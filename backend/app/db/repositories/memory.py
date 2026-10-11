"""Queries over long-lived user memory.

Semantic search always scopes to a single ``user_id``; there is no code path
that can read another user's memories, which is what keeps the future
multi-user migration a configuration change rather than a security review.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.enums import MemoryKind, MemoryProvenance, MemoryStatus
from app.db.models import DecisionEvent, Memory, MemoryAudit


async def create_memory(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    kind: MemoryKind,
    content: str,
    provenance: MemoryProvenance,
    status: MemoryStatus,
    normalized_key: str = "",
    confidence: float = 0.6,
    importance: int = 3,
    evidence: str | None = None,
    tags: list[str] | None = None,
    embedding: list[float] | None = None,
    source_session_id: uuid.UUID | None = None,
    source_message_id: uuid.UUID | None = None,
) -> Memory:
    memory = Memory(
        user_id=user_id,
        kind=str(kind),
        content=content,
        normalized_key=normalized_key[:160],
        status=str(status),
        provenance=str(provenance),
        confidence=confidence,
        importance=importance,
        evidence=evidence,
        tags=tags or [],
        embedding=embedding,
        source_session_id=source_session_id,
        source_message_id=source_message_id,
    )
    session.add(memory)
    await session.flush()
    return memory


async def get_memory(
    session: AsyncSession, memory_id: uuid.UUID, *, user_id: uuid.UUID, include_deleted: bool = False
) -> Memory | None:
    stmt = select(Memory).where(Memory.id == memory_id, Memory.user_id == user_id)
    if not include_deleted:
        stmt = stmt.where(Memory.deleted_at.is_(None))
    return (await session.execute(stmt)).scalar_one_or_none()


async def list_memories(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    kinds: list[MemoryKind] | None = None,
    statuses: list[MemoryStatus] | None = None,
    include_deleted: bool = False,
    limit: int = 200,
    offset: int = 0,
) -> list[Memory]:
    stmt = select(Memory).where(Memory.user_id == user_id)
    if not include_deleted:
        stmt = stmt.where(Memory.deleted_at.is_(None))
    if kinds:
        stmt = stmt.where(Memory.kind.in_([str(k) for k in kinds]))
    if statuses:
        stmt = stmt.where(Memory.status.in_([str(s) for s in statuses]))
    stmt = stmt.order_by(Memory.importance.desc(), Memory.created_at.desc()).limit(limit).offset(offset)
    return list((await session.execute(stmt)).scalars())


async def count_memories(session: AsyncSession, *, user_id: uuid.UUID) -> dict[str, int]:
    stmt = (
        select(Memory.status, func.count(Memory.id))
        .where(Memory.user_id == user_id, Memory.deleted_at.is_(None))
        .group_by(Memory.status)
    )
    return {status: int(count) for status, count in (await session.execute(stmt)).all()}


async def search_memories(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    query_embedding: list[float],
    top_k: int,
    min_score: float,
    kinds: list[MemoryKind] | None = None,
    statuses: list[MemoryStatus] | None = None,
    include_candidates: bool = True,
) -> list[tuple[Memory, float]]:
    distance = Memory.embedding.cosine_distance(query_embedding)
    stmt = (
        select(Memory, distance.label("distance"))
        .where(
            Memory.user_id == user_id,
            Memory.deleted_at.is_(None),
            Memory.embedding.is_not(None),
        )
        .order_by(distance)
        .limit(top_k)
    )
    if kinds:
        stmt = stmt.where(Memory.kind.in_([str(k) for k in kinds]))
    if statuses:
        stmt = stmt.where(Memory.status.in_([str(s) for s in statuses]))
    elif not include_candidates:
        stmt = stmt.where(Memory.status == str(MemoryStatus.ACTIVE))

    hits: list[tuple[Memory, float]] = []
    for memory, dist in (await session.execute(stmt)).all():
        score = 1.0 - float(dist)
        if score < min_score:
            continue
        hits.append((memory, round(score, 4)))
    return hits


async def find_duplicates(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    kind: MemoryKind,
    query_embedding: list[float],
    threshold: float,
    limit: int = 5,
) -> list[tuple[Memory, float]]:
    """Nearest neighbours of the same kind, for merge/conflict decisions."""
    distance = Memory.embedding.cosine_distance(query_embedding)
    stmt = (
        select(Memory, distance.label("distance"))
        .where(
            Memory.user_id == user_id,
            Memory.kind == str(kind),
            Memory.deleted_at.is_(None),
            Memory.embedding.is_not(None),
            Memory.status != str(MemoryStatus.SUPERSEDED),
        )
        .order_by(distance)
        .limit(limit)
    )
    results: list[tuple[Memory, float]] = []
    for memory, dist in (await session.execute(stmt)).all():
        score = 1.0 - float(dist)
        if score >= threshold:
            results.append((memory, round(score, 4)))
    return results


def memory_to_audit_payload(memory: Memory) -> dict[str, Any]:
    return {
        "kind": memory.kind,
        "content": memory.content,
        "status": memory.status,
        "provenance": memory.provenance,
        "confidence": memory.confidence,
        "importance": memory.importance,
        "normalized_key": memory.normalized_key,
        "tags": list(memory.tags or []),
    }


async def record_audit(
    session: AsyncSession,
    *,
    memory: Memory,
    action: str,
    actor: str = "agent",
    before: dict[str, Any] | None = None,
    after: dict[str, Any] | None = None,
    reason: str | None = None,
) -> MemoryAudit:
    audit = MemoryAudit(
        memory_id=memory.id,
        user_id=memory.user_id,
        action=action,
        actor=actor,
        before=before,
        after=after,
        reason=reason,
    )
    session.add(audit)
    return audit


async def touch_memories_used(session: AsyncSession, memory_ids: list[uuid.UUID]) -> None:
    if not memory_ids:
        return
    await session.execute(
        update(Memory)
        .where(Memory.id.in_(memory_ids))
        .values(last_used_at=datetime.now(tz=UTC), use_count=Memory.use_count + 1)
    )


async def list_audit(
    session: AsyncSession, *, memory_id: uuid.UUID, user_id: uuid.UUID, limit: int = 100
) -> list[MemoryAudit]:
    stmt = (
        select(MemoryAudit)
        .join(Memory, Memory.id == MemoryAudit.memory_id)
        .where(MemoryAudit.memory_id == memory_id, Memory.user_id == user_id)
        .order_by(MemoryAudit.created_at.desc())
        .limit(limit)
    )
    return list((await session.execute(stmt)).scalars())


# ------------------------------------------------------------- decision events
async def create_decision_event(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    session_id: uuid.UUID | None,
    title: str,
    situation: str,
    options: list[dict[str, Any]],
    chosen_option: str | None,
    greed: list[str],
    fear: list[str],
    cost: list[str],
    risks: list[str],
    summary: str | None,
    embedding: list[float] | None,
) -> DecisionEvent:
    event = DecisionEvent(
        user_id=user_id,
        session_id=session_id,
        title=title,
        situation=situation,
        options=options,
        chosen_option=chosen_option,
        greed=greed,
        fear=fear,
        cost=cost,
        risks=risks,
        summary=summary,
        embedding=embedding,
    )
    session.add(event)
    await session.flush()
    return event


async def search_decision_events(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    query_embedding: list[float],
    top_k: int,
    min_score: float,
) -> list[tuple[DecisionEvent, float]]:
    distance = DecisionEvent.embedding.cosine_distance(query_embedding)
    stmt = (
        select(DecisionEvent, distance.label("distance"))
        .where(
            DecisionEvent.user_id == user_id,
            DecisionEvent.deleted_at.is_(None),
            DecisionEvent.embedding.is_not(None),
        )
        .order_by(distance)
        .limit(top_k)
    )
    hits: list[tuple[DecisionEvent, float]] = []
    for event, dist in (await session.execute(stmt)).all():
        score = 1.0 - float(dist)
        if score >= min_score:
            hits.append((event, round(score, 4)))
    return hits


async def list_decision_events(
    session: AsyncSession, *, user_id: uuid.UUID, limit: int = 100, offset: int = 0
) -> list[DecisionEvent]:
    stmt = (
        select(DecisionEvent)
        .where(DecisionEvent.user_id == user_id, DecisionEvent.deleted_at.is_(None))
        .order_by(DecisionEvent.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return list((await session.execute(stmt)).scalars())


async def get_decision_event(session: AsyncSession, event_id: uuid.UUID, *, user_id: uuid.UUID) -> DecisionEvent | None:
    stmt = select(DecisionEvent).where(DecisionEvent.id == event_id, DecisionEvent.user_id == user_id)
    return (await session.execute(stmt)).scalar_one_or_none()
