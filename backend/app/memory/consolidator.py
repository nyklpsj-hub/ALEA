"""De-duplication, updating and conflict handling for long-term memory.

Three things can happen when a new candidate arrives next to something already
stored: it is the same fact (merge), it is a newer version of it (supersede), or
it contradicts it (supersede and keep both in the audit trail). All three are
recorded, so a memory can always be explained.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.logging import get_logger
from app.db.enums import MemoryAuditAction, MemoryProvenance, MemoryStatus
from app.db.models import Memory
from app.db.repositories import memory as memory_repo
from app.llm import client as llm
from app.memory.prompts import CONFLICT_SYSTEM, CONFLICT_USER
from app.schemas.agent import MemoryCandidate

logger = get_logger(__name__)

#: A stored memory is trusted more when it came from the user directly.
PROVENANCE_RANK: dict[str, int] = {
    MemoryProvenance.AGENT_INFERRED: 0,
    MemoryProvenance.USER_CONFIRMED: 1,
    MemoryProvenance.USER_STATED: 2,
}

#: Lower bound for "this might be about the same thing"; below it we do not even
#: ask the model, we just store a new memory.
NEIGHBOUR_THRESHOLD = 0.72


class NeighbourVerdict(BaseModel):
    index: int = Field(description="已有记忆在列表中的序号，从 0 开始")
    verdict: Literal["duplicate", "update", "contradict", "distinct"]
    note: str = ""


class ConflictResolution(BaseModel):
    verdicts: list[NeighbourVerdict] = Field(default_factory=list)


@dataclass(slots=True)
class ConsolidationResult:
    action: Literal["created", "merged", "superseded", "skipped"]
    memory: Memory
    superseded: list[Memory]


async def consolidate_candidate(
    session: AsyncSession,
    *,
    settings: Settings,
    user_id: uuid.UUID,
    session_id: uuid.UUID | None,
    message_id: uuid.UUID | None,
    candidate: MemoryCandidate,
) -> ConsolidationResult:
    embedding = await llm.embed_query(candidate.content)
    target_status = (
        MemoryStatus.CANDIDATE if candidate.provenance is MemoryProvenance.AGENT_INFERRED else MemoryStatus.ACTIVE
    )

    exact = await memory_repo.find_duplicates(
        session,
        user_id=user_id,
        kind=candidate.kind,
        query_embedding=embedding,
        threshold=settings.memory_dedup_similarity,
        limit=1,
    )
    neighbours = await memory_repo.find_duplicates(
        session,
        user_id=user_id,
        kind=candidate.kind,
        query_embedding=embedding,
        threshold=NEIGHBOUR_THRESHOLD,
        limit=4,
    )

    merge_into: Memory | None = exact[0][0] if exact else None
    supersede: list[Memory] = []

    if merge_into is None and neighbours:
        verdicts = await _judge_neighbours(
            settings=settings, candidate=candidate, neighbours=[m for m, _ in neighbours]
        )
        for index, verdict in verdicts.items():
            neighbour = neighbours[index][0]
            if verdict == "duplicate":
                merge_into = neighbour
                break
            if verdict in {"update", "contradict"}:
                supersede.append(neighbour)

    if merge_into is not None:
        memory = await _merge(session, target=merge_into, candidate=candidate, embedding=embedding)
        return ConsolidationResult(action="merged", memory=memory, superseded=[])

    memory = await memory_repo.create_memory(
        session,
        user_id=user_id,
        kind=candidate.kind,
        content=candidate.content,
        provenance=candidate.provenance,
        status=target_status,
        normalized_key=candidate.normalized_key,
        confidence=candidate.confidence,
        importance=candidate.importance,
        evidence=candidate.evidence or None,
        tags=[],
        embedding=embedding,
        source_session_id=session_id,
        source_message_id=message_id,
    )
    await memory_repo.record_audit(
        session,
        memory=memory,
        action=str(MemoryAuditAction.CREATED),
        actor="agent",
        after=memory_repo.memory_to_audit_payload(memory),
        reason=candidate.reason or None,
    )

    for old in supersede:
        await _supersede(session, old=old, new=memory, reason="与更新的信息冲突，旧记忆归档")

    return ConsolidationResult(
        action="superseded" if supersede else "created",
        memory=memory,
        superseded=supersede,
    )


async def _judge_neighbours(
    *,
    settings: Settings,
    candidate: MemoryCandidate,
    neighbours: list[Memory],
) -> dict[int, str]:
    """Ask the model whether each near neighbour is the same thing, newer, or opposite."""
    listing = "\n".join(
        f"{index}. {item.content}（录入于 {item.created_at:%Y-%m-%d}，状态 {item.status}）"
        for index, item in enumerate(neighbours)
    )
    model = llm.get_chat_model(temperature=0.0)
    try:
        resolution = await llm.call_structured(
            model,
            ConflictResolution,
            [
                SystemMessage(content=CONFLICT_SYSTEM),
                HumanMessage(
                    content=CONFLICT_USER.format(
                        kind=candidate.kind,
                        content=candidate.content,
                        evidence=candidate.evidence or "（无）",
                        existing=listing,
                    )
                ),
            ],
            node="memory_conflict",
        )
    except Exception as exc:  # noqa: BLE001 - never fail a turn over this
        logger.warning("memory_conflict_check_failed", extra={"error": type(exc).__name__})
        return {}

    return {item.index: item.verdict for item in resolution.verdicts if 0 <= item.index < len(neighbours)}


async def _merge(
    session: AsyncSession,
    *,
    target: Memory,
    candidate: MemoryCandidate,
    embedding: list[float],
) -> Memory:
    before = memory_repo.memory_to_audit_payload(target)

    new_rank = PROVENANCE_RANK[candidate.provenance]
    old_rank = PROVENANCE_RANK.get(target.provenance, 0)
    if new_rank > old_rank:
        target.provenance = str(candidate.provenance)
    if candidate.provenance is not MemoryProvenance.AGENT_INFERRED:
        target.status = str(MemoryStatus.ACTIVE)

    # Keep the more specific wording.
    if len(candidate.content) > len(target.content) * 1.2:
        target.content = candidate.content
        target.embedding = embedding

    if candidate.evidence and candidate.evidence not in (target.evidence or ""):
        target.evidence = ((target.evidence or "") + "\n" + candidate.evidence).strip()[:2000]

    target.confidence = max(target.confidence, candidate.confidence)
    target.importance = max(target.importance, candidate.importance)
    if candidate.normalized_key:
        target.normalized_key = candidate.normalized_key[:160]

    await session.flush()
    await memory_repo.record_audit(
        session,
        memory=target,
        action=str(MemoryAuditAction.MERGED),
        actor="agent",
        before=before,
        after=memory_repo.memory_to_audit_payload(target),
        reason="与已有记忆合并，补充证据与置信度",
    )
    return target


async def _supersede(session: AsyncSession, *, old: Memory, new: Memory, reason: str) -> None:
    old.status = str(MemoryStatus.SUPERSEDED)
    old.superseded_by = new.id
    await session.flush()
    await memory_repo.record_audit(
        session,
        memory=old,
        action=str(MemoryAuditAction.SUPERSEDED),
        actor="agent",
        after={"superseded_by": str(new.id)},
        reason=reason,
    )


async def apply_user_verdict(
    session: AsyncSession,
    *,
    memory: Memory,
    action: Literal["confirm", "correct", "reject"],
    corrected_text: str | None = None,
    note: str | None = None,
) -> Memory:
    """Apply the user's own answer to a candidate memory."""
    before = memory_repo.memory_to_audit_payload(memory)
    if action == "confirm":
        memory.status = str(MemoryStatus.ACTIVE)
        memory.provenance = str(MemoryProvenance.USER_CONFIRMED)
        memory.confidence = min(1.0, max(memory.confidence, 0.9))
        audit_action = MemoryAuditAction.CONFIRMED
    elif action == "correct":
        memory.status = str(MemoryStatus.ACTIVE)
        memory.provenance = str(MemoryProvenance.USER_STATED)
        memory.confidence = 0.95
        if corrected_text:
            memory.content = corrected_text.strip()
            memory.embedding = await llm.embed_query(memory.content)
        audit_action = MemoryAuditAction.UPDATED
    else:  # reject
        memory.status = str(MemoryStatus.REJECTED)
        memory.confidence = 0.1
        audit_action = MemoryAuditAction.REJECTED

    memory.last_used_at = datetime.now(tz=UTC)
    await session.flush()
    await memory_repo.record_audit(
        session,
        memory=memory,
        action=str(audit_action),
        actor="user",
        before=before,
        after=memory_repo.memory_to_audit_payload(memory),
        reason=note or f"用户在对话中选择了 {action}",
    )
    return memory
