"""Long-term memory management.

Everything a user can do to their own memory: read it, search it, correct it,
delete it, and see who changed what and why.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Query, status

from app.api.deps import ContainerDep, UserIdDep
from app.core.errors import NotFoundError
from app.db.enums import MemoryKind, MemoryStatus
from app.db.repositories import memory as memory_repo
from app.llm import client as llm
from app.schemas.api import (
    MemoryAuditRead,
    MemoryCreate,
    MemoryRead,
    MemorySearchHit,
    MemorySearchRequest,
    MemoryUpdate,
)

router = APIRouter(prefix="/memories", tags=["memory"])


@router.get("", response_model=list[MemoryRead])
async def list_memories(
    container: ContainerDep,
    user_id: UserIdDep,
    kind: list[MemoryKind] | None = Query(default=None),
    status_filter: list[MemoryStatus] | None = Query(default=None, alias="status"),
    include_deleted: bool = Query(default=False),
    limit: int = Query(default=200, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[MemoryRead]:
    async with container.database.session() as session:
        rows = await memory_repo.list_memories(
            session,
            user_id=user_id,
            kinds=kind,
            statuses=status_filter,
            include_deleted=include_deleted,
            limit=limit,
            offset=offset,
        )
        return [MemoryRead.model_validate(row) for row in rows]


@router.get("/stats", response_model=dict)
async def memory_stats(container: ContainerDep, user_id: UserIdDep) -> dict:
    async with container.database.session() as session:
        return await memory_repo.count_memories(session, user_id=user_id)


@router.post("/search", response_model=list[MemorySearchHit])
async def search_memories(
    payload: MemorySearchRequest, container: ContainerDep, user_id: UserIdDep
) -> list[MemorySearchHit]:
    embedding = await llm.embed_query(payload.query)
    async with container.database.session() as session:
        hits = await memory_repo.search_memories(
            session,
            user_id=user_id,
            query_embedding=embedding,
            top_k=payload.top_k,
            min_score=container.settings.memory_min_score,
            kinds=payload.kinds or None,
        )
    return [MemorySearchHit(memory=MemoryRead.model_validate(memory), score=score) for memory, score in hits]


@router.post("", response_model=MemoryRead, status_code=status.HTTP_201_CREATED)
async def create_memory(payload: MemoryCreate, container: ContainerDep, user_id: UserIdDep) -> MemoryRead:
    embedding = await llm.embed_query(payload.content)
    async with container.database.session() as session:
        memory = await memory_repo.create_memory(
            session,
            user_id=user_id,
            kind=payload.kind,
            content=payload.content,
            provenance=payload.provenance,
            status=MemoryStatus.ACTIVE,
            confidence=payload.confidence,
            importance=payload.importance,
            evidence=None,
            tags=payload.tags,
            embedding=embedding,
        )
        await memory_repo.record_audit(
            session,
            memory=memory,
            action="created",
            actor="user",
            after=memory_repo.memory_to_audit_payload(memory),
            reason="用户在记忆管理页面手动添加",
        )
        return MemoryRead.model_validate(memory)


@router.patch("/{memory_id}", response_model=MemoryRead)
async def update_memory(
    memory_id: uuid.UUID,
    payload: MemoryUpdate,
    container: ContainerDep,
    user_id: UserIdDep,
) -> MemoryRead:
    async with container.database.session() as session:
        memory = await memory_repo.get_memory(session, memory_id, user_id=user_id)
        if memory is None:
            raise NotFoundError("记忆不存在")
        before = memory_repo.memory_to_audit_payload(memory)

        if payload.content is not None and payload.content != memory.content:
            memory.content = payload.content
            memory.embedding = await llm.embed_query(payload.content)
        if payload.kind is not None:
            memory.kind = str(payload.kind)
        if payload.status is not None:
            memory.status = str(payload.status)
        if payload.importance is not None:
            memory.importance = payload.importance
        if payload.confidence is not None:
            memory.confidence = payload.confidence
        if payload.tags is not None:
            memory.tags = payload.tags
        await session.flush()
        # PostgreSQL `onupdate` values are not returned by UPDATE, so the row's
        # updated_at would still be expired; refresh makes it readable here.
        await session.refresh(memory)

        await memory_repo.record_audit(
            session,
            memory=memory,
            action="updated",
            actor="user",
            before=before,
            after=memory_repo.memory_to_audit_payload(memory),
            reason="用户在记忆管理页面修改",
        )
        return MemoryRead.model_validate(memory)


@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(memory_id: uuid.UUID, container: ContainerDep, user_id: UserIdDep) -> None:
    async with container.database.session() as session:
        memory = await memory_repo.get_memory(session, memory_id, user_id=user_id)
        if memory is None:
            raise NotFoundError("记忆不存在")
        memory.deleted_at = datetime.now(tz=UTC)
        await session.flush()
        await memory_repo.record_audit(
            session,
            memory=memory,
            action="deleted",
            actor="user",
            reason="用户在记忆管理页面删除（保留审计记录，可恢复）",
        )


@router.post("/{memory_id}/restore", response_model=MemoryRead)
async def restore_memory(memory_id: uuid.UUID, container: ContainerDep, user_id: UserIdDep) -> MemoryRead:
    async with container.database.session() as session:
        memory = await memory_repo.get_memory(session, memory_id, user_id=user_id, include_deleted=True)
        if memory is None:
            raise NotFoundError("记忆不存在")
        memory.deleted_at = None
        await session.flush()
        await session.refresh(memory)
        await memory_repo.record_audit(session, memory=memory, action="restored", actor="user", reason="用户恢复该记忆")
        return MemoryRead.model_validate(memory)


@router.get("/{memory_id}/history", response_model=list[MemoryAuditRead])
async def memory_history(memory_id: uuid.UUID, container: ContainerDep, user_id: UserIdDep) -> list[MemoryAuditRead]:
    async with container.database.session() as session:
        memory = await memory_repo.get_memory(session, memory_id, user_id=user_id, include_deleted=True)
        if memory is None:
            raise NotFoundError("记忆不存在")
        rows = await memory_repo.list_audit(session, memory_id=memory_id, user_id=user_id)
        return [MemoryAuditRead.model_validate(row) for row in rows]
