"""Session lifecycle and history."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Query, status

from app.api.deps import ChatServiceDep, UserIdDep
from app.db.enums import SessionStatus
from app.schemas.api import MessageRead, SessionCreate, SessionDetail, SessionRead, SessionUpdate

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=SessionDetail, status_code=status.HTTP_201_CREATED)
async def create_session(payload: SessionCreate, service: ChatServiceDep, user_id: UserIdDep) -> SessionDetail:
    row = await service.create_session(user_id=user_id, title=payload.title)
    return await service.get_detail(user_id=user_id, session_id=row.id)


@router.get("", response_model=list[SessionRead])
async def list_sessions(
    service: ChatServiceDep,
    user_id: UserIdDep,
    status_filter: SessionStatus | None = Query(default=None, alias="status"),
    limit: int = Query(default=100, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[SessionRead]:
    rows = await service.list_sessions(
        user_id=user_id,
        statuses=[status_filter] if status_filter else None,
        limit=limit,
        offset=offset,
    )
    return [
        SessionRead(
            id=row.id,
            title=row.title,
            status=SessionStatus(row.status),
            phase=row.phase,
            turn_count=row.turn_count,
            info_sufficiency=row.info_sufficiency,
            last_active_at=row.last_active_at,
            created_at=row.created_at,
            has_decision=bool(row.decision and row.decision.get("status") == "recommend"),
        )
        for row in rows
    ]


@router.get("/{session_id}", response_model=SessionDetail)
async def get_session(session_id: uuid.UUID, service: ChatServiceDep, user_id: UserIdDep) -> SessionDetail:
    return await service.get_detail(user_id=user_id, session_id=session_id)


@router.patch("/{session_id}", response_model=SessionRead)
async def update_session(
    session_id: uuid.UUID,
    payload: SessionUpdate,
    service: ChatServiceDep,
    user_id: UserIdDep,
) -> SessionRead:
    row = await service.update_session(
        user_id=user_id, session_id=session_id, title=payload.title, status=payload.status
    )
    return SessionRead(
        id=row.id,
        title=row.title,
        status=SessionStatus(row.status),
        phase=row.phase,
        turn_count=row.turn_count,
        info_sufficiency=row.info_sufficiency,
        last_active_at=row.last_active_at,
        created_at=row.created_at,
        has_decision=bool(row.decision and row.decision.get("status") == "recommend"),
    )


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: uuid.UUID, service: ChatServiceDep, user_id: UserIdDep) -> None:
    await service.delete_session(user_id=user_id, session_id=session_id)


@router.get("/{session_id}/messages", response_model=list[MessageRead])
async def list_messages(
    session_id: uuid.UUID,
    service: ChatServiceDep,
    user_id: UserIdDep,
    after_seq: int = Query(default=0, ge=0),
) -> list[MessageRead]:
    detail = await service.get_detail(user_id=user_id, session_id=session_id)
    return [m for m in detail.messages if m.seq > after_seq]
