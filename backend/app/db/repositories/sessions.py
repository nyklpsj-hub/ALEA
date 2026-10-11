"""Queries over sessions and their message log."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.enums import MessageRole, SessionStatus
from app.db.models import Message, Session


async def create_session(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    title: str = "新的困境",
) -> Session:
    row = Session(user_id=user_id, title=title, status=str(SessionStatus.ACTIVE), analysis={})
    session.add(row)
    await session.flush()
    return row


async def get_session(
    session: AsyncSession,
    session_id: uuid.UUID,
    *,
    user_id: uuid.UUID,
    include_deleted: bool = False,
) -> Session | None:
    stmt = select(Session).where(Session.id == session_id, Session.user_id == user_id)
    if not include_deleted:
        stmt = stmt.where(Session.deleted_at.is_(None))
    return (await session.execute(stmt)).scalar_one_or_none()


async def list_sessions(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    statuses: list[SessionStatus] | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[Session]:
    stmt = select(Session).where(Session.user_id == user_id, Session.deleted_at.is_(None))
    if statuses:
        stmt = stmt.where(Session.status.in_([str(s) for s in statuses]))
    stmt = stmt.order_by(Session.last_active_at.desc()).limit(limit).offset(offset)
    return list((await session.execute(stmt)).scalars())


async def next_seq(session: AsyncSession, *, session_id: uuid.UUID) -> int:
    stmt = select(func.coalesce(func.max(Message.seq), 0)).where(Message.session_id == session_id)
    return int((await session.execute(stmt)).scalar_one()) + 1


async def append_message(
    session: AsyncSession,
    *,
    session_id: uuid.UUID,
    user_id: uuid.UUID,
    role: MessageRole,
    content: str,
    node: str | None = None,
    meta: dict[str, Any] | None = None,
) -> Message:
    message = Message(
        session_id=session_id,
        user_id=user_id,
        seq=await next_seq(session, session_id=session_id),
        role=str(role),
        content=content,
        node=node,
        meta=meta or {},
    )
    session.add(message)
    await session.flush()
    return message


async def list_messages(
    session: AsyncSession, *, session_id: uuid.UUID, limit: int = 500, after_seq: int = 0
) -> list[Message]:
    stmt = (
        select(Message)
        .where(Message.session_id == session_id, Message.seq > after_seq)
        .order_by(Message.seq)
        .limit(limit)
    )
    return list((await session.execute(stmt)).scalars())


async def update_session_progress(
    session: AsyncSession,
    row: Session,
    *,
    analysis: dict[str, Any] | None = None,
    decision: dict[str, Any] | None = None,
    phase: str | None = None,
    turn_count: int | None = None,
    sufficiency: float | None = None,
    title: str | None = None,
) -> Session:
    if analysis is not None:
        row.analysis = analysis
    if decision is not None:
        row.decision = decision
    if phase is not None:
        row.phase = phase
    if turn_count is not None:
        row.turn_count = turn_count
    if sufficiency is not None:
        row.info_sufficiency = sufficiency
    if title is not None:
        row.title = title
    row.last_active_at = datetime.now(tz=UTC)
    await session.flush()
    # server-side onupdate values are not returned by UPDATE; reload them so the
    # row can be serialised after the transaction closes.
    await session.refresh(row)
    return row


async def soft_delete_session(session: AsyncSession, row: Session) -> None:
    row.deleted_at = datetime.now(tz=UTC)
    await session.flush()
