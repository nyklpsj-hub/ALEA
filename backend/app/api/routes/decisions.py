"""Historical decision events."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Query

from app.api.deps import ContainerDep, UserIdDep
from app.core.errors import NotFoundError
from app.db.repositories import memory as memory_repo
from app.schemas.api import DecisionEventOutcomeUpdate, DecisionEventRead

router = APIRouter(prefix="/decisions", tags=["decisions"])


@router.get("", response_model=list[DecisionEventRead])
async def list_decisions(
    container: ContainerDep,
    user_id: UserIdDep,
    limit: int = Query(default=100, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[DecisionEventRead]:
    async with container.database.session() as session:
        rows = await memory_repo.list_decision_events(session, user_id=user_id, limit=limit, offset=offset)
        return [DecisionEventRead.model_validate(row) for row in rows]


@router.patch("/{event_id}/outcome", response_model=DecisionEventRead)
async def update_outcome(
    event_id: uuid.UUID,
    payload: DecisionEventOutcomeUpdate,
    container: ContainerDep,
    user_id: UserIdDep,
) -> DecisionEventRead:
    """Record what actually happened — the only way memory gets a track record."""
    async with container.database.session() as session:
        event = await memory_repo.get_decision_event(session, event_id, user_id=user_id)
        if event is None:
            raise NotFoundError("决策记录不存在")
        event.outcome = payload.outcome
        await session.flush()
        return DecisionEventRead.model_validate(event)
