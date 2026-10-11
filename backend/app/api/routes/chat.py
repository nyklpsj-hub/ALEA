"""Conversation endpoints.

Two ways to run a turn:

* ``POST /sessions/{id}/messages`` — run it and return the finished result.
* ``POST /sessions/{id}/stream``   — server-sent events while the graph runs.

Both drive the same service method, so the two paths can never drift apart.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.deps import ChatServiceDep, UserIdDep
from app.core.errors import AgentRunError
from app.core.logging import get_logger
from app.schemas.api import (
    ChatRequest,
    ChatResponse,
    ConfirmationResolve,
    MemoryChange,
    MessageRead,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/sessions", tags=["chat"])

SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


@router.post("/{session_id}/messages", response_model=ChatResponse)
async def send_message(
    session_id: uuid.UUID,
    payload: ChatRequest,
    service: ChatServiceDep,
    user_id: UserIdDep,
) -> ChatResponse:
    result = await service.run_turn(user_id=user_id, session_id=session_id, content=payload.content)
    return ChatResponse(
        session_id=session_id,
        message=MessageRead.model_validate(result.message),
        analysis=result.analysis,
        decision=result.decision,
        phase=result.phase,
        turn_count=result.turn_count,
        recalled_memory_count=len(result.analysis.recalled_memories),
        citations=result.analysis.citations,
        memory_changes=[
            MemoryChange(
                action=change["action"],
                memory_id=uuid.UUID(change["memory_id"]),
                content=change["content"],
                kind=change["kind"],
                status=change["status"],
            )
            for change in result.memory_changes
        ],
    )


@router.post("/{session_id}/stream")
async def stream_message(
    session_id: uuid.UUID,
    payload: ChatRequest,
    service: ChatServiceDep,
    user_id: UserIdDep,
) -> StreamingResponse:
    async def event_source() -> AsyncIterator[str]:
        try:
            async for event in service.stream_turn(user_id=user_id, session_id=session_id, content=payload.content):
                yield _sse(_serialize(event))
        except AgentRunError as exc:
            yield _sse({"type": "error", "message": exc.message})
        except Exception:  # noqa: BLE001 - the stream must end with a usable event
            logger.exception("stream_turn_failed", extra={"session_id": str(session_id)})
            yield _sse({"type": "error", "message": "服务在处理这条消息时出错，请重试。"})

    return StreamingResponse(event_source(), media_type="text/event-stream", headers=SSE_HEADERS)


@router.post("/{session_id}/confirmations/{confirmation_id}", response_model=dict)
async def resolve_confirmation(
    session_id: uuid.UUID,
    confirmation_id: str,
    payload: ConfirmationResolve,
    service: ChatServiceDep,
    user_id: UserIdDep,
) -> dict:
    analysis, memory_id = await service.resolve_confirmation(
        user_id=user_id,
        session_id=session_id,
        confirmation_id=confirmation_id,
        action=payload.action,
        corrected_text=payload.corrected_text,
        note=payload.note,
    )
    return {
        "analysis": analysis.model_dump(mode="json"),
        "memory_id": str(memory_id) if memory_id else None,
    }


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


def _serialize(event: dict) -> dict:
    """Convert dataclasses/UUIDs inside an event into JSON-safe values."""
    # In-process callers reuse the original exception; it never goes on the wire.
    event.pop("exception", None)
    if event.get("type") == "done":
        event.pop("result", None)
        message = event.get("message")
        if message is not None and not isinstance(message, dict):
            event["message"] = MessageRead.model_validate(message).model_dump(mode="json")
    return event
