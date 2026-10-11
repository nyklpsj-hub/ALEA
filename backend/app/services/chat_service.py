"""Orchestration of a conversation turn.

The graph owns the reasoning; this service owns the transaction around it:
persisting the user's message, seeding or resuming checkpointer state, turning
graph events into a stream the browser can render, and writing the durable
snapshot (analysis, decision, title) back to the session row.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.graph import AgentRuntime
from app.agent.nodes import REPLY_NODES
from app.agent.state import empty_state
from app.core.config import Settings
from app.core.errors import AgentRunError, AppError, NotFoundError
from app.core.logging import get_logger, redact
from app.db.enums import (
    AgentPhase,
    ConfirmationStatus,
    MemoryKind,
    MemoryProvenance,
    MessageRole,
    SessionStatus,
)
from app.db.models import Message, Session
from app.db.repositories import sessions as sessions_repo
from app.db.session import Database
from app.memory.service import MemoryService
from app.schemas.agent import (
    Confirmation,
    Decision,
    MemoryCandidate,
    SessionAnalysis,
)
from app.schemas.api import MessageRead, SessionDetail

logger = get_logger(__name__)

DEFAULT_TITLE = "新的困境"

StreamMode = Literal["values", "updates", "checkpoints", "tasks", "debug", "messages", "custom"]

#: Exactly the stream modes one turn needs: token deltas plus node updates.
_TURN_STREAM_MODES: tuple[StreamMode, StreamMode] = ("messages", "updates")


@dataclass
class TurnResult:
    session: Session
    message: Message
    analysis: SessionAnalysis
    decision: Decision | None
    phase: str
    turn_count: int
    memory_changes: list[dict[str, Any]] = field(default_factory=list)


class ChatService:
    def __init__(
        self,
        *,
        settings: Settings,
        database: Database,
        runtime: AgentRuntime,
        memory_service: MemoryService,
    ) -> None:
        self.settings = settings
        self.database = database
        self.runtime = runtime
        self.memory_service = memory_service

    # ------------------------------------------------------------- sessions
    async def create_session(self, *, user_id: uuid.UUID, title: str | None = None) -> Session:
        async with self.database.session() as session:
            return await sessions_repo.create_session(session, user_id=user_id, title=title or DEFAULT_TITLE)

    async def list_sessions(
        self,
        *,
        user_id: uuid.UUID,
        statuses: list[SessionStatus] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Session]:
        async with self.database.session() as session:
            return await sessions_repo.list_sessions(
                session, user_id=user_id, statuses=statuses, limit=limit, offset=offset
            )

    async def get_session(self, *, user_id: uuid.UUID, session_id: uuid.UUID) -> Session:
        async with self.database.session() as session:
            row = await sessions_repo.get_session(session, session_id, user_id=user_id)
            if row is None:
                raise NotFoundError("会话不存在")
            return row

    async def get_detail(self, *, user_id: uuid.UUID, session_id: uuid.UUID) -> SessionDetail:
        async with self.database.session() as session:
            row = await sessions_repo.get_session(session, session_id, user_id=user_id)
            if row is None:
                raise NotFoundError("会话不存在")
            messages = await sessions_repo.list_messages(session, session_id=session_id)
            analysis = SessionAnalysis.model_validate(row.analysis or {})
            decision = Decision.model_validate(row.decision) if row.decision else None
            return SessionDetail(
                id=row.id,
                title=row.title,
                status=SessionStatus(row.status),
                phase=row.phase,
                turn_count=row.turn_count,
                info_sufficiency=row.info_sufficiency,
                last_active_at=row.last_active_at,
                created_at=row.created_at,
                has_decision=decision is not None and decision.status == "recommend",
                analysis=analysis,
                decision=decision,
                messages=[MessageRead.model_validate(message) for message in messages],
            )

    async def update_session(
        self,
        *,
        user_id: uuid.UUID,
        session_id: uuid.UUID,
        title: str | None = None,
        status: SessionStatus | None = None,
    ) -> Session:
        async with self.database.session() as session:
            row = await sessions_repo.get_session(session, session_id, user_id=user_id)
            if row is None:
                raise NotFoundError("会话不存在")
            if title is not None:
                row.title = title
            if status is not None:
                row.status = str(status)
            await session.flush()
            await session.refresh(row)
            return row

    async def delete_session(self, *, user_id: uuid.UUID, session_id: uuid.UUID) -> None:
        async with self.database.session() as session:
            row = await sessions_repo.get_session(session, session_id, user_id=user_id)
            if row is None:
                raise NotFoundError("会话不存在")
            await sessions_repo.soft_delete_session(session, row)

    # ------------------------------------------------------------- turn run
    async def run_turn(self, *, user_id: uuid.UUID, session_id: uuid.UUID, content: str) -> TurnResult:
        result: TurnResult | None = None
        async for event in self.stream_turn(user_id=user_id, session_id=session_id, content=content):
            if event["type"] == "done":
                result = event["result"]
            elif event["type"] == "error":
                # Re-raise the original error so the HTTP layer can map it to the
                # right status (a missing API key is a 503, not a 500).
                raise event.get("exception") or AgentRunError(event["message"])
        if result is None:  # pragma: no cover - stream_turn always ends with done/error
            raise AgentRunError("对话未产生结果")
        return result

    async def stream_turn(
        self, *, user_id: uuid.UUID, session_id: uuid.UUID, content: str
    ) -> AsyncIterator[dict[str, Any]]:
        deps = self._deps(user_id)
        graph = self.runtime.require_graph()

        async with self.database.session() as session:
            row = await sessions_repo.get_session(session, session_id, user_id=user_id)
            if row is None:
                raise NotFoundError("会话不存在")
            user_message = await sessions_repo.append_message(
                session,
                session_id=session_id,
                user_id=user_id,
                role=MessageRole.USER,
                content=content,
            )
            user_message_id = user_message.id
            state_seed = await self._build_input(
                session, row=row, session_id=session_id, user_id=user_id, content=content
            )

        logger.info(
            "turn_started",
            extra={
                "session_id": str(session_id),
                "turn": row.turn_count + 1,
                "user_text": redact(content),
            },
        )
        yield {"type": "start", "session_id": str(session_id), "user_message_id": str(user_message_id)}

        config = self.runtime.thread_config(str(session_id), deps)
        seen_nodes: set[str] = set()
        final_state: dict[str, Any] = {}

        try:
            async for chunk in graph.astream(state_seed, config, stream_mode=_TURN_STREAM_MODES):
                mode, payload = _split_stream_chunk(chunk)
                if mode == "messages":
                    message_chunk, metadata = payload
                    node = (metadata or {}).get("langgraph_node")
                    if node not in REPLY_NODES:
                        continue
                    text = _chunk_text(message_chunk)
                    if text:
                        yield {"type": "token", "node": node, "text": text}
                elif mode == "updates":
                    for node, update in (payload or {}).items():
                        if not isinstance(update, dict):
                            continue
                        final_state.update(update)
                        if node not in seen_nodes:
                            seen_nodes.add(node)
                            yield {
                                "type": "node",
                                "node": node,
                                "phase": _phase_of(update) or None,
                            }
        except AppError as exc:
            logger.warning(
                "agent_run_failed",
                extra={"session_id": str(session_id), "code": exc.code},
            )
            yield {
                "type": "error",
                "message": exc.message,
                "code": exc.code,
                "status": exc.http_status,
                "exception": exc,
            }
            return
        except Exception as exc:  # noqa: BLE001 - surface a usable message
            logger.exception("agent_run_crashed", extra={"session_id": str(session_id)})
            yield {
                "type": "error",
                "message": f"决策流程执行失败：{type(exc).__name__}",
                "code": "agent_run_failed",
            }
            return

        snapshot = await graph.aget_state(config)
        if snapshot is not None and snapshot.values:
            final_state = {**final_state, **snapshot.values}

        analysis = _analysis_from_state(final_state)
        decision = _decision_from_state(final_state)
        reply = final_state.get("assistant_reply") or "（本轮没有产生回复）"
        memory_changes = list(final_state.get("memory_changes") or [])

        if decision is not None and decision.status == "hold_safety":
            yield {"type": "decision", "decision": decision.model_dump(mode="json")}
        if memory_changes:
            yield {"type": "memory", "changes": memory_changes}

        async with self.database.session() as session:
            row = await sessions_repo.get_session(session, session_id, user_id=user_id)
            if row is None:
                raise NotFoundError("会话不存在")
            assistant_message = await sessions_repo.append_message(
                session,
                session_id=session_id,
                user_id=user_id,
                role=MessageRole.ASSISTANT,
                content=reply,
                node=final_state.get("phase"),
                meta={
                    "route": final_state.get("route"),
                    "citations": [c.model_dump(mode="json") for c in (final_state.get("citations") or [])],
                    "memory_changes": memory_changes,
                },
            )
            await sessions_repo.update_session_progress(
                session,
                row,
                analysis=analysis.model_dump(mode="json"),
                decision=decision.model_dump(mode="json") if decision else None,
                phase=str(final_state.get("phase") or row.phase),
                turn_count=int(final_state.get("turn_count") or row.turn_count),
                sufficiency=float(final_state.get("sufficiency") or 0.0),
                title=self._maybe_title(row.title, analysis),
            )
            if decision is not None and decision.is_final:
                await self._record_decision_event(
                    session, row=row, analysis=analysis, decision=decision, state=final_state
                )
            session_id_snapshot = row.id
            turn_count = row.turn_count

        result = TurnResult(
            session=row,
            message=assistant_message,
            analysis=analysis,
            decision=decision,
            phase=str(final_state.get("phase") or row.phase),
            turn_count=turn_count,
            memory_changes=memory_changes,
        )
        logger.info(
            "turn_finished",
            extra={
                "session_id": str(session_id_snapshot),
                "phase": result.phase,
                "route": final_state.get("route"),
                "route_reason": final_state.get("route_reason"),
                "citations": len(final_state.get("citations") or []),
            },
        )
        yield {
            "type": "done",
            "result": result,
            # Both a model and its JSON form are provided: the HTTP layer
            # serialises `message`, in-process callers read `result`.
            "message": assistant_message,
            "analysis": analysis.model_dump(mode="json"),
            "decision": decision.model_dump(mode="json") if decision else None,
            "phase": result.phase,
            "turn_count": turn_count,
            "citations": [c.model_dump(mode="json") for c in (final_state.get("citations") or [])],
            "memory_changes": memory_changes,
            "message_id": str(assistant_message.id),
        }

    # -------------------------------------------------- confirmation resolve
    async def resolve_confirmation(
        self,
        *,
        user_id: uuid.UUID,
        session_id: uuid.UUID,
        confirmation_id: str,
        action: str,
        corrected_text: str | None = None,
        note: str | None = None,
    ) -> tuple[SessionAnalysis, uuid.UUID | None]:
        """Apply the user's own verdict on an inferred claim, from the UI."""
        deps = self._deps(user_id)
        snapshot = await self.runtime.get_state(str(session_id), deps)
        confirmations_raw = (snapshot or {}).get("confirmations") or []
        confirmations = [Confirmation.model_validate(c) for c in confirmations_raw]

        target = next((c for c in confirmations if c.id == confirmation_id), None)
        if target is None:
            raise NotFoundError("该确认项不存在或已被处理")

        status = {
            "confirm": ConfirmationStatus.CONFIRMED,
            "correct": ConfirmationStatus.CORRECTED,
            "reject": ConfirmationStatus.REJECTED,
        }[action]
        target.status = status
        target.resolution_note = (corrected_text or note or "").strip()

        if action == "correct" and corrected_text:
            target.claim = corrected_text.strip()

        # Pass the models themselves: the checkpointer's serialiser is told which
        # state types to expect, and raw dicts would come back as dicts.
        await self.runtime.update_confirmations(str(session_id), list(confirmations), deps)

        memory_id: uuid.UUID | None = None
        if action in {"confirm", "correct"}:
            memory_id = await self._persist_confirmed_memory(
                user_id=user_id, session_id=session_id, confirmation=target, action=action
            )

        state = await self.runtime.get_state(str(session_id), deps) or {}
        analysis = _analysis_from_state({**state, "confirmations": confirmations})
        async with self.database.session() as session:
            row = await sessions_repo.get_session(session, session_id, user_id=user_id)
            if row is not None:
                await sessions_repo.update_session_progress(session, row, analysis=analysis.model_dump(mode="json"))
        return analysis, memory_id

    async def _persist_confirmed_memory(
        self,
        *,
        user_id: uuid.UUID,
        session_id: uuid.UUID,
        confirmation: Confirmation,
        action: str,
    ) -> uuid.UUID | None:
        kind_map = {
            "greed": MemoryKind.GREED,
            "fear": MemoryKind.FEAR,
            "constraint": MemoryKind.CONSTRAINT,
            "goal": MemoryKind.GOAL,
            "value": MemoryKind.VALUE,
            "contradiction": MemoryKind.FACT,
        }
        provenance = MemoryProvenance.USER_STATED if action == "correct" else MemoryProvenance.USER_CONFIRMED
        candidate = MemoryCandidate(
            kind=kind_map.get(confirmation.topic, MemoryKind.FACT),
            content=confirmation.claim,
            normalized_key=confirmation.topic,
            provenance=provenance,
            confidence=0.95,
            importance=4,
            evidence=confirmation.resolution_note or confirmation.evidence,
            reason="用户在确认环节确认/修正了这条判断",
        )
        try:
            async with self.database.session() as session:
                outcomes = await self.memory_service.persist(
                    session,
                    user_id=user_id,
                    session_id=session_id,
                    message_id=None,
                    candidates=[candidate],
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("confirmed_memory_persist_failed", extra={"error": type(exc).__name__})
            return None
        return outcomes[0].memory.id if outcomes else None

    # ------------------------------------------------------------- internals
    def _deps(self, user_id: uuid.UUID):
        from app.agent.deps import AgentDeps

        return AgentDeps(
            settings=self.settings,
            database=self.database,
            memory_service=self.memory_service,
            user_id=user_id,
        )

    async def _build_input(
        self,
        session: AsyncSession,
        *,
        row: Session,
        session_id: uuid.UUID,
        user_id: uuid.UUID,
        content: str,
    ) -> dict[str, Any]:
        """Seed the checkpointer thread on first use, otherwise send just the delta."""
        snapshot = await self.runtime.get_state(str(session_id))
        has_state = bool(snapshot and snapshot.get("messages"))

        base: dict[str, Any] = {
            "latest_user_text": content,
            "messages": [HumanMessage(content=content)],
            "user_id": str(user_id),
            "session_id": str(session_id),
        }
        if has_state:
            return base

        # Rebuild from the durable snapshot: this is what makes a session
        # resumable even if its checkpoint thread was lost.
        history = await sessions_repo.list_messages(session, session_id=session_id)
        seed: dict[str, Any] = dict(empty_state(user_id=str(user_id), session_id=str(session_id)))
        persisted = SessionAnalysis.model_validate(row.analysis or {})
        seed.update(
            {
                "dilemma": persisted.dilemma,
                "greeds": persisted.greeds,
                "fears": persisted.fears,
                "constraints": persisted.constraints,
                "contradictions": persisted.contradictions,
                "open_questions": persisted.open_questions,
                "confirmations": persisted.confirmations,
                "citations": persisted.citations,
                "sufficiency": persisted.sufficiency,
                "sufficiency_rationale": persisted.sufficiency_rationale,
                "risk_flags": persisted.risk_flags,
                "turn_count": row.turn_count,
                "phase": AgentPhase(row.phase) if row.phase else AgentPhase.UNDERSTANDING,
                "messages": [
                    HumanMessage(content=m.content) if m.role == str(MessageRole.USER) else AIMessage(content=m.content)
                    for m in history
                ],
            }
        )
        seed.update(base)
        logger.info(
            "graph_thread_seeded",
            extra={"session_id": str(session_id), "history_messages": len(history)},
        )
        return seed

    async def _record_decision_event(
        self,
        session: AsyncSession,
        *,
        row: Session,
        analysis: SessionAnalysis,
        decision: Decision,
        state: dict[str, Any],
    ) -> None:
        turns = _transcript_from_state(state)
        await self.memory_service.record_decision_event(
            session,
            user_id=row.user_id,
            session_id=row.id,
            analysis=analysis,
            decision=decision,
            turns=turns,
        )

    @staticmethod
    def _maybe_title(current: str, analysis: SessionAnalysis) -> str:
        if current and current != DEFAULT_TITLE:
            return current
        statement = analysis.dilemma.statement.strip()
        if statement:
            return statement[:40]
        return current or DEFAULT_TITLE


#: Every mode name LangGraph can emit.
_STREAM_MODES = frozenset({"values", "updates", "checkpoints", "tasks", "debug", "messages", "custom"})


def _split_stream_chunk(chunk: Any) -> tuple[str | None, Any]:
    """Normalise a LangGraph stream chunk into ``(mode, payload)``.

    Depending on the installed version and how ``stream_mode`` was passed,
    chunks arrive in one of three shapes:

    * ``("messages", (message, metadata))`` — the documented wrapper;
    * ``(message, metadata)``                — unwrapped message chunk;
    * ``{"node": update}``                   — unwrapped node update.

    Getting this wrong silently drops every event, so all three are handled
    explicitly rather than assumed.
    """
    if isinstance(chunk, tuple) and len(chunk) == 2:
        head = chunk[0]
        if isinstance(head, str) and head in _STREAM_MODES:
            return head, chunk[1]
        return "messages", chunk
    if isinstance(chunk, Mapping):
        # StreamPart mappings carry exactly type/ns/data; a node-update mapping
        # is keyed by node name instead.
        if set(chunk) <= {"type", "ns", "data"} and chunk.get("type") in _STREAM_MODES:
            return str(chunk["type"]), chunk.get("data")
        return "updates", chunk
    return None, None


def _chunk_text(chunk: Any) -> str:
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _phase_of(update: dict[str, Any]) -> str | None:
    phase = update.get("phase")
    return str(phase) if phase else None


def _analysis_from_state(state: dict[str, Any]) -> SessionAnalysis:
    return SessionAnalysis(
        dilemma=state.get("dilemma") or SessionAnalysis().dilemma,
        greeds=list(state.get("greeds") or []),
        fears=list(state.get("fears") or []),
        constraints=list(state.get("constraints") or []),
        contradictions=list(state.get("contradictions") or []),
        open_questions=list(state.get("open_questions") or []),
        confirmations=list(state.get("confirmations") or []),
        recalled_memories=list(state.get("recalled_memory_refs") or []),
        citations=list(state.get("citations") or []),
        sufficiency=float(state.get("sufficiency") or 0.0),
        sufficiency_rationale=state.get("sufficiency_rationale") or "",
        risk_flags=list(state.get("risk_flags") or []),
    )


def _decision_from_state(state: dict[str, Any]) -> Decision | None:
    raw = state.get("decision")
    if raw is None:
        return None
    if isinstance(raw, Decision):
        return raw
    try:
        return Decision.model_validate(raw)
    except Exception:  # noqa: BLE001
        return None


def _transcript_from_state(state: dict[str, Any]) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    for message in state.get("messages") or []:
        if isinstance(message, HumanMessage):
            turns.append(("user", str(message.content)))
        elif isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
            turns.append(("assistant", str(message.content)))
    return turns
