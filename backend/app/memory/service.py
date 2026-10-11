"""The long-term memory system.

Responsibilities, in the order the agent uses them:

1. **recall** — given the current dilemma, fetch the few memories and past
   decisions that are actually relevant. Never the whole history.
2. **persist** — turn this turn's candidates into stored memories, de-duplicated
   and conflict-resolved.
3. **record_decision** — once a session concludes, distil it into a decision
   event so future sessions can learn from what the user chose last time.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.logging import get_logger
from app.db.enums import MemoryKind, MemoryStatus
from app.db.models import DecisionEvent
from app.db.repositories import memory as memory_repo
from app.llm import client as llm
from app.memory.consolidator import ConsolidationResult, consolidate_candidate
from app.memory.prompts import SUMMARIZE_SESSION_SYSTEM, SUMMARIZE_SESSION_USER
from app.schemas.agent import Decision, MemoryCandidate, MemoryRef, SessionAnalysis

logger = get_logger(__name__)


class DecisionEventDraft(BaseModel):
    title: str = Field(description="不超过 20 字的选择名称")
    situation: str = Field(description="当时处境，两三句话")
    greed: list[str] = Field(default_factory=list)
    fear: list[str] = Field(default_factory=list)
    cost: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    summary: str = Field(default="")


@dataclass(slots=True)
class RecallResult:
    memories: list[MemoryRef] = field(default_factory=list)
    events: list[DecisionEvent] = field(default_factory=list)

    def render(self) -> str:
        """Render recalled context for the prompt, provenance included."""
        if not self.memories and not self.events:
            return "（没有检索到相关的历史记忆）"
        blocks: list[str] = []
        if self.memories:
            lines = []
            for ref in self.memories:
                tag = {
                    "user_stated": "用户亲述",
                    "user_confirmed": "用户确认",
                    "agent_inferred": "Agent 推测（未经确认，只能作为假设提出）",
                }.get(ref.provenance, ref.provenance)
                lines.append(f"- [{ref.kind} · {tag}] {ref.content}")
            blocks.append("【关于这位用户的长期记忆】\n" + "\n".join(lines))
        if self.events:
            lines = []
            for event in self.events:
                choice = event.chosen_option or "（未记录最终选择）"
                lines.append(
                    f"- {event.title}：处境={event.situation[:80]}；最终选择={choice}；"
                    f"代价={'、'.join(event.cost) or '未记录'}"
                )
            blocks.append("【用户过去的选择】\n" + "\n".join(lines))
        return "\n\n".join(blocks)


class MemoryService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    # ---------------------------------------------------------------- recall
    async def recall(
        self,
        session: AsyncSession,
        *,
        user_id: uuid.UUID,
        query: str,
        kinds: list[MemoryKind] | None = None,
        top_k: int | None = None,
    ) -> RecallResult:
        top_k = top_k or self.settings.memory_top_k
        embedding = await llm.embed_query(query)
        result = RecallResult()

        hits = await memory_repo.search_memories(
            session,
            user_id=user_id,
            query_embedding=embedding,
            top_k=top_k,
            min_score=self.settings.memory_min_score,
            kinds=kinds,
            include_candidates=True,
        )
        for memory, score in hits:
            if memory.status == str(MemoryStatus.REJECTED):
                continue
            result.memories.append(
                MemoryRef(
                    id=str(memory.id),
                    kind=MemoryKind(memory.kind),
                    content=memory.content,
                    provenance=memory.provenance,  # type: ignore[arg-type]
                    status=memory.status,
                    score=score,
                    source_session_id=str(memory.source_session_id) if memory.source_session_id else None,
                    created_at=memory.created_at.isoformat() if memory.created_at else None,
                )
            )

        events = await memory_repo.search_decision_events(
            session,
            user_id=user_id,
            query_embedding=embedding,
            top_k=max(2, top_k // 2),
            min_score=self.settings.memory_min_score,
        )
        result.events = [event for event, _score in events]
        if result.memories:
            await memory_repo.touch_memories_used(session, [uuid.UUID(ref.id) for ref in result.memories])
        return result

    # --------------------------------------------------------------- persist
    async def persist(
        self,
        session: AsyncSession,
        *,
        user_id: uuid.UUID,
        session_id: uuid.UUID | None,
        message_id: uuid.UUID | None,
        candidates: list[MemoryCandidate],
    ) -> list[ConsolidationResult]:
        results: list[ConsolidationResult] = []
        for candidate in candidates:
            try:
                outcome = await consolidate_candidate(
                    session,
                    settings=self.settings,
                    user_id=user_id,
                    session_id=session_id,
                    message_id=message_id,
                    candidate=candidate,
                )
            except Exception as exc:  # noqa: BLE001 - one bad candidate must not lose the turn
                logger.warning("memory_persist_failed", extra={"error": type(exc).__name__})
                continue
            results.append(outcome)
        return results

    # ------------------------------------------------------- decision events
    async def record_decision_event(
        self,
        session: AsyncSession,
        *,
        user_id: uuid.UUID,
        session_id: uuid.UUID,
        analysis: SessionAnalysis,
        decision: Decision,
        turns: list[tuple[str, str]],
    ) -> DecisionEvent | None:
        if decision.status == "hold_safety":
            return None
        model = llm.get_chat_model(temperature=0.2, max_tokens=900)
        transcript = "\n".join(
            f"{'用户' if role == 'user' else 'Agent'}：{content.strip()}" for role, content in turns[-20:]
        )
        try:
            draft = await llm.call_structured(
                model,
                DecisionEventDraft,
                [
                    SystemMessage(content=SUMMARIZE_SESSION_SYSTEM),
                    HumanMessage(
                        content=SUMMARIZE_SESSION_USER.format(
                            dilemma=analysis.dilemma.statement or "（未明确）",
                            options="、".join(opt.label for opt in analysis.dilemma.options) or "（未明确）",
                            decision=decision.headline or decision.defer_reason or "（暂缓建议）",
                            transcript=transcript,
                        )
                    ),
                ],
                node="decision_event",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("decision_event_summary_failed", extra={"error": type(exc).__name__})
            return None

        embedding = await llm.embed_query(f"{draft.title} {draft.situation} {draft.summary}")
        return await memory_repo.create_decision_event(
            session,
            user_id=user_id,
            session_id=session_id,
            title=draft.title[:200],
            situation=draft.situation,
            options=[opt.model_dump() for opt in analysis.dilemma.options],
            chosen_option=decision.chosen_label,
            greed=draft.greed,
            fear=draft.fear,
            cost=draft.cost or decision.must_give_up[:3],
            risks=draft.risks or [risk.description for risk in decision.risks[:3]],
            summary=draft.summary or decision.headline,
            embedding=embedding,
        )
