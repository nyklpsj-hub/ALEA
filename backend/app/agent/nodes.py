"""The graph's nodes.

Flow (see ``graph.py`` for the wiring):

    understand → recall → ⟨route⟩ → explore ─┐
                                     confirm ─┼→ understand (next turn)
                                     knowledge → decide → finalize → END

Every node is small on purpose: one model call, one bounded job. The reasoning
lives in the prompts; the control flow lives in the router.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.config import get_config
from pydantic import BaseModel, Field

from app.agent.deps import AgentDeps
from app.agent.prompts import (
    BASE_IDENTITY,
    CONFIRM_SYSTEM,
    CONFIRM_USER,
    DECIDE_SYSTEM,
    DECIDE_USER,
    EXPLORE_SYSTEM,
    EXPLORE_USER,
    KNOWLEDGE_SYSTEM,
    KNOWLEDGE_USER,
    SAFETY_RESOURCES,
    SAFETY_SYSTEM,
    SAFETY_USER,
    UNDERSTAND_SYSTEM,
    UNDERSTAND_USER,
)
from app.agent.state import AleaState
from app.agent.tools import ToolTrace, build_knowledge_tools, search_knowledge
from app.core.config import Settings
from app.core.logging import get_logger, redact
from app.db.enums import AgentPhase, ConfirmationStatus, MemoryProvenance
from app.llm import client as llm
from app.memory.extractor import extract_candidates
from app.schemas.agent import (
    Citation,
    Confirmation,
    Constraint,
    Contradiction,
    Decision,
    Dilemma,
    Motive,
)

logger = get_logger(__name__)

REPLY_NODES = frozenset({"explore", "confirm", "decide", "safety"})

#: Deterministic backstop for the safety-critical path. The model also screens
#: for these signals; a keyword hit only ever *adds* a flag, never removes one.
_CRISIS_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"自杀|轻生|不想活|活不下去|结束生命|去死", "自伤/自杀倾向"),
    (r"自残|割腕|伤害自己", "自残倾向"),
    (r"杀(了)?(他|她|他们)|报复社会|同归于尽", "伤害他人的想法"),
    (r"被(家暴|殴打|威胁|胁迫)|人身安全", "人身安全受威胁"),
    (r"吸毒|赌博输光|被催收|高利贷", "成瘾或严重债务风险"),
)


def _deps() -> AgentDeps:
    config = get_config()
    deps = (config.get("configurable") or {}).get("deps")
    if deps is None:
        raise RuntimeError("AgentDeps 未注入：请通过 config={'configurable': {'deps': ...}} 调用")
    return deps


def _transcript(messages: list[AnyMessage], *, limit: int = 16) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    for message in messages[-limit:]:
        if isinstance(message, HumanMessage):
            turns.append(("user", _text_of(message)))
        elif isinstance(message, AIMessage) and not message.tool_calls:
            turns.append(("assistant", _text_of(message)))
    return [(role, text) for role, text in turns if text.strip()]


def _text_of(message: AnyMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)


def _render_turns(turns: list[tuple[str, str]]) -> str:
    return (
        "\n".join(f"{'用户' if role == 'user' else 'Agent'}：{text.strip()}" for role, text in turns)
        or "（还没有对话）"
    )


def _slug_id(prefix: str, text: str) -> str:
    digest = hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:8]
    return f"{prefix}-{digest}"


def _bullet(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "（暂无）"


def _analysis_json(state: AleaState) -> str:
    payload = {
        "dilemma": state.get("dilemma", Dilemma()).model_dump(exclude_none=True),
        "greeds": [m.model_dump(exclude_none=True) for m in state.get("greeds", [])],
        "fears": [m.model_dump(exclude_none=True) for m in state.get("fears", [])],
        "constraints": [c.model_dump(exclude_none=True) for c in state.get("constraints", [])],
        "contradictions": [c.model_dump(exclude_none=True) for c in state.get("contradictions", [])],
        "open_questions": state.get("open_questions", []),
        "confirmations": [c.model_dump(exclude_none=True) for c in state.get("confirmations", [])],
        "sufficiency": state.get("sufficiency", 0.0),
    }
    return json.dumps(payload, ensure_ascii=False, indent=1)


def screen_risk(text: str) -> list[str]:
    flags: list[str] = []
    for pattern, label in _CRISIS_PATTERNS:
        if re.search(pattern, text):
            flags.append(label)
    return flags


def normalize_state(state: AleaState) -> None:
    """Coerce anything that round-tripped through the checkpointer back to models.

    State written by ``aupdate_state`` (for example when the user answers a
    confirmation card in the UI) can arrive as plain dicts. Normalising once, at
    the entry node, means no downstream node has to defend against two shapes.
    """
    state["greeds"] = _coerce_list(state.get("greeds"), Motive, "greeds")
    state["fears"] = _coerce_list(state.get("fears"), Motive, "fears")
    state["constraints"] = _coerce_list(state.get("constraints"), Constraint, "constraints")
    state["contradictions"] = _coerce_list(state.get("contradictions"), Contradiction, "contradictions")
    state["confirmations"] = _coerce_list(state.get("confirmations"), Confirmation, "confirmations")
    state["citations"] = _coerce_list(state.get("citations"), Citation, "citations")

    dilemma = state.get("dilemma")
    if dilemma is not None and not isinstance(dilemma, Dilemma):
        state["dilemma"] = _coerce_model(dilemma, Dilemma, "dilemma") or Dilemma()

    decision = state.get("decision")
    if decision is not None and not isinstance(decision, Decision):
        state["decision"] = _coerce_model(decision, Decision, "decision")


def _coerce_list[ModelT: BaseModel](items: Any, model: type[ModelT], field: str) -> list[ModelT]:
    coerced: list[ModelT] = []
    for item in items or []:
        if isinstance(item, model):
            coerced.append(item)
            continue
        try:
            coerced.append(model.model_validate(item))
        except Exception:  # noqa: BLE001 - drop what cannot be understood
            logger.warning("state_item_dropped", extra={"field": field, "item": type(item).__name__})
    return coerced


def _coerce_model[ModelT: BaseModel](value: Any, model: type[ModelT], field: str) -> ModelT | None:
    try:
        return model.model_validate(value)
    except Exception:  # noqa: BLE001
        logger.warning("state_item_dropped", extra={"field": field})
        return None


# =============================================================== understanding
class Understanding(BaseModel):
    """Structured reading of the conversation so far."""

    dilemma: Dilemma = Field(default_factory=Dilemma)
    greeds: list[Motive] = Field(default_factory=list)
    fears: list[Motive] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    confirmations: list[Confirmation] = Field(default_factory=list)
    #: Ids of previously recorded motives/constraints the user has since
    #: corrected or denied. Without this the agent cannot tell "the model
    #: forgot to repeat it" from "it no longer applies".
    retired_ids: list[str] = Field(default_factory=list)
    sufficiency: float = Field(default=0.0, ge=0.0, le=1.0)
    sufficiency_rationale: str = ""
    risk_flags: list[str] = Field(default_factory=list)
    user_requests_conclusion: bool = Field(default=False, description="用户是否明确要求现在就给结论")


async def understand(state: AleaState) -> dict[str, Any]:
    normalize_state(state)
    turns = _transcript(state.get("messages", []))
    latest = state.get("latest_user_text", "")

    model = llm.get_chat_model(temperature=0.0)
    result = await llm.call_structured(
        model,
        Understanding,
        [
            SystemMessage(content=BASE_IDENTITY + "\n\n" + UNDERSTAND_SYSTEM),
            HumanMessage(
                content=UNDERSTAND_USER.format(
                    analysis_json=_analysis_json(state),
                    recalled_block=state.get("recalled_block") or "（本轮尚未检索）",
                    transcript=_render_turns(turns),
                    latest_user_text=latest or "（无）",
                )
            ),
        ],
        node="understand",
    )

    update = _reconcile(state, result)
    update["phase"] = AgentPhase.UNDERSTANDING
    logger.info(
        "understand_done",
        extra={
            "session_id": state.get("session_id"),
            "sufficiency": update["sufficiency"],
            "greeds": len(update["greeds"]),
            "fears": len(update["fears"]),
            "risk_flags": len(update["risk_flags"]),
            "user_text": redact(latest),
        },
    )
    return update


def _reconcile(state: AleaState, result: Understanding) -> dict[str, Any]:
    """Merge the model's reading with what is already established.

    The model may forget a confirmed item; the user's confirmation is the more
    reliable record, so confirmed items are carried over unless the model
    explicitly replaced them under the same id.
    """

    def _ensure_ids(items: list[Any], prefix: str) -> list[Any]:
        for index, item in enumerate(items, start=1):
            if not getattr(item, "id", ""):
                item.id = f"{prefix}{index}"
        return items

    #: Ids retired earlier in this session stay retired.
    retired = {*(state.get("retired_ids") or []), *(result.retired_ids or [])}

    # `kind` is the model's explicit classification; the list it happened to
    # write the item into is the weaker signal. Small models do put a fear in
    # `greeds` — filing by the label keeps the fear instead of silently
    # reclassifying it as a want.
    motives = [*result.greeds, *result.fears]
    greeds = [item for item in motives if item.kind == "greed"]
    fears = [item for item in motives if item.kind != "greed"]
    if len(greeds) != len(result.greeds):
        logger.info(
            "motive_list_corrected",
            extra={"moved": abs(len(greeds) - len(result.greeds))},
        )

    greeds = _ensure_ids(greeds, "g")
    fears = _ensure_ids(fears, "f")
    constraints = _ensure_ids(list(result.constraints), "c")

    for motive in greeds:
        _enforce_provenance(motive)
    for motive in fears:
        _enforce_provenance(motive)
    for constraint in constraints:
        _enforce_provenance(constraint)

    previous_confirmed = {
        item.id: item
        for item in [*state.get("greeds", []), *state.get("fears", [])]
        if item.status == ConfirmationStatus.CONFIRMED and item.id
    }
    seen = {item.id for item in [*greeds, *fears]}
    for item_id, item in previous_confirmed.items():
        if item_id not in seen and item_id not in retired:
            if item.kind == "greed":
                greeds.append(item)
            else:
                fears.append(item)

    previous_constraints = {
        item.id: item
        for item in state.get("constraints", [])
        if item.status == ConfirmationStatus.CONFIRMED and item.id
    }
    seen_constraints = {item.id for item in constraints}
    for item_id, constraint in previous_constraints.items():
        if item_id not in seen_constraints and item_id not in retired:
            constraints.append(constraint)

    # A retired item may still be echoed back by the model; drop it either way.
    greeds = [item for item in greeds if item.id not in retired]
    fears = [item for item in fears if item.id not in retired]
    constraints = [item for item in constraints if item.id not in retired]

    resolved: dict[str, Confirmation] = {
        c.id: c for c in state.get("confirmations", []) if c.status != ConfirmationStatus.PENDING and c.id
    }
    pending: list[Confirmation] = []
    for confirmation in result.confirmations[:3]:
        if not confirmation.id:
            confirmation.id = _slug_id("q", confirmation.claim)
        if confirmation.id in resolved:
            continue
        pending.append(confirmation)

    known_claims = {c.claim.strip() for c in resolved.values()}
    pending = [c for c in pending if c.claim.strip() not in known_claims]

    # The model often stops repeating the option list once it is established;
    # losing it would break the decision stage, so keep the last known one.
    dilemma = result.dilemma
    if not dilemma.options:
        previous_options = state.get("dilemma", Dilemma()).options
        if previous_options:
            dilemma = dilemma.model_copy(update={"options": previous_options})
    if not dilemma.statement:
        dilemma = dilemma.model_copy(update={"statement": state.get("dilemma", Dilemma()).statement})

    risk_flags = list(dict.fromkeys([*result.risk_flags, *screen_risk(state.get("latest_user_text", ""))]))

    return {
        "dilemma": dilemma,
        "greeds": greeds,
        "fears": fears,
        "constraints": constraints,
        "contradictions": [
            Contradiction(
                description=c.description.strip(),
                between=[b.strip() for b in c.between][:4],
                evidence=c.evidence,
                is_inference=True,
            )
            for c in result.contradictions
            if c.description.strip()
        ][:4],
        "open_questions": [q.strip() for q in result.open_questions if q.strip()][:6],
        "confirmations": [*resolved.values(), *pending],
        "retired_ids": sorted(retired),
        "sufficiency": result.sufficiency,
        "sufficiency_rationale": result.sufficiency_rationale,
        "risk_flags": risk_flags,
        "user_requests_conclusion": result.user_requests_conclusion,
    }


def _enforce_provenance(item: Motive | Constraint) -> None:
    """A claim with no quoted evidence cannot be presented as the user's own."""
    if not item.evidence.strip():
        item.provenance = MemoryProvenance.AGENT_INFERRED
        item.confidence = min(item.confidence, 0.6)
        if item.status == ConfirmationStatus.CONFIRMED:
            # Confirmed without evidence is possible only if the user confirmed it
            # in words, in which case the model should have quoted them.
            item.status = ConfirmationStatus.PENDING


# ====================================================================== memory
async def recall(state: AleaState) -> dict[str, Any]:
    deps = _deps()
    dilemma = state.get("dilemma", Dilemma())
    query_parts = [
        dilemma.statement,
        state.get("latest_user_text", ""),
        " ".join(m.statement for m in state.get("fears", [])[:2]),
        " ".join(m.statement for m in state.get("greeds", [])[:2]),
    ]
    query = " ".join(part for part in query_parts if part).strip() or "用户处境"
    try:
        async with deps.database.session() as session:
            result = await deps.memory_service.recall(session, user_id=deps.user_id, query=query)
    except Exception as exc:  # noqa: BLE001 - memory is an enhancement, not a dependency
        logger.warning("memory_recall_failed", extra={"error": type(exc).__name__})
        return {
            "recalled_block": "（记忆检索失败，本轮不参考历史记忆）",
            "recalled_memory_refs": [],
            "recalled_event_count": 0,
        }

    logger.info(
        "memory_recalled",
        extra={
            "session_id": state.get("session_id"),
            "memories": len(result.memories),
            "events": len(result.events),
        },
    )
    return {
        "recalled_block": result.render(),
        "recalled_memory_refs": result.memories,
        "recalled_event_count": len(result.events),
        "phase": AgentPhase.MEMORY,
    }


# ====================================================================== router
def route(state: AleaState) -> dict[str, Any]:
    """Deterministic control flow — no model call, so cost and behaviour are predictable."""
    deps = _deps()
    settings = deps.settings
    turn = int(state.get("turn_count", 0))
    sufficiency = float(state.get("sufficiency", 0.0))
    risk_flags = state.get("risk_flags", [])
    confirmations = state.get("confirmations", [])
    asked = set(state.get("asked_confirmation_ids", []))
    citations = state.get("citations", [])
    wants_conclusion = bool(state.get("user_requests_conclusion", False))
    has_options = bool(state.get("dilemma", Dilemma()).options)

    def decided(name: str, reason: str) -> dict[str, Any]:
        return {"route": name, "route_reason": reason}

    if risk_flags:
        return decided("safety", f"检测到风险信号：{'、'.join(risk_flags)}")

    if turn >= settings.agent_hard_turn_limit:
        return decided(
            "knowledge" if not citations else "decide",
            f"已达到对话轮次上限（{settings.agent_hard_turn_limit}），必须收敛",
        )

    pending_unasked = [c for c in confirmations if c.status == ConfirmationStatus.PENDING and c.id not in asked]

    if wants_conclusion and sufficiency >= 0.35 and has_options:
        return decided(
            "knowledge" if not citations else "decide",
            "用户明确要求给出结论，且已有足够背景",
        )

    if sufficiency < 0.35:
        return decided("explore", f"信息量不足（sufficiency={sufficiency:.2f}），继续了解处境")

    if pending_unasked:
        return decided("confirm", f"有 {len(pending_unasked)} 条关键推断需要用户确认后才能作为依据")

    if sufficiency < settings.agent_sufficiency_threshold:
        if turn >= settings.agent_soft_turn_limit:
            return decided(
                "knowledge" if not citations else "decide",
                "已达到软轮次上限，需要给出明确结论（可以是暂缓）",
            )
        return decided("explore", f"信息仍不充分（sufficiency={sufficiency:.2f}），继续追问")

    if not citations:
        return decided("knowledge", "信息已充分，先检索理论依据")
    return decided("decide", "信息充分且已获得理论依据，可以给出建议")


# ===================================================================== explore
async def explore(state: AleaState) -> dict[str, Any]:
    turns = _transcript(state.get("messages", []))
    model = llm.get_chat_model(temperature=0.7, streaming=True)

    prompt = EXPLORE_USER.format(
        dilemma=state.get("dilemma", Dilemma()).statement or "（还在了解）",
        greeds=_bullet([m.statement for m in state.get("greeds", [])]),
        fears=_bullet([m.statement for m in state.get("fears", [])]),
        constraints=_bullet([c.statement for c in state.get("constraints", [])]),
        open_questions=_bullet(state.get("open_questions", [])),
        asked_directions=_bullet(state.get("asked_directions", [])),
        transcript=_render_turns(turns[-8:]),
    )
    response = await model.ainvoke(
        [SystemMessage(content=BASE_IDENTITY + "\n\n" + EXPLORE_SYSTEM), HumanMessage(content=prompt)]
    )
    reply = _text_of(response).strip()
    asked = list(state.get("asked_directions", []))
    question = next((q for q in state.get("open_questions", []) if q not in asked), None)
    if question:
        asked.append(question)

    logger.info(
        "explore_done",
        extra={"session_id": state.get("session_id"), "reply_preview": redact(reply)},
    )
    return {
        "assistant_reply": reply,
        "messages": [AIMessage(content=reply)],
        "phase": AgentPhase.EXPLORING,
        "asked_directions": asked[-8:],
        "turn_count": int(state.get("turn_count", 0)) + 1,
        "decision": None,
    }


# ===================================================================== confirm
async def confirm(state: AleaState) -> dict[str, Any]:
    turns = _transcript(state.get("messages", []))
    pending = [c for c in state.get("confirmations", []) if c.status == ConfirmationStatus.PENDING][:3]

    if not pending:
        return await explore(state)

    claims = "\n".join(
        f"{index}. 我的推测：{c.claim}\n   依据：{c.evidence or '（你之前的表述）'}"
        for index, c in enumerate(pending, start=1)
    )
    model = llm.get_chat_model(temperature=0.6, streaming=True)
    response = await model.ainvoke(
        [
            SystemMessage(content=BASE_IDENTITY + "\n\n" + CONFIRM_SYSTEM),
            HumanMessage(
                content=CONFIRM_USER.format(
                    dilemma=state.get("dilemma", Dilemma()).statement or "（还在了解）",
                    claims=claims,
                    transcript=_render_turns(turns[-6:]),
                )
            ),
        ]
    )
    reply = _text_of(response).strip()
    asked = list(state.get("asked_confirmation_ids", []))
    asked.extend(c.id for c in pending if c.id)
    pending_claims = [c.claim for c in pending if c.claim]

    logger.info(
        "confirm_done",
        extra={"session_id": state.get("session_id"), "claims": len(pending)},
    )
    return {
        "assistant_reply": reply,
        "messages": [AIMessage(content=reply)],
        "phase": AgentPhase.CONFIRMING,
        "asked_confirmation_ids": asked[-12:],
        "turn_count": int(state.get("turn_count", 0)) + 1,
        "decision": None,
        "open_questions": [
            *state.get("open_questions", []),
            *[claim for claim in pending_claims if claim not in state.get("open_questions", [])],
        ],
    }


# =================================================================== knowledge
async def knowledge(state: AleaState) -> dict[str, Any]:
    deps = _deps()
    settings = deps.settings
    dilemma = state.get("dilemma", Dilemma())
    trace = ToolTrace()

    async with deps.database.session() as session:
        tools = build_knowledge_tools(
            session=session,
            settings=settings,
            memory_service=deps.memory_service,
            user_id=deps.user_id,
            trace=trace,
        )
        model = llm.get_chat_model(temperature=0.0)
        bound = model.bind_tools(tools)
        by_name = {tool.name: tool for tool in tools}

        messages: list[AnyMessage] = [
            SystemMessage(content=KNOWLEDGE_SYSTEM),
            HumanMessage(
                content=KNOWLEDGE_USER.format(
                    dilemma=dilemma.statement or "（未明确）",
                    greeds=_bullet([m.statement for m in state.get("greeds", [])]),
                    fears=_bullet([m.statement for m in state.get("fears", [])]),
                    constraints=_bullet([c.statement for c in state.get("constraints", [])]),
                    contradictions=_bullet([c.description for c in state.get("contradictions", [])]),
                )
            ),
        ]

        closing = ""
        closing_is_grounded = False
        for _round in range(settings.agent_max_tool_rounds):
            answer = await bound.ainvoke(messages)
            messages.append(answer)
            calls = getattr(answer, "tool_calls", None) or []
            if not calls:
                # The model answered instead of calling another tool. If it had
                # already retrieved something this round, that answer has the
                # material in context and *is* the summary — asking it to repeat
                # itself would be a wasted round trip.
                closing = _text_of(answer)
                closing_is_grounded = bool(trace.citations)
                break
            for call in calls:
                tool = by_name.get(call["name"])
                if tool is None:
                    continue
                try:
                    output = await tool.ainvoke(call.get("args") or {})
                except Exception as exc:  # noqa: BLE001
                    output = f"检索失败：{type(exc).__name__}"
                    logger.warning("tool_failed", extra={"tool": call["name"], "error": type(exc).__name__})
                messages.append(ToolMessage(content=str(output), tool_call_id=call["id"]))

        if not trace.citations:
            await _fallback_retrieval(session=session, settings=settings, state=state, trace=trace)

        grounded_closing = closing if closing_is_grounded else ""
        notes_text = _notes_from(grounded_closing, trace) or await _summarize_knowledge(model, messages, trace)
    logger.info(
        "knowledge_done",
        extra={
            "session_id": state.get("session_id"),
            "citations": len(trace.citations),
            "tool_calls": len(trace.calls),
        },
    )
    return {
        "citations": trace.citations,
        "knowledge_notes": notes_text,
        "phase": AgentPhase.KNOWLEDGE,
        "tool_calls": trace.calls,
    }


async def _fallback_retrieval(
    *,
    session: Any,
    settings: Settings,
    state: AleaState,
    trace: ToolTrace,
) -> None:
    """Retrieve once with a query derived from the analysis.

    Some OpenAI-compatible endpoints — most local and self-hosted models — do not
    implement tool calling at all. Without this, such a provider would silently
    lose its theoretical grounding and produce advice with no basis, which is
    precisely what this project must not do.

    Only runs when the model retrieved nothing itself, and results are subject to
    the same similarity floor, so an irrelevant fallback still comes back empty.
    """
    dilemma = state.get("dilemma", Dilemma())
    parts = [
        dilemma.statement,
        *(item.description for item in state.get("contradictions", [])[:2]),
        *(item.statement for item in state.get("fears", [])[:2]),
        *(item.statement for item in state.get("greeds", [])[:2]),
    ]
    query = " ".join(part for part in parts if part).strip()
    if not query:
        return

    logger.info(
        "knowledge_fallback_retrieval",
        extra={"session_id": state.get("session_id"), "reason": "模型未调用检索工具"},
    )
    await search_knowledge(
        session=session,
        settings=settings,
        query=query,
        top_k=settings.rag_top_k,
        trace=trace,
        source_types=["curated_note", "verified_excerpt"],
        tool_name="fallback_retrieval",
    )
    await search_knowledge(
        session=session,
        settings=settings,
        query=query,
        top_k=2,
        trace=trace,
        source_types=["case"],
        tool_name="fallback_retrieval",
    )


def _notes_from(text: str, trace: ToolTrace) -> list[str]:
    """Accept a model-written summary only when it stands on retrieved material."""
    if not text.strip() or not trace.citations:
        return []
    lines = [line.strip(" -•\t#") for line in text.splitlines() if line.strip()]
    return [line for line in lines if len(line) > 4][:6]


async def _summarize_knowledge(model: Any, messages: list[AnyMessage], trace: ToolTrace) -> list[str]:
    if not trace.citations:
        return ["知识库中没有检索到与当前困境直接相关的条目，本次分析将不使用理论引用。"]

    # The tool transcript is re-rendered into a single clean turn rather than
    # replayed verbatim: a history containing tool calls but no bound tools is
    # rejected by some OpenAI-compatible providers.
    found = "\n\n".join(
        f"来源 {index}｜{citation.document_title}"
        f"（{'原文摘录' if citation.verified else '整理笔记'}，{citation.source_path}）\n{citation.excerpt[:500]}"
        for index, citation in enumerate(trace.citations, start=1)
    )
    request = HumanMessage(
        content=(
            "以下是检索到的资料：\n\n"
            f"{found}\n\n"
            "请用 3-6 条简短中文要点总结与当前困境相关的理论，"
            "每条注明来源编号（格式：要点 —— 来源 N）。只写上面确实有的内容。"
        )
    )
    try:
        response = await model.ainvoke([messages[0], request])
        text = _text_of(response)
    except Exception as exc:  # noqa: BLE001
        logger.warning("knowledge_summary_failed", extra={"error": type(exc).__name__})
        return [citation.excerpt[:160] for citation in trace.citations[:4]]
    lines = [line.strip(" -•\t") for line in text.splitlines() if line.strip()]
    return [line for line in lines if len(line) > 4][:6]


# ====================================================================== decide
async def decide(state: AleaState) -> dict[str, Any]:
    deps = _deps()
    settings = deps.settings
    turns = _transcript(state.get("messages", []))
    dilemma = state.get("dilemma", Dilemma())

    unconfirmed = [
        *(m.statement for m in state.get("greeds", []) if m.status != ConfirmationStatus.CONFIRMED),
        *(m.statement for m in state.get("fears", []) if m.status != ConfirmationStatus.CONFIRMED),
        *(c.statement for c in state.get("constraints", []) if c.status != ConfirmationStatus.CONFIRMED),
    ]
    citations = state.get("citations", [])
    source_list = (
        "\n".join(
            f"来源 {index}｜{c.document_title}"
            f"（{'原文摘录' if c.verified else '整理笔记'}，{c.source_path}，相似度 {c.score}）\n    {c.excerpt[:400]}"
            for index, c in enumerate(citations, start=1)
        )
        or "（本轮没有检索到可引用的来源）"
    )

    model = llm.get_chat_model(temperature=0.3)
    decision = await llm.call_structured(
        model,
        Decision,
        [
            SystemMessage(content=BASE_IDENTITY + "\n\n" + DECIDE_SYSTEM),
            HumanMessage(
                content=DECIDE_USER.format(
                    dilemma=dilemma.statement or "（未明确）",
                    options="\n".join(f"- {opt.id}｜{opt.label}：{opt.description}" for opt in dilemma.options)
                    or "（未明确列出选项）",
                    greeds=_bullet(
                        [m.statement for m in state.get("greeds", []) if m.status == ConfirmationStatus.CONFIRMED]
                    ),
                    fears=_bullet(
                        [m.statement for m in state.get("fears", []) if m.status == ConfirmationStatus.CONFIRMED]
                    ),
                    constraints=_bullet(
                        [
                            f"{c.statement}（{'硬约束' if c.hard else '软约束'}）"
                            for c in state.get("constraints", [])
                            if c.status == ConfirmationStatus.CONFIRMED
                        ]
                    ),
                    unconfirmed=_bullet(unconfirmed),
                    knowledge_notes=_bullet(state.get("knowledge_notes", [])),
                    citations=source_list,
                    recalled_block=state.get("recalled_block") or "（无）",
                    transcript=_render_turns(turns[-14:]),
                )
            ),
        ],
        node="decide",
    )

    decision = _enforce_decision(decision, state, settings)
    reply = render_decision(decision, dilemma)
    final = decision.status == "recommend"

    logger.info(
        "decide_done",
        extra={
            "session_id": state.get("session_id"),
            "status": decision.status,
            "confidence": decision.confidence,
            "citations": len(decision.citations),
        },
    )
    return {
        "decision": decision,
        "assistant_reply": reply,
        "messages": [AIMessage(content=reply)],
        "phase": AgentPhase.DECIDED if final else AgentPhase.DECIDING,
        "turn_count": int(state.get("turn_count", 0)) + 1,
    }


def _enforce_decision(decision: Decision, state: AleaState, settings: Settings) -> Decision:
    """Post-conditions the model must not be able to talk itself out of."""
    # 1. A low-sufficiency recommendation is downgraded to an explicit defer.
    if (
        decision.status == "recommend"
        and float(state.get("sufficiency", 0.0)) < settings.agent_sufficiency_threshold * 0.75
    ):
        decision.status = "defer"
        decision.defer_reason = decision.defer_reason or "目前对关键动机的把握还不够，给出确定建议可能误导你。"

    # 2. Citations must be ones we actually retrieved.
    #
    # `decision.citations` means "the sources this advice stands on", not "the
    # sources the model happened to name". Anything invented is dropped; if the
    # model named nothing valid, the retrieved set — which was literally put in
    # front of it for this decision — stands in, so an advice card can never
    # look unsourced when a real retrieval did happen.
    retrieved = list(state.get("citations", []))
    allowed = {c.chunk_id: c for c in retrieved}
    chosen = [allowed[c.chunk_id] for c in decision.citations if c.chunk_id in allowed]
    if not chosen and decision.status == "recommend":
        chosen = retrieved[:3]
    decision.citations = chosen

    # 3. A recommendation needs a chosen option.
    if decision.status == "recommend" and not (decision.chosen_option_id or decision.chosen_label):
        decision.status = "defer"
        decision.defer_reason = "还需要先确认你真正愿意承担哪一种代价。"

    # 4. Risk flags always win.
    if state.get("risk_flags"):
        decision.status = "hold_safety"
        decision.safety_note = decision.safety_note or "、".join(state["risk_flags"])

    decision.is_final = decision.status == "recommend"
    if not decision.headline:
        decision.headline = {
            "recommend": "建议已给出",
            "defer": "暂缓给出建议",
            "hold_safety": "暂停建议",
        }[decision.status]
    return decision


def render_decision(decision: Decision, dilemma: Dilemma) -> str:
    """Render the structured decision as markdown.

    Rendered from the object rather than written by a second model call, so the
    prose and the card in the UI can never disagree.
    """
    lines: list[str] = []
    if decision.status == "hold_safety":
        lines.append("## 我需要先停下这里的分析")
        lines.append(decision.safety_note or "你描述的情况超出了我适合给出建议的范围。")
        lines.append("")
        lines.append("请优先联系现实中可以帮到你的人或机构：全国 24 小时心理援助热线 12356；紧急情况请拨 120 / 110。")
        return "\n".join(lines)

    if decision.status == "defer":
        lines.append("## 暂时不给你结论")
        lines.append(decision.defer_reason or "信息还不够，现在给建议不负责任。")
        if decision.missing_information:
            lines.append("")
            lines.append("**我还需要知道：**")
            lines.extend(f"- {item}" for item in decision.missing_information)
        if decision.reasons:
            lines.append("")
            lines.append("**目前已经能看清的部分：**")
            lines.extend(f"- {item}" for item in decision.reasons)
        return "\n".join(lines)

    lines.append(f"## {decision.headline}")
    lines.append("")
    lines.append(decision.why_this_choice)
    if decision.reasons:
        lines.append("")
        lines.append("**为什么是它：**")
        lines.extend(f"- {item}" for item in decision.reasons)
    if decision.must_give_up:
        lines.append("")
        lines.append("**选择它，你必须放弃：**")
        lines.extend(f"- {item}" for item in decision.must_give_up)
    if decision.risks:
        lines.append("")
        lines.append("**风险与应对：**")
        for risk in decision.risks:
            level = {"low": "低", "medium": "中", "high": "高", "critical": "极高"}.get(str(risk.level), "中")
            mitigation = f"（应对：{risk.mitigation}）" if risk.mitigation else ""
            lines.append(f"- [{level}] {risk.description}{mitigation}")
    if decision.bottom_lines:
        lines.append("")
        lines.append("**底线：**")
        lines.extend(f"- {item}" for item in decision.bottom_lines)
    if decision.reassess_if:
        lines.append("")
        lines.append("**出现以下情况请重新评估：**")
        lines.extend(f"- {item}" for item in decision.reassess_if)
    if decision.citations:
        lines.append("")
        lines.append("**依据的理论来源：**")
        lines.extend(
            f"- {c.document_title}（{'原文摘录' if c.verified else '本项目整理笔记'}｜{c.source_path}）"
            for c in decision.citations
        )
    lines.append("")
    lines.append(f"_置信度：{decision.confidence:.2f}_")
    return "\n".join(lines)


# ====================================================================== safety
async def safety(state: AleaState) -> dict[str, Any]:
    risk_flags = state.get("risk_flags", [])
    model = llm.get_chat_model(temperature=0.3, streaming=True)
    response = await model.ainvoke(
        [
            SystemMessage(content=BASE_IDENTITY + "\n\n" + SAFETY_SYSTEM),
            HumanMessage(
                content=SAFETY_USER.format(
                    risk_flags="、".join(risk_flags),
                    latest_user_text=state.get("latest_user_text", ""),
                )
            ),
        ]
    )
    reply = _text_of(response).strip()
    if "12356" not in reply:
        reply = (reply + SAFETY_RESOURCES).strip()
    decision = Decision(
        status="hold_safety",
        headline="暂停建议",
        safety_note="、".join(risk_flags),
        is_final=False,
    )
    logger.warning("safety_hold", extra={"session_id": state.get("session_id"), "flags": risk_flags})
    return {
        "assistant_reply": reply,
        "messages": [AIMessage(content=reply)],
        "decision": decision,
        "phase": AgentPhase.PAUSED,
        "turn_count": int(state.get("turn_count", 0)) + 1,
    }


# ==================================================================== finalize
async def finalize(state: AleaState) -> dict[str, Any]:
    """Persist what this turn taught us, then clear the per-turn accumulators."""
    deps = _deps()
    analysis = _state_to_analysis(state)
    turns = _transcript(state.get("messages", []))
    changes: list[dict[str, Any]] = []
    candidates: list[Any] = []

    if not state.get("risk_flags"):
        try:
            candidates = await extract_candidates(
                settings=deps.settings,
                analysis=analysis,
                turns=turns,
                latest_user_text=state.get("latest_user_text", ""),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("memory_extract_failed", extra={"error": type(exc).__name__})

    if candidates:
        try:
            async with deps.database.session() as session:
                outcomes = await deps.memory_service.persist(
                    session,
                    user_id=deps.user_id,
                    session_id=uuid.UUID(state["session_id"]) if state.get("session_id") else None,
                    message_id=None,
                    candidates=candidates,
                )
                changes = [
                    {
                        "action": outcome.action,
                        "memory_id": str(outcome.memory.id),
                        "content": outcome.memory.content,
                        "kind": outcome.memory.kind,
                        "status": outcome.memory.status,
                    }
                    for outcome in outcomes
                ]
        except Exception as exc:  # noqa: BLE001
            logger.warning("memory_persist_failed", extra={"error": type(exc).__name__})

    logger.info(
        "finalize_done",
        extra={
            "session_id": state.get("session_id"),
            "memory_changes": len(changes),
            "candidates": len(candidates),
        },
    )
    return {
        "memory_changes": changes,
        "memory_candidates": [],
        "confirmations_just_resolved": False,
    }


def _state_to_analysis(state: AleaState) -> Any:
    from app.schemas.agent import SessionAnalysis

    return SessionAnalysis(
        dilemma=state.get("dilemma", Dilemma()),
        greeds=state.get("greeds", []),
        fears=state.get("fears", []),
        constraints=state.get("constraints", []),
        contradictions=state.get("contradictions", []),
        open_questions=state.get("open_questions", []),
        confirmations=state.get("confirmations", []),
        recalled_memories=state.get("recalled_memory_refs", []),
        citations=state.get("citations", []),
        sufficiency=float(state.get("sufficiency", 0.0)),
        sufficiency_rationale=state.get("sufficiency_rationale", ""),
        risk_flags=state.get("risk_flags", []),
    )
