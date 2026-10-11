"""Graph state.

One dict flows through every node. It is checkpointed after each step, so a
session can be resumed — or a run inspected — exactly where it stopped.

Conventions:
* Structured analysis (dilemma/motives/constraints) is Pydantic, not prose. The
  frontend renders it and the user can correct it, so it must be typed.
* ``messages`` is the only field with a reducer; every node writes the analysis
  fields it owns wholesale.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from app.db.enums import AgentPhase
from app.schemas.agent import (
    Citation,
    Confirmation,
    Constraint,
    Contradiction,
    Decision,
    Dilemma,
    MemoryCandidate,
    MemoryRef,
    Motive,
)


class AleaState(TypedDict, total=False):
    # ------------------------------------------------------------- identity
    user_id: str
    session_id: str

    # --------------------------------------------------------- conversation
    messages: Annotated[list[AnyMessage], add_messages]
    latest_user_text: str
    transcript: list[tuple[str, str]]

    # ------------------------------------------------------------- analysis
    dilemma: Dilemma
    greeds: list[Motive]
    fears: list[Motive]
    constraints: list[Constraint]
    contradictions: list[Contradiction]
    open_questions: list[str]
    confirmations: list[Confirmation]
    #: Ids of motives/constraints the user corrected or denied; never re-added.
    retired_ids: list[str]
    sufficiency: float
    sufficiency_rationale: str
    risk_flags: list[str]

    # --------------------------------------------------------------- memory
    recalled_block: str
    recalled_memory_refs: list[MemoryRef]
    recalled_event_count: int
    memory_candidates: list[MemoryCandidate]
    memory_changes: list[dict[str, Any]]

    # ------------------------------------------------------------ knowledge
    citations: list[Citation]
    knowledge_notes: list[str]

    # ------------------------------------------------------------- decision
    decision: Decision | None

    # -------------------------------------------------------------- control
    phase: AgentPhase
    turn_count: int
    route: str
    route_reason: str
    assistant_reply: str
    errors: list[str]
    #: Confirmation ids already put to the user, so the agent does not nag.
    asked_confirmation_ids: list[str]
    #: Question topics already explored without new information.
    asked_directions: list[str]
    #: Set when the user asks for a conclusion outright.
    user_requests_conclusion: bool
    #: Which tools ran this turn (for the debug/quality panel).
    tool_calls: list[dict[str, Any]]
    #: Set when the user answered confirmation cards through the UI.
    confirmations_just_resolved: bool


def empty_state(*, user_id: str, session_id: str) -> AleaState:
    return AleaState(
        user_id=user_id,
        session_id=session_id,
        messages=[],
        transcript=[],
        dilemma=Dilemma(),
        greeds=[],
        fears=[],
        constraints=[],
        contradictions=[],
        open_questions=[],
        confirmations=[],
        retired_ids=[],
        sufficiency=0.0,
        sufficiency_rationale="",
        risk_flags=[],
        recalled_block="",
        recalled_memory_refs=[],
        recalled_event_count=0,
        memory_candidates=[],
        memory_changes=[],
        citations=[],
        knowledge_notes=[],
        decision=None,
        phase=AgentPhase.UNDERSTANDING,
        turn_count=0,
        route="",
        route_reason="",
        assistant_reply="",
        errors=[],
        asked_confirmation_ids=[],
        asked_directions=[],
        user_requests_conclusion=False,
        tool_calls=[],
        confirmations_just_resolved=False,
    )
