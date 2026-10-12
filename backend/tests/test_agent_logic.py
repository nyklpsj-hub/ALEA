"""Pure decision logic: risk screening, state reconciliation, decision guards.

None of these need a model or a database — which is exactly why they are worth
testing: they are the guard rails that must hold no matter what the model says.
"""

from __future__ import annotations

import pytest

from app.agent.nodes import (
    Understanding,
    _enforce_decision,
    _reconcile,
    render_decision,
    screen_risk,
)
from app.core.config import get_settings
from app.db.enums import ConfirmationStatus, MemoryProvenance
from app.schemas.agent import (
    Confirmation,
    Constraint,
    Decision,
    Dilemma,
    Motive,
    Option,
    RiskItem,
)

pytestmark = pytest.mark.no_db


# --------------------------------------------------------------- risk screen
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("我觉得活着没什么意思，不想活了", "自伤/自杀倾向"),
        ("我有时候会想伤害自己", "自残倾向"),
        ("我恨不得杀了他", "伤害他人的想法"),
        ("我在家里被家暴过", "人身安全受威胁"),
        ("我欠了高利贷被人催收", "成瘾或严重债务风险"),
    ],
)
def test_crisis_language_is_flagged(text: str, expected: str):
    assert expected in screen_risk(text)


@pytest.mark.parametrize(
    "text",
    [
        "我想辞职，但怕家里反对",
        "这个选择让我很痛苦，也很纠结",
        "我怕创业失败，压力很大",
        "分手以后我一直很难过",
    ],
)
def test_ordinary_hardship_is_not_flagged(text: str):
    """Everyday hard choices are the product's job, not a crisis trigger."""
    assert screen_risk(text) == []


# --------------------------------------------------------------- reconcile
def _state(**overrides):
    base = {
        "greeds": [],
        "fears": [],
        "constraints": [],
        "confirmations": [],
        "latest_user_text": "",
    }
    base.update(overrides)
    return base


def test_confirmed_motive_survives_a_model_that_forgets_it():
    confirmed = Motive(
        id="f1",
        kind="fear",
        statement="你害怕失去稳定的收入",
        evidence="我怕没了工资",
        provenance=MemoryProvenance.USER_CONFIRMED,
        status=ConfirmationStatus.CONFIRMED,
    )
    result = Understanding(dilemma=Dilemma(statement="要不要辞职"), fears=[])
    update = _reconcile(_state(fears=[confirmed]), result)
    assert [m.statement for m in update["fears"]] == ["你害怕失去稳定的收入"]
    assert update["fears"][0].status is ConfirmationStatus.CONFIRMED


def test_rejected_motive_is_dropped():
    rejected = Motive(
        id="f1",
        kind="fear",
        statement="你害怕失去稳定的收入",
        status=ConfirmationStatus.REJECTED,
    )
    result = Understanding(fears=[Motive(kind="fear", statement="你害怕被同行超越")])
    update = _reconcile(_state(fears=[rejected]), result)
    assert [m.statement for m in update["fears"]] == ["你害怕被同行超越"]


def test_motive_without_evidence_cannot_claim_to_be_user_stated():
    result = Understanding(
        greeds=[
            Motive(
                kind="greed",
                statement="你想要更大的舞台",
                evidence="",
                provenance=MemoryProvenance.USER_STATED,
                confidence=0.95,
                status=ConfirmationStatus.CONFIRMED,
            )
        ]
    )
    update = _reconcile(_state(), result)
    motive = update["greeds"][0]
    assert motive.provenance is MemoryProvenance.AGENT_INFERRED
    assert motive.confidence <= 0.6
    assert motive.status is ConfirmationStatus.PENDING


def test_resolved_confirmations_are_not_asked_again():
    resolved = Confirmation(
        id="q-1",
        topic="fear",
        claim="你怕的是失去选择权",
        status=ConfirmationStatus.CORRECTED,
        resolution_note="不是失去选择权，是怕没面子",
    )
    result = Understanding(
        confirmations=[
            Confirmation(id="q-1", topic="fear", claim="你怕的是失去选择权"),
            Confirmation(id="q-2", topic="greed", claim="你想要被认可"),
        ]
    )
    update = _reconcile(_state(confirmations=[resolved]), result)
    ids = [item.id for item in update["confirmations"]]
    assert "q-1" in ids and "q-2" in ids
    assert {c.id: c.status for c in update["confirmations"]}["q-1"] is ConfirmationStatus.CORRECTED


def test_no_more_than_three_questions_per_turn():
    result = Understanding(confirmations=[Confirmation(id=f"q{i}", claim=f"判断{i}") for i in range(6)])
    update = _reconcile(_state(), result)
    assert len([c for c in update["confirmations"] if c.status is ConfirmationStatus.PENDING]) == 3


def test_risk_flags_from_the_model_and_the_keyword_backstop_are_merged():
    result = Understanding(risk_flags=["用户提到过被胁迫"])
    update = _reconcile(_state(latest_user_text="我不想活了"), result)
    assert set(update["risk_flags"]) == {"用户提到过被胁迫", "自伤/自杀倾向"}


def test_a_fear_written_into_the_greeds_list_is_filed_as_a_fear():
    """Weaker models do put a fear in `greeds`; the explicit `kind` label wins.

    Filing by list membership instead would silently turn a fear into a want,
    which then feeds the wrong motive into the recommendation.
    """
    wanted = Motive(kind="greed", statement="你想被认可为专业人士", evidence="我想被认可")
    misplaced = Motive(kind="fear", statement="你害怕失去稳定的收入", evidence="我怕没工资")
    result = Understanding(greeds=[wanted, misplaced])
    update = _reconcile(_state(), result)

    assert [m.statement for m in update["greeds"]] == ["你想被认可为专业人士"]
    assert [m.statement for m in update["fears"]] == ["你害怕失去稳定的收入"]
    assert [m.kind for m in update["fears"]] == ["fear"]


def test_constraints_get_stable_ids():
    result = Understanding(constraints=[Constraint(statement="房贷不能断")])
    update = _reconcile(_state(), result)
    assert update["constraints"][0].id == "c1"


# ----------------------------------------------------------- decision guards
def _decision(**overrides) -> Decision:
    payload = {
        "status": "recommend",
        "chosen_option_id": "stay",
        "chosen_label": "留在现职",
        "headline": "先留下",
    }
    payload.update(overrides)
    return Decision(**payload)


def test_low_sufficiency_recommendation_is_downgraded_to_defer():
    settings = get_settings()
    state = {"sufficiency": 0.1, "risk_flags": [], "citations": []}
    decision = _enforce_decision(_decision(), state, settings)
    assert decision.status == "defer"
    assert decision.defer_reason
    assert decision.is_final is False


def test_recommendation_without_a_chosen_option_becomes_defer():
    settings = get_settings()
    state = {"sufficiency": 0.9, "risk_flags": [], "citations": []}
    decision = _enforce_decision(_decision(chosen_option_id=None, chosen_label=None), state, settings)
    assert decision.status == "defer"


def test_risk_flags_always_override_a_recommendation():
    settings = get_settings()
    state = {"sufficiency": 0.9, "risk_flags": ["人身安全受威胁"], "citations": []}
    decision = _enforce_decision(_decision(), state, settings)
    assert decision.status == "hold_safety"
    assert "人身安全受威胁" in decision.safety_note


def test_citations_must_come_from_what_was_retrieved():
    settings = get_settings()
    from app.schemas.agent import Citation

    real = Citation(chunk_id="real-1", document_title="真实的资料", source_path="a.md")
    invented = Citation(chunk_id="made-up", document_title="编造的引用", source_path="nope.md")
    state = {"sufficiency": 0.9, "risk_flags": [], "citations": [real]}
    decision = _enforce_decision(_decision(citations=[real, invented]), state, settings)
    assert [c.chunk_id for c in decision.citations] == ["real-1"]


def test_an_unsourced_recommendation_falls_back_to_what_was_retrieved():
    """The advice card must never look unsourced when a real retrieval happened."""
    settings = get_settings()
    from app.schemas.agent import Citation

    real = Citation(chunk_id="real-1", document_title="真实的资料", source_path="a.md")
    state = {"sufficiency": 0.9, "risk_flags": [], "citations": [real]}
    decision = _enforce_decision(_decision(citations=[]), state, settings)
    assert [c.chunk_id for c in decision.citations] == ["real-1"]


def test_a_deferral_cites_nothing():
    settings = get_settings()
    from app.schemas.agent import Citation

    real = Citation(chunk_id="real-1", document_title="真实的资料", source_path="a.md")
    state = {"sufficiency": 0.9, "risk_flags": [], "citations": [real]}
    decision = _enforce_decision(_decision(status="defer", chosen_option_id=None, chosen_label=None), state, settings)
    assert decision.citations == []


# ------------------------------------------------------------------ rendering
def test_render_includes_costs_risks_and_bottom_lines():
    decision = _decision(
        why_this_choice="因为它可逆",
        reasons=["满足你的恐惧：保住收入"],
        must_give_up=["一年内不做全职摄影"],
        risks=[RiskItem(description="接单不足", level="high", mitigation="保留备用金")],
        bottom_lines=["不动用应急储备金"],
        reassess_if=["半年后接单仍低于房贷"],
    )
    text = render_decision(decision, Dilemma(statement="要不要辞职"))
    expected = [
        "先留下",
        "因为它可逆",
        "一年内不做全职摄影",
        "接单不足",
        "不动用应急储备金",
        "半年后接单仍低于房贷",
    ]
    for fragment in expected:
        assert fragment in text


def test_render_defer_is_explicit_about_what_is_missing():
    decision = Decision(
        status="defer",
        defer_reason="还不知道你能动用多少现金",
        missing_information=["手上有多少可动用现金"],
    )
    text = render_decision(decision, Dilemma())
    assert "暂时不给你结论" in text
    assert "手上有多少可动用现金" in text


def test_render_safety_points_at_real_help():
    decision = Decision(status="hold_safety", safety_note="人身安全受威胁")
    text = render_decision(decision, Dilemma())
    assert "12356" in text
    assert "选" not in text.split("\n")[0]


def test_options_carry_an_id_even_for_chinese_labels():
    option = Option(id="留下", label="留下", description="")
    assert option.id
