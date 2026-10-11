"""The decision workflow, driven end to end with scripted model output.

These tests exercise the real graph, the real database, the real checkpointer and
the real retrieval SQL. Only the model and embedding calls are scripted, so what
is verified is the *control flow and the guard rails* — the parts that must
behave predictably regardless of how the model happens to phrase things.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from app.db.enums import ConfirmationStatus, MemoryKind, MemoryProvenance, MemoryStatus
from app.db.models import Memory, Message

from .fakes import one_hot

pytestmark = pytest.mark.db

DIM = 1536


def understanding(**overrides) -> dict:
    payload: dict = {
        "dilemma": {
            "statement": "要不要辞掉稳定的工作去做摄影",
            "domain": "职业",
            "options": [
                {"id": "stay", "label": "留在现职", "description": "继续现在的行政工作"},
                {"id": "leave", "label": "全职做摄影", "description": "加入工作室合伙"},
            ],
            "deadline": "",
            "is_user_decision": True,
            "other_parties": ["伴侣"],
        },
        "greeds": [],
        "fears": [],
        "constraints": [],
        "contradictions": [],
        "open_questions": ["你最怕失去的具体是什么"],
        "confirmations": [],
        "retired_ids": [],
        "sufficiency": 0.2,
        "sufficiency_rationale": "只知道表面选项",
        "risk_flags": [],
        "user_requests_conclusion": False,
    }
    payload.update(overrides)
    return payload


def motive(mid: str, kind: str, statement: str, evidence: str = "", status: str = "confirmed") -> dict:
    return {
        "id": mid,
        "kind": kind,
        "statement": statement,
        "intensity": 4,
        "evidence": evidence,
        "provenance": "user_stated" if evidence else "agent_inferred",
        "confidence": 0.9 if evidence else 0.5,
        "status": status,
        "note": "",
    }


def confirmation(cid: str, topic: str, claim: str, evidence: str = "", status: str = "pending") -> dict:
    return {
        "id": cid,
        "topic": topic,
        "claim": claim,
        "evidence": evidence,
        "question": "我理解得对吗？",
        "status": status,
        "resolution_note": "",
    }


DECISION = {
    "status": "recommend",
    "chosen_option_id": "stay",
    "chosen_label": "留在现职，同时用一年时间把接单量做起来",
    "headline": "建议先不辞职，用一年把摄影做成能覆盖房贷的收入",
    "why_this_choice": "你要的是身份认同，不是立刻的全部收入；而可回退的路径一旦失去就不可逆。",
    "reasons": ["满足你想被认可为摄影师", "不触碰房贷这条底线"],
    "must_give_up": ["这一年的周末和晚上"],
    "risks": [{"description": "接单不足", "level": "medium", "mitigation": "保留六个月现金流"}],
    "bottom_lines": ["不动用应急储备金", "房贷断供前必须重新找工作"],
    "reassess_if": ["一年后接单仍低于房贷"],
    "options_analysis": [
        {
            "option_id": "stay",
            "label": "留在现职",
            "satisfies": ["保住收入"],
            "costs": ["一年的业余时间"],
            "risks": [],
            "reversibility": "reversible",
            "cost_level": 2,
        }
    ],
    "citations": [],
    "confidence": 0.72,
    "defer_reason": "",
    "missing_information": [],
    "safety_note": "",
}

DECISION_EVENT = {
    "title": "要不要辞职做摄影",
    "situation": "在国企做行政六年，想全职做摄影。",
    "greed": ["被认可为摄影师"],
    "fear": ["失去可回退的路径"],
    "cost": ["一年的业余时间"],
    "risks": ["接单不足"],
    "summary": "先不辞职，用一年验证接单能力。",
}


async def _run_turn(service, session_id: uuid.UUID, user_id: uuid.UUID, text: str):
    events = []
    async for event in service.stream_turn(user_id=user_id, session_id=session_id, content=text):
        events.append(event)
    return events, next(event for event in events if event["type"] == "done")


async def _run(container, session_id, user_id, text):
    return await _run_turn(container.chat_service, session_id, user_id, text)


async def _new_session(container, user_id: uuid.UUID) -> uuid.UUID:
    row = await container.chat_service.create_session(user_id=user_id)
    return row.id


# ------------------------------------------------------------------ turn one
async def test_first_turn_explores_instead_of_answering(container, user_id, patch_llm):
    patch_llm(
        [
            [understanding()],
            ["你说到了辞职这件事。如果一年后回头看，你最怕失去的是什么？"],
            [{"memories": []}],
        ]
    )
    session_id = await _new_session(container, user_id)
    events, done = await _run(container, session_id, user_id, "我在国企做行政六年了，想辞职做摄影，但很犹豫")

    assert done["phase"] == "exploring"
    assert done["decision"] is None
    assert "最怕失去" in done["message"].content
    nodes = [event["node"] for event in events if event["type"] == "node"]
    assert nodes[0] == "understand"
    assert "explore" in nodes
    assert "decide" not in nodes


async def test_every_turn_is_persisted(container, user_id, patch_llm):
    patch_llm([[understanding()], ["你在担心什么？"], [{"memories": []}]])
    session_id = await _new_session(container, user_id)
    await _run(container, session_id, user_id, "要不要辞职")

    async with container.database.session() as session:
        rows = list(
            (
                await session.execute(select(Message).where(Message.session_id == session_id).order_by(Message.seq))
            ).scalars()
        )
    assert [row.role for row in rows] == ["user", "assistant"]
    assert [row.seq for row in rows] == [1, 2]


async def test_session_title_follows_the_dilemma(container, user_id, patch_llm):
    patch_llm([[understanding()], ["继续说说？"], [{"memories": []}]])
    session_id = await _new_session(container, user_id)
    await _run(container, session_id, user_id, "要不要辞职")
    detail = await container.chat_service.get_detail(user_id=user_id, session_id=session_id)
    assert detail.title == "要不要辞掉稳定的工作去做摄影"


# --------------------------------------------------------- corrections stick
async def test_user_correction_replaces_the_old_reading(container, user_id, patch_llm):
    """Acceptance item: correcting a fear really changes the stored state."""
    patch_llm(
        [
            [
                understanding(
                    fears=[motive("f1", "fear", "你害怕失去稳定的收入", "我怕没有工资")],
                    sufficiency=0.5,
                )
            ],
            ["所以你最在意的是收入，对吗？"],
            [{"memories": []}],
        ]
    )
    session_id = await _new_session(container, user_id)
    _, done = await _run(container, session_id, user_id, "我怕没有工资")
    assert [m["statement"] for m in done["analysis"]["fears"]] == ["你害怕失去稳定的收入"]

    patch_llm(
        [
            [
                understanding(
                    fears=[
                        motive(
                            "f2",
                            "fear",
                            "你害怕的是失去自己安排时间的能力",
                            "不是怕没钱，是怕再也做不了自己的主",
                        )
                    ],
                    retired_ids=["f1"],
                    sufficiency=0.55,
                )
            ],
            ["明白了，是时间上的自主权。那先做副业呢？"],
            [{"memories": []}],
        ]
    )
    _, done2 = await _run(container, session_id, user_id, "不是怕没钱，是怕再也做不了自己的主")
    statements = [m["statement"] for m in done2["analysis"]["fears"]]
    assert statements == ["你害怕的是失去自己安排时间的能力"]
    assert "你害怕失去稳定的收入" not in statements


async def test_a_confirmed_motive_is_not_lost_when_the_model_omits_it(container, user_id, patch_llm):
    patch_llm(
        [
            [understanding(fears=[motive("f1", "fear", "你害怕失去稳定的收入", "我怕没有工资")])],
            ["第一轮追问"],
            [{"memories": []}],
        ]
    )
    session_id = await _new_session(container, user_id)
    await _run(container, session_id, user_id, "我怕没有工资")

    patch_llm(
        [
            [understanding(greeds=[motive("g1", "greed", "你想被人当成专业人士", "我想被当成专业人士")])],
            ["第二轮追问"],
            [{"memories": []}],
        ]
    )
    _, done = await _run(container, session_id, user_id, "我想被人当成专业人士")

    assert {m["statement"] for m in done["analysis"]["fears"]} == {"你害怕失去稳定的收入"}


# ------------------------------------------------------ confirmation → memory
async def test_confirmation_writes_a_user_confirmed_memory(container, user_id, patch_llm):
    claim = confirmation("q-abc", "fear", "你害怕的其实是失去可回退的路径", "我怕回不去原来的单位")
    patch_llm(
        [
            [
                understanding(
                    fears=[motive("f1", "fear", claim["claim"], claim["evidence"])],
                    confirmations=[claim],
                    sufficiency=0.5,
                )
            ],
            ["我猜你怕的是失去可回退的路径，对吗？"],
            [{"memories": []}],
        ]
    )
    session_id = await _new_session(container, user_id)
    _, done = await _run(container, session_id, user_id, "我怕回不去原来的单位")
    assert any(item["id"] == "q-abc" and item["status"] == "pending" for item in done["analysis"]["confirmations"])

    patch_llm([[]])  # resolving only embeds; no chat call is needed
    analysis, memory_id = await container.chat_service.resolve_confirmation(
        user_id=user_id, session_id=session_id, confirmation_id="q-abc", action="confirm"
    )
    assert memory_id is not None
    resolved = next(item for item in analysis.confirmations if item.id == "q-abc")
    assert resolved.status == ConfirmationStatus.CONFIRMED

    async with container.database.session() as session:
        memory = (await session.execute(select(Memory).where(Memory.id == memory_id))).scalar_one()
    assert memory.provenance == str(MemoryProvenance.USER_CONFIRMED)
    assert memory.status == str(MemoryStatus.ACTIVE)
    assert memory.kind == str(MemoryKind.FEAR)


async def test_correcting_a_claim_stores_the_users_own_wording(container, user_id, patch_llm):
    claim = confirmation("q-fear", "fear", "你害怕的是收入下降")
    patch_llm(
        [
            [understanding(confirmations=[claim], sufficiency=0.5)],
            ["我猜你怕的是收入下降？"],
            [{"memories": []}],
        ]
    )
    session_id = await _new_session(container, user_id)
    await _run(container, session_id, user_id, "说不太清楚")

    patch_llm([[]])
    _, memory_id = await container.chat_service.resolve_confirmation(
        user_id=user_id,
        session_id=session_id,
        confirmation_id="q-fear",
        action="correct",
        corrected_text="我怕的是被人看成失败者",
    )
    async with container.database.session() as session:
        memory = (await session.execute(select(Memory).where(Memory.id == memory_id))).scalar_one()
    assert memory.content == "我怕的是被人看成失败者"
    assert memory.provenance == str(MemoryProvenance.USER_STATED)


async def test_rejecting_a_claim_writes_no_memory(container, user_id, patch_llm):
    claim = confirmation("q-no", "greed", "你想要更高的收入")
    patch_llm(
        [
            [understanding(confirmations=[claim], sufficiency=0.5)],
            ["你想要的是更高的收入吗？"],
            [{"memories": []}],
        ]
    )
    session_id = await _new_session(container, user_id)
    await _run(container, session_id, user_id, "说不太清楚")

    patch_llm([[]])
    analysis, memory_id = await container.chat_service.resolve_confirmation(
        user_id=user_id, session_id=session_id, confirmation_id="q-no", action="reject"
    )
    assert memory_id is None
    rejected = next(item for item in analysis.confirmations if item.id == "q-no")
    assert rejected.status == ConfirmationStatus.REJECTED


# ------------------------------------------------- knowledge → decide → cost
async def test_sufficient_information_produces_a_grounded_recommendation(container, knowledge, user_id, patch_llm):
    """Acceptance items: real retrieval with real sources, plus an explicit recommendation."""
    patch_llm(
        [
            [
                understanding(
                    sufficiency=0.85,
                    greeds=[motive("g1", "greed", "你想被认可为摄影师", "我不想只当拍得还行的行政")],
                    fears=[motive("f1", "fear", "你害怕失去可回退的路径", "我怕回不去")],
                    constraints=[
                        {
                            "id": "c1",
                            "statement": "房贷每月 8000 元不能断",
                            "kind": "money",
                            "hard": True,
                            "evidence": "房贷 8000 一个月",
                            "provenance": "user_stated",
                            "confidence": 0.95,
                            "status": "confirmed",
                        }
                    ],
                )
            ],
            [
                {
                    "tool_calls": [
                        {
                            "name": "search_theory",
                            "args": {"query": "代价 可逆 底线"},
                            "id": "call-1",
                        }
                    ]
                },
                "可逆的代价优先于不可逆的代价 —— 来源 1\n底线必须写成具体动作 —— 来源 2",
            ],
            [DECISION],
            [{"memories": []}],
            [DECISION_EVENT],
        ]
    )

    session_id = await _new_session(container, user_id)
    events, done = await _run(container, session_id, user_id, "房贷不能断，但我想全职做摄影")

    assert done["citations"], "知识库检索必须返回来源"
    assert all(citation["source_path"].endswith(".md") for citation in done["citations"])
    assert any(citation["score"] >= 0.0 for citation in done["citations"])
    assert done["phase"] == "decided"
    assert done["decision"]["status"] == "recommend"
    assert done["decision"]["chosen_label"] == "留在现职，同时用一年时间把接单量做起来"
    assert done["decision"]["must_give_up"]
    assert done["decision"]["bottom_lines"]
    assert "必须放弃" in done["message"].content

    nodes = [event["node"] for event in events if event["type"] == "node"]
    assert nodes[:3] == ["understand", "recall", "route"]
    assert "knowledge" in nodes and "decide" in nodes

    detail = await container.chat_service.get_detail(user_id=user_id, session_id=session_id)
    assert detail.has_decision
    assert detail.decision is not None


async def test_decision_event_is_recorded_after_a_recommendation(container, knowledge, user_id, patch_llm):
    from app.db.models import DecisionEvent

    patch_llm(
        [
            [understanding(sufficiency=0.85)],
            [
                {"tool_calls": [{"name": "search_theory", "args": {"query": "代价"}, "id": "c1"}]},
                "要点 —— 来源 1",
            ],
            [DECISION],
            [{"memories": []}],
            [DECISION_EVENT],
        ]
    )
    session_id = await _new_session(container, user_id)
    await _run(container, session_id, user_id, "房贷不能断，但我想全职做摄影")

    async with container.database.session() as session:
        events = list((await session.execute(select(DecisionEvent))).scalars())
    assert len(events) == 1
    assert events[0].title == "要不要辞职做摄影"
    assert events[0].chosen_option


async def test_citations_are_attached_to_the_rendered_advice(container, knowledge, user_id, patch_llm):
    # Turn one only gathers the theory; the ids are then known to the test.
    patch_llm(
        [
            [understanding(sufficiency=0.85)],
            [
                {"tool_calls": [{"name": "search_theory", "args": {"query": "代价 底线"}, "id": "c1"}]},
                "要点 —— 来源 1",
            ],
            [DECISION],
            [{"memories": []}],
            [DECISION_EVENT],
        ]
    )
    session_id = await _new_session(container, user_id)
    _, first = await _run(container, session_id, user_id, "房贷不能断，但我想全职做摄影")
    citation = first["citations"][0]
    chunk_id, title = citation["chunk_id"], citation["document_title"]

    # Turn two: the agent already holds citations, so it goes straight to deciding.
    patch_llm(
        [
            [understanding(sufficiency=0.85)],
            [DECISION | {"citations": [citation]}],
            [{"memories": []}],
            [DECISION_EVENT],
        ]
    )
    _, second = await _run(container, session_id, user_id, "我还在犹豫，不知道该不该答应他")

    assert [c["chunk_id"] for c in second["decision"]["citations"]] == [chunk_id]
    assert title in second["message"].content


# ------------------------------------------------------------------- safety
async def test_crisis_language_halts_the_decision(container, user_id, patch_llm):
    patch_llm(
        [
            [understanding(risk_flags=["自伤/自杀倾向"])],
            ["我注意到你说到了活着这件事本身。这些日子很难熬，我先不和你分析选择。"],
        ]
    )
    session_id = await _new_session(container, user_id)
    events, done = await _run(container, session_id, user_id, "我不想活了")

    assert done["phase"] == "paused"
    assert done["decision"]["status"] == "hold_safety"
    assert "12356" in done["message"].content
    nodes = [event["node"] for event in events if event["type"] == "node"]
    assert nodes == ["understand", "recall", "route", "safety", "finalize"]


async def test_keyword_backstop_flags_a_crisis_even_if_the_model_misses_it(container, user_id, patch_llm):
    patch_llm([[understanding(risk_flags=[])], ["我很担心你"]])
    session_id = await _new_session(container, user_id)
    _, done = await _run(container, session_id, user_id, "我想自杀")

    assert done["decision"] is not None
    assert done["decision"]["status"] == "hold_safety"


async def test_no_memory_is_written_when_a_crisis_is_detected(container, user_id, patch_llm):
    patch_llm([[understanding(risk_flags=["自伤/自杀倾向"])], ["我很担心你"]])
    session_id = await _new_session(container, user_id)
    await _run(container, session_id, user_id, "我被家暴了，想死")

    async with container.database.session() as session:
        rows = list((await session.execute(select(Memory))).scalars())
    assert rows == []


# ----------------------------------------------------------------- bounding
async def test_the_workflow_converges_at_the_hard_turn_limit(container, user_id, patch_llm, settings, monkeypatch):
    """A conversation that never becomes clear must still end — with a deferral."""
    monkeypatch.setattr(settings, "agent_soft_turn_limit", 2)
    monkeypatch.setattr(settings, "agent_hard_turn_limit", 3)

    session_id = await _new_session(container, user_id)
    done = None
    for turn in range(1, 5):
        if turn < 4:
            patch_llm([[understanding(sufficiency=0.1)], [f"第 {turn} 轮追问"], [{"memories": []}]])
        else:
            patch_llm(
                [
                    [understanding(sufficiency=0.1)],
                    [
                        {"tool_calls": [{"name": "search_theory", "args": {"query": "代价"}, "id": "c"}]},
                        "要点 —— 来源 1",
                    ],
                    [DECISION],
                    [{"memories": []}],
                ]
            )
        _, done = await _run(container, session_id, user_id, f"第 {turn} 次回答")

    assert done is not None
    assert done["turn_count"] >= settings.agent_hard_turn_limit
    assert done["decision"] is not None
    assert done["decision"]["status"] == "defer"
    assert "暂时不给你结论" in done["message"].content


async def test_a_recommendation_below_the_sufficiency_floor_is_downgraded(container, user_id, patch_llm):
    patch_llm(
        [
            [understanding(sufficiency=0.40, user_requests_conclusion=True)],
            [
                {"tool_calls": [{"name": "search_theory", "args": {"query": "代价"}, "id": "c"}]},
                "要点 —— 来源 1",
            ],
            [DECISION],
            [{"memories": []}],
        ]
    )
    session_id = await _new_session(container, user_id)
    _, done = await _run(container, session_id, user_id, "直接给我结论吧")

    assert done["decision"] is not None
    assert done["decision"]["status"] == "defer"
    assert "暂时不给你结论" in done["message"].content


# -------------------------------------------------------- session restoration
async def test_session_state_survives_a_restart(container, user_id, patch_llm, settings):
    """The durable snapshot must rebuild the graph state if the thread is gone."""
    patch_llm(
        [
            [understanding(fears=[motive("f1", "fear", "你害怕失去稳定的收入", "我怕没工资")], sufficiency=0.5)],
            ["第一轮"],
            [{"memories": []}],
        ]
    )
    session_id = await _new_session(container, user_id)
    await _run(container, session_id, user_id, "我怕没工资")

    from app.agent.graph import AgentRuntime
    from app.services.chat_service import ChatService

    fresh_runtime = AgentRuntime(settings)
    await fresh_runtime.start()
    fresh = ChatService(
        settings=settings,
        database=container.database,
        runtime=fresh_runtime,
        memory_service=container.memory_service,
    )
    try:
        detail = await fresh.get_detail(user_id=user_id, session_id=session_id)
        assert detail.analysis.fears[0].statement == "你害怕失去稳定的收入"

        patch_llm(
            [
                [understanding(fears=[motive("f1", "fear", "你害怕失去稳定的收入", "我怕没工资")], sufficiency=0.6)],
                ["第二轮"],
                [{"memories": []}],
            ]
        )
        _, done = await _run_turn(fresh, session_id, user_id, "我还是很犹豫")
        assert [m["statement"] for m in done["analysis"]["fears"]] == ["你害怕失去稳定的收入"]
    finally:
        await fresh_runtime.stop()


# ------------------------------------------------------------- cross-session
async def test_a_confirmed_memory_is_recalled_in_a_later_session(container, user_id, patch_llm):
    """Acceptance item: closing a session and starting a new one still uses memory."""
    install = patch_llm([[]])
    install.embedder.default_vector = one_hot(DIM, 42)

    claim = confirmation("q-mem", "constraint", "你的硬约束是每月房贷 8000 元不能断", "房贷 8000 一个月")
    first_session = await _new_session(container, user_id)
    patch_llm(
        [
            [understanding(confirmations=[claim], sufficiency=0.5)],
            ["你提到房贷不能断，我理解得对吗？"],
            [{"memories": []}],
        ]
    )
    await _run(container, first_session, user_id, "房贷 8000 一个月")

    resolver = patch_llm([[]])
    resolver.embedder.default_vector = one_hot(DIM, 42)
    await container.chat_service.resolve_confirmation(
        user_id=user_id, session_id=first_session, confirmation_id="q-mem", action="confirm"
    )

    # A different session, later — the same embedding keeps similarity meaningful.
    second = patch_llm(
        [
            [understanding(sufficiency=0.3)],
            ["你之前提到房贷，这次的情况一样吗？"],
            [{"memories": []}],
        ]
    )
    second.embedder.default_vector = one_hot(DIM, 42)
    second_session = await _new_session(container, user_id)
    _, done = await _run(container, second_session, user_id, "我又在纠结要不要换工作")

    recalled = done["analysis"]["recalled_memories"]
    assert recalled, "新会话必须能检索到上一个会话确认过的记忆"
    assert recalled[0]["provenance"] == "user_confirmed"
    assert "房贷" in recalled[0]["content"]
    assert done["analysis"]["recalled_memories"] != []


async def test_recalled_memory_is_labelled_with_its_provenance(container, user_id, patch_llm):
    install = patch_llm([[]])
    install.embedder.default_vector = one_hot(DIM, 43)

    from app.db.repositories import memory as memory_repo
    from app.memory.service import MemoryService

    async with container.database.session() as session:
        await memory_repo.create_memory(
            session,
            user_id=user_id,
            kind=MemoryKind.FEAR,
            content="你害怕失去稳定的收入",
            provenance=MemoryProvenance.USER_STATED,
            status=MemoryStatus.ACTIVE,
            embedding=one_hot(DIM, 43),
        )
        await memory_repo.create_memory(
            session,
            user_id=user_id,
            kind=MemoryKind.GREED,
            content="你可能更在意别人怎么看你，而不是收入",
            provenance=MemoryProvenance.AGENT_INFERRED,
            status=MemoryStatus.CANDIDATE,
            embedding=one_hot(DIM, 43),
        )

    service = MemoryService(container.settings)
    async with container.database.session() as session:
        result = await service.recall(session, user_id=user_id, query="换工作")

    rendered = result.render()
    assert "用户亲述" in rendered
    assert "未经确认" in rendered
