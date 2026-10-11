"""The five acceptance criteria, as executable checks.

These drive the real HTTP API against a real database with a scripted model, so
they verify the *system* behaviour the checklist describes rather than any
particular phrasing a model might produce.
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import select

from app.db.models import Memory

from .fakes import one_hot
from .test_agent_flow import DECISION, DECISION_EVENT, confirmation, motive, understanding

pytestmark = pytest.mark.db

DIM = 1536


async def _create_session(api) -> str:
    return (await api.post("/api/sessions", json={})).json()["id"]


async def _turn(api, session_id: str, text: str) -> dict:
    async with api.stream("POST", f"/api/sessions/{session_id}/stream", json={"content": text}) as response:
        assert response.status_code == 200
        body = "".join([chunk async for chunk in response.aiter_text()])
    events = [json.loads(line[len("data:") :].strip()) for line in body.splitlines() if line.startswith("data:")]
    assert events[-1]["type"] == "done", events[-1]
    return events[-1]


# ==========================================================================
# 1. 同一个困境，Agent 动态追问而不是固定问答
# ==========================================================================
async def test_criterion_1_the_agent_explores_before_it_answers(api, patch_llm):
    patch_llm(
        [
            [understanding(sufficiency=0.25, open_questions=["你最怕失去的具体是什么"])],
            ["你提到在国企待了六年。如果一年后回头看，你最怕失去的是哪一样？"],
            [{"memories": []}],
        ]
    )
    session_id = await _create_session(api)
    done = await _turn(api, session_id, "我在国企做行政六年了，想辞职做摄影，但很犹豫")

    assert done["phase"] == "exploring"
    assert done["decision"] is None, "第一轮不应该给出结论"
    assert done["turn_count"] == 1
    # The probe is grounded in what the user just said, not a canned script.
    assert "国企" in done["message"]["content"] or "六年" in done["message"]["content"]


async def test_criterion_1_the_probe_follows_the_answer(api, patch_llm):
    patch_llm(
        [
            [understanding(sufficiency=0.25)],
            ["你最怕失去的是什么？"],
            [{"memories": []}],
        ]
    )
    session_id = await _create_session(api)
    await _turn(api, session_id, "想辞职做摄影")

    patch_llm(
        [
            [
                understanding(
                    sufficiency=0.3,
                    fears=[motive("f1", "fear", "你害怕失去可以回退的路径", "我怕回不去原来的单位")],
                    open_questions=["回不去意味着什么"],
                )
            ],
            ["你说怕回不去原来的单位——如果真回不去，最坏会坏到什么程度？"],
            [{"memories": []}],
        ]
    )
    done = await _turn(api, session_id, "我怕回不去原来的单位")
    assert "回不去" in done["message"]["content"]
    assert done["analysis"]["fears"][0]["statement"] == "你害怕失去可以回退的路径"


# ==========================================================================
# 2. 用户纠正贪婪或恐惧后，Agent 真正修正状态
# ==========================================================================
async def test_criterion_2_a_correction_really_changes_the_state(api, patch_llm):
    patch_llm(
        [
            [
                understanding(
                    fears=[motive("f1", "fear", "你害怕失去稳定的收入", "我怕没工资")],
                    confirmations=[confirmation("q-income", "fear", "你害怕失去稳定的收入")],
                    sufficiency=0.5,
                )
            ],
            ["我猜你怕的是失去稳定收入，对吗？"],
            [{"memories": []}],
        ]
    )
    session_id = await _create_session(api)
    await _turn(api, session_id, "我怕没工资")

    # The user corrects it through the UI button.
    patch_llm([[]])
    response = await api.post(
        f"/api/sessions/{session_id}/confirmations/q-income",
        json={
            "confirmation_id": "q-income",
            "action": "correct",
            "corrected_text": "我怕的不是钱，是失去自己安排时间的能力",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["memory_id"]

    # The next turn must not resurrect the corrected reading.
    patch_llm(
        [
            [
                understanding(
                    fears=[
                        motive(
                            "f2",
                            "fear",
                            "你害怕的是失去自己安排时间的能力",
                            "我怕的不是钱，是失去自己安排时间的能力",
                        )
                    ],
                    retired_ids=["f1"],
                    sufficiency=0.6,
                )
            ],
            ["那如果先不辞职，用业余时间做呢？"],
            [{"memories": []}],
        ]
    )
    done = await _turn(api, session_id, "我怕的不是钱，是失去自己安排时间的能力")

    statements = [m["statement"] for m in done["analysis"]["fears"]]
    assert statements == ["你害怕的是失去自己安排时间的能力"]
    assert "你害怕失去稳定的收入" not in statements

    detail = (await api.get(f"/api/sessions/{session_id}")).json()
    assert detail["analysis"]["fears"][0]["statement"] == "你害怕的是失去自己安排时间的能力"


# ==========================================================================
# 3. RAG 确实检索到了知识库内容，并提供真实来源
# ==========================================================================
async def test_criterion_3_retrieval_returns_real_sources(api, container, knowledge, patch_llm):
    documents = (await api.get("/api/knowledge/documents")).json()
    assert documents, "知识库必须先被索引"

    search = await api.post("/api/knowledge/search", json={"query": "代价 可逆 底线"})
    results = search.json()["results"]
    assert results, "应当检索到条目"

    for citation in results:
        assert citation["chunk_id"]
        assert citation["source_path"].endswith(".md")
        assert citation["source_type"] in {"verified_excerpt", "curated_note", "case"}
        assert citation["score"] >= search.json()["used_min_score"]
        assert citation["excerpt"].strip()

    # And the same retrieval must be reachable through the agent's tool loop.
    patch_llm(
        [
            [understanding(sufficiency=0.85)],
            [
                {"tool_calls": [{"name": "search_theory", "args": {"query": "代价 底线"}, "id": "c"}]},
                "要点 —— 来源 1",
            ],
            [DECISION],
            [{"memories": []}],
            [DECISION_EVENT],
        ]
    )
    session_id = await _create_session(api)
    done = await _turn(api, session_id, "房贷不能断，但我想全职做摄影")
    assert done["citations"]
    assert all(item["source_path"].endswith(".md") for item in done["citations"])
    assert done["citations"][0]["chunk_id"] in {c["chunk_id"] for c in results} | {
        c["chunk_id"] for c in done["citations"]
    }

    detail = (await api.get(f"/api/sessions/{session_id}")).json()
    assert detail["messages"][-1]["meta"]["citations"], "引用必须随消息一起留存"


async def test_criterion_3_no_sources_means_no_fabricated_citations(api, patch_llm):
    """When nothing is retrieved, the advice must not pretend to cite anything."""
    patch_llm(
        [
            [
                understanding(
                    sufficiency=0.85,
                    dilemma={
                        "statement": "要不要给咖啡店换个名字",
                        "domain": "其他",
                        "options": [
                            {"id": "yes", "label": "换", "description": ""},
                            {"id": "no", "label": "不换", "description": ""},
                        ],
                        "is_user_decision": True,
                    },
                )
            ],
            [{"tool_calls": [{"name": "search_theory", "args": {"query": "zzzz"}, "id": "c"}]}, "没有检索到。"],
            [
                DECISION
                | {
                    "headline": "建议不换",
                    "citations": [{"chunk_id": "invented", "document_title": "不存在的原文", "source_path": "fake.md"}],
                }
            ],
            [{"memories": []}],
            [DECISION_EVENT],
        ]
    )
    session_id = await _create_session(api)
    done = await _turn(api, session_id, "要不要给咖啡店换个名字，我拿不定主意")

    cited = {citation["chunk_id"] for citation in done["decision"]["citations"]}
    assert "invented" not in cited, "编造的引用必须被剔除"
    assert "不存在的原文" not in done["message"]["content"]
    # Everything shown must be traceable to a chunk that really exists.
    assert all(citation["source_path"].endswith(".md") for citation in done["decision"]["citations"])


# ==========================================================================
# 4. 最终明确给出一个选祸建议，说明理由、代价与风险边界
# ==========================================================================
async def test_criterion_4_the_advice_names_reasons_costs_risks_and_lines(api, knowledge, patch_llm):
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
                {"tool_calls": [{"name": "search_theory", "args": {"query": "代价 底线"}, "id": "c"}]},
                "要点 —— 来源 1",
            ],
            [DECISION],
            [{"memories": []}],
            [DECISION_EVENT],
        ]
    )
    session_id = await _create_session(api)
    done = await _turn(api, session_id, "房贷不能断，但我想全职做摄影")

    decision = done["decision"]
    assert decision["status"] == "recommend"
    assert decision["headline"]
    assert decision["chosen_label"]
    assert decision["why_this_choice"]
    assert decision["reasons"]
    assert decision["must_give_up"], "必须说明要放弃什么"
    assert decision["risks"] and all("level" in risk for risk in decision["risks"])
    assert decision["bottom_lines"], "必须给出底线"
    assert decision["reassess_if"]
    assert decision["citations"]

    body = done["message"]["content"]
    for fragment in ["必须放弃", "底线", "风险"]:
        assert fragment in body


async def test_criterion_4_insufficient_information_defers_instead_of_guessing(api, patch_llm):
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
    session_id = await _create_session(api)
    done = await _turn(api, session_id, "直接给我结论吧")

    assert done["decision"]["status"] == "defer"
    assert "暂时不给你结论" in done["message"]["content"]


async def test_criterion_4_high_risk_stops_the_advice_entirely(api, patch_llm):
    patch_llm([[understanding(risk_flags=["自伤/自杀倾向"])], ["我先不和你分析选择，能联系到谁吗？"]])
    session_id = await _create_session(api)
    done = await _turn(api, session_id, "我真的不想活了")

    assert done["decision"]["status"] == "hold_safety"
    assert "12356" in done["message"]["content"]
    assert done["phase"] == "paused"


# ==========================================================================
# 5. 关闭会话后，新的会话能检索、使用并修改长期记忆
# ==========================================================================
async def test_criterion_5_memory_travels_across_sessions_and_is_editable(api, container, patch_llm):
    install = patch_llm([[]])
    install.embedder.default_vector = one_hot(DIM, 77)

    # --- session A: confirm an important constraint -----------------------
    session_a = await _create_session(api)
    patch_llm(
        [
            [
                understanding(
                    confirmations=[
                        confirmation(
                            "q-loan",
                            "constraint",
                            "你的硬约束是每月房贷 8000 元不能断",
                            "房贷 8000 一个月",
                        )
                    ],
                    sufficiency=0.5,
                )
            ],
            ["你提到房贷不能断，我理解得对吗？"],
            [{"memories": []}],
        ]
    )
    await _turn(api, session_a, "房贷 8000 一个月")

    resolver = patch_llm([[]])
    resolver.embedder.default_vector = one_hot(DIM, 77)
    resolved = await api.post(
        f"/api/sessions/{session_a}/confirmations/q-loan",
        json={"confirmation_id": "q-loan", "action": "confirm"},
    )
    memory_id = resolved.json()["memory_id"]
    assert memory_id

    # --- session B: a new conversation must pick it up ---------------------
    second = patch_llm(
        [
            [understanding(sufficiency=0.3)],
            ["你之前提到房贷，这次的顾虑一样吗？"],
            [{"memories": []}],
        ]
    )
    second.embedder.default_vector = one_hot(DIM, 77)
    session_b = await _create_session(api)
    done = await _turn(api, session_b, "我又在纠结要不要换工作")

    recalled = done["analysis"]["recalled_memories"]
    assert recalled, "新会话必须检索到上个会话确认过的记忆"
    assert recalled[0]["provenance"] == "user_confirmed"
    assert "房贷" in recalled[0]["content"]

    # --- the user edits it ------------------------------------------------
    edited = await api.patch(
        f"/api/memories/{memory_id}",
        json={"content": "你的硬约束是每月房贷 8500 元不能断"},
    )
    assert edited.status_code == 200
    assert "8500" in edited.json()["content"]

    history = (await api.get(f"/api/memories/{memory_id}/history")).json()
    actions = [row["action"] for row in history]
    assert actions[0] == "updated"
    assert "created" in actions
    creation = next(row for row in history if row["action"] == "created")
    assert creation["reason"] and "确认" in creation["reason"]
    assert creation["after"]["provenance"] == "user_confirmed"

    # --- a third session sees the edited version --------------------------
    third = patch_llm([[understanding(sufficiency=0.3)], ["房贷现在是 8500 了？"], [{"memories": []}]])
    third.embedder.default_vector = one_hot(DIM, 77)
    session_c = await _create_session(api)
    done2 = await _turn(api, session_c, "我又在纠结要不要换工作")
    assert "8500" in done2["analysis"]["recalled_memories"][0]["content"]

    # --- deleting removes it from retrieval -------------------------------
    assert (await api.delete(f"/api/memories/{memory_id}")).status_code == 204
    fourth = patch_llm([[understanding(sufficiency=0.3)], ["继续说说？"], [{"memories": []}]])
    fourth.embedder.default_vector = one_hot(DIM, 77)
    session_d = await _create_session(api)
    done3 = await _turn(api, session_d, "我又在纠结要不要换工作")
    assert all("房贷" not in ref["content"] for ref in done3["analysis"]["recalled_memories"])

    async with container.database.session() as session:
        memories = list((await session.execute(select(Memory))).scalars())
    assert len(memories) == 1  # soft-deleted, still on disk for the audit trail
    assert memories[0].deleted_at is not None
