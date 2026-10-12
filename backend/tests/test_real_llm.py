"""Tests against the real, configured model endpoint.

Marked ``llm`` and skipped unless a genuine API key is present. They cost tokens
and are non-deterministic by nature, so they assert *properties* (the reply is
Chinese prose, the analysis is populated, retrieval is on-topic) rather than
exact wording.
"""

from __future__ import annotations

import pytest

from app.agent.nodes import Understanding
from app.llm.client import call_structured, embed_query, get_chat_model

pytestmark = [pytest.mark.llm, pytest.mark.db]


async def test_the_configured_model_returns_structured_output(real_llm):
    from langchain_core.messages import HumanMessage, SystemMessage

    # This provider's model emits reasoning tokens, billed against the same
    # budget as the answer; a small cap starves the tool call entirely.
    model = get_chat_model(temperature=0.0)
    result = await call_structured(
        model,
        Understanding,
        [
            SystemMessage(
                content=(
                    "你是一个决策分析助手。请读懂这段对话并输出结构化理解。"
                    "greeds 是用户想得到的，fears 是用户害怕失去的。"
                )
            ),
            HumanMessage(
                content=(
                    "用户：我在国企做行政六年了，想辞职做摄影，但我最怕的是回不去原来的单位，"
                    "而且房贷 8000 一个月不能断。\n"
                    "Agent：你说怕回不去原来的单位，能具体说说吗？"
                )
            ),
        ],
        node="test_real_model",
    )
    assert result.dilemma.statement
    assert result.sufficiency > 0
    # The model must not claim the user said something they did not.
    for item in [*result.greeds, *result.fears]:
        if item.provenance.value == "user_stated":
            assert item.evidence.strip(), f"{item.statement} 声称是用户亲述却没有原话依据"


async def test_real_embeddings_retrieve_the_right_document(container, real_llm):
    await container.knowledge_service.ingest(force=True)
    results = await container.knowledge_service.search("不可逆的代价和底线", top_k=3)
    assert results, "真实向量模型应当能检索到相关条目"
    assert results[0].score > 0.3
    combined = " ".join(item.document_title + item.excerpt for item in results)
    assert any(keyword in combined for keyword in ("代价", "底线", "可逆"))


async def test_a_real_conversation_turn_produces_a_usable_reply(container, real_llm, user_id):
    session = await container.chat_service.create_session(user_id=user_id)
    events = []
    async for event in container.chat_service.stream_turn(
        user_id=user_id,
        session_id=session.id,
        content="我在现在的公司待了六年，稳定但不开心，有个创业机会在等我，我该走吗？",
    ):
        events.append(event)

    done = next(event for event in events if event["type"] == "done")
    reply = done["message"].content
    assert len(reply) > 30
    assert any("一" <= char <= "鿿" for char in reply), "回复应当是中文"
    assert done["analysis"]["dilemma"]["statement"]
    assert done["turn_count"] == 1

    # First turn: it must be asking, not concluding.
    assert done["decision"] is None
    assert done["phase"] in {"exploring", "confirming"}


async def test_embedding_dimension_matches_the_schema(container, real_llm):
    settings = container.settings
    vector = await embed_query("测试向量维度")
    assert len(vector) == settings.embedding_dim
