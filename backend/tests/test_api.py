"""HTTP layer: contract, error shapes, streaming and per-user scoping."""

from __future__ import annotations

import json
import uuid

import pytest

from app.db.models import User

from .test_agent_flow import DECISION, DECISION_EVENT, understanding

pytestmark = pytest.mark.db


async def _create_session(api) -> str:
    response = await api.post("/api/sessions", json={})
    assert response.status_code == 201
    return response.json()["id"]


# ------------------------------------------------------------------- system
async def test_health_reports_readiness(api):
    response = await api.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["database"] is True
    assert payload["llm_configured"] is True


async def test_runtime_config_never_exposes_secrets(api):
    response = await api.get("/api/config")
    assert response.status_code == 200
    payload = response.json()
    assert "llm_api_key" not in payload
    assert "api_key" not in json.dumps(payload)
    assert payload["llm_model"]


# ----------------------------------------------------------------- sessions
async def test_session_lifecycle(api):
    session_id = await _create_session(api)

    listed = await api.get("/api/sessions")
    assert [row["id"] for row in listed.json()] == [session_id]

    renamed = await api.patch(f"/api/sessions/{session_id}", json={"title": "换城市的犹豫"})
    assert renamed.json()["title"] == "换城市的犹豫"

    archived = await api.patch(f"/api/sessions/{session_id}", json={"status": "archived"})
    assert archived.json()["status"] == "archived"

    deleted = await api.delete(f"/api/sessions/{session_id}")
    assert deleted.status_code == 204
    assert (await api.get(f"/api/sessions/{session_id}")).status_code == 404


async def test_unknown_session_is_a_structured_404(api):
    response = await api.get(f"/api/sessions/{uuid.uuid4()}")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


async def test_a_session_belonging_to_another_user_is_invisible(api, container):
    other_user = uuid.uuid4()
    async with container.database.session() as session:
        session.add(User(id=other_user, display_name="另一个人"))
    from app.db.repositories import sessions as sessions_repo

    async with container.database.session() as session:
        foreign = await sessions_repo.create_session(session, user_id=other_user, title="别人的会话")

    response = await api.get(f"/api/sessions/{foreign.id}")
    assert response.status_code == 404
    assert (await api.get("/api/sessions")).json() == []


async def test_invalid_payload_is_reported_consistently(api):
    session_id = await _create_session(api)
    response = await api.post(f"/api/sessions/{session_id}/messages", json={"content": ""})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


# --------------------------------------------------------------------- chat
async def test_non_streaming_turn_returns_the_analysis(api, patch_llm):
    patch_llm([[understanding()], ["你最怕失去的是什么？"], [{"memories": []}]])
    session_id = await _create_session(api)

    response = await api.post(
        f"/api/sessions/{session_id}/messages",
        json={"content": "要不要辞掉稳定的工作去做摄影"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["phase"] == "exploring"
    assert payload["message"]["role"] == "assistant"
    assert payload["analysis"]["dilemma"]["statement"]
    assert payload["turn_count"] == 1


async def test_streaming_turn_emits_ordered_events(api, patch_llm):
    patch_llm([[understanding()], ["你最怕失去的是什么？"], [{"memories": []}]])
    session_id = await _create_session(api)

    async with api.stream(
        "POST",
        f"/api/sessions/{session_id}/stream",
        json={"content": "要不要辞掉稳定的工作去做摄影"},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join([chunk async for chunk in response.aiter_text()])

    events = [json.loads(line[len("data:") :].strip()) for line in body.splitlines() if line.startswith("data:")]
    kinds = [event["type"] for event in events]
    assert kinds[0] == "start"
    assert kinds[-1] == "done"
    assert "node" in kinds
    done = events[-1]
    assert done["message"]["role"] == "assistant"
    assert done["analysis"]["dilemma"]["statement"]


async def test_streaming_a_recommendation_includes_citations(api, knowledge, patch_llm):
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
    async with api.stream(
        "POST", f"/api/sessions/{session_id}/stream", json={"content": "房贷不能断，但我想全职做摄影"}
    ) as response:
        body = "".join([chunk async for chunk in response.aiter_text()])

    events = [json.loads(line[len("data:") :].strip()) for line in body.splitlines() if line.startswith("data:")]
    done = events[-1]
    assert done["decision"]["status"] == "recommend"
    assert done["citations"]
    assert done["citations"][0]["source_path"].endswith(".md")
    assert done["citations"][0]["source_type"] in {"curated_note", "verified_excerpt", "case"}


async def test_resolving_a_confirmation_over_http(api, patch_llm):
    claim = {
        "id": "q-http",
        "topic": "fear",
        "claim": "你怕的是失去可回退的路径",
        "evidence": "",
        "question": "对吗？",
        "status": "pending",
        "resolution_note": "",
    }
    patch_llm(
        [
            [understanding(confirmations=[claim], sufficiency=0.5)],
            ["我猜你怕的是失去回退路径？"],
            [{"memories": []}],
        ]
    )
    session_id = await _create_session(api)
    await api.post(f"/api/sessions/{session_id}/messages", json={"content": "我怕回不去"})

    patch_llm([[]])
    response = await api.post(
        f"/api/sessions/{session_id}/confirmations/q-http",
        json={"confirmation_id": "q-http", "action": "correct", "corrected_text": "我怕的是被人看轻"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["memory_id"]
    item = next(c for c in payload["analysis"]["confirmations"] if c["id"] == "q-http")
    assert item["status"] == "corrected"
    assert item["claim"] == "我怕的是被人看轻"


async def test_a_missing_model_key_is_a_setup_error_not_a_crash(api, container):
    """Without a key the user must get an actionable 503, not a generic 500."""
    from pydantic import SecretStr

    original = container.settings.llm_api_key
    container.settings.llm_api_key = SecretStr("")
    try:
        session_id = await _create_session(api)
        response = await api.post(f"/api/sessions/{session_id}/messages", json={"content": "要不要辞职"})
    finally:
        container.settings.llm_api_key = original

    assert response.status_code == 503
    payload = response.json()["error"]
    assert payload["code"] == "configuration_error"
    assert "LLM_API_KEY" in payload["message"] or "LLM_API_KEY" in str(payload["details"])


# ------------------------------------------------------------------- memory
async def test_memory_crud_and_history(api, patch_llm):
    install = patch_llm([[]])
    from tests.fakes import one_hot

    install.embedder.default_vector = one_hot(1536, 91)

    created = await api.post(
        "/api/memories",
        json={"kind": "constraint", "content": "房贷每月 8000 元不能断", "importance": 5},
    )
    assert created.status_code == 201
    memory = created.json()
    assert memory["status"] == "active"
    assert memory["provenance"] == "user_stated"

    updated = await api.patch(f"/api/memories/{memory['id']}", json={"content": "房贷每月 8500 元不能断"})
    assert updated.json()["content"] == "房贷每月 8500 元不能断"

    history = await api.get(f"/api/memories/{memory['id']}/history")
    assert [row["action"] for row in history.json()] == ["updated", "created"]

    stats = await api.get("/api/memories/stats")
    assert stats.json()["active"] == 1

    assert (await api.delete(f"/api/memories/{memory['id']}")).status_code == 204
    assert (await api.get("/api/memories")).json() == []

    with_deleted = await api.get("/api/memories", params={"include_deleted": "true"})
    assert len(with_deleted.json()) == 1

    restored = await api.post(f"/api/memories/{memory['id']}/restore")
    assert restored.status_code == 200
    assert (await api.get("/api/memories")).json() != []


async def test_memory_search_returns_scores(api, patch_llm):
    install = patch_llm([[]])
    from tests.fakes import one_hot

    install.embedder.default_vector = one_hot(1536, 92)
    await api.post("/api/memories", json={"kind": "fear", "content": "你害怕失去稳定的收入"})

    response = await api.post("/api/memories/search", json={"query": "收入"})
    assert response.status_code == 200
    hits = response.json()
    assert hits and hits[0]["score"] > 0
    assert hits[0]["memory"]["content"] == "你害怕失去稳定的收入"


async def test_memories_cannot_be_read_across_users(api, container, patch_llm):
    install = patch_llm([[]])
    from tests.fakes import one_hot

    install.embedder.default_vector = one_hot(1536, 93)
    created = await api.post("/api/memories", json={"kind": "goal", "content": "你想换一条职业路径"})
    memory_id = created.json()["id"]

    other_user = uuid.uuid4()
    async with container.database.session() as session:
        session.add(User(id=other_user, display_name="另一个人"))
    # Point the app at the other user for one request.
    container.user_id = other_user
    try:
        assert (await api.get(f"/api/memories/{memory_id}/history")).status_code == 404
        assert (await api.get("/api/memories")).json() == []
        assert (await api.delete(f"/api/memories/{memory_id}")).status_code == 404
    finally:
        container.user_id = uuid.UUID(container.settings.default_user_id)


# ---------------------------------------------------------------- knowledge
async def test_knowledge_documents_and_direct_search(api, knowledge, container, patch_llm):
    # The fixture already ingested once; a second pass over unchanged files must
    # be a no-op rather than duplicating chunks.
    again = await container.knowledge_service.ingest()
    assert again.documents_seen >= 5
    assert again.documents_unchanged == again.documents_seen
    assert again.chunks_written == 0

    documents = await api.get("/api/knowledge/documents")
    assert documents.status_code == 200
    rows = documents.json()
    assert len(rows) >= 5
    assert all(row["chunk_count"] > 0 for row in rows)
    assert {row["source_type"] for row in rows} <= {"curated_note", "verified_excerpt", "case"}

    search = await api.post("/api/knowledge/search", json={"query": "代价 可逆 底线"})
    assert search.status_code == 200
    payload = search.json()
    assert payload["results"], "应当检索到与查询相关的条目"
    top = payload["results"][0]
    assert top["source_path"].endswith(".md")
    assert top["chunk_id"]
    assert top["score"] >= payload["used_min_score"]


async def test_reindex_endpoint(api, patch_llm):
    patch_llm([[]])
    response = await api.post("/api/knowledge/reindex")
    assert response.status_code == 200
    assert response.json()["documents_seen"] >= 5


# ---------------------------------------------------------------- decisions
async def test_decision_events_are_listed_and_annotated(api, knowledge, patch_llm):
    patch_llm(
        [
            [understanding(sufficiency=0.85)],
            [
                {"tool_calls": [{"name": "search_theory", "args": {"query": "代价"}, "id": "c"}]},
                "要点 —— 来源 1",
            ],
            [DECISION],
            [{"memories": []}],
            [DECISION_EVENT],
        ]
    )
    session_id = await _create_session(api)
    await api.post(f"/api/sessions/{session_id}/messages", json={"content": "房贷不能断，但我想全职做摄影"})

    listed = await api.get("/api/decisions")
    assert len(listed.json()) == 1
    event_id = listed.json()[0]["id"]

    updated = await api.patch(f"/api/decisions/{event_id}/outcome", json={"outcome": "半年后接单确实不够，回去上班了"})
    assert updated.status_code == 200
    assert updated.json()["outcome"] == "半年后接单确实不够，回去上班了"


async def test_session_detail_exposes_the_full_picture(api, knowledge, patch_llm):
    patch_llm(
        [
            [understanding(sufficiency=0.85)],
            [
                {"tool_calls": [{"name": "search_theory", "args": {"query": "代价"}, "id": "c"}]},
                "要点 —— 来源 1",
            ],
            [DECISION],
            [{"memories": []}],
            [DECISION_EVENT],
        ]
    )
    session_id = await _create_session(api)
    await api.post(f"/api/sessions/{session_id}/messages", json={"content": "房贷不能断，但我想全职做摄影"})

    detail = (await api.get(f"/api/sessions/{session_id}")).json()
    assert detail["has_decision"] is True
    assert detail["decision"]["status"] == "recommend"
    assert detail["analysis"]["citations"]
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
    assert detail["messages"][1]["meta"]["citations"]


async def test_deleted_sessions_do_not_appear(api):
    session_id = await _create_session(api)
    await api.delete(f"/api/sessions/{session_id}")
    assert (await api.get("/api/sessions")).json() == []
