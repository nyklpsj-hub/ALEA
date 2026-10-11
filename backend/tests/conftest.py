"""Shared fixtures.

Database-backed tests run against a **real** PostgreSQL + pgvector instance —
the same one `docker compose up -d db` or `scripts.local_postgres.py` provides.
Nothing in this project fakes the database, because the parts most likely to
break (pgvector queries, HNSW indexes, checkpointer round-trips) only exist
there.

Configuration is injected through environment variables *before* the app is
imported, so every layer — including `.env` reading — sees the test target.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest

from app.core.compat import configure_event_loop_policy

# Must happen before pytest-asyncio builds the first event loop, otherwise the
# psycopg-backed checkpointer pool can never connect on Windows.
configure_event_loop_policy()

BACKEND_DIR = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- environment
PG_HOST = os.environ.get("ALEA_TEST_PG_HOST", "127.0.0.1")
PG_PORT = int(os.environ.get("ALEA_TEST_PG_PORT", "55432"))
PG_USER = os.environ.get("ALEA_TEST_PG_USER", "alea")
PG_PASSWORD = os.environ.get("ALEA_TEST_PG_PASSWORD", "alea")
TEST_DB = os.environ.get("ALEA_TEST_DB", "alea_test")

os.environ.setdefault("APP_ENV", "test")
os.environ["DATABASE_URL"] = f"postgresql+asyncpg://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{TEST_DB}"
os.environ["CHECKPOINT_DATABASE_URL"] = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{TEST_DB}"

# Real credentials from backend/.env survive into the test session so the
# `llm`-marked tests can talk to the configured provider; everything else gets a
# placeholder because the default suite must never hit the network.
_PLACEHOLDER = "test-key-not-used"
_ENV_FILE = BACKEND_DIR / ".env"
if _ENV_FILE.exists():
    for _raw in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        _line = _raw.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _key, _, _value = _line.partition("=")
        _key, _value = _key.strip(), _value.strip()
        if _key.startswith(("LLM_", "EMBEDDING_")) and _value:
            os.environ.setdefault(_key, _value)

for _key in ("LLM_API_KEY", "EMBEDDING_API_KEY"):
    os.environ.setdefault(_key, _PLACEHOLDER)

os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_DIM", "1536")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ["LOG_USER_CONTENT"] = "true"

REQUIRE_DB = os.environ.get("ALEA_REQUIRE_DB") == "1"

ADMIN_DSN = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/postgres"

TABLES = (
    "memory_audit",
    "decision_events",
    "memories",
    "knowledge_chunks",
    "knowledge_documents",
    "messages",
    "sessions",
    "users",
)


def _database_reachable() -> tuple[bool, str]:
    try:
        import psycopg

        with psycopg.connect(ADMIN_DSN, connect_timeout=5):
            return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


@pytest.fixture(scope="session")
def test_database() -> Iterator[str]:
    """Create the test database if needed and bring it to the latest revision."""
    reachable, error = _database_reachable()
    if not reachable:
        message = (
            f"PostgreSQL 不可达（{ADMIN_DSN}）：{error}\n"
            "先运行：python -m scripts.local_postgres start  （或 docker compose up -d db）"
        )
        if REQUIRE_DB:
            pytest.fail(message, pytrace=False)
        pytest.skip(message)

    import psycopg

    with psycopg.connect(ADMIN_DSN, autocommit=True) as connection:
        exists = connection.execute("SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB,)).fetchone()
        if not exists:
            connection.execute(f'CREATE DATABASE "{TEST_DB}"')

    from alembic.config import Config

    from alembic import command
    from app.core.config import get_settings

    settings = get_settings()
    config = Config(str(BACKEND_DIR / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    config.set_main_option("sqlalchemy.url", settings.database_url)
    config.set_main_option("embedding_dim", str(settings.embedding_dim))
    command.upgrade(config, "head")
    return TEST_DB


@pytest.fixture(scope="session")
def real_llm(settings):
    """Skip `llm`-marked tests unless a real model endpoint is configured."""
    if not settings.llm_configured or settings.llm_api_key.get_secret_value() == _PLACEHOLDER:
        pytest.skip("未配置真实 LLM_API_KEY（在 backend/.env 里填写后重跑）")
    return settings


@pytest.fixture(scope="session")
def settings(test_database: str):  # noqa: ARG001 - ordering dependency
    from app.core.config import get_settings

    return get_settings()


@pytest.fixture(autouse=True)
async def clean_database(settings, request) -> AsyncIterator[None]:
    """Empty every table between tests so they cannot see each other's rows."""
    if "no_db" in request.keywords:
        yield
        return

    from sqlalchemy import text

    from app.db.session import Database

    database = Database(settings)
    try:
        async with database.engine.begin() as connection:
            await connection.execute(text("TRUNCATE " + ", ".join(TABLES) + " RESTART IDENTITY CASCADE"))
        await database.ensure_default_user()
        yield
    finally:
        await database.dispose()


@pytest.fixture
async def database(settings) -> AsyncIterator[object]:
    from app.db.session import Database

    instance = Database(settings)
    try:
        yield instance
    finally:
        await instance.dispose()


@pytest.fixture
def user_id(settings) -> uuid.UUID:
    return uuid.UUID(settings.default_user_id)


# ------------------------------------------------------------------ LLM fakes
@pytest.fixture
def patch_llm(monkeypatch, settings):
    """Install fake chat + embedding providers for the duration of a test.

    Calling the returned object scripts the model calls for one turn and returns
    the same object, so ``install = patch_llm([...])`` gives access to
    ``install.embedder`` (to pin specific vectors) and ``install.models`` (the
    scripted models that were handed out).
    """

    from app.llm import client as llm_client

    from .fakes import EmbeddingStub, make_model_factory

    embedder = EmbeddingStub(dim=settings.embedding_dim)

    def install(scripts: list[list[object]]):
        factory, created = make_model_factory(*scripts)
        monkeypatch.setattr(llm_client, "get_chat_model", factory)

        async def embed_texts(texts: list[str]) -> list[list[float]]:
            return embedder.embed_documents(texts)

        async def embed_query(query: str) -> list[float]:
            return embedder.embed_query(query)

        monkeypatch.setattr(llm_client, "embed_texts", embed_texts)
        monkeypatch.setattr(llm_client, "embed_query", embed_query)
        install.models = created
        return install

    install.embedder = embedder  # type: ignore[attr-defined]
    install.models = []  # type: ignore[attr-defined]
    return install


@pytest.fixture
async def container(settings) -> AsyncIterator[object]:
    """A fully wired application container pointing at the test database."""
    from app.services.container import AppContainer

    instance = await AppContainer.create(settings)
    try:
        yield instance
    finally:
        await instance.shutdown()


@pytest.fixture
async def knowledge(container, settings, patch_llm):
    """Ingest the shipped corpus using the fake embedder.

    Ingestion must happen *after* the embedding provider is stubbed, otherwise
    the test would talk to the real provider (and fail on the placeholder key).

    The stub runs in uniform mode so every query matches every chunk: these
    tests are about the retrieval *plumbing* (SQL, thresholds, citation
    assembly, provenance). Whether the ranking is semantically good is a
    question for the real provider, covered by tests/test_real_llm.py.
    """
    from .fakes import one_hot

    install = patch_llm([[]])
    install.embedder.default_vector = one_hot(settings.embedding_dim, 1)
    await container.knowledge_service.ingest()
    return container.knowledge_service


@pytest.fixture
async def api(settings, container) -> AsyncIterator[object]:
    """An HTTP client wired straight to the ASGI app (no network, no lifespan)."""
    import httpx

    from app.main import create_app

    app = create_app(settings)
    app.state.container = container
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
