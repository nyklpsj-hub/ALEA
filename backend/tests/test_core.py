"""Configuration, logging redaction and error mapping."""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.errors import AppError, ConfigurationError, NotFoundError
from app.core.logging import redact

pytestmark = pytest.mark.no_db


def test_settings_requires_an_api_key_for_llm(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("EMBEDDING_API_KEY", "")
    settings = Settings(_env_file=None)
    assert settings.llm_configured is False
    assert settings.resolved_embedding_api_key == ""


def test_embedding_falls_back_to_the_llm_credentials(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "llm-key")
    monkeypatch.setenv("EMBEDDING_API_KEY", "")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "")
    settings = Settings(_env_file=None)
    assert settings.resolved_embedding_api_key == "llm-key"
    assert settings.resolved_embedding_base_url == settings.llm_base_url


def test_local_embedding_provider_swaps_in_a_compatible_model(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    settings = Settings(_env_file=None)
    assert settings.embedding_model == "BAAI/bge-small-zh-v1.5"


def test_log_level_is_validated(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "chatty")
    with pytest.raises(ValueError):
        Settings(_env_file=None)


def test_cors_origins_are_split_and_trimmed(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "http://a.test , http://b.test,")
    settings = Settings(_env_file=None)
    assert settings.cors_origin_list == ["http://a.test", "http://b.test"]


# ------------------------------------------------------------------- logging
def test_redact_never_leaks_user_text():
    secret = "我打算下个月辞掉工作去创业，但家里完全不知道这件事"
    masked = redact(secret)
    for fragment in ("创业", "辞职", "我打算", secret[:6]):
        assert fragment not in masked
    assert str(len(secret)) in masked


def test_redact_is_stable_for_the_same_text():
    """The digest lets an operator correlate log lines without the content."""
    text = "我在纠结要不要离开现在的城市"
    assert redact(text) == redact(text)
    assert redact(text) != redact(text + "。")


def test_redact_handles_short_and_empty_inputs():
    assert redact("") == ""
    assert redact(None) == ""
    assert redact("很短").startswith("<text len=2")


# -------------------------------------------------------------------- errors
def test_app_error_payload_shape():
    error = NotFoundError("会话不存在")
    assert error.http_status == 404
    assert error.to_payload() == {"error": {"code": "not_found", "message": "会话不存在", "details": {}}}


def test_app_error_accepts_details_and_codes():
    error = ConfigurationError(details={"env": "LLM_API_KEY"})
    payload = error.to_payload()
    assert payload["error"]["code"] == "configuration_error"
    assert payload["error"]["details"] == {"env": "LLM_API_KEY"}
    assert isinstance(error, AppError)
