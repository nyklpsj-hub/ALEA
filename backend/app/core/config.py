"""Application settings.

All configuration is environment driven (see `.env.example`). Nothing that could
identify a user or grant access to a paid model is ever stored in code.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(BACKEND_DIR / ".env", REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ app
    app_name: str = "Alea"
    app_env: Literal["local", "test", "staging", "production"] = "local"
    debug: bool = False
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    log_level: str = "INFO"
    #: When false (default) user-authored text is never written to logs.
    log_user_content: bool = False
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # ------------------------------------------------------------- database
    database_url: str = "postgresql+asyncpg://alea:alea@127.0.0.1:5432/alea"
    #: LangGraph checkpointer speaks psycopg, not asyncpg.
    checkpoint_database_url: str = "postgresql://alea:alea@127.0.0.1:5432/alea"
    db_pool_size: int = 10
    db_max_overflow: int = 5
    db_statement_timeout_ms: int = 15_000
    #: Seconds to wait for the LangGraph checkpointer connection pool.
    db_pool_open_timeout: float = 20.0

    # ------------------------------------------------------------ identity
    #: V1 runs a single local profile; the column and every query still carry a
    #: user id so that multi-user support is a data change, not a rewrite.
    default_user_id: str = "00000000-0000-0000-0000-000000000001"
    default_user_name: str = "本地用户"

    # ------------------------------------------------------------ llm (chat)
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.5
    #: One budget for every model call, structured ones included. Reasoning
    #: models count their thinking against it, so a value tuned for a
    #: non-reasoning model will truncate the answer — the symptom is a
    #: structured call that returns nothing at all.
    llm_max_tokens: int = 4000
    llm_timeout_seconds: float = 90.0
    llm_max_retries: int = 2
    #: How structured output is requested. `function_calling` is what hosted
    #: providers (OpenAI, DeepSeek, Qwen, …) handle best; `json_mode` is the
    #: fallback for local and self-hosted models that do not implement tools.
    llm_structured_output_method: Literal["function_calling", "json_mode", "json_schema"] = "function_calling"
    #: Passed to the provider for structured calls. "required" is the honest
    #: default: every structured call here *must* return an object, so leaving
    #: the model free to answer in prose only creates intermittency — with a
    #: longer prompt, weaker models do exactly that. Set to "" to let the model
    #: choose, or to a function name for providers that require one.
    llm_tool_choice: str = "required"
    #: Hard ceiling for a single graph run, guards against a hung provider.
    agent_run_timeout_seconds: float = 600.0

    # ------------------------------------------------------- llm (embedding)
    embedding_provider: Literal["openai", "local"] = "openai"
    embedding_base_url: str = ""
    embedding_api_key: SecretStr = SecretStr("")
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    embedding_batch_size: int = 64
    embedding_timeout_seconds: float = 60.0

    # ----------------------------------------------------------------- rag
    knowledge_dir: Path = BACKEND_DIR / "knowledge"
    rag_chunk_size: int = 700
    rag_chunk_overlap: int = 120
    rag_top_k: int = 5
    #: Cosine similarity floor. Knowledge chunks below it are dropped.
    rag_min_score: float = 0.15

    # -------------------------------------------------------------- memory
    memory_top_k: int = 6
    memory_min_score: float = 0.2
    #: Above this cosine similarity a new candidate is merged into the existing row.
    memory_dedup_similarity: float = 0.9
    memory_min_importance: int = 2
    #: Extraction is skipped for very short acknowledgements.
    memory_min_message_chars: int = 12

    # --------------------------------------------------------------- agent
    #: Soft budget: once reached the agent must either decide or explicitly defer.
    agent_soft_turn_limit: int = 8
    #: Hard budget: the graph stops exploring no matter what.
    agent_hard_turn_limit: int = 16
    #: Below this self-reported sufficiency the agent will not force a decision.
    agent_sufficiency_threshold: float = 0.62
    #: Max tool-calling rounds inside the knowledge node.
    agent_max_tool_rounds: int = 3

    # ------------------------------------------------------------ retrieval
    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def resolved_embedding_base_url(self) -> str:
        return self.embedding_base_url or self.llm_base_url

    @property
    def resolved_embedding_api_key(self) -> str:
        key = self.embedding_api_key.get_secret_value()
        return key or self.llm_api_key.get_secret_value()

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key.get_secret_value())

    @field_validator("knowledge_dir", mode="before")
    @classmethod
    def _resolve_knowledge_dir(cls, value: object) -> object:
        if isinstance(value, str) and value and not Path(value).is_absolute():
            return (REPO_ROOT / value).resolve()
        return value

    @field_validator("log_level")
    @classmethod
    def _upper_level(cls, value: str) -> str:
        level = value.upper()
        allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
        if level not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return level

    @model_validator(mode="after")
    def _warn_defaults(self) -> Settings:
        # Local embeddings ship a 512-dimension model; a silent mismatch here
        # would only surface as a database error much later.
        if self.embedding_provider == "local" and self.embedding_model.startswith("text-embedding"):
            object.__setattr__(self, "embedding_model", "BAAI/bge-small-zh-v1.5")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
