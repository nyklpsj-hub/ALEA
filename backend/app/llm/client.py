"""Chat and embedding model factories.

The chat model is a plain OpenAI-compatible client, so any provider that speaks
that protocol (OpenAI, DeepSeek, Qwen, Moonshot, vLLM, Ollama, …) works by
changing three environment variables.
"""

from __future__ import annotations

import asyncio
import json
from functools import lru_cache
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, SecretStr
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from app.core.config import Settings, get_settings
from app.core.errors import (
    ConfigurationError,
    EmbeddingError,
    LLMResponseError,
    LLMUnavailableError,
)
from app.core.logging import get_logger

logger = get_logger(__name__)

#: Exceptions that mean "the provider hiccuped"); worth another attempt.
_TRANSIENT = (TimeoutError, ConnectionError, asyncio.TimeoutError)


def build_chat_model(
    settings: Settings,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    streaming: bool = False,
    model: str | None = None,
) -> BaseChatModel:
    if not settings.llm_configured:
        raise ConfigurationError(
            "未配置 LLM_API_KEY，请在 backend/.env 中填写兼容 OpenAI 协议的模型密钥",
            details={"env": "LLM_API_KEY"},
        )
    return ChatOpenAI(
        model=model or settings.llm_model,
        api_key=SecretStr(settings.llm_api_key.get_secret_value()),
        base_url=settings.llm_base_url,
        temperature=settings.llm_temperature if temperature is None else temperature,
        # mypy only sees the pydantic alias (`max_completion_tokens`), but the
        # field name is the documented spelling and produces an identical
        # request payload.
        max_tokens=settings.llm_max_tokens if max_tokens is None else max_tokens,  # type: ignore[call-arg]
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        streaming=streaming,
    )


@lru_cache
def _cached_chat_model(
    temperature: float | None,
    max_tokens: int | None,
    streaming: bool,
) -> BaseChatModel:
    return build_chat_model(get_settings(), temperature=temperature, max_tokens=max_tokens, streaming=streaming)


def get_chat_model(
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    streaming: bool = False,
) -> BaseChatModel:
    """Return a cached chat model for the given generation settings."""
    return _cached_chat_model(temperature, max_tokens, streaming)


@lru_cache
def _cached_embeddings() -> Any:
    settings = get_settings()
    if settings.embedding_provider == "openai":
        if not settings.resolved_embedding_api_key:
            raise ConfigurationError(
                "未配置 EMBEDDING_API_KEY（或 LLM_API_KEY），无法生成向量",
                details={"env": "EMBEDDING_API_KEY"},
            )
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=SecretStr(settings.resolved_embedding_api_key),
            base_url=settings.resolved_embedding_base_url,
            dimensions=settings.embedding_dim,
            check_embedding_ctx_length=False,
            timeout=settings.embedding_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    if settings.embedding_provider == "local":
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ConfigurationError(
                "EMBEDDING_PROVIDER=local 需要安装可选依赖：pip install '.[local-embeddings]'",
                details={"extra": "local-embeddings"},
            ) from exc
        return _LocalEmbeddings(TextEmbedding(model_name=settings.embedding_model))

    raise ConfigurationError(f"未知的 EMBEDDING_PROVIDER: {settings.embedding_provider}")


class _LocalEmbeddings:
    """Offline semantic embeddings backed by fastembed (ONNX, no torch).

    Exposes the two methods of ``langchain_core.embeddings.Embeddings`` that this
    project uses. Kept private and provider-checked so it can never silently
    stand in for the configured remote provider.
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, vec)) for vec in self._model.embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        embeddings = list(self._model.embed([text]))
        return list(map(float, embeddings[0]))

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)


def get_embeddings() -> Any:
    return _cached_embeddings()


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of documents, splitting on the configured batch size."""
    if not texts:
        return []
    settings = get_settings()
    engine = get_embeddings()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), settings.embedding_batch_size):
        batch = texts[start : start + settings.embedding_batch_size]

        @retry(
            stop=stop_after_attempt(settings.llm_max_retries + 1),
            wait=wait_exponential_jitter(initial=0.6, max=8),
            retry=retry_if_exception_type(_TRANSIENT),
            reraise=True,
        )
        async def _run(payload: list[str] = batch) -> list[list[float]]:
            return await engine.aembed_documents(payload)

        try:
            vectors.extend(await _run())
        except _TRANSIENT as exc:
            raise EmbeddingError(f"向量化请求超时：{exc}") from exc
        except Exception as exc:  # noqa: BLE001 - provider errors are opaque
            raise EmbeddingError(f"向量化失败：{type(exc).__name__}") from exc
    return vectors


async def embed_query(text: str) -> list[float]:
    engine = get_embeddings()
    try:
        return await engine.aembed_query(text)
    except Exception as exc:  # noqa: BLE001
        raise EmbeddingError(f"查询向量化失败：{type(exc).__name__}") from exc


def _json_schema_hint(schema: type[BaseModel]) -> str:
    """Describe the required object for providers that only support JSON mode.

    With ``method="json_mode"`` the provider guarantees a JSON object but not its
    shape, so the schema has to travel in the prompt.
    """
    return "只输出一个 JSON 对象，不要解释、不要 Markdown 代码块。它必须符合下面的 JSON Schema：\n" + json.dumps(
        schema.model_json_schema(), ensure_ascii=False
    )


def _why(exc: Exception) -> str:
    """A short, log-safe reason.

    Only the validation *shape* is kept (field names and allowed values), never
    the payload it came from — that payload is derived from the conversation.
    """
    text = " ".join(str(exc).split())
    return text[:240]


async def call_structured[SchemaT: BaseModel](
    model: BaseChatModel,
    schema: type[SchemaT],
    messages: list[BaseMessage],
    *,
    max_retries: int | None = None,
    node: str = "unknown",
) -> SchemaT:
    """Ask the model for a structured object matching ``schema``.

    A rejected object is retried *with the validation error fed back*, which
    turns a hard failure into a correction for the common case of one wrong enum
    or a missing field — a whole turn is too much to lose over that.

    Raises ``LLMUnavailableError`` when the provider still cannot comply, which
    the graph turns into a graceful degradation instead of a 500.
    """
    settings = get_settings()
    method = settings.llm_structured_output_method
    attempts = (max_retries if max_retries is not None else settings.llm_max_retries) + 1
    options: dict[str, Any] = {}
    if method == "function_calling" and settings.llm_tool_choice:
        options["tool_choice"] = settings.llm_tool_choice
    runnable = model.with_structured_output(schema, method=method, include_raw=False, **options)
    base_messages = list(messages)
    if method == "json_mode":
        base_messages = [*base_messages, HumanMessage(content=_json_schema_hint(schema))]

    last_error: Exception | None = None
    repair: HumanMessage | None = None
    for attempt in range(1, attempts + 1):
        attempt_messages = [*base_messages, repair] if repair else base_messages
        try:
            result = await runnable.ainvoke(attempt_messages)
            if isinstance(result, schema):
                return result
            if isinstance(result, dict):
                return schema.model_validate(result)
            raise LLMResponseError("模型未返回结构化结果")
        except (LLMResponseError, ValueError) as exc:
            last_error = exc
            problem = _why(exc)
            logger.warning(
                "structured_output_invalid",
                extra={"node": node, "attempt": attempt, "why": problem},
            )
            repair = HumanMessage(
                content=(
                    "上一次的输出不符合要求，无法解析：\n"
                    f"{problem}\n"
                    "请修正后重新输出，只输出一个符合 schema 的 JSON 对象，不要解释。"
                )
            )
        except _TRANSIENT as exc:
            last_error = exc
            logger.warning("structured_output_timeout", extra={"node": node, "attempt": attempt})
        except Exception as exc:  # noqa: BLE001 - provider SDK raises many shapes
            last_error = exc
            logger.warning(
                "structured_output_failed",
                extra={"node": node, "attempt": attempt, "error": type(exc).__name__},
            )
        if attempt < attempts:
            await asyncio.sleep(min(2 ** (attempt - 1) * 0.5, 4))
    raise LLMUnavailableError(
        f"模型未能返回有效结果（{type(last_error).__name__}）",
        details={"node": node},
    )
