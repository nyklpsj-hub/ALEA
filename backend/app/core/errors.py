"""Domain errors and their HTTP mapping.

Every failure that can reach a client is expressed as an ``AppError`` so the API
layer can map it to a stable machine-readable ``code`` without leaking internals.
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base class for expected, client-visible failures."""

    code = "internal_error"
    http_status = 500
    message = "服务内部错误"

    def __init__(
        self,
        message: str | None = None,
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message or self.message
        self.details = details or {}
        if code:
            self.code = code
        super().__init__(self.message)

    def to_payload(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}


class ConfigurationError(AppError):
    code = "configuration_error"
    http_status = 503
    message = "服务尚未完成配置"


class NotFoundError(AppError):
    code = "not_found"
    http_status = 404
    message = "资源不存在"


class InvalidRequestError(AppError):
    code = "invalid_request"
    http_status = 400
    message = "请求参数不合法"


class ConflictError(AppError):
    code = "conflict"
    http_status = 409
    message = "资源状态冲突"


class LLMUnavailableError(AppError):
    """The upstream model endpoint failed after retries."""

    code = "llm_unavailable"
    http_status = 502
    message = "模型服务暂时不可用，请稍后重试"


class LLMResponseError(AppError):
    """The model answered, but not in the shape we require."""

    code = "llm_invalid_response"
    http_status = 502
    message = "模型返回内容无法解析"


class AgentRunError(AppError):
    code = "agent_run_failed"
    http_status = 500
    message = "决策流程执行失败"


class EmbeddingError(AppError):
    code = "embedding_failed"
    http_status = 502
    message = "文本向量化失败"


class SafetyHoldError(AppError):
    """The agent refuses to continue: a decision here could cause real harm."""

    code = "safety_hold"
    http_status = 200
    message = "当前情境存在严重风险，需要先获得专业支持"
