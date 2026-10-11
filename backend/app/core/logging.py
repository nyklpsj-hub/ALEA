"""Structured logging with built-in redaction of user-authored content."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from app.core.config import Settings

_RESERVED = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "taskName",
    "message",
}


def redact(text: str | None) -> str:
    """Return a log-safe rendering of possibly private text.

    No fragment of the original is ever emitted — only its length and a short
    digest, which is enough to correlate log lines about the same message
    without writing the message itself into a log file.

    Used for every user-authored string that reaches a log line unless
    ``LOG_USER_CONTENT=true`` was explicitly set by the operator.
    """
    if not text:
        return ""
    collapsed = " ".join(str(text).split())
    digest = hashlib.sha1(collapsed.encode("utf-8")).hexdigest()[:8]
    return f"<text len={len(collapsed)} sha1={digest}>"


class JsonFormatter(logging.Formatter):
    def __init__(self, *, include_user_content: bool) -> None:
        super().__init__()
        self._include_user_content = include_user_content

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_"):
                continue
            payload[key] = self._sanitize(key, value)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)

    def _sanitize(self, key: str, value: Any) -> Any:
        if self._include_user_content:
            return value
        if key in {"user_text", "content", "message_text", "preview", "answer", "query"}:
            return redact(value if isinstance(value, str) else str(value))
        return value


def configure_logging(settings: Settings) -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stdout)
    if settings.app_env in {"local"} and sys.stdout.isatty():
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s :: %(message)s", "%H:%M:%S"))
    else:
        handler.setFormatter(JsonFormatter(include_user_content=settings.log_user_content))
    root.addHandler(handler)
    root.setLevel(settings.log_level)

    for noisy in ("httpx", "httpcore", "openai", "urllib3", "asyncio", "watchfiles"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
