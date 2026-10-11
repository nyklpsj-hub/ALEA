"""Test doubles for the model and embedding providers.

These are *transport* fakes, not feature mocks: the real prompts, the real
graph, the real database and the real RAG code all run. Only the two network
calls (chat completion, embedding) are replaced, so the tests exercise the same
logic that production runs.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from langchain_core.messages import AIMessage
from pydantic import BaseModel


class ScriptExhausted(BaseException):
    """Raised when a test consumed more model calls than it scripted.

    Derives from BaseException on purpose: the production retry helpers catch
    ``Exception``, and a misscripted test must fail loudly instead of being
    silently retried into a wrong answer.
    """


class ScriptMismatch(BaseException):
    """A scripted response did not match the schema the caller asked for.

    Also a BaseException, for the same reason: this is a *test* bug (usually a
    script list that no longer lines up with the nodes that run), and swallowing
    it into a retry loop turns a clear mistake into a confusing timeout.
    """


class FakeChatModel:
    """Duck-typed stand-in for ``BaseChatModel``.

    The graph only ever uses ``ainvoke``, ``bind_tools`` and
    ``with_structured_output``, so those are all that need to exist.

    Script items are consumed in call order:
      * ``str``                     -> plain text reply
      * ``dict`` / ``BaseModel``    -> structured output for the requested schema
      * ``{"tool_calls": [...]}``   -> an AIMessage carrying tool calls

    One script per ``get_chat_model`` call, in node order. Two things to keep in
    mind when writing a script:

    * the knowledge node uses one model for both the tool round and the summary
      (2 items when it calls one tool);
    * ``finalize`` skips memory extraction for short user messages
      (``MEMORY_MIN_MESSAGE_CHARS``), which shifts every later script by one.
      A mismatch raises :class:`ScriptMismatch` rather than silently retrying.
    """

    def __init__(self, script: list[Any], *, label: str = "fake") -> None:
        self.script = list(script)
        self.label = label
        self.calls: list[list[Any]] = []

    # ------------------------------------------------------------ internals
    def _pop(self) -> Any:
        if not self.script:
            raise ScriptExhausted(f"{self.label}: 模型调用次数超出脚本")
        return self.script.pop(0)

    def remaining(self) -> int:
        return len(self.script)

    # ------------------------------------------------------------ interface
    def bind_tools(self, tools: list[Any]) -> FakeChatModel:  # noqa: ARG002
        return self

    def with_structured_output(self, schema: type[BaseModel], **kwargs: Any) -> FakeStructured:
        return FakeStructured(self, schema, **kwargs)

    async def ainvoke(self, messages: list[Any], **kwargs: Any) -> AIMessage:  # noqa: ARG002
        self.calls.append(list(messages))
        item = self._pop()
        if isinstance(item, dict) and "tool_calls" in item:
            return AIMessage(content=item.get("content", ""), tool_calls=item["tool_calls"])
        if isinstance(item, str):
            return AIMessage(content=item)
        return AIMessage(content=json.dumps(item, ensure_ascii=False, default=str))


class FakeStructured:
    def __init__(self, model: FakeChatModel, schema: type[BaseModel], **kwargs: Any) -> None:
        self.model = model
        self.schema = schema
        self.options = kwargs

    def _validate(self, item: Any) -> BaseModel:
        if isinstance(item, self.schema):
            return item
        try:
            if isinstance(item, BaseModel):
                return self.schema.model_validate(item.model_dump())
            if isinstance(item, dict):
                return self.schema.model_validate(item)
        except Exception as exc:  # noqa: BLE001 - reported as a test failure
            raise ScriptMismatch(
                f"{self.model.label}: 期望 {self.schema.__name__}，脚本给的是 {item!r}（{type(exc).__name__}）"
            ) from exc
        raise ScriptMismatch(f"{self.model.label}: 期望 {self.schema.__name__} 的结构化输出，收到 {item!r}")

    async def ainvoke(self, messages: list[Any], **kwargs: Any) -> BaseModel:  # noqa: ARG002
        self.model.calls.append(list(messages))
        return self._validate(self.model._pop())

    def invoke(self, messages: list[Any], **kwargs: Any) -> BaseModel:
        return self._validate(self.model._pop())


class EmbeddingStub:
    """Deterministic embeddings.

    Unmapped text gets a hash-derived unit vector, so unrelated strings are
    orthogonal (similarity 0) and never collide accidentally. Tests that care
    about similarity register explicit vectors via :meth:`set_vector`.
    """

    def __init__(self, dim: int = 1536) -> None:
        self.dim = dim
        self.vectors: dict[str, list[float]] = {}
        self.calls: list[str] = []
        #: When set, every unmapped text maps here. Used by tests that care about
        #: the retrieval *pipeline* rather than embedding quality.
        self.default_vector: list[float] | None = None

    def set_vector(self, text: str, vector: list[float]) -> None:
        if len(vector) != self.dim:
            raise ValueError(f"向量维度应为 {self.dim}")
        self.vectors[text] = vector

    def vector_for(self, text: str) -> list[float]:
        self.calls.append(text)
        if text in self.vectors:
            return list(self.vectors[text])
        if self.default_vector is not None:
            return list(self.default_vector)
        return _hash_vector(text, self.dim)

    # LangChain Embeddings-shaped surface
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.vector_for(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.vector_for(text)


def _hash_vector(text: str, dim: int) -> list[float]:
    """A deterministic lexical embedding (character + bigram bag of words).

    It is *not* a semantic model — it is a stand-in that makes texts sharing
    characters measurably closer, so retrieval plumbing (SQL, thresholds,
    ordering, citation assembly) can be exercised deterministically. Semantic
    quality is covered by the ``llm``-marked test against the real provider.
    """
    accumulator = [0.0] * dim
    compact = "".join(text.split())
    tokens = list(compact) + [compact[index : index + 2] for index in range(max(0, len(compact) - 1))]
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        index = value % dim
        accumulator[index] += 1.0 if (value >> 63) & 1 else -1.0
    norm = math.sqrt(sum(item * item for item in accumulator)) or 1.0
    return [item / norm for item in accumulator]


def one_hot(dim: int, index: int) -> list[float]:
    vector = [0.0] * dim
    vector[index % dim] = 1.0
    return vector


def blend(dim: int, index: int, other: int, ratio: float) -> list[float]:
    """A vector at a chosen cosine distance from ``one_hot(dim, index)``."""
    vector = [0.0] * dim
    vector[index % dim] = ratio
    vector[other % dim] = math.sqrt(max(0.0, 1 - ratio * ratio))
    return vector


def make_model_factory(*scripts: list[Any]) -> tuple[Any, list[FakeChatModel]]:
    """Return a ``get_chat_model`` replacement plus the models it handed out.

    The first script is handed to the first caller; each additional script to
    the next, so a test can separate "the analysis call" from "the knowledge
    call" when both happen in one turn.
    """
    created: list[FakeChatModel] = []
    queue = list(scripts)

    def factory(*args: Any, **kwargs: Any) -> FakeChatModel:  # noqa: ARG001
        script = queue.pop(0) if queue else []
        model = FakeChatModel(script, label=f"fake#{len(created)}")
        created.append(model)
        return model

    return factory, created
