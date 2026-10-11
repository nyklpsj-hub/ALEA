"""HTTP request/response contracts."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.db.enums import (
    MemoryKind,
    MemoryProvenance,
    MemoryStatus,
    MessageRole,
    SessionStatus,
    SourceType,
)
from app.schemas.agent import Citation, Decision, SessionAnalysis


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# --------------------------------------------------------------------- health
class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = "ok"
    version: str
    database: bool
    llm_configured: bool
    embedding_provider: str
    knowledge_chunks: int = 0


class ConfigResponse(BaseModel):
    """Non-secret runtime configuration, so the UI can tell users what they run."""

    app_name: str
    app_env: str
    llm_model: str
    llm_base_url: str
    embedding_provider: str
    embedding_model: str
    embedding_dim: int
    agent_soft_turn_limit: int
    agent_hard_turn_limit: int
    agent_sufficiency_threshold: float


# ------------------------------------------------------------------- sessions
class SessionCreate(BaseModel):
    title: str | None = Field(default=None, max_length=200)


class SessionUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    status: SessionStatus | None = None


class MessageRead(ORMModel):
    id: uuid.UUID
    session_id: uuid.UUID
    seq: int
    role: MessageRole
    content: str
    node: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class SessionRead(ORMModel):
    id: uuid.UUID
    title: str
    status: SessionStatus
    phase: str
    turn_count: int
    info_sufficiency: float
    last_active_at: datetime
    created_at: datetime
    has_decision: bool = False


class SessionDetail(SessionRead):
    analysis: SessionAnalysis = Field(default_factory=SessionAnalysis)
    decision: Decision | None = None
    messages: list[MessageRead] = Field(default_factory=list)


# ----------------------------------------------------------------------- chat
class ChatRequest(BaseModel):
    content: str = Field(min_length=1, max_length=8000)


class MemoryChange(BaseModel):
    """A long-term memory written during this turn."""

    action: str
    memory_id: uuid.UUID
    content: str
    kind: MemoryKind
    status: MemoryStatus


class ChatResponse(BaseModel):
    session_id: uuid.UUID
    message: MessageRead
    analysis: SessionAnalysis
    decision: Decision | None = None
    phase: str
    turn_count: int
    recalled_memory_count: int = 0
    citations: list[Citation] = Field(default_factory=list)
    memory_changes: list[MemoryChange] = Field(default_factory=list)


class ConfirmationResolve(BaseModel):
    """The user's answer to a single confirmation card."""

    confirmation_id: str
    action: Literal["confirm", "correct", "reject"]
    corrected_text: str | None = Field(default=None, max_length=2000)
    note: str | None = Field(default=None, max_length=1000)


# -------------------------------------------------------------------- memory
class MemoryCreate(BaseModel):
    kind: MemoryKind
    content: str = Field(min_length=2, max_length=4000)
    provenance: MemoryProvenance = MemoryProvenance.USER_STATED
    importance: int = Field(default=3, ge=1, le=5)
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)


class MemoryUpdate(BaseModel):
    content: str | None = Field(default=None, min_length=2, max_length=4000)
    kind: MemoryKind | None = None
    status: MemoryStatus | None = None
    importance: int | None = Field(default=None, ge=1, le=5)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    tags: list[str] | None = None


class MemoryRead(ORMModel):
    id: uuid.UUID
    user_id: uuid.UUID
    kind: MemoryKind
    content: str
    normalized_key: str
    status: MemoryStatus
    provenance: MemoryProvenance
    confidence: float
    importance: int
    evidence: str | None = None
    source_session_id: uuid.UUID | None = None
    source_message_id: uuid.UUID | None = None
    superseded_by: uuid.UUID | None = None
    tags: list[str] = Field(default_factory=list)
    use_count: int
    last_used_at: datetime | None = None
    deleted_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class MemoryAuditRead(ORMModel):
    id: uuid.UUID
    memory_id: uuid.UUID
    action: str
    actor: str
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    reason: str | None = None
    created_at: datetime


class MemorySearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    kinds: list[MemoryKind] = Field(default_factory=list)
    top_k: int = Field(default=6, ge=1, le=30)


class MemorySearchHit(BaseModel):
    memory: MemoryRead
    score: float


# ------------------------------------------------------------------ decisions
class DecisionEventRead(ORMModel):
    id: uuid.UUID
    session_id: uuid.UUID | None = None
    title: str
    situation: str
    chosen_option: str | None = None
    greed: list[str] = Field(default_factory=list)
    fear: list[str] = Field(default_factory=list)
    cost: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    outcome: str | None = None
    summary: str | None = None
    created_at: datetime


class DecisionEventOutcomeUpdate(BaseModel):
    outcome: str = Field(min_length=2, max_length=4000)


# ------------------------------------------------------------------ knowledge
class KnowledgeDocumentRead(ORMModel):
    id: uuid.UUID
    source_path: str
    title: str
    source_type: SourceType
    verified: bool
    version: int
    updated_at: datetime
    chunk_count: int = 0


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class KnowledgeSearchResponse(BaseModel):
    query: str
    results: list[Citation] = Field(default_factory=list)
    used_min_score: float


class IngestReport(BaseModel):
    documents_seen: int
    documents_created: int
    documents_updated: int
    documents_unchanged: int
    chunks_written: int
    embedding_provider: str
    embedding_dim: int
    duration_seconds: float
