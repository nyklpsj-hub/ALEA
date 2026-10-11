"""ORM models.

Layout of the persisted world:

* ``users`` / ``sessions`` / ``messages`` — the conversation record.
* ``memories`` / ``memory_audit`` — long-lived, user-scoped semantic memory.
* ``decision_events`` — the user's history of past choices and their costs.
* ``knowledge_documents`` / ``knowledge_chunks`` — the *public* theory corpus.

The last two groups are deliberately separate tables with separate retrieval
code paths: private memory can never be served out of the public knowledge
index, and vice versa.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import get_settings
from app.db.base import Base, Timestamped, UUIDPrimaryKey

_DIM = get_settings().embedding_dim


class User(UUIDPrimaryKey, Timestamped, Base):
    __tablename__ = "users"

    display_name: Mapped[str] = mapped_column(String(120), nullable=False)
    #: Reserved for future multi-user auth; unused in V1.
    external_ref: Mapped[str | None] = mapped_column(String(255), nullable=True)


class Session(UUIDPrimaryKey, Timestamped, Base):
    __tablename__ = "sessions"

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False, default="新的困境")
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="active")
    phase: Mapped[str] = mapped_column(String(24), nullable=False, default="understanding")
    turn_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    info_sufficiency: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    last_active_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    #: Latest structured snapshot of the agent's analysis (mirrors graph state).
    analysis: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    decision: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    messages: Mapped[list[Message]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="Message.seq"
    )

    __table_args__ = (Index("ix_sessions_user_active", "user_id", "last_active_at"),)


class Message(UUIDPrimaryKey, Base):
    __tablename__ = "messages"

    session_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    #: Which graph node produced the turn, for debuggability.
    node: Mapped[str | None] = mapped_column(String(48), nullable=True)
    meta: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    session: Mapped[Session] = relationship(back_populates="messages")

    __table_args__ = (
        UniqueConstraint("session_id", "seq", name="uq_messages_session_seq"),
        Index("ix_messages_session", "session_id", "seq"),
    )


class Memory(UUIDPrimaryKey, Timestamped, Base):
    """A single long-lived fact/motive about a user."""

    __tablename__ = "memories"

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[str] = mapped_column(String(24), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    #: Short label used for cheap de-duplication ("换城市工作", "父母养老").
    normalized_key: Mapped[str] = mapped_column(String(160), nullable=False, default="")
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="candidate")
    provenance: Mapped[str] = mapped_column(String(24), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.6)
    importance: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    #: The user's own words that justify this row (source traceability).
    evidence: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_session_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("sessions.id", ondelete="SET NULL"), nullable=True
    )
    source_message_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("messages.id", ondelete="SET NULL"), nullable=True
    )
    superseded_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("memories.id", ondelete="SET NULL"), nullable=True
    )
    tags: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(_DIM), nullable=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    use_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    #: Set when the user deletes a memory; the row survives for the audit trail.
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        CheckConstraint("importance >= 1 AND importance <= 5", name="importance_range"),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="confidence_range"),
        Index("ix_memories_user_kind_status", "user_id", "kind", "status"),
        Index(
            "ix_memories_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class MemoryAudit(UUIDPrimaryKey, Base):
    """Append-only trail of every mutation applied to a memory."""

    __tablename__ = "memory_audit"

    memory_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("memories.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    action: Mapped[str] = mapped_column(String(24), nullable=False)
    #: ``user`` when the person did it from the UI, ``agent`` when the graph did.
    actor: Mapped[str] = mapped_column(String(16), nullable=False, default="agent")
    before: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    after: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (Index("ix_memory_audit_memory", "memory_id", "created_at"),)


class DecisionEvent(UUIDPrimaryKey, Timestamped, Base):
    """A past decision (and its price) distilled from a finished session."""

    __tablename__ = "decision_events"

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("sessions.id", ondelete="SET NULL"), nullable=True
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    situation: Mapped[str] = mapped_column(Text, nullable=False)
    options: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)
    chosen_option: Mapped[str | None] = mapped_column(Text, nullable=True)
    greed: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    fear: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    cost: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    risks: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    #: Filled in when the user later reports how it went.
    outcome: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(_DIM), nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_decision_events_user", "user_id", "created_at"),
        Index(
            "ix_decision_events_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class KnowledgeDocument(UUIDPrimaryKey, Timestamped, Base):
    __tablename__ = "knowledge_documents"

    source_path: Mapped[str] = mapped_column(String(400), nullable=False, unique=True)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False, default="curated_note")
    #: True only when a human checked the text against the primary source.
    verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    meta: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    chunks: Mapped[list[KnowledgeChunk]] = relationship(back_populates="document", cascade="all, delete-orphan")


class KnowledgeChunk(UUIDPrimaryKey, Base):
    __tablename__ = "knowledge_chunks"

    document_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("knowledge_documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    heading: Mapped[str] = mapped_column(String(400), nullable=False, default="")
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(_DIM), nullable=True)
    meta: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    document: Mapped[KnowledgeDocument] = relationship(back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("document_id", "ordinal", name="uq_knowledge_chunks_doc_ordinal"),
        Index(
            "ix_knowledge_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class SchemaMeta(Base):
    """Single-row-per-key table recording schema-level facts."""

    __tablename__ = "schema_meta"

    key: Mapped[str] = mapped_column(String(80), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
