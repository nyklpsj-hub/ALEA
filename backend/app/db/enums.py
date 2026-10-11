"""Domain enumerations shared by ORM models, schemas and the agent."""

from __future__ import annotations

from enum import StrEnum


class SessionStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class MemoryKind(StrEnum):
    """What a long-lived memory is about."""

    GOAL = "goal"  # 重要目标
    VALUE = "value"  # 价值观
    GREED = "greed"  # 贪婪：想得到什么
    FEAR = "fear"  # 恐惧：怕失去什么
    CONSTRAINT = "constraint"  # 现实约束
    DECISION = "decision"  # 历史选择及其代价
    PREFERENCE = "preference"  # 沟通或决策偏好
    FACT = "fact"  # 稳定的客观事实


class MemoryStatus(StrEnum):
    """Lifecycle of a memory row."""

    #: Agent inferred it and the user has neither confirmed nor denied it.
    CANDIDATE = "candidate"
    #: Either stated outright by the user or confirmed after being inferred.
    ACTIVE = "active"
    #: The user explicitly said this is not true.
    REJECTED = "rejected"
    #: Replaced by a newer, conflicting memory.
    SUPERSEDED = "superseded"


class MemoryProvenance(StrEnum):
    """Where a memory came from — the honesty axis of the memory system."""

    #: The user said it in their own words.
    USER_STATED = "user_stated"
    #: The agent inferred it and the user explicitly confirmed it.
    USER_CONFIRMED = "user_confirmed"
    #: The agent's own inference; never treated as fact.
    AGENT_INFERRED = "agent_inferred"


class MemoryAuditAction(StrEnum):
    CREATED = "created"
    UPDATED = "updated"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    MERGED = "merged"
    SUPERSEDED = "superseded"
    DELETED = "deleted"
    RESTORED = "restored"
    USED = "used"


class SourceType(StrEnum):
    """Provenance of a knowledge-base document."""

    #: An excerpt verified against the primary source by a human editor.
    VERIFIED_EXCERPT = "verified_excerpt"
    #: This repository's own editorial summary of the framework.
    CURATED_NOTE = "curated_note"
    #: A worked case, either from the source or reconstructed for illustration.
    CASE = "case"


class AgentPhase(StrEnum):
    """High-level stage of the decision workflow (mirrors the LangGraph route)."""

    UNDERSTANDING = "understanding"
    MEMORY = "memory"
    EXPLORING = "exploring"
    CONFIRMING = "confirming"
    KNOWLEDGE = "knowledge"
    DECIDING = "deciding"
    DECIDED = "decided"
    PAUSED = "paused"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ConfirmationStatus(StrEnum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    CORRECTED = "corrected"
    REJECTED = "rejected"
