"""Shared runtime dependencies handed to graph nodes."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from app.core.config import Settings
from app.db.session import Database
from app.memory.service import MemoryService


@dataclass
class AgentDeps:
    """Everything a node needs that is not part of the graph state."""

    settings: Settings
    database: Database
    memory_service: MemoryService
    user_id: uuid.UUID
