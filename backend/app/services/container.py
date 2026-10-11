"""Composition root.

One container holds every long-lived collaborator; FastAPI receives it through a
dependency, graph nodes receive it through the LangGraph config. There is no
service locator magic and no circular imports.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from app.agent.graph import AgentRuntime
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.db.session import Database
from app.memory.service import MemoryService
from app.services.chat_service import ChatService
from app.services.knowledge_service import KnowledgeService

logger = get_logger(__name__)


@dataclass
class AppContainer:
    settings: Settings
    database: Database
    runtime: AgentRuntime
    memory_service: MemoryService
    knowledge_service: KnowledgeService
    chat_service: ChatService
    user_id: uuid.UUID

    @classmethod
    async def create(cls, settings: Settings | None = None) -> AppContainer:
        settings = settings or get_settings()
        database = Database(settings)
        runtime = AgentRuntime(settings)
        memory_service = MemoryService(settings)
        knowledge_service = KnowledgeService(settings, database)
        chat_service = ChatService(
            settings=settings,
            database=database,
            runtime=runtime,
            memory_service=memory_service,
        )
        container = cls(
            settings=settings,
            database=database,
            runtime=runtime,
            memory_service=memory_service,
            knowledge_service=knowledge_service,
            chat_service=chat_service,
            user_id=uuid.UUID(settings.default_user_id),
        )
        await container.startup()
        return container

    async def startup(self) -> None:
        await self.database.ping()
        await self.database.verify_extensions()
        await self.database.ensure_default_user()
        await self.runtime.start()
        logger.info(
            "container_ready",
            extra={
                "app_env": self.settings.app_env,
                "llm_model": self.settings.llm_model,
                "embedding_provider": self.settings.embedding_provider,
                "llm_configured": self.settings.llm_configured,
            },
        )

    async def shutdown(self) -> None:
        await self.runtime.stop()
        await self.database.dispose()
