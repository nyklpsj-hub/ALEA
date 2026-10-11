"""FastAPI dependencies."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import Depends, Request

from app.core.config import Settings, get_settings
from app.core.errors import ConfigurationError
from app.services.chat_service import ChatService
from app.services.container import AppContainer
from app.services.knowledge_service import KnowledgeService


def get_container(request: Request) -> AppContainer:
    container = getattr(request.app.state, "container", None)
    if container is None:  # pragma: no cover - startup always sets this
        raise ConfigurationError("应用尚未完成初始化")
    return container


def get_chat_service(container: Annotated[AppContainer, Depends(get_container)]) -> ChatService:
    return container.chat_service


def get_knowledge_service(
    container: Annotated[AppContainer, Depends(get_container)],
) -> KnowledgeService:
    return container.knowledge_service


def get_current_user_id(container: Annotated[AppContainer, Depends(get_container)]) -> uuid.UUID:
    """V1 has a single local profile.

    Every query still filters by this id, so adding real authentication later
    means replacing this function — not auditing every call site.
    """
    return container.user_id


def get_app_settings() -> Settings:
    return get_settings()


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
KnowledgeServiceDep = Annotated[KnowledgeService, Depends(get_knowledge_service)]
UserIdDep = Annotated[uuid.UUID, Depends(get_current_user_id)]
SettingsDep = Annotated[Settings, Depends(get_app_settings)]
ContainerDep = Annotated[AppContainer, Depends(get_container)]
