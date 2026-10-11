"""Health and runtime configuration endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from app import __version__
from app.api.deps import ContainerDep, SettingsDep
from app.schemas.api import ConfigResponse, HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health(container: ContainerDep) -> HealthResponse:
    database_ok = True
    chunks = 0
    try:
        await container.database.ping()
        chunks = await container.knowledge_service.count_chunks()
    except Exception:  # noqa: BLE001 - health must always answer
        database_ok = False
    return HealthResponse(
        status="ok" if database_ok else "degraded",
        version=__version__,
        database=database_ok,
        llm_configured=container.settings.llm_configured,
        embedding_provider=container.settings.embedding_provider,
        knowledge_chunks=chunks,
    )


@router.get("/config", response_model=ConfigResponse)
async def runtime_config(settings: SettingsDep) -> ConfigResponse:
    """Expose what the model setup is — never the keys themselves."""
    return ConfigResponse(
        app_name=settings.app_name,
        app_env=settings.app_env,
        llm_model=settings.llm_model,
        llm_base_url=settings.llm_base_url,
        embedding_provider=settings.embedding_provider,
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
        agent_soft_turn_limit=settings.agent_soft_turn_limit,
        agent_hard_turn_limit=settings.agent_hard_turn_limit,
        agent_sufficiency_threshold=settings.agent_sufficiency_threshold,
    )
