"""Knowledge base: inspect, verify retrieval, re-index."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import ContainerDep, KnowledgeServiceDep, SettingsDep
from app.db.enums import SourceType
from app.schemas.api import (
    IngestReport,
    KnowledgeDocumentRead,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
)

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.get("/documents", response_model=list[KnowledgeDocumentRead])
async def list_documents(service: KnowledgeServiceDep) -> list[KnowledgeDocumentRead]:
    rows = await service.list_documents()
    return [
        KnowledgeDocumentRead(
            id=document.id,
            source_path=document.source_path,
            title=document.title,
            source_type=SourceType(document.source_type),
            verified=document.verified,
            version=document.version,
            updated_at=document.updated_at,
            chunk_count=chunk_count,
        )
        for document, chunk_count in rows
    ]


@router.post("/search", response_model=KnowledgeSearchResponse)
async def search(
    payload: KnowledgeSearchRequest, service: KnowledgeServiceDep, settings: SettingsDep
) -> KnowledgeSearchResponse:
    """Direct retrieval, used by the UI to show exactly what the agent can cite."""
    results = await service.search(payload.query, top_k=payload.top_k)
    return KnowledgeSearchResponse(query=payload.query, results=results, used_min_score=settings.rag_min_score)


@router.post("/reindex", response_model=IngestReport)
async def reindex(container: ContainerDep, force: bool = False) -> IngestReport:
    return await container.knowledge_service.ingest(force=force)
