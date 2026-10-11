"""Queries over the public knowledge base."""

from __future__ import annotations

import uuid

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.enums import SourceType
from app.db.models import KnowledgeChunk, KnowledgeDocument
from app.schemas.agent import Citation


async def get_document(session: AsyncSession, source_path: str) -> KnowledgeDocument | None:
    result = await session.execute(select(KnowledgeDocument).where(KnowledgeDocument.source_path == source_path))
    return result.scalar_one_or_none()


async def list_documents(session: AsyncSession) -> list[tuple[KnowledgeDocument, int]]:
    stmt = (
        select(KnowledgeDocument, func.count(KnowledgeChunk.id))
        .outerjoin(KnowledgeChunk, KnowledgeChunk.document_id == KnowledgeDocument.id)
        .group_by(KnowledgeDocument.id)
        .order_by(KnowledgeDocument.source_path)
    )
    return [(doc, count) for doc, count in (await session.execute(stmt)).all()]


async def delete_document(session: AsyncSession, document_id: uuid.UUID) -> None:
    await session.execute(delete(KnowledgeDocument).where(KnowledgeDocument.id == document_id))


async def delete_documents_not_in(session: AsyncSession, source_paths: list[str]) -> list[str]:
    stmt = select(KnowledgeDocument.source_path)
    if source_paths:
        stmt = stmt.where(KnowledgeDocument.source_path.not_in(source_paths))
    removed = list((await session.execute(stmt)).scalars())
    if removed:
        await session.execute(delete(KnowledgeDocument).where(KnowledgeDocument.source_path.in_(removed)))
    return removed


async def count_chunks(session: AsyncSession) -> int:
    return int((await session.execute(select(func.count(KnowledgeChunk.id)))).scalar_one())


async def search_chunks(
    session: AsyncSession,
    query_embedding: list[float],
    *,
    top_k: int,
    min_score: float,
    source_types: list[str] | None = None,
) -> list[Citation]:
    """Cosine-similarity search over the theory corpus.

    Only rows with an embedding are considered, and the similarity floor is
    applied in SQL so an empty result is an honest "the corpus says nothing
    about this" rather than a weak match dressed up as evidence.

    ``source_types`` is filtered in SQL rather than afterwards: filtering a
    top-k result set would silently drop matches whenever the dominant type
    fills the limit.
    """
    distance = KnowledgeChunk.embedding.cosine_distance(query_embedding)
    stmt = (
        select(KnowledgeChunk, KnowledgeDocument, distance.label("distance"))
        .join(KnowledgeDocument, KnowledgeDocument.id == KnowledgeChunk.document_id)
        .where(KnowledgeChunk.embedding.is_not(None))
    )
    if source_types:
        stmt = stmt.where(KnowledgeDocument.source_type.in_(source_types))
    # A stable tie-break keeps results reproducible when scores are equal.
    stmt = stmt.order_by(distance, KnowledgeChunk.id).limit(top_k)

    citations: list[Citation] = []
    for chunk, document, dist in (await session.execute(stmt)).all():
        score = 1.0 - float(dist)
        if score < min_score:
            continue
        citations.append(
            Citation(
                chunk_id=str(chunk.id),
                document_title=document.title,
                heading=chunk.heading,
                source_path=document.source_path,
                source_type=SourceType(document.source_type),
                verified=document.verified,
                score=round(score, 4),
                excerpt=chunk.content.strip(),
            )
        )
    return citations
