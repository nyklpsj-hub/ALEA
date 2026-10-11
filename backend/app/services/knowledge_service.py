"""Ingestion and querying of the public knowledge base."""

from __future__ import annotations

import time

from sqlalchemy import delete as sa_delete

from app.core.config import Settings
from app.core.logging import get_logger
from app.db.models import KnowledgeChunk, KnowledgeDocument
from app.db.repositories import knowledge as knowledge_repo
from app.db.session import Database
from app.llm import client as llm
from app.rag.chunker import chunk_document
from app.rag.loader import iter_documents
from app.schemas.agent import Citation
from app.schemas.api import IngestReport

logger = get_logger(__name__)


class KnowledgeService:
    def __init__(self, settings: Settings, database: Database) -> None:
        self.settings = settings
        self.database = database

    async def ingest(self, *, force: bool = False) -> IngestReport:
        started = time.perf_counter()
        docs = iter_documents(self.settings.knowledge_dir)
        created = updated = unchanged = chunks_written = 0

        async with self.database.session() as session:
            for doc in docs:
                existing = await knowledge_repo.get_document(session, doc.source_path)
                if existing is not None and existing.checksum == doc.checksum and not force:
                    unchanged += 1
                    continue

                chunks = chunk_document(
                    doc,
                    chunk_size=self.settings.rag_chunk_size,
                    chunk_overlap=self.settings.rag_chunk_overlap,
                )
                vectors = await llm.embed_texts([chunk.content for chunk in chunks])

                if existing is None:
                    document = KnowledgeDocument(
                        source_path=doc.source_path,
                        title=doc.title,
                        checksum=doc.checksum,
                        source_type=doc.source_type.value,
                        verified=doc.verified,
                        version=1,
                        meta=doc.meta,
                    )
                    session.add(document)
                    created += 1
                else:
                    document = existing
                    document.title = doc.title
                    document.checksum = doc.checksum
                    document.source_type = doc.source_type.value
                    document.verified = doc.verified
                    document.version += 1
                    document.meta = doc.meta
                    updated += 1
                    await session.execute(sa_delete(KnowledgeChunk).where(KnowledgeChunk.document_id == document.id))

                await session.flush()
                for chunk, vector in zip(chunks, vectors, strict=True):
                    session.add(
                        KnowledgeChunk(
                            document_id=document.id,
                            ordinal=chunk.ordinal,
                            heading=chunk.heading[:400],
                            content=chunk.content,
                            token_count=len(chunk.content),
                            embedding=vector,
                            meta={},
                        )
                    )
                    chunks_written += 1

            removed = await knowledge_repo.delete_documents_not_in(session, [doc.source_path for doc in docs])
            if removed:
                logger.info("knowledge_documents_removed", extra={"paths": removed})

        report = IngestReport(
            documents_seen=len(docs),
            documents_created=created,
            documents_updated=updated,
            documents_unchanged=unchanged,
            chunks_written=chunks_written,
            embedding_provider=self.settings.embedding_provider,
            embedding_dim=self.settings.embedding_dim,
            duration_seconds=round(time.perf_counter() - started, 3),
        )
        logger.info("knowledge_ingested", extra=report.model_dump())
        return report

    async def search(self, query: str, *, top_k: int | None = None) -> list[Citation]:
        top_k = top_k or self.settings.rag_top_k
        embedding = await llm.embed_query(query)
        async with self.database.session() as session:
            return await knowledge_repo.search_chunks(
                session,
                embedding,
                top_k=top_k,
                min_score=self.settings.rag_min_score,
            )

    async def list_documents(self) -> list[tuple[KnowledgeDocument, int]]:
        async with self.database.session() as session:
            return await knowledge_repo.list_documents(session)

    async def count_chunks(self) -> int:
        async with self.database.session() as session:
            return await knowledge_repo.count_chunks(session)
