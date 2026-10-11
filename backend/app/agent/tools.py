"""Tools the agent can call.

Two hard boundaries are enforced here rather than in prompts:

* ``search_theory``/``search_cases`` only ever read the public knowledge base.
* ``recall_user_memory`` only ever reads rows belonging to the current user.

Every citation returned carries its real ``source_path``, ``source_type`` and a
``verified`` flag, so a curated note cannot be shown to the user as a quotation.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.logging import get_logger
from app.db.repositories import knowledge as knowledge_repo
from app.llm import client as llm
from app.memory.service import MemoryService
from app.schemas.agent import Citation

logger = get_logger(__name__)


class SearchTheoryArgs(BaseModel):
    query: str = Field(description="中文检索词，围绕理论概念，例如「代价 不可逆」「贪婪与恐惧」")
    top_k: int = Field(default=5, ge=1, le=10, description="返回条目数")


class SearchCasesArgs(BaseModel):
    query: str = Field(description="中文检索词，描述与当前困境相似的处境")
    top_k: int = Field(default=3, ge=1, le=10)


class RecallMemoryArgs(BaseModel):
    query: str = Field(description="中文检索词，描述要回忆的用户信息主题")
    top_k: int = Field(default=5, ge=1, le=10)


@dataclass
class ToolTrace:
    """Everything the tools returned, for the state and the UI."""

    citations: list[Citation] = field(default_factory=list)
    memory_snippets: list[str] = field(default_factory=list)
    calls: list[dict[str, str]] = field(default_factory=list)

    def add_citations(self, found: list[Citation]) -> None:
        known = {c.chunk_id for c in self.citations}
        for citation in found:
            if citation.chunk_id not in known:
                self.citations.append(citation)
                known.add(citation.chunk_id)


def _serialize(citations: list[Citation]) -> str:
    """Render tool output as compact numbered JSON the model can cite by index."""
    if not citations:
        return json.dumps({"results": [], "note": "知识库中没有检索到相关条目。"}, ensure_ascii=False)
    return json.dumps(
        {
            "results": [
                {
                    "index": index,
                    "title": item.document_title,
                    "heading": item.heading,
                    "source_type": item.source_type.value if hasattr(item.source_type, "value") else item.source_type,
                    "verified": item.verified,
                    "similarity": item.score,
                    "content": item.excerpt,
                }
                for index, item in enumerate(citations, start=1)
            ]
        },
        ensure_ascii=False,
    )


async def search_knowledge(
    *,
    session: AsyncSession,
    settings: Settings,
    query: str,
    top_k: int,
    trace: ToolTrace,
    source_types: list[str] | None = None,
    tool_name: str = "search_theory",
) -> str:
    embedding = await llm.embed_query(query)
    citations = await knowledge_repo.search_chunks(
        session,
        embedding,
        top_k=top_k,
        min_score=settings.rag_min_score,
        source_types=source_types,
    )
    trace.calls.append({"tool": tool_name, "query": query})
    trace.add_citations(citations)
    return _serialize(citations)


def build_knowledge_tools(
    *,
    session: AsyncSession,
    settings: Settings,
    memory_service: MemoryService,
    user_id: uuid.UUID,
    trace: ToolTrace,
) -> list[StructuredTool]:
    async def _search_theory(query: str, top_k: int = 5) -> str:
        return await search_knowledge(
            session=session,
            settings=settings,
            query=query,
            top_k=top_k,
            trace=trace,
            source_types=["curated_note", "verified_excerpt"],
            tool_name="search_theory",
        )

    async def _search_cases(query: str, top_k: int = 3) -> str:
        return await search_knowledge(
            session=session,
            settings=settings,
            query=query,
            top_k=top_k,
            trace=trace,
            source_types=["case"],
            tool_name="search_cases",
        )

    async def _recall_user_memory(query: str, top_k: int = 5) -> str:
        result = await memory_service.recall(session, user_id=user_id, query=query, top_k=top_k)
        trace.calls.append({"tool": "recall_user_memory", "query": query})
        if not result.memories and not result.events:
            return json.dumps({"memories": [], "note": "没有相关记忆。"}, ensure_ascii=False)
        rendered = result.render()
        trace.memory_snippets.append(rendered)
        return rendered

    return [
        StructuredTool.from_function(
            coroutine=_search_theory,
            name="search_theory",
            description="检索《人选天选论》的理论条目（整理笔记与已核对原文摘录）。",
            args_schema=SearchTheoryArgs,
        ),
        StructuredTool.from_function(
            coroutine=_search_cases,
            name="search_cases",
            description="检索知识库中的案例，用于类比相似处境。",
            args_schema=SearchCasesArgs,
        ),
        StructuredTool.from_function(
            coroutine=_recall_user_memory,
            name="recall_user_memory",
            description="检索这位用户过去确认过的长期记忆与历史选择。",
            args_schema=RecallMemoryArgs,
        ),
    ]
