"""Check the knowledge base and its retrieval, with whichever embedder is configured.

    python -m scripts.check_retrieval
    python -m scripts.check_retrieval "代价 不可逆 底线" --top-k 5

This deliberately needs no chat model: it validates the half of RAG that decides
whether the agent has anything real to cite — the corpus parses, the chunks are
embedded, the pgvector query returns sensible neighbours, and every hit carries
usable provenance. Run it after changing EMBEDDING_* or editing `knowledge/`.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from app.core.compat import configure_event_loop_policy
from app.core.config import get_settings
from app.core.errors import AppError
from app.core.logging import configure_logging
from app.db.session import Database
from app.services.knowledge_service import KnowledgeService

configure_event_loop_policy()

DEFAULT_QUERIES = [
    "代价 不可逆 底线",
    "贪婪与恐惧 错位",
    "要不要辞职 稳定 收入",
]


async def run(queries: list[str], *, top_k: int, reindex: bool) -> int:
    settings = get_settings()
    configure_logging(settings)

    print(f"向量：{settings.embedding_provider} · {settings.embedding_model} · {settings.embedding_dim} 维")
    print(f"知识库：{settings.knowledge_dir}")
    print(f"阈值：相似度 ≥ {settings.rag_min_score}\n")

    database = Database(settings)
    try:
        await database.verify_extensions()
        service = KnowledgeService(settings, database)
        if reindex or await service.count_chunks() == 0:
            report = await service.ingest(force=reindex)
            print(
                f"索引完成：{report.documents_seen} 个文件，"
                f"新建 {report.documents_created}，更新 {report.documents_updated}，"
                f"写入 {report.chunks_written} 个切片\n"
            )

        documents = await service.list_documents()
        print(f"知识库共 {len(documents)} 个文档：")
        for document, chunk_count in documents:
            tag = (
                "原文摘录"
                if document.source_type == "verified_excerpt"
                else ("案例" if document.source_type == "case" else "整理笔记")
            )
            print(f"  · {document.title}（{tag}，{chunk_count} 切片）  {document.source_path}")
        print()

        failures = 0
        for query in queries:
            hits = await service.search(query, top_k=top_k)
            print(f"查询：{query!r} → {len(hits)} 条")
            if not hits:
                print("  （没有命中的片段：要么知识库确实没有相关内容，要么阈值过高）\n")
                failures += 1
                continue
            for rank, hit in enumerate(hits, start=1):
                tag = "原文摘录" if hit.verified else ("案例" if hit.source_type == "case" else "整理笔记")
                print(f"  {rank}. [{hit.score:.3f}] {hit.document_title}｜{hit.heading or '（无章节）'}")
                print(f"     {tag}｜{hit.source_path}")
                print(f"     {hit.excerpt[:120].replace(chr(10), ' ')}…")
            print()

        return 1 if failures == len(queries) else 0
    except AppError as exc:
        print(f"失败：{exc.message} {exc.details}", file=sys.stderr)
        return 1
    finally:
        await database.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(description="Check knowledge-base retrieval")
    parser.add_argument("query", nargs="*", help="检索词；不填则跑一组默认查询")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--reindex", action="store_true", help="先强制重建索引")
    args = parser.parse_args()
    return asyncio.run(run(args.query or DEFAULT_QUERIES, top_k=args.top_k, reindex=args.reindex))


if __name__ == "__main__":
    raise SystemExit(main())
