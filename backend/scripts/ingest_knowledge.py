"""Index the knowledge directory into PostgreSQL + pgvector.

    python -m scripts.ingest_knowledge           # incremental (skips unchanged files)
    python -m scripts.ingest_knowledge --force   # re-embed everything

Requires a reachable database and a configured embedding provider; both are read
from `backend/.env`.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from app.core.config import get_settings
from app.core.errors import AppError
from app.core.logging import configure_logging
from app.db.session import Database
from app.services.knowledge_service import KnowledgeService


async def run(force: bool) -> int:
    settings = get_settings()
    configure_logging(settings)
    database = Database(settings)
    try:
        await database.verify_extensions()
        service = KnowledgeService(settings, database)
        report = await service.ingest(force=force)
    except AppError as exc:
        print(f"失败：{exc.message} {exc.details}", file=sys.stderr)
        return 1
    finally:
        await database.dispose()

    print("知识库索引完成：")
    print(f"  扫描文件      {report.documents_seen}")
    print(f"  新建 / 更新   {report.documents_created} / {report.documents_updated}")
    print(f"  未变化        {report.documents_unchanged}")
    print(f"  写入切片      {report.chunks_written}")
    print(f"  向量模型      {report.embedding_provider} · {report.embedding_dim} 维")
    print(f"  耗时          {report.duration_seconds}s")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Index the Alea knowledge base")
    parser.add_argument("--force", action="store_true", help="重新向量化所有文件")
    args = parser.parse_args()
    return asyncio.run(run(args.force))


if __name__ == "__main__":
    raise SystemExit(main())
