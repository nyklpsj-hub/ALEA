"""Async database engine and session plumbing."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import Settings, get_settings
from app.core.errors import ConfigurationError
from app.core.logging import get_logger
from app.db.models import User

logger = get_logger(__name__)


def create_engine(settings: Settings) -> AsyncEngine:
    return create_async_engine(
        settings.database_url,
        echo=False,
        pool_pre_ping=True,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        connect_args={
            "server_settings": {
                "statement_timeout": str(settings.db_statement_timeout_ms),
                "application_name": "alea-api",
            }
        },
    )


class Database:
    """Owns the engine and hands out request-scoped sessions."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.engine: AsyncEngine = create_engine(self.settings)
        self._sessionmaker = async_sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self._sessionmaker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def ping(self) -> None:
        async with self.engine.connect() as conn:
            await conn.execute(text("SELECT 1"))

    async def verify_extensions(self) -> None:
        async with self.engine.connect() as conn:
            row = (
                await conn.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'vector'"))
            ).scalar_one_or_none()
            if row is None:
                raise ConfigurationError(
                    "数据库缺少 pgvector 扩展，请执行 CREATE EXTENSION vector",
                    details={"extension": "vector"},
                )
            migrated = (await conn.execute(text("SELECT to_regclass('public.schema_meta')"))).scalar_one_or_none()
            if migrated is None:
                raise ConfigurationError(
                    "数据库尚未初始化，请先执行 `alembic upgrade head`",
                    details={"command": "alembic upgrade head"},
                )
            stored_dim = (
                await conn.execute(text("SELECT value FROM schema_meta WHERE key = 'embedding_dim'"))
            ).scalar_one_or_none()
        if stored_dim and int(stored_dim) != self.settings.embedding_dim:
            raise ConfigurationError(
                "EMBEDDING_DIM 与数据库中的向量维度不一致，请重新索引或修正配置",
                details={
                    "configured": self.settings.embedding_dim,
                    "database": int(stored_dim),
                },
            )

    async def ensure_default_user(self) -> uuid.UUID:
        """Create the single local profile if it is missing."""
        user_id = uuid.UUID(self.settings.default_user_id)
        async with self.session() as session:
            user = await session.get(User, user_id)
            if user is None:
                session.add(User(id=user_id, display_name=self.settings.default_user_name))
                logger.info("default_user_created", extra={"user_id": str(user_id)})
        return user_id

    async def dispose(self) -> None:
        await self.engine.dispose()


_database: Database | None = None


def get_database() -> Database:
    global _database
    if _database is None:
        _database = Database()
    return _database


def set_database(database: Database | None) -> None:
    """Test hook: swap in a database bound to a throwaway instance."""
    global _database
    _database = database
