"""Graph construction and runtime.

The workflow is a single bounded pass per user message — the agent decides how
to spend the turn, then hands control back. There is deliberately no unbounded
agent loop: every path ends in ``finalize``.

    START → understand → recall → route → ┬ explore ──┐
                                          ├ confirm ──┼→ finalize → END
                                          ├ safety ───┘
                                          └ knowledge → decide → finalize

Persistence uses LangGraph's PostgreSQL checkpointer, so a session resumes
exactly where it stopped — including the structured analysis, not just the
messages. Long-term memory lives in the application's own tables instead of the
LangGraph Store: it needs provenance, de-duplication, conflict resolution,
audit history and per-user isolation, which a KV+vector namespace cannot express
cleanly. See docs/architecture.md for the full rationale.
"""

from __future__ import annotations

from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from psycopg_pool import AsyncConnectionPool

from app.agent import nodes
from app.agent.serde import build_serde
from app.agent.state import AleaState
from app.core.compat import configure_event_loop_policy, is_selector_loop_running
from app.core.config import Settings
from app.core.errors import ConfigurationError
from app.core.logging import get_logger

# psycopg's async driver needs a Selector loop on Windows; must run before any
# event loop is created. See app/core/compat.py.
configure_event_loop_policy()

logger = get_logger(__name__)


def build_graph() -> StateGraph:
    graph: StateGraph = StateGraph(AleaState)

    graph.add_node("understand", nodes.understand)
    graph.add_node("recall", nodes.recall)
    graph.add_node("route", nodes.route)
    graph.add_node("explore", nodes.explore)
    graph.add_node("confirm", nodes.confirm)
    graph.add_node("knowledge", nodes.knowledge)
    graph.add_node("decide", nodes.decide)
    graph.add_node("safety", nodes.safety)
    graph.add_node("finalize", nodes.finalize)

    graph.add_edge(START, "understand")
    graph.add_edge("understand", "recall")
    graph.add_edge("recall", "route")

    graph.add_conditional_edges(
        "route",
        lambda state: state.get("route", "explore"),
        {
            "explore": "explore",
            "confirm": "confirm",
            "knowledge": "knowledge",
            "decide": "decide",
            "safety": "safety",
        },
    )

    graph.add_edge("explore", "finalize")
    graph.add_edge("confirm", "finalize")
    graph.add_edge("safety", "finalize")
    graph.add_edge("knowledge", "decide")
    graph.add_edge("decide", "finalize")
    graph.add_edge("finalize", END)
    return graph


class AgentRuntime:
    """Owns the checkpointer pool and the compiled graph."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._pool: AsyncConnectionPool | None = None
        self._saver: AsyncPostgresSaver | None = None
        self.graph: CompiledStateGraph | None = None

    async def start(self) -> None:
        if not is_selector_loop_running():  # pragma: no cover - Windows only
            logger.error(
                "incompatible_event_loop",
                extra={
                    "hint": "Windows 上 psycopg 异步驱动需要 SelectorEventLoop；"
                    "请用 `python -m scripts.serve` 启动服务，"
                    "或在使用 uvicorn 前调用 app.core.compat.configure_event_loop_policy()"
                },
            )

        self._pool = AsyncConnectionPool(
            conninfo=self.settings.checkpoint_database_url,
            min_size=1,
            max_size=max(4, self.settings.db_pool_size // 2),
            open=False,
            kwargs={"autocommit": True, "prepare_threshold": 0},
        )
        try:
            await self._pool.open(wait=True, timeout=self.settings.db_pool_open_timeout)
        except Exception as exc:
            await self._pool.close()
            raise ConfigurationError(
                "无法连接 LangGraph checkpointer 使用的数据库连接池",
                details={
                    "dsn_host": self.settings.checkpoint_database_url.rsplit("@", 1)[-1],
                    "error": type(exc).__name__,
                    "hint": "检查数据库是否启动；Windows 上需使用 scripts.serve 启动以选用 Selector 事件循环",
                },
            ) from exc

        # psycopg_pool's row-factory generic does not line up with the saver's
        # annotation even though the runtime contract matches.
        self._saver = AsyncPostgresSaver(self._pool, serde=build_serde())  # type: ignore[arg-type]
        await self._saver.setup()
        self.graph = build_graph().compile(checkpointer=self._saver)
        logger.info("agent_runtime_started", extra={"model": self.settings.llm_model})

    async def stop(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def require_graph(self) -> CompiledStateGraph:
        if self.graph is None:
            raise RuntimeError("AgentRuntime 尚未启动")
        return self.graph

    @staticmethod
    def thread_config(session_id: str, deps: Any | None = None) -> RunnableConfig:
        configurable: dict[str, Any] = {"thread_id": session_id}
        if deps is not None:
            configurable["deps"] = deps
        return {"configurable": configurable, "recursion_limit": 40}

    async def get_state(self, session_id: str, deps: Any | None = None) -> AleaState | None:
        snapshot = await self.require_graph().aget_state(self.thread_config(session_id, deps))
        if snapshot is None or not snapshot.values:
            return None
        return cast(AleaState, snapshot.values)

    async def update_confirmations(self, session_id: str, confirmations: list[Any], deps: Any | None = None) -> None:
        """Push a UI-side decision back into the checkpointer."""
        await self.require_graph().aupdate_state(
            self.thread_config(session_id, deps),
            {"confirmations": confirmations, "confirmations_just_resolved": True},
        )
