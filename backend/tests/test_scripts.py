"""The command-line helpers are deliverables too, so execute them.

Linting a script proves nothing about whether it runs; these tests drive the same
code paths a user would hit from a terminal.
"""

from __future__ import annotations

import pytest

from app.core.config import get_settings
from scripts import ingest_knowledge, smoke_llm

from .fakes import one_hot
from .test_agent_flow import understanding

pytestmark = pytest.mark.db


async def test_smoke_script_completes_a_full_run(knowledge, patch_llm, capsys):
    """`scripts.smoke_llm` must survive a real replay, not just import."""
    turns = smoke_llm.SCRIPTED_TURNS
    scripts: list[list[object]] = []
    for index in range(len(turns)):
        scripts.append([understanding(sufficiency=0.2)])
        scripts.append([f"第 {index + 1} 轮追问：你更怕失去的是哪一样？"])
        scripts.append([{"memories": []}])
    patch_llm(scripts)

    exit_code = await smoke_llm.run(list(turns), interactive=False)
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "信息充分度" in output
    assert "结束：" in output
    # Every scripted turn must have been consumed — no leftover model calls.
    assert all(model.remaining() == 0 for model in patch_llm.models)


async def test_ingest_script_reports_a_missing_embedding_key(patch_llm, monkeypatch, capsys):
    """Without a key the CLI must explain the fix and exit non-zero, not traceback."""
    from app.llm import client as llm

    async def explode(_texts: list[str]) -> list[list[float]]:
        from app.core.errors import ConfigurationError

        raise ConfigurationError("未配置 EMBEDDING_API_KEY", details={"env": "EMBEDDING_API_KEY"})

    monkeypatch.setattr(llm, "embed_texts", explode)
    exit_code = await ingest_knowledge.run(force=False)
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "EMBEDDING_API_KEY" in (captured.out + captured.err)


async def test_ingest_script_writes_the_corpus_and_is_idempotent(container, patch_llm, capsys):
    install = patch_llm([[]])
    install.embedder.default_vector = one_hot(get_settings().embedding_dim, 3)

    assert await ingest_knowledge.run(force=False) == 0
    first = capsys.readouterr().out
    assert "写入切片" in first

    # Second run over unchanged files must not rewrite anything.
    assert await ingest_knowledge.run(force=False) == 0
    second = capsys.readouterr().out
    assert "未变化" in second

    assert await container.knowledge_service.count_chunks() > 0
