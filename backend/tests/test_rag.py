"""Knowledge base loading and chunking."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.errors import InvalidRequestError
from app.db.enums import SourceType
from app.rag.chunker import chunk_document
from app.rag.loader import iter_documents, parse_document

pytestmark = pytest.mark.no_db


def _write(directory: Path, name: str, content: str) -> Path:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_front_matter_is_parsed_into_provenance(tmp_path: Path):
    _write(
        tmp_path,
        "a.md",
        "---\ntitle: 标题\nsource: 人选天选论\nsource_type: curated_note\nverified: false\n---\n\n"
        "## 小节\n\n正文内容。\n",
    )
    doc = parse_document(tmp_path / "a.md", root=tmp_path)
    assert doc.title == "标题"
    assert doc.source_type is SourceType.CURATED_NOTE
    assert doc.verified is False
    assert doc.source_path == "a.md"


def test_verified_excerpt_must_actually_be_marked_verified(tmp_path: Path):
    """A document cannot claim to be a checked quotation without saying so."""
    _write(
        tmp_path,
        "b.md",
        "---\ntitle: 原文\nsource_type: verified_excerpt\nverified: false\n---\n\n内容\n",
    )
    with pytest.raises(InvalidRequestError):
        parse_document(tmp_path / "b.md", root=tmp_path)


def test_unknown_source_type_is_rejected(tmp_path: Path):
    _write(tmp_path, "c.md", "---\ntitle: x\nsource_type: hearsay\n---\n\n内容\n")
    with pytest.raises(InvalidRequestError):
        parse_document(tmp_path / "c.md", root=tmp_path)


def test_title_falls_back_to_the_first_heading(tmp_path: Path):
    _write(tmp_path, "d.md", "---\nsource_type: case\n---\n\n# 从标题来的名字\n\n内容\n")
    doc = parse_document(tmp_path / "d.md", root=tmp_path)
    assert doc.title == "从标题来的名字"


def test_readme_and_underscore_files_are_not_indexed(tmp_path: Path):
    _write(tmp_path, "README.md", "# 说明\n")
    _write(tmp_path, "_draft.md", "---\ntitle: 草稿\n---\n\n草稿内容\n")
    _write(tmp_path, "_templates/template.md", "---\ntitle: 模板\n---\n\n模板内容\n")
    _write(tmp_path, "real.md", "---\ntitle: 正式\n---\n\n正文\n")
    paths = [doc.source_path for doc in iter_documents(tmp_path)]
    assert paths == ["real.md"]


def test_the_shipped_template_is_not_retrievable():
    """The excerpt template declares `verified: true`; indexing it would let the
    agent 'quote' a form."""
    root = Path(__file__).resolve().parents[1] / "knowledge"
    paths = [doc.source_path for doc in iter_documents(root)]
    assert not any("template" in path for path in paths)


def test_checksum_changes_only_when_content_changes(tmp_path: Path):
    path = _write(tmp_path, "e.md", "---\ntitle: t\n---\n\n原始正文\n")
    first = parse_document(path, root=tmp_path).checksum
    path.write_text("---\ntitle: t\n---\n\n修改后的正文\n", encoding="utf-8")
    assert parse_document(path, root=tmp_path).checksum != first


# ------------------------------------------------------------------ chunking
def test_chunking_keeps_the_heading_breadcrumb():
    from app.rag.loader import KnowledgeDoc

    body = "\n".join(
        [
            "# 总纲",
            "",
            "总纲的第一段文字，用来填充内容长度，让切分器有东西可以切。" * 2,
            "",
            "## 代价",
            "",
            "代价这一节解释可逆与不可逆的区别，需要足够长才会产生切片。" * 3,
        ]
    )
    doc = KnowledgeDoc(
        source_path="x.md",
        title="x",
        body=body,
        source_type=SourceType.CURATED_NOTE,
        verified=False,
    )
    chunks = chunk_document(doc, chunk_size=120, chunk_overlap=20)
    assert chunks
    assert any("总纲" in chunk.heading for chunk in chunks)
    assert any("代价" in chunk.heading for chunk in chunks)
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))


def test_tiny_fragments_are_dropped():
    from app.rag.loader import KnowledgeDoc

    doc = KnowledgeDoc(
        source_path="y.md",
        title="y",
        body="# 标题\n\n短\n",
        source_type=SourceType.CURATED_NOTE,
        verified=False,
    )
    assert chunk_document(doc, chunk_size=200, chunk_overlap=0) == []


def test_bundled_knowledge_base_parses_and_chunks():
    """The shipped corpus must be valid, or ingestion would fail at runtime."""
    root = Path(__file__).resolve().parents[1] / "knowledge"
    docs = iter_documents(root)
    assert len(docs) >= 5
    for doc in docs:
        assert doc.body.strip(), f"{doc.source_path} 没有正文"
        if doc.source_type is SourceType.VERIFIED_EXCERPT:
            assert doc.verified is True
        chunks = chunk_document(doc, chunk_size=700, chunk_overlap=120)
        assert chunks, f"{doc.source_path} 切不出切片"
