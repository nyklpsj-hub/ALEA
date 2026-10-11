"""Loading and parsing of the public knowledge base.

The corpus is a directory of Markdown files with YAML front-matter:

    ---
    title: 选祸：把代价拿到台面上
    source: 人选天选论（路飞）
    source_type: curated_note
    verified: false
    tags: [核心概念]
    ---

``source_type`` and ``verified`` travel all the way to the citations the user
sees, so a curated note can never be displayed as if it were the author's own
words.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.core.errors import InvalidRequestError
from app.db.enums import SourceType

_FRONT_MATTER_DELIM = "---"


@dataclass(slots=True)
class KnowledgeDoc:
    source_path: str
    title: str
    body: str
    source_type: SourceType
    verified: bool
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def checksum(self) -> str:
        payload = f"{self.title}\n{self.source_type}\n{self.verified}\n{self.body}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    if not text.lstrip().startswith(_FRONT_MATTER_DELIM):
        return {}, text
    stripped = text.lstrip()
    parts = stripped.split(_FRONT_MATTER_DELIM, 2)
    if len(parts) < 3:
        return {}, text
    raw_meta, body = parts[1], parts[2]
    try:
        meta = yaml.safe_load(raw_meta) or {}
    except yaml.YAMLError as exc:
        raise InvalidRequestError(f"知识库 front-matter 解析失败：{exc}") from exc
    if not isinstance(meta, dict):
        raise InvalidRequestError("知识库 front-matter 必须是键值对")
    return meta, body.lstrip("\n")


def parse_document(path: Path, *, root: Path) -> KnowledgeDoc:
    text = path.read_text(encoding="utf-8")
    meta, body = _split_front_matter(text)
    rel = path.relative_to(root).as_posix()

    raw_type = str(meta.get("source_type", SourceType.CURATED_NOTE)).strip()
    try:
        source_type = SourceType(raw_type)
    except ValueError as exc:
        raise InvalidRequestError(f"{rel}: source_type 必须是 {[t.value for t in SourceType]} 之一") from exc

    title = str(meta.get("title") or _first_heading(body) or path.stem)
    verified = bool(meta.get("verified", False))
    if source_type is SourceType.VERIFIED_EXCERPT and not verified:
        raise InvalidRequestError(
            f"{rel}: source_type=verified_excerpt 必须同时标注 verified: true，未核对原文时请使用 curated_note"
        )

    return KnowledgeDoc(
        source_path=rel,
        title=title,
        body=body.strip(),
        source_type=source_type,
        verified=verified,
        meta={k: v for k, v in meta.items() if k not in {"title", "source_type", "verified"}},
    )


def _first_heading(body: str) -> str | None:
    for line in body.splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return None


def iter_documents(root: Path) -> list[KnowledgeDoc]:
    if not root.exists():
        raise InvalidRequestError(f"知识库目录不存在：{root}")
    docs: list[KnowledgeDoc] = []
    for path in sorted(root.rglob("*.md")):
        if path.name.lower().startswith("readme"):
            continue
        if path.name.lower().startswith("_"):
            # Templates and drafts are kept out of the index on purpose.
            continue
        docs.append(parse_document(path, root=root))
    return docs
