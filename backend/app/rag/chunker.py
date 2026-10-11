"""Markdown-aware chunking.

Headings are preserved as breadcrumbs (``# 总论 > ## 代价``) so that a retrieved
chunk can always be traced back to a specific section of a specific file — the
citation is only useful if it points at something the reader can check.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.rag.loader import KnowledgeDoc

_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]

#: Chinese text has no spaces, so sentence punctuation has to do the work.
_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]

#: Fragments shorter than this carry no retrievable meaning and are dropped.
_MIN_CHUNK_CHARS = 20


@dataclass(slots=True)
class Chunk:
    ordinal: int
    heading: str
    content: str


def chunk_document(doc: KnowledgeDoc, *, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS, strip_headers=False, return_each_line=False
    )
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_SEPARATORS,
        length_function=len,
        keep_separator=True,
    )

    sections = header_splitter.split_text(doc.body) or []
    chunks: list[Chunk] = []
    ordinal = 0
    for section in sections:
        breadcrumb = " > ".join(str(value) for key, value in (section.metadata or {}).items() if value)
        for piece in size_splitter.split_text(section.page_content):
            text = piece.strip()
            if len(text) < _MIN_CHUNK_CHARS:  # headings or stray lines on their own
                continue
            chunks.append(Chunk(ordinal=ordinal, heading=breadcrumb, content=text))
            ordinal += 1

    if not chunks and len(doc.body.strip()) >= _MIN_CHUNK_CHARS:
        # A document with no headings still has content worth indexing — but
        # only if there is enough of it to be retrievable at all.
        chunks.append(Chunk(ordinal=0, heading="", content=doc.body.strip()[:chunk_size]))
    return chunks
