"""Turning conversation into long-term memory candidates.

The extractor is deliberately conservative: it would rather miss a durable fact
than store an inference as if the user had said it.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.core.config import Settings
from app.core.logging import get_logger
from app.db.enums import MemoryKind, MemoryProvenance
from app.llm import client as llm
from app.memory.prompts import EXTRACTION_SYSTEM, EXTRACTION_USER
from app.schemas.agent import MemoryCandidate, SessionAnalysis

logger = get_logger(__name__)


class ExtractedMemory(BaseModel):
    kind: MemoryKind
    content: str = Field(description="一条独立、自洽、可被证伪的陈述")
    normalized_key: str = Field(default="", description="不超过 12 个字的话题标签")
    provenance: Literal["user_stated", "user_confirmed", "agent_inferred"]
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    importance: int = Field(default=3, ge=1, le=5)
    evidence: str = Field(default="", description="用户原话片段，逐字引用")


class MemoryExtraction(BaseModel):
    memories: list[ExtractedMemory] = Field(default_factory=list)


def _render_transcript(turns: list[tuple[str, str]], *, limit: int = 12) -> str:
    lines: list[str] = []
    for role, content in turns[-limit:]:
        speaker = "用户" if role == "user" else "Agent"
        lines.append(f"{speaker}：{content.strip()}")
    return "\n".join(lines)


def _render_analysis(analysis: SessionAnalysis) -> str:
    """Show the extractor what has already been confirmed, so it can label the
    provenance of anything it re-derives instead of guessing."""
    lines: list[str] = []
    for kind, items in (
        ("贪婪", analysis.greeds),
        ("恐惧", analysis.fears),
        ("约束", analysis.constraints),
    ):
        for item in items:
            lines.append(f"- [{kind}][{item.status}] {item.statement}")
    for confirmation in analysis.confirmations:
        lines.append(f"- [待确认项][{confirmation.status}] {confirmation.claim}")
    return "\n".join(lines) if lines else "（本轮没有记录到具体判断）"


async def extract_candidates(
    *,
    settings: Settings,
    analysis: SessionAnalysis,
    turns: list[tuple[str, str]],
    latest_user_text: str,
) -> list[MemoryCandidate]:
    """Return memory candidates worth considering for long-term storage."""
    if len(latest_user_text.strip()) < settings.memory_min_message_chars:
        return []

    model = llm.get_chat_model(temperature=0.0, max_tokens=1200)
    prompt = EXTRACTION_USER.format(
        dilemma=analysis.dilemma.statement or "（尚未明确）",
        analysis_summary=_render_analysis(analysis),
        transcript=_render_transcript(turns),
    )
    try:
        result = await llm.call_structured(
            model,
            MemoryExtraction,
            [SystemMessage(content=EXTRACTION_SYSTEM), HumanMessage(content=prompt)],
            node="memory_extract",
        )
    except Exception as exc:  # noqa: BLE001 - extraction must never break a turn
        logger.warning("memory_extraction_failed", extra={"error": type(exc).__name__})
        return []

    candidates = [
        MemoryCandidate(
            kind=item.kind,
            content=item.content.strip(),
            normalized_key=(item.normalized_key or item.content)[:120],
            provenance=MemoryProvenance(item.provenance),
            confidence=item.confidence,
            importance=item.importance,
            evidence=item.evidence.strip()[:1000],
            reason="对话提取",
        )
        for item in result.memories[:5]
        if item.content.strip()
    ]
    return [c for c in candidates if c.importance >= settings.memory_min_importance]
