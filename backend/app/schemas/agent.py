"""Domain models produced and consumed by the decision workflow.

These types are the contract between three layers: the LLM (via structured
output), the LangGraph state, and the HTTP API. Keeping one definition means the
frontend can rely on exactly the shape the agent reasoned about.

A note on honesty: every motive/constraint carries ``provenance``. Nothing the
agent merely *suspects* is ever presented as a fact the user said.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.db.enums import (
    ConfirmationStatus,
    MemoryKind,
    MemoryProvenance,
    RiskLevel,
    SourceType,
)

MotiveKind = Literal["greed", "fear"]
ConstraintKind = Literal["time", "money", "family", "health", "legal", "skill", "other"]


class Option(BaseModel):
    """One path the user could take."""

    id: str = Field(description="短横线连接的小写英文 slug，例如 stay / leave")
    label: str = Field(description="选项的简短中文名称")
    description: str = Field(default="", description="一句话说明该选项具体意味着什么")

    @field_validator("id", mode="before")
    @classmethod
    def _slugify(cls, value: object) -> str:
        raw = str(value or "").strip().lower()
        cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in raw)
        return cleaned.strip("-") or "option"


class Motive(BaseModel):
    """A greed (what the user wants to gain) or a fear (what they dread losing)."""

    id: str = ""
    kind: MotiveKind
    statement: str = Field(description="用第二人称、具体、不含评判的一句话，例如「你害怕失去稳定的现金流」")
    intensity: int = Field(default=3, ge=1, le=5, description="强度 1-5")
    evidence: str = Field(default="", description="用户在对话中的原话片段，无则为空")
    provenance: MemoryProvenance = MemoryProvenance.AGENT_INFERRED
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    status: ConfirmationStatus = ConfirmationStatus.PENDING
    note: str = Field(default="", description="Agent 的补充说明，可为空")


class Constraint(BaseModel):
    """A real-world limit that any recommendation has to respect."""

    id: str = ""
    statement: str
    kind: ConstraintKind = "other"
    hard: bool = Field(default=True, description="true=硬约束（不可逾越），false=软约束（可协商）")
    evidence: str = ""
    provenance: MemoryProvenance = MemoryProvenance.AGENT_INFERRED
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    status: ConfirmationStatus = ConfirmationStatus.PENDING


class Contradiction(BaseModel):
    """Something the user said that does not line up with something else."""

    description: str = Field(description="中立描述，不给人格或心理贴标签")
    between: list[str] = Field(default_factory=list, description="互相张力的两个说法")
    evidence: str = ""
    #: Always true: a noticed tension is a hypothesis about the *words*, not the person.
    is_inference: bool = True


class Dilemma(BaseModel):
    statement: str = Field(default="", description="用一两句话概括用户面临的困境")
    domain: str = Field(default="其他", description="职业/关系/财务/健康/学业/其他")
    options: list[Option] = Field(default_factory=list)
    deadline: str = Field(default="", description="如果有时间压力，何时；否则留空")
    is_user_decision: bool = Field(default=True, description="这个选择是否由用户本人做出")
    other_parties: list[str] = Field(default_factory=list, description="受影响的其他人")


class Confirmation(BaseModel):
    """A judgement the agent wants the user to confirm, correct or reject."""

    id: str = ""
    topic: Literal["greed", "fear", "constraint", "goal", "value", "contradiction"] = "fear"
    claim: str = Field(description="Agent 想确认的判断，用第二人称表述")
    evidence: str = Field(default="", description="支撑该判断的用户原话")
    question: str = Field(default="", description="向用户提出的确认问题，要自然、不逼供")
    status: ConfirmationStatus = ConfirmationStatus.PENDING
    resolution_note: str = Field(default="", description="用户确认/纠正后的原话")


class Citation(BaseModel):
    """A retrieved knowledge-base chunk, with its real provenance."""

    chunk_id: str
    document_title: str
    heading: str = ""
    source_path: str
    source_type: SourceType = SourceType.CURATED_NOTE
    verified: bool = False
    score: float = 0.0
    excerpt: str = ""


class RiskItem(BaseModel):
    description: str
    level: RiskLevel = RiskLevel.MEDIUM
    mitigation: str = Field(default="", description="如果真的发生，可以做什么")


class OptionCost(BaseModel):
    """What each option costs — the heart of the analysis."""

    option_id: str
    label: str
    satisfies: list[str] = Field(default_factory=list, description="满足的贪婪/回避的恐惧")
    costs: list[str] = Field(default_factory=list, description="必须承担的代价")
    risks: list[RiskItem] = Field(default_factory=list)
    reversibility: Literal["reversible", "partially", "irreversible"] = "partially"
    cost_level: int = Field(default=3, ge=1, le=5)


class Decision(BaseModel):
    """The final recommendation, or an explicit refusal to recommend."""

    #: recommend=明确建议；defer=信息不足，暂缓；hold_safety=存在严重风险，停止
    status: Literal["recommend", "defer", "hold_safety"] = "defer"
    chosen_option_id: str | None = None
    chosen_label: str | None = None
    headline: str = Field(default="", description="一句话结论")
    why_this_choice: str = Field(default="", description="为什么这个祸值得主动承担")
    reasons: list[str] = Field(default_factory=list)
    must_give_up: list[str] = Field(default_factory=list, description="选它必须放弃什么")
    risks: list[RiskItem] = Field(default_factory=list)
    bottom_lines: list[str] = Field(default_factory=list, description="不可逾越的底线")
    reassess_if: list[str] = Field(default_factory=list, description="出现什么情况应当重新评估")
    options_analysis: list[OptionCost] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    defer_reason: str = Field(default="", description="status=defer 时说明还缺什么信息")
    missing_information: list[str] = Field(default_factory=list)
    safety_note: str = Field(default="", description="status=hold_safety 时的说明与求助建议")
    is_final: bool = False


class MemoryRef(BaseModel):
    """A recalled long-term memory, as shown to the user and to the prompt."""

    id: str
    kind: MemoryKind
    content: str
    provenance: MemoryProvenance
    status: str
    score: float = 0.0
    source_session_id: str | None = None
    created_at: str | None = None


class MemoryCandidate(BaseModel):
    """A long-term memory the agent proposes to store or update."""

    kind: MemoryKind
    content: str
    normalized_key: str = ""
    provenance: MemoryProvenance
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    importance: int = Field(default=3, ge=1, le=5)
    evidence: str = ""
    reason: str = Field(default="", description="为什么值得长期保存")


class SessionAnalysis(BaseModel):
    """Everything the agent currently believes about the user's situation."""

    dilemma: Dilemma = Field(default_factory=Dilemma)
    greeds: list[Motive] = Field(default_factory=list)
    fears: list[Motive] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    confirmations: list[Confirmation] = Field(default_factory=list)
    recalled_memories: list[MemoryRef] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    sufficiency: float = 0.0
    sufficiency_rationale: str = ""
    risk_flags: list[str] = Field(default_factory=list)
