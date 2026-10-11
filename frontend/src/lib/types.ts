/**
 * Mirrors app/schemas/*.py on the backend.
 *
 * Kept hand-written rather than generated so that the contract is visible in
 * review; the backend is the source of truth and any change there must be
 * reflected here.
 */

export type MemoryKind =
  | "goal"
  | "value"
  | "greed"
  | "fear"
  | "constraint"
  | "decision"
  | "preference"
  | "fact";

export type MemoryStatus = "candidate" | "active" | "rejected" | "superseded";

export type MemoryProvenance = "user_stated" | "user_confirmed" | "agent_inferred";

export type ConfirmationStatus = "pending" | "confirmed" | "corrected" | "rejected";

export type SourceType = "verified_excerpt" | "curated_note" | "case";

export type RiskLevel = "low" | "medium" | "high" | "critical";

export type DecisionStatus = "recommend" | "defer" | "hold_safety";

export interface Option {
  id: string;
  label: string;
  description: string;
}

export interface Motive {
  id: string;
  kind: "greed" | "fear";
  statement: string;
  intensity: number;
  evidence: string;
  provenance: MemoryProvenance;
  confidence: number;
  status: ConfirmationStatus;
  note: string;
}

export interface Constraint {
  id: string;
  statement: string;
  kind: string;
  hard: boolean;
  evidence: string;
  provenance: MemoryProvenance;
  confidence: number;
  status: ConfirmationStatus;
}

export interface Contradiction {
  description: string;
  between: string[];
  evidence: string;
  is_inference: boolean;
}

export interface Dilemma {
  statement: string;
  domain: string;
  options: Option[];
  deadline: string;
  is_user_decision: boolean;
  other_parties: string[];
}

export interface Confirmation {
  id: string;
  topic: "greed" | "fear" | "constraint" | "goal" | "value" | "contradiction";
  claim: string;
  evidence: string;
  question: string;
  status: ConfirmationStatus;
  resolution_note: string;
}

export interface Citation {
  chunk_id: string;
  document_title: string;
  heading: string;
  source_path: string;
  source_type: SourceType;
  verified: boolean;
  score: number;
  excerpt: string;
}

export interface RiskItem {
  description: string;
  level: RiskLevel;
  mitigation: string;
}

export interface OptionCost {
  option_id: string;
  label: string;
  satisfies: string[];
  costs: string[];
  risks: RiskItem[];
  reversibility: "reversible" | "partially" | "irreversible";
  cost_level: number;
}

export interface Decision {
  status: DecisionStatus;
  chosen_option_id: string | null;
  chosen_label: string | null;
  headline: string;
  why_this_choice: string;
  reasons: string[];
  must_give_up: string[];
  risks: RiskItem[];
  bottom_lines: string[];
  reassess_if: string[];
  options_analysis: OptionCost[];
  citations: Citation[];
  confidence: number;
  defer_reason: string;
  missing_information: string[];
  safety_note: string;
  is_final: boolean;
}

export interface MemoryRef {
  id: string;
  kind: MemoryKind;
  content: string;
  provenance: MemoryProvenance;
  status: string;
  score: number;
  source_session_id: string | null;
  created_at: string | null;
}

export interface SessionAnalysis {
  dilemma: Dilemma;
  greeds: Motive[];
  fears: Motive[];
  constraints: Constraint[];
  contradictions: Contradiction[];
  open_questions: string[];
  confirmations: Confirmation[];
  recalled_memories: MemoryRef[];
  citations: Citation[];
  sufficiency: number;
  sufficiency_rationale: string;
  risk_flags: string[];
}

export interface SessionSummary {
  id: string;
  title: string;
  status: "active" | "archived";
  phase: string;
  turn_count: number;
  info_sufficiency: number;
  last_active_at: string;
  created_at: string;
  has_decision: boolean;
}

export interface Message {
  id: string;
  session_id: string;
  seq: number;
  role: "user" | "assistant" | "system" | "tool";
  content: string;
  node: string | null;
  meta: Record<string, unknown>;
  created_at: string;
}

export interface SessionDetail extends SessionSummary {
  analysis: SessionAnalysis;
  decision: Decision | null;
  messages: Message[];
}

export interface Memory {
  id: string;
  user_id: string;
  kind: MemoryKind;
  content: string;
  normalized_key: string;
  status: MemoryStatus;
  provenance: MemoryProvenance;
  confidence: number;
  importance: number;
  evidence: string | null;
  source_session_id: string | null;
  source_message_id: string | null;
  superseded_by: string | null;
  tags: string[];
  use_count: number;
  last_used_at: string | null;
  deleted_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface MemoryAudit {
  id: string;
  memory_id: string;
  action: string;
  actor: string;
  before: Record<string, unknown> | null;
  after: Record<string, unknown> | null;
  reason: string | null;
  created_at: string;
}

export interface DecisionEvent {
  id: string;
  session_id: string | null;
  title: string;
  situation: string;
  chosen_option: string | null;
  greed: string[];
  fear: string[];
  cost: string[];
  risks: string[];
  outcome: string | null;
  summary: string | null;
  created_at: string;
}

export interface KnowledgeDocument {
  id: string;
  source_path: string;
  title: string;
  source_type: SourceType;
  verified: boolean;
  version: number;
  updated_at: string;
  chunk_count: number;
}

export interface Health {
  status: "ok" | "degraded";
  version: string;
  database: boolean;
  llm_configured: boolean;
  embedding_provider: string;
  knowledge_chunks: number;
}

export interface RuntimeConfig {
  app_name: string;
  app_env: string;
  llm_model: string;
  llm_base_url: string;
  embedding_provider: string;
  embedding_model: string;
  embedding_dim: number;
  agent_soft_turn_limit: number;
  agent_hard_turn_limit: number;
  agent_sufficiency_threshold: number;
}

export interface MemoryChange {
  action: string;
  memory_id: string;
  content: string;
  kind: MemoryKind;
  status: MemoryStatus;
}

/** Events emitted by POST /api/sessions/{id}/stream */
export type StreamEvent =
  | { type: "start"; session_id: string; user_message_id: string }
  | { type: "node"; node: string; phase: string | null }
  | { type: "token"; node: string; text: string }
  | { type: "memory"; changes: MemoryChange[] }
  | { type: "decision"; decision: Decision }
  | {
      type: "done";
      message: Message;
      analysis: SessionAnalysis;
      decision: Decision | null;
      phase: string;
      turn_count: number;
      citations: Citation[];
      memory_changes: MemoryChange[];
      message_id: string;
    }
  | { type: "error"; message: string };
