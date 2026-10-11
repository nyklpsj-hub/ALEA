import type {
  DecisionEvent,
  Health,
  KnowledgeDocument,
  Memory,
  MemoryAudit,
  MemoryChange,
  MemoryKind,
  MemoryStatus,
  RuntimeConfig,
  SessionDetail,
  SessionSummary,
  StreamEvent,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ?? "http://127.0.0.1:8000";

export class ApiError extends Error {
  code: string;
  status: number;
  details: Record<string, unknown>;

  constructor(message: string, code: string, status: number, details: Record<string, unknown> = {}) {
    super(message);
    this.code = code;
    this.status = status;
    this.details = details;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    cache: "no-store",
  });

  if (response.status === 204) {
    return undefined as T;
  }

  const text = await response.text();
  const payload = text ? safeJson(text) : {};

  if (!response.ok) {
    const error = (payload as { error?: { code?: string; message?: string; details?: object } })
      .error;
    throw new ApiError(
      error?.message ?? `请求失败 (${response.status})`,
      error?.code ?? "http_error",
      response.status,
      (error?.details as Record<string, unknown>) ?? {},
    );
  }
  return payload as T;
}

function safeJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return {};
  }
}

// ------------------------------------------------------------------ system
export const getHealth = () => request<Health>("/api/health");
export const getRuntimeConfig = () => request<RuntimeConfig>("/api/config");

// ---------------------------------------------------------------- sessions
export const listSessions = () => request<SessionSummary[]>("/api/sessions");

export const createSession = (title?: string) =>
  request<SessionDetail>("/api/sessions", {
    method: "POST",
    body: JSON.stringify({ title: title ?? null }),
  });

export const getSession = (id: string) => request<SessionDetail>(`/api/sessions/${id}`);

export const updateSession = (id: string, payload: { title?: string; status?: string }) =>
  request<SessionSummary>(`/api/sessions/${id}`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });

export const deleteSession = (id: string) =>
  request<void>(`/api/sessions/${id}`, { method: "DELETE" });

// -------------------------------------------------------------- chat/SSE
export interface StreamHandlers {
  onEvent: (event: StreamEvent) => void;
  signal?: AbortSignal;
}

/**
 * Streams a turn. The endpoint is a POST, so EventSource cannot be used; we
 * read the SSE frames off the response body ourselves.
 */
export async function streamMessage(
  sessionId: string,
  content: string,
  { onEvent, signal }: StreamHandlers,
): Promise<void> {
  const response = await fetch(`${API_BASE}/api/sessions/${sessionId}/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content }),
    signal,
  });

  if (!response.ok || !response.body) {
    const text = await response.text().catch(() => "");
    const payload = safeJson(text) as { error?: { message?: string } };
    throw new ApiError(
      payload?.error?.message ?? `无法开始对话 (${response.status})`,
      "stream_failed",
      response.status,
    );
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const dataLine = frame
        .split("\n")
        .find((line) => line.startsWith("data:"));
      if (dataLine) {
        const raw = dataLine.slice(5).trim();
        if (raw) {
          try {
            onEvent(JSON.parse(raw) as StreamEvent);
          } catch {
            // A malformed frame should not kill the stream.
          }
        }
      }
      boundary = buffer.indexOf("\n\n");
    }
  }
}

export const resolveConfirmation = (
  sessionId: string,
  confirmationId: string,
  action: "confirm" | "correct" | "reject",
  correctedText?: string,
) =>
  request<{ analysis: SessionDetail["analysis"]; memory_id: string | null }>(
    `/api/sessions/${sessionId}/confirmations/${confirmationId}`,
    {
      method: "POST",
      body: JSON.stringify({ confirmation_id: confirmationId, action, corrected_text: correctedText ?? null }),
    },
  );

// ---------------------------------------------------------------- memory
export interface MemoryFilters {
  kinds?: MemoryKind[];
  statuses?: MemoryStatus[];
  includeDeleted?: boolean;
}

export function listMemories(filters: MemoryFilters = {}): Promise<Memory[]> {
  const params = new URLSearchParams();
  filters.kinds?.forEach((kind) => params.append("kind", kind));
  filters.statuses?.forEach((status) => params.append("status", status));
  if (filters.includeDeleted) params.set("include_deleted", "true");
  const query = params.toString();
  return request<Memory[]>(`/api/memories${query ? `?${query}` : ""}`);
}

export const getMemoryStats = () => request<Record<string, number>>("/api/memories/stats");

/** Semantic search over this user's memories — the same path the agent uses. */
export const searchMemories = (payload: { query: string; kinds?: MemoryKind[]; top_k?: number }) =>
  request<Array<{ memory: Memory; score: number }>>("/api/memories/search", {
    method: "POST",
    body: JSON.stringify({
      query: payload.query,
      kinds: payload.kinds ?? [],
      top_k: payload.top_k ?? 6,
    }),
  });

export const updateMemory = (id: string, payload: Partial<Memory>) =>
  request<Memory>(`/api/memories/${id}`, { method: "PATCH", body: JSON.stringify(payload) });

export const deleteMemory = (id: string) =>
  request<void>(`/api/memories/${id}`, { method: "DELETE" });

export const restoreMemory = (id: string) =>
  request<Memory>(`/api/memories/${id}/restore`, { method: "POST" });

export const getMemoryHistory = (id: string) =>
  request<MemoryAudit[]>(`/api/memories/${id}/history`);

// ------------------------------------------------------------- decisions
export const listDecisionEvents = () => request<DecisionEvent[]>("/api/decisions");

export const updateDecisionOutcome = (id: string, outcome: string) =>
  request<DecisionEvent>(`/api/decisions/${id}/outcome`, {
    method: "PATCH",
    body: JSON.stringify({ outcome }),
  });

// ------------------------------------------------------------- knowledge
export const listKnowledgeDocuments = () =>
  request<KnowledgeDocument[]>("/api/knowledge/documents");

export const searchKnowledge = (query: string, topK = 5) =>
  request<{ query: string; results: SessionDetail["analysis"]["citations"]; used_min_score: number }>(
    "/api/knowledge/search",
    { method: "POST", body: JSON.stringify({ query, top_k: topK }) },
  );

export const reindexKnowledge = (force = false) =>
  request<Record<string, unknown>>(`/api/knowledge/reindex?force=${force}`, { method: "POST" });

export type { MemoryChange };
