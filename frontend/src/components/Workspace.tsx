"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import * as api from "@/lib/api";
import type {
  Decision,
  Health,
  MemoryChange,
  MemoryRef,
  Message,
  SessionAnalysis,
  SessionSummary,
  StreamEvent,
} from "@/lib/types";
import { AnalysisPanel } from "./AnalysisPanel";
import { ChatPanel } from "./ChatPanel";
import { DecisionCard } from "./DecisionCard";
import { SessionSidebar } from "./SessionSidebar";
import { Button, Panel } from "./ui";

const EMPTY_ANALYSIS: SessionAnalysis = {
  dilemma: {
    statement: "",
    domain: "其他",
    options: [],
    deadline: "",
    is_user_decision: true,
    other_parties: [],
  },
  greeds: [],
  fears: [],
  constraints: [],
  contradictions: [],
  open_questions: [],
  confirmations: [],
  recalled_memories: [],
  citations: [],
  sufficiency: 0,
  sufficiency_rationale: "",
  risk_flags: [],
};

export function Workspace({ sessionId }: { sessionId: string | null }) {
  const router = useRouter();

  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [messages, setMessages] = useState<Array<Pick<Message, "role" | "content" | "node">>>([]);
  const [analysis, setAnalysis] = useState<SessionAnalysis>(EMPTY_ANALYSIS);
  const [decision, setDecision] = useState<Decision | null>(null);
  const [liveText, setLiveText] = useState("");
  const [phase, setPhase] = useState("");
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [memoryFeed, setMemoryFeed] = useState<MemoryChange[]>([]);
  const [rightOpen, setRightOpen] = useState(true);
  const [loading, setLoading] = useState(false);

  const abortRef = useRef<AbortController | null>(null);
  // While a turn is streaming, the session id may change (a new session is
  // created on first send). Reloading then would wipe the streaming state.
  const streamingRef = useRef(false);

  // -------------------------------------------------------------- bootstrap
  const refreshSessions = useCallback(async () => {
    try {
      setSessions(await api.listSessions());
    } catch {
      /* sidebar failure is not fatal */
    }
  }, []);

  useEffect(() => {
    void refreshSessions();
    api
      .getHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, [refreshSessions]);

  useEffect(() => {
    if (!sessionId) {
      setMessages([]);
      setAnalysis(EMPTY_ANALYSIS);
      setDecision(null);
      setLiveText("");
      setPhase("");
      setMemoryFeed([]);
      return;
    }
    if (streamingRef.current) return;
    let cancelled = false;
    setLoading(true);
    api
      .getSession(sessionId)
      .then((detail) => {
        if (cancelled) return;
        setMessages(
          detail.messages.map((message) => ({
            role: message.role,
            content: message.content,
            node: message.node,
          })),
        );
        setAnalysis(detail.analysis);
        setDecision(detail.decision);
        setPhase(detail.phase);
        setLiveText("");
        setError(null);
      })
      .catch((cause: unknown) => {
        if (!cancelled) setError(cause instanceof Error ? cause.message : "加载会话失败");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  // ------------------------------------------------------------------ turn
  const handleEvent = useCallback((event: StreamEvent) => {
    switch (event.type) {
      case "token":
        setLiveText((previous) => previous + event.text);
        break;
      case "node":
        if (event.phase) setPhase(event.phase);
        break;
      case "memory":
        setMemoryFeed((previous) => [...previous, ...event.changes]);
        break;
      case "done":
        setMessages((previous) => [
          ...previous,
          { role: "assistant", content: event.message.content, node: event.phase },
        ]);
        setLiveText("");
        setAnalysis(event.analysis);
        setDecision(event.decision);
        setPhase(event.phase);
        break;
      case "error":
        setError(event.message);
        setLiveText("");
        break;
      default:
        break;
    }
  }, []);

  const send = useCallback(
    async (text: string) => {
      const content = text.trim();
      if (!content || busy) return;

      setInput("");
      setError(null);
      setBusy(true);
      setLiveText("");
      setMemoryFeed([]);
      setMessages((previous) => [...previous, { role: "user", content, node: null }]);

      const controller = new AbortController();
      abortRef.current = controller;
      streamingRef.current = true;

      try {
        let targetId = sessionId;
        if (!targetId) {
          const created = await api.createSession();
          targetId = created.id;
          router.replace(`/s/${targetId}`);
        }
        await api.streamMessage(targetId, content, {
          onEvent: handleEvent,
          signal: controller.signal,
        });
        void refreshSessions();
      } catch (cause) {
        if ((cause as Error)?.name !== "AbortError") {
          setError(cause instanceof Error ? cause.message : "发送失败");
          setMessages((previous) => {
            const last = previous[previous.length - 1];
            return last?.role === "user" ? previous.slice(0, -1) : previous;
          });
          setInput(content);
        }
      } finally {
        setBusy(false);
        setLiveText("");
        abortRef.current = null;
        streamingRef.current = false;
      }
    },
    [busy, handleEvent, refreshSessions, router, sessionId],
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
    setBusy(false);
  }, []);

  const resolveConfirmation = useCallback(
    async (id: string, action: "confirm" | "correct" | "reject", correctedText?: string) => {
      if (!sessionId) return;
      setError(null);
      try {
        const result = await api.resolveConfirmation(sessionId, id, action, correctedText);
        setAnalysis(result.analysis);
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : "更新确认项失败");
      }
    },
    [sessionId],
  );

  const startNew = useCallback(() => {
    router.push("/");
  }, [router]);

  const removeSession = useCallback(
    async (id: string) => {
      try {
        await api.deleteSession(id);
        await refreshSessions();
        if (id === sessionId) router.push("/");
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : "删除会话失败");
      }
    },
    [refreshSessions, router, sessionId],
  );

  const openMemory = useCallback(
    (memory: MemoryRef) => {
      router.push(`/memories?focus=${memory.id}`);
    },
    [router],
  );

  return (
    <div className="flex h-screen overflow-hidden">
      <SessionSidebar
        sessions={sessions}
        activeId={sessionId}
        health={health}
        onNew={startNew}
        onDelete={removeSession}
      />

      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center justify-between border-b border-ink-800 px-5 py-2.5">
          <div className="flex items-center gap-3">
            <h2 className="text-[13.5px] text-ink-200">
              {loading
                ? "加载中…"
                : messages.length === 0
                  ? "新的困境"
                  : sessions.find((item) => item.id === sessionId)?.title ?? "对话"}
            </h2>
            {phase && messages.length > 0 && (
              <span className="text-[11px] text-ink-500">阶段：{phase}</span>
            )}
          </div>
          <div className="flex items-center gap-2">
            {health && !health.llm_configured && (
              <span className="rounded-full border border-amber-500/30 bg-amber-500/10 px-2 py-[3px] text-[11px] text-amber-300">
                未配置 LLM_API_KEY
              </span>
            )}
            <Button size="sm" variant="subtle" onClick={() => setRightOpen((value) => !value)}>
              {rightOpen ? "收起分析 →" : "← 展开分析"}
            </Button>
          </div>
        </header>

        <div className="flex min-h-0 flex-1">
          <ChatPanel
            messages={messages}
            liveText={liveText}
            busy={busy}
            phase={phase}
            error={error}
            input={input}
            onInputChange={setInput}
            onSend={() => void send(input)}
            onStop={stop}
            canStop={busy}
            onQuickStart={(text) => void send(text)}
            startDisabled={busy}
          />

          {rightOpen && (
            <aside className="flex w-[368px] shrink-0 flex-col border-l border-ink-800 bg-ink-950/60">
              {(decision || memoryFeed.length > 0 || messages.length === 0) && (
                <div className="max-h-[52%] shrink-0 space-y-3 overflow-y-auto border-b border-ink-800 p-3">
                  {decision && <DecisionCard decision={decision} />}
                  {memoryFeed.length > 0 && (
                    <Panel title="本轮写入的长期记忆" subtitle="可在记忆管理中修改或删除">
                      <div className="space-y-1.5">
                        {memoryFeed.map((change) => (
                          <div
                            key={`${change.memory_id}-${change.content}`}
                            className="text-[12.5px] text-ink-300"
                          >
                            <span className="text-ink-500">
                              [
                              {change.action === "merged"
                                ? "合并"
                                : change.action === "superseded"
                                  ? "更新"
                                  : "新增"}
                              ]
                            </span>{" "}
                            {change.content}
                          </div>
                        ))}
                      </div>
                    </Panel>
                  )}
                  {!decision && memoryFeed.length === 0 && messages.length === 0 && (
                    <Panel title="这会话面板">
                      <p className="text-[12.5px] leading-relaxed text-ink-400">
                        开始对话后，这里会实时显示 Agent 对处境的判断、需要你确认的推测、
                        检索到的理论来源，以及最终的选祸建议。
                      </p>
                    </Panel>
                  )}
                </div>
              )}

              <div className="min-h-0 flex-1">
                <AnalysisPanel
                  analysis={analysis}
                  busy={busy}
                  onResolveConfirmation={resolveConfirmation}
                  onOpenMemory={openMemory}
                />
              </div>
            </aside>
          )}
        </div>
      </main>
    </div>
  );
}
