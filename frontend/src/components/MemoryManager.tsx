"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import * as api from "@/lib/api";
import type {
  DecisionEvent,
  KnowledgeDocument,
  Memory,
  MemoryAudit,
  MemoryKind,
  MemoryStatus,
  SessionDetail,
} from "@/lib/types";
import { CitationList } from "./CitationList";
import { Button, Chip, Empty, Panel, cx } from "./ui";

const KINDS: Array<{ value: MemoryKind; label: string }> = [
  { value: "goal", label: "目标" },
  { value: "value", label: "价值观" },
  { value: "greed", label: "贪婪" },
  { value: "fear", label: "恐惧" },
  { value: "constraint", label: "约束" },
  { value: "decision", label: "历史选择" },
  { value: "preference", label: "偏好" },
  { value: "fact", label: "事实" },
];

const KIND_LABEL = Object.fromEntries(KINDS.map((kind) => [kind.value, kind.label]));

const STATUS: Record<MemoryStatus, { label: string; tone: "neutral" | "good" | "warn" }> = {
  candidate: { label: "待确认", tone: "warn" },
  active: { label: "生效中", tone: "good" },
  rejected: { label: "已否认", tone: "neutral" },
  superseded: { label: "已归档", tone: "neutral" },
};

const PROVENANCE: Record<string, { label: string; tone: "good" | "warn" }> = {
  user_stated: { label: "用户亲述", tone: "good" },
  user_confirmed: { label: "用户确认", tone: "good" },
  agent_inferred: { label: "Agent 推测", tone: "warn" },
};

type Tab = "memories" | "decisions" | "knowledge";

export function MemoryManager() {
  const [tab, setTab] = useState<Tab>("memories");
  const [memories, setMemories] = useState<Memory[]>([]);
  const [decisions, setDecisions] = useState<DecisionEvent[]>([]);
  const [documents, setDocuments] = useState<KnowledgeDocument[]>([]);
  const [stats, setStats] = useState<Record<string, number>>({});
  const [kindFilter, setKindFilter] = useState<MemoryKind | "">("");
  const [statusFilter, setStatusFilter] = useState<MemoryStatus | "">("");
  const [showDeleted, setShowDeleted] = useState(false);
  const [query, setQuery] = useState("");
  const [searchHits, setSearchHits] = useState<Array<{ memory: Memory; score: number }> | null>(null);
  const [retrieval, setRetrieval] = useState<SessionDetail["analysis"]["citations"] | null>(null);
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [history, setHistory] = useState<{ id: string; rows: MemoryAudit[] } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const reload = useCallback(async () => {
    try {
      const [items, statsPayload] = await Promise.all([
        api.listMemories({
          kinds: kindFilter ? [kindFilter] : undefined,
          statuses: statusFilter ? [statusFilter] : undefined,
          includeDeleted: showDeleted,
        }),
        api.getMemoryStats(),
      ]);
      setMemories(items);
      setStats(statsPayload);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "加载记忆失败");
    }
  }, [kindFilter, showDeleted, statusFilter]);

  useEffect(() => {
    void reload();
  }, [reload]);

  useEffect(() => {
    if (tab === "decisions" && decisions.length === 0) {
      api.listDecisionEvents().then(setDecisions).catch(() => undefined);
    }
    if (tab === "knowledge" && documents.length === 0) {
      api.listKnowledgeDocuments().then(setDocuments).catch(() => undefined);
    }
  }, [tab, decisions.length, documents.length]);

  const runSearch = useCallback(async () => {
    if (!query.trim()) {
      setSearchHits(null);
      return;
    }
    setBusy(true);
    try {
      setSearchHits(await api.searchMemories({ query }));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "检索失败");
    } finally {
      setBusy(false);
    }
  }, [query]);

  const saveEdit = useCallback(
    async (id: string) => {
      if (!draft.trim()) return;
      try {
        await api.updateMemory(id, { content: draft.trim() });
        setEditing(null);
        await reload();
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : "保存失败");
      }
    },
    [draft, reload],
  );

  const summary = useMemo(
    () => [
      { label: "生效中", value: stats.active ?? 0 },
      { label: "待确认", value: stats.candidate ?? 0 },
      { label: "已否认", value: stats.rejected ?? 0 },
      { label: "已归档", value: stats.superseded ?? 0 },
    ],
    [stats],
  );

  return (
    <div className="mx-auto flex h-screen max-w-5xl flex-col">
      <header className="flex items-center justify-between border-b border-ink-800 px-5 py-3">
        <div>
          <h1 className="font-serif text-[18px] text-ink-100">长期记忆</h1>
          <p className="mt-0.5 text-[11.5px] text-ink-400">
            这里保存的是你自己的信息，与公共知识库完全隔离。你可以随时修改或删除。
          </p>
        </div>
        <Link href="/" className="text-[12.5px] text-ember-400 hover:text-ember-500">
          ← 回到对话
        </Link>
      </header>

      <div className="flex items-center gap-1 border-b border-ink-800 px-3">
        {(
          [
            ["memories", "记忆条目"],
            ["decisions", "历史决策"],
            ["knowledge", "知识库"],
          ] as Array<[Tab, string]>
        ).map(([value, label]) => (
          <button
            key={value}
            type="button"
            onClick={() => setTab(value)}
            className={cx(
              "-mb-px border-b-2 px-3.5 py-2.5 text-[13px] transition-colors",
              tab === value
                ? "border-ember-500 text-ink-100"
                : "border-transparent text-ink-400 hover:text-ink-200",
            )}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-4">
        {error && (
          <div className="mb-3 rounded-lg border border-red-900/60 bg-red-950/30 px-3.5 py-2 text-[13px] text-red-200">
            {error}
          </div>
        )}

        {tab === "memories" && (
          <div className="space-y-4">
            <div className="grid grid-cols-4 gap-2">
              {summary.map((item) => (
                <div key={item.label} className="rounded-xl border border-ink-800 bg-ink-900/50 px-3 py-2.5">
                  <p className="text-[11px] text-ink-400">{item.label}</p>
                  <p className="mt-0.5 text-[19px] tabular-nums text-ink-100">{item.value}</p>
                </div>
              ))}
            </div>

            <Panel title="按语义检索记忆" subtitle="和 Agent 在新会话里检索的方式相同">
              <div className="flex gap-2">
                <input
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  onKeyDown={(event) => event.key === "Enter" && void runSearch()}
                  placeholder="例如：换城市、父母的期待、怕失败"
                  className="flex-1 rounded-lg border border-ink-700 bg-ink-900 px-3 py-2 text-[13px] text-ink-100 outline-none placeholder:text-ink-600 focus:border-ember-600/60"
                />
                <Button variant="primary" onClick={() => void runSearch()} disabled={busy}>
                  检索
                </Button>
                {searchHits && (
                  <Button
                    onClick={() => {
                      setSearchHits(null);
                      setQuery("");
                    }}
                  >
                    清除
                  </Button>
                )}
              </div>
              {searchHits && (
                <div className="mt-3 space-y-2">
                  {searchHits.length === 0 && <Empty>没有相关记忆。</Empty>}
                  {searchHits.map(({ memory, score }) => (
                    <div
                      key={memory.id}
                      className="rounded-lg border border-ink-800 bg-ink-850/50 px-3 py-2"
                    >
                      <p className="text-[13px] text-ink-100">{memory.content}</p>
                      <div className="mt-1 flex items-center gap-2">
                        <Chip tone="neutral">{KIND_LABEL[memory.kind] ?? memory.kind}</Chip>
                        <span className="text-[10px] tabular-nums text-ink-500">
                          相关度 {score.toFixed(3)}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </Panel>

            <div className="flex flex-wrap items-center gap-2">
              <select
                value={kindFilter}
                onChange={(event) => setKindFilter(event.target.value as MemoryKind | "")}
                className="rounded-lg border border-ink-700 bg-ink-900 px-2.5 py-1.5 text-[12.5px] text-ink-200 outline-none"
              >
                <option value="">全部类别</option>
                {KINDS.map((kind) => (
                  <option key={kind.value} value={kind.value}>
                    {kind.label}
                  </option>
                ))}
              </select>
              <select
                value={statusFilter}
                onChange={(event) => setStatusFilter(event.target.value as MemoryStatus | "")}
                className="rounded-lg border border-ink-700 bg-ink-900 px-2.5 py-1.5 text-[12.5px] text-ink-200 outline-none"
              >
                <option value="">全部状态</option>
                {(Object.keys(STATUS) as MemoryStatus[]).map((status) => (
                  <option key={status} value={status}>
                    {STATUS[status].label}
                  </option>
                ))}
              </select>
              <label className="flex items-center gap-1.5 text-[12px] text-ink-400">
                <input
                  type="checkbox"
                  checked={showDeleted}
                  onChange={(event) => setShowDeleted(event.target.checked)}
                />
                显示已删除
              </label>
              <span className="ml-auto text-[12px] text-ink-500">{memories.length} 条</span>
            </div>

            <div className="space-y-2">
              {memories.length === 0 && <Empty>还没有长期记忆。</Empty>}
              {memories.map((memory) => (
                <div
                  key={memory.id}
                  className={cx(
                    "rounded-xl border bg-ink-900/50 p-3",
                    memory.deleted_at ? "border-ink-800 opacity-60" : "border-ink-800",
                  )}
                >
                  {editing === memory.id ? (
                    <div className="space-y-2">
                      <textarea
                        value={draft}
                        onChange={(event) => setDraft(event.target.value)}
                        rows={3}
                        className="w-full resize-y rounded-lg border border-ink-700 bg-ink-900 px-2.5 py-2 text-[13px] text-ink-100 outline-none focus:border-ember-600/60"
                      />
                      <div className="flex gap-2">
                        <Button size="sm" variant="primary" onClick={() => void saveEdit(memory.id)}>
                          保存
                        </Button>
                        <Button size="sm" onClick={() => setEditing(null)}>
                          取消
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <>
                      <p className="text-[13.5px] leading-relaxed text-ink-100">{memory.content}</p>

                      {memory.evidence && (
                        <p className="mt-1.5 border-l-2 border-ink-700 pl-2 text-[12px] text-ink-400">
                          依据原话：{memory.evidence}
                        </p>
                      )}

                      <div className="mt-2 flex flex-wrap items-center gap-1.5">
                        <Chip tone="neutral">{KIND_LABEL[memory.kind] ?? memory.kind}</Chip>
                        <Chip tone={STATUS[memory.status].tone}>{STATUS[memory.status].label}</Chip>
                        <Chip tone={PROVENANCE[memory.provenance]?.tone ?? "warn"}>
                          {PROVENANCE[memory.provenance]?.label ?? memory.provenance}
                        </Chip>
                        <span className="text-[10px] text-ink-500">重要度 {memory.importance}/5</span>
                        <span className="text-[10px] text-ink-500">
                          置信度 {memory.confidence.toFixed(2)}
                        </span>
                        <span className="text-[10px] text-ink-500">使用 {memory.use_count} 次</span>
                      </div>

                      <div className="mt-2 flex flex-wrap gap-2">
                        <Button
                          size="sm"
                          onClick={() => {
                            setEditing(memory.id);
                            setDraft(memory.content);
                          }}
                        >
                          修改
                        </Button>
                        <Button
                          size="sm"
                          onClick={async () => {
                            try {
                              setHistory({ id: memory.id, rows: await api.getMemoryHistory(memory.id) });
                            } catch (cause) {
                              setError(cause instanceof Error ? cause.message : "读取历史失败");
                            }
                          }}
                        >
                          变更记录
                        </Button>
                        {memory.deleted_at ? (
                          <Button
                            size="sm"
                            onClick={async () => {
                              await api.restoreMemory(memory.id);
                              await reload();
                            }}
                          >
                            恢复
                          </Button>
                        ) : (
                          <Button
                            size="sm"
                            variant="danger"
                            onClick={async () => {
                              await api.deleteMemory(memory.id);
                              await reload();
                            }}
                          >
                            删除
                          </Button>
                        )}
                        {memory.status === "candidate" && (
                          <Button
                            size="sm"
                            variant="primary"
                            onClick={async () => {
                              await api.updateMemory(memory.id, { status: "active" });
                              await reload();
                            }}
                          >
                            确认这条
                          </Button>
                        )}
                      </div>

                      {history?.id === memory.id && (
                        <div className="mt-2.5 space-y-1.5 rounded-lg border border-ink-800 bg-ink-950/60 p-2.5">
                          <p className="text-[11px] text-ink-400">变更记录</p>
                          {history.rows.map((row) => (
                            <div key={row.id} className="text-[11.5px] text-ink-300">
                              <span className="text-ink-500">
                                {new Date(row.created_at).toLocaleString("zh-CN")} · {row.actor} ·{" "}
                                {row.action}
                              </span>
                              {row.reason && <span className="text-ink-400"> — {row.reason}</span>}
                            </div>
                          ))}
                        </div>
                      )}
                    </>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {tab === "decisions" && (
          <div className="space-y-3">
            <p className="text-[12.5px] text-ink-400">
              每次给出选祸建议后，这次决策会被压缩成一条记录，用于未来相似困境的参考。
              {decisions.length === 0 && " 目前还没有记录。"}
            </p>
            {decisions.map((event) => (
              <Panel key={event.id} title={event.title} subtitle={new Date(event.created_at).toLocaleString("zh-CN")}>
                <p className="text-[13px] leading-relaxed text-ink-200">{event.situation}</p>
                {event.chosen_option && (
                  <p className="mt-1.5 text-[12.5px] text-jade-400">最终选择：{event.chosen_option}</p>
                )}
                <div className="mt-2 grid grid-cols-2 gap-2 text-[12px]">
                  <div>
                    <p className="text-ink-500">贪婪</p>
                    <p className="text-ink-300">{event.greed.join("、") || "—"}</p>
                  </div>
                  <div>
                    <p className="text-ink-500">恐惧</p>
                    <p className="text-ink-300">{event.fear.join("、") || "—"}</p>
                  </div>
                  <div>
                    <p className="text-ink-500">代价</p>
                    <p className="text-ink-300">{event.cost.join("、") || "—"}</p>
                  </div>
                  <div>
                    <p className="text-ink-500">风险</p>
                    <p className="text-ink-300">{event.risks.join("、") || "—"}</p>
                  </div>
                </div>
                {event.summary && (
                  <p className="mt-2 text-[12.5px] text-ink-400">小结：{event.summary}</p>
                )}
                <div className="mt-2.5 flex items-center gap-2">
                  <input
                    defaultValue={event.outcome ?? ""}
                    placeholder="后来怎么样了？（记录下来帮助下次判断）"
                    onKeyDown={async (nativeEvent) => {
                      if (nativeEvent.key !== "Enter") return;
                      const value = (nativeEvent.target as HTMLInputElement).value.trim();
                      if (!value) return;
                      const updated = await api.updateDecisionOutcome(event.id, value);
                      setDecisions((previous) =>
                        previous.map((item) => (item.id === updated.id ? updated : item)),
                      );
                    }}
                    className="flex-1 rounded-lg border border-ink-700 bg-ink-900 px-2.5 py-1.5 text-[12.5px] text-ink-100 outline-none focus:border-ember-600/60"
                  />
                  <span className="text-[11px] text-ink-500">回车保存</span>
                </div>
              </Panel>
            ))}
          </div>
        )}

        {tab === "knowledge" && (
          <div className="space-y-4">
            <Panel
              title="公共知识库"
              subtitle="Agent 只能引用这里的内容，且会标注来源类型"
              action={
                <Button
                  size="sm"
                  onClick={async () => {
                    setBusy(true);
                    try {
                      await api.reindexKnowledge(false);
                      setDocuments(await api.listKnowledgeDocuments());
                    } catch (cause) {
                      setError(cause instanceof Error ? cause.message : "重新索引失败");
                    } finally {
                      setBusy(false);
                    }
                  }}
                  disabled={busy}
                >
                  重新索引
                </Button>
              }
            >
              <div className="space-y-2">
                {documents.length === 0 && <Empty>知识库为空，请先运行索引脚本。</Empty>}
                {documents.map((document) => (
                  <div
                    key={document.id}
                    className="flex items-center justify-between gap-3 rounded-lg border border-ink-800 bg-ink-900/40 px-3 py-2"
                  >
                    <div className="min-w-0">
                      <p className="truncate text-[13px] text-ink-100">{document.title}</p>
                      <p className="truncate text-[11px] text-ink-500">{document.source_path}</p>
                    </div>
                    <div className="flex shrink-0 items-center gap-1.5">
                      <Chip tone={document.verified ? "good" : "neutral"}>
                        {document.source_type === "verified_excerpt"
                          ? "原文摘录"
                          : document.source_type === "case"
                            ? "案例"
                            : "整理笔记"}
                      </Chip>
                      <span className="text-[10px] text-ink-500">{document.chunk_count} 片段</span>
                    </div>
                  </div>
                ))}
              </div>
            </Panel>

            <Panel title="检索验证" subtitle="直接查看检索到什么，以及相似度">
              <div className="flex gap-2">
                <input
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  onKeyDown={async (nativeEvent) => {
                    if (nativeEvent.key !== "Enter" || !query.trim()) return;
                    setBusy(true);
                    try {
                      const result = await api.searchKnowledge(query);
                      setRetrieval(result.results);
                    } catch (cause) {
                      setError(cause instanceof Error ? cause.message : "检索失败");
                    } finally {
                      setBusy(false);
                    }
                  }}
                  placeholder="例如：代价 不可逆、贪婪与恐惧"
                  className="flex-1 rounded-lg border border-ink-700 bg-ink-900 px-3 py-2 text-[13px] text-ink-100 outline-none placeholder:text-ink-600 focus:border-ember-600/60"
                />
                <Button
                  variant="primary"
                  disabled={busy || !query.trim()}
                  onClick={async () => {
                    setBusy(true);
                    try {
                      const result = await api.searchKnowledge(query);
                      setRetrieval(result.results);
                    } catch (cause) {
                      setError(cause instanceof Error ? cause.message : "检索失败");
                    } finally {
                      setBusy(false);
                    }
                  }}
                >
                  检索
                </Button>
              </div>
              {retrieval && (
                <div className="mt-3">
                  <CitationList citations={retrieval} emptyText="相似度低于阈值，没有命中任何片段。" />
                </div>
              )}
            </Panel>
          </div>
        )}
      </div>
    </div>
  );
}
