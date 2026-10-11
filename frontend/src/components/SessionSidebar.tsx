"use client";

import Link from "next/link";
import type { Health, SessionSummary } from "@/lib/types";
import { Button, Chip, cx } from "./ui";

const PHASE_LABEL: Record<string, string> = {
  understanding: "理解中",
  exploring: "追问中",
  confirming: "待确认",
  knowledge: "检索中",
  deciding: "分析中",
  decided: "已给建议",
  paused: "已暂停",
};

function timeAgo(iso: string): string {
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const diff = Date.now() - then;
  const minutes = Math.floor(diff / 60000);
  if (minutes < 1) return "刚刚";
  if (minutes < 60) return `${minutes} 分钟前`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} 小时前`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days} 天前`;
  return new Date(iso).toLocaleDateString("zh-CN");
}

export function SessionSidebar({
  sessions,
  activeId,
  health,
  onNew,
  onDelete,
}: {
  sessions: SessionSummary[];
  activeId: string | null;
  health: Health | null;
  onNew: () => void;
  onDelete: (id: string) => void;
}) {
  return (
    <aside className="flex h-full w-[248px] shrink-0 flex-col border-r border-ink-800 bg-ink-900/50">
      <div className="border-b border-ink-800 px-3.5 py-3">
        <div className="flex items-baseline gap-2">
          <span className="font-serif text-[17px] tracking-wide text-ink-100">人选祸</span>
          <span className="text-[10px] uppercase tracking-[0.18em] text-ink-500">AlEA</span>
        </div>
        <p className="mt-1 text-[11px] leading-relaxed text-ink-400">
          人选的只有祸，福由天定。
        </p>
        <Button variant="primary" size="sm" className="mt-2.5 w-full" onClick={onNew}>
          ＋ 新的困境
        </Button>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-2 py-2">
        {sessions.length === 0 && (
          <p className="px-2 py-6 text-center text-xs text-ink-500">
            还没有对话。
            <br />
            从一个真实的选择开始。
          </p>
        )}
        <ul className="space-y-1">
          {sessions.map((session) => {
            const active = session.id === activeId;
            return (
              <li key={session.id} className="group relative">
                <Link
                  href={`/s/${session.id}`}
                  className={cx(
                    "block rounded-lg border px-2.5 py-2 transition-colors",
                    active
                      ? "border-ember-600/40 bg-ember-600/10"
                      : "border-transparent hover:border-ink-700 hover:bg-ink-850/70",
                  )}
                >
                  <p className="truncate pr-5 text-[13px] text-ink-100">{session.title}</p>
                  <div className="mt-1 flex items-center gap-1.5">
                    <span className="text-[10px] text-ink-500">{timeAgo(session.last_active_at)}</span>
                    {session.has_decision ? (
                      <Chip tone="good">已给建议</Chip>
                    ) : (
                      <span className="text-[10px] text-ink-500">
                        {PHASE_LABEL[session.phase] ?? session.phase}
                      </span>
                    )}
                  </div>
                </Link>
                <button
                  type="button"
                  title="删除会话"
                  onClick={(event) => {
                    event.preventDefault();
                    onDelete(session.id);
                  }}
                  className="absolute right-1.5 top-1.5 hidden rounded px-1 text-[13px] leading-none text-ink-500 hover:text-red-300 group-hover:block"
                >
                  ×
                </button>
              </li>
            );
          })}
        </ul>
      </div>

      <div className="border-t border-ink-800 px-3.5 py-2.5 text-[11px]">
        <Link href="/memories" className="block text-ink-300 transition-colors hover:text-ember-400">
          ⛁ 长期记忆管理
        </Link>
        <div className="mt-1.5 flex items-center gap-1.5 text-ink-500">
          <span
            className={cx(
              "h-1.5 w-1.5 rounded-full",
              health?.llm_configured && health?.database ? "bg-jade-500" : "bg-amber-ink",
            )}
          />
          {health
            ? health.llm_configured
              ? health.database
                ? `知识库 ${health.knowledge_chunks} 片段`
                : "数据库未连接"
              : "未配置模型密钥"
            : "连接中…"}
        </div>
      </div>
    </aside>
  );
}
