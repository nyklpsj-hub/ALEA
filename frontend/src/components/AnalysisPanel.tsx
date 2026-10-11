"use client";

import { useState } from "react";
import type { ReactNode } from "react";
import Link from "next/link";
import type { Constraint, MemoryRef, Motive, SessionAnalysis } from "@/lib/types";
import { CitationList } from "./CitationList";
import { ConfirmationCards } from "./ConfirmationCards";
import { Chip, Empty, Panel, Progress, cx } from "./ui";

const PROVENANCE: Record<string, { label: string; tone: "good" | "warn" | "neutral" }> = {
  user_stated: { label: "亲述", tone: "good" },
  user_confirmed: { label: "已确认", tone: "good" },
  agent_inferred: { label: "推测", tone: "warn" },
};

const KIND_LABEL: Record<string, string> = {
  goal: "目标",
  value: "价值观",
  greed: "贪婪",
  fear: "恐惧",
  constraint: "约束",
  decision: "历史选择",
  preference: "偏好",
  fact: "事实",
};

function MotiveRow({ motive, tone }: { motive: Motive; tone: "greed" | "fear" }) {
  const provenance = PROVENANCE[motive.provenance] ?? PROVENANCE.agent_inferred;
  return (
    <div className="rounded-lg border border-ink-800 bg-ink-900/40 px-2.5 py-2">
      <p className="text-[13px] leading-relaxed text-ink-100">{motive.statement}</p>
      <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
        <Chip tone={tone}>{tone === "greed" ? "贪婪" : "恐惧"}</Chip>
        <Chip tone={provenance.tone}>{provenance.label}</Chip>
        {motive.status === "rejected" && <Chip tone="neutral">已否认</Chip>}
        {motive.status === "corrected" && <Chip tone="warn">已纠正</Chip>}
        <span className="text-[10px] text-ink-600">强度 {motive.intensity}/5</span>
      </div>
      {motive.evidence && (
        <p className="mt-1 border-l-2 border-ink-700 pl-2 text-[11.5px] text-ink-400">
          {motive.evidence}
        </p>
      )}
    </div>
  );
}

function ConstraintRow({ constraint }: { constraint: Constraint }) {
  const provenance = PROVENANCE[constraint.provenance] ?? PROVENANCE.agent_inferred;
  return (
    <div className="rounded-lg border border-ink-800 bg-ink-900/40 px-2.5 py-2">
      <p className="text-[13px] leading-relaxed text-ink-100">{constraint.statement}</p>
      <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
        <Chip tone={constraint.hard ? "constraint" : "neutral"}>
          {constraint.hard ? "硬约束" : "软约束"}
        </Chip>
        <Chip tone={provenance.tone}>{provenance.label}</Chip>
      </div>
    </div>
  );
}

function Tab({ active, children, onClick }: { active: boolean; children: ReactNode; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cx(
        "-mb-px border-b-2 px-3 py-2 text-[12.5px] transition-colors",
        active
          ? "border-ember-500 text-ink-100"
          : "border-transparent text-ink-400 hover:text-ink-200",
      )}
    >
      {children}
    </button>
  );
}

export function AnalysisPanel({
  analysis,
  busy,
  onResolveConfirmation,
  onOpenMemory,
}: {
  analysis: SessionAnalysis;
  busy: boolean;
  onResolveConfirmation: (
    id: string,
    action: "confirm" | "correct" | "reject",
    correctedText?: string,
  ) => Promise<void>;
  onOpenMemory: (memory: MemoryRef) => void;
}) {
  const [tab, setTab] = useState<"analysis" | "memory" | "sources">("analysis");
  const pendingCount = analysis.confirmations.filter((item) => item.status === "pending").length;

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-1 border-b border-ink-800 px-2">
        <Tab active={tab === "analysis"} onClick={() => setTab("analysis")}>
          分析
        </Tab>
        <Tab active={tab === "memory"} onClick={() => setTab("memory")}>
          记忆{analysis.recalled_memories.length ? ` (${analysis.recalled_memories.length})` : ""}
        </Tab>
        <Tab active={tab === "sources"} onClick={() => setTab("sources")}>
          引用{analysis.citations.length ? ` (${analysis.citations.length})` : ""}
        </Tab>
      </div>

      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-3">
        {analysis.risk_flags.length > 0 && (
          <div className="rounded-xl border border-red-900/50 bg-red-950/30 p-3">
            <h3 className="text-[13px] font-medium text-red-200">检测到风险信号</h3>
            <p className="mt-1 text-[12.5px] text-red-100/80">
              {analysis.risk_flags.join("、")}——已暂停给出决策建议。
            </p>
          </div>
        )}

        {tab === "analysis" && (
          <>
            <Panel title="当前理解">
              <p className="text-[13px] leading-relaxed text-ink-100">
                {analysis.dilemma.statement || "还没有描述出具体困境。"}
              </p>
              {analysis.dilemma.domain && analysis.dilemma.domain !== "其他" && (
                <div className="mt-1.5">
                  <Chip tone="neutral">{analysis.dilemma.domain}</Chip>
                </div>
              )}
              {analysis.dilemma.options.length > 0 && (
                <div className="mt-2 space-y-1">
                  {analysis.dilemma.options.map((option) => (
                    <div key={option.id} className="text-[12.5px] text-ink-300">
                      · <span className="text-ink-100">{option.label}</span>
                      {option.description && <span className="text-ink-400"> — {option.description}</span>}
                    </div>
                  ))}
                </div>
              )}
              <div className="mt-3">
                <Progress value={analysis.sufficiency} />
                {analysis.sufficiency_rationale && (
                  <p className="mt-1 text-[11px] leading-relaxed text-ink-400">
                    {analysis.sufficiency_rationale}
                  </p>
                )}
              </div>
            </Panel>

            <Panel
              title="需要你确认"
              subtitle={pendingCount ? `${pendingCount} 条待确认` : "目前没有待确认项"}
            >
              <ConfirmationCards
                confirmations={analysis.confirmations}
                busy={busy}
                onResolve={onResolveConfirmation}
              />
            </Panel>

            <Panel title="贪婪：想要得到什么" subtitle={`${analysis.greeds.length} 条`}>
              <div className="space-y-2">
                {analysis.greeds.length ? (
                  analysis.greeds.map((motive) => (
                    <MotiveRow key={motive.id} motive={motive} tone="greed" />
                  ))
                ) : (
                  <Empty>还没有识别出明确的渴望。</Empty>
                )}
              </div>
            </Panel>

            <Panel title="恐惧：害怕失去什么" subtitle={`${analysis.fears.length} 条`}>
              <div className="space-y-2">
                {analysis.fears.length ? (
                  analysis.fears.map((motive) => (
                    <MotiveRow key={motive.id} motive={motive} tone="fear" />
                  ))
                ) : (
                  <Empty>还没有识别出明确的恐惧。</Empty>
                )}
              </div>
            </Panel>

            <Panel title="现实约束" subtitle={`${analysis.constraints.length} 条`}>
              <div className="space-y-2">
                {analysis.constraints.length ? (
                  analysis.constraints.map((constraint) => (
                    <ConstraintRow key={constraint.id} constraint={constraint} />
                  ))
                ) : (
                  <Empty>还没有提到硬性约束。</Empty>
                )}
              </div>
            </Panel>

            {analysis.contradictions.length > 0 && (
              <Panel title="你说法之间的张力" subtitle="Agent 的观察，不是对你的评价">
                <div className="space-y-2">
                  {analysis.contradictions.map((item, index) => (
                    <div key={`${item.description}-${index}`} className="text-[13px] leading-relaxed text-ink-200">
                      · {item.description}
                      {item.between.length > 0 && (
                        <span className="text-ink-400">（{item.between.join(" / ")}）</span>
                      )}
                    </div>
                  ))}
                </div>
              </Panel>
            )}

            {analysis.open_questions.length > 0 && (
              <Panel title="还没弄清的问题">
                <ul className="space-y-1">
                  {analysis.open_questions.map((question) => (
                    <li key={question} className="text-[12.5px] leading-relaxed text-ink-300">
                      · {question}
                    </li>
                  ))}
                </ul>
              </Panel>
            )}
          </>
        )}

        {tab === "memory" && (
          <>
            <Panel
              title="本轮检索到的长期记忆"
              subtitle="只在与当前困境相关时才会被取出"
              action={
                <Link href="/memories" className="text-[11px] text-ember-400 hover:text-ember-500">
                  管理记忆 →
                </Link>
              }
            >
              {analysis.recalled_memories.length ? (
                <div className="space-y-2">
                  {analysis.recalled_memories.map((memory) => (
                    <button
                      key={memory.id}
                      type="button"
                      onClick={() => onOpenMemory(memory)}
                      className="w-full rounded-lg border border-ink-800 bg-ink-900/40 px-2.5 py-2 text-left transition-colors hover:border-ink-600"
                    >
                      <p className="text-[13px] leading-relaxed text-ink-100">{memory.content}</p>
                      <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
                        <Chip tone="neutral">{KIND_LABEL[memory.kind] ?? memory.kind}</Chip>
                        <Chip tone={PROVENANCE[memory.provenance]?.tone ?? "neutral"}>
                          {PROVENANCE[memory.provenance]?.label ?? memory.provenance}
                        </Chip>
                        <span className="text-[10px] tabular-nums text-ink-600">
                          相关度 {memory.score.toFixed(2)}
                        </span>
                      </div>
                    </button>
                  ))}
                </div>
              ) : (
                <Empty>本轮没有检索到相关记忆。</Empty>
              )}
            </Panel>
          </>
        )}

        {tab === "sources" && (
          <Panel title="本轮引用的资料" subtitle="点击可展开原文摘录">
            <CitationList citations={analysis.citations} emptyText="本轮知识库没有检索到相关条目。" />
          </Panel>
        )}
      </div>
    </div>
  );
}
