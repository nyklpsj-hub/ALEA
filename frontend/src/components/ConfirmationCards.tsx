"use client";

import { useState } from "react";
import type { Confirmation } from "@/lib/types";
import { Button, Chip } from "./ui";

const TOPIC_LABEL: Record<string, string> = {
  greed: "贪婪",
  fear: "恐惧",
  constraint: "约束",
  goal: "目标",
  value: "价值观",
  contradiction: "矛盾",
};

const STATUS_LABEL: Record<string, string> = {
  pending: "待确认",
  confirmed: "已确认",
  corrected: "已纠正",
  rejected: "已否认",
};

export function ConfirmationCards({
  confirmations,
  busy,
  onResolve,
}: {
  confirmations: Confirmation[];
  busy: boolean;
  onResolve: (
    id: string,
    action: "confirm" | "correct" | "reject",
    correctedText?: string,
  ) => Promise<void>;
}) {
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");

  const pending = confirmations.filter((item) => item.status === "pending");
  const resolved = confirmations.filter((item) => item.status !== "pending");

  if (!confirmations.length) {
    return (
      <p className="py-2 text-center text-xs text-ink-400">
        还没有需要你确认的判断。Agent 会在有把握的推测出现时请你核对。
      </p>
    );
  }

  return (
    <div className="space-y-2.5">
      {pending.map((item) => (
        <div
          key={item.id}
          className="alea-fade-up rounded-lg border border-ember-600/30 bg-ember-600/[0.07] p-3"
        >
          <div className="mb-1.5 flex items-center gap-2">
            <Chip tone="accent">{TOPIC_LABEL[item.topic] ?? item.topic}</Chip>
            <span className="text-[11px] text-ink-400">Agent 的推测，等你确认</span>
          </div>

          <p className="text-[13.5px] leading-relaxed text-ink-100">{item.claim}</p>

          {item.evidence && (
            <p className="mt-1.5 border-l-2 border-ink-700 pl-2 text-[12px] text-ink-400">
              依据你的原话：{item.evidence}
            </p>
          )}

          {editing === item.id ? (
            <div className="mt-2.5 space-y-2">
              <textarea
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                rows={3}
                placeholder="写出更准确的说法，例如「不是怕没钱，是怕失去能自己决定时间的能力」"
                className="w-full resize-y rounded-lg border border-ink-700 bg-ink-900 px-2.5 py-2 text-[13px] text-ink-100 outline-none placeholder:text-ink-600 focus:border-ember-600"
              />
              <div className="flex gap-2">
                <Button
                  size="sm"
                  variant="primary"
                  disabled={busy || !draft.trim()}
                  onClick={async () => {
                    await onResolve(item.id, "correct", draft.trim());
                    setEditing(null);
                    setDraft("");
                  }}
                >
                  保存我的说法
                </Button>
                <Button size="sm" onClick={() => setEditing(null)}>
                  取消
                </Button>
              </div>
            </div>
          ) : (
            <div className="mt-2.5 flex flex-wrap gap-2">
              <Button
                size="sm"
                variant="primary"
                disabled={busy}
                onClick={() => onResolve(item.id, "confirm")}
              >
                是我
              </Button>
              <Button
                size="sm"
                disabled={busy}
                onClick={() => {
                  setEditing(item.id);
                  setDraft(item.claim);
                }}
              >
                不完全是，我来改
              </Button>
              <Button
                size="sm"
                variant="danger"
                disabled={busy}
                onClick={() => onResolve(item.id, "reject")}
              >
                不是
              </Button>
            </div>
          )}
        </div>
      ))}

      {resolved.length > 0 && (
        <div className="space-y-1.5 pt-1">
          {resolved.map((item) => (
            <div
              key={item.id}
              className="flex items-start gap-2 rounded-lg border border-ink-800 bg-ink-900/40 px-2.5 py-2"
            >
              <Chip
                tone={
                  item.status === "rejected"
                    ? "neutral"
                    : item.status === "corrected"
                      ? "warn"
                      : "good"
                }
              >
                {STATUS_LABEL[item.status] ?? item.status}
              </Chip>
              <div className="min-w-0">
                <p className="text-[12.5px] leading-relaxed text-ink-300">{item.claim}</p>
                {item.resolution_note && (
                  <p className="mt-0.5 text-[11px] text-ink-400">你的更正：{item.resolution_note}</p>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
