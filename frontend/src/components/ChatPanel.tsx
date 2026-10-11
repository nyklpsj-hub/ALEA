"use client";

import { useEffect, useRef, useState } from "react";
import type { Message } from "@/lib/types";
import { MessageBubble } from "./MessageBubble";
import { Button, ThinkingDots, cx } from "./ui";

const EXAMPLES = [
  "我在现在的公司待了六年，稳定但不开心，有个创业机会在等我。",
  "男朋友想让我搬去他的城市，我在这里有工作和家人。",
  "攒了一笔钱，是提前还房贷还是拿去读一个学位。",
];

export function ChatPanel({
  messages,
  liveText,
  busy,
  phase,
  error,
  input,
  onInputChange,
  onSend,
  onStop,
  canStop,
  onQuickStart,
  startDisabled,
}: {
  messages: Array<Pick<Message, "role" | "content" | "node">>;
  liveText: string;
  busy: boolean;
  phase: string;
  error: string | null;
  input: string;
  onInputChange: (value: string) => void;
  onSend: () => void;
  onStop: () => void;
  canStop: boolean;
  onQuickStart: (text: string) => void;
  startDisabled: boolean;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);

  useEffect(() => {
    if (!autoScroll) return;
    const node = scrollRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [messages.length, liveText, autoScroll]);

  useEffect(() => {
    const node = textareaRef.current;
    if (!node) return;
    node.style.height = "auto";
    node.style.height = `${Math.min(node.scrollHeight, 200)}px`;
  }, [input]);

  const empty = messages.length === 0 && !liveText;

  return (
    <div className="flex h-full min-w-0 flex-1 flex-col">
      <div
        ref={scrollRef}
        onScroll={(event) => {
          const node = event.currentTarget;
          setAutoScroll(node.scrollHeight - node.scrollTop - node.clientHeight < 80);
        }}
        className="min-h-0 flex-1 overflow-y-auto px-6 py-6"
      >
        <div className="mx-auto flex max-w-3xl flex-col gap-4">
          {empty && (
            <div className="alea-fade-up mx-auto max-w-xl py-10 text-center">
              <h1 className="font-serif text-[26px] leading-snug text-ink-100">
                说的不是「选什么最好」，
                <br />
                而是「你愿意承担哪个祸」。
              </h1>
              <p className="mt-3 text-[13px] leading-relaxed text-ink-400">
                把你正面临的选择说出来。我会先弄清你真正想要什么、又怕失去什么，
                再和你一起把每个选项的代价摊开，最后给出一个明确、可追溯的建议。
              </p>
              <div className="mt-6 space-y-2 text-left">
                {EXAMPLES.map((example) => (
                  <button
                    key={example}
                    type="button"
                    disabled={startDisabled}
                    onClick={() => onQuickStart(example)}
                    className="w-full rounded-lg border border-ink-800 bg-ink-900/50 px-3.5 py-2.5 text-left text-[13px] text-ink-300 transition-colors hover:border-ember-600/40 hover:text-ink-100 disabled:opacity-50"
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((message, index) => (
            <MessageBubble key={`${message.role}-${index}`} message={message} />
          ))}

          {liveText && (
            <MessageBubble
              message={{ role: "assistant", content: liveText, node: phase }}
              streaming
            />
          )}

          {busy && !liveText && (
            <div className="flex items-center gap-2 px-1 text-[12px] text-ink-400">
              <ThinkingDots />
              <span>正在梳理你的处境…</span>
            </div>
          )}

          {error && (
            <div className="rounded-lg border border-red-900/60 bg-red-950/30 px-3.5 py-2.5 text-[13px] text-red-200">
              {error}
            </div>
          )}
        </div>
      </div>

      <div className="border-t border-ink-800 bg-ink-900/40 px-6 py-4">
        <div className="mx-auto max-w-3xl">
          <div
            className={cx(
              "flex items-end gap-2 rounded-2xl border bg-ink-900 px-3 py-2.5 transition-colors",
              busy ? "border-ink-800" : "border-ink-700 focus-within:border-ember-600/60",
            )}
          >
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(event) => onInputChange(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                  event.preventDefault();
                  onSend();
                }
              }}
              rows={1}
              placeholder="说说你现在的选择困境…（Enter 发送，Shift+Enter 换行）"
              className="max-h-[200px] min-h-[26px] flex-1 resize-none bg-transparent text-[14px] leading-relaxed text-ink-100 outline-none placeholder:text-ink-600"
            />
            {canStop ? (
              <Button variant="ghost" size="sm" onClick={onStop}>
                停止
              </Button>
            ) : (
              <Button
                variant="primary"
                size="sm"
                disabled={busy || !input.trim()}
                onClick={onSend}
              >
                发送
              </Button>
            )}
          </div>
          <p className="mt-2 text-center text-[11px] text-ink-600">
            建议不构成医疗、法律或投资意见。涉及重大不可逆决定时，请同时咨询专业人士。
          </p>
        </div>
      </div>
    </div>
  );
}
