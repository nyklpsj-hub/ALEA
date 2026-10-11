import type { Message } from "@/lib/types";
import { Markdown } from "./Markdown";
import { ThinkingDots } from "./ui";

const PHASE_LABEL: Record<string, string> = {
  understanding: "理解处境",
  memory: "调取记忆",
  exploring: "追问",
  confirming: "确认判断",
  knowledge: "检索理论",
  deciding: "形成建议",
  decided: "给出建议",
  paused: "暂停",
};

export function MessageBubble({
  message,
  streaming,
}: {
  message: Pick<Message, "role" | "content" | "node">;
  streaming?: boolean;
}) {
  const isUser = message.role === "user";
  return (
    <div className={`alea-fade-up flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`max-w-[min(46rem,88%)] ${isUser ? "order-2" : ""}`}>
        {!isUser && message.node && (
          <div className="mb-1 pl-1 text-[11px] text-ink-400">
            {PHASE_LABEL[message.node] ?? message.node}
          </div>
        )}
        <div
          className={
            isUser
              ? "rounded-2xl rounded-br-md bg-ember-600/20 px-4 py-2.5 text-[15px] leading-relaxed text-ink-100 ring-1 ring-ember-600/25"
              : "rounded-2xl rounded-bl-md border border-ink-800 bg-ink-900/70 px-4 py-3"
          }
        >
          {isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <>
              <Markdown>{message.content}</Markdown>
              {streaming && (
                <span className="ml-1 inline-block align-middle">
                  <ThinkingDots />
                </span>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
