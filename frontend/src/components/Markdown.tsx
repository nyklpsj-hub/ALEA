"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/**
 * Chat replies are markdown produced by the backend. Rendered as-is (no raw
 * HTML passthrough), so model output cannot inject markup.
 */
export function Markdown({ children }: { children: string }) {
  return (
    <div className="alea-prose">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{children}</ReactMarkdown>
    </div>
  );
}
