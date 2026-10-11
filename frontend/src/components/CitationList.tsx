"use client";

import { useState } from "react";
import type { Citation } from "@/lib/types";
import { Chip, cx } from "./ui";

const SOURCE_LABEL: Record<string, string> = {
  verified_excerpt: "原文摘录",
  curated_note: "整理笔记",
  case: "案例",
};

export function CitationBadge({ citation }: { citation: Citation }) {
  const isExcerpt = citation.source_type === "verified_excerpt";
  return (
    <Chip tone={isExcerpt ? "good" : "neutral"} title={citation.source_path}>
      {isExcerpt ? "原文" : SOURCE_LABEL[citation.source_type] ?? "资料"}
      {!isExcerpt && citation.verified ? " · 已核对" : ""}
    </Chip>
  );
}

export function CitationCard({ citation }: { citation: Citation }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-lg border border-ink-800 bg-ink-850/60 p-2.5">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="truncate text-[13px] font-medium text-ink-100">{citation.document_title}</p>
          {citation.heading && (
            <p className="mt-0.5 truncate text-[11px] text-ink-400">{citation.heading}</p>
          )}
        </div>
        <CitationBadge citation={citation} />
      </div>

      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="mt-1.5 text-[11px] text-ink-400 hover:text-ink-200"
      >
        {open ? "收起摘录" : "查看摘录"} · {citation.source_path} · 相似度 {citation.score.toFixed(2)}
      </button>

      {open && (
        <p className="mt-2 whitespace-pre-wrap border-l-2 border-ink-700 pl-2.5 text-[12.5px] leading-relaxed text-ink-300">
          {citation.excerpt}
        </p>
      )}
    </div>
  );
}

export function CitationList({
  citations,
  emptyText = "本轮没有引用理论资料。",
}: {
  citations: Citation[];
  emptyText?: string;
}) {
  if (!citations.length) {
    return <p className="py-2 text-center text-xs text-ink-400">{emptyText}</p>;
  }
  return (
    <div className={cx("space-y-2")}>
      {citations.map((citation) => (
        <CitationCard key={citation.chunk_id} citation={citation} />
      ))}
    </div>
  );
}
