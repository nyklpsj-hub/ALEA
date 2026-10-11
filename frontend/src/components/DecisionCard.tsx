"use client";

import type { ReactNode } from "react";

import type { Decision, RiskItem } from "@/lib/types";
import { CitationList } from "./CitationList";
import { Chip } from "./ui";

const LEVEL: Record<string, { label: string; tone: "neutral" | "warn" | "fear" | "greed" }> = {
  low: { label: "低", tone: "neutral" },
  medium: { label: "中", tone: "warn" },
  high: { label: "高", tone: "greed" },
  critical: { label: "极高", tone: "fear" },
};

const REVERSIBILITY: Record<string, string> = {
  reversible: "可逆",
  partially: "部分可逆",
  irreversible: "不可逆",
};

function RiskLine({ risk }: { risk: RiskItem }) {
  const meta = LEVEL[risk.level] ?? LEVEL.medium;
  return (
    <li className="flex gap-2">
      <Chip tone={meta.tone}>{meta.label}</Chip>
      <span className="text-[13px] leading-relaxed text-ink-200">
        {risk.description}
        {risk.mitigation && <span className="text-ink-400">（应对：{risk.mitigation}）</span>}
      </span>
    </li>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="mt-3.5">
      <h4 className="mb-1.5 text-[12px] font-medium tracking-wide text-ink-400">{title}</h4>
      {children}
    </div>
  );
}

export function DecisionCard({ decision }: { decision: Decision }) {
  if (decision.status === "hold_safety") {
    return (
      <div className="rounded-xl border border-red-900/50 bg-red-950/30 p-4">
        <h3 className="text-[15px] font-medium text-red-200">这里先停下</h3>
        <p className="mt-2 text-[13.5px] leading-relaxed text-red-100/90">
          {decision.safety_note || "你描述的情况超出了我适合给出决策建议的范围。"}
        </p>
        <p className="mt-2 text-[12.5px] leading-relaxed text-red-200/70">
          请优先联系现实中能支持你的人或机构：全国 24 小时心理援助热线 12356；紧急情况请拨
          120 / 110。
        </p>
      </div>
    );
  }

  if (decision.status === "defer") {
    return (
      <div className="rounded-xl border border-amber-500/25 bg-amber-500/[0.06] p-4">
        <div className="flex items-center gap-2">
          <Chip tone="warn">暂缓建议</Chip>
        </div>
        <p className="mt-2.5 text-[13.5px] leading-relaxed text-ink-200">
          {decision.defer_reason}
        </p>
        {decision.missing_information.length > 0 && (
          <Section title="还需要知道">
            <ul className="space-y-1">
              {decision.missing_information.map((item) => (
                <li key={item} className="text-[13px] text-ink-200">
                  · {item}
                </li>
              ))}
            </ul>
          </Section>
        )}
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-jade-500/25 bg-jade-500/[0.05] p-4">
      <div className="flex items-center justify-between gap-2">
        <Chip tone="good">选祸建议</Chip>
        <span className="text-[11px] tabular-nums text-ink-400">
          置信度 {decision.confidence.toFixed(2)}
        </span>
      </div>

      <h3 className="mt-2.5 text-[16px] font-medium leading-snug text-ink-100">
        {decision.headline}
      </h3>
      {decision.chosen_label && (
        <p className="mt-1 text-[13px] text-jade-400">建议承担：{decision.chosen_label}</p>
      )}

      {decision.why_this_choice && (
        <p className="mt-2.5 text-[13.5px] leading-relaxed text-ink-200">
          {decision.why_this_choice}
        </p>
      )}

      {decision.reasons.length > 0 && (
        <Section title="为什么是它">
          <ul className="space-y-1">
            {decision.reasons.map((item) => (
              <li key={item} className="text-[13px] leading-relaxed text-ink-200">
                · {item}
              </li>
            ))}
          </ul>
        </Section>
      )}

      {decision.must_give_up.length > 0 && (
        <Section title="选择它，必须放弃">
          <ul className="space-y-1">
            {decision.must_give_up.map((item) => (
              <li key={item} className="text-[13px] leading-relaxed text-ember-400">
                · {item}
              </li>
            ))}
          </ul>
        </Section>
      )}

      {decision.risks.length > 0 && (
        <Section title="风险与应对">
          <ul className="space-y-1.5">
            {decision.risks.map((risk) => (
              <RiskLine key={risk.description} risk={risk} />
            ))}
          </ul>
        </Section>
      )}

      {decision.bottom_lines.length > 0 && (
        <Section title="底线">
          <ul className="space-y-1">
            {decision.bottom_lines.map((item) => (
              <li key={item} className="text-[13px] leading-relaxed text-ink-100">
                · {item}
              </li>
            ))}
          </ul>
        </Section>
      )}

      {decision.options_analysis.length > 0 && (
        <Section title="各选项的代价">
          <div className="space-y-2">
            {decision.options_analysis.map((option) => (
              <div key={option.option_id} className="rounded-lg border border-ink-800 bg-ink-900/50 p-2.5">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-[13px] font-medium text-ink-100">{option.label}</span>
                  <div className="flex items-center gap-1.5">
                    <Chip tone={option.cost_level >= 4 ? "greed" : "neutral"}>
                      代价 {option.cost_level}/5
                    </Chip>
                    <Chip tone="neutral">{REVERSIBILITY[option.reversibility]}</Chip>
                  </div>
                </div>
                {option.satisfies.length > 0 && (
                  <p className="mt-1.5 text-[12px] text-jade-400">满足：{option.satisfies.join("、")}</p>
                )}
                {option.costs.length > 0 && (
                  <p className="mt-0.5 text-[12px] text-ember-400">代价：{option.costs.join("、")}</p>
                )}
              </div>
            ))}
          </div>
        </Section>
      )}

      {decision.reassess_if.length > 0 && (
        <Section title="出现这些情况请重新评估">
          <ul className="space-y-1">
            {decision.reassess_if.map((item) => (
              <li key={item} className="text-[13px] leading-relaxed text-ink-300">
                · {item}
              </li>
            ))}
          </ul>
        </Section>
      )}

      {decision.citations.length > 0 && (
        <Section title="依据的理论来源">
          <CitationList citations={decision.citations} />
        </Section>
      )}
    </div>
  );
}
