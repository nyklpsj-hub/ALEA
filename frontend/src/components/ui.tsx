"use client";

import type { ReactNode } from "react";

export function cx(...values: Array<string | false | null | undefined>): string {
  return values.filter(Boolean).join(" ");
}

export function Chip({
  children,
  tone = "neutral",
  title,
}: {
  children: ReactNode;
  tone?: "neutral" | "greed" | "fear" | "constraint" | "warn" | "good" | "accent";
  title?: string;
}) {
  const tones: Record<string, string> = {
    neutral: "bg-ink-800 text-ink-300 border-ink-700",
    greed: "bg-ember-600/15 text-ember-400 border-ember-600/30",
    fear: "bg-sky-500/10 text-sky-300 border-sky-500/25",
    constraint: "bg-ink-800 text-ink-200 border-ink-600",
    warn: "bg-amber-500/10 text-amber-300 border-amber-500/30",
    good: "bg-jade-500/15 text-jade-400 border-jade-500/30",
    accent: "bg-ember-500/20 text-ember-400 border-ember-500/40",
  };
  return (
    <span
      title={title}
      className={cx(
        "inline-flex items-center gap-1 rounded-full border px-2 py-[3px] text-[11px] leading-none",
        tones[tone],
      )}
    >
      {children}
    </span>
  );
}

export function Button({
  children,
  onClick,
  variant = "ghost",
  size = "md",
  disabled,
  type = "button",
  title,
  className,
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: "primary" | "ghost" | "danger" | "subtle";
  size?: "sm" | "md";
  disabled?: boolean;
  type?: "button" | "submit";
  title?: string;
  className?: string;
}) {
  const variants: Record<string, string> = {
    primary:
      "bg-ember-500 text-ink-950 hover:bg-ember-400 disabled:bg-ink-700 disabled:text-ink-400 font-medium",
    ghost: "border border-ink-700 text-ink-200 hover:border-ink-600 hover:bg-ink-850",
    danger: "border border-red-900/60 text-red-300 hover:bg-red-950/40",
    subtle: "text-ink-400 hover:text-ink-200",
  };
  return (
    <button
      type={type}
      title={title}
      disabled={disabled}
      onClick={onClick}
      className={cx(
        "rounded-lg transition-colors disabled:cursor-not-allowed",
        size === "sm" ? "px-2.5 py-1 text-xs" : "px-3.5 py-2 text-sm",
        variants[variant],
        className,
      )}
    >
      {children}
    </button>
  );
}

export function Panel({
  title,
  subtitle,
  action,
  children,
  className,
}: {
  title?: string;
  subtitle?: string;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={cx("rounded-xl border border-ink-800 bg-ink-900/60", className)}>
      {(title || action) && (
        <header className="flex items-start justify-between gap-2 border-b border-ink-800 px-3.5 py-2.5">
          <div>
            {title && <h3 className="text-[13px] font-medium text-ink-100">{title}</h3>}
            {subtitle && <p className="mt-0.5 text-[11px] text-ink-400">{subtitle}</p>}
          </div>
          {action}
        </header>
      )}
      <div className="px-3.5 py-3">{children}</div>
    </section>
  );
}

export function Progress({ value, label }: { value: number; label?: string }) {
  const pct = Math.round(Math.min(Math.max(value, 0), 1) * 100);
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-[11px] text-ink-400">
        <span>{label ?? "信息充分度"}</span>
        <span className="tabular-nums">{pct}%</span>
      </div>
      <div className="h-1 w-full overflow-hidden rounded-full bg-ink-800">
        <div
          className={cx(
            "h-full rounded-full transition-all duration-500",
            pct >= 62 ? "bg-jade-500" : pct >= 35 ? "bg-amber-ink" : "bg-ember-600",
          )}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="py-3 text-center text-xs text-ink-400">{children}</p>;
}

export function ThinkingDots() {
  return (
    <span className="inline-flex items-center gap-1">
      {[0, 1, 2].map((index) => (
        <span
          key={index}
          className="alea-thinking-dot h-1.5 w-1.5 rounded-full bg-ember-400"
          style={{ animationDelay: `${index * 0.18}s` }}
        />
      ))}
    </span>
  );
}
