import type { Metadata } from "next";
import type { ReactNode } from "react";
import "./globals.css";

export const metadata: Metadata = {
  title: "人选祸 · Alea",
  description:
    "看清选择背后的贪婪与恐惧，理解每个选择必须承担的代价，得到一个明确、可追溯的选祸建议。",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="zh-CN">
      <body className="min-h-screen bg-ink-950 text-ink-100">{children}</body>
    </html>
  );
}
