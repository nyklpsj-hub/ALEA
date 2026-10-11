"""End-to-end smoke run against the real configured model.

    python -m scripts.smoke_llm                 # replay a scripted 4-turn dilemma
    python -m scripts.smoke_llm --interactive   # type your own

Prints, turn by turn, what the agent decided to do next, what it believes about
the user's greed/fear, which theory chunks it retrieved, and the final advice.
This is the fastest way to confirm the real chain works before opening the UI.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from app.core.compat import configure_event_loop_policy
from app.core.config import get_settings
from app.core.errors import AppError
from app.core.logging import configure_logging
from app.services.container import AppContainer

configure_event_loop_policy()

SCRIPTED_TURNS = [
    "我在一家国企做行政六年了，稳定但没什么成就感。有个朋友邀我一起做摄影工作室，"
    "要我全职投入至少一年，收入不确定。我一直在犹豫。",
    "我最怕的其实不是没钱，是怕回不去原来那种安稳的状态。房贷一个月 8000，不能断。",
    "对，我怕的是失去可以回头的路。我老婆也提到过房贷的事，但她说尊重我的选择。",
    "摄影我做了三年副业，目前一年能接十几单，大概 6 万。如果全职，我觉得能翻倍，但不敢保证。",
    "我还是不知道自己是不是在赌。你觉得我该怎么办？",
]


def _line(char: str = "─", width: int = 78) -> str:
    return char * width


def _render_analysis(analysis: dict) -> str:
    parts: list[str] = []
    if analysis.get("greeds"):
        parts.append("  贪婪：" + "；".join(item["statement"] for item in analysis["greeds"]))
    if analysis.get("fears"):
        parts.append("  恐惧：" + "；".join(item["statement"] for item in analysis["fears"]))
    if analysis.get("constraints"):
        parts.append("  约束：" + "；".join(item["statement"] for item in analysis["constraints"]))
    pending = [c for c in analysis.get("confirmations", []) if c["status"] == "pending"]
    if pending:
        parts.append("  待确认：" + "；".join(c["claim"] for c in pending))
    parts.append(f"  信息充分度：{analysis.get('sufficiency', 0):.2f}")
    return "\n".join(parts)


async def run(turns: list[str], *, interactive: bool) -> int:
    settings = get_settings()
    configure_logging(settings)

    if not settings.llm_configured:
        print("未配置 LLM_API_KEY：请在 backend/.env 中填写后重试。", file=sys.stderr)
        return 2

    container = await AppContainer.create(settings)
    print(f"模型：{settings.llm_model} @ {settings.llm_base_url}")
    print(f"向量：{settings.embedding_provider} · {settings.embedding_model} · {settings.embedding_dim} 维")
    chunks = await container.knowledge_service.count_chunks()
    if chunks == 0:
        print("知识库为空，正在索引…")
        await container.knowledge_service.ingest()
        chunks = await container.knowledge_service.count_chunks()
    print(f"知识库：{chunks} 个片段")
    print(_line())

    user_id = container.user_id
    session = await container.chat_service.create_session(user_id=user_id)
    print(f"会话：{session.id}\n")

    queue = list(turns)
    try:
        while True:
            if interactive:
                try:
                    content = (await asyncio.to_thread(input, "你 > ")).strip()
                except EOFError:
                    break
                if not content or content in {"exit", "quit"}:
                    break
            else:
                if not queue:
                    break
                content = queue.pop(0)
                print(f"你 > {content}\n")

            reply_parts: list[str] = []
            done: dict | None = None
            async for event in container.chat_service.stream_turn(
                user_id=user_id, session_id=session.id, content=content
            ):
                if event["type"] == "node":
                    print(f"  · {event['node']}", end="", flush=True)
                elif event["type"] == "token":
                    reply_parts.append(event["text"])
                elif event["type"] == "done":
                    done = event
            print()

            if done is None:
                print("没有收到结果。", file=sys.stderr)
                return 1

            print(_line())
            print(done["message"].content)
            print(_line("·"))
            print(_render_analysis(done["analysis"]))
            if done["citations"]:
                print("  引用：")
                for citation in done["citations"]:
                    kind = "原文摘录" if citation["verified"] else "整理笔记"
                    print(
                        f"    - {citation['document_title']}"
                        f"（{kind}｜{citation['source_path']}｜相似度 {citation['score']:.3f}）"
                    )
            if done["memory_changes"]:
                print("  写入长期记忆：")
                for change in done["memory_changes"]:
                    print(f"    - [{change['action']}] {change['content']}")
            print()

            if done["decision"]:
                decision = done["decision"]
                print(f"结论状态：{decision['status']}")
                if decision["status"] == "recommend":
                    print("验收：已给出明确的选祸建议（理由 / 代价 / 风险 / 底线齐备）")
                    return 0
                if decision["status"] == "hold_safety":
                    print("验收：检测到严重风险，已停止给出建议并转为求助引导")
                    return 0
                print("结论：暂缓（信息不足或需要用户确认）")

        detail = await container.chat_service.get_detail(user_id=user_id, session_id=session.id)
        print(_line("═"))
        print(f"结束：{detail.turn_count} 轮，阶段 {detail.phase}")
        if detail.decision:
            print(f"最终结论：{detail.decision.status} — {detail.decision.headline}")
        else:
            print("没有产生最终结论（对话在了解阶段结束）。")
        return 0
    except AppError as exc:
        print(f"失败：{exc.message} {exc.details}", file=sys.stderr)
        return 1
    finally:
        await container.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description="Alea end-to-end smoke run")
    parser.add_argument("--interactive", "-i", action="store_true", help="自己输入困境")
    parser.add_argument("--turn", "-t", action="append", default=[], help="追加一轮用户消息（可重复）")
    args = parser.parse_args()
    turns = args.turn or ([] if args.interactive else SCRIPTED_TURNS)
    return asyncio.run(run(list(turns), interactive=args.interactive))


if __name__ == "__main__":
    raise SystemExit(main())
