"""Recommended way to run the API.

    python -m scripts.serve                 # 127.0.0.1:8000
    python -m scripts.serve --reload        # development

A bare ``uvicorn app.main:app`` also works on Linux/macOS and inside Docker, but
on Windows it builds a Proactor event loop before the app is imported, and the
LangGraph checkpointer (psycopg async) cannot run there. This entry point
selects the right loop policy first — see ``app/core/compat.py``.
"""

from __future__ import annotations

import argparse
import sys

from app.core.compat import configure_event_loop_policy

# Must happen before uvicorn creates its loop. The import above is deliberate
# and must stay above the uvicorn import in source order for readability; the
# call below is what actually matters.
configure_event_loop_policy()

import uvicorn  # noqa: E402

from app.core.config import get_settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Alea API")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--reload", action="store_true", help="开发模式，代码改动自动重启")
    args = parser.parse_args()

    settings = get_settings()
    host = args.host or settings.api_host
    port = args.port or settings.api_port
    print(f"[alea] http://{host}:{port}  (docs at /docs)")

    # uvicorn's own asyncio factory picks ProactorEventLoop on Windows, which
    # would override the policy set above; our factory always gives a loop that
    # psycopg's async driver can use.
    loop_option = "app.core.compat:loop_factory"

    if args.reload:
        # The reloader runs the server in a child process which re-imports this
        # module, so the policy call above runs there too.
        uvicorn.run(
            "app.main:app",
            host=host,
            port=port,
            reload=True,
            loop=loop_option,
            reload_dirs=[str(settings.knowledge_dir.parent)],
        )
    else:
        from app.main import app

        uvicorn.run(app, host=host, port=port, loop=loop_option)
    return 0


if __name__ == "__main__":
    sys.exit(main())
