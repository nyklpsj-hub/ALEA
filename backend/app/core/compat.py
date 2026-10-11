"""Platform compatibility shims.

## Why this exists

LangGraph's PostgreSQL checkpointer is built on psycopg's *async* driver. psycopg
documents that it is not compatible with the default ``ProactorEventLoop`` on
Windows — that loop does not implement ``add_reader``/``add_writer``, which
psycopg relies on, so connections never complete and pool startup times out
after 30 seconds with a bare ``PoolTimeout``.

The fix is to select the ``SelectorEventLoop`` policy *before* any event loop is
created. Calling this after uvicorn (or pytest) has already built its loop has
no effect, which is why the serving entry point is ``scripts/serve.py`` rather
than a bare ``uvicorn`` invocation, and why pytest calls it at import time.
"""

from __future__ import annotations

import asyncio
import sys

_configured = False


def configure_event_loop_policy() -> bool:
    """Select a psycopg-compatible event loop policy. Returns True if applied.

    Safe to call repeatedly and safe to call on platforms that do not need it.
    """
    global _configured
    if _configured:
        return False
    if sys.platform != "win32":
        _configured = True
        return False

    policy = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
    if policy is None:  # pragma: no cover - non-CPython interpreters
        return False

    asyncio.set_event_loop_policy(policy())
    _configured = True
    return True


def is_selector_loop_running() -> bool:
    """True when the running loop can host psycopg connections."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return True  # no loop yet; the policy call will take effect
    name = type(loop).__name__
    return "Selector" in name or sys.platform != "win32"


def loop_factory() -> asyncio.AbstractEventLoop:
    """uvicorn loop factory (``--loop app.core.compat:loop_factory``).

    uvicorn's built-in ``asyncio`` factory deliberately returns
    ``ProactorEventLoop`` on Windows, overriding any policy set beforehand.
    Supplying our own is the only way to make the choice stick for a directly
    run server.

    Note the zero-argument signature: uvicorn passes a custom ``--loop`` target
    straight to ``asyncio.Runner(loop_factory=...)`` without the
    ``use_subprocess`` call it applies to its own factories, so this must
    return a ready loop instance rather than a loop class.
    """
    if sys.platform == "win32":
        return asyncio.SelectorEventLoop()
    return asyncio.new_event_loop()
