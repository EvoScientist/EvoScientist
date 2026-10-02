"""Windows asyncio event-loop policy compatibility.

On Windows, MCP stdio servers are launched as subprocesses by the MCP SDK's
stdio transport, which uses ``anyio.open_process`` → ``asyncio`` async
subprocess support. ``asyncio``'s *Selector* event loop does **not** implement
async subprocess creation, so on a Selector loop the stdio transport falls back
to a synchronous ``subprocess.Popen`` inside an ``async`` function. Under
``langgraph dev`` (which enables ``blockbuster`` by default to police blocking
I/O) that synchronous call is flagged as a ``BlockingError`` — see issue #283.

The *Proactor* loop supports async subprocesses natively, so the fallback never
happens and ``blockbuster`` allows the (now genuinely async) spawn.

Windows has defaulted to ``WindowsProactorEventLoopPolicy`` since Python 3.8, so
this is normally a no-op. We set it explicitly anyway as a safeguard: a
dependency, IDE, or notebook host may have installed a Selector policy earlier
in the process, and the ``langgraph dev`` subprocess in particular runs code we
don't fully control. Calling this at each process entrypoint — **before any
event loop is created** — guarantees the MCP subprocess path stays async.

This must run at import/startup time, ahead of the first ``asyncio.run`` /
``new_event_loop`` call; once a loop exists, swapping the policy does not change
the already-running loop.
"""

from __future__ import annotations

import sys


def ensure_proactor_event_loop_policy() -> bool:
    """Install ``WindowsProactorEventLoopPolicy`` on Windows if needed.

    Returns ``True`` if a Proactor policy is in effect afterwards (always
    ``False`` off Windows, where the concept doesn't apply). Safe and idempotent
    to call multiple times; a no-op on non-Windows platforms.
    """
    if sys.platform != "win32":
        return False

    import asyncio

    proactor_policy = getattr(asyncio, "WindowsProactorEventLoopPolicy", None)
    if proactor_policy is None:  # pragma: no cover - non-Windows / stripped build
        return False

    current = asyncio.get_event_loop_policy()
    if not isinstance(current, proactor_policy):
        asyncio.set_event_loop_policy(proactor_policy())
    return True
