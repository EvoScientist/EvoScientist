"""Background refresh of the deployed main agent's expert registry.

``task`` and ``start_async_task`` resolve ``subagent_type`` against dicts frozen
inside ``create_deep_agent``, so an expert that appears after the agent was built
is unreachable until it is rebuilt. In the CLI/TUI that rebuild happens at a turn
boundary (``cli/_agent_loader.py``). The deployed graph has no turn boundary to
hang it on, and the request path is the wrong place for it twice over: the work
is blocking, which langgraph-dev's blockbuster guard turns into a raised
``BlockingError`` rather than a slow request, and a rebuild takes seconds, which
would stall an unrelated ``GET /threads/{id}/state``.

So the check runs here instead, off-request, and publishes through
``main_graph.refresh_main_graph``. The request path becomes a cached-object
return.

**Why both pushed and polled.** Only ``install_skill`` and ``uninstall_skill``
publish through ``skills_manager._notify_skills_changed``. Every other way an
expert can appear is silent:

- the sandbox shell — ``CompositeBackend.execute`` is not path-routed, and
  ``USER_SKILLS_DIR`` defaults to ``WORKSPACE_ROOT / "skills"``, so
  ``mkdir skills/<name>`` is an ordinary in-workspace command no Python code sees;
- autoskills proposal approval, which copies into ``USER_SKILLS_DIR`` directly;
- ``write_file`` / ``edit_file`` onto the writable workspace tier, which
  ``MergedSkillsBackend`` routes through three independent mutators with no shared
  choke point;
- anything edited by hand.

Subscribing to the push hook alone would therefore catch skill *installs* and miss
the agent authoring an expert, which is the case this feature exists to support.
Polling alone would make every install wait out the interval. Together: installs
take effect as soon as the rebuild finishes, and the silent paths within one poll.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

_logger = logging.getLogger(__name__)

# How long to wait for a push before walking the tree anyway. Only bounds the
# paths that emit no signal — install/uninstall wake the loop immediately. At
# ~18 ms per walk this is well under 1% of one worker thread, and the walk runs
# in a thread so it never touches the event loop.
POLL_INTERVAL_SECONDS = 3.0


def _is_deploy_mode() -> bool:
    """True when this process serves the live main agent.

    ``EvoSci deploy`` sets ``EVOSCIENTIST_DEPLOY_MODE=full``. Under ``EvoSci`` /
    ``EvoSci serve`` it is ``stripped``: the agent lives in the parent CLI
    process and the deployed main graph is dead code, so refreshing it would be
    pure waste. Mirrors the gate in ``manager.py`` and ``EvoScientist.py``.
    """
    return os.environ.get("EVOSCIENTIST_DEPLOY_MODE", "").lower() == "full"


async def _refresh_once(last_token: str | None) -> str | None:
    """Detect a registry change and rebuild if there is one.

    Returns the token to compare against next time. On failure returns
    *last_token* unchanged, so the next pass retries rather than treating the
    failed state as current.
    """
    from ..subagents.expert_container import dispatchable_experts_token

    # Detection only, and the token cannot stamp the cue's memo even if a
    # caller wanted it to. ``refresh_main_graph`` -> ``_get_default_agent``
    # stamps on its success branch, so the cue starts naming the new expert
    # at the moment ``task()`` can route to it — not for the seconds before.
    token = await asyncio.to_thread(dispatchable_experts_token)
    if token == last_token:
        return last_token

    if last_token is not None:
        _logger.info("Expert registry changed; rebuilding the deployed graph.")

    from .main_graph import refresh_main_graph

    await asyncio.to_thread(refresh_main_graph)
    return token


async def _refresh_loop(wake: asyncio.Event) -> None:
    """Watch the expert registry until cancelled."""
    last_token: str | None = None
    while True:
        try:
            last_token = await _refresh_once(last_token)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Never let one bad pass end the watch — a transient failure in
            # the surrounding build (backends against live paths, the chat
            # model) would otherwise disable refresh for the whole process.
            # ``_get_default_agent`` keeps the previous agent seated, so the
            # deployment stays usable; we simply try again next tick.
            _logger.warning(
                "Expert-registry refresh pass failed; keeping the current "
                "graph and retrying.",
                exc_info=True,
            )
        try:
            await asyncio.wait_for(wake.wait(), timeout=POLL_INTERVAL_SECONDS)
        except TimeoutError:
            pass
        wake.clear()


@asynccontextmanager
async def expert_registry_refresher(_app: Any = None) -> AsyncIterator[None]:
    """Lifespan hook that keeps the deployed graph's expert set current.

    Wired as ``Starlette(lifespan=...)`` on the custom app in ``http.py``.
    langgraph-api merges that lifespan with its own; it must be a lifespan
    context manager rather than ``on_startup``, which
    ``langgraph_api.server.validate_router_lifespan_hooks`` rejects outright.
    """
    if not _is_deploy_mode():
        yield
        return

    from ..tools.skills_manager import register_skills_changed_callback

    loop = asyncio.get_running_loop()
    wake = asyncio.Event()

    def _on_skills_changed() -> None:
        # Fires on a default-executor worker thread (the ``skill_manager``
        # tool is a sync ``def``), where there is no running loop — so hop
        # back rather than touching the Event directly.
        loop.call_soon_threadsafe(wake.set)

    # Process-scoped and never unregistered, which is fine: one refresher per
    # process, and the callback only sets an Event.
    register_skills_changed_callback(_on_skills_changed)

    task = asyncio.create_task(
        _refresh_loop(wake), name="evoscientist-expert-registry-refresh"
    )
    _logger.info(
        "Watching the expert registry; polling every %.1fs plus immediate "
        "wake-ups on skill install/uninstall.",
        POLL_INTERVAL_SECONDS,
    )
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
