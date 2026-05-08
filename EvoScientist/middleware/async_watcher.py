"""Spawn async-task watchers when the agent invokes start/update_async_task.

Hooks via ``awrap_tool_call`` so it only fires on the two launch tools — every
other tool call is a no-op pass-through. The middleware does not register tools
of its own; deepagents' built-in async-subagents middleware already publishes
``start_async_task`` / ``update_async_task`` to the agent.

Stable contract this depends on:

* The two public tool names ``start_async_task`` and ``update_async_task``.
* The ``Command(update={"async_tasks": {task_id: AsyncTask}})`` state schema
  returned by both tools.
* ``runtime.config["configurable"]["thread_id"]`` for capturing the originating
  CLI thread.

It also imports ``_ClientCache`` from deepagents — that is a private symbol but
a stable typed class, used here only as a connection-pool helper keyed by
``(url, headers)``.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

logger = logging.getLogger(__name__)

_LAUNCH_TOOL_NAMES = ("start_async_task", "update_async_task")


class AsyncWatcherMiddleware(AgentMiddleware):
    """Spawn an ``async_notifier`` watcher whenever the agent launches or
    updates an async sub-agent task.

    Args:
        async_agents: Mapping of subagent name → ``AsyncSubAgent`` TypedDict
            (must contain at least ``url`` and ``graph_id``). Used to construct
            a ``_ClientCache`` for resolving the LangGraph client per agent.
    """

    def __init__(self, async_agents: dict[str, Any]) -> None:
        from deepagents.middleware.async_subagents import _ClientCache

        super().__init__()
        self._clients = _ClientCache(async_agents)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        from EvoScientist.cli import async_notifier

        name = request.tool_call.get("name")
        args = request.tool_call.get("args") or {}

        # Pre-cancel the existing watcher BEFORE the new run interrupts the old
        # one. ``update_async_task`` creates a new run on the same thread_id with
        # ``multitask_strategy="interrupt"``, which closes the old run's stream
        # cleanly — without pre-cancellation the old watcher would observe a
        # clean exit and enqueue a stale "success" notification before the new
        # spawn can replace it.
        if name == "update_async_task" and (tid := args.get("task_id")):
            try:
                old = async_notifier._watcher_by_thread.get(tid)
                if old is not None and not old.done():
                    old.cancel()
            except Exception:
                logger.warning(
                    "Pre-cancel of stale watcher for task %s failed; a stale "
                    "success notification may be enqueued",
                    tid,
                    exc_info=True,
                )

        result = await handler(request)

        if name in _LAUNCH_TOOL_NAMES and isinstance(result, Command):
            cli_thread_id = None
            cfg = getattr(getattr(request, "runtime", None), "config", None)
            if isinstance(cfg, dict):
                cli_thread_id = cfg.get("configurable", {}).get("thread_id")

            # Tool-name-gated prompt field — `start_async_task` defines
            # `description`, `update_async_task` defines `message`.
            prompt_field = "description" if name == "start_async_task" else "message"
            prompt = args.get(prompt_field, "")

            tasks_update = (result.update or {}).get("async_tasks") or {}
            for task_id, task in tasks_update.items():
                try:
                    client = self._clients.get_async(task["agent_name"])
                    async_notifier.spawn_watcher(
                        client,
                        task_id,
                        task["run_id"],
                        task["agent_name"],
                        prompt=prompt,
                        origin_cli_thread_id=cli_thread_id,
                    )
                except Exception:
                    logger.warning(
                        "Failed to spawn watcher for task %s",
                        task_id,
                        exc_info=True,
                    )

        return result
