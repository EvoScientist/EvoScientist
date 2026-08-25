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
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

if TYPE_CHECKING:
    from .notifier import NotifierPort

logger = logging.getLogger(__name__)

_LAUNCH_TOOL_NAMES = ("start_async_task", "update_async_task")


class AsyncWatcherMiddleware(AgentMiddleware):
    """Spawn an ``async_notifier`` watcher whenever the agent launches or
    updates an async sub-agent task.

    Args:
        async_agents: Mapping of subagent name → ``AsyncSubAgent`` TypedDict
            (must contain at least ``url`` and ``graph_id``). Used to construct
            a ``_ClientCache`` for resolving the LangGraph client per agent.
        notifier: Injected :class:`~EvoScientist.middleware.notifier.NotifierPort`
            used to pre-cancel stale watchers and spawn new ones. The composition
            root supplies ``EvoScientist.cli.async_notifier`` in the local CLI
            process, where the notifier's queues have a live drain. ``None`` when
            the main agent is built inside the langgraph dev server process
            (``EVOSCIENTIST_DEPLOY_MODE`` set): there the queues have no consumer
            and completions are delivered client-side by the state-driven reader
            (``async_notifier.enqueue_completions_from_state``), so the watcher
            has nothing to do and every method below no-ops. The middleware
            still stays in the stack so the expert-spec client-cache extension in
            ``_route_async_specs_through_evo_middleware`` keeps its mutation
            target.
    """

    def __init__(
        self, async_agents: dict[str, Any], notifier: NotifierPort | None = None
    ) -> None:
        from deepagents.middleware.async_subagents import _ClientCache

        super().__init__()
        self._clients = _ClientCache(async_agents)
        self._notifier = notifier

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        name = request.tool_call.get("name")
        args = request.tool_call.get("args") or {}

        # No notifier means the main agent is built in the langgraph dev server
        # process, where the notifier's queues have no drain. The state-driven
        # reader delivers completions client-side instead, so the watcher is a
        # pure pass-through: don't pre-cancel and don't spawn.
        if self._notifier is None:
            return await handler(request)

        # Pre-cancel the existing watcher BEFORE the new run interrupts the old
        # one (see NotifierPort.pre_cancel_watcher for the full rationale).
        if name == "update_async_task" and (tid := args.get("task_id")):
            self._notifier.pre_cancel_watcher(tid)

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
                    self._notifier.spawn_watcher(
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
