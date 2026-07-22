"""Payload-aware AsyncSubAgentMiddleware for the expert-skill dispatch pattern.

Upstream ``deepagents.AsyncSubAgentMiddleware`` hardcodes the invocation
input to ``{"messages": [{"role": "user", "content": description}]}`` — no
way for ``start_async_task`` to pass per-run state to the target graph. That
blocks the generic-container async pattern we need for agent-teams v2's
expert dispatch (one container graph, parameterised by which skill is active
via ``skill_name`` + ``output_path`` in the initial state).

Multiple community issues on the deepagents tracker target this gap
(``#2440``, ``#3838``, ``#4668``, ``#606``, ``#2512``) and the maintainers
have been closing implementation PRs (``#2617``, ``#3839``, ``#4669``) with
process-gate comments, none assigned. Upstream fix is not expected on any
predictable timeline; this subclass gives us the mechanism locally.

Design
------
- Subclass ``AsyncSubAgentMiddleware``; call ``super().__init__()`` for spec
  validation + default 5-tool build, then swap in a payload-aware start tool
  in place of upstream's (keeping check / update / cancel / list unchanged).
- Extend the ``AsyncSubAgent`` typed dict with an optional ``is_expert``
  marker. Expert specs require a non-empty payload; standard specs (existing
  ``writing-agent`` / ``data-analysis-agent`` / ``scheduler``) reject a
  non-None payload with a clear tool-result error.
- Payload keys are merged into ``client.runs.create(input=...)`` so the
  target graph's state schema can declare them as regular fields
  (``skill_name``, ``output_path``, ``started_at``, ...).

If deepagents ever lands a payload-passthrough of its own, delete this file
and rebind ``EvoAsyncSubAgentMiddleware`` → ``AsyncSubAgentMiddleware`` in
one commit; the state-schema shape on the container graph doesn't change.

Do NOT add ``from __future__ import annotations`` to this module. langchain's
``StructuredTool._injected_args_keys`` uses ``inspect.signature(fn)`` (raw
annotations, not ``get_type_hints``) to decide which parameters are injected
runtime args. With PEP 563 in effect ``runtime: ToolRuntime`` becomes the
string ``"ToolRuntime"``, fails the ``issubclass(type_, _DirectlyInjectedToolArg)``
check, and gets stripped from tool_input at parse time — the coroutine is
then called without ``runtime`` and raises ``TypeError``.
"""

import logging
from datetime import UTC, datetime
from typing import Any, NotRequired

from deepagents.middleware.async_subagents import (
    ASYNC_TASK_TOOL_DESCRIPTION,
    AsyncSubAgent,
    AsyncSubAgentMiddleware,
    AsyncTask,
    _build_cancel_tool,
    _build_check_tool,
    _build_list_tasks_tool,
    _build_update_tool,
    _ClientCache,
    _validate_agent_type,
)
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.types import Command
from pydantic import BaseModel, Field

_logger = logging.getLogger(__name__)


class ExpertAsyncSubAgent(AsyncSubAgent):
    """AsyncSubAgent spec extended with the expert-dispatch marker.

    Same wire fields as upstream ``AsyncSubAgent`` plus an internal
    ``is_expert`` marker. Expert specs are dispatched with a required
    payload dict (containing ``skill_name``, ``output_path``, ...); standard
    specs reject payload.
    """

    is_expert: NotRequired[bool]


class ExpertStartAsyncTaskSchema(BaseModel):
    """Input schema for the payload-aware ``start_async_task``.

    Extends upstream's ``StartAsyncTaskSchema`` with an optional ``payload``
    dict. Presence / absence rules depend on the target subagent's
    ``is_expert`` marker — enforced at tool-call time.
    """

    description: str = Field(
        description="A detailed description of the task for the async subagent to perform."
    )
    subagent_type: str = Field(
        description=(
            "The type of async subagent to use. Must be one of the available "
            "types listed in the tool description."
        )
    )
    payload: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Per-invocation state for expert subagents. Required when "
            "``subagent_type`` refers to an expert (e.g. ``literature-review``); "
            "must include ``skill_name`` and ``output_path``. Not accepted "
            "for standard subagents (e.g. ``writing-agent``)."
        ),
    )


def _payload_validation_error(
    spec: AsyncSubAgent,
    subagent_type: str,
    payload: dict[str, Any] | None,
) -> str | None:
    """Return an error string when payload presence doesn't match the spec.

    Rules:
    - Expert spec (``is_expert=True``): payload MUST be a non-empty dict
      containing at minimum ``skill_name``. ``output_path`` is checked at
      the container graph — not required at the tool-call boundary — because
      some experts may compute the path themselves from ``skill_name`` + user
      goal in a preflight step.
    - Standard spec: payload MUST be ``None`` or absent. A non-None payload
      indicates the caller confused the two subagent shapes; reject rather
      than silently drop the payload.
    """
    is_expert = bool(spec.get("is_expert", False))
    if is_expert:
        if not payload:
            return (
                f"payload required for expert subagent '{subagent_type}': "
                "must be a dict including at least `skill_name`"
            )
        if not isinstance(payload, dict):
            return (
                f"payload for expert subagent '{subagent_type}' must be a dict, "
                f"got {type(payload).__name__}"
            )
        if not payload.get("skill_name"):
            return (
                f"payload for expert subagent '{subagent_type}' must include "
                "non-empty `skill_name`"
            )
        return None
    # Standard subagent — payload must be absent.
    if payload:
        return (
            f"payload not accepted for standard subagent '{subagent_type}'; "
            "payload is only used by expert subagents"
        )
    return None


def _build_expert_start_tool(
    agent_map: dict[str, AsyncSubAgent],
    clients: _ClientCache,
    tool_description: str,
) -> StructuredTool:
    """Build the payload-aware ``start_async_task`` tool.

    Mirrors ``deepagents.middleware.async_subagents._build_start_tool`` line
    for line, with two additions: (1) accepts ``payload: dict | None``,
    (2) merges it into the ``client.runs.create(input=...)`` dict so the
    target graph receives it as initial state.
    """

    def start_async_task(
        description: str,
        subagent_type: str,
        payload: dict[str, Any] | None,
        runtime: ToolRuntime,
    ) -> str | Command:
        error = _validate_agent_type(agent_map, subagent_type)
        if error:
            return error
        spec = agent_map[subagent_type]
        payload_error = _payload_validation_error(spec, subagent_type, payload)
        if payload_error:
            return payload_error

        input_dict: dict[str, Any] = {
            "messages": [{"role": "user", "content": description}]
        }
        if payload:
            input_dict.update(payload)

        try:
            client = clients.get_sync(subagent_type)
            thread = client.threads.create()
            run = client.runs.create(
                thread_id=thread["thread_id"],
                assistant_id=spec["graph_id"],
                input=input_dict,
            )
        except Exception as e:
            _logger.warning(
                "Failed to launch async subagent '%s': %s", subagent_type, e
            )
            return f"Failed to launch async subagent '{subagent_type}': {e}"

        task_id = thread["thread_id"]
        now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        task: AsyncTask = {
            "task_id": task_id,
            "agent_name": subagent_type,
            "thread_id": task_id,
            "run_id": run["run_id"],
            "status": "running",
            "created_at": now,
            "last_checked_at": now,
            "last_updated_at": now,
        }
        msg = f"Launched async subagent. task_id: {task_id}"
        return Command(
            update={
                "messages": [ToolMessage(msg, tool_call_id=runtime.tool_call_id)],
                "async_tasks": {task_id: task},
            }
        )

    async def astart_async_task(
        description: str,
        subagent_type: str,
        payload: dict[str, Any] | None,
        runtime: ToolRuntime,
    ) -> str | Command:
        error = _validate_agent_type(agent_map, subagent_type)
        if error:
            return error
        spec = agent_map[subagent_type]
        payload_error = _payload_validation_error(spec, subagent_type, payload)
        if payload_error:
            return payload_error

        input_dict: dict[str, Any] = {
            "messages": [{"role": "user", "content": description}]
        }
        if payload:
            input_dict.update(payload)

        try:
            client = clients.get_async(subagent_type)
            thread = await client.threads.create()
            run = await client.runs.create(
                thread_id=thread["thread_id"],
                assistant_id=spec["graph_id"],
                input=input_dict,
            )
        except Exception as e:
            _logger.warning(
                "Failed to launch async subagent '%s': %s", subagent_type, e
            )
            return f"Failed to launch async subagent '{subagent_type}': {e}"

        task_id = thread["thread_id"]
        now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        task: AsyncTask = {
            "task_id": task_id,
            "agent_name": subagent_type,
            "thread_id": task_id,
            "run_id": run["run_id"],
            "status": "running",
            "created_at": now,
            "last_checked_at": now,
            "last_updated_at": now,
        }
        msg = f"Launched async subagent. task_id: {task_id}"
        return Command(
            update={
                "messages": [ToolMessage(msg, tool_call_id=runtime.tool_call_id)],
                "async_tasks": {task_id: task},
            }
        )

    return StructuredTool.from_function(
        name="start_async_task",
        func=start_async_task,
        coroutine=astart_async_task,
        description=tool_description,
        infer_schema=False,
        args_schema=ExpertStartAsyncTaskSchema,
    )


class EvoAsyncSubAgentMiddleware(AsyncSubAgentMiddleware):
    """AsyncSubAgentMiddleware with payload-aware ``start_async_task``.

    Composes exactly like upstream — same constructor kwargs, same
    ``system_prompt`` handling, same ``wrap_model_call`` / ``awrap_model_call``.
    Only difference: the ``start_async_task`` tool accepts an optional
    ``payload: dict | None`` that is forwarded into the target graph's
    initial state, enabling the expert-container dispatch pattern.

    Existing async subagents (``writing-agent``, ``data-analysis-agent``,
    ``scheduler``) work unchanged — they are declared without ``is_expert``
    (or with ``is_expert=False``), and callers that pass no payload get
    upstream's exact behaviour.
    """

    def __init__(
        self,
        *,
        async_subagents: list[AsyncSubAgent],
        system_prompt: str | None = None,
    ) -> None:
        # Upstream's __init__ validates spec shape, builds the default 5-tool
        # list, and composes the system_prompt. Delegate to it, then swap in
        # the payload-aware start tool. This wastes one tool-build cycle
        # (~microseconds at construction) but avoids duplicating upstream's
        # validation and system-prompt-composition logic.
        from deepagents.middleware.async_subagents import ASYNC_TASK_SYSTEM_PROMPT

        super().__init__(
            async_subagents=async_subagents,
            system_prompt=system_prompt
            if system_prompt is not None
            else ASYNC_TASK_SYSTEM_PROMPT,
        )
        agent_map: dict[str, AsyncSubAgent] = {a["name"]: a for a in async_subagents}
        clients = _ClientCache(agent_map)
        agents_desc = "\n".join(
            f"- {a['name']}: {a['description']}" for a in async_subagents
        )
        launch_desc = ASYNC_TASK_TOOL_DESCRIPTION.format(available_agents=agents_desc)
        self.tools = [
            _build_expert_start_tool(agent_map, clients, launch_desc),
            _build_check_tool(clients),
            _build_update_tool(agent_map, clients),
            _build_cancel_tool(clients),
            _build_list_tasks_tool(clients),
        ]
