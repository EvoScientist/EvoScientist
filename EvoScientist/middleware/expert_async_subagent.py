"""Skill-name-injecting AsyncSubAgentMiddleware for expert dispatch.

Upstream ``deepagents.AsyncSubAgentMiddleware`` hardcodes the invocation
input to ``{"messages": [{"role": "user", "content": description}]}`` — no
way for ``start_async_task`` to pass per-run state to the target graph. That
blocks the generic-container async pattern we need for agent-teams' expert
dispatch (one container graph, parameterised by which skill is active via
``skill_name`` in the initial state).

Multiple community issues on the deepagents tracker target this gap
(``#2440``, ``#3838``, ``#4668``, ``#606``, ``#2512``) and the maintainers
have been closing implementation PRs (``#2617``, ``#3839``, ``#4669``) with
process-gate comments, none assigned. Upstream fix is not expected on any
predictable timeline; this subclass gives us the mechanism locally.

Design
------
- Subclass ``AsyncSubAgentMiddleware``; call ``super().__init__()`` for spec
  validation + default 5-tool build, then swap in a start tool that injects
  ``skill_name=subagent_type`` by construction (keeping check / update /
  cancel / list unchanged).
- The tool signature matches upstream exactly: ``(description, subagent_type,
  runtime)``. No LLM-visible ``payload`` field: every value the middleware
  can derive itself (the skill name) is injected inside the middleware, not
  entrusted to a channel the model can get wrong. Any run-specific
  information the model uniquely holds (e.g. the desired ``output_path``)
  belongs in the description string.
- Extend the ``AsyncSubAgent`` typed dict with an optional ``is_expert``
  marker so the middleware knows when to add ``skill_name`` to the run
  input. Standard specs (``writing-agent`` / ``data-analysis-agent`` /
  ``scheduler``) reach ``client.runs.create`` with the upstream shape.

If deepagents ever lands a skill-name-passthrough of its own, delete this
file and rebind ``EvoAsyncSubAgentMiddleware`` → ``AsyncSubAgentMiddleware``
in one commit; the state-schema shape on the container graph doesn't change.

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
    StartAsyncTaskSchema,
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

_logger = logging.getLogger(__name__)


class ExpertAsyncSubAgent(AsyncSubAgent):
    """AsyncSubAgent spec extended with the expert-dispatch marker.

    Same wire fields as upstream ``AsyncSubAgent`` plus an internal
    ``is_expert`` marker. Expert specs get ``skill_name`` injected into
    the run input by construction so the shared container graph knows
    which persona to load; standard specs reach ``runs.create`` with the
    upstream shape.
    """

    is_expert: NotRequired[bool]


def _build_run_input(
    spec: AsyncSubAgent, subagent_type: str, description: str
) -> dict[str, Any]:
    """Build the ``input`` dict for ``client.runs.create``.

    ``skill_name`` is injected by construction for expert specs — never
    accepted from the LLM, because the value is derivable from
    ``subagent_type`` and every LLM-authored field is a field the LLM can
    get wrong (silently overwriting ``messages`` was the pre-fix bug).
    Standard specs (``writing-agent`` / ``data-analysis-agent`` /
    ``scheduler``) reach ``runs.create`` with the upstream single-key shape.
    """
    input_dict: dict[str, Any] = {
        "messages": [{"role": "user", "content": description}]
    }
    if spec.get("is_expert"):
        input_dict["skill_name"] = subagent_type
    return input_dict


def _build_task_envelope(
    subagent_type: str, thread_id: str, run_id: str, tool_call_id: str
) -> Command:
    """Wrap a successful launch in the ``Command`` shape the router expects."""
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    task: AsyncTask = {
        "task_id": thread_id,
        "agent_name": subagent_type,
        "thread_id": thread_id,
        "run_id": run_id,
        "status": "running",
        "created_at": now,
        "last_checked_at": now,
        "last_updated_at": now,
    }
    msg = f"Launched async subagent. task_id: {thread_id}"
    return Command(
        update={
            "messages": [ToolMessage(msg, tool_call_id=tool_call_id)],
            "async_tasks": {thread_id: task},
        }
    )


def _build_expert_start_tool(
    agent_map: dict[str, AsyncSubAgent],
    clients: _ClientCache,
    tool_description: str,
) -> StructuredTool:
    """Build the skill-name-injecting ``start_async_task`` tool.

    Tool signature is upstream's exact shape (``description``,
    ``subagent_type``, ``runtime``). For expert specs the middleware
    injects ``skill_name=subagent_type`` into the run input before
    dispatch, so the container graph resolves the right persona without
    the model contributing (or being able to corrupt) that value.
    """

    def start_async_task(
        description: str,
        subagent_type: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        error = _validate_agent_type(agent_map, subagent_type)
        if error:
            return error
        spec = agent_map[subagent_type]
        input_dict = _build_run_input(spec, subagent_type, description)
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
        return _build_task_envelope(
            subagent_type, thread["thread_id"], run["run_id"], runtime.tool_call_id
        )

    async def astart_async_task(
        description: str,
        subagent_type: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        error = _validate_agent_type(agent_map, subagent_type)
        if error:
            return error
        spec = agent_map[subagent_type]
        input_dict = _build_run_input(spec, subagent_type, description)
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
        return _build_task_envelope(
            subagent_type, thread["thread_id"], run["run_id"], runtime.tool_call_id
        )

    return StructuredTool.from_function(
        name="start_async_task",
        func=start_async_task,
        coroutine=astart_async_task,
        description=tool_description,
        infer_schema=False,
        args_schema=StartAsyncTaskSchema,
    )


class EvoAsyncSubAgentMiddleware(AsyncSubAgentMiddleware):
    """AsyncSubAgentMiddleware with skill-name-injecting ``start_async_task``.

    Composes exactly like upstream — same constructor kwargs, same
    ``system_prompt`` handling, same ``wrap_model_call`` / ``awrap_model_call``,
    same tool signature (``description``, ``subagent_type``, ``runtime``).
    Only difference: for expert specs (``is_expert=True``) the middleware
    injects ``skill_name=subagent_type`` into ``client.runs.create(input=...)``
    so the shared container graph resolves the right persona.

    Existing async subagents (``writing-agent``, ``data-analysis-agent``,
    ``scheduler``) work unchanged — they are declared without ``is_expert``
    and reach ``runs.create`` with the upstream single-key shape.
    """

    def __init__(
        self,
        *,
        async_subagents: list[AsyncSubAgent],
        system_prompt: str | None = None,
    ) -> None:
        # Install the model-passthrough patch BEFORE ``super().__init__(...)``
        # so upstream's ``_build_async_subagent_tools`` sees the patched
        # ``_build_start_tool`` / ``_build_update_tool`` module attributes.
        # Idempotent (guarded by ``_model_passthrough_patched`` in
        # ``llm/patches.py``), so re-invocation on repeated middleware
        # construction is a no-op. Without this, super()'s vanilla tools
        # would still ignore ``cfg.model`` — including ``update_async_task``,
        # which we inherit unchanged below.
        from ..llm.patches import (
            _ClientCacheProxy,
            _patch_deepagents_model_passthrough,
        )

        _patch_deepagents_model_passthrough()

        # Upstream's __init__ validates spec shape, builds the default 5-tool
        # list, and composes the system_prompt. Delegate to it, then swap in
        # the skill-name-injecting start tool. This wastes one tool-build cycle
        # (~microseconds at construction) but avoids duplicating upstream's
        # validation and system-prompt-composition logic. Pass ``system_prompt``
        # through unchanged — deepagents 0.7.0 dropped its ``ASYNC_TASK_SYSTEM_PROMPT``
        # default text; callers that want extra guidance in the async-task
        # section of the prompt now supply it explicitly.
        super().__init__(
            async_subagents=async_subagents,
            system_prompt=system_prompt,
        )
        agent_map: dict[str, AsyncSubAgent] = {a["name"]: a for a in async_subagents}
        # Wrap the client cache in ``_ClientCacheProxy`` so ``client.runs.create``
        # in our replacement start tool (and in the rebuilt check / update /
        # cancel / list tools below) injects ``configurable.model`` /
        # ``configurable.model_provider`` per run. ``_ClientCacheProxy`` exposes
        # the same ``get_sync`` / ``get_async`` surface as ``_ClientCache``, so
        # the upstream tool builders accept it without a type change.
        clients = _ClientCacheProxy(_ClientCache(agent_map))
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
