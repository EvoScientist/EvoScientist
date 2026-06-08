"""Factory for building deployable sub-agent graphs from yaml definitions.

Lives in ``EvoScientist/subagents/`` next to the canonical yaml entries
because the factory is "build a graph from a sub-agent name" — a generic
construction utility, not a deployment concern. Any deployment surface
(``EvoScientist/langgraph_dev/``, future ``langgraph_platform/``, custom
servers) can call ``build_async_subagent_graph(name)`` to materialize the
runnable graph.

Reuses the main EvoScientist agent's chat model, backend, and middleware so
the deployed sub-agent has full capability parity with its in-process
synchronous counterpart: same workspace files, same ``/skills/`` and
``/memories/`` routes, same error-handling and context-overflow middleware.
"""

from __future__ import annotations

import os
from typing import Any


def build_async_subagent_graph(name: str) -> Any:
    """Build a deployable graph for the ``name`` sub-agent defined in yaml.

    Args:
        name: The sub-agent's key in one of the ``EvoScientist/subagents/*.yaml``
            files (e.g. ``"writing-agent"``).

    Returns:
        A compiled ``langgraph`` graph ready for registration in ``langgraph.json``.

    Raises:
        ValueError: If ``name`` is not defined under ``EvoScientist/subagents/``.
    """
    # Lazy imports — the factory is invoked at langgraph dev startup time, so
    # all heavy modules (deepagents, llm, MCP) are pulled in here rather than
    # at package import.
    from deepagents import create_deep_agent

    from EvoScientist.config import apply_config_to_env, get_effective_config
    from EvoScientist.EvoScientist import (
        SUBAGENTS_CONFIG,
        _ensure_chat_model,
        _ensure_general_purpose_subagent,
        _get_default_backend,
        _get_default_middleware,
        _inject_subagent_middleware,
    )
    from EvoScientist.tools import tavily_search, think_tool
    from EvoScientist.utils import load_subagents

    # Surface API keys as env vars so downstream SDKs (openai, anthropic, …)
    # find them on subprocess invocations from langgraph dev.
    cfg = get_effective_config()
    apply_config_to_env(cfg)

    # Mirror the tool registry constructed in EvoScientist._build_base_kwargs.
    tool_registry = {"think_tool": think_tool}
    if os.environ.get("TAVILY_API_KEY"):
        tool_registry["tavily_search"] = tavily_search

    # Use the official loader so resolved tools, prompt_refs, and skills are
    # all wired the same way as the in-process sync version.
    specs = load_subagents(
        SUBAGENTS_CONFIG,
        tool_registry=tool_registry,
    )
    spec = next((s for s in specs if s.get("name") == name), None)
    if spec is None:
        raise ValueError(
            f"Sub-agent {name!r} not found in {SUBAGENTS_CONFIG}. "
            f"Available: {[s.get('name') for s in specs]}"
        )

    # Load MCP tools routed to THIS agent via ``expose_to: <name>`` in
    # ``mcp.yaml``. Use the cached helper so multiple ``build_async_subagent_graph``
    # calls in the same langgraph dev subprocess (one per registered async graph)
    # share a single MCP connection set per server instead of re-spawning.
    from EvoScientist.EvoScientist import _load_mcp_tools_cached

    mcp_tools_by_agent = _load_mcp_tools_cached()
    agent_mcp_tools = mcp_tools_by_agent.get(name, [])

    # NOTE on HITL: async sub-agents intentionally do NOT set ``interrupt_on``,
    # even though the deployed main agent does. They run as standalone graphs
    # on the langgraph dev subprocess; the parent (CLI main agent) only sees a
    # ``task_id`` from ``start_async_task`` and has no UI path to surface a
    # paused-on-interrupt child to the user. Setting ``interrupt_on`` here
    # would hang the sub-agent on its first ``execute`` call with no one to
    # approve. The user-visible HITL boundary is the parent's
    # ``start_async_task`` decision; restrict the child's reach by limiting
    # ``tools`` in ``subagents/<name>.yaml`` instead.
    #
    # ``for_async_subagent=True`` propagates the same reasoning to the
    # middleware list — specifically, it suppresses ``AskUserMiddleware``,
    # which uses ``interrupt()`` for the same purpose (waiting on a user
    # reply) and would deadlock an async sub-agent for the same reason.
    #
    # Memory middleware is included so async sub-agents get the same profile
    # context and `/memories/profile/...` file guidance as the main agent.
    subagents = []
    _ensure_general_purpose_subagent(subagents)
    _inject_subagent_middleware(subagents)

    return create_deep_agent(
        name=name,
        model=_ensure_chat_model(),
        system_prompt=spec.get("system_prompt", ""),
        tools=spec.get("tools", []) + agent_mcp_tools,
        skills=spec.get("skills"),
        backend=_get_default_backend(),
        middleware=_get_default_middleware(
            for_async_subagent=True,
            memory_source_agent=name,
        ),
        subagents=subagents,
    ).with_config({"recursion_limit": cfg.recursion_limit})
