"""Shared construction helpers for background EvoMemory agents."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepagents.backends.protocol import BackendProtocol
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from ... import paths as _paths

MEMORY_AGENT_RECURSION_LIMIT = 100
MEMORY_MAINTENANCE_EXCLUDED_TOOLS = frozenset(
    {
        "edit_file",
        "execute",
        "task",
        "write_file",
        "write_todos",
    }
)


@dataclass(frozen=True, slots=True)
class MemoryAgentPaths:
    memory_dir: Path
    workspace_dir: Path


def resolve_memory_agent_paths(
    *,
    memory_dir: str | Path | None = None,
    workspace_dir: str | Path | None = None,
) -> MemoryAgentPaths:
    """Resolve the default workspace and memory roots for memory graphs."""
    resolved_memory_dir = Path(
        _paths.MEMORIES_DIR if memory_dir is None else memory_dir
    ).expanduser()
    resolved_workspace_dir = Path(
        _paths.WORKSPACE_ROOT if workspace_dir is None else workspace_dir
    ).expanduser()
    return MemoryAgentPaths(
        memory_dir=resolved_memory_dir,
        workspace_dir=resolved_workspace_dir,
    )


def memory_agent_middleware(
    *extra_middleware: AgentMiddleware,
    excluded_tools: Iterable[str] = MEMORY_MAINTENANCE_EXCLUDED_TOOLS,
) -> list[AgentMiddleware]:
    """Compose the standard middleware stack for unattended memory agents."""
    from deepagents.middleware._tool_exclusion import _ToolExclusionMiddleware

    from ...middleware.tool_error_handler import ToolErrorHandlerMiddleware

    middleware: list[AgentMiddleware] = [
        ToolErrorHandlerMiddleware(),
        *extra_middleware,
    ]
    excluded = frozenset(excluded_tools)
    if excluded:
        middleware.append(_ToolExclusionMiddleware(excluded=excluded))
    return middleware


def build_memory_agent_graph(
    *,
    name: str,
    system_prompt: str,
    memory_dir: str | Path,
    workspace_dir: str | Path,
    tools: Sequence[BaseTool],
    middleware: Sequence[AgentMiddleware],
    recursion_limit: int = MEMORY_AGENT_RECURSION_LIMIT,
    response_format: type[BaseModel] | None = None,
    skills: list[str] | None = None,
    backend: BackendProtocol | None = None,
) -> CompiledStateGraph:
    """Build a background memory graph with the shared model/backend wiring."""
    from deepagents import create_deep_agent

    from ...backends import build_memory_agent_backend
    from ...EvoScientist import _ensure_auxiliary_chat_model

    kwargs: dict[str, Any] = {}
    if response_format is not None:
        kwargs["response_format"] = response_format

    if backend is None:
        backend = build_memory_agent_backend(
            workspace_dir=workspace_dir,
            memory_dir=memory_dir,
        )

    agent = create_deep_agent(
        name=name,
        model=_ensure_auxiliary_chat_model(),
        system_prompt=system_prompt,
        tools=list(tools),
        backend=backend,
        middleware=list(middleware),
        subagents=[],
        skills=skills,
        **kwargs,
    )
    return agent.with_config({"recursion_limit": recursion_limit})
