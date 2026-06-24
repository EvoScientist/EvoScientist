"""Observation-linking background memory agent."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from ... import paths as _paths
from ...config import MemoryControls, get_effective_config
from ..observations import (
    create_link_observations_tool,
    create_read_memory_tool,
    create_search_observations_tool,
)
from ..project import resolve_project_id

logger = logging.getLogger(__name__)

OBSERVATION_LINKER_RECURSION_LIMIT = 100
_OBSERVATION_LINKER_EXCLUDED_TOOLS = frozenset(
    {
        "edit_file",
        "execute",
        "task",
        "write_file",
        "write_todos",
    }
)


def _observation_linker_system_prompt() -> str:
    return (
        "You maintain links between existing observation memory files.\n\n"
        "Read each newly recorded observation id you are given. Search and "
        "read existing observations that may be strongly related. When a "
        "durable relationship exists, call `link_observations` with the "
        "new observation id, the related existing observation id, and a short "
        "reason. Use relation `related`, `contradicts`, or `supersedes`. Link "
        "only strong, reusable relationships.\n\n"
        "Do not create new observations. Do not manually edit memory markdown "
        "or frontmatter. Do not edit profile memory. Do not continue the "
        "source task. If the relationship is weak or duplicative, finish "
        "without file changes."
    )


def _build_observation_linker_backend(
    *,
    workspace_dir: str | Path,
    memory_dir: str | Path,
):
    from deepagents.backends import CompositeBackend, FilesystemBackend

    from ...backends import MemoryFilesystemBackend

    return CompositeBackend(
        default=FilesystemBackend(root_dir=str(workspace_dir), virtual_mode=True),
        routes={
            "/memories/": MemoryFilesystemBackend(
                root_dir=str(memory_dir),
                virtual_mode=True,
            )
        },
    )


def _observation_linker_tools(
    *,
    memory_dir: str | Path,
    workspace_dir: str | Path,
) -> list[BaseTool]:
    project_id = resolve_project_id(workspace_dir)
    return [
        create_search_observations_tool(
            memory_dir=memory_dir,
            project_id=project_id,
        ),
        create_read_memory_tool(
            memory_dir=memory_dir,
            project_id=project_id,
        ),
        create_link_observations_tool(
            memory_dir=memory_dir,
            project_id=project_id,
        ),
    ]


def build_observation_linker_graph(
    *,
    memory_dir: str | Path | None = None,
    workspace_dir: str | Path | None = None,
) -> CompiledStateGraph:
    """Build the registered LangGraph observation linker."""
    from deepagents.middleware._tool_exclusion import _ToolExclusionMiddleware

    memory_controls = MemoryControls.from_config(get_effective_config())
    worker_memory_dir = Path(
        _paths.MEMORIES_DIR if memory_dir is None else memory_dir
    ).expanduser()
    worker_workspace_dir = Path(
        _paths.WORKSPACE_ROOT if workspace_dir is None else workspace_dir
    ).expanduser()
    middleware: list[AgentMiddleware] = [
        _ToolExclusionMiddleware(excluded=_OBSERVATION_LINKER_EXCLUDED_TOOLS)
    ]
    tools: list[BaseTool] = []
    if memory_controls.observations_enabled:
        observation_tools = _observation_linker_tools(
            memory_dir=worker_memory_dir,
            workspace_dir=worker_workspace_dir,
        )
        tools.extend(observation_tools)

    from deepagents import create_deep_agent

    from ...EvoScientist import _ensure_auxiliary_chat_model

    agent = create_deep_agent(
        name="evomemory-observation-linker",
        model=_ensure_auxiliary_chat_model(),
        system_prompt=_observation_linker_system_prompt(),
        tools=tools,
        backend=_build_observation_linker_backend(
            workspace_dir=worker_workspace_dir,
            memory_dir=worker_memory_dir,
        ),
        middleware=middleware,
        subagents=[],
    )
    return agent.with_config({"recursion_limit": OBSERVATION_LINKER_RECURSION_LIMIT})
