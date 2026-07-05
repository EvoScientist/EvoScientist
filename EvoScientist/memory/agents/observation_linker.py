"""Observation-linking background memory agent."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from ..observations import (
    create_link_observations_tool,
    create_read_memory_tool,
    create_search_observations_tool,
)
from ..project import resolve_project_id
from ._factory import (
    build_memory_agent_graph,
    memory_agent_middleware,
    resolve_memory_agent_paths,
)

logger = logging.getLogger(__name__)


def _observation_linker_system_prompt() -> str:
    return (
        "You maintain links between observation memory files.\n\n"
        "Read each newly recorded observation id you are given. Other newly "
        "recorded ids in the same batch are link candidates too. Search and "
        "read observations that may be strongly related. When a "
        "durable relationship exists, call `link_observations` with the "
        "new observation id, the related observation id, and a short "
        "reason. Use relation `complements`, `contradicts`, or `supersedes`. "
        "For bidirectional links, write the reason so it remains true from "
        "either observation's perspective; set `bidirectional=false` when the "
        "explanation is directional. "
        "Link only strong, reusable relationships.\n\n"
        "Do not create new observations. Do not manually edit memory markdown "
        "or frontmatter. Do not edit profile memory. Do not continue the "
        "source task. If the relationship is weak or duplicative, finish "
        "without file changes."
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
    agent_paths = resolve_memory_agent_paths(
        memory_dir=memory_dir,
        workspace_dir=workspace_dir,
    )
    tools = _observation_linker_tools(
        memory_dir=agent_paths.memory_dir,
        workspace_dir=agent_paths.workspace_dir,
    )
    return build_memory_agent_graph(
        name="evomemory-observation-linker",
        system_prompt=_observation_linker_system_prompt(),
        tools=tools,
        memory_dir=agent_paths.memory_dir,
        workspace_dir=agent_paths.workspace_dir,
        middleware=memory_agent_middleware(),
    )
