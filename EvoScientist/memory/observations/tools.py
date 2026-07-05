"""LangChain tool wrappers for observation memory."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, InjectedToolArg, StructuredTool
from pydantic import BaseModel, Field

from ..types import (
    MemoryScope,
    MemorySourceType,
    MemoryType,
    ObservationRecordResult,
    ObservationRelation,
    ObservationSearchMode,
)
from .relations import link_observation_files
from .store import (
    read_observation_file,
    record_observation_file,
    search_observation_files,
)

logger = logging.getLogger(__name__)
ObservationRecordedHook = Callable[[ObservationRecordResult], None]


class RecordObservationArgs(BaseModel):
    """Model-facing arguments for the `record_observation` tool."""

    memory_type: MemoryType = Field(
        description=(
            "semantic for reusable facts/findings; procedural for reusable "
            "commands, tool constraints, workarounds, or operating recipes; "
            "episodic only for notable one-time session events needed for "
            "future debugging or handoff."
        ),
    )
    summary: str = Field(
        min_length=1,
        description=(
            "One-line summary for the observation index. Include the concrete "
            "pattern, trigger, or outcome a future agent would search for."
        ),
    )
    observation: str = Field(
        min_length=1,
        description=(
            "Concise reusable lesson, fact, or procedure. State the durable "
            "finding and the action or interpretation it implies for future "
            "work."
        ),
    )
    why_it_matters: str = Field(
        min_length=1,
        description=(
            "Explain the future value of the observation: what mistake it "
            "prevents, what decision it accelerates, or what behavior it should "
            "change."
        ),
    )
    evidence: str | None = Field(
        default=None,
        description=(
            "Optional compact support for the observation: source URLs, arXiv "
            "IDs, file paths, exact commands, issue IDs, commit hashes, or run "
            "provenance."
        ),
    )
    scope: MemoryScope = Field(
        description=(
            "global for cross-project findings and general tool/platform "
            "behavior; project only for workspace-specific facts, commands, "
            "or conventions."
        ),
    )
    runtime: Annotated[object | None, InjectedToolArg] = None


class SearchObservationsArgs(BaseModel):
    """Model-facing arguments for the `search_observations` tool."""

    query: str = Field(
        min_length=1,
        description=(
            "Search text. In ranked mode, provide compact natural-language "
            "keywords or short phrases that describe the issue, constraint, "
            "procedure, or prior result to find. In regex mode, provide a "
            "case-insensitive grep-like pattern."
        ),
    )
    mode: ObservationSearchMode = Field(
        default=ObservationSearchMode.RANKED,
        description=(
            "ranked interprets query as keyword text and returns relevance-"
            "ordered observations. regex interprets query as a grep-like "
            "pattern and falls back to literal matching when the pattern is "
            "invalid."
        ),
    )
    scope: MemoryScope | None = Field(
        default=None,
        description=(
            "Optional scope filter. Use project for workspace-local notes, "
            "global for cross-project notes, or omit to search both."
        ),
    )
    memory_type: MemoryType | None = Field(
        default=None,
        description=(
            "Optional type filter: procedural for commands/workarounds, "
            "semantic for reusable facts/findings, episodic for notable events."
        ),
    )
    limit: int = Field(
        default=8,
        ge=1,
        le=20,
        description="Maximum number of matching observations to return.",
    )
    runtime: Annotated[object | None, InjectedToolArg] = None


class ReadMemoryArgs(BaseModel):
    """Model-facing arguments for the `read_memory` tool."""

    observation_id: str = Field(
        min_length=1,
        description=(
            "Exact observation ID to read, such as an ID returned by "
            "`search_observations` or listed in the inlined observation index."
        ),
    )
    runtime: Annotated[object | None, InjectedToolArg] = None


class LinkObservationsArgs(BaseModel):
    """Model-facing arguments for the `link_observations` tool."""

    source_observation_id: str = Field(
        min_length=1,
        description="Exact ID of the newly recorded observation to annotate.",
    )
    target_observation_id: str = Field(
        min_length=1,
        description="Exact ID of the related observation.",
    )
    relation: ObservationRelation = Field(
        default=ObservationRelation.COMPLEMENTS,
        description=(
            "Relationship label. Use `complements` when observations should "
            "be considered together, `contradicts` for incompatible claims, "
            "and `supersedes` when the source should replace the target."
        ),
    )
    reason: str = Field(
        min_length=1,
        max_length=500,
        description=(
            "One concise sentence explaining why future agents should consider "
            "these observations together. For bidirectional links, write a "
            "relationship-level reason that remains true from either "
            "observation's perspective."
        ),
    )
    bidirectional: bool = Field(
        default=True,
        description=(
            "When true, write symmetric relationships to both observations. Use "
            "false when the reason is directional. `supersedes` is directional "
            "and remains source-to-target only."
        ),
    )
    runtime: Annotated[object | None, InjectedToolArg] = None


@dataclass(frozen=True)
class _ObservationContext:
    """Concrete source metadata attached to an observation file."""

    project_id: str
    source_session_id: str
    source_agent: str


def _runtime_config_value(runtime: ToolRuntime | None, key: str) -> str | None:
    """Read one optional string override from runtime configurable config."""
    if runtime is None:
        return None
    config = runtime.config or {}
    if not isinstance(config, Mapping):
        return None
    configurable = config.get("configurable", {})
    if not isinstance(configurable, Mapping):
        return None
    value = configurable.get(key)
    return value if isinstance(value, str) and value else None


def _runtime_project_id(runtime: ToolRuntime | None, default_project_id: str) -> str:
    return _runtime_config_value(runtime, "evomemory_project_id") or default_project_id


def _runtime_session_id(runtime: ToolRuntime | None) -> str | None:
    """Extract the source thread id from tool runtime metadata when present."""
    source_session_id = _runtime_config_value(runtime, "evomemory_source_session_id")
    if source_session_id:
        return source_session_id
    if runtime is not None:
        if runtime.execution_info and runtime.execution_info.thread_id:
            return str(runtime.execution_info.thread_id)
        thread_id = _runtime_config_value(runtime, "thread_id")
        if thread_id:
            return thread_id
    return None


def _resolve_observation_context(
    runtime: ToolRuntime | None,
    *,
    project_id: str,
    source_agent: str,
) -> _ObservationContext | None:
    """Resolve required observation metadata from fixed values and runtime."""
    source_session_id = _runtime_session_id(runtime)
    if source_session_id is None:
        return None
    return _ObservationContext(
        project_id=_runtime_project_id(runtime, project_id),
        source_session_id=source_session_id,
        source_agent=_runtime_config_value(runtime, "evomemory_source_agent")
        or source_agent,
    )


def create_search_observations_tool(
    *,
    memory_dir: str | Path,
    project_id: str,
) -> BaseTool:
    """Build the read-only `search_observations` tool for one project context."""

    def _search_observations(
        query: str,
        mode: ObservationSearchMode = ObservationSearchMode.RANKED,
        scope: MemoryScope | None = None,
        memory_type: MemoryType | None = None,
        limit: int = 8,
        runtime: Annotated[ToolRuntime | None, InjectedToolArg] = None,
    ) -> str:
        search_mode = ObservationSearchMode(mode)
        results = search_observation_files(
            memory_dir=memory_dir,
            project_id=_runtime_project_id(runtime, project_id),
            query=query,
            scope=scope,
            memory_type=memory_type,
            limit=limit,
            mode=search_mode,
        )
        return json.dumps(
            {"results": results},
            ensure_ascii=False,
            sort_keys=True,
        )

    return StructuredTool.from_function(
        func=_search_observations,
        name="search_observations",
        description=(
            "Search EvoMemory observation summaries and bodies with ranked "
            "free-text retrieval. Use a few distinctive words or short phrases "
            "that describe the issue, constraint, procedure, or prior result "
            "to find. For exact grep-like matching, pass `mode=regex`. For "
            "substantial coding, debugging, research, planning, or evaluation "
            "work, use this as the memory preflight before inspecting workspace "
            "files unless the inlined observation index already gives an exact "
            "observation ID to read. Read promising hits with `read_memory`."
        ),
        args_schema=SearchObservationsArgs,
        infer_schema=False,
    )


def create_read_memory_tool(
    *,
    memory_dir: str | Path,
    project_id: str,
) -> BaseTool:
    """Build the read-only `read_memory` tool for one project context."""

    def _read_memory(
        observation_id: str,
        runtime: Annotated[ToolRuntime | None, InjectedToolArg] = None,
    ) -> str:
        requested_id = observation_id.strip()
        result = read_observation_file(
            memory_dir=memory_dir,
            project_id=_runtime_project_id(runtime, project_id),
            observation_id=requested_id,
        )
        if result is None:
            return json.dumps(
                {
                    "error": "No observation with that ID exists in global or current-project memory.",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        payload: dict[str, object] = {"text": result["text"]}
        if "related_observations" in result:
            payload["related_observations"] = result["related_observations"]
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    return StructuredTool.from_function(
        func=_read_memory,
        name="read_memory",
        description=(
            "Read the full markdown for an EvoMemory observation by exact "
            "observation ID. Use this after `search_observations` or the "
            "inlined observation index identifies a promising memory."
        ),
        args_schema=ReadMemoryArgs,
        infer_schema=False,
    )


def create_record_observation_tool(
    *,
    memory_dir: str | Path,
    project_id: str,
    source_type: MemorySourceType,
    source_agent: str,
    on_observation_recorded: ObservationRecordedHook | None = None,
) -> BaseTool:
    """Build the `record_observation` tool for one agent context."""

    def _record_observation(
        memory_type: MemoryType,
        summary: str,
        observation: str,
        why_it_matters: str,
        scope: MemoryScope,
        evidence: str | None = None,
        runtime: Annotated[ToolRuntime | None, InjectedToolArg] = None,
    ) -> str:
        context = _resolve_observation_context(
            runtime,
            project_id=project_id,
            source_agent=source_agent,
        )
        if context is None:
            return json.dumps(
                {
                    "error": "Cannot record observation without a source session id.",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        result = record_observation_file(
            memory_dir=memory_dir,
            project_id=context.project_id,
            memory_type=memory_type,
            summary=summary,
            observation=observation,
            why_it_matters=why_it_matters,
            evidence=evidence,
            scope=scope,
            source_type=source_type,
            source_session_id=context.source_session_id,
            source_agent=context.source_agent,
        )
        if result["created"] and on_observation_recorded is not None:
            try:
                on_observation_recorded(result)
            except Exception:
                logger.warning("Failed to schedule observation linking", exc_info=True)
        return json.dumps(result, ensure_ascii=False, sort_keys=True)

    return StructuredTool.from_function(
        func=_record_observation,
        name="record_observation",
        description=(
            "Record compact reusable memory as a structured EvoMemory "
            "observation markdown file. Use procedural/global for reusable "
            "tool or platform behavior unless it is project-specific."
        ),
        args_schema=RecordObservationArgs,
        infer_schema=False,
    )


def create_link_observations_tool(
    *,
    memory_dir: str | Path,
    project_id: str,
) -> BaseTool:
    """Build the `link_observations` tool for frontmatter-native links."""

    def _link_observations(
        source_observation_id: str,
        target_observation_id: str,
        reason: str,
        relation: ObservationRelation = ObservationRelation.COMPLEMENTS,
        bidirectional: bool = True,
        runtime: Annotated[ToolRuntime | None, InjectedToolArg] = None,
    ) -> str:
        result = link_observation_files(
            memory_dir=memory_dir,
            project_id=_runtime_project_id(runtime, project_id),
            source_observation_id=source_observation_id,
            target_observation_id=target_observation_id,
            reason=reason,
            relation=relation,
            bidirectional=bidirectional,
        )
        return json.dumps(result, ensure_ascii=False, sort_keys=True)

    return StructuredTool.from_function(
        func=_link_observations,
        name="link_observations",
        description=(
            "Add or update a frontmatter `related_observations` link between "
            "two existing EvoMemory observations. Use this only after reading "
            "or searching enough memory to establish a strong durable "
            "relationship; do not use it to create new observations."
        ),
        args_schema=LinkObservationsArgs,
        infer_schema=False,
    )
