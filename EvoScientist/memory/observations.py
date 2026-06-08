"""File-backed observation memory.

Observations are small markdown files under `/memories/observations/`. Each
file has stable frontmatter for future indexing plus a short body that agents
can grep and read with ordinary file tools today.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, NotRequired, TypedDict

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, InjectedToolArg, StructuredTool
from pydantic import BaseModel, ConfigDict, Field

OBSERVATION_DIR = "/observations"


class MemoryType(StrEnum):
    """Kinds of reusable memory an observation can represent."""

    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    EPISODIC = "episodic"


class MemoryScope(StrEnum):
    """Whether an observation is global or tied to the active project."""

    GLOBAL = "global"
    PROJECT = "project"


class MemorySourceType(StrEnum):
    """Where an observation came from in the agent lifecycle."""

    SUBAGENT = "subagent"
    TURN = "turn"


class ObservationRecordResult(TypedDict):
    """Result returned by `record_observation`."""

    observation_id: str
    path: str
    created: bool
    memory_type: MemoryType
    scope: MemoryScope
    project_id: NotRequired[str]


class RecordObservationArgs(BaseModel):
    """Model-facing arguments for the `record_observation` tool."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

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
            "One-line agent-generated summary used in the observation index. "
            "Make it specific enough to decide whether to read the full file."
        ),
    )
    observation: str = Field(
        min_length=1,
        description=(
            "Concise reusable memory. Do not include raw traces, long citation "
            "dumps, or claims that are not supported by the trajectory."
        ),
    )
    why_it_matters: str = Field(
        min_length=1,
        description=(
            "Why this will matter in future work, including compact evidence "
            "or provenance when relevant."
        ),
    )
    evidence: str | None = Field(
        default=None,
        description=(
            "Optional compact support such as source URLs, arXiv IDs, artifact "
            "paths, exact commands, or 'observed in this run'. Use this for "
            "bibliographic, benchmark, or date-sensitive claims."
        ),
    )
    scope: MemoryScope = Field(
        description=(
            "global for cross-project findings and general tool/platform "
            "behavior; project only for workspace-specific facts, commands, "
            "or conventions."
        ),
    )
    runtime: Annotated[ToolRuntime | None, InjectedToolArg] = None


@dataclass(frozen=True)
class _ObservationContext:
    """Concrete source metadata attached to an observation file."""

    project_id: str
    source_session_id: str
    source_agent: str
    source_trajectory_digest: str | None
    record_tool_call_id: str | None
    record_worker_agent: str


def _normalize(text: str) -> str:
    """Collapse whitespace before deriving the dedupe id."""
    return " ".join(text.strip().split())


def _observation_id(
    *,
    memory_type: MemoryType,
    scope: MemoryScope,
    observation: str,
    why_it_matters: str,
) -> str:
    """Return a deterministic id for semantically identical observations."""
    key = "\n".join(
        [
            memory_type.value,
            scope.value,
            _normalize(observation).casefold(),
            _normalize(why_it_matters).casefold(),
        ]
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"O-{digest}"


def _agent_path(memory_path: str) -> str:
    """Translate a memory-relative path to the virtual path agents see."""
    return f"/memories{memory_path}"


def _memory_path(
    *,
    observation_id: str,
    scope: MemoryScope,
    project_id: str,
) -> str:
    """Return the memory-relative path for an observation id."""
    if scope == MemoryScope.PROJECT:
        return f"{OBSERVATION_DIR}/projects/{project_id}/{observation_id}.md"
    return f"{OBSERVATION_DIR}/global/{observation_id}.md"


def _json_string(value: str) -> str:
    """Render a string as a YAML-safe JSON scalar."""
    return json.dumps(value, ensure_ascii=False)


def _format_frontmatter(
    *,
    observation_id: str,
    created_at: str,
    memory_type: MemoryType,
    summary: str,
    scope: MemoryScope,
    source_type: MemorySourceType,
    source_agent: str,
    project_id: str,
) -> str:
    """Build the frontmatter block for an observation file."""
    lines = [
        "---",
        f"id: {_json_string(observation_id)}",
        f"created_at: {_json_string(created_at)}",
        f"summary: {_json_string(summary)}",
        f"memory_type: {memory_type.value}",
        f"scope: {scope.value}",
    ]
    if scope == MemoryScope.PROJECT:
        lines.append(f"project_id: {_json_string(project_id)}")
    lines.extend(
        [
            "source:",
            f"  type: {source_type.value}",
            f"  agent: {_json_string(source_agent)}",
        ]
    )
    lines.append("---")
    return "\n".join(lines)


def _format_observation_markdown(
    *,
    observation_id: str,
    created_at: str,
    memory_type: MemoryType,
    summary: str,
    observation: str,
    why_it_matters: str,
    evidence: str | None,
    scope: MemoryScope,
    source_type: MemorySourceType,
    source_agent: str,
    project_id: str,
) -> str:
    """Render a complete observation markdown document."""
    frontmatter = _format_frontmatter(
        observation_id=observation_id,
        created_at=created_at,
        memory_type=memory_type,
        summary=summary,
        scope=scope,
        source_type=source_type,
        source_agent=source_agent,
        project_id=project_id,
    )
    body = (
        f"{frontmatter}\n\n"
        "## Observation\n\n"
        f"{observation.strip()}\n\n"
        "## Why It Matters\n\n"
        f"{why_it_matters.strip()}\n"
    )
    if evidence and evidence.strip():
        body += f"\n## Evidence\n\n{evidence.strip()}\n"
    return body


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


def _runtime_session_id(runtime: ToolRuntime | None) -> str:
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
    return "unknown"


def _runtime_tool_call_id(runtime: ToolRuntime | None) -> str | None:
    """Extract the active tool call id from runtime metadata when present."""
    if runtime is None or not runtime.tool_call_id:
        return None
    return str(runtime.tool_call_id)


def _resolve_observation_context(
    runtime: ToolRuntime | None,
    *,
    project_id: str,
    source_agent: str,
    source_tool_call_id: str | None,
) -> _ObservationContext:
    """Resolve required observation metadata from fixed values and runtime."""
    return _ObservationContext(
        project_id=_runtime_config_value(runtime, "evomemory_project_id") or project_id,
        source_session_id=_runtime_session_id(runtime),
        source_agent=_runtime_config_value(runtime, "evomemory_source_agent")
        or source_agent,
        source_trajectory_digest=_runtime_config_value(
            runtime, "evomemory_trajectory_digest"
        ),
        record_tool_call_id=source_tool_call_id
        if source_tool_call_id is not None
        else _runtime_tool_call_id(runtime),
        record_worker_agent=source_agent,
    )


def record_observation_file(
    *,
    memory_dir: str | Path,
    project_id: str,
    memory_type: MemoryType,
    summary: str,
    observation: str,
    why_it_matters: str,
    scope: MemoryScope,
    source_type: MemorySourceType,
    source_session_id: str,
    source_agent: str,
    source_trajectory_digest: str | None = None,
    source_tool_call_id: str | None = None,
    record_worker_agent: str | None = None,
    evidence: str | None = None,
) -> ObservationRecordResult:
    """Create an observation markdown file unless an equivalent one exists.

    The id is derived from the normalized observation text, rationale, type, and
    scope, so repeated attempts to save the same observation return the existing
    path instead of creating duplicates.
    """

    summary_text = summary.strip()
    observation_text = observation.strip()
    why_text = why_it_matters.strip()
    if not summary_text:
        raise ValueError("summary must not be empty")
    if not observation_text:
        raise ValueError("observation must not be empty")
    if not why_text:
        raise ValueError("why_it_matters must not be empty")

    observation_id = _observation_id(
        memory_type=memory_type,
        scope=scope,
        observation=observation_text,
        why_it_matters=why_text,
    )
    memory_path = _memory_path(
        observation_id=observation_id,
        scope=scope,
        project_id=project_id,
    )
    path = Path(memory_dir).expanduser() / memory_path.lstrip("/")
    created = False
    if not path.exists():
        created_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        content = _format_observation_markdown(
            observation_id=observation_id,
            created_at=created_at,
            memory_type=memory_type,
            summary=summary_text,
            observation=observation_text,
            why_it_matters=why_text,
            evidence=evidence.strip() if evidence else None,
            scope=scope,
            source_type=source_type,
            source_agent=source_agent,
            project_id=project_id,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        created = True

    result: ObservationRecordResult = {
        "observation_id": observation_id,
        "path": _agent_path(memory_path),
        "created": created,
        "memory_type": memory_type,
        "scope": scope,
    }
    if scope == MemoryScope.PROJECT:
        result["project_id"] = project_id
    return result


def create_record_observation_tool(
    *,
    memory_dir: str | Path,
    project_id: str,
    source_type: MemorySourceType,
    source_agent: str,
    source_tool_call_id: str | None = None,
) -> BaseTool:
    """Build the `record_observation` tool for one agent context."""

    def _record_observation(
        memory_type: MemoryType,
        summary: str,
        observation: str,
        why_it_matters: str,
        scope: MemoryScope,
        evidence: str | None = None,
        runtime: ToolRuntime | None = None,
    ) -> str:
        context = _resolve_observation_context(
            runtime,
            project_id=project_id,
            source_agent=source_agent,
            source_tool_call_id=source_tool_call_id,
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
            source_trajectory_digest=context.source_trajectory_digest,
            source_tool_call_id=context.record_tool_call_id,
            record_worker_agent=context.record_worker_agent,
        )
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
