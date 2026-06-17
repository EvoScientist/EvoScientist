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
from pathlib import Path
from typing import Annotated

import yaml
from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, InjectedToolArg, StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from .search import (
    search_documents,
)
from .types import (
    MemoryScope,
    MemorySourceType,
    MemoryType,
    ObservationReadResult,
    ObservationRecordResult,
    ObservationSearchDocument,
    ObservationSearchHit,
    ObservationSearchMode,
)

OBSERVATION_DIR = "/observations"


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
    runtime: Annotated[ToolRuntime | None, InjectedToolArg] = None


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


class ReadMemoryArgs(BaseModel):
    """Model-facing arguments for the `read_memory` tool."""

    observation_id: str = Field(
        min_length=1,
        description=(
            "Exact observation ID to read, such as an ID returned by "
            "`search_observations` or listed in the inlined observation index."
        ),
    )


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


def _read_observation_document(path: Path) -> tuple[dict[str, object], str] | None:
    """Read an observation markdown document and parse its frontmatter."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    if not text.startswith("---\n"):
        return None
    try:
        frontmatter, body = text.removeprefix("---\n").split("\n---\n", 1)
        metadata = yaml.safe_load(frontmatter)
    except (ValueError, yaml.YAMLError):
        return None
    if not isinstance(metadata, dict):
        return None
    return {key: value for key, value in metadata.items() if isinstance(key, str)}, body


def _observation_files(
    *,
    memory_dir: str | Path,
    project_id: str,
    scope: MemoryScope | None,
) -> list[Path]:
    """Return candidate observation files for the current project context."""
    root = Path(memory_dir).expanduser()
    memory_paths: list[str] = []
    if scope in {None, MemoryScope.GLOBAL}:
        memory_paths.append(f"{OBSERVATION_DIR}/global")
    if scope in {None, MemoryScope.PROJECT}:
        memory_paths.append(f"{OBSERVATION_DIR}/projects/{project_id}")

    paths: list[Path] = []
    for memory_path in memory_paths:
        directory = root / memory_path.lstrip("/")
        try:
            paths.extend(sorted(directory.glob("*.md")))
        except OSError:
            continue
    return paths


def _candidate_observation_documents(
    *,
    memory_dir: str | Path,
    project_id: str,
    scope: MemoryScope | None = None,
    memory_type: MemoryType | None = None,
) -> list[ObservationSearchDocument]:
    """Read candidate observations for the current filters."""
    documents: list[ObservationSearchDocument] = []
    for path in _observation_files(
        memory_dir=memory_dir,
        project_id=project_id,
        scope=scope,
    ):
        document = _read_observation_document(path)
        if document is None:
            continue
        metadata, body = document
        observation_id = str(metadata.get("id") or "").strip()
        summary = str(metadata.get("summary") or "").strip()
        memory_type_value = str(metadata.get("memory_type") or "").strip()
        scope_value = str(metadata.get("scope") or "").strip()
        if (
            not observation_id
            or not summary
            or not memory_type_value
            or not scope_value
        ):
            continue
        try:
            record_type = MemoryType(memory_type_value)
            record_scope = MemoryScope(scope_value)
        except ValueError:
            continue
        if memory_type is not None and record_type != memory_type:
            continue

        try:
            memory_path = (
                "/" + path.relative_to(Path(memory_dir).expanduser()).as_posix()
            )
        except ValueError:
            continue
        documents.append(
            ObservationSearchDocument(
                observation_id=observation_id,
                path=_agent_path(memory_path),
                memory_type=record_type,
                scope=record_scope,
                summary=summary,
                body=body,
            )
        )
    return documents


def search_observation_files(
    *,
    memory_dir: str | Path,
    project_id: str,
    query: str,
    scope: MemoryScope | None = None,
    memory_type: MemoryType | None = None,
    limit: int = 8,
    mode: ObservationSearchMode = ObservationSearchMode.RANKED,
) -> list[ObservationSearchHit]:
    """Search global/current-project observations by ranked relevance by default."""
    query_text = query.strip()
    if not query_text:
        return []
    search_mode = ObservationSearchMode(mode)

    documents = _candidate_observation_documents(
        memory_dir=memory_dir,
        project_id=project_id,
        scope=scope,
        memory_type=memory_type,
    )
    return search_documents(
        documents=documents,
        query=query_text,
        limit=limit,
        mode=search_mode,
    )


def read_observation_file(
    *,
    memory_dir: str | Path,
    project_id: str,
    observation_id: str,
) -> ObservationReadResult | None:
    """Read a full observation document by frontmatter id."""
    requested_id = observation_id.strip()
    if not requested_id:
        return None

    root = Path(memory_dir).expanduser()
    for path in _observation_files(
        memory_dir=root,
        project_id=project_id,
        scope=None,
    ):
        document = _read_observation_document(path)
        if document is None:
            continue
        metadata, _body = document
        record_id = str(metadata.get("id") or "").strip()
        if record_id != requested_id:
            continue

        summary = str(metadata.get("summary") or "").strip()
        memory_type_value = str(metadata.get("memory_type") or "").strip()
        scope_value = str(metadata.get("scope") or "").strip()
        if not summary or not memory_type_value or not scope_value:
            return None
        try:
            memory_type = MemoryType(memory_type_value)
            scope = MemoryScope(scope_value)
            memory_path = "/" + path.relative_to(root).as_posix()
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError, ValueError):
            return None

        return {
            "observation_id": record_id,
            "path": _agent_path(memory_path),
            "memory_type": memory_type,
            "scope": scope,
            "summary": summary,
            "text": text,
        }
    return None


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
    ) -> str:
        search_mode = ObservationSearchMode(mode)
        results = search_observation_files(
            memory_dir=memory_dir,
            project_id=project_id,
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

    def _read_memory(observation_id: str) -> str:
        requested_id = observation_id.strip()
        result = read_observation_file(
            memory_dir=memory_dir,
            project_id=project_id,
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
        return json.dumps(
            {"text": result["text"]},
            ensure_ascii=False,
            sort_keys=True,
        )

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
