"""File-backed observation memory.

Observations are small markdown files under `/memories/observations/`. Each
file has stable frontmatter for future indexing plus a short body that agents
can grep and read with ordinary file tools today.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import yaml

from ..search import (
    search_documents,
)
from ..types import (
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


def read_observation_id_from_path(path: str | Path) -> str | None:
    """Read an observation id from a concrete markdown file path."""
    document = _read_observation_document(Path(path).expanduser())
    if document is None:
        return None
    metadata, _body = document
    observation_id = str(metadata.get("id") or "").strip()
    return observation_id or None


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


def observation_document_by_id(
    *,
    memory_dir: str | Path,
    project_id: str,
    observation_id: str,
) -> tuple[Path, dict[str, object], str] | None:
    """Return the stored document tuple for one observation id."""
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
        metadata, body = document
        if str(metadata.get("id") or "").strip() == requested_id:
            return path, metadata, body
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
