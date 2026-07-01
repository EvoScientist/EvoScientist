"""File-backed observation memory.

Observations are small markdown files under `/memories/observations/`. Each
file has stable frontmatter for future indexing plus a short body that agents
can grep and read with ordinary file tools today.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ..search import (
    search_documents,
)
from ..types import (
    MemoryScope,
    MemorySourceType,
    MemoryType,
    ObservationReadResult,
    ObservationRecordResult,
    ObservationRelation,
    ObservationSearchDocument,
    ObservationSearchHit,
    ObservationSearchMode,
    RelatedObservationResult,
)

OBSERVATION_DIR = "/observations"


ObservationFrontmatterValue = str | dict[str, str] | list[dict[str, str]]
ObservationFrontmatterPayload = dict[str, ObservationFrontmatterValue]


class RelatedObservationEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(min_length=1, strict=True)
    relation: ObservationRelation
    reason: str = Field(min_length=1, strict=True)
    linked_at: str = Field(min_length=1, strict=True)

    @field_validator("id", "reason", "linked_at")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    def to_frontmatter_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "relation": self.relation.value,
            "reason": self.reason,
            "linked_at": self.linked_at,
        }


class ObservationSourceFrontmatter(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: MemorySourceType
    agent: str = Field(min_length=1, strict=True)
    session_id: str | None = Field(default=None, min_length=1, strict=True)

    @field_validator("agent", "session_id")
    @classmethod
    def _non_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must not be blank")
        return value

    def to_frontmatter_dict(self) -> dict[str, str]:
        payload = {
            "type": self.type.value,
            "agent": self.agent,
        }
        if self.session_id is not None:
            payload["session_id"] = self.session_id
        return payload


class ObservationFrontmatter(BaseModel):
    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    id: str = Field(min_length=1, strict=True)
    created_at: str | None = Field(default=None, min_length=1, strict=True)
    summary: str = Field(min_length=1, strict=True)
    memory_type: MemoryType
    scope: MemoryScope
    project_id: str | None = Field(default=None, min_length=1, strict=True)
    source: ObservationSourceFrontmatter | None = None
    related_observations: list[RelatedObservationEntry] = Field(default_factory=list)

    @field_validator("id", "summary", "created_at", "project_id")
    @classmethod
    def _non_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("created_at", mode="before")
    @classmethod
    def _coerce_yaml_timestamp(cls, value: object) -> object:
        if isinstance(value, datetime):
            if value.tzinfo is not None:
                value = value.astimezone(UTC)
                return value.strftime("%Y-%m-%dT%H:%M:%SZ")
            return value.strftime("%Y-%m-%dT%H:%M:%S")
        if isinstance(value, date):
            return value.isoformat()
        return value

    def to_frontmatter_dict(self) -> ObservationFrontmatterPayload:
        payload: ObservationFrontmatterPayload = {
            "id": self.id,
        }
        if self.created_at is not None:
            payload["created_at"] = self.created_at
        payload["summary"] = self.summary
        payload["memory_type"] = self.memory_type.value
        payload["scope"] = self.scope.value
        if self.project_id is not None:
            payload["project_id"] = self.project_id
        if self.source is not None:
            payload["source"] = self.source.to_frontmatter_dict()
        if self.related_observations:
            payload["related_observations"] = [
                entry.to_frontmatter_dict() for entry in self.related_observations
            ]
        return payload


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


def _read_observation_document_with_text(
    path: str | Path,
) -> tuple[ObservationFrontmatter, str, str] | None:
    """Read an observation markdown document, body, and original text."""
    document_path = Path(path).expanduser()
    try:
        text = document_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    if not text.startswith("---\n"):
        return None
    try:
        frontmatter, body = text.removeprefix("---\n").split("\n---\n", 1)
        metadata = ObservationFrontmatter.model_validate(yaml.safe_load(frontmatter))
    except (ValueError, ValidationError, yaml.YAMLError):
        return None
    return metadata, body, text


def read_observation_document(
    path: str | Path,
) -> tuple[ObservationFrontmatter, str] | None:
    """Read an observation markdown document and parse its frontmatter."""
    document = _read_observation_document_with_text(path)
    if document is None:
        return None
    metadata, body, _text = document
    return metadata, body


def write_observation_document(
    path: str | Path,
    *,
    metadata: ObservationFrontmatter,
    body: str,
) -> None:
    """Write an observation markdown document with frontmatter."""
    frontmatter = yaml.safe_dump(
        metadata.to_frontmatter_dict(),
        allow_unicode=True,
        sort_keys=False,
    )
    Path(path).write_text(f"---\n{frontmatter}---\n{body}", encoding="utf-8")


def read_observation_id_from_path(path: str | Path) -> str | None:
    """Read an observation id from a concrete markdown file path."""
    document = read_observation_document(path)
    if document is None:
        return None
    metadata, _body = document
    return metadata.id.strip()


def related_observation_entries(
    metadata: ObservationFrontmatter,
) -> list[RelatedObservationEntry]:
    """Return related-observation frontmatter entries."""
    return list(metadata.related_observations)


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


def _all_observation_files(root: Path) -> list[Path]:
    observation_root = root / OBSERVATION_DIR.lstrip("/")
    try:
        return sorted(path for path in observation_root.rglob("*.md") if path.is_file())
    except OSError:
        return []


def _resolve_related_observations(
    entries: list[RelatedObservationEntry],
    *,
    documents_by_id: dict[str, ObservationSearchDocument],
) -> tuple[RelatedObservationResult, ...]:
    related_observations: list[RelatedObservationResult] = []
    for entry in entries:
        related_id = entry.id
        if related_id not in documents_by_id:
            continue
        target = documents_by_id[related_id]
        related: RelatedObservationResult = {
            "observation_id": target.observation_id,
            "path": target.path,
            "memory_type": target.memory_type,
            "scope": target.scope,
            "summary": target.summary,
            "relation": entry.relation,
            "reason": entry.reason,
        }
        related_observations.append(related)
    return tuple(related_observations)


def _parse_observation_search_document(
    *,
    root: Path,
    path: Path,
) -> tuple[ObservationSearchDocument, list[RelatedObservationEntry]] | None:
    document = _read_observation_document_with_text(path)
    if document is None:
        return None
    metadata, body, text = document
    try:
        memory_path = "/" + path.relative_to(root).as_posix()
    except ValueError:
        return None

    return (
        ObservationSearchDocument(
            observation_id=metadata.id,
            path=_agent_path(memory_path),
            memory_type=metadata.memory_type,
            scope=metadata.scope,
            summary=metadata.summary,
            body=body,
            text=text,
        ),
        related_observation_entries(metadata),
    )


def _resolve_document_links(
    parsed: list[tuple[ObservationSearchDocument, list[RelatedObservationEntry]]],
    *,
    root: Path,
) -> list[ObservationSearchDocument]:
    documents_by_id = {document.observation_id: document for document, _ in parsed}
    missing_related_ids = {
        entry.id
        for _document, entries in parsed
        for entry in entries
        if entry.id not in documents_by_id
    }
    if missing_related_ids:
        for path in _all_observation_files(root):
            if not missing_related_ids:
                break
            parsed_document = _parse_observation_search_document(root=root, path=path)
            if parsed_document is None:
                continue
            document, _entries = parsed_document
            if document.observation_id not in missing_related_ids:
                continue
            documents_by_id[document.observation_id] = document
            missing_related_ids.remove(document.observation_id)

    return [
        replace(
            document,
            related_observations=_resolve_related_observations(
                entries,
                documents_by_id=documents_by_id,
            ),
        )
        for document, entries in parsed
    ]


def list_observation_documents(
    *,
    memory_dir: str | Path,
    project_id: str,
    scope: MemoryScope | None = None,
    memory_type: MemoryType | None = None,
) -> list[ObservationSearchDocument]:
    """Read candidate observations for the current filters."""
    root = Path(memory_dir).expanduser()
    parsed: list[tuple[ObservationSearchDocument, list[RelatedObservationEntry]]] = []
    for path in _observation_files(
        memory_dir=root,
        project_id=project_id,
        scope=scope,
    ):
        parsed_document = _parse_observation_search_document(root=root, path=path)
        if parsed_document is not None:
            parsed.append(parsed_document)

    # Resolve links before filtering by memory_type so a procedural hit can still
    # surface a linked semantic observation, and vice versa.
    documents = _resolve_document_links(parsed, root=root)
    if memory_type is not None:
        return [
            document for document in documents if document.memory_type == memory_type
        ]
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

    documents = list_observation_documents(
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
    for document in list_observation_documents(
        memory_dir=root,
        project_id=project_id,
        scope=None,
    ):
        if document.observation_id != requested_id:
            continue
        result: ObservationReadResult = {
            "observation_id": document.observation_id,
            "path": document.path,
            "memory_type": document.memory_type,
            "scope": document.scope,
            "summary": document.summary,
            "text": document.text,
        }
        if document.related_observations:
            result["related_observations"] = list(document.related_observations)
        return result
    return None


def observation_document_by_id(
    *,
    memory_dir: str | Path,
    project_id: str,
    observation_id: str,
) -> tuple[Path, ObservationFrontmatter, str] | None:
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
        document = read_observation_document(path)
        if document is None:
            continue
        metadata, body = document
        if metadata.id == requested_id:
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
    source_session_id: str,
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
    lines.append(f"  session_id: {_json_string(source_session_id.strip())}")
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
    source_session_id: str,
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
        source_session_id=source_session_id,
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
    if not source_session_id.strip():
        raise ValueError("source_session_id must not be empty")

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
            source_session_id=source_session_id,
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
