"""File-backed synthesized knowledge memory.

Knowledge records are compact abstractions over observations. They live beside
observation memory as markdown files, but they carry explicit supporting
observation IDs so future agents can inspect the underlying evidence when the
abstraction matters.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import yaml
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from .observations import (
    candidate_observation_documents,
    read_observation_file,
)
from .search import search_memory_documents
from .types import (
    KnowledgeReadResult,
    KnowledgeRecordResult,
    KnowledgeSearchDocument,
    KnowledgeStatus,
    MemoryLevel,
    MemoryLevelFilter,
    MemoryScope,
    MemorySearchDocument,
    MemorySearchHit,
    MemorySearchMode,
    MemorySourceType,
    MemoryType,
    ObservationReadResult,
)

KNOWLEDGE_DIR = "/knowledge"
SYNTHESIS_AGENT_NAME = "evomemory-synthesizer"


class SearchMemoryArgs(BaseModel):
    """Model-facing arguments for the combined `search_memory` tool."""

    query: str = Field(
        min_length=1,
        description=(
            "Search text. Ranked mode prefers synthesized knowledge and omits "
            "observations already covered by active knowledge. Use distinctive "
            "keywords or short phrases describing the reusable fact, rule, "
            "procedure, issue, or prior result."
        ),
    )
    mode: MemorySearchMode = Field(
        default=MemorySearchMode.RANKED,
        description=(
            "ranked interprets query as keyword text. regex interprets query as "
            "a case-insensitive grep-like pattern and falls back to literal "
            "matching when the pattern is invalid."
        ),
    )
    memory_level: MemoryLevelFilter = Field(
        default=MemoryLevelFilter.ANY,
        description=(
            "Which memory level to search: any searches active knowledge plus "
            "uncovered observations; knowledge searches synthesized K-* "
            "records only; observation searches raw O-* evidence only."
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
            "Optional memory-type filter: procedural for commands/workarounds, "
            "semantic for reusable facts/findings, episodic for notable events. "
            "This is distinct from memory_level."
        ),
    )
    include_archived_knowledge: bool = Field(
        default=False,
        description="Whether archived knowledge records may appear in results.",
    )
    limit: int = Field(
        default=8,
        ge=1,
        le=20,
        description="Maximum number of matching memories to return.",
    )


class ReadMemoryArgs(BaseModel):
    """Model-facing arguments for the combined `read_memory` tool."""

    memory_id: str = Field(
        min_length=1,
        description=(
            "Exact memory ID to read, such as a K-* knowledge ID or O-* "
            "observation ID returned by `search_memory`."
        ),
    )


def _normalize(text: str) -> str:
    return " ".join(text.strip().split())


def _knowledge_id(
    *,
    memory_type: MemoryType,
    scope: MemoryScope,
    knowledge: str,
    supporting_observation_ids: list[str],
) -> str:
    key = "\n".join(
        [
            memory_type.value,
            scope.value,
            _normalize(knowledge).casefold(),
            "\n".join(sorted(supporting_observation_ids)),
        ]
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"K-{digest}"


def _agent_path(memory_path: str) -> str:
    return f"/memories{memory_path}"


def _memory_path(
    *,
    knowledge_id: str,
    scope: MemoryScope,
    project_id: str,
) -> str:
    if scope == MemoryScope.PROJECT:
        return f"{KNOWLEDGE_DIR}/projects/{project_id}/{knowledge_id}.md"
    return f"{KNOWLEDGE_DIR}/global/{knowledge_id}.md"


def _json_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _json_list(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=False)


def _read_memory_document(path: Path) -> tuple[dict[str, object], str] | None:
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


def _string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [item for raw in value if (item := str(raw).strip())]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _memory_level_filter(
    value: MemoryLevelFilter | MemoryLevel | str,
) -> MemoryLevelFilter | None:
    try:
        return MemoryLevelFilter(str(value))
    except ValueError:
        return None


def _knowledge_files(
    *,
    memory_dir: str | Path,
    project_id: str,
    scope: MemoryScope | None,
) -> list[Path]:
    root = Path(memory_dir).expanduser()
    memory_paths: list[str] = []
    if scope in {None, MemoryScope.GLOBAL}:
        memory_paths.append(f"{KNOWLEDGE_DIR}/global")
    if scope in {None, MemoryScope.PROJECT}:
        memory_paths.append(f"{KNOWLEDGE_DIR}/projects/{project_id}")

    paths: list[Path] = []
    for memory_path in memory_paths:
        directory = root / memory_path.lstrip("/")
        try:
            paths.extend(sorted(directory.glob("*.md")))
        except OSError:
            continue
    return paths


def _find_knowledge_file(
    *,
    memory_dir: str | Path,
    project_id: str,
    knowledge_id: str,
) -> Path | None:
    requested_id = knowledge_id.strip()
    if not requested_id:
        return None
    for path in _knowledge_files(memory_dir=memory_dir, project_id=project_id, scope=None):
        document = _read_memory_document(path)
        if document is None:
            continue
        metadata, _body = document
        if str(metadata.get("id") or "").strip() == requested_id:
            return path
    return None


def _knowledge_document_from_path(
    *,
    path: Path,
    memory_dir: str | Path,
) -> KnowledgeSearchDocument | None:
    document = _read_memory_document(path)
    if document is None:
        return None
    metadata, body = document
    knowledge_id = str(metadata.get("id") or "").strip()
    summary = str(metadata.get("summary") or "").strip()
    memory_type_value = str(metadata.get("memory_type") or "").strip()
    scope_value = str(metadata.get("scope") or "").strip()
    status_value = str(metadata.get("status") or KnowledgeStatus.ACTIVE).strip()
    if not knowledge_id or not summary or not memory_type_value or not scope_value:
        return None
    try:
        memory_type = MemoryType(memory_type_value)
        scope = MemoryScope(scope_value)
        status = KnowledgeStatus(status_value)
        memory_path = "/" + path.relative_to(Path(memory_dir).expanduser()).as_posix()
    except ValueError:
        return None
    return KnowledgeSearchDocument(
        knowledge_id=knowledge_id,
        path=_agent_path(memory_path),
        memory_type=memory_type,
        scope=scope,
        summary=summary,
        body=body,
        status=status,
        supporting_observation_ids=tuple(
            _string_list(metadata.get("supporting_observation_ids"))
        ),
    )


def knowledge_search_documents(
    *,
    memory_dir: str | Path,
    project_id: str,
    scope: MemoryScope | None = None,
    memory_type: MemoryType | None = None,
    status: KnowledgeStatus | None = KnowledgeStatus.ACTIVE,
) -> list[KnowledgeSearchDocument]:
    """Read candidate knowledge documents for the current filters."""
    documents: list[KnowledgeSearchDocument] = []
    for path in _knowledge_files(
        memory_dir=memory_dir,
        project_id=project_id,
        scope=scope,
    ):
        document = _knowledge_document_from_path(path=path, memory_dir=memory_dir)
        if document is None:
            continue
        if memory_type is not None and document.memory_type != memory_type:
            continue
        if status is not None and document.status != status:
            continue
        documents.append(document)
    return documents


def _markdown_sections(body: str) -> dict[str, str]:
    matches = list(re.finditer(r"^## (?P<title>.+)$", body, flags=re.MULTILINE))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        sections[match.group("title")] = body[start:end].strip()
    return sections


def _existing_created_at(path: Path) -> str | None:
    document = _read_memory_document(path)
    if document is None:
        return None
    metadata, _body = document
    value = metadata.get("created_at")
    return value if isinstance(value, str) and value else None


def _support_lines(
    *,
    memory_dir: str | Path,
    project_id: str,
    supporting_observation_ids: list[str],
) -> list[str]:
    lines: list[str] = []
    for observation_id in supporting_observation_ids:
        observation = read_observation_file(
            memory_dir=memory_dir,
            project_id=project_id,
            observation_id=observation_id,
        )
        if observation is None:
            lines.append(f"- {observation_id}")
        else:
            lines.append(f"- {observation_id}: {observation['summary']}")
    return lines


def _format_frontmatter(
    *,
    knowledge_id: str,
    created_at: str,
    updated_at: str,
    memory_type: MemoryType,
    summary: str,
    scope: MemoryScope,
    project_id: str,
    status: KnowledgeStatus,
    supporting_observation_ids: list[str],
    supersedes_knowledge_ids: list[str],
    archived_reason: str | None,
    archived_at: str | None,
    source_agent: str,
) -> str:
    lines = [
        "---",
        f"id: {_json_string(knowledge_id)}",
        f"created_at: {_json_string(created_at)}",
        f"updated_at: {_json_string(updated_at)}",
        f"summary: {_json_string(summary)}",
        f"memory_type: {memory_type.value}",
        f"scope: {scope.value}",
    ]
    if scope == MemoryScope.PROJECT:
        lines.append(f"project_id: {_json_string(project_id)}")
    lines.extend(
        [
            f"status: {status.value}",
            f"supporting_observation_ids: {_json_list(supporting_observation_ids)}",
        ]
    )
    if supersedes_knowledge_ids:
        lines.append(f"supersedes_knowledge_ids: {_json_list(supersedes_knowledge_ids)}")
    if archived_at:
        lines.append(f"archived_at: {_json_string(archived_at)}")
    if archived_reason:
        lines.append(f"archived_reason: {_json_string(archived_reason)}")
    lines.extend(
        [
            "source:",
            f"  type: {MemorySourceType.SYNTHESIS.value}",
            f"  agent: {_json_string(source_agent)}",
            "---",
        ]
    )
    return "\n".join(lines)


def _format_knowledge_markdown(
    *,
    knowledge_id: str,
    created_at: str,
    updated_at: str,
    memory_type: MemoryType,
    summary: str,
    knowledge: str,
    when_to_use: str | None,
    scope: MemoryScope,
    project_id: str,
    status: KnowledgeStatus,
    supporting_observation_ids: list[str],
    supersedes_knowledge_ids: list[str],
    archived_reason: str | None,
    archived_at: str | None,
    source_agent: str,
    support_lines: list[str],
) -> str:
    frontmatter = _format_frontmatter(
        knowledge_id=knowledge_id,
        created_at=created_at,
        updated_at=updated_at,
        memory_type=memory_type,
        summary=summary,
        scope=scope,
        project_id=project_id,
        status=status,
        supporting_observation_ids=supporting_observation_ids,
        supersedes_knowledge_ids=supersedes_knowledge_ids,
        archived_reason=archived_reason,
        archived_at=archived_at,
        source_agent=source_agent,
    )
    body = (
        f"{frontmatter}\n\n"
        "## Knowledge\n\n"
        f"{knowledge.strip()}\n\n"
        "## When To Use\n\n"
        f"{(when_to_use or '').strip() or 'Use when this pattern appears again.'}\n\n"
        "## Support\n\n"
        f"{chr(10).join(support_lines) if support_lines else 'No support listed.'}\n"
    )
    if archived_reason:
        body += f"\n## Archive Reason\n\n{archived_reason.strip()}\n"
    return body


def _unique_ids(ids: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw_id in ids:
        memory_id = raw_id.strip()
        if not memory_id or memory_id in seen:
            continue
        seen.add(memory_id)
        out.append(memory_id)
    return out


def _validate_supporting_observations(
    *,
    memory_dir: str | Path,
    project_id: str,
    supporting_observation_ids: list[str],
) -> None:
    if not supporting_observation_ids:
        raise ValueError("knowledge requires at least one supporting observation")
    missing = [
        observation_id
        for observation_id in supporting_observation_ids
        if read_observation_file(
            memory_dir=memory_dir,
            project_id=project_id,
            observation_id=observation_id,
        )
        is None
    ]
    if missing:
        raise ValueError(
            "supporting observations do not exist in global or current-project "
            f"memory: {', '.join(missing)}"
        )


def record_knowledge_file(
    *,
    memory_dir: str | Path,
    project_id: str,
    memory_type: MemoryType,
    summary: str,
    knowledge: str,
    supporting_observation_ids: list[str],
    scope: MemoryScope,
    when_to_use: str | None = None,
    knowledge_id: str | None = None,
    status: KnowledgeStatus = KnowledgeStatus.ACTIVE,
    supersedes_knowledge_ids: list[str] | None = None,
    archived_reason: str | None = None,
    source_agent: str = SYNTHESIS_AGENT_NAME,
) -> KnowledgeRecordResult:
    """Create or update a synthesized knowledge markdown file."""
    summary_text = summary.strip()
    knowledge_text = knowledge.strip()
    support_ids = _unique_ids(supporting_observation_ids)
    supersedes_ids = _unique_ids(supersedes_knowledge_ids or [])
    if not summary_text:
        raise ValueError("summary must not be empty")
    if not knowledge_text:
        raise ValueError("knowledge must not be empty")
    if status == KnowledgeStatus.ACTIVE:
        _validate_supporting_observations(
            memory_dir=memory_dir,
            project_id=project_id,
            supporting_observation_ids=support_ids,
        )

    resolved_id = (knowledge_id or "").strip() or _knowledge_id(
        memory_type=memory_type,
        scope=scope,
        knowledge=knowledge_text,
        supporting_observation_ids=support_ids,
    )
    if not resolved_id.startswith("K-"):
        raise ValueError("knowledge_id must start with K-")

    memory_path = _memory_path(
        knowledge_id=resolved_id,
        scope=scope,
        project_id=project_id,
    )
    path = Path(memory_dir).expanduser() / memory_path.lstrip("/")
    created = not path.exists()
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    created_at = _existing_created_at(path) or now
    archived_at = now if status == KnowledgeStatus.ARCHIVED else None
    support_lines = _support_lines(
        memory_dir=memory_dir,
        project_id=project_id,
        supporting_observation_ids=support_ids,
    )
    content = _format_knowledge_markdown(
        knowledge_id=resolved_id,
        created_at=created_at,
        updated_at=now,
        memory_type=memory_type,
        summary=summary_text,
        knowledge=knowledge_text,
        when_to_use=when_to_use,
        scope=scope,
        project_id=project_id,
        status=status,
        supporting_observation_ids=support_ids,
        supersedes_knowledge_ids=supersedes_ids,
        archived_reason=archived_reason.strip() if archived_reason else None,
        archived_at=archived_at,
        source_agent=source_agent,
        support_lines=support_lines,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

    result: KnowledgeRecordResult = {
        "knowledge_id": resolved_id,
        "path": _agent_path(memory_path),
        "created": created,
        "memory_type": memory_type,
        "scope": scope,
        "status": status,
    }
    if scope == MemoryScope.PROJECT:
        result["project_id"] = project_id
    return result


def read_knowledge_file(
    *,
    memory_dir: str | Path,
    project_id: str,
    knowledge_id: str,
) -> KnowledgeReadResult | None:
    """Read a full knowledge document by frontmatter id."""
    path = _find_knowledge_file(
        memory_dir=memory_dir,
        project_id=project_id,
        knowledge_id=knowledge_id,
    )
    if path is None:
        return None
    document = _knowledge_document_from_path(path=path, memory_dir=memory_dir)
    if document is None:
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    return {
        "knowledge_id": document.knowledge_id,
        "path": document.path,
        "memory_type": document.memory_type,
        "scope": document.scope,
        "summary": document.summary,
        "status": document.status,
        "supporting_observation_ids": list(document.supporting_observation_ids),
        "text": text,
    }


def read_memory_file(
    *,
    memory_dir: str | Path,
    project_id: str,
    memory_id: str,
) -> KnowledgeReadResult | ObservationReadResult | None:
    """Read either a knowledge or observation memory by exact id."""
    requested_id = memory_id.strip()
    if not requested_id:
        return None
    if requested_id.startswith("K-"):
        return read_knowledge_file(
            memory_dir=memory_dir,
            project_id=project_id,
            knowledge_id=requested_id,
        )
    if requested_id.startswith("O-"):
        return read_observation_file(
            memory_dir=memory_dir,
            project_id=project_id,
            observation_id=requested_id,
        )
    return None


def search_knowledge_files(
    *,
    memory_dir: str | Path,
    project_id: str,
    query: str,
    scope: MemoryScope | None = None,
    memory_type: MemoryType | None = None,
    include_archived: bool = False,
    limit: int = 8,
    mode: MemorySearchMode = MemorySearchMode.RANKED,
) -> list[MemorySearchHit]:
    """Search global/current-project knowledge by ranked relevance by default."""
    query_text = query.strip()
    if not query_text:
        return []
    status = None if include_archived else KnowledgeStatus.ACTIVE
    documents = knowledge_search_documents(
        memory_dir=memory_dir,
        project_id=project_id,
        scope=scope,
        memory_type=memory_type,
        status=status,
    )
    return search_memory_documents(
        documents=documents,
        query=query_text,
        limit=limit,
        mode=MemorySearchMode(mode),
    )


def search_memory_files(
    *,
    memory_dir: str | Path,
    project_id: str,
    query: str,
    memory_level: MemoryLevelFilter | MemoryLevel | str = MemoryLevelFilter.ANY,
    scope: MemoryScope | None = None,
    memory_type: MemoryType | None = None,
    include_archived_knowledge: bool = False,
    include_covered_observations: bool = False,
    limit: int = 8,
    mode: MemorySearchMode = MemorySearchMode.RANKED,
) -> list[MemorySearchHit]:
    """Search knowledge plus raw observations with knowledge preferred."""
    query_text = query.strip()
    if not query_text:
        return []

    resolved_level = _memory_level_filter(memory_level)
    if resolved_level is None:
        return []
    knowledge_status = None if include_archived_knowledge else KnowledgeStatus.ACTIVE
    documents: list[MemorySearchDocument] = []
    knowledge_documents: list[KnowledgeSearchDocument] = []
    if resolved_level in {MemoryLevelFilter.ANY, MemoryLevelFilter.KNOWLEDGE}:
        knowledge_documents = knowledge_search_documents(
            memory_dir=memory_dir,
            project_id=project_id,
            scope=scope,
            memory_type=memory_type,
            status=knowledge_status,
        )
        documents.extend(knowledge_documents)

    observation_documents = []
    if resolved_level in {MemoryLevelFilter.ANY, MemoryLevelFilter.OBSERVATION}:
        observation_documents = candidate_observation_documents(
            memory_dir=memory_dir,
            project_id=project_id,
            scope=scope,
            memory_type=memory_type,
        )
        if (
            resolved_level == MemoryLevelFilter.ANY
            and not include_covered_observations
        ):
            covered = {
                observation_id
                for document in knowledge_documents
                if document.status == KnowledgeStatus.ACTIVE
                for observation_id in document.supporting_observation_ids
            }
            observation_documents = [
                document
                for document in observation_documents
                if document.observation_id not in covered
            ]
        documents.extend(observation_documents)

    return search_memory_documents(
        documents=documents,
        query=query_text,
        limit=limit,
        mode=MemorySearchMode(mode),
    )


def archive_knowledge_file(
    *,
    memory_dir: str | Path,
    project_id: str,
    knowledge_id: str,
    reason: str,
    source_agent: str = SYNTHESIS_AGENT_NAME,
) -> KnowledgeRecordResult | None:
    """Mark an existing knowledge file archived while preserving its content."""
    path = _find_knowledge_file(
        memory_dir=memory_dir,
        project_id=project_id,
        knowledge_id=knowledge_id,
    )
    if path is None:
        return None
    raw = _read_memory_document(path)
    if raw is None:
        return None
    metadata, body = raw
    sections = _markdown_sections(body)
    try:
        memory_type = MemoryType(str(metadata.get("memory_type") or ""))
        scope = MemoryScope(str(metadata.get("scope") or ""))
    except ValueError:
        return None
    return record_knowledge_file(
        memory_dir=memory_dir,
        project_id=project_id,
        knowledge_id=knowledge_id,
        memory_type=memory_type,
        summary=str(metadata.get("summary") or "Archived knowledge"),
        knowledge=sections.get("Knowledge", body.strip()),
        when_to_use=sections.get("When To Use"),
        supporting_observation_ids=_string_list(
            metadata.get("supporting_observation_ids")
        ),
        scope=scope,
        status=KnowledgeStatus.ARCHIVED,
        supersedes_knowledge_ids=_string_list(metadata.get("supersedes_knowledge_ids")),
        archived_reason=reason,
        source_agent=source_agent,
    )


def create_search_memory_tool(
    *,
    memory_dir: str | Path,
    project_id: str,
) -> BaseTool:
    """Build the read-only `search_memory` tool for one project context."""

    def _search_memory(
        query: str,
        mode: MemorySearchMode = MemorySearchMode.RANKED,
        memory_level: MemoryLevelFilter = MemoryLevelFilter.ANY,
        scope: MemoryScope | None = None,
        memory_type: MemoryType | None = None,
        include_archived_knowledge: bool = False,
        limit: int = 8,
    ) -> str:
        results = search_memory_files(
            memory_dir=memory_dir,
            project_id=project_id,
            query=query,
            memory_level=memory_level,
            scope=scope,
            memory_type=memory_type,
            include_archived_knowledge=include_archived_knowledge,
            limit=limit,
            mode=mode,
        )
        return json.dumps({"results": results}, ensure_ascii=False, sort_keys=True)

    return StructuredTool.from_function(
        func=_search_memory,
        name="search_memory",
        description=(
            "Search EvoMemory synthesized knowledge and observations together. "
            "Use memory_level=any for general memory preflight, knowledge for "
            "synthesized K-* records, or observation for raw O-* evidence. "
            "Use memory_type to filter semantic/procedural/episodic memories. "
            "Read promising K-* or O-* hits with read_memory."
        ),
        args_schema=SearchMemoryArgs,
        infer_schema=False,
    )


def create_read_memory_tool(
    *,
    memory_dir: str | Path,
    project_id: str,
) -> BaseTool:
    """Build the read-only `read_memory` tool for one project context."""

    def _read_memory(memory_id: str) -> str:
        requested_id = memory_id.strip()
        result = read_memory_file(
            memory_dir=memory_dir,
            project_id=project_id,
            memory_id=requested_id,
        )
        if result is None:
            return json.dumps(
                {
                    "error": (
                        "No memory with that ID exists in global or "
                        "current-project memory."
                    ),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        return json.dumps({"text": result["text"]}, ensure_ascii=False, sort_keys=True)

    return StructuredTool.from_function(
        func=_read_memory,
        name="read_memory",
        description=(
            "Read the full markdown for an EvoMemory knowledge or observation "
            "record by exact K-* or O-* ID."
        ),
        args_schema=ReadMemoryArgs,
        infer_schema=False,
    )
