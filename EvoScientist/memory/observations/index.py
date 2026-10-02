"""Prompt-facing observation memory indexes."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from ..types import MemoryScope, MemoryType, ObservationSearchDocument
from .store import list_observation_documents

DEFAULT_MAX_INLINE_OBSERVATION_INDEX_CHARS = 12_000


def build_observation_index_context(
    *,
    memory_dir: str | Path,
    project_id: str,
    max_inline_chars: int = DEFAULT_MAX_INLINE_OBSERVATION_INDEX_CHARS,
) -> str:
    """Build a compact observation-memory index for prompts."""
    return _format_observation_index_context(
        _observation_documents(memory_dir=memory_dir, project_id=project_id),
        include_counts=True,
        include_paths=True,
        include_search_hints=True,
        empty_context=True,
        intro="Indexed observations:",
        max_inline_chars=max_inline_chars,
    )


def build_observation_linker_index_context(
    *,
    memory_dir: str | Path,
    project_id: str,
    exclude_ids: Iterable[str],
    max_inline_chars: int = DEFAULT_MAX_INLINE_OBSERVATION_INDEX_CHARS,
) -> str:
    """Build the existing-observation index included in linker launches."""
    return _format_observation_index_context(
        _observation_documents(
            memory_dir=memory_dir,
            project_id=project_id,
            exclude_ids=exclude_ids,
        ),
        include_counts=False,
        include_paths=False,
        include_search_hints=False,
        empty_context=False,
        intro=(
            "Stored observation snapshot excluding the current batch "
            "(id [type/scope]: summary). Read before linking when needed."
        ),
        max_inline_chars=max_inline_chars,
    )


def _observation_documents(
    *,
    memory_dir: str | Path,
    project_id: str,
    exclude_ids: Iterable[str] = (),
) -> list[ObservationSearchDocument]:
    excluded = set(exclude_ids)
    return sorted(
        (
            document
            for document in list_observation_documents(
                memory_dir=memory_dir,
                project_id=project_id,
            )
            if document.observation_id not in excluded
        ),
        key=lambda document: document.observation_id,
    )


def _format_observation_index_context(
    documents: Sequence[ObservationSearchDocument],
    *,
    include_counts: bool = True,
    include_paths: bool = True,
    include_search_hints: bool = True,
    empty_context: bool = True,
    intro: str = "Indexed observations:",
    max_inline_chars: int = DEFAULT_MAX_INLINE_OBSERVATION_INDEX_CHARS,
) -> str:
    """Format parsed observation documents as a prompt index."""
    if not documents and not empty_context:
        return ""

    header = ["<observation_memory>"]
    if include_counts:
        header.append(_observation_index_count_line(documents))

    footer = [_observation_search_hints()] if include_search_hints else []
    if not documents:
        return "\n".join([*header, *footer, "</observation_memory>"])

    lines = [
        _observation_index_line(document, include_paths=include_paths)
        for document in documents
    ]
    full = "\n".join(
        [
            *header,
            intro,
            *lines,
            *footer,
            "</observation_memory>",
        ]
    )
    if len(full) <= max_inline_chars:
        return full

    return _truncated_observation_index_context(
        header=header,
        intro=intro,
        lines=lines,
        footer=footer,
        max_inline_chars=max_inline_chars,
    )


def _truncated_observation_index_context(
    *,
    header: Sequence[str],
    intro: str,
    lines: Sequence[str],
    footer: Sequence[str],
    max_inline_chars: int,
) -> str:
    prefix = [
        *header,
        "Observation index truncated to entries that fit.",
        intro,
    ]
    suffix = [*footer, "</observation_memory>"]
    selected: list[str] = []
    for line in lines:
        candidate = "\n".join([*prefix, *selected, line, *suffix])
        if len(candidate) <= max_inline_chars:
            selected.append(line)
    if selected:
        return "\n".join([*prefix, *selected, *suffix])

    return "\n".join(
        [
            *header,
            "Observation summaries are too large to inline; search on demand.",
            *footer,
            "</observation_memory>",
        ]
    )


def _observation_index_line(
    document: ObservationSearchDocument,
    *,
    include_paths: bool,
) -> str:
    typed_scope = f"[{document.memory_type.value}/{document.scope.value}]"
    if include_paths:
        return (
            f"- {document.observation_id} {typed_scope} "
            f"{document.path}: {document.summary}"
        )
    return f"- {document.observation_id} {typed_scope}: {document.summary}"


def _observation_index_count_line(
    documents: Sequence[ObservationSearchDocument],
) -> str:
    """Return compact observation counts by scope and memory type."""
    scope_counts = dict.fromkeys(MemoryScope, 0)
    type_counts = dict.fromkeys(MemoryType, 0)
    for document in documents:
        scope_counts[document.scope] += 1
        type_counts[document.memory_type] += 1
    return (
        f"Counts: total={len(documents)}; "
        f"scope global={scope_counts[MemoryScope.GLOBAL]}, "
        f"project={scope_counts[MemoryScope.PROJECT]}; "
        f"type semantic={type_counts[MemoryType.SEMANTIC]}, "
        f"procedural={type_counts[MemoryType.PROCEDURAL]}, "
        f"episodic={type_counts[MemoryType.EPISODIC]}."
    )


def _observation_search_hints() -> str:
    """Return stable search hints for observation memory."""
    return "\n".join(
        [
            "Search hints:",
            "- Each line gives id, type/scope, path, and summary.",
            (
                "- Use `search_observations` for ranked keyword search "
                "and `read_memory` for known observation IDs."
            ),
            "- Use `mode=regex` only when exact grep-like matching is required.",
            "- Search by id when you already know it from the index.",
            (
                "- Filter by type when appropriate: "
                "`memory_type: procedural`, `memory_type: semantic`, or "
                "`memory_type: episodic`."
            ),
            (
                "- Filter by scope when appropriate: "
                "`scope: project` or `scope: global`."
            ),
            (
                "- Search with a few distinctive words or phrases from "
                "the current work that describe the issue, constraint, "
                "procedure, or prior result to find."
            ),
        ]
    )
