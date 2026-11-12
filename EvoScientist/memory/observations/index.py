"""Prompt-facing observation memory indexes."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from ..types import MemoryScope, MemoryType, ObservationSearchDocument
from .store import list_observation_documents

DEFAULT_MAX_INLINE_OBSERVATION_INDEX_CHARS = 12_000
DEFAULT_INLINE_ROWS_PER_SESSION = 3


def build_observation_index_context(
    *,
    memory_dir: str | Path,
    project_id: str,
    max_inline_chars: int = DEFAULT_MAX_INLINE_OBSERVATION_INDEX_CHARS,
    include_search_hints: bool = True,
    rows_per_session: int | None = DEFAULT_INLINE_ROWS_PER_SESSION,
) -> str:
    """Build a compact observation-memory index for prompts.

    ``include_search_hints=False`` omits the tool-usage footer, for model calls
    that carry no memory tools. When the index does not fit, every source
    session's newest ``rows_per_session`` rows claim space first and the rest
    fill what is left, so one session that wrote many observations cannot push
    the others out; ``None`` fills by time alone.
    """
    return _format_observation_index_context(
        _observation_documents(memory_dir=memory_dir, project_id=project_id),
        include_counts=True,
        include_paths=True,
        include_search_hints=include_search_hints,
        can_search=include_search_hints,
        empty_context=True,
        intro="Indexed observations:",
        max_inline_chars=max_inline_chars,
        rows_per_session=rows_per_session,
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
        # The linker agent carries search_observations / read_memory.
        can_search=True,
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
    documents = sorted(
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
    # Newest first, so a truncated index drops the oldest entries; undated ones go last.
    return sorted(documents, key=_created_at_key, reverse=True)


def _created_at_key(document: ObservationSearchDocument) -> tuple[int, float]:
    """Order by the instant, not the string: ``+02:00`` and ``Z`` spellings mix."""
    raw = (document.created_at or "").strip()
    if not raw:
        return (0, 0.0)
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return (0, 0.0)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return (1, parsed.timestamp())


def _fill_order(
    documents: Sequence[ObservationSearchDocument],
    rows_per_session: int,
) -> list[int]:
    """Return the order in which rows claim space.

    ``documents`` is newest first. Each source session's newest
    ``rows_per_session`` rows come first; documents without a source session are
    never deferred.
    """
    seen: dict[str, int] = {}
    first: list[int] = []
    deferred: list[int] = []
    for index, document in enumerate(documents):
        session = document.source_session_id
        if session is not None:
            seen[session] = seen.get(session, 0) + 1
            if seen[session] > rows_per_session:
                deferred.append(index)
                continue
        first.append(index)
    return first + deferred


def _session_share_note(rows_per_session: int, *, can_search: bool) -> str:
    note = (
        f"Each session's newest {rows_per_session} observations are listed "
        "before its older ones"
    )
    if can_search:
        return f"{note}; search_observations finds the rest."
    return f"{note}."


def _format_observation_index_context(
    documents: Sequence[ObservationSearchDocument],
    *,
    include_counts: bool = True,
    include_paths: bool = True,
    include_search_hints: bool = True,
    can_search: bool = True,
    empty_context: bool = True,
    intro: str = "Indexed observations:",
    max_inline_chars: int = DEFAULT_MAX_INLINE_OBSERVATION_INDEX_CHARS,
    rows_per_session: int | None = None,
) -> str:
    """Format parsed observation documents as a prompt index.

    ``can_search`` says whether the reader has a search tool; it decides
    whether an index too large to inline tells the reader to search.
    ``rows_per_session`` shares the space by source session when the index
    does not fit, as described in ``build_observation_index_context``.
    """
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

    shared = rows_per_session is not None
    return _truncated_observation_index_context(
        header=header,
        intro=intro,
        lines=lines,
        footer=footer,
        can_search=can_search,
        max_inline_chars=max_inline_chars,
        order=_fill_order(documents, rows_per_session) if shared else None,
        note=(
            _session_share_note(rows_per_session, can_search=can_search)
            if shared
            else None
        ),
    )


def _truncated_observation_index_context(
    *,
    header: Sequence[str],
    intro: str,
    lines: Sequence[str],
    footer: Sequence[str],
    can_search: bool,
    max_inline_chars: int,
    order: Sequence[int] | None = None,
    note: str | None = None,
) -> str:
    """Fill the budget with ``lines`` taken in ``order``, shown in their own order.

    ``note`` is shown, with its space reserved, only when ``order`` keeps other
    lines than taking them by time would.
    """
    prefix = [*header, "Observation index truncated to entries that fit.", intro]
    suffix = [*footer, "</observation_memory>"]
    selected = _fill(lines, range(len(lines)), [*prefix, *suffix], max_inline_chars)
    notes: list[str] = []
    if order is not None:
        reordered = _fill(lines, order, [*prefix, *suffix], max_inline_chars)
        if note is not None and set(reordered) != set(selected):
            with_note = _fill(lines, order, [*prefix, note, *suffix], max_inline_chars)
            # A note that leaves no room for any row is dropped instead.
            if with_note:
                notes = [note]
                reordered = with_note
        selected = reordered
    if selected:
        kept = [lines[index] for index in sorted(selected)]
        return "\n".join([*prefix, *notes, *kept, *suffix])

    # A reader without a search tool (a tool-less model call) is not told to
    # search.
    too_large = "Observation summaries are too large to inline"
    return "\n".join(
        [
            *header,
            f"{too_large}; search on demand." if can_search else f"{too_large}.",
            *footer,
            "</observation_memory>",
        ]
    )


def _fill(
    lines: Sequence[str],
    order: Iterable[int],
    frame: Sequence[str],
    max_inline_chars: int,
) -> list[int]:
    """Take lines in ``order`` while they still fit next to ``frame``."""
    size = len("\n".join(frame))
    taken: list[int] = []
    for index in order:
        # Each added line costs its length plus one newline.
        if size + 1 + len(lines[index]) <= max_inline_chars:
            taken.append(index)
            size += 1 + len(lines[index])
    return taken


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
