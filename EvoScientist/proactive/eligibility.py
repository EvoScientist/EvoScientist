"""Server-scoped candidate enumeration for the proactive cron.

The proactive cron must pick which threads to run a check on. Candidates are
enumerated from the **server registry** via ``client.threads.search`` restricted
to threads carrying a channel-origin marker — NOT from a raw ``sessions.db`` scan
(which also holds CLI/TUI threads the cron must never touch, and whose delivery
has no channel origin). So the cron only ever sees server-born channel threads.

Serve-side companion: ``remember_channel_origin`` records the origin only in the
in-process ``_thread_channel_origins`` dict, invisible to the server/cron
process. Serve therefore also stamps ``CHANNEL_ORIGIN_MARKER`` into the thread's
server metadata (``stamp_channel_marker``) when it registers an origin. Without
the stamp the search returns nothing — fail-closed, which is the safe direction
(no thread gets a proactive push rather than the wrong one).

This function returns coarse *candidates* with their last-activity time; the
authoritative per-thread decision (quiet hours, precise idle, in-flight, origin
presence) stays with ``gate.evaluate_gate`` in ``service.run_proactive_check``. It
deliberately does not re-implement the idle test, to avoid a divergent second copy
of the gate logic.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable
from datetime import UTC, datetime
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# Server-metadata marker serve stamps on a thread when it registers a channel
# origin (companion change). Threads.search matches metadata by containment, so a
# value-equality marker is the queryable shape.
CHANNEL_ORIGIN_MARKER: dict[str, Any] = {"has_channel_origin": True}


class _ThreadsSearchClient(Protocol):
    """Minimal client surface: ``client.threads.search(...)`` (langgraph_sdk)."""

    @property
    def threads(self) -> Any: ...


async def list_channel_thread_candidates(
    client: _ThreadsSearchClient,
    *,
    marker: dict[str, Any] | None = None,
    limit: int = 50,
    status: str | None = "idle",
) -> list[tuple[str, datetime | None]]:
    """Return ``(thread_id, last_activity)`` for server-registered channel threads.

    Queries ``client.threads.search`` for threads whose server metadata contains
    ``marker`` (default :data:`CHANNEL_ORIGIN_MARKER`) and whose ``status`` matches
    ``status`` (default ``"idle"`` — no run in flight, the gate's in-flight
    condition enforced cheaply server-side; pass ``None`` to skip it), newest-active
    first, capped at ``limit``. ``last_activity`` is parsed from the thread's
    ``updated_at`` (UTC; ``None`` when absent/unparseable) and is meant to be fed
    straight into the gate. Threads without an id are skipped. An empty list means
    "no eligible candidates" (including the fail-closed case where serve hasn't
    stamped the marker yet).
    """
    search = client.threads.search(
        metadata=marker if marker is not None else CHANNEL_ORIGIN_MARKER,
        status=status,
        sort_by="updated_at",
        sort_order="desc",
        limit=limit,
    )
    threads = await search if isinstance(search, Awaitable) else search

    candidates: list[tuple[str, datetime | None]] = []
    for thread in threads or []:
        thread_id = thread.get("thread_id") if isinstance(thread, dict) else None
        if not thread_id:
            continue
        candidates.append((str(thread_id), _thread_last_activity(thread)))

    logger.debug(
        "[proactive cron] %d channel-thread candidate(s) from server registry",
        len(candidates),
    )
    return candidates


def _thread_last_activity(thread: dict[str, Any]) -> datetime | None:
    """Parse a thread's ``updated_at`` into a UTC datetime (None if absent/bad)."""
    value = thread.get("updated_at")
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    return _parse_iso_utc(str(value))


def _parse_iso_utc(value: str) -> datetime | None:
    """Parse an ISO-8601 timestamp; treat naive values as UTC."""
    try:
        parsed = datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


async def stamp_channel_marker(
    client: Any,
    thread_id: str,
    *,
    marker: dict[str, Any] | None = None,
) -> None:
    """Stamp the channel-origin marker into ``thread_id``'s server metadata.

    Write-side companion to :func:`list_channel_thread_candidates`:
    ``remember_channel_origin`` records the origin only in serve's in-process dict,
    invisible to the cron/server process. Serve calls this (when it holds a server
    client) so the thread becomes discoverable by the eligibility search.
    ``client.threads.update`` merges metadata server-side, so the marker does not
    clobber ``agent_name``/``graph_id``/etc.
    """
    update = client.threads.update(
        thread_id, metadata=marker if marker is not None else CHANNEL_ORIGIN_MARKER
    )
    if isinstance(update, Awaitable):
        await update
