"""Proactive-check cron registration.

The proactive MVP "reuses cron, checks every 10 min". This is the thin wrapper —
sibling of ``EvoScientist/cron/schedule.py`` — that registers that periodic cron.
Unlike ``schedule.py`` (which targets the throwaway ``scheduler`` graph), this
targets the dedicated ``proactive`` graph (registered in ``langgraph.json``): a
periodic scan whose run enumerates server-registered channel-thread
candidates (``eligibility.list_channel_thread_candidates``) and runs a proactive
check on each. So there is no per-run message input — the graph enumerates threads
itself; ``proactive_mode`` is set by the shadow turns the check spawns, not baked
into the cron.

Like ``schedule.py``, isolation is process-level: crons live in the langgraph-dev
process's ``.langgraph_api`` store, so this reaches whichever dev server the active
workspace's config points at. ``is_available()`` gates on that server being up.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langgraph_sdk.schema import Cron

from ..langgraph_dev.sdk import (
    configured_langgraph_dev_url,
    default_scheduler_timezone,
    get_langgraph_sync_client,
)

# Registered in langgraph.json. Distinct from scheduler's graph so
# proactive crons never collide with user-scheduled tasks.
PROACTIVE_GRAPH_ID = "proactive"
PROACTIVE_RUN_KIND = "proactive_check"
# Metadata of the single server thread every proactive tick runs on. Pinning the
# cron to one thread lets the server serialize ticks: with
# ``multitask_strategy="reject"`` a fire that overlaps a still-running tick is
# refused, so two ticks can never check (and push to) the same thread at once.
PROACTIVE_THREAD_METADATA = {"run_kind": PROACTIVE_RUN_KIND}
PROACTIVE_MULTITASK_STRATEGY = "reject"
# Every 10 minutes (MVP cadence). Quiet-hours / idle are enforced per-thread by the
# gate inside the check, not by the cron schedule.
DEFAULT_PROACTIVE_SCHEDULE = "*/10 * * * *"


def _proactive_url() -> str:
    return configured_langgraph_dev_url()


def _client():
    return get_langgraph_sync_client(url=_proactive_url())


def _default_timezone() -> str | None:
    return default_scheduler_timezone()


def is_available() -> bool:
    """True when the langgraph dev backend (which fires crons) is reachable."""
    from ..langgraph_dev.manager import is_langgraph_dev_running

    return bool(is_langgraph_dev_running(base_url=_proactive_url()))


def ensure_proactive_thread(client: Any | None = None) -> str:
    """Return the id of the dedicated proactive tick thread, creating it once.

    Looked up by :data:`PROACTIVE_THREAD_METADATA` so serve restarts reuse it.
    """
    client = client or _client()
    found = client.threads.search(metadata=PROACTIVE_THREAD_METADATA, limit=1)
    for thread in found or []:
        thread_id = thread.get("thread_id") if isinstance(thread, dict) else None
        if thread_id:
            return str(thread_id)
    created = client.threads.create(metadata=PROACTIVE_THREAD_METADATA)
    return str(created["thread_id"])


def create_proactive_schedule(
    *, schedule: str = DEFAULT_PROACTIVE_SCHEDULE, timezone: str | None = None
) -> Cron:
    """Register the periodic proactive-check cron against the ``proactive`` graph.

    The cron is pinned to the dedicated tick thread with
    ``multitask_strategy="reject"`` so overlapping fires are serialized by the
    server (a tick that outlasts the cadence simply makes the next fire a no-op).
    Each fire is a bare tick, not a message turn: the graph enumerates eligible
    threads itself and ignores the (empty, non-null) input.
    """
    client = _client()
    return client.crons.create_for_thread(
        ensure_proactive_thread(client),
        PROACTIVE_GRAPH_ID,
        schedule=schedule,
        # Empty (not None): langgraph rejects a null run input (EmptyInputError).
        input={},
        metadata={"run_kind": PROACTIVE_RUN_KIND},
        multitask_strategy=PROACTIVE_MULTITASK_STRATEGY,
        timezone=timezone or _default_timezone(),
    )


def list_proactive_schedules() -> list[Cron]:
    """Return only proactive-check crons (server-side ``run_kind`` filter)."""
    return _client().crons.search(
        metadata={"run_kind": PROACTIVE_RUN_KIND},
        limit=1000,
    )


def delete_proactive_schedule(cron_id: str) -> None:
    """Delete a proactive-check cron by id."""
    _client().crons.delete(cron_id)


def set_proactive_enabled(cron_id: str, enabled: bool) -> Cron:
    """Enable or disable a proactive-check cron by id."""
    return _client().crons.update(cron_id, enabled=enabled)
