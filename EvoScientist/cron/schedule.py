"""Thin wrapper over the langgraph dev built-in cron API (langgraph_sdk).

EvoScientist scheduled tasks ARE langgraph crons targeting the ``scheduler``
graph. This module is the single choke-point so the ``/schedule`` command and the
NL ``schedule_task`` tool share one implementation.

Isolation is **process-level**, not data-level: EvoScientist's manager.py restarts
langgraph dev when the active workspace changes, so each workspace gets its own
langgraph-dev process and its own ``.langgraph_api`` cron store. If you point
multiple clients at one hand-started server they will share the same cron store.
"""

from __future__ import annotations

from typing import Any

SCHEDULER_GRAPH_ID = "scheduler"
SCHEDULED_RUN_KIND = "scheduled_task"


def _scheduler_url() -> str:
    from ..EvoScientist import _ensure_config

    cfg = _ensure_config()
    port = int(getattr(cfg, "langgraph_dev_port", 6174))
    return f"http://localhost:{port}"


def _client():
    from langgraph_sdk import get_sync_client

    return get_sync_client(url=_scheduler_url(), headers={"x-auth-scheme": "langsmith"})


def _default_timezone() -> str | None:
    from ..EvoScientist import _ensure_config

    tz = getattr(_ensure_config(), "scheduler_default_timezone", "") or ""
    if tz:
        return tz
    # Resolve the host's real IANA zone (e.g. "Asia/Shanghai") so absolute-time
    # schedules fire in local time and track DST. Falls back to None (-> UTC in
    # the cron backend) when the local zone can't be determined.
    try:
        from tzlocal import get_localzone_name

        return get_localzone_name()
    except Exception:
        return None


def _as_dict(rec: Any) -> dict[str, Any]:
    return rec if isinstance(rec, dict) else dict(rec)


def is_available() -> bool:
    """True when the langgraph dev backend (which fires crons) is reachable."""
    from ..langgraph_dev.manager import is_langgraph_dev_running

    return bool(is_langgraph_dev_running(base_url=_scheduler_url()))


def create_schedule(
    *, name: str, schedule: str, prompt: str, timezone: str | None = None
) -> dict[str, Any]:
    """Create a recurring scheduled task on the scheduler graph."""
    # Crons are stored in the langgraph-dev process's .langgraph_api store, not
    # tagged by workspace. Isolation is process-level (see module docstring).
    cron = _client().crons.create(
        assistant_id=SCHEDULER_GRAPH_ID,
        schedule=schedule,
        input={"messages": [{"role": "user", "content": prompt}]},
        metadata={"run_kind": SCHEDULED_RUN_KIND, "name": name, "prompt": prompt},
        timezone=timezone or _default_timezone(),
    )
    return _as_dict(cron)


def list_schedules() -> list[dict[str, Any]]:
    """Return only EvoScientist scheduled tasks (filtered by run_kind metadata)."""
    out = _client().crons.search(limit=1000)
    recs = [_as_dict(c) for c in out]
    return [
        c
        for c in recs
        if (c.get("metadata") or {}).get("run_kind") == SCHEDULED_RUN_KIND
    ]


def delete_schedule(cron_id: str) -> None:
    """Delete a scheduled task by cron id."""
    _client().crons.delete(cron_id)


def set_enabled(cron_id: str, enabled: bool) -> dict[str, Any]:
    """Enable or disable a scheduled task by cron id."""
    return _as_dict(_client().crons.update(cron_id, enabled=enabled))


def run_now(prompt: str) -> dict[str, Any]:
    """Fire a one-off scheduler run immediately (for ``/schedule run``).

    Output goes wherever the task's prompt specifies; there is no push notification.
    """
    client = _client()
    thread = _as_dict(client.threads.create(graph_id=SCHEDULER_GRAPH_ID))
    run = client.runs.create(
        thread_id=str(thread["thread_id"]),
        assistant_id=SCHEDULER_GRAPH_ID,
        input={"messages": [{"role": "user", "content": prompt}]},
        metadata={
            "run_kind": SCHEDULED_RUN_KIND,
            "name": "manual-run",
            "prompt": prompt,
        },
    )
    return _as_dict(run)
