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

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langgraph_sdk.schema import Cron, Run

from ..langgraph_dev.sdk import (
    configured_langgraph_dev_url,
    default_scheduler_timezone,
    get_langgraph_sync_client,
    messages_input,
)

SCHEDULER_GRAPH_ID = "scheduler"
SCHEDULED_RUN_KIND = "scheduled_task"


def _normalize_rubric(rubric: str | None) -> str | None:
    text = (rubric or "").strip()
    return text or None


def _scheduled_input(prompt: str, rubric: str | None) -> dict[str, Any]:
    """Run input for the scheduler graph; ``rubric`` rides along only when set.

    The key is read by ``RubricMiddleware`` mounted on the scheduler graph — an
    absent key means no grading pass at all, so unset stays byte-identical to
    the pre-rubric payload.
    """
    payload: dict[str, Any] = messages_input(prompt)
    if rubric:
        payload["rubric"] = rubric
    return payload


def _scheduled_metadata(
    *, name: str, prompt: str, rubric: str | None
) -> dict[str, str]:
    metadata = {"run_kind": SCHEDULED_RUN_KIND, "name": name, "prompt": prompt}
    if rubric:
        metadata["rubric"] = rubric
    return metadata


def _scheduler_url() -> str:
    return configured_langgraph_dev_url()


def _client():
    return get_langgraph_sync_client(url=_scheduler_url())


def _default_timezone() -> str | None:
    return default_scheduler_timezone()


def is_available() -> bool:
    """True when the langgraph dev backend (which fires crons) is reachable."""
    from ..langgraph_dev.manager import is_langgraph_dev_running

    return bool(is_langgraph_dev_running(base_url=_scheduler_url()))


def create_schedule(
    *,
    name: str,
    schedule: str,
    prompt: str,
    timezone: str | None = None,
    rubric: str | None = None,
) -> Cron:
    """Create a recurring scheduled task on the scheduler graph.

    ``rubric`` is an optional acceptance checklist graded after each run; blank
    means the run is never graded.
    """
    rubric = _normalize_rubric(rubric)
    # Crons are stored in the langgraph-dev process's .langgraph_api store, not
    # tagged by workspace. Isolation is process-level (see module docstring).
    return _client().crons.create(
        assistant_id=SCHEDULER_GRAPH_ID,
        schedule=schedule,
        input=_scheduled_input(prompt, rubric),
        metadata=_scheduled_metadata(name=name, prompt=prompt, rubric=rubric),
        timezone=timezone or _default_timezone(),
    )


def list_schedules() -> list[Cron]:
    """Return only EvoScientist scheduled tasks.

    Filtered server-side by ``run_kind`` metadata (the cron backend matches by
    metadata containment), so we never page through unrelated crons; ``limit`` is
    a ceiling on OUR schedules (far below 1000 in practice). We filter on metadata
    rather than ``assistant_id`` because the stored ``assistant_id`` is a resolved
    UUID, not the ``scheduler`` graph name we create with.
    """
    return _client().crons.search(
        metadata={"run_kind": SCHEDULED_RUN_KIND},
        limit=1000,
    )


def delete_schedule(cron_id: str) -> None:
    """Delete a scheduled task by cron id."""
    _client().crons.delete(cron_id)


def set_enabled(cron_id: str, enabled: bool) -> Cron:
    """Enable or disable a scheduled task by cron id."""
    return _client().crons.update(cron_id, enabled=enabled)


def run_now(prompt: str, *, rubric: str | None = None) -> Run:
    """Fire a one-off scheduler run immediately (for ``/schedule run``).

    Output goes wherever the task's prompt specifies; there is no push notification.
    """
    rubric = _normalize_rubric(rubric)
    client = _client()
    thread = client.threads.create(graph_id=SCHEDULER_GRAPH_ID)
    return client.runs.create(
        thread_id=str(thread["thread_id"]),
        assistant_id=SCHEDULER_GRAPH_ID,
        input=_scheduled_input(prompt, rubric),
        metadata=_scheduled_metadata(name="manual-run", prompt=prompt, rubric=rubric),
    )
