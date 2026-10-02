"""Thin wrapper over the langgraph dev built-in cron API (langgraph_sdk).

EvoScientist scheduled tasks ARE langgraph crons targeting the ``scheduler``
graph. This module is the single choke-point so the ``/schedule`` command and the
NL ``schedule_task`` tool share one implementation.

Every scheduled task belongs to a workspace: it carries the workspace root in
``config.configurable`` and ``metadata`` (``workspace_dir``), runs in that root,
and is listed, paused and deleted only from that workspace. Tasks created before
they were tagged belong to whichever workspace the server serves.
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
from ..paths import SessionDirs, Workspace

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
    *, name: str, prompt: str, rubric: str | None, workspace: Workspace
) -> dict[str, str]:
    metadata = {
        "run_kind": SCHEDULED_RUN_KIND,
        "name": name,
        "prompt": prompt,
        "workspace_dir": workspace.key,
    }
    if rubric:
        metadata["rubric"] = rubric
    return metadata


def run_config(workspace: Workspace, **configurable: Any) -> dict[str, Any]:
    """Run config of a scheduled run: it works in the workspace root."""
    return {"configurable": {**configurable, **SessionDirs(workspace).metadata()}}


def workspace_key(value: Any) -> str | None:
    """The workspace a stored folder names, in the stored form, or None."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return Workspace(value).key
    except (OSError, RuntimeError, ValueError):
        # A path that cannot exist on this machine names no workspace here.
        return None


def belongs_to(cron: Any, workspace: Workspace) -> bool:
    """True if *cron* is one of *workspace*'s scheduled tasks.

    Tags are compared in the stored form, so tags written before it (an
    unresolved or Windows path) still match. Untagged crons were created
    before tasks were tagged; the server only loads the store of the
    workspace it serves, so they belong to it.
    """
    tag = ((cron or {}).get("metadata") or {}).get("workspace_dir")
    if not isinstance(tag, str) or not tag:
        return True
    return workspace_key(tag) == workspace.key


# Crons per search page. Workspaces are filtered after the search, so every
# page is read: a store full of other workspaces' crons cannot hide ours.
_SEARCH_PAGE = 1000


def search_all(client: Any, *, metadata: dict[str, Any]) -> list[Cron]:
    """Every cron whose metadata contains *metadata*, page by page."""
    rows: list[Cron] = []
    while True:
        page = client.crons.search(
            metadata=metadata, limit=_SEARCH_PAGE, offset=len(rows)
        )
        rows.extend(page)
        if len(page) < _SEARCH_PAGE:
            return rows


async def asearch_all(client: Any, *, metadata: dict[str, Any]) -> list[Cron]:
    """Async variant of :func:`search_all`."""
    rows: list[Cron] = []
    while True:
        page = await client.crons.search(
            metadata=metadata, limit=_SEARCH_PAGE, offset=len(rows)
        )
        rows.extend(page)
        if len(page) < _SEARCH_PAGE:
            return rows


def owned(rows: list[Cron], workspace: Workspace) -> list[Cron]:
    """The crons of *rows* that belong to *workspace* (resolves paths)."""
    return [row for row in rows if belongs_to(row, workspace)]


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
    workspace: Workspace,
) -> Cron:
    """Create a recurring scheduled task of *workspace* on the scheduler graph.

    ``rubric`` is an optional acceptance checklist graded after each run; blank
    means the run is never graded.
    """
    rubric = _normalize_rubric(rubric)
    return _client().crons.create(
        assistant_id=SCHEDULER_GRAPH_ID,
        schedule=schedule,
        input=_scheduled_input(prompt, rubric),
        metadata=_scheduled_metadata(
            name=name, prompt=prompt, rubric=rubric, workspace=workspace
        ),
        config=run_config(workspace),
        timezone=timezone or _default_timezone(),
    )


def list_schedules(*, workspace: Workspace) -> list[Cron]:
    """Return *workspace*'s EvoScientist scheduled tasks.

    Filtered server-side by ``run_kind`` metadata (the cron backend matches by
    metadata containment), so other kinds of crons are never read. We filter
    on metadata rather than ``assistant_id`` because the stored ``assistant_id``
    is a resolved UUID, not the ``scheduler`` graph name we create with. The
    workspace is filtered here rather than in the search, which matches values
    exactly: see :func:`belongs_to`.
    """
    rows = search_all(_client(), metadata={"run_kind": SCHEDULED_RUN_KIND})
    return owned(rows, workspace)


def _require_own(cron: Cron, workspace: Workspace) -> str:
    if not belongs_to(cron, workspace):
        raise LookupError(f"No scheduled task {cron.get('cron_id')} in this workspace.")
    return str(cron["cron_id"])


def delete_schedule(cron: Cron, *, workspace: Workspace) -> None:
    """Delete *cron*, one of *workspace*'s scheduled tasks."""
    _client().crons.delete(_require_own(cron, workspace))


def set_enabled(cron: Cron, enabled: bool, *, workspace: Workspace) -> Cron:
    """Enable or disable *cron*, one of *workspace*'s scheduled tasks."""
    return _client().crons.update(_require_own(cron, workspace), enabled=enabled)


def run_now(prompt: str, *, workspace: Workspace, rubric: str | None = None) -> Run:
    """Fire a one-off scheduler run of *workspace* now (for ``/schedule run``).

    Output goes wherever the task's prompt specifies; there is no push notification.
    """
    rubric = _normalize_rubric(rubric)
    client = _client()
    thread = client.threads.create(
        graph_id=SCHEDULER_GRAPH_ID,
        metadata={"run_kind": SCHEDULED_RUN_KIND, **SessionDirs(workspace).metadata()},
    )
    return client.runs.create(
        thread_id=str(thread["thread_id"]),
        assistant_id=SCHEDULER_GRAPH_ID,
        input=_scheduled_input(prompt, rubric),
        metadata=_scheduled_metadata(
            name="manual-run", prompt=prompt, rubric=rubric, workspace=workspace
        ),
        config=run_config(workspace),
    )
