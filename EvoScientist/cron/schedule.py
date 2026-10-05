"""Thin wrapper over the langgraph dev built-in cron API (langgraph_sdk).

EvoScientist scheduled tasks ARE langgraph crons targeting the ``scheduler``
graph. This module is the single choke-point so the ``/schedule`` command and the
NL ``schedule_task`` tool share one implementation, and every cron (AutoSkills'
too) is created with :func:`create_cron`, which tags it, and deleted with
:func:`delete_cron`, which checks it is the workspace's own.

Every cron belongs to a workspace: it carries the workspace root in
``config.configurable`` and ``metadata`` (``workspace_dir``), runs in that root,
and is listed, paused and deleted only from that workspace. Crons nobody else
can own (untagged, or tagged with a folder that no longer exists) belong to
whichever workspace the server serves.
"""

from __future__ import annotations

from pathlib import Path
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
    *, name: str, prompt: str, rubric: str | None
) -> dict[str, str]:
    metadata = {
        "run_kind": SCHEDULED_RUN_KIND,
        "name": name,
        "prompt": prompt,
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
    unresolved or Windows path) still match. The server only loads the store
    of the workspace it serves, so a cron nobody else can own belongs to it:
    one created before tasks were tagged, or one tagged with a folder that no
    longer exists (the project was moved or renamed, and its store with it).
    """
    tag = ((cron or {}).get("metadata") or {}).get("workspace_dir")
    if not isinstance(tag, str) or not tag:
        return True
    key = workspace_key(tag)
    if key == workspace.key:
        return True
    return key is not None and not Path(key).is_dir()


def adopt_stale_tasks(client: Any, workspace: Workspace, *, run_kind: str) -> int:
    """Tag *workspace*'s crons of *run_kind* that carry no live tag.

    Crons from before tasks were tagged, and crons tagged with a folder that
    no longer exists, get this workspace's tag and run config, so they list,
    pause and delete here and their runs pass the server's folder check.
    Returns how many were re-tagged.
    """
    adopted = 0
    for cron in search_all(client, metadata={"run_kind": run_kind}):
        if not belongs_to(cron, workspace):
            continue
        tag = (cron.get("metadata") or {}).get("workspace_dir")
        if isinstance(tag, str) and workspace_key(tag) == workspace.key:
            continue
        client.crons.update(
            str(cron["cron_id"]),
            metadata={"workspace_dir": workspace.key},
            config=run_config(workspace),
        )
        adopted += 1
    return adopted


# Crons per search page (the server's maximum). Workspaces are filtered after
# the search, so every page is read: a store full of other workspaces' crons
# cannot hide ours. Pages are newest first, so a cron created between two
# fetches shifts the rest down by one; rows are therefore collected by id.
_SEARCH_PAGE = 1000


def _extend_unseen(rows: list[Cron], seen: set[str], page: list[Cron]) -> None:
    for cron in page:
        cron_id = str(cron.get("cron_id"))
        if cron_id not in seen:
            seen.add(cron_id)
            rows.append(cron)


def search_all(client: Any, *, metadata: dict[str, Any]) -> list[Cron]:
    """Every cron whose metadata contains *metadata*, page by page."""
    rows: list[Cron] = []
    seen: set[str] = set()
    offset = 0
    while True:
        page = client.crons.search(metadata=metadata, limit=_SEARCH_PAGE, offset=offset)
        _extend_unseen(rows, seen, page)
        offset += len(page)
        if len(page) < _SEARCH_PAGE:
            return rows


async def asearch_all(client: Any, *, metadata: dict[str, Any]) -> list[Cron]:
    """Async variant of :func:`search_all`."""
    rows: list[Cron] = []
    seen: set[str] = set()
    offset = 0
    while True:
        page = await client.crons.search(
            metadata=metadata, limit=_SEARCH_PAGE, offset=offset
        )
        _extend_unseen(rows, seen, page)
        offset += len(page)
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


def create_cron(
    client: Any,
    *,
    workspace: Workspace,
    assistant_id: str,
    schedule: str,
    input: dict[str, Any],
    metadata: dict[str, str],
    timezone: str | None,
) -> Cron:
    """Create a cron of *workspace*: tagged with it, and its runs work in it."""
    return client.crons.create(
        assistant_id=assistant_id,
        schedule=schedule,
        input=input,
        metadata={**metadata, "workspace_dir": workspace.key},
        config=run_config(workspace),
        timezone=timezone,
    )


def _require_own(cron: Cron, workspace: Workspace) -> str:
    if not belongs_to(cron, workspace):
        raise LookupError(f"No scheduled task {cron.get('cron_id')} in this workspace.")
    return str(cron["cron_id"])


def delete_cron(client: Any, cron: Cron, *, workspace: Workspace) -> None:
    """Delete *cron*, one of *workspace*'s crons."""
    client.crons.delete(_require_own(cron, workspace))


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
    return create_cron(
        _client(),
        workspace=workspace,
        assistant_id=SCHEDULER_GRAPH_ID,
        schedule=schedule,
        input=_scheduled_input(prompt, rubric),
        metadata=_scheduled_metadata(name=name, prompt=prompt, rubric=rubric),
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


def delete_schedule(cron: Cron, *, workspace: Workspace) -> None:
    """Delete *cron*, one of *workspace*'s scheduled tasks."""
    delete_cron(_client(), cron, workspace=workspace)


def set_enabled(cron: Cron, enabled: bool, *, workspace: Workspace) -> Cron:
    """Enable or disable *cron*, one of *workspace*'s scheduled tasks."""
    return _client().crons.update(_require_own(cron, workspace), enabled=enabled)


def run_now(prompt: str, *, workspace: Workspace, rubric: str | None = None) -> Run:
    """Fire a one-off scheduler run of *workspace* now (for ``/schedule run``).

    Output goes wherever the task's prompt specifies; there is no push notification.
    """
    rubric = _normalize_rubric(rubric)
    client = _client()
    folders = SessionDirs(workspace).metadata()
    thread = client.threads.create(
        graph_id=SCHEDULER_GRAPH_ID,
        metadata={"run_kind": SCHEDULED_RUN_KIND, **folders},
    )
    return client.runs.create(
        thread_id=str(thread["thread_id"]),
        assistant_id=SCHEDULER_GRAPH_ID,
        input=_scheduled_input(prompt, rubric),
        metadata={
            **_scheduled_metadata(name="manual-run", prompt=prompt, rubric=rubric),
            **folders,
        },
        config=run_config(workspace),
    )
