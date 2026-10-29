"""Thin wrapper over the langgraph dev built-in cron API (langgraph_sdk).

EvoScientist scheduled tasks ARE langgraph crons targeting the ``scheduler``
graph. This module is the single choke-point so the ``/schedule`` command and the
NL ``schedule_task`` tool share one implementation: a client creates a cron with
:func:`create_cron`, which tags it, and deletes it with :func:`delete_cron`,
which checks it is the workspace's own. Every cron is built by
:func:`cron_request`, also inside the server, which reconciles the AutoSkills
crons in process (``langgraph_dev.server_crons``).

Every cron belongs to the workspace whose assistant it runs on, and is listed,
paused and deleted only from that workspace. It also carries the workspace root
in ``config.configurable`` and ``metadata`` (``workspace_dir``), for display.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langgraph_sdk.schema import Cron, Run

from ..langgraph_dev.assistants import launch_assistant_id, workspace_assistant_ids
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


def _runs_on(cron: Any, assistant_ids: frozenset[str]) -> bool:
    return str((cron or {}).get("assistant_id")) in assistant_ids


def belongs_to(cron: Any, workspace: Workspace) -> bool:
    """True if *cron* is one of *workspace*'s: it runs on one of *workspace*'s
    assistants. Its ``workspace_dir`` tag is only for display."""
    return _runs_on(cron, workspace_assistant_ids(workspace))


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
    """The crons of *rows* that belong to *workspace*."""
    assistant_ids = workspace_assistant_ids(workspace)
    return [row for row in rows if _runs_on(row, assistant_ids)]


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


def cron_request(
    *,
    workspace: Workspace,
    graph_id: str,
    schedule: str,
    input: dict[str, Any],
    metadata: dict[str, str],
    timezone: str | None,
    multitask_strategy: str | None = None,
    durability: str | None = None,
    recursion_limit: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """The assistant and the fields of a cron of *workspace* on its
    *graph_id* assistant: tagged with it, and its runs work in it, taking at
    most *recursion_limit* steps when given. Shared by :func:`create_cron` and
    the server, which creates crons in process."""
    config = run_config(workspace)
    if recursion_limit is not None:
        config["recursion_limit"] = recursion_limit
    kwargs: dict[str, Any] = {
        "schedule": schedule,
        "input": input,
        "metadata": {**metadata, "workspace_dir": workspace.key},
        "config": config,
        "timezone": timezone,
    }
    if multitask_strategy is not None:
        kwargs["multitask_strategy"] = multitask_strategy
    if durability is not None:
        kwargs["durability"] = durability
    return launch_assistant_id(graph_id, workspace), kwargs


def create_cron(
    client: Any,
    *,
    workspace: Workspace,
    graph_id: str,
    schedule: str,
    input: dict[str, Any],
    metadata: dict[str, str],
    timezone: str | None,
    thread_id: str | None = None,
    multitask_strategy: str | None = None,
    durability: str | None = None,
) -> Cron:
    """Create a cron of *workspace* on its *graph_id* assistant: tagged with
    it, and its runs work in it.

    Without *thread_id* every run gets a fresh thread, which the runtime
    deletes after the run. With it the cron is bound to that existing thread,
    so its runs share one state; *multitask_strategy* (``"reject"`` keeps runs
    from overlapping) applies only then. *durability* sets how often a run
    checkpoints (``"exit"``: once, at the end).
    """
    assistant_id, kwargs = cron_request(
        workspace=workspace,
        graph_id=graph_id,
        schedule=schedule,
        input=input,
        metadata=metadata,
        timezone=timezone,
        multitask_strategy=multitask_strategy,
        durability=durability,
    )
    if thread_id is not None:
        return client.crons.create_for_thread(thread_id, assistant_id, **kwargs)
    return client.crons.create(assistant_id=assistant_id, **kwargs)


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
        graph_id=SCHEDULER_GRAPH_ID,
        schedule=schedule,
        input=_scheduled_input(prompt, rubric),
        metadata=_scheduled_metadata(name=name, prompt=prompt, rubric=rubric),
        timezone=timezone or _default_timezone(),
    )


def list_schedules(*, workspace: Workspace) -> list[Cron]:
    """Return *workspace*'s EvoScientist scheduled tasks.

    Filtered server-side by ``run_kind`` metadata (the cron backend matches by
    metadata containment), so other kinds of crons are never read; the
    project here, by the cron's assistant (:func:`belongs_to`), since a
    search takes one assistant and a project has one per graph.
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
        assistant_id=launch_assistant_id(SCHEDULER_GRAPH_ID, workspace),
        input=_scheduled_input(prompt, rubric),
        metadata={
            **_scheduled_metadata(name="manual-run", prompt=prompt, rubric=rubric),
            **folders,
        },
        config=run_config(workspace),
    )
