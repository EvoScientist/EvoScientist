"""A project's crons, read and changed inside the server through the
runtime's own store.

A cron belongs to the project of the workspace assistant it runs on. User
crons (scheduled tasks) are the user's; the AutoSkills and proactive crons are
the system's, recreated by their reconciles when the project is used again.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..cron.schedule import SCHEDULED_RUN_KIND, belongs_to, cron_request, owned
from ..memory.autoskills.schedule import (
    AUTOSKILL_GRAPH_ID,
    AUTOSKILL_RUN_KIND,
    plan_autoskill_schedule,
)
from ..memory.types import MEMORY_AGENT_RECURSION_LIMIT
from ..paths import Workspace
from ..proactive.cron import PROACTIVE_RUN_KIND

logger = logging.getLogger(__name__)

SYSTEM_RUN_KINDS = frozenset({AUTOSKILL_RUN_KIND, PROACTIVE_RUN_KIND})

# Every cron the store holds fits in one search.
_SEARCH_LIMIT = 100_000


def metadata_of(row: Any) -> dict[str, Any]:
    """The metadata of a stored *row* (a cron, a thread), or an empty dict
    when it has none, or the row is malformed."""
    metadata = row.get("metadata") if isinstance(row, dict) else None
    return metadata if isinstance(metadata, dict) else {}


def _run_kind(cron: Any) -> str | None:
    run_kind = metadata_of(cron).get("run_kind")
    return run_kind if isinstance(run_kind, str) else None


def is_system_cron(cron: Any) -> bool:
    """Whether *cron* is an AutoSkills or proactive cron."""
    return _run_kind(cron) in SYSTEM_RUN_KINDS


async def project_crons(workspace: Workspace) -> list[dict[str, Any]]:
    """The crons that run on one of *workspace*'s assistants."""
    from langgraph_runtime.database import connect
    from langgraph_runtime.ops import Crons

    async with connect() as conn:
        rows, _count = await Crons.search(
            conn,
            assistant_id=None,
            thread_id=None,
            enabled=None,
            limit=_SEARCH_LIMIT,
            offset=0,
        )
        return owned([cron async for cron in rows], workspace)


def enabled_user_crons(crons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The enabled scheduled tasks among *crons*."""
    return [
        cron
        for cron in crons
        if cron.get("enabled") and _run_kind(cron) == SCHEDULED_RUN_KIND
    ]


def system_crons(crons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The AutoSkills and proactive crons among *crons*."""
    return [cron for cron in crons if is_system_cron(cron)]


async def _delete(conn: Any, cron_id: Any) -> None:
    from langgraph_runtime.ops import Crons

    async for _deleted in await Crons.delete(conn, cron_id):
        pass


async def delete_crons(cron_ids: list[Any]) -> None:
    from langgraph_runtime.database import connect

    async with connect() as conn:
        for cron_id in cron_ids:
            await _delete(conn, cron_id)


async def drop_system_cron(cron_id: str, workspace: Workspace) -> None:
    """Delete *cron_id* if it is one of *workspace*'s system crons, which fired
    into its missing folder; its reconcile recreates it.

    User crons are left as they are: each fire is refused until the folder is
    back. A task that deletes its run's thread leaves nothing in the store; one
    that keeps it (``on_run_completed="keep"``, as the WebUI creates them)
    leaves that thread and its failed run, until the server restarts.
    """
    from langgraph_runtime.database import connect
    from langgraph_runtime.ops import Crons

    async with connect() as conn:
        found = [cron async for cron in await Crons.get(conn, cron_id)]
        if not found:
            return
        (cron,) = found
        if belongs_to(cron, workspace) and is_system_cron(cron):
            await _delete(conn, cron["cron_id"])


async def reconcile_autoskills(config: Any, workspaces: list[Workspace]) -> list[str]:
    """Make the AutoSkills cron of each of *workspaces* match *config*, in
    process (the server never calls itself over HTTP); see
    :func:`~EvoScientist.memory.autoskills.schedule.plan_autoskill_schedule`.
    Returns why each one that failed did; a failure is logged, and the others
    go on."""
    from langgraph_runtime.database import connect
    from langgraph_runtime.ops import Crons

    failed: list[str] = []
    async with connect() as conn:
        rows, _count = await Crons.search(
            conn,
            assistant_id=None,
            thread_id=None,
            enabled=None,
            limit=_SEARCH_LIMIT,
            offset=0,
            metadata={"run_kind": AUTOSKILL_RUN_KIND},
        )
        autoskills = [cron async for cron in rows]
        for workspace in workspaces:
            try:
                existing = owned(autoskills, workspace)
                plan = await asyncio.to_thread(
                    plan_autoskill_schedule, config, workspace, existing
                )
                if plan.create is not None:
                    await _put(conn, workspace, plan.create)
                for cron in plan.delete:
                    await _delete(conn, cron["cron_id"])
            except Exception as exc:
                logger.warning(
                    "Could not reconcile the AutoSkills schedule of %s.",
                    workspace.root,
                    exc_info=True,
                )
                failed.append(f"{workspace.root}: {exc}")
    return failed


def request_payload(assistant_id: str, fields: dict[str, Any]) -> dict[str, Any]:
    """The payload the HTTP route stores for a cron of *fields* (from
    :func:`~EvoScientist.cron.schedule.cron_request`): its request body."""
    return {
        "assistant_id": assistant_id,
        **{key: value for key, value in fields.items() if value is not None},
    }


async def _put(conn: Any, workspace: Workspace, cron: Any) -> None:
    """Create *cron* as the HTTP route would: its payload is the request
    body."""
    from langgraph_runtime.ops import Crons

    assistant_id, fields = cron_request(
        workspace=workspace,
        graph_id=AUTOSKILL_GRAPH_ID,
        schedule=cron.schedule,
        input=cron.input,
        metadata=cron.metadata,
        timezone=cron.timezone,
        recursion_limit=MEMORY_AGENT_RECURSION_LIMIT,
    )
    payload = request_payload(assistant_id, fields)
    async for _created in await Crons.put(
        conn,
        payload=payload,
        schedule=cron.schedule,
        metadata=fields["metadata"],
        enabled=True,
        on_run_completed="delete",
        timezone=cron.timezone,
    ):
        pass
