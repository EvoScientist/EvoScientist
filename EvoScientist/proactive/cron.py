"""The proactive cron: one per workspace, bound to its own thread.

The cron runs the ``proactive`` graph every 10 minutes on a dedicated thread,
so ticks share one state (the decision record) and, with
``multitask_strategy="reject"``, never overlap. It goes through
:func:`EvoScientist.cron.schedule.create_cron`, so it carries the workspace tag
and run config like every other cron. It is a system cron: it has its own
``run_kind``, so ``/schedule`` never lists it.
"""

from __future__ import annotations

from typing import Any

from ..cron.schedule import create_cron, owned, search_all
from ..paths import SessionDirs, Workspace

PROACTIVE_GRAPH_ID = "proactive"
PROACTIVE_RUN_KIND = "proactive_check"
PROACTIVE_SCHEDULE = "*/10 * * * *"
# Reject a fire while the previous tick still runs. Never "rollback": the
# custom checkpointer cannot delete a run's checkpoints.
PROACTIVE_MULTITASK_STRATEGY = "reject"
# One checkpoint per tick instead of one per step.
PROACTIVE_DURABILITY = "exit"


def cron_input(thread_ids: list[str]) -> dict[str, Any]:
    """The cron's run input: the threads serve can deliver to.

    This is the whole input, never a patch: ``crons.update`` replaces it.
    """
    return {"candidates": [{"thread_id": tid} for tid in thread_ids]}


def list_proactive_crons(client: Any, workspace: Workspace) -> list[Any]:
    """*workspace*'s proactive crons (normally zero or one)."""
    rows = search_all(client, metadata={"run_kind": PROACTIVE_RUN_KIND})
    return owned(rows, workspace)


def create_proactive_thread(client: Any, workspace: Workspace) -> str:
    """Create the thread a proactive cron is bound to, and return its id."""
    thread = client.threads.create(
        metadata={
            "run_kind": PROACTIVE_RUN_KIND,
            # Set before the first run, so state can be written to the thread.
            "graph_id": PROACTIVE_GRAPH_ID,
            **SessionDirs(workspace).metadata(),
        }
    )
    return str(thread["thread_id"])


def create_proactive_cron(
    client: Any,
    workspace: Workspace,
    *,
    thread_id: str,
    candidates: list[str],
    timezone: str | None,
) -> Any:
    """Create *workspace*'s proactive cron on *thread_id* (which must exist)."""
    return create_cron(
        client,
        workspace=workspace,
        assistant_id=PROACTIVE_GRAPH_ID,
        schedule=PROACTIVE_SCHEDULE,
        input=cron_input(candidates),
        metadata={"run_kind": PROACTIVE_RUN_KIND},
        timezone=timezone,
        thread_id=thread_id,
        multitask_strategy=PROACTIVE_MULTITASK_STRATEGY,
        durability=PROACTIVE_DURABILITY,
    )
