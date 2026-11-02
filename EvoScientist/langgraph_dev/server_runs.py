"""Interrupt the runs belonging to an unregistered project."""

from __future__ import annotations

import asyncio
from uuid import UUID

from starlette.exceptions import HTTPException

from ..paths import Workspace
from ..sessions import thread_binding
from .assistants import workspace_assistant_ids

_SEARCH_LIMIT = 100_000


async def cancel_project_runs(workspace: Workspace) -> None:
    """Interrupt workspace-assistant runs and runs on bound legacy threads.

    The project must already be unregistered so newly started runs are refused.
    A run can use a graph's default assistant on a previously bound thread, so
    cancelling workspace assistants alone does not cover every project run.
    """
    from langgraph_runtime.database import connect
    from langgraph_runtime.ops import Runs, Threads

    async with connect() as conn:
        bound_threads = []
        rows, cursor = await Threads.search(
            conn, metadata=None, values=None, status=None, limit=_SEARCH_LIMIT, offset=0
        )
        if cursor is not None:
            raise RuntimeError("Too many threads to safely unregister the project")
        for thread in [row async for row in rows]:
            thread_id = thread["thread_id"]
            binding = await asyncio.to_thread(thread_binding, str(thread_id))
            if binding is not None and binding.workspace == workspace:
                bound_threads.append(thread_id)

        for assistant_id in workspace_assistant_ids(workspace):
            await Runs.cancel(conn, assistant_id=UUID(assistant_id))

        for thread_id in bound_threads:
            run_ids = []
            for status in ("pending", "running"):
                rows = await Runs.search(
                    conn, thread_id, status=status, limit=_SEARCH_LIMIT, offset=0
                )
                page = [row async for row in rows]
                if len(page) == _SEARCH_LIMIT:
                    raise RuntimeError("Too many runs to safely unregister the project")
                run_ids.extend(row["run_id"] for row in page)
            if run_ids:
                try:
                    await Runs.cancel(conn, run_ids, thread_id=thread_id)
                except HTTPException as exc:
                    if exc.status_code != 404:
                        raise
