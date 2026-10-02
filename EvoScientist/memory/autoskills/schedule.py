"""LangGraph scheduling helpers for EvoMemory AutoSkills."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ...config import EvoScientistConfig
from ...cron.schedule import asearch_all, owned, run_config, search_all, workspace_key
from ...langgraph_dev.sdk import (
    default_scheduler_timezone,
    get_langgraph_async_client,
    get_langgraph_sync_client,
    langgraph_dev_url,
    messages_input,
)
from ...paths import Workspace

AUTOSKILL_GRAPH_ID = "evomemory-autoskills"
AUTOSKILL_RUN_KIND = "evomemory_autoskills"


def autoskill_cron(cadence: str, time_hhmm: str) -> str:
    """Translate public cadence settings to a 5-field cron expression."""
    hour_text, minute_text = time_hhmm.split(":", 1)
    hour = int(hour_text)
    minute = int(minute_text)
    cadence_value = str(getattr(cadence, "value", cadence)).strip().lower()
    if cadence_value == "nightly":
        return f"{minute} {hour} * * *"
    if cadence_value == "weekly":
        return f"{minute} {hour} * * 0"
    if cadence_value == "monthly":
        return f"{minute} {hour} 1 * *"
    raise ValueError(f"Unsupported AutoSkills cadence: {cadence!r}")


def _autoskill_input() -> dict[str, Any]:
    return messages_input(
        "Run EvoMemory AutoSkills maintenance. Inspect candidate "
        "observation clusters, then propose at most a small number "
        "of high-confidence skills."
    )


def _autoskill_metadata(
    *,
    config: EvoScientistConfig,
    workspace_dir: str | Path,
    schedule: str,
) -> dict[str, str]:
    return {
        "run_kind": AUTOSKILL_RUN_KIND,
        "name": "EvoMemory AutoSkills",
        "workspace_dir": Workspace(workspace_dir).key,
        "mode": config.memory_skill_synthesis_mode.value,
        "cadence": config.memory_skill_synthesis_cadence.value,
        "time": config.memory_skill_synthesis_time,
        "schedule": schedule,
    }


def _owned_by(
    rows: list[dict[str, Any]], workspace_dir: str | Path
) -> list[dict[str, Any]]:
    return owned(rows, Workspace(workspace_dir))


def list_autoskill_schedules(
    config: EvoScientistConfig,
    *,
    workspace_dir: str | Path,
) -> list[dict[str, Any]]:
    """Return the workspace's internal AutoSkills cron records."""
    rows = search_all(
        get_langgraph_sync_client(url=langgraph_dev_url(config)),
        metadata={"run_kind": AUTOSKILL_RUN_KIND},
    )
    return _owned_by(rows, workspace_dir)


async def alist_autoskill_schedules(
    config: EvoScientistConfig,
    *,
    workspace_dir: str | Path,
) -> list[dict[str, Any]]:
    """Async variant of :func:`list_autoskill_schedules`."""
    rows = await asearch_all(
        get_langgraph_async_client(url=langgraph_dev_url(config)),
        metadata={"run_kind": AUTOSKILL_RUN_KIND},
    )
    return await asyncio.to_thread(_owned_by, rows, workspace_dir)


def _forwarded_workspace(cron: dict[str, Any]) -> str | None:
    """The workspace a cron's runs forward (its run config), in stored form."""
    payload = cron.get("payload") or {}
    configurable = (payload.get("config") or {}).get("configurable") or {}
    return workspace_key(configurable.get("workspace_dir"))


def reconcile_autoskill_schedule(
    config: EvoScientistConfig,
    *,
    workspace_dir: str | Path,
) -> dict[str, Any]:
    """Ensure the workspace's hidden AutoSkills cron matches config.

    Only this workspace's AutoSkills crons are listed, replaced or deleted.
    """
    from ...langgraph_dev.manager import is_langgraph_dev_running

    if not is_langgraph_dev_running(base_url=langgraph_dev_url(config)):
        return {"status": "unavailable"}

    client = get_langgraph_sync_client(url=langgraph_dev_url(config))
    existing = list_autoskill_schedules(config, workspace_dir=workspace_dir)
    if not config.memory_skill_synthesis_enabled:
        for row in existing:
            client.crons.delete(str(row["cron_id"]))
        return {"status": "disabled", "deleted": len(existing)}

    schedule = autoskill_cron(
        config.memory_skill_synthesis_cadence,
        config.memory_skill_synthesis_time,
    )
    metadata = _autoskill_metadata(
        config=config,
        workspace_dir=workspace_dir,
        schedule=schedule,
    )
    # A cron created before its runs forwarded the workspace is replaced, so
    # its runs carry ``configurable.workspace_dir`` too.
    matching = [
        row
        for row in existing
        if row.get("schedule") == schedule
        and bool(row.get("enabled", True))
        and (row.get("metadata") or {}).get("mode") == metadata["mode"]
        and _forwarded_workspace(row) == metadata["workspace_dir"]
    ]
    if len(matching) == 1 and len(existing) == 1:
        return {"status": "unchanged", "cron_id": matching[0].get("cron_id")}

    for row in existing:
        client.crons.delete(str(row["cron_id"]))
    created = client.crons.create(
        assistant_id=AUTOSKILL_GRAPH_ID,
        schedule=schedule,
        input=_autoskill_input(),
        metadata=metadata,
        config=run_config(Workspace(workspace_dir)),
        timezone=default_scheduler_timezone(config),
    )
    return {
        "status": "created",
        "cron_id": created.get("cron_id"),
        "schedule": schedule,
    }


def run_autoskill_now(
    config: EvoScientistConfig,
    *,
    workspace_dir: str | Path,
) -> dict[str, Any]:
    """Launch a one-off AutoSkills run immediately."""
    client = get_langgraph_sync_client(url=langgraph_dev_url(config))
    thread = client.threads.create(
        graph_id=AUTOSKILL_GRAPH_ID,
        metadata={
            "run_kind": AUTOSKILL_RUN_KIND,
            "workspace_dir": Workspace(workspace_dir).key,
        },
    )
    run = client.runs.create(
        thread_id=str(thread["thread_id"]),
        assistant_id=AUTOSKILL_GRAPH_ID,
        input=_autoskill_input(),
        metadata=_autoskill_metadata(
            config=config,
            workspace_dir=workspace_dir,
            schedule="manual",
        ),
        config=run_config(Workspace(workspace_dir), thread_id=str(thread["thread_id"])),
    )
    return {"thread_id": thread["thread_id"], "run_id": run["run_id"]}


async def arun_autoskill_now(
    config: EvoScientistConfig,
    *,
    workspace_dir: str | Path,
) -> dict[str, Any]:
    """Async variant of :func:`run_autoskill_now`."""
    client = get_langgraph_async_client(url=langgraph_dev_url(config))
    thread = await client.threads.create(
        graph_id=AUTOSKILL_GRAPH_ID,
        metadata={
            "run_kind": AUTOSKILL_RUN_KIND,
            "workspace_dir": Workspace(workspace_dir).key,
        },
    )
    run = await client.runs.create(
        thread_id=str(thread["thread_id"]),
        assistant_id=AUTOSKILL_GRAPH_ID,
        input=_autoskill_input(),
        metadata=_autoskill_metadata(
            config=config,
            workspace_dir=workspace_dir,
            schedule="manual",
        ),
        config=run_config(Workspace(workspace_dir), thread_id=str(thread["thread_id"])),
    )
    return {"thread_id": thread["thread_id"], "run_id": run["run_id"]}
