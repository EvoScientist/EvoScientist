"""LangGraph scheduling helpers for EvoMemory AutoSkills."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...config import EvoScientistConfig
from ...langgraph_dev.sdk import (
    default_scheduler_timezone,
    get_langgraph_async_client,
    get_langgraph_sync_client,
    langgraph_dev_url,
    messages_input,
)

AUTOSKILL_GRAPH_ID = "evomemory-autoskills"
AUTOSKILL_RUN_KIND = "evomemory_autoskills"
AUTOSKILL_SCHEDULE_SEARCH_LIMIT = 100


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
        "workspace_dir": str(Path(workspace_dir).expanduser().resolve()),
        "mode": config.memory_skill_synthesis_mode.value,
        "cadence": config.memory_skill_synthesis_cadence.value,
        "time": config.memory_skill_synthesis_time,
        "schedule": schedule,
    }


def list_autoskill_schedules(
    config: EvoScientistConfig,
    *,
    limit: int = AUTOSKILL_SCHEDULE_SEARCH_LIMIT,
) -> list[dict[str, Any]]:
    """Return internal AutoSkills cron records."""
    return list(
        get_langgraph_sync_client(url=langgraph_dev_url(config)).crons.search(
            metadata={"run_kind": AUTOSKILL_RUN_KIND},
            limit=limit,
        )
    )


async def alist_autoskill_schedules(
    config: EvoScientistConfig,
    *,
    limit: int = AUTOSKILL_SCHEDULE_SEARCH_LIMIT,
) -> list[dict[str, Any]]:
    """Async variant of :func:`list_autoskill_schedules`."""
    rows = await get_langgraph_async_client(url=langgraph_dev_url(config)).crons.search(
        metadata={"run_kind": AUTOSKILL_RUN_KIND},
        limit=limit,
    )
    return list(rows)


def reconcile_autoskill_schedule(
    config: EvoScientistConfig,
    *,
    workspace_dir: str | Path,
) -> dict[str, Any]:
    """Ensure the hidden AutoSkills cron matches config."""
    from ...langgraph_dev.manager import is_langgraph_dev_running

    if not is_langgraph_dev_running(base_url=langgraph_dev_url(config)):
        return {"status": "unavailable"}

    client = get_langgraph_sync_client(url=langgraph_dev_url(config))
    existing = list_autoskill_schedules(
        config,
        limit=AUTOSKILL_SCHEDULE_SEARCH_LIMIT,
    )
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
    matching = [
        row
        for row in existing
        if row.get("schedule") == schedule
        and bool(row.get("enabled", True))
        and (row.get("metadata") or {}).get("workspace_dir")
        == metadata["workspace_dir"]
        and (row.get("metadata") or {}).get("mode") == metadata["mode"]
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
            "workspace_dir": str(Path(workspace_dir).expanduser().resolve()),
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
        config={"configurable": {"thread_id": str(thread["thread_id"])}},
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
            "workspace_dir": str(Path(workspace_dir).expanduser().resolve()),
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
        config={"configurable": {"thread_id": str(thread["thread_id"])}},
    )
    return {"thread_id": thread["thread_id"], "run_id": run["run_id"]}
