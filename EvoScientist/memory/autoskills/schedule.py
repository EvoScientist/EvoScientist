"""LangGraph scheduling helpers for EvoMemory AutoSkills."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ...config import EvoScientistConfig
from ...cron.schedule import (
    asearch_all,
    owned,
    run_config,
    workspace_key,
)
from ...langgraph_dev.assistants import launch_assistant_id
from ...langgraph_dev.sdk import (
    default_scheduler_timezone,
    get_langgraph_async_client,
    get_langgraph_sync_client,
    langgraph_dev_url,
    messages_input,
)
from ...paths import SessionDirs, Workspace
from ..types import MEMORY_AGENT_RECURSION_LIMIT

AUTOSKILL_GRAPH_ID = "evomemory-autoskills"
AUTOSKILL_RUN_KIND = "evomemory_autoskills"
# The settings an AutoSkills cron is built from (see plan_autoskill_schedule):
# changing one reschedules every project's cron.
AUTOSKILL_SETTINGS = frozenset(
    {
        "memory_skill_synthesis_enabled",
        "memory_skill_synthesis_mode",
        "memory_skill_synthesis_cadence",
        "memory_skill_synthesis_time",
        "scheduler_default_timezone",
    }
)


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


def _autoskill_metadata(*, config: EvoScientistConfig, schedule: str) -> dict[str, str]:
    return {
        "run_kind": AUTOSKILL_RUN_KIND,
        "name": "EvoMemory AutoSkills",
        "mode": config.memory_skill_synthesis_mode.value,
        "cadence": config.memory_skill_synthesis_cadence.value,
        "time": config.memory_skill_synthesis_time,
        "schedule": schedule,
    }


def _owned_by(
    rows: list[dict[str, Any]], workspace_dir: str | Path
) -> list[dict[str, Any]]:
    return owned(rows, Workspace(workspace_dir))


async def alist_autoskill_schedules(
    config: EvoScientistConfig,
    *,
    workspace_dir: str | Path,
) -> list[dict[str, Any]]:
    """The workspace's internal AutoSkills cron records."""
    rows = await asearch_all(
        get_langgraph_async_client(url=langgraph_dev_url(config)),
        metadata={"run_kind": AUTOSKILL_RUN_KIND},
    )
    return await asyncio.to_thread(_owned_by, rows, workspace_dir)


def _run_config(cron: dict[str, Any]) -> dict[str, Any]:
    return (cron.get("payload") or {}).get("config") or {}


def _forwarded_workspace(cron: dict[str, Any]) -> str | None:
    """The workspace a cron's runs forward (its run config), in stored form."""
    configurable = _run_config(cron).get("configurable") or {}
    return workspace_key(configurable.get("workspace_dir"))


@dataclass(frozen=True)
class AutoskillCron:
    """The AutoSkills cron the configuration asks for (the fields of
    :func:`~EvoScientist.cron.schedule.cron_request`)."""

    schedule: str
    input: dict[str, Any]
    metadata: dict[str, str]
    timezone: str | None


@dataclass(frozen=True)
class AutoskillPlan:
    """What a reconcile does with a workspace's AutoSkills crons: keep
    *keep* (a matching cron) or create *create*, then delete *delete*."""

    keep: dict[str, Any] | None
    create: AutoskillCron | None
    delete: list[dict[str, Any]]


def plan_autoskill_schedule(
    config: EvoScientistConfig,
    workspace: Workspace,
    existing: list[dict[str, Any]],
) -> AutoskillPlan:
    """The reconcile of *workspace*'s AutoSkills crons *existing* against
    *config*: disabled, every one goes; else a matching one is kept and the
    rest go, or the replacement is created before the old ones go, so a
    failed create leaves them running. Blocking (the default timezone reads
    the system's); ``ValueError`` for an unknown timezone."""
    if not config.memory_skill_synthesis_enabled:
        return AutoskillPlan(keep=None, create=None, delete=list(existing))
    schedule = autoskill_cron(
        config.memory_skill_synthesis_cadence,
        config.memory_skill_synthesis_time,
    )
    metadata = _autoskill_metadata(config=config, schedule=schedule)
    timezone = default_scheduler_timezone(config)
    if timezone is not None:
        try:
            ZoneInfo(timezone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"Unknown scheduler timezone {timezone!r}.") from exc
    # A cron created before its runs forwarded the workspace, ran on the
    # workspace's assistant or carried the memory step limit does not match.
    assistant_id = launch_assistant_id(AUTOSKILL_GRAPH_ID, workspace)
    matching = [
        row
        for row in existing
        if row.get("schedule") == schedule
        and row.get("timezone") == timezone
        and bool(row.get("enabled", True))
        and (row.get("metadata") or {}).get("mode") == metadata["mode"]
        and _forwarded_workspace(row) == workspace.key
        and str(row.get("assistant_id")) == assistant_id
        and _run_config(row).get("recursion_limit") == MEMORY_AGENT_RECURSION_LIMIT
    ]
    if matching:
        kept = matching[0]
        return AutoskillPlan(
            keep=kept, create=None, delete=[r for r in existing if r is not kept]
        )
    create = AutoskillCron(
        schedule=schedule,
        input=_autoskill_input(),
        metadata=metadata,
        timezone=timezone,
    )
    return AutoskillPlan(keep=None, create=create, delete=list(existing))


def run_autoskill_now(
    config: EvoScientistConfig,
    *,
    workspace_dir: str | Path,
) -> dict[str, Any]:
    """Launch a one-off AutoSkills run immediately."""
    client = get_langgraph_sync_client(url=langgraph_dev_url(config))
    workspace = Workspace(workspace_dir)
    folders = SessionDirs(workspace).metadata()
    thread = client.threads.create(
        graph_id=AUTOSKILL_GRAPH_ID,
        metadata={"run_kind": AUTOSKILL_RUN_KIND, **folders},
    )
    run = client.runs.create(
        thread_id=str(thread["thread_id"]),
        assistant_id=launch_assistant_id(AUTOSKILL_GRAPH_ID, workspace),
        input=_autoskill_input(),
        metadata={**_autoskill_metadata(config=config, schedule="manual"), **folders},
        config={
            **run_config(workspace, thread_id=str(thread["thread_id"])),
            "recursion_limit": MEMORY_AGENT_RECURSION_LIMIT,
        },
    )
    return {"thread_id": thread["thread_id"], "run_id": run["run_id"]}


async def arun_autoskill_now(
    config: EvoScientistConfig,
    *,
    workspace_dir: str | Path,
) -> dict[str, Any]:
    """Async variant of :func:`run_autoskill_now`."""
    client = get_langgraph_async_client(url=langgraph_dev_url(config))
    workspace = Workspace(workspace_dir)
    folders = SessionDirs(workspace).metadata()
    thread = await client.threads.create(
        graph_id=AUTOSKILL_GRAPH_ID,
        metadata={"run_kind": AUTOSKILL_RUN_KIND, **folders},
    )
    run = await client.runs.create(
        thread_id=str(thread["thread_id"]),
        assistant_id=launch_assistant_id(AUTOSKILL_GRAPH_ID, workspace),
        input=_autoskill_input(),
        metadata={**_autoskill_metadata(config=config, schedule="manual"), **folders},
        config={
            **run_config(workspace, thread_id=str(thread["thread_id"])),
            "recursion_limit": MEMORY_AGENT_RECURSION_LIMIT,
        },
    )
    return {"thread_id": thread["thread_id"], "run_id": run["run_id"]}
