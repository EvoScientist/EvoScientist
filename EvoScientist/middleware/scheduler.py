"""Scheduling middleware and tools for the main agent.

Bundles three NL scheduling tools (schedule_task, list_scheduled_tasks,
cancel_scheduled_task) together with the SchedulerMiddleware that:
- Injects scheduling guidance + a live ``<scheduled_tasks>`` block into every
  system prompt (static→dynamic, mirroring EvoMemoryMiddleware).
- Contributes the three tools via ``self.tools`` so agent wiring only needs to
  append this middleware — no separate tool-list management required.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.tools import tool

from .utils import append_to_system_message

_CACHE_TTL_SECONDS = 15.0

# Static guidance injected into the system prompt; the dynamic <scheduled_tasks>
# list follows it (static→dynamic, mirroring EvoMemoryMiddleware's
# <memory_instructions> + <profile_memory> ordering).
_SCHEDULING_INSTRUCTIONS = """<scheduling_instructions>
When the user asks for something on a recurring schedule ("every 10 minutes",
"each morning at 7"), translate the timing into a 5-field cron expression and
call `schedule_task(name, cron, prompt, timezone)`. The `prompt` must be a
complete, self-contained instruction. Spell out the task fully AND name an
explicit destination for its result, e.g. "...and save the summary to
`/memories/daily-papers.md`" or "...and append the status to `experiment_log.json`"
or "...and update my research memory". It can use skills (e.g.
`paper-navigator`) and the workspace; a task that never says where to put its
output leaves no trace. Use `list_scheduled_tasks` / `cancel_scheduled_task` to
manage them.
</scheduling_instructions>"""


# ---------------------------------------------------------------------------
# NL scheduling tools
# ---------------------------------------------------------------------------


@tool
def schedule_task(name: str, cron: str, prompt: str, timezone: str = "") -> str:
    """Create a recurring scheduled task that runs unattended in the background.

    Translate the user's natural-language timing into a standard 5-field cron
    expression yourself before calling (e.g. 'every 10 minutes' -> '*/10 * * * *',
    'every day at 7am' -> '0 7 * * *', 'every Monday 9am' -> '0 9 * * 1').

    Args:
        name: short human label for the task (e.g. "uk-weather").
        cron: 5-field cron expression.
        prompt: the full instruction the background scheduler runs each time.
        timezone: optional IANA tz (e.g. "Europe/London"); empty = host local zone.
    """
    from ..cron import schedule as crons

    if not crons.is_available():
        return "Scheduler unavailable: the langgraph dev backend is not running."
    try:
        rec = crons.create_schedule(
            name=name, schedule=cron, prompt=prompt, timezone=timezone or None
        )
    except Exception as e:
        return f"Error: {e}"
    return (
        f"Scheduled '{name}' [{cron}] — id {rec.get('cron_id')}. It runs unattended in the "
        "background; output goes wherever the task's prompt specifies. Use list_scheduled_tasks to review."
    )


@tool
def list_scheduled_tasks() -> str:
    """List the user's recurring scheduled tasks (id, name, schedule, enabled)."""
    from ..cron import schedule as crons

    if not crons.is_available():
        return "Scheduler unavailable: the langgraph dev backend is not running."
    try:
        rows = crons.list_schedules()
    except Exception as e:
        return f"Error: {e}"
    if not rows:
        return "No scheduled tasks."
    lines = []
    for r in rows:
        meta = r.get("metadata") or {}
        lines.append(
            f"- {str(r.get('cron_id', ''))[:8]} | {meta.get('name', '')} | "
            f"{r.get('schedule', '')} | {'on' if r.get('enabled', True) else 'off'}"
        )
    return "\n".join(lines)


@tool
def cancel_scheduled_task(cron_id: str) -> str:
    """Cancel (delete) a scheduled task. Pass the id (or its prefix) shown by list_scheduled_tasks."""
    from ..cron import schedule as crons

    if not crons.is_available():
        return "Scheduler unavailable: the langgraph dev backend is not running."
    if not (requested_id := cron_id.strip()):
        # Empty prefix would match (and delete) the only cron — refuse it.
        return "Provide the id (or a prefix) of the task to cancel."
    try:
        rows = crons.list_schedules()
        # B2: collect ALL prefix matches before acting to detect ambiguity.
        matches = [
            r for r in rows if str(r.get("cron_id", "")).startswith(requested_id)
        ]
        if not matches:
            return f"No scheduled task matching '{requested_id}'."
        if len(matches) > 1:
            ids = ", ".join(str(r.get("cron_id", ""))[:8] for r in matches)
            return (
                f"Multiple schedules match '{requested_id}' ({ids}) — use a longer id."
            )
        target = str(matches[0]["cron_id"])
        crons.delete_schedule(target)
    except Exception as e:
        return f"Error: {e}"
    return f"Cancelled scheduled task {target}."


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class SchedulerMiddleware(AgentMiddleware):
    """Inject scheduling awareness + contribute scheduling tools to the main agent."""

    name = "scheduler"

    def __init__(self) -> None:
        super().__init__()
        self._cache: str | None = None
        self._cache_at: float = 0.0
        self.tools = [schedule_task, list_scheduled_tasks, cancel_scheduled_task]

    def _schedules_block(self) -> str:
        """Build the dynamic ``<scheduled_tasks>`` block (empty if none / down)."""
        from ..cron import schedule as crons

        def _clean(value: object) -> str:
            # Flatten to one line + drop angle brackets so a task's own name or
            # prompt can't break the <scheduled_tasks> block or inject pseudo-tags.
            return " ".join(str(value).split()).replace("<", "").replace(">", "")

        try:
            if not crons.is_available():
                return ""
            rows = crons.list_schedules()
        except Exception:
            return ""
        if not rows:
            return ""
        lines = [
            "<scheduled_tasks>",
            "Background cron tasks currently scheduled (they run unattended; you "
            "need do nothing — this is for your awareness, e.g. to answer "
            "questions about them or avoid creating duplicates):",
        ]
        for r in rows:
            meta = r.get("metadata") or {}
            lines.append(
                f"- id={str(r.get('cron_id', ''))[:8]} | {_clean(meta.get('name', ''))} | "
                f"{_clean(r.get('schedule', ''))} | "
                f"{'on' if r.get('enabled', True) else 'off'} | "
                f"{_clean(meta.get('prompt', ''))[:80]}"
            )
        lines.append("</scheduled_tasks>")
        return "\n".join(lines)

    def _cached_schedules_block(self) -> str:
        now = time.monotonic()
        if self._cache is None or (now - self._cache_at) > _CACHE_TTL_SECONDS:
            self._cache = self._schedules_block()
            self._cache_at = now
        return self._cache

    def _injection(self, schedules_block: str) -> str:
        """Static instructions, then the dynamic list (static→dynamic, like memory)."""
        parts = [_SCHEDULING_INSTRUCTIONS]
        if schedules_block:
            parts.append(schedules_block)
        return "\n\n".join(parts)

    def modify_request(self, request: ModelRequest) -> ModelRequest:
        injection = self._injection(self._cached_schedules_block())
        new_system = append_to_system_message(request.system_message, injection)
        return request.override(system_message=new_system)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(self.modify_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        # Offload the (SDK-touching) list build to a thread so we never block the
        # event loop or trip langgraph dev's blockbuster detector.
        block = await asyncio.to_thread(self._cached_schedules_block)
        injection = self._injection(block)
        request = request.override(
            system_message=append_to_system_message(request.system_message, injection)
        )
        return await handler(request)


def create_scheduler_middleware() -> SchedulerMiddleware:
    """Factory for the scheduler middleware (main agent only)."""
    return SchedulerMiddleware()
