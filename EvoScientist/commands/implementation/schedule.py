from __future__ import annotations

import asyncio
import re
from typing import ClassVar

from rich.table import Table

from ..base import Command, CommandContext, SubCommand
from ..manager import manager


class ScheduleCommand(Command):
    """Manage scheduled (cron) tasks."""

    name = "/schedule"
    description = "Manage scheduled (cron) tasks"
    subcommands: ClassVar[list[SubCommand]] = [
        SubCommand("add", 'Add: /schedule add <m h dom mon dow> "<prompt>"'),
        SubCommand("list", "List scheduled tasks"),
        SubCommand("remove", "Remove a schedule by id"),
        SubCommand("run", "Run a schedule's prompt once now (test)"),
        SubCommand("pause", "Disable a schedule by id"),
        SubCommand("resume", "Enable a schedule by id"),
    ]

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        """Dispatch to the appropriate /schedule subcommand."""
        cfg = getattr(ctx, "config", None)
        if cfg is not None and not getattr(cfg, "enable_scheduler", True):
            ctx.ui.append_system(
                "Scheduled tasks are disabled (`enable_scheduler` is off).",
                style="yellow",
            )
            return

        from ...cron import schedule as crons

        # Cron SDK calls are sync HTTP; offload to a thread so backend latency
        # can never freeze the interactive event loop.
        if not await asyncio.to_thread(crons.is_available):
            ctx.ui.append_system(
                "Scheduler unavailable: the langgraph dev backend is not running.",
                style="yellow",
            )
            return

        if not args or args[0].lower() == "list":
            await self._list(ctx, crons)
            return

        sub = args[0].lower()
        rest = args[1:]
        if sub == "add":
            await self._add(ctx, crons, rest)
        elif sub == "remove":
            await self._remove(ctx, crons, rest[0] if rest else "")
        elif sub == "run":
            await self._run(ctx, crons, rest[0] if rest else "")
        elif sub in ("pause", "resume"):
            await self._set_enabled(
                ctx, crons, rest[0] if rest else "", sub == "resume"
            )
        else:
            ctx.ui.append_system("Schedule commands:", style="bold")
            for s in self.subcommands:
                ctx.ui.append_system(
                    f"  /schedule {s.name:<8} {s.description}", style="dim"
                )

    async def _add(self, ctx: CommandContext, crons, rest: list[str]) -> None:
        # Cron may arrive as 5 separate tokens (unquoted) or 1 token (shlex-quoted).
        # Split on any whitespace so extra spaces don't break detection; the
        # backend rejects genuinely malformed expressions.
        if rest and len(rest[0].split()) == 5:  # quoted 5-field cron
            schedule, prompt_tokens = " ".join(rest[0].split()), rest[1:]
        elif len(rest) >= 5:  # 5 separate cron fields
            schedule, prompt_tokens = " ".join(rest[:5]), rest[5:]
        else:
            ctx.ui.append_system(
                'Usage: /schedule add "<m h dom mon dow>" "<prompt>"', style="yellow"
            )
            return
        prompt = " ".join(prompt_tokens).strip().strip('"').strip("'")
        if not prompt:
            ctx.ui.append_system("A task prompt is required.", style="yellow")
            return
        # B3: strip unsafe chars; keep only alphanumerics + hyphens (kebab-case).
        raw = prompt[:48].lower()
        name = re.sub(r"[^a-z0-9]+", "-", raw).strip("-")[:32] or "task"
        try:
            rec = await asyncio.to_thread(
                crons.create_schedule, name=name, schedule=schedule, prompt=prompt
            )
        except Exception as exc:
            ctx.ui.append_system(f"Error: {exc}", style="red")
            return
        ctx.ui.append_system(
            f"Scheduled '{name}' [{schedule}] — id {rec.get('cron_id')}. "
            "Runs unattended in the background.",
            style="green",
        )

    async def _list(self, ctx: CommandContext, crons) -> None:
        # B1: guard SDK call — backend may die after the is_available() check.
        try:
            rows = await asyncio.to_thread(crons.list_schedules)
        except Exception as exc:
            ctx.ui.append_system(f"Error: {exc}", style="red")
            return
        if not rows:
            ctx.ui.append_system(
                "No scheduled tasks. Add one: /schedule add ...", style="dim"
            )
            return
        table = Table(title="Scheduled Tasks", show_header=True)
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="magenta")
        table.add_column("Schedule", style="green")
        table.add_column("Enabled", style="yellow")
        table.add_column("Next run (UTC)", style="white")
        for r in rows:
            meta = r.get("metadata") or {}
            table.add_row(
                str(r.get("cron_id", ""))[:8],
                str(meta.get("name", "")),
                str(r.get("schedule", "")),
                "yes" if r.get("enabled", True) else "no",
                str(r.get("next_run_date", "")),
            )
        ctx.ui.mount_renderable(table)

    _AMBIGUOUS = object()  # B2: sentinel returned when multiple crons match a prefix
    _BACKEND_ERROR = object()  # sentinel returned when list_schedules() raises

    async def _resolve(self, crons, prefix: str):
        """Return the unique matching record, _AMBIGUOUS if >1 match, _BACKEND_ERROR on error, or None."""
        # B1: guard SDK call — backend may die after is_available() check.
        try:
            all_rows = await asyncio.to_thread(crons.list_schedules)
        except Exception as exc:
            # Store the exception text so _resolve_or_report can surface it.
            self._last_backend_exc = exc
            return self._BACKEND_ERROR
        # B2: collect ALL matches; ambiguous prefix → sentinel so callers can warn.
        matches = [r for r in all_rows if str(r.get("cron_id", "")).startswith(prefix)]
        if len(matches) > 1:
            return self._AMBIGUOUS
        return matches[0] if matches else None

    async def _resolve_or_report(self, ctx: CommandContext, crons, prefix: str):
        """Resolve prefix → record, emit UI error on ambiguity/miss/error, return None on failure."""
        match = await self._resolve(crons, prefix)
        if match is self._BACKEND_ERROR:
            exc = getattr(self, "_last_backend_exc", None)
            ctx.ui.append_system(
                f"Error: scheduler backend unavailable ({exc})",
                style="red",
            )
            return None
        if match is self._AMBIGUOUS:
            ctx.ui.append_system(
                f"Multiple schedules match '{prefix}' — use a longer id.",
                style="yellow",
            )
            return None
        if not match:
            ctx.ui.append_system(f"No schedule matching {prefix}.", style="yellow")
            return None
        return match

    async def _remove(self, ctx: CommandContext, crons, prefix: str) -> None:
        if not prefix:
            ctx.ui.append_system("Usage: /schedule remove <id>", style="yellow")
            return
        match = await self._resolve_or_report(ctx, crons, prefix)
        if match is None:
            return
        cron_id = str(match.get("cron_id", ""))
        try:
            await asyncio.to_thread(crons.delete_schedule, cron_id)
        except Exception as exc:
            ctx.ui.append_system(f"Error: {exc}", style="red")
            return
        ctx.ui.append_system(f"Removed schedule {cron_id}.", style="green")

    async def _run(self, ctx: CommandContext, crons, prefix: str) -> None:
        if not prefix:
            ctx.ui.append_system("Usage: /schedule run <id>", style="yellow")
            return
        match = await self._resolve_or_report(ctx, crons, prefix)
        if match is None:
            return
        prompt = (match.get("metadata") or {}).get("prompt", "")
        if not str(prompt).strip():
            ctx.ui.append_system(
                f"Schedule {prefix} has no stored prompt — cannot run it.",
                style="yellow",
            )
            return
        try:
            rec = await asyncio.to_thread(crons.run_now, prompt)
        except Exception as exc:
            ctx.ui.append_system(f"Error: {exc}", style="red")
            return
        # Don't promise a location; the task's own prompt decides where output goes.
        ctx.ui.append_system(
            f"Fired schedule {prefix} once now (run {rec.get('run_id')}). "
            "Any output goes wherever the task's instruction specifies.",
            style="green",
        )

    async def _set_enabled(
        self, ctx: CommandContext, crons, prefix: str, enabled: bool
    ) -> None:
        if not prefix:
            ctx.ui.append_system("Usage: /schedule pause|resume <id>", style="yellow")
            return
        match = await self._resolve_or_report(ctx, crons, prefix)
        if match is None:
            return
        cron_id = str(match.get("cron_id", ""))
        try:
            await asyncio.to_thread(crons.set_enabled, cron_id, enabled)
        except Exception as exc:
            ctx.ui.append_system(f"Error: {exc}", style="red")
            return
        ctx.ui.append_system(
            f"{'Resumed' if enabled else 'Paused'} schedule {cron_id}.", style="green"
        )


# Register schedule command
manager.register(ScheduleCommand())
