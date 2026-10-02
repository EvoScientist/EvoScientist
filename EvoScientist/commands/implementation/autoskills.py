from __future__ import annotations

import asyncio
from enum import Enum
from typing import ClassVar

from rich.table import Table

from ..base import Command, CommandContext, SubCommand
from ..manager import manager

AUTOSKILLS_COMMAND = "/autoskills"
_PROPOSAL_STATUSES = {
    "review": "pending",
    "approved": "approved",
    "rejected": "rejected",
}


class AutoSkillsCommand(Command):
    """Manage EvoMemory AutoSkills proposals."""

    name = AUTOSKILLS_COMMAND
    alias: ClassVar[list[str]] = ["/skills-review"]
    description = "Review EvoMemory autoskill proposals"
    subcommands: ClassVar[list[SubCommand]] = [
        SubCommand("status", "Show AutoSkills config and proposals for review"),
        SubCommand("help", "Show AutoSkills command examples"),
        SubCommand("list", "List autoskill proposals, optionally filtered by status"),
        SubCommand("review", "Review autoskill proposals awaiting a decision"),
        SubCommand("approve", "Approve an autoskill proposal by id"),
        SubCommand("reject", "Reject an autoskill proposal by id"),
        SubCommand("run", "Run AutoSkills once now"),
        SubCommand("on", "Enable periodic AutoSkills"),
        SubCommand("off", "Disable periodic AutoSkills"),
        SubCommand("mode", "Set review or auto approval mode"),
        SubCommand("cadence", "Set nightly, weekly, or monthly cadence"),
        SubCommand("time", "Set local run time as HH:MM"),
    ]

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        sub = args[0].lower() if args else "help"
        rest = args[1:]
        if sub in {"help", "-h", "--help", "?"}:
            self._show_help(ctx)
        elif sub in {"status", "show"}:
            await self._status(ctx)
        elif sub in {"list", "ls", "proposals"}:
            await self._list_command(ctx, rest)
        elif sub == "review":
            await self._list(ctx, status="pending")
        elif sub in {"approve", "accept"}:
            await self._approve(ctx, self._first_arg(rest))
        elif sub in {"reject", "deny", "decline"}:
            await self._reject(ctx, self._first_arg(rest))
        elif sub in {"run", "now"}:
            await self._run(ctx)
        elif sub in {"on", "enable"}:
            await self._set_config(ctx, "memory_skill_synthesis_enabled", "true")
        elif sub in {"off", "disable"}:
            await self._set_config(ctx, "memory_skill_synthesis_enabled", "false")
        elif sub == "mode":
            await self._set_config(
                ctx,
                "memory_skill_synthesis_mode",
                self._first_arg(rest),
            )
        elif sub in {"auto", "automatic"}:
            await self._set_config(ctx, "memory_skill_synthesis_mode", "auto")
        elif sub == "manual":
            await self._set_config(ctx, "memory_skill_synthesis_mode", "review")
        elif sub == "cadence":
            await self._set_config(
                ctx,
                "memory_skill_synthesis_cadence",
                self._first_arg(rest),
            )
        elif sub in {"nightly", "weekly", "monthly"}:
            await self._set_config(ctx, "memory_skill_synthesis_cadence", sub)
        elif sub == "time":
            await self._set_config(
                ctx,
                "memory_skill_synthesis_time",
                self._first_arg(rest),
            )
        else:
            self._show_help(ctx, prefix=f"Unknown AutoSkills command: {sub}")

    async def _status(self, ctx: CommandContext) -> None:
        from ... import paths
        from ...config import get_effective_config
        from ...memory.autoskills.proposals import list_skill_proposals
        from ...memory.autoskills.schedule import alist_autoskill_schedules

        cfg = get_effective_config()
        workspace_dir = self._workspace_dir(ctx)
        pending = list_skill_proposals(
            paths.MEMORIES_DIR,
            status="pending",
            workspace_dir=workspace_dir,
        )
        ctx.ui.append_system(
            (
                "AutoSkills: "
                f"{'on' if cfg.memory_skill_synthesis_enabled else 'off'} | "
                f"mode={cfg.memory_skill_synthesis_mode.value} | "
                f"cadence={cfg.memory_skill_synthesis_cadence.value} | "
                f"time={cfg.memory_skill_synthesis_time}"
            ),
            style="dim",
        )
        ctx.ui.append_system(
            f"AutoSkill proposal(s) ready for review: {len(pending)}",
            style="yellow" if pending else "dim",
        )
        if pending:
            ctx.ui.append_system(
                (
                    f"Next: {AUTOSKILLS_COMMAND} review, then "
                    f"{AUTOSKILLS_COMMAND} approve <id> or "
                    f"{AUTOSKILLS_COMMAND} reject <id>."
                ),
                style="dim",
            )
        elif cfg.memory_skill_synthesis_enabled:
            ctx.ui.append_system(
                f"Next: {AUTOSKILLS_COMMAND} run to search now, or "
                f"{AUTOSKILLS_COMMAND} help for commands.",
                style="dim",
            )
        else:
            ctx.ui.append_system(
                f"Next: {AUTOSKILLS_COMMAND} run to search once, or "
                f"{AUTOSKILLS_COMMAND} on to enable scheduled runs.",
                style="dim",
            )
        if cfg.memory_skill_synthesis_enabled:
            try:
                rows = await alist_autoskill_schedules(cfg, limit=1)
            except Exception:
                rows = []
            if rows:
                ctx.ui.append_system(
                    f"Background schedule id: {str(rows[0].get('cron_id', ''))[:8]}",
                    style="dim",
                )

    async def _list_command(self, ctx: CommandContext, args: list[str]) -> None:
        if not args or args[0].lower() == "all":
            await self._list(ctx)
            return
        status = _PROPOSAL_STATUSES.get(args[0].lower())
        if status is None:
            ctx.ui.append_system(
                f"Usage: {AUTOSKILLS_COMMAND} list [review|approved|rejected|all]",
                style="yellow",
            )
            return
        await self._list(ctx, status=status)

    async def _list(self, ctx: CommandContext, *, status: str | None = None) -> None:
        from ... import paths
        from ...memory.autoskills.proposals import list_skill_proposals

        workspace_dir = self._workspace_dir(ctx)
        proposals = list_skill_proposals(
            paths.MEMORIES_DIR,
            status=status,
            workspace_dir=workspace_dir,
        )
        if not proposals:
            if status:
                label = self._status_label(status)
                ctx.ui.append_system(
                    f"No autoskill proposals {label}.",
                    style="dim",
                )
            else:
                ctx.ui.append_system("No autoskill proposals.", style="dim")
            return

        title = "EvoMemory AutoSkill Proposals"
        if status:
            title = (
                f"EvoMemory AutoSkill Proposals {self._status_label(status).title()}"
            )
        table = Table(title=title, show_header=True)
        table.add_column("ID", style="cyan")
        table.add_column("Action", style="magenta")
        table.add_column("AutoSkill", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Observations", justify="right")
        table.add_column("Description", style="dim")
        for proposal in proposals:
            table.add_row(
                proposal.proposal_id,
                proposal.operation,
                proposal.skill_name,
                proposal.status,
                str(len(proposal.source_observation_ids)),
                proposal.description,
            )
        ctx.ui.mount_renderable(table)
        ctx.ui.append_system(
            f"Use {AUTOSKILLS_COMMAND} approve <id> or "
            f"{AUTOSKILLS_COMMAND} reject <id>.",
            style="dim",
        )

    async def _approve(self, ctx: CommandContext, proposal_id: str | None) -> None:
        from ... import paths
        from ...memory.autoskills.proposals import approve_skill_proposal

        if not proposal_id:
            ctx.ui.append_system(
                f"Usage: {AUTOSKILLS_COMMAND} approve <id>",
                style="yellow",
            )
            ctx.ui.append_system(
                f"Run {AUTOSKILLS_COMMAND} review to copy a proposal ID.",
                style="dim",
            )
            return
        workspace_dir = self._workspace_dir(ctx)
        result = await asyncio.to_thread(
            approve_skill_proposal,
            paths.MEMORIES_DIR,
            proposal_id,
            workspace_dir=workspace_dir,
        )
        if result.get("approved"):
            verb = "Updated" if result.get("operation") == "update" else "Approved"
            ctx.ui.append_system(
                f"{verb} autoskill: {result['skill_name']} ({result['path']})",
                style="green",
            )
            ctx.ui.append_system(
                "Reload with /new to apply the new skill.", style="dim"
            )
        else:
            ctx.ui.append_system(f"Approval failed: {result.get('error')}", style="red")

    async def _reject(self, ctx: CommandContext, proposal_id: str | None) -> None:
        from ... import paths
        from ...memory.autoskills.proposals import reject_skill_proposal

        if not proposal_id:
            ctx.ui.append_system(
                f"Usage: {AUTOSKILLS_COMMAND} reject <id>",
                style="yellow",
            )
            ctx.ui.append_system(
                f"Run {AUTOSKILLS_COMMAND} review to copy a proposal ID.",
                style="dim",
            )
            return
        workspace_dir = self._workspace_dir(ctx)
        result = await asyncio.to_thread(
            reject_skill_proposal,
            paths.MEMORIES_DIR,
            proposal_id,
            workspace_dir=workspace_dir,
        )
        if result.get("rejected"):
            ctx.ui.append_system(
                f"Rejected proposal: {result['proposal_id']}",
                style="green",
            )
        else:
            ctx.ui.append_system(f"Reject failed: {result.get('error')}", style="red")

    async def _run(self, ctx: CommandContext) -> None:
        from ...config import get_effective_config
        from ...memory.autoskills.schedule import arun_autoskill_now

        workspace_dir = self._workspace_dir(ctx)
        try:
            result = await arun_autoskill_now(
                get_effective_config(),
                workspace_dir=workspace_dir,
            )
        except Exception as exc:
            ctx.ui.append_system(f"Failed to start AutoSkills: {exc}", style="red")
            return
        ctx.ui.append_system(
            f"Started AutoSkills run {result['run_id']}.",
            style="green",
        )

    async def _set_config(
        self,
        ctx: CommandContext,
        key: str,
        value: str | None,
    ) -> None:
        from ...config import get_effective_config, set_config_value
        from ...memory.autoskills.schedule import reconcile_autoskill_schedule

        workspace_dir = self._workspace_dir(ctx)
        if not value:
            cfg = get_effective_config()
            current = self._display_value(getattr(cfg, key))
            ctx.ui.append_system(
                f"Current {self._config_label(key)}: {current}",
                style="dim",
            )
            ctx.ui.append_system(
                f"Usage: {self._config_usage(key)}",
                style="yellow",
            )
            return
        if not await asyncio.to_thread(set_config_value, key, value):
            valid = self._config_values(key)
            suffix = f" Valid values: {valid}." if valid else ""
            ctx.ui.append_system(
                f"Invalid value for {self._config_label(key)}: {value}.{suffix}",
                style="red",
            )
            return
        cfg = get_effective_config()
        if ctx.config is not None and hasattr(ctx.config, key):
            setattr(ctx.config, key, getattr(cfg, key))
        await asyncio.to_thread(
            reconcile_autoskill_schedule,
            cfg,
            workspace_dir=workspace_dir,
        )
        ctx.ui.append_system(
            f"Updated {self._config_label(key)} = {self._display_value(getattr(cfg, key))}",
            style="green",
        )

    @staticmethod
    def _config_label(key: str) -> str:
        labels = {
            "memory_skill_synthesis_enabled": "AutoSkills",
            "memory_skill_synthesis_mode": "AutoSkills mode",
            "memory_skill_synthesis_cadence": "AutoSkills cadence",
            "memory_skill_synthesis_time": "AutoSkills time",
        }
        return labels.get(key, key)

    @staticmethod
    def _display_value(value: object) -> object:
        return getattr(value, "value", value)

    @staticmethod
    def _enum_values(enum_type: type[Enum], *, separator: str = ", ") -> str:
        return separator.join(str(member.value) for member in enum_type)

    @classmethod
    def _config_usage(cls, key: str) -> str:
        from ...config import MemorySkillSynthesisCadence, MemorySkillSynthesisMode

        if key == "memory_skill_synthesis_mode":
            values = cls._enum_values(MemorySkillSynthesisMode, separator="|")
            return f"{AUTOSKILLS_COMMAND} mode {values}"
        if key == "memory_skill_synthesis_cadence":
            values = cls._enum_values(MemorySkillSynthesisCadence, separator="|")
            return f"{AUTOSKILLS_COMMAND} cadence {values}"
        if key == "memory_skill_synthesis_time":
            return f"{AUTOSKILLS_COMMAND} time HH:MM"
        return f"{AUTOSKILLS_COMMAND} <value>"

    @classmethod
    def _config_values(cls, key: str) -> str | None:
        from ...config import MemorySkillSynthesisCadence, MemorySkillSynthesisMode

        if key == "memory_skill_synthesis_mode":
            return cls._enum_values(MemorySkillSynthesisMode)
        if key == "memory_skill_synthesis_cadence":
            return cls._enum_values(MemorySkillSynthesisCadence)
        if key == "memory_skill_synthesis_time":
            return "24-hour local time, for example 03:00"
        return None

    @staticmethod
    def _status_label(status: str) -> str:
        if status == "pending":
            return "ready for review"
        return status

    @staticmethod
    def _show_help(ctx: CommandContext, *, prefix: str | None = None) -> None:
        if prefix:
            ctx.ui.append_system(prefix, style="yellow")
        ctx.ui.append_system(
            (
                f"Usage: {AUTOSKILLS_COMMAND} "
                "[status|review|approve|reject|run|on|off|mode|cadence|time]"
            ),
            style="bold",
        )
        table = Table(title="AutoSkills Commands", show_header=True)
        table.add_column("Command", style="cyan")
        table.add_column("Use when", style="dim")
        rows = [
            (AUTOSKILLS_COMMAND, "Show this command reference"),
            (f"{AUTOSKILLS_COMMAND} status", "Show config and the next useful action"),
            (f"{AUTOSKILLS_COMMAND} review", "Review proposals waiting for a decision"),
            (f"{AUTOSKILLS_COMMAND} approve <id>", "Install a reviewed autoskill"),
            (f"{AUTOSKILLS_COMMAND} reject <id>", "Dismiss a reviewed proposal"),
            (f"{AUTOSKILLS_COMMAND} run", "Start a one-off background autoskill run"),
            (f"{AUTOSKILLS_COMMAND} on|off", "Enable or disable scheduled runs"),
            (f"{AUTOSKILLS_COMMAND} auto|manual", "Switch approval behavior"),
            (
                f"{AUTOSKILLS_COMMAND} nightly|weekly|monthly",
                "Set the built-in schedule cadence",
            ),
            (f"{AUTOSKILLS_COMMAND} time 03:00", "Set the local schedule time"),
            (
                f"{AUTOSKILLS_COMMAND} list [status]",
                "List all proposals or filter by review, approved, or rejected",
            ),
        ]
        for command, description in rows:
            table.add_row(command, description)
        ctx.ui.mount_renderable(table)
        ctx.ui.append_system(
            "Aliases: /skills-review, ls, proposals, accept, deny, enable, disable, now.",
            style="dim",
        )

    @staticmethod
    def _workspace_dir(ctx: CommandContext) -> str:
        from ... import paths

        return str(ctx.workspace_dir or paths.WORKSPACE_ROOT)

    @staticmethod
    def _first_arg(args: list[str]) -> str | None:
        return args[0] if args else None


manager.register(AutoSkillsCommand())
