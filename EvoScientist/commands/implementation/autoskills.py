from __future__ import annotations

import asyncio
from typing import ClassVar

from rich.table import Table

from ..base import Command, CommandContext, SubCommand
from ..manager import manager

AUTOSKILLS_COMMAND = "/autoskills"


class AutoSkillsCommand(Command):
    """Manage EvoMemory AutoSkills proposals."""

    name = AUTOSKILLS_COMMAND
    alias: ClassVar[list[str]] = ["/skills-review"]
    description = "Review EvoMemory autoskill proposals"
    subcommands: ClassVar[list[SubCommand]] = [
        SubCommand("status", "Show AutoSkills config and pending proposals"),
        SubCommand("list", "List autoskill proposals"),
        SubCommand("approve", "Approve a pending autoskill proposal by id"),
        SubCommand("reject", "Reject a pending autoskill proposal by id"),
        SubCommand("run", "Run AutoSkills once now"),
        SubCommand("on", "Enable periodic AutoSkills"),
        SubCommand("off", "Disable periodic AutoSkills"),
        SubCommand("mode", "Set review or auto approval mode"),
        SubCommand("cadence", "Set nightly, weekly, or monthly cadence"),
        SubCommand("time", "Set local run time as HH:MM"),
    ]

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        sub = args[0].lower() if args else "status"
        rest = args[1:]
        if sub == "status":
            await self._status(ctx)
        elif sub == "list":
            await self._list(ctx)
        elif sub == "approve":
            await self._approve(ctx, self._first_arg(rest))
        elif sub == "reject":
            await self._reject(ctx, self._first_arg(rest))
        elif sub == "run":
            await self._run(ctx)
        elif sub == "on":
            await self._set_config(ctx, "memory_skill_synthesis_enabled", "true")
        elif sub == "off":
            await self._set_config(ctx, "memory_skill_synthesis_enabled", "false")
        elif sub == "mode":
            await self._set_config(
                ctx,
                "memory_skill_synthesis_mode",
                self._first_arg(rest),
            )
        elif sub == "cadence":
            await self._set_config(
                ctx,
                "memory_skill_synthesis_cadence",
                self._first_arg(rest),
            )
        elif sub == "time":
            await self._set_config(
                ctx,
                "memory_skill_synthesis_time",
                self._first_arg(rest),
            )
        else:
            ctx.ui.append_system("AutoSkills commands:", style="bold")
            for item in self.subcommands:
                ctx.ui.append_system(
                    f"  {AUTOSKILLS_COMMAND} {item.name:<8} {item.description}",
                    style="dim",
                )

    async def _status(self, ctx: CommandContext) -> None:
        from ... import paths
        from ...config import get_effective_config
        from ...memory.autoskills.proposals import list_skill_proposals
        from ...memory.autoskills.schedule import alist_autoskill_schedules

        cfg = get_effective_config()
        pending = list_skill_proposals(paths.MEMORIES_DIR, status="pending")
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
            f"Pending autoskill proposal(s): {len(pending)}",
            style="yellow" if pending else "dim",
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

    async def _list(self, ctx: CommandContext) -> None:
        from ... import paths
        from ...memory.autoskills.proposals import list_skill_proposals

        proposals = list_skill_proposals(paths.MEMORIES_DIR)
        if not proposals:
            ctx.ui.append_system("No autoskill proposals.", style="dim")
            return

        table = Table(title="EvoMemory AutoSkill Proposals", show_header=True)
        table.add_column("ID", style="cyan")
        table.add_column("AutoSkill", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Observations", justify="right")
        table.add_column("Description", style="dim")
        for proposal in proposals:
            table.add_row(
                proposal.proposal_id,
                proposal.skill_name,
                proposal.status,
                str(len(proposal.source_observation_ids)),
                proposal.description,
            )
        ctx.ui.mount_renderable(table)
        ctx.ui.append_system(
            f"Use {AUTOSKILLS_COMMAND} approve <id> or {AUTOSKILLS_COMMAND} reject <id>.",
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
            return
        result = await asyncio.to_thread(
            approve_skill_proposal,
            paths.MEMORIES_DIR,
            proposal_id,
            skills_dir=paths.USER_SKILLS_DIR,
        )
        if result.get("approved"):
            ctx.ui.append_system(
                f"Approved autoskill: {result['skill_name']} ({result['path']})",
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
            return
        result = await asyncio.to_thread(
            reject_skill_proposal,
            paths.MEMORIES_DIR,
            proposal_id,
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

        workspace_dir = ctx.workspace_dir or "."
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

        if not value:
            ctx.ui.append_system(
                f"Missing value for {self._config_label(key)}.",
                style="yellow",
            )
            return
        if not await asyncio.to_thread(set_config_value, key, value):
            ctx.ui.append_system(
                f"Invalid value for {self._config_label(key)}: {value}",
                style="red",
            )
            return
        cfg = get_effective_config()
        if ctx.config is not None and hasattr(ctx.config, key):
            setattr(ctx.config, key, getattr(cfg, key))
        workspace_dir = ctx.workspace_dir or "."
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
    def _first_arg(args: list[str]) -> str | None:
        return args[0] if args else None


manager.register(AutoSkillsCommand())
