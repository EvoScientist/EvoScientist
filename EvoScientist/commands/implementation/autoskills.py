from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

from rich.console import Group
from rich.table import Table
from rich.text import Text

from ..base import Command, CommandContext, SubCommand
from ..manager import manager

AUTOSKILLS_COMMAND = "/autoskills"
_PROPOSAL_STATUSES = {
    "review": "pending",
    "approved": "approved",
    "rejected": "rejected",
}


@dataclass(frozen=True)
class _ProposalSnapshot:
    updated_at: str
    content_digest: str | None = None


class AutoSkillsCommand(Command):
    """Manage EvoMemory AutoSkills proposals."""

    name = AUTOSKILLS_COMMAND
    alias: ClassVar[list[str]] = ["/skills-review"]
    description = "Review EvoMemory autoskill proposals"
    subcommands: ClassVar[list[SubCommand]] = [
        SubCommand("status", "Show AutoSkills config and proposals for review"),
        SubCommand("help", "Show AutoSkills command examples"),
        SubCommand("list", "List autoskill proposals, optionally filtered by status"),
        SubCommand("review", "Review one autoskill proposal by id"),
        SubCommand("approve", "Approve an autoskill proposal by id"),
        SubCommand("reject", "Reject an autoskill proposal by id"),
        SubCommand("run", "Run AutoSkills once now"),
        SubCommand("on", "Enable periodic AutoSkills"),
        SubCommand("off", "Disable periodic AutoSkills"),
        SubCommand("mode", "Set review or auto approval mode"),
        SubCommand("cadence", "Set nightly, weekly, or monthly cadence"),
        SubCommand("time", "Set local run time as HH:MM"),
    ]

    def __init__(self) -> None:
        self._snapshots: dict[tuple[str, str], _ProposalSnapshot] = {}

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        """Dispatch one ``autoskills`` subcommand."""
        sub = args[0].lower() if args else "help"
        rest = args[1:]
        if sub in {"help", "-h", "--help", "?"}:
            self._show_help(ctx)
        elif sub in {"status", "show"}:
            await self._status(ctx)
        elif sub in {"list", "ls", "proposals"}:
            await self._list_command(ctx, rest)
        elif sub == "review":
            await self._review_command(ctx, rest)
        elif sub in {"approve", "accept"}:
            scope, scope_error = self._scope_arg(rest)
            if scope_error is not None:
                ctx.ui.append_system(
                    f"autoskills approve: {scope_error}",
                    style="red",
                )
                return
            await self._approve(ctx, self._first_arg(rest), scope=scope)
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
                    f"Next: {AUTOSKILLS_COMMAND} review <id>, then "
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
                rows = await alist_autoskill_schedules(cfg, workspace_dir=workspace_dir)
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

    async def _review_command(self, ctx: CommandContext, args: list[str]) -> None:
        if len(args) != 1:
            ctx.ui.append_system(
                f"Usage: {AUTOSKILLS_COMMAND} review <id>",
                style="yellow",
            )
            return
        await self._review(ctx, args[0])

    async def _review(self, ctx: CommandContext, proposal_id: str) -> None:
        """Render the copied files of one pending proposal."""
        from ... import paths
        from ...memory.autoskills.proposals import (
            _content_digest,
            _copied_contents,
            list_skill_proposals,
        )

        workspace_dir = self._workspace_dir(ctx)
        proposals = await asyncio.to_thread(
            list_skill_proposals,
            paths.MEMORIES_DIR,
            status="pending",
            workspace_dir=workspace_dir,
        )
        proposal = self._match_proposal(proposals, proposal_id)
        if proposal is None:
            ctx.ui.append_system(
                f"No unique pending autoskill proposal matching {proposal_id!r}.",
                style="yellow",
            )
            return
        try:
            contents = await asyncio.to_thread(_copied_contents, proposal.path)
            content_digest = _content_digest(contents)
        except (OSError, ValueError) as exc:
            ctx.ui.append_system(
                f"Could not read proposal {proposal.proposal_id}: {exc}",
                style="red",
            )
            return
        files = [
            (name, contents[name].decode("utf-8", errors="replace"))
            for name in sorted(contents)
        ]

        shown = [
            Text(f"Draft {proposal.proposal_id}", style="bold"),
            Text(f"Scope: {proposal.effective_scope}", style="dim"),
        ]
        for relative, content in files:
            shown.append(Text(f"--- {relative}", style="cyan"))
            shown.append(Text(content))
        ctx.ui.mount_renderable(Group(*shown))
        self._remember(ctx, proposal, content_digest=content_digest)

    async def _list(self, ctx: CommandContext, *, status: str | None = None) -> None:
        """Render the proposal table, optionally filtered by status."""
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
        table.add_column("Scope", style="cyan")
        table.add_column("Provenance", style="blue")
        table.add_column("Status", style="yellow")
        table.add_column("Observations", justify="right")
        table.add_column("Description", style="dim")
        for proposal in proposals:
            table.add_row(
                proposal.proposal_id,
                proposal.operation,
                proposal.skill_name,
                proposal.effective_scope,
                (
                    f"upstream: {proposal.upstream_source}"
                    if proposal.upstream_source
                    else "local"
                ),
                proposal.status,
                str(len(proposal.source_observation_ids)),
                proposal.description,
            )
        ctx.ui.mount_renderable(table)
        for proposal in proposals:
            self._remember(ctx, proposal)
        ctx.ui.append_system(
            f"Use {AUTOSKILLS_COMMAND} review <id> to inspect a draft, then "
            f"{AUTOSKILLS_COMMAND} approve <id> or "
            f"{AUTOSKILLS_COMMAND} reject <id>.",
            style="dim",
        )

    async def _approve(
        self,
        ctx: CommandContext,
        proposal_id: str | None,
        *,
        scope: str | None = None,
    ) -> None:
        """Approve a proposal, optionally into a given scope."""
        from ... import paths
        from ...memory.autoskills.proposals import approve_skill_proposal

        if not proposal_id:
            ctx.ui.append_system(
                f"Usage: {AUTOSKILLS_COMMAND} approve <id> [workspace|global]",
                style="yellow",
            )
            ctx.ui.append_system(
                f"Run {AUTOSKILLS_COMMAND} list to copy a proposal ID.",
                style="dim",
            )
            return
        workspace_dir = self._workspace_dir(ctx)
        snapshot = self._cached_snapshot(ctx, proposal_id)
        if snapshot is None:
            snapshot = await asyncio.to_thread(self._read_snapshot, ctx, proposal_id)
        result = await asyncio.to_thread(
            approve_skill_proposal,
            paths.MEMORIES_DIR,
            proposal_id,
            skills_dir=ctx.workspace.skills_dir,
            workspace_dir=workspace_dir,
            scope=scope,
            expected_updated_at=snapshot.updated_at if snapshot else None,
            expected_content_digest=snapshot.content_digest if snapshot else None,
        )
        if result.get("approved"):
            verb = "Updated" if result.get("operation") == "update" else "Approved"
            upstream = result.get("upstream_source")
            provenance_note = (
                f"; upstream provenance: {upstream} (a future reinstall may overwrite this change)"
                if upstream
                else ""
            )
            shadow = result.get("workspace_shadow")
            shadow_note = f"; workspace shadow remains at {shadow}" if shadow else ""
            ctx.ui.append_system(
                f"{verb} autoskill: {result['skill_name']} ({result['path']})"
                f"{provenance_note}{shadow_note}",
                style="green",
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
                f"Run {AUTOSKILLS_COMMAND} list to copy a proposal ID.",
                style="dim",
            )
            return
        workspace_dir = self._workspace_dir(ctx)
        snapshot = self._cached_snapshot(ctx, proposal_id)
        if snapshot is None:
            snapshot = await asyncio.to_thread(self._read_snapshot, ctx, proposal_id)
        result = await asyncio.to_thread(
            reject_skill_proposal,
            paths.MEMORIES_DIR,
            proposal_id,
            workspace_dir=workspace_dir,
            expected_updated_at=snapshot.updated_at if snapshot else None,
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
        from ...config.store import ConfigError

        try:
            saved = await asyncio.to_thread(set_config_value, key, value)
        except ConfigError as e:
            ctx.ui.append_system(f"Could not save to config: {e}", style="red")
            return
        if not saved:
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
        ctx.ui.append_system(
            f"Updated {self._config_label(key)} = {self._display_value(getattr(cfg, key))}",
            style="green",
        )
        from ...langgraph_dev.registration import apply_autoskills_settings

        note = await asyncio.to_thread(apply_autoskills_settings, cfg)
        if note is not None:
            ctx.ui.append_system(note, style="yellow")

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
        """Render the command reference."""
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
            (
                f"{AUTOSKILLS_COMMAND} review <id>",
                "Show one proposal's files for review",
            ),
            (
                f"{AUTOSKILLS_COMMAND} approve <id> [scope]",
                "Install a reviewed autoskill; scope is workspace or global",
            ),
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
    def _match_proposal(proposals: list[Any], proposal_id: str) -> Any | None:
        exact = [item for item in proposals if item.proposal_id == proposal_id]
        if len(exact) == 1:
            return exact[0]
        matches = [
            item for item in proposals if item.proposal_id.startswith(proposal_id)
        ]
        return matches[0] if len(matches) == 1 else None

    def _remember(
        self,
        ctx: CommandContext,
        proposal: Any,
        *,
        content_digest: str | None = None,
    ) -> None:
        if content_digest is None:
            content_digest = getattr(proposal, "content_digest", None)
        if not isinstance(content_digest, str):
            key = (self._workspace_dir(ctx), proposal.proposal_id)
            previous = self._snapshots.get(key)
            content_digest = (
                previous.content_digest
                if previous is not None and previous.updated_at == proposal.updated_at
                else None
            )
        key = (self._workspace_dir(ctx), proposal.proposal_id)
        self._snapshots[key] = _ProposalSnapshot(
            updated_at=proposal.updated_at,
            content_digest=content_digest,
        )

    def _cached_snapshot(
        self, ctx: CommandContext, proposal_id: str
    ) -> _ProposalSnapshot | None:
        workspace_dir = self._workspace_dir(ctx)
        exact = self._snapshots.get((workspace_dir, proposal_id))
        if exact is not None:
            return exact
        matches = [
            snapshot
            for (workspace, listed_id), snapshot in self._snapshots.items()
            if workspace == workspace_dir and listed_id.startswith(proposal_id)
        ]
        return matches[0] if len(matches) == 1 else None

    def _read_snapshot(
        self, ctx: CommandContext, proposal_id: str
    ) -> _ProposalSnapshot | None:
        from ... import paths
        from ...memory.autoskills.proposals import list_skill_proposals

        proposals = list_skill_proposals(
            paths.MEMORIES_DIR,
            workspace_dir=self._workspace_dir(ctx),
        )
        proposal = self._match_proposal(proposals, proposal_id)
        if proposal is None:
            return None
        self._remember(ctx, proposal)
        return self._snapshots[(self._workspace_dir(ctx), proposal.proposal_id)]

    @staticmethod
    def _workspace_dir(ctx: CommandContext) -> str:

        return str(ctx.workspace.root)

    @staticmethod
    def _first_arg(args: list[str]) -> str | None:
        """The first argument, or ``None`` when there is none."""
        return args[0] if args else None

    @staticmethod
    def _scope_arg(args: list[str]) -> tuple[str | None, str | None]:
        """The optional scope word that follows the proposal id.

        Returns ``(scope, error)``. A scope word the user typed but that is
        not a tier is an error, not a missing word: falling back to the
        scope the proposal recorded would install the skill somewhere the
        user did not ask for.
        """
        extra = args[1:]
        if not extra:
            return None, None
        if len(extra) > 1:
            return None, "expected one optional scope: workspace or global"
        word = extra[0].strip().lower()
        if word not in {"workspace", "global"}:
            return None, (f"unknown scope {extra[0]!r}: expected workspace or global")
        return word, None


manager.register(AutoSkillsCommand())
