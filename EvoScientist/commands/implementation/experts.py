"""Slash commands for TUI expert-skill selection.

``/experts`` — list installed expert skills.
``/expert <name>`` — toggle an expert into the current session's
``active_teams`` list; the next turn's ``configurable.active_teams`` picks
this up and ``ActiveTeamMiddleware`` biases the main-agent's delegation
toward the invited expert(s).
``/expert clear`` — reset the list.

User-facing verbs match the WebUI gallery: **invite** to add an expert,
**dismiss** to remove one. Internal state field stays ``active_teams``
for wire compatibility.

Backing store is ``ChannelRuntime.active_teams`` (see
``EvoScientist/commands/base.py``). WebUI users get the same effect via
its gallery + langgraph-sdk ``config.configurable``; these commands are
the TUI-side equivalent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from rich.table import Table

from ..base import Argument, Command, CommandContext, SubCommand
from ..manager import manager

if TYPE_CHECKING:
    from ...tools.skills_manager import SkillInfo

_dispatchable_experts_cache: list[SkillInfo] | None = None


def invalidate_experts_cache() -> None:
    """Reset the /expert dispatchable-experts cache.

    Called after ``install_skill`` / ``uninstall_skill`` mutations so a
    freshly installed expert shows up in the /expert popup on the next
    keystroke.
    """
    global _dispatchable_experts_cache
    _dispatchable_experts_cache = None


def _subscribe_cache_invalidation() -> None:
    """Register with ``skills_manager`` so every install/uninstall path
    (slash commands, agent ``skill_manager`` @tool, onboarding) busts
    the /expert popup — no caller has to remember.
    """
    try:
        from ...tools.skills_manager import register_skills_changed_callback

        register_skills_changed_callback(invalidate_experts_cache)
    except Exception:
        # ``skills_manager`` not importable in some early-init contexts;
        # cache staleness is a UX inconvenience, not a correctness bug.
        pass


_subscribe_cache_invalidation()


def _dispatchable_experts() -> list[SkillInfo]:
    """Cached list of experts that /expert can safely invite.

    Filters ``list_expert_skills`` down to those that pass the same
    empty-body + name-collision guards ``build_expert_subagent_specs``
    and ``_fold_expert_subagents`` apply at agent-construction time, so
    the /expert popup and invite-accept path only ever surface names
    that will actually reach ``ActiveTeamMiddleware``'s cue.
    """
    global _dispatchable_experts_cache
    if _dispatchable_experts_cache is None:
        try:
            from ...subagents.expert_container import list_dispatchable_experts

            _dispatchable_experts_cache = list_dispatchable_experts(include_system=True)
        except Exception:
            return []
    return _dispatchable_experts_cache


class ExpertsCommand(Command):
    """List installed expert skills."""

    name: ClassVar[str] = "/experts"
    description: ClassVar[str] = "List installed expert skills"
    category: ClassVar[str] = "Experts"

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        from ...tools.skills_manager import list_expert_skills

        experts = list_expert_skills(include_system=True)
        active = _current_active_teams(ctx)

        if not experts:
            ctx.ui.append_system("No expert skills installed.", style="dim")
            ctx.ui.append_system(
                "Install with: /install-skill <path-or-url>", style="dim"
            )
            return

        table = Table(title=f"Expert Skills ({len(experts)})", show_header=True)
        table.add_column("Name", style="cyan")
        table.add_column("Role", style="dim")
        table.add_column("Active", style="green")
        for skill in experts:
            marker = "*" if skill.name in active else ""
            table.add_row(
                skill.name,
                skill.role or skill.description,
                marker,
            )
        ctx.ui.mount_renderable(table)

        if active:
            ctx.ui.append_system(
                f"Active: {', '.join(active)}. Toggle with `/expert <name>`, "
                "clear with `/expert clear`.",
                style="dim",
            )
        else:
            ctx.ui.append_system(
                "No experts invited. `/expert <name>` to invite one.",
                style="dim",
            )


class ExpertCommand(Command):
    """Invite, dismiss, or clear expert skills for the current thread."""

    name: ClassVar[str] = "/expert"
    description: ClassVar[str] = "Invite or dismiss an expert skill"
    category: ClassVar[str] = "Experts"
    arguments: ClassVar[list[Argument]] = [
        Argument(
            name="name_or_clear",
            type=str,
            description="Expert skill name to toggle, or 'clear' to reset",
            required=True,
        )
    ]
    subcommands: ClassVar[list[SubCommand]] = [
        SubCommand("clear", "Dismiss all invited experts"),
    ]

    def _get_expert_candidates(self) -> list[tuple[str, str]]:
        return [(s.name, s.role or s.description) for s in _dispatchable_experts()]

    def get_completions(self, tokens: list[str]) -> list[tuple[str, str]]:
        """Complete expert names + the ``clear`` subcommand."""
        # /expert takes a single positional arg; anything past it (including a
        # trailing space that turns tokens into ["name", ""]) has nothing to offer.
        if len(tokens) > 1:
            return []
        prefix = tokens[0].lower() if tokens else ""
        candidates = [
            *self._get_expert_candidates(),
            ("clear", "Dismiss all invited experts"),
        ]
        matches = [
            (name, desc) for name, desc in candidates if name.lower().startswith(prefix)
        ]
        # Exact match — argument already complete, hide the popup.
        if len(matches) == 1 and matches[0][0].lower() == prefix:
            return []
        return matches

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        runtime = ctx.channel_runtime
        if runtime is None:
            ctx.ui.append_system(
                "/expert requires a session runtime; not available in this context.",
                style="yellow",
            )
            return

        if not args:
            ctx.ui.append_system(
                "Usage: /expert <name>   toggle an expert into the invited list",
                style="yellow",
            )
            ctx.ui.append_system(
                "       /expert clear    dismiss all invited experts",
                style="dim",
            )
            return

        target = args[0].strip()
        if target.lower() == "clear":
            if not runtime.active_teams:
                ctx.ui.append_system("No experts invited.", style="dim")
                return
            dismissed = list(runtime.active_teams)
            runtime.active_teams = []
            ctx.ui.append_system(
                f"Dismissed experts: {', '.join(dismissed)}", style="dim"
            )
            return

        # Completion matches case-insensitively; honour the same here by
        # resolving a case-variant to the on-disk name before membership.
        by_lower = {s.name.lower(): s.name for s in _dispatchable_experts()}
        canonical = by_lower.get(target.lower())
        if canonical is None:
            from ...tools.skills_manager import list_expert_skills

            installed = {
                s.name.lower() for s in list_expert_skills(include_system=True)
            }
            if target.lower() not in installed:
                ctx.ui.append_system(
                    f"No expert skill named '{target}'. `/experts` lists "
                    "installed ones.",
                    style="red",
                )
            else:
                ctx.ui.append_system(
                    f"Expert '{target}' can't be dispatched (empty actor "
                    "definition or name collision with a built-in sub-agent).",
                    style="red",
                )
            return

        if canonical in runtime.active_teams:
            runtime.active_teams = [n for n in runtime.active_teams if n != canonical]
            ctx.ui.append_system(f"Dismissed expert: {canonical}", style="dim")
        else:
            runtime.active_teams = [*runtime.active_teams, canonical]
            ctx.ui.append_system(f"Invited expert: {canonical}", style="green")
            # Case (c): expert was installed after agent construction, so it
            # will not reach ``task()`` until the graph is rebuilt. Cheap
            # always-print hint mirrors the /install-skill success message.
            ctx.ui.append_system(
                "If just installed, run /new to activate it.", style="dim"
            )
        if runtime.active_teams:
            ctx.ui.append_system(
                f"Active: {', '.join(runtime.active_teams)}", style="dim"
            )


def _current_active_teams(ctx: CommandContext) -> list[str]:
    runtime = ctx.channel_runtime
    return list(runtime.active_teams) if runtime is not None else []


manager.register(ExpertsCommand())
manager.register(ExpertCommand())
