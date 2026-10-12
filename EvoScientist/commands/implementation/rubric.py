"""``/rubric``: acceptance criteria for the current thread.

The command records operations in the session (``ChannelRuntime.rubric``);
the next message the user sends carries them to the grader, which stores the
checklists on the thread and grades every marked run against them.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from rich.text import Text

from ...rubric_session import runs_in_process
from ..base import Command, SubCommand
from ..manager import manager

if TYPE_CHECKING:
    from ..base import ChannelRuntime, CommandContext

logger = logging.getLogger(__name__)

_USAGE = (
    "Usage: /rubric set <criteria | @file>    grade every turn from the next message on",
    "       /rubric next <criteria | @file>   grade only the next task",
    "       /rubric show                      show the checklists",
    "       /rubric clear                     remove both checklists",
)


async def read_thread_values(ctx: CommandContext, who: str) -> dict | None:
    """The thread's state values, or ``None`` when they cannot be read now."""
    if ctx.agent is None or ctx.graph_gateway is None:
        return None
    from ...gateway import GraphTarget

    target = GraphTarget(local_graph=ctx.agent, **ctx.dirs.metadata())
    try:
        return await ctx.graph_gateway.get_state_values(target, ctx.thread_id) or {}
    except Exception:
        logger.debug("%s: thread state unavailable", who, exc_info=True)
        return None


async def _note_goal(ctx: CommandContext) -> None:
    """Say that replacing the sticky checklist also ends the thread's goal."""
    from ...rubric_session import thread_goal

    if thread_goal(await read_thread_values(ctx, "/rubric")) is not None:
        ctx.ui.append_system(
            "This also cancels the thread's goal (see /goal).", style="yellow"
        )


def _text_after_subcommand(raw_args: str) -> str:
    parts = raw_args.split(None, 1)
    text = parts[1].strip() if len(parts) > 1 else ""
    # A quoted file name: @"criteria space.md". Inline criteria keep their quotes.
    if len(text) >= 3 and text[0] == "@" and text[1] in "\"'" and text[-1] == text[1]:
        return "@" + text[2:-1]
    return text


class RubricCommand(Command):
    """Set, show or clear the acceptance criteria of the current thread."""

    name = "/rubric"
    description = "Set acceptance criteria for this thread"
    category = "Session"
    subcommands: ClassVar[list[SubCommand]] = [
        SubCommand("set", "Grade every turn from the next message on"),
        SubCommand("next", "Grade only the next task"),
        SubCommand("show", "Show the checklists"),
        SubCommand("clear", "Remove both checklists"),
    ]

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        runtime = ctx.channel_runtime
        if runtime is None or not ctx.ui.supports_interactive:
            ctx.ui.append_system(
                "/rubric is available in the terminal session only.", style="yellow"
            )
            return
        if not runs_in_process(ctx.graph_gateway):
            ctx.ui.append_system(
                "/rubric needs the local gateway; this session executes its runs "
                "on the langgraph server.",
                style="yellow",
            )
            return

        sub = args[0].lower() if args else "show"
        if sub == "show":
            await self._show(ctx, runtime)
            return
        if sub == "clear":
            runtime.rubric.record(ctx.thread_id, sticky=None, next=None)
            ctx.ui.append_system(
                "Acceptance criteria cleared; applies from your next message.",
                style="dim",
            )
            await _note_goal(ctx)
            return
        text = _text_after_subcommand(ctx.raw_args)
        if sub not in ("set", "next") or not text:
            for line in _USAGE:
                ctx.ui.append_system(line, style="yellow")
            return

        from ...rubric_text import RubricTextError, read_rubric_text

        try:
            criteria = read_rubric_text(text, base_dir=ctx.work_dir)
        except RubricTextError as exc:
            # A path the user typed: shown as it is in both UIs.
            ctx.ui.mount_renderable(Text(f"/rubric {sub} {exc}", style="red"))
            return
        if sub == "set":
            runtime.rubric.record(ctx.thread_id, sticky=criteria)
            ctx.ui.append_system(
                "Acceptance criteria set for every turn from your next message on.",
                style="dim",
            )
            await _note_goal(ctx)
        else:
            runtime.rubric.record(ctx.thread_id, next=criteria)
            ctx.ui.append_system(
                "Acceptance criteria set for the next task; they apply from your "
                "next message.",
                style="dim",
            )

    async def _show(self, ctx: CommandContext, runtime: ChannelRuntime) -> None:
        pending = runtime.rubric.changes_for(ctx.thread_id)
        values = await read_thread_values(ctx, "/rubric show")
        if values is None:
            unreadable = (
                "Agent not loaded yet"
                if ctx.agent is None
                else "Thread state could not be read"
            )
            ctx.ui.append_system(
                f"{unreadable}; showing only the changes waiting for your next "
                "message.",
                style="dim",
            )
        else:
            # Operations the thread already applied are not "waiting".
            runtime.rubric.acknowledge(ctx.thread_id, values.get("_rubric_applied"))
            pending = runtime.rubric.changes_for(ctx.thread_id)
        sticky = (values or {}).get("_sticky_rubric") or None
        next_ = (values or {}).get("_next_rubric") or None
        for operation in pending:
            if "sticky" in operation:
                sticky = {"text": operation["sticky"]} if operation["sticky"] else None
            if "next" in operation:
                next_ = {"text": operation["next"]} if operation["next"] else None
        # Checklist text is the user's own: a plain renderable, never markup.
        if sticky:
            ctx.ui.append_system("Every turn:", style="bold")
            ctx.ui.mount_renderable(Text(str(sticky.get("text", "")), style="dim"))
        if next_:
            ctx.ui.append_system("Next task only:", style="bold")
            ctx.ui.mount_renderable(Text(str(next_.get("text", "")), style="dim"))
        if not sticky and not next_:
            ctx.ui.append_system(
                "No acceptance criteria set for this thread.", style="dim"
            )
        if pending:
            count = len(pending)
            ctx.ui.append_system(
                f"({count} change{'s' if count != 1 else ''} not yet applied; "
                "they apply with your next message)",
                style="dim",
            )


manager.register(RubricCommand())
