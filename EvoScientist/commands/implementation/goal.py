"""``/goal``: a goal for the current thread, with drafted acceptance criteria.

The command never writes thread state. It reads the thread to say what can
be done, then leaves a request on the context; the surface starts a run that
carries only that request, and the graph drafts, asks for the review and
changes the goal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from rich.text import Text

from ...rubric_session import runs_in_process
from ..base import Command, SubCommand
from ..manager import manager
from .rubric import read_thread_values

if TYPE_CHECKING:
    from ..base import ChannelRuntime, CommandContext

_USAGE = (
    "Usage: /goal <objective>        set a goal; you review the drafted criteria",
    "       /goal                    show the goal (also: show, status)",
    "       /goal amend <feedback>   revise the goal and its criteria",
    "       /goal pause              stop working toward it and grading against it",
    "       /goal resume             continue a paused goal",
    "       /goal clear              drop the goal and its criteria",
)
_SHOW_WORDS = ("", "show", "status")
_STATUS_WORDS = ("pause", "resume", "clear")
_NO_GOAL = "No goal set for this thread."
_NOTE_LABELS = {"blocked": "Blocked", "complete": "Result"}


def _refusal(kind: str, status: str) -> str:
    """Why a goal with *status* cannot take the *kind* request."""
    if kind == "amend" and status == "paused":
        return "The goal is paused; resume it first with /goal resume."
    if kind == "amend":
        return f"The goal is {status}; set a new one with /goal <objective>."
    if kind == "pause" and status == "blocked":
        return "The goal is blocked and already waits for your message."
    wanted = {"pause": "an active", "resume": "a paused"}[kind]
    return f"The goal is {status}; only {wanted} goal can be {kind}d."


class GoalCommand(Command):
    """Set, show, amend, pause, resume or clear the goal of the current thread."""

    name = "/goal"
    description = "Set a goal with drafted acceptance criteria"
    category = "Session"
    subcommands: ClassVar[list[SubCommand]] = [
        SubCommand("show", "Show the goal and its criteria"),
        SubCommand("amend", "Revise the goal and its criteria"),
        SubCommand("pause", "Stop working toward the goal"),
        SubCommand("resume", "Continue a paused goal"),
        SubCommand("clear", "Drop the goal and its criteria"),
    ]

    def needs_agent(self, args: list[str]) -> bool:
        # Everything but ``show`` starts a run.
        return " ".join(args).strip().lower() not in _SHOW_WORDS

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        runtime = ctx.channel_runtime
        if runtime is None or not ctx.ui.supports_interactive:
            ctx.ui.append_system(
                "/goal is available in the terminal session only.", style="yellow"
            )
            return
        if not runs_in_process(ctx.graph_gateway):
            ctx.ui.append_system(
                "/goal needs the local gateway; this session executes its runs "
                "on the langgraph server.",
                style="yellow",
            )
            return

        # A subcommand is the whole remainder, so "/goal show me the money"
        # sets a goal; only ``amend`` takes text after it.
        raw = ctx.raw_args.strip()
        word = raw.lower()
        if word in _SHOW_WORDS:
            await self._show(ctx, runtime)
            return
        if ctx.agent is None:
            ctx.ui.append_system(
                "Agent not loaded yet; try /goal again in a moment.", style="yellow"
            )
            return

        from ...middleware.goal import (
            GOAL_REQUEST_STATUSES,
            GoalTextError,
            check_goal_text,
            new_goal_request,
        )

        review = not bool(getattr(ctx.config, "auto_mode", False))
        head, *tail = raw.split(None, 1)
        kind = word if word in _STATUS_WORDS else None
        feedback = None
        if kind is None and head.lower() == "amend":
            kind, feedback = "amend", (tail[0].strip() if tail else "")
            if not feedback:
                for line in _USAGE:
                    ctx.ui.append_system(line, style="yellow")
                return
        if kind is None:
            try:
                check_goal_text(raw)
            except GoalTextError as exc:
                ctx.ui.append_system(str(exc), style="red")
                return
            ctx.goal_request = new_goal_request("create", objective=raw, review=review)
            return

        values = await read_thread_values(ctx, "/goal")
        if values is None:
            ctx.ui.append_system(
                "The thread could not be read; nothing was done.", style="yellow"
            )
            return
        from ...rubric_session import thread_goal

        goal = thread_goal(values)
        if goal is None:
            ctx.ui.append_system(_NO_GOAL, style="yellow")
            return
        status = str(goal.get("status") or "")
        if status not in GOAL_REQUEST_STATUSES[kind]:
            ctx.ui.append_system(_refusal(kind, status), style="yellow")
            return
        ctx.goal_request = new_goal_request(
            kind, feedback=feedback, target=goal["rubric_id"], review=review
        )

    async def _show(self, ctx: CommandContext, runtime: ChannelRuntime) -> None:
        from ...rubric_session import thread_goal

        values = await read_thread_values(ctx, "/goal show")
        if values is None:
            ctx.ui.append_system(
                "Agent not loaded yet; the goal cannot be read."
                if ctx.agent is None
                else "The thread could not be read.",
                style="dim",
            )
            return
        goal = thread_goal(values)
        if goal is None:
            ctx.ui.append_system(_NO_GOAL, style="dim")
            for line in _USAGE:
                ctx.ui.append_system(line, style="dim")
            return
        status = str(goal.get("status") or "")
        # The objective, the note and the criteria are the user's or the
        # model's text: plain renderables, never markup.
        ctx.ui.append_system(f"Goal ({status})", style="bold")
        ctx.ui.mount_renderable(Text(str(goal.get("objective") or ""), style="dim"))
        note = str(goal.get("note") or "").strip()
        if note:
            label = _NOTE_LABELS.get(status, "Note")
            ctx.ui.mount_renderable(Text(f"{label}: {note}", style="dim"))
        ctx.ui.append_system("Acceptance criteria:", style="bold")
        criteria = (values.get("_sticky_rubric") or {}).get("text") or ""
        ctx.ui.mount_renderable(Text(str(criteria), style="dim"))
        # Operations the thread already applied are not "waiting".
        runtime.rubric.acknowledge(ctx.thread_id, values.get("_rubric_applied"))
        pending = runtime.rubric.changes_for(ctx.thread_id)
        if any("sticky" in operation for operation in pending):
            ctx.ui.append_system(
                "A /rubric change is waiting: the goal is cancelled with your "
                "next message.",
                style="yellow",
            )


manager.register(GoalCommand())
