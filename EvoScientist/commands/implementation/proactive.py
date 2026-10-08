"""``/proactive``: arm the current conversation for proactive pushes (#263).

``/proactive on`` lets EvoScientist send one unprompted message to this chat
once the conversation has gone quiet and something new has happened;
``/proactive off`` stops it; ``/proactive`` shows the state and the settings.

Only ``EvoSci serve`` can deliver a push, so only serve hands this command a
controller (``ChannelRuntime.proactive``). Elsewhere the command explains
that and changes nothing. In serve on the ``local`` backend, or after the dev
server was unavailable at startup, the controller carries the reason instead.
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from ..base import Command, CommandContext, SubCommand
from ..manager import manager

_SERVE_ONLY = (
    "Proactive follow-ups run under `EvoSci serve` and reach you through a "
    "channel; nothing changed."
)


def _controller(ctx: CommandContext) -> Any:
    return getattr(ctx.channel_runtime, "proactive", None)


def _settings_lines() -> list[str]:
    from ...config.settings import get_effective_config

    cfg = get_effective_config()
    quiet = cfg.proactive_quiet_hours or "off"
    zone = cfg.proactive_timezone or "host zone"
    return [
        f"  quiet for at least: {cfg.proactive_idle_minutes} min",
        f"  quiet hours: {quiet} ({zone})",
    ]


class ProactiveCommand(Command):
    """Arm or disarm proactive pushes for the current conversation."""

    name: ClassVar[str] = "/proactive"
    description: ClassVar[str] = (
        "Proactive follow-ups for this conversation: on, off, or status"
    )
    category: ClassVar[str] = "General"
    subcommands: ClassVar[list[SubCommand]] = [
        SubCommand("on", "Follow up on this conversation when it goes quiet"),
        SubCommand("off", "Stop proactive follow-ups for this conversation"),
    ]

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        controller = _controller(ctx)
        if controller is None:
            ctx.ui.append_system(_SERVE_ONLY, style="yellow")
            return
        if not controller.available:
            ctx.ui.append_system(
                f"Proactive follow-ups are not available: "
                f"{controller.unavailable_reason}.",
                style="yellow",
            )
            return

        action = args[0].lower() if args else ""
        if action not in ("", "on", "off"):
            ctx.ui.append_system("Usage: /proactive [on|off]", style="yellow")
            return
        thread_id = ctx.thread_id
        try:
            if action == "on":
                await asyncio.to_thread(controller.arm, thread_id)
            elif action == "off":
                await asyncio.to_thread(controller.disarm, thread_id)
            settings = await asyncio.to_thread(_settings_lines)
        except Exception as exc:
            ctx.ui.append_system(
                f"Could not change proactive follow-ups: {exc}", style="red"
            )
            return

        if action == "on":
            ctx.ui.append_system(
                "Proactive follow-ups are on for this conversation.", style="green"
            )
        elif action == "off":
            ctx.ui.append_system(
                "Proactive follow-ups are off for this conversation.", style="dim"
            )
        else:
            state = "on" if controller.is_armed(thread_id) else "off"
            ctx.ui.append_system(
                f"Proactive follow-ups: {state} for this conversation.", style="dim"
            )
        for line in settings:
            ctx.ui.append_system(line, style="dim")


manager.register(ProactiveCommand())
