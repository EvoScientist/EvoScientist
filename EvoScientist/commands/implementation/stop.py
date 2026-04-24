from __future__ import annotations

from typing import ClassVar

from ..base import Command, CommandContext
from ..manager import manager


class StopCommand(Command):
    """Request the in-flight streaming response to terminate."""

    name = "/stop"
    alias: ClassVar[list[str]] = ["/cancel"]
    description = "Stop the current streaming response"

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        from ...stream.display import (
            is_stream_cancel_requested,
            request_stream_cancel,
        )

        if is_stream_cancel_requested():
            ctx.ui.append_system("Stop already requested.", style="dim")
            return
        request_stream_cancel()
        ctx.ui.append_system("Stop requested.", style="dim")


manager.register(StopCommand())
