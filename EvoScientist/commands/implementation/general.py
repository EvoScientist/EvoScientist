from __future__ import annotations

from rich.text import Text

from ..base import Command, CommandContext
from ..manager import manager


class HelpCommand(Command):
    """Show available commands."""

    name = "/help"
    description = "Show available commands"

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        help_text = Text("Available commands:\n", style="bold")
        for cmd in sorted(manager.get_all_commands(), key=lambda c: c.name):
            # Build usage hint from arguments
            usage_parts = [cmd.name]
            for arg in cmd.arguments:
                if arg.required:
                    usage_parts.append(f"<{arg.name}>")
                else:
                    usage_parts.append(f"[{arg.name}]")

            usage = " ".join(usage_parts)
            help_text.append(f"  {usage:<30}", style="cyan")
            desc = cmd.description
            if cmd.alias:
                desc += f" (aliases: {', '.join(cmd.alias)})"
            help_text.append(f"{desc}\n", style="dim")
        ctx.ui.mount_renderable(help_text)


class CurrentCommand(Command):
    """Show current session info."""

    name = "/current"
    description = "Show current session info"

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        from ... import paths

        ctx.ui.append_system(f"Thread: {ctx.thread_id}", style="dim")
        if ctx.workspace_dir:
            from ...cli.agent import _shorten_path

            ctx.ui.append_system(
                f"Workspace: {_shorten_path(ctx.workspace_dir)}",
                style="dim",
            )
        memory_path = paths.MEMORIES_DIR
        if memory_path:
            from ...cli.agent import _shorten_path

            ctx.ui.append_system(
                f"Memory dir: {_shorten_path(str(memory_path))}", style="dim"
            )


# Register commands
manager.register(HelpCommand())
manager.register(CurrentCommand())
