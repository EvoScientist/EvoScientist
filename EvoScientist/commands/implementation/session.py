from __future__ import annotations

from typing import ClassVar

from rich.table import Table

from ..base import Argument, Command, CommandContext
from ..manager import manager


class CompactCommand(Command):
    """Compact conversation to free context."""

    name = "/compact"
    description = "Compact conversation to free context"

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        from ...cli.commands import compact_conversation, render_compact_result

        ctx.ui.append_system("Compacting conversation...")
        result = await compact_conversation(
            agent=ctx.agent,
            thread_id=ctx.thread_id,
        )
        ctx.ui.mount_renderable(render_compact_result(result))


class ThreadsCommand(Command):
    """List recent sessions."""

    name = "/threads"
    description = "List recent sessions"

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        from ...sessions import _format_relative_time, list_threads

        threads = await list_threads(
            limit=0,
            include_message_count=True,
            include_preview=True,
        )

        if not threads:
            ctx.ui.append_system("No saved sessions.", style="yellow")
            return

        # Use protocol property to adapt output for non-interactive UIs (channels)
        is_channel = not ctx.ui.supports_interactive

        table = Table(title="Sessions", show_header=True, header_style="bold cyan")
        table.add_column("ID", style="bold")
        table.add_column(
            "Preview", style="dim", max_width=40 if is_channel else 50, no_wrap=True
        )
        table.add_column("Msgs" if is_channel else "Messages", justify="right")
        if not is_channel:
            table.add_column("Model", style="dim")
        table.add_column("Last Used", style="dim")

        for thread in threads:
            thread_id_value = thread["thread_id"]
            marker = " *" if thread_id_value == ctx.thread_id else ""

            row = [
                f"{thread_id_value}{marker}",
                thread.get("preview", "") or "",
                str(thread.get("message_count", 0)),
            ]
            if not is_channel:
                row.append(thread.get("model", "") or "")
            row.append(_format_relative_time(thread.get("updated_at")))

            table.add_row(*row)
        ctx.ui.mount_renderable(table)


class ResumeCommand(Command):
    """Resume a previous session."""

    name = "/resume"
    description = "Resume a previous session"
    arguments: ClassVar[list[Argument]] = [
        Argument(
            name="thread_id",
            type=str,
            description="Thread ID or prefix to resume",
            required=False,
        )
    ]

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:

        from ...sessions import (
            get_thread_metadata,
            list_threads,
        )

        arg = args[0] if args else ""
        if not arg:
            threads = await list_threads(
                limit=0,
                include_message_count=True,
                include_preview=True,
            )
            if not threads:
                ctx.ui.append_system("No sessions to resume.", style="yellow")
                return

            # Interactive pick
            selected = await ctx.ui.wait_for_thread_pick(
                threads,
                current_thread=ctx.thread_id,
                title=">>> Select session to resume <<<",
            )
            if selected is None:
                return
            arg = selected

        # Resolve thread_id
        resolved = await self._resolve_thread_id(arg, ctx)
        if not resolved:
            return

        metadata = await get_thread_metadata(resolved)
        restored_workspace = (metadata or {}).get("workspace_dir", "")
        if restored_workspace:
            ctx.workspace_dir = restored_workspace

        ctx.thread_id = resolved

        # Signal session change to UI
        if hasattr(ctx.ui, "handle_session_resume"):
            await ctx.ui.handle_session_resume(resolved, restored_workspace)

    async def _resolve_thread_id(self, prefix: str, ctx: CommandContext) -> str | None:
        from ...sessions import find_similar_threads, thread_exists

        if await thread_exists(prefix):
            return prefix

        similar = await find_similar_threads(prefix)
        if len(similar) == 1:
            return similar[0]

        if len(similar) > 1:
            ctx.ui.append_system(
                f"Ambiguous thread ID '{prefix}'. Use a longer prefix.",
                style="yellow",
            )
            for thread in similar:
                ctx.ui.append_system(f"  - {thread}", style="dim")
            return None

        ctx.ui.append_system(f"Thread '{prefix}' not found.", style="red")
        return None


class NewCommand(Command):
    """Start a new session."""

    name = "/new"
    description = "Start a new session"

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        ctx.ui.start_new_session()


class ClearCommand(Command):
    """Clear chat history."""

    name = "/clear"
    description = "Clear chat history"

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        ctx.ui.clear_chat()


class DeleteCommand(Command):
    """Delete a saved session."""

    name = "/delete"
    description = "Delete a saved session"
    arguments: ClassVar[list[Argument]] = [
        Argument(
            name="thread_id",
            type=str,
            description="Thread ID or prefix to delete",
            required=False,
        )
    ]

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        from ...sessions import (
            delete_thread,
            find_similar_threads,
            list_threads,
            thread_exists,
        )

        arg = args[0] if args else ""
        if not arg:
            threads = await list_threads(
                limit=0,
                include_message_count=True,
                include_preview=True,
            )
            if not threads:
                ctx.ui.append_system("No sessions to delete.", style="yellow")
                return

            # Interactive pick
            selected = await ctx.ui.wait_for_thread_pick(
                threads,
                current_thread=ctx.thread_id,
                title=">>> Select session to delete <<<",
            )
            if selected is None:
                return
            arg = selected

        # Resolve thread_id
        resolved = None
        if await thread_exists(arg):
            resolved = arg
        else:
            similar = await find_similar_threads(arg)
            if len(similar) == 1:
                resolved = similar[0]
            elif len(similar) > 1:
                ctx.ui.append_system(
                    f"Ambiguous thread ID '{arg}'. Use a longer prefix.",
                    style="yellow",
                )
                for thread in similar:
                    ctx.ui.append_system(f"  - {thread}", style="dim")
                return

        if not resolved:
            ctx.ui.append_system(f"Session '{arg}' not found.", style="red")
            return

        if resolved == ctx.thread_id:
            ctx.ui.append_system(
                "Cannot delete the current session.",
                style="yellow",
            )
            return

        deleted = await delete_thread(resolved)
        if deleted:
            ctx.ui.append_system(f"Deleted session {resolved}.", style="green")
        else:
            ctx.ui.append_system(f"Session {resolved} not found.", style="red")


class ExitCommand(Command):
    """Quit EvoScientist."""

    name = "/exit"
    alias: ClassVar[list[str]] = ["/quit", "/q"]
    description = "Quit EvoScientist"

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        ctx.ui.force_quit()


# Register session commands
manager.register(CompactCommand())
manager.register(ThreadsCommand())
manager.register(ResumeCommand())
manager.register(NewCommand())
manager.register(ClearCommand())
manager.register(DeleteCommand())
manager.register(ExitCommand())
