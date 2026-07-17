from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

from ..base import Argument, Command, CommandContext


class InstallMCPCommand(Command):
    """Browse and install MCP servers."""

    name = "/install-mcp"
    description = "Browse and install MCP servers"
    category = "MCP"
    arguments: ClassVar[list[Argument]] = [
        Argument(
            name="source",
            type=str,
            description="Server name or tag filter",
            required=False,
        )
    ]

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        import asyncio
        import threading

        from ...mcp.registry import (
            fetch_marketplace_index,
            find_server_by_name,
            get_all_tags,
            get_installed_names,
            install_mcp_server,
            install_mcp_servers,
        )

        source = args[0] if args else ""

        ctx.ui.append_system("Fetching MCP server index...", style="dim")
        await ctx.ui.flush()
        try:
            servers = await asyncio.to_thread(fetch_marketplace_index)
        except Exception as e:
            ctx.ui.append_system(f"Failed to fetch server index: {e}", style="red")
            return

        if not servers:
            ctx.ui.append_system("No MCP servers found.", style="yellow")
            return

        # Direct name match
        if source:
            match = find_server_by_name(source, servers)
            if match:
                installed = get_installed_names()
                if match.name in installed:
                    ctx.ui.append_system(
                        f"{match.name} is already configured.", style="yellow"
                    )
                    return

                cancel_event = threading.Event()
                committed: list[str] = []
                committed_lock = threading.Lock()

                def _commit_gate(name: str, write: Callable[[], bool]) -> bool:
                    # The write and its ledger record happen under one lock, so
                    # a cancellation lands before the write (nothing applied) or
                    # after a recorded success; a failed write records nothing.
                    with committed_lock:
                        if cancel_event.is_set():
                            return False
                        if not write():
                            return False
                        committed.append(name)
                        return True

                def _install_one():
                    messages: list[tuple[str, str]] = []

                    def _collect(text: str, style: str = "") -> None:
                        messages.append((text, style))

                    ok = install_mcp_server(
                        match,
                        print_fn=_collect,
                        commit_gate=_commit_gate,
                    )
                    return ok, messages

                try:
                    installed_ok, messages = await asyncio.to_thread(_install_one)
                except asyncio.CancelledError:
                    with committed_lock:
                        cancel_event.set()
                        already = list(committed)
                    if already:
                        ctx.ui.append_system(
                            f"{match.name} finished installing before the cancel "
                            "and was applied.",
                            style="yellow",
                        )
                    else:
                        ctx.ui.append_system(
                            f"Cancelled before {match.name} was applied. Nothing "
                            "will be written; any in-flight download finishes in "
                            "the background and is safe to retry.",
                            style="yellow",
                        )
                    await ctx.ui.flush()
                    raise
                for text, style in messages:
                    ctx.ui.append_system(text, style=style)
                if installed_ok:
                    ctx.ui.append_system(f"Configured: {match.name}", style="green")
                    ctx.ui.append_system("Reload with /new to apply.", style="dim")
                else:
                    ctx.ui.append_system(
                        f"Failed to configure {match.name}.", style="red"
                    )
                return

            # Check if it's a tag — fall through to browser
            if source.lower() not in get_all_tags(servers):
                ctx.ui.append_system(
                    f"No server or tag found matching: {source}", style="red"
                )
                close = [s.name for s in servers if source.lower() in s.name.lower()]
                if close:
                    ctx.ui.append_system(
                        f"Did you mean: {', '.join(close)}?", style="dim"
                    )
                return

        # Interactive browse (or pre-filtered by tag)
        installed_names = get_installed_names()

        selected_entries = await ctx.ui.wait_for_mcp_browse(
            servers, installed_names, pre_filter_tag=source
        )

        if selected_entries is None:
            ctx.ui.append_system("Browse cancelled.", style="dim")
            return
        if not selected_entries:
            ctx.ui.append_system("No servers selected.", style="dim")
            return

        requested = [entry.name for entry in selected_entries]
        cancel_event = threading.Event()
        committed: list[str] = []
        committed_lock = threading.Lock()

        def _commit_gate(name: str, write: Callable[[], bool]) -> bool:
            with committed_lock:
                if cancel_event.is_set():
                    return False
                if not write():
                    return False
                committed.append(name)
                return True

        def _install_selected():
            messages: list[tuple[str, str]] = []

            def _collect(text: str, style: str = "") -> None:
                messages.append((text, style))

            count = install_mcp_servers(
                selected_entries,
                print_fn=_collect,
                cancel_event=cancel_event,
                commit_gate=_commit_gate,
            )
            return count, messages

        try:
            count, messages = await asyncio.to_thread(_install_selected)
        except asyncio.CancelledError:
            cancel_event.set()
            with committed_lock:
                done = list(committed)
            not_done = [name for name in requested if name not in done]
            if not done:
                ctx.ui.append_system(
                    "Cancelled before any server was installed. Nothing will be "
                    "written; any in-flight download finishes in the background "
                    "and is safe to retry.",
                    style="yellow",
                )
            else:
                ctx.ui.append_system(
                    f"Cancelled partway. Installed before the cancel: "
                    f"{', '.join(done)}.",
                    style="yellow",
                )
                if not_done:
                    ctx.ui.append_system(
                        f"Not installed: {', '.join(not_done)}.",
                        style="dim",
                    )
            await ctx.ui.flush()
            raise
        for text, style in messages:
            ctx.ui.append_system(text, style=style)
        if count:
            ctx.ui.append_system(
                f"{count} server(s) configured. Reload with /new to apply.",
                style="green",
            )
