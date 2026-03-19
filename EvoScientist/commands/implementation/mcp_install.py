from __future__ import annotations

from ..base import Argument, Command, CommandContext


class InstallMCPCommand(Command):
    """Browse and install MCP servers."""

    name = "/install-mcp"
    description = "Browse and install MCP servers"
    arguments = [
        Argument(
            name="source",
            type=str,
            description="Server name or tag filter",
            required=False,
        )
    ]

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
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
            import asyncio

            servers = await asyncio.get_event_loop().run_in_executor(
                None, fetch_marketplace_index
            )
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
                if install_mcp_server(
                    match, print_fn=ctx.ui.append_system
                ):
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

        count = install_mcp_servers(
            selected_entries, print_fn=ctx.ui.append_system
        )
        if count:
            ctx.ui.append_system(
                f"{count} server(s) configured. Reload with /new to apply.",
                style="green",
            )
