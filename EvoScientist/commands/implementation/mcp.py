from __future__ import annotations

from rich.table import Table

from ..base import Command, CommandContext
from ..manager import manager


class MCPCommand(Command):
    """Manage MCP servers."""

    name = "/mcp"
    description = "Manage MCP servers"

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:

        if not args or args[0] == "list":
            await self._mcp_list(ctx)
            return

        subcmd = args[0].lower()
        subargs = args[1:]

        if subcmd == "config":
            await self._mcp_config(ctx, subargs[0] if subargs else "")
        elif subcmd == "add":
            await self._mcp_add(ctx, subargs)
        elif subcmd == "edit":
            await self._mcp_edit(ctx, subargs)
        elif subcmd == "remove":
            await self._mcp_remove(ctx, subargs[0] if subargs else "")
        elif subcmd == "install":
            from .mcp_install import InstallMCPCommand
            await InstallMCPCommand().execute(ctx, subargs)
        else:
            ctx.ui.append_system("MCP commands:", style="bold")
            ctx.ui.append_system(
                "  /mcp              List configured servers", style="dim"
            )
            ctx.ui.append_system(
                "  /mcp list         List configured servers", style="dim"
            )
            ctx.ui.append_system(
                "  /mcp config       Show detailed server config", style="dim"
            )
            ctx.ui.append_system("  /mcp add ...      Add a server", style="dim")
            ctx.ui.append_system(
                "  /mcp edit ...     Edit an existing server", style="dim"
            )
            ctx.ui.append_system("  /mcp remove ...   Remove a server", style="dim")
            ctx.ui.append_system("  /mcp install ...  Browse and install servers", style="dim")

    async def _mcp_list(self, ctx: CommandContext) -> None:
        from ...mcp import load_mcp_config
        from ...mcp.client import USER_MCP_CONFIG

        config = load_mcp_config()
        if not config:
            ctx.ui.append_system("No MCP servers configured.", style="dim")
            ctx.ui.append_system(
                "Add one with: /mcp add <name> <command-or-url> [args...]",
                style="dim",
            )
            return

        table = Table(title="MCP Servers", show_header=True)
        table.add_column("Server", style="cyan")
        table.add_column("Transport", style="green")
        table.add_column("Tools", style="yellow")
        table.add_column("Expose To", style="magenta")

        for name, server in config.items():
            transport = server.get("transport", "?")
            tools = server.get("tools")
            tools_str = ", ".join(tools) if tools else "(all)"
            expose_to = server.get("expose_to", ["main"])
            if isinstance(expose_to, str):
                expose_to = [expose_to]
            expose_str = ", ".join(expose_to)
            table.add_row(name, transport, tools_str, expose_str)

        ctx.ui.mount_renderable(table)
        ctx.ui.append_system(f"Config file: {USER_MCP_CONFIG}", style="dim")

    async def _mcp_config(self, ctx: CommandContext, name: str) -> None:
        from ...mcp import load_mcp_config
        from ...mcp.client import USER_MCP_CONFIG

        config = load_mcp_config()
        if not config:
            ctx.ui.append_system("No MCP servers configured.", style="dim")
            return

        if name and name not in config:
            ctx.ui.append_system(f"Server not found: {name}", style="red")
            return

        servers = {name: config[name]} if name else config
        for srv_name, srv in servers.items():
            table = Table(
                title=f"MCP Server: {srv_name}",
                show_header=True,
                title_style="bold cyan",
            )
            table.add_column("Setting", style="cyan")
            table.add_column("Value")

            table.add_row("transport", str(srv.get("transport", "(not set)")))
            if srv.get("command"):
                table.add_row("command", str(srv["command"]))
            if srv.get("args"):
                table.add_row("args", " ".join(str(a) for a in srv["args"]))
            if srv.get("url"):
                table.add_row("url", str(srv["url"]))
            if srv.get("headers"):
                headers_str = ", ".join(f"{k}: {v}" for k, v in srv["headers"].items())
                table.add_row("headers", headers_str)
            if srv.get("env"):
                env_str = ", ".join(f"{k}={v}" for k, v in srv["env"].items())
                table.add_row("env", env_str)

            tools = srv.get("tools")
            table.add_row("tools", ", ".join(tools) if tools else "[dim](all)[/dim]")
            expose_to = srv.get("expose_to", ["main"])
            if isinstance(expose_to, str):
                expose_to = [expose_to]
            table.add_row("expose_to", ", ".join(expose_to))

            ctx.ui.mount_renderable(table)

        ctx.ui.append_system(f"Config file: {USER_MCP_CONFIG}", style="dim")

    async def _mcp_add(self, ctx: CommandContext, tokens: list[str]) -> None:
        from ...mcp import add_mcp_server, parse_mcp_add_args

        if not tokens:
            ctx.ui.append_system(
                "Usage: /mcp add <name> <command-or-url> [args...]", style="yellow"
            )
            return

        try:
            kwargs = parse_mcp_add_args(tokens)
            entry = add_mcp_server(**kwargs)
            ctx.ui.append_system(
                f"Added MCP server: {kwargs['name']} ({entry['transport']})",
                style="green",
            )
            ctx.ui.append_system("Reload with /new to apply.", style="dim")
        except Exception as exc:
            ctx.ui.append_system(f"Error: {exc}", style="red")

    async def _mcp_edit(self, ctx: CommandContext, tokens: list[str]) -> None:
        from ...mcp import edit_mcp_server, parse_mcp_edit_args

        if not tokens:
            ctx.ui.append_system(
                "Usage: /mcp edit <name> --<field> <value> ...", style="yellow"
            )
            return

        try:
            name, fields = parse_mcp_edit_args(tokens)
            edit_mcp_server(name, **fields)
            ctx.ui.append_system(f"Updated MCP server: {name}", style="green")
            ctx.ui.append_system("Reload with /new to apply.", style="dim")
        except Exception as exc:
            ctx.ui.append_system(f"Error: {exc}", style="red")

    async def _mcp_remove(self, ctx: CommandContext, name: str) -> None:
        from ...mcp import remove_mcp_server

        if not name:
            ctx.ui.append_system("Usage: /mcp remove <name>", style="yellow")
            return

        if remove_mcp_server(name):
            ctx.ui.append_system(f"Removed MCP server: {name}", style="green")
            ctx.ui.append_system("Reload with /new to apply.", style="dim")
        else:
            ctx.ui.append_system(f"Server not found: {name}", style="red")


# Register MCP command
manager.register(MCPCommand())
