"""Shared UI helpers for MCP server display and operations (used by the Typer `mcp` commands)."""

from typing import Any

from rich.table import Table

from ..stream.console import console


def _mcp_list_servers() -> None:
    """Print a table of configured MCP servers."""
    from ..mcp import load_mcp_config
    from ..mcp.client import USER_MCP_CONFIG

    config = load_mcp_config()

    if not config:
        console.print("[dim]No MCP servers configured.[/dim]")
        console.print(
            "[dim]Add one with:[/dim] /mcp add <name> <command-or-url> [args...]"
        )
        console.print()
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

    console.print(table)
    console.print(f"\n[dim]Config file: {USER_MCP_CONFIG}[/dim]")
    console.print()


def _mcp_add_server_from_kwargs(
    kwargs: dict[str, Any],
    *,
    show_reload_hint: bool = False,
) -> bool:
    """Add an MCP server from prepared kwargs."""
    from ..mcp import add_mcp_server

    try:
        entry = add_mcp_server(**kwargs)
        console.print(
            f"[green]Added MCP server:[/green] [cyan]{kwargs['name']}[/cyan] ({entry['transport']})"
        )
        if show_reload_hint:
            console.print("[dim]Reload with /new to apply.[/dim]")
        return True
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return False


def _mcp_edit_server_fields(
    name: str,
    fields: dict[str, Any],
    *,
    show_reload_hint: bool = False,
) -> bool:
    """Edit an MCP server from prepared field updates."""
    from ..mcp import edit_mcp_server

    if not fields:
        console.print(
            "[red]No fields to edit. Use --transport, --command, --url, --tools, --expose-to, etc.[/red]"
        )
        return False

    try:
        edit_mcp_server(name, **fields)
        console.print(f"[green]Updated MCP server:[/green] [cyan]{name}[/cyan]")
        for k, v in fields.items():
            console.print(f"  [dim]{k}:[/dim] {v}")
        if show_reload_hint:
            console.print("[dim]Reload with /new to apply.[/dim]")
        return True
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        return False
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return False


def _mcp_remove_server(name: str, *, show_reload_hint: bool = False) -> bool:
    """Remove an MCP server by name."""
    from ..mcp import remove_mcp_server

    clean_name = name.strip()
    if not clean_name:
        console.print("[red]Usage:[/red] /mcp remove <name>")
        return False

    if remove_mcp_server(clean_name):
        console.print(f"[green]Removed MCP server:[/green] [cyan]{clean_name}[/cyan]")
        if show_reload_hint:
            console.print("[dim]Reload with /new to apply.[/dim]")
        return True

    console.print(f"[red]Server not found:[/red] {clean_name}")
    return False


def _render_mcp_server_config_table(name: str, server: dict[str, Any]) -> None:
    """Render one MCP server config table."""
    table = Table(
        title=f"MCP Server: {name}",
        show_header=True,
        title_style="bold cyan",
    )
    table.add_column("Setting", style="cyan")
    table.add_column("Value")

    table.add_row("transport", str(server.get("transport", "(not set)")))
    if server.get("command"):
        table.add_row("command", str(server["command"]))
    if server.get("args"):
        table.add_row("args", " ".join(str(a) for a in server["args"]))
    if server.get("url"):
        table.add_row("url", str(server["url"]))
    if server.get("headers"):
        for k, v in server["headers"].items():
            table.add_row(f"header: {k}", str(v))
    if server.get("env"):
        for k, v in server["env"].items():
            table.add_row(f"env: {k}", str(v))

    tools = server.get("tools")
    table.add_row("tools", ", ".join(tools) if tools else "[dim](all)[/dim]")
    expose_to = server.get("expose_to", ["main"])
    if isinstance(expose_to, str):
        expose_to = [expose_to]
    table.add_row("expose_to", ", ".join(expose_to))

    console.print(table)
    console.print()


def _show_mcp_config(name: str = "", *, show_blank_line: bool = True) -> str:
    """Show MCP config details.

    Returns:
        "ok" when rendered, "empty" when no config exists, "missing" when
        a specific server name is requested but not found.
    """
    from ..mcp import load_mcp_config
    from ..mcp.client import USER_MCP_CONFIG

    config = load_mcp_config()
    if not config:
        console.print("[dim]No MCP servers configured.[/dim]")
        if show_blank_line:
            console.print()
        return "empty"

    name = name.strip()
    if name and name not in config:
        console.print(f"[red]Server not found:[/red] {name}")
        if show_blank_line:
            console.print()
        return "missing"

    servers = {name: config[name]} if name else config
    for srv_name, srv in servers.items():
        _render_mcp_server_config_table(srv_name, srv)

    console.print(f"[dim]Config file: {USER_MCP_CONFIG}[/dim]")
    if show_blank_line:
        console.print()
    return "ok"
