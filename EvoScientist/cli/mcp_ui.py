"""MCP server display, operations, and /mcp slash-command dispatcher."""

from typing import Any

from rich.table import Table

from ..stream.display import console


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


def _cmd_mcp_add(args_str: str) -> None:
    """Handle ``/mcp add ...``."""
    import shlex

    from ..mcp import parse_mcp_add_args

    if not args_str.strip():
        console.print("[bold]Usage:[/bold] /mcp add <name> <command-or-url> [args...]")
        console.print()
        console.print(
            "[dim]Transport is auto-detected: URLs \u2192 http, commands \u2192 stdio[/dim]"
        )
        console.print()
        console.print("[bold]Examples:[/bold]")
        console.print(
            "  /mcp add sequential-thinking npx -y @modelcontextprotocol/server-sequential-thinking"
        )
        console.print("  /mcp add docs-langchain https://docs.langchain.com/mcp")
        console.print(
            "  /mcp add my-sse http://localhost:9090/sse --transport sse --expose-to research-agent"
        )
        console.print()
        console.print("[dim]Options:[/dim]")
        console.print("  --transport T          Transport type (default: auto-detect)")
        console.print(
            "  --tools t1,t2          Tool allowlist (supports wildcards: *_exa, read_*)"
        )
        console.print("  --expose-to a1,a2      Target agents (default: main)")
        console.print("  --header Key:Value     HTTP header (repeatable)")
        console.print("  --env KEY=VALUE        Env var for stdio (repeatable)")
        console.print(
            "  --env-ref KEY          Env var as runtime ${KEY} reference (repeatable)"
        )
        console.print()
        return

    try:
        tokens = shlex.split(args_str)
        kwargs = parse_mcp_add_args(tokens)
        _mcp_add_server_from_kwargs(kwargs, show_reload_hint=True)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
    console.print()


def _cmd_mcp_edit(args_str: str) -> None:
    """Handle ``/mcp edit <name> --field value ...``."""
    import shlex

    from ..mcp import parse_mcp_edit_args

    if not args_str.strip():
        console.print("[bold]Usage:[/bold] /mcp edit <name> --<field> <value> ...")
        console.print()
        console.print(
            "[dim]Fields:[/dim] --transport, --command, --url, --args, --tools, --expose-to, --header, --env"
        )
        console.print(
            "[dim]Use[/dim] --tools none [dim]or[/dim] --expose-to none [dim]to clear a field.[/dim]"
        )
        console.print()
        console.print("[bold]Examples:[/bold]")
        console.print("  /mcp edit filesystem --expose-to main,code-agent")
        console.print("  /mcp edit filesystem --tools read_file,write_file")
        console.print("  /mcp edit my-api --url http://new-host:8080/mcp")
        console.print("  /mcp edit my-api --tools none")
        console.print()
        return

    try:
        tokens = shlex.split(args_str)
        name, fields = parse_mcp_edit_args(tokens)
        _mcp_edit_server_fields(name, fields, show_reload_hint=True)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
    console.print()


def _cmd_mcp_remove(name: str) -> None:
    """Handle ``/mcp remove <name>``."""
    _mcp_remove_server(name, show_reload_hint=True)
    console.print()


def _cmd_mcp_config(name: str) -> None:
    """Handle ``/mcp config [name]``."""
    _show_mcp_config(name, show_blank_line=True)


def _cmd_mcp(args: str) -> None:
    """Dispatch ``/mcp`` subcommands."""
    args = args.strip()
    if not args:
        _mcp_list_servers()
        return

    parts = args.split(maxsplit=1)
    subcmd = parts[0].lower()
    subargs = parts[1] if len(parts) > 1 else ""

    if subcmd == "list":
        _mcp_list_servers()
    elif subcmd == "add":
        _cmd_mcp_add(subargs)
    elif subcmd == "edit":
        _cmd_mcp_edit(subargs)
    elif subcmd == "remove":
        _cmd_mcp_remove(subargs)
    elif subcmd == "config":
        _cmd_mcp_config(subargs)
    elif subcmd == "install":
        from .mcp_install_cmd import _cmd_install_mcp

        _cmd_install_mcp(subargs)
    else:
        console.print("[bold]MCP commands:[/bold]")
        console.print("  /mcp              List configured servers")
        console.print("  /mcp list         List configured servers")
        console.print("  /mcp config       Show detailed server config")
        console.print("  /mcp add ...      Add a server")
        console.print("  /mcp edit ...     Edit an existing server")
        console.print("  /mcp remove ...   Remove a server")
        console.print("  /mcp install ...  Browse and install servers")
        console.print()
