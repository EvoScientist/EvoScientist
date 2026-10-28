"""``EvoSci service`` and ``EvoSci ccproxy``: the background agent service.

The service is what runs in the background for every EvoSci session: the
langgraph dev server (``EvoSci server``) and, when Anthropic uses OAuth,
the ccproxy it and the sessions route through (``EvoSci ccproxy``).
``EvoSci service`` starts, shows and stops them together.
"""

from __future__ import annotations

import typer  # type: ignore[import-untyped]
from rich.markup import escape

from ..stream.console import console
from ._app import ccproxy_app, service_app
from .server_cmd import server_status, start_background, stop_server


@service_app.command("start")
def service_start(
    workdir: str | None = typer.Option(
        None,
        "--workdir",
        help="The project to start it from (default: default_workdir, else "
        "the current folder)",
    ),
) -> None:
    """Start the background agent service (the server, and ccproxy with
    Anthropic OAuth) and leave it running for every EvoSci session;
    `EvoSci service stop` stops it."""
    start_background(workdir, with_ccproxy=True)


@service_app.command("stop")
def service_stop(
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Stop it even while the server runs work"
    ),
) -> None:
    """Stop the background agent service: the server and ccproxy. A server
    that EvoSci deploy or the desktop app owns is left running, with the
    ccproxy it may route through, and a busy one is stopped after asking."""
    if stop_server(keep_owned=True, yes=yes):
        from ..ccproxy_manager import recorded_ccproxy_running

        if recorded_ccproxy_running() is not None:
            console.print(
                "[dim]ccproxy was left running too, since that server may route "
                "through it; EvoSci ccproxy stop stops it.[/dim]"
            )
        return
    ccproxy_stop()


@service_app.command("status")
def service_status() -> None:
    """Show the background agent service's state."""
    server_status()
    ccproxy_status()


@ccproxy_app.command("start")
def ccproxy_start() -> None:
    """Start ccproxy, which Anthropic OAuth routes through, and leave it
    running for every EvoSci session; `EvoSci ccproxy stop` stops it."""
    from ..ccproxy_manager import is_ccproxy_running, maybe_start_ccproxy, uses_oauth
    from ..config import get_effective_config

    config = get_effective_config()
    if not uses_oauth(config):
        console.print(
            "[dim]Anthropic does not use OAuth here, so ccproxy is not needed.[/dim]"
        )
        return
    port = config.ccproxy_port
    was_running = is_ccproxy_running(port)
    try:
        with console.status("[dim]Starting ccproxy...[/dim]", spinner="dots"):
            maybe_start_ccproxy(config, lasting=True)
    except (RuntimeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc
    if was_running:
        console.print(f"[green]✓[/green] ccproxy was already running on port {port}.")
    else:
        console.print(
            f"[green]✓[/green] ccproxy running on port {port}. It keeps running "
            "until [bold]EvoSci ccproxy stop[/bold] (or EvoSci service stop)."
        )


@ccproxy_app.command("stop")
def ccproxy_stop() -> None:
    """Stop the ccproxy EvoSci started for its sessions."""
    from ..ccproxy_manager import (
        is_ccproxy_running,
        recorded_ccproxy_running,
        stop_recorded_ccproxy,
    )
    from ..config import get_effective_config

    pid = stop_recorded_ccproxy()
    if pid is not None:
        console.print(f"[green]✓[/green] Stopped ccproxy (pid {pid}).")
        return
    kept = recorded_ccproxy_running()
    if kept is not None:
        console.print(
            f"[yellow]⚠ Could not stop the ccproxy EvoSci started (pid {kept}): "
            "its process cannot be inspected. Stop it by hand.[/yellow]"
        )
        return
    port = get_effective_config().ccproxy_port
    if is_ccproxy_running(port):
        console.print(
            f"[yellow]⚠ A ccproxy still answers on port {port}, but EvoSci did not "
            "start it for its sessions (an EvoSci deploy or the desktop app owns "
            "it, or it was started by hand), so it was not touched.[/yellow]"
        )
    else:
        console.print("[dim]No ccproxy started by EvoSci is running.[/dim]")


@ccproxy_app.command("status")
def ccproxy_status() -> None:
    """Show whether ccproxy is running."""
    from ..ccproxy_manager import (
        is_ccproxy_running,
        recorded_ccproxy_running,
        uses_oauth,
    )
    from ..config import get_effective_config

    config = get_effective_config()
    port = config.ccproxy_port
    if is_ccproxy_running(port):
        pid = recorded_ccproxy_running()
        owner = f" (pid {pid})" if pid is not None else " (not started by EvoSci)"
        console.print(
            f"[bold]ccproxy[/bold] on port {port}: [green]running[/green]{owner}"
        )
    elif uses_oauth(config):
        console.print(f"[bold]ccproxy[/bold] on port {port}: [red]not running[/red]")
    else:
        console.print("[dim]ccproxy: not needed (Anthropic does not use OAuth).[/dim]")
