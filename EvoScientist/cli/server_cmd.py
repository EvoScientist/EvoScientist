"""``EvoSci server`` — inspect and stop the background langgraph dev server.

The explicit counterpart to ``langgraph_dev_keepalive``: an opt-in server
that outlives its CLI needs an equally explicit way to see and stop it.
"""

from __future__ import annotations

from ..stream.console import console
from ._app import server_app


@server_app.command("status")
def server_status() -> None:
    """Show the background langgraph dev server's state."""
    from ..config import get_effective_config
    from ..langgraph_dev.manager import (
        _DEFAULT_HOST,
        _DEFAULT_PORT,
        _read_workspace_sidecar,
        is_langgraph_dev_running,
    )

    config = get_effective_config()
    port = int(getattr(config, "langgraph_dev_port", _DEFAULT_PORT))
    host = (
        str(getattr(config, "langgraph_dev_host", _DEFAULT_HOST) or _DEFAULT_HOST)
    ).strip() or _DEFAULT_HOST
    running = is_langgraph_dev_running(port=port, host=host)
    sidecar = _read_workspace_sidecar()

    if not running and sidecar is None:
        console.print("[dim]No background langgraph dev server is running.[/dim]")
        return
    state = "[green]running[/green]" if running else "[red]not responding[/red]"
    console.print(f"[bold]langgraph dev[/bold] on port {port}: {state}")
    if sidecar is not None:
        console.print(f"  workspace: {sidecar.get('workspace')}")
        console.print(f"  pid:       {sidecar.get('pid')}")
    elif running:
        console.print(
            "  [yellow]no sidecar — externally managed or pre-keepalive server[/yellow]"
        )


@server_app.command("stop")
def server_stop() -> None:
    """Stop the background langgraph dev server started by EvoSci."""
    from ..langgraph_dev.manager import stop_recorded_server

    pid = stop_recorded_server()
    if pid is None:
        console.print(
            "[dim]No EvoSci-owned langgraph dev server to stop "
            "(stale state, if any, was cleaned up).[/dim]"
        )
    else:
        console.print(f"[green]✓[/green] Stopped langgraph dev (pid {pid}).")
