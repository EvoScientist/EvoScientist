"""``EvoSci server`` — inspect and stop the background langgraph dev server.

The explicit counterpart to ``langgraph_dev_keepalive``: an opt-in server
that outlives its CLI needs an equally explicit way to see and stop it.
"""

from __future__ import annotations

import sys

from ..stream.console import console
from ._app import server_app


@server_app.command("status")
def server_status() -> None:
    """Show the background langgraph dev server's state."""
    from ..config import get_effective_config
    from ..langgraph_dev.manager import (
        _DEFAULT_HOST,
        _DEFAULT_PORT,
        _pid_serves_port,
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
        pid = sidecar.get("pid")
        if _pid_serves_port(pid, port):
            console.print(f"  pid:       {pid}")
        else:
            console.print(
                f"  pid:       {pid} [yellow](stale record — this pid does "
                f"not serve port {port})[/yellow]"
            )
    elif running:
        console.print(
            "  [yellow]no sidecar — externally managed or pre-keepalive server[/yellow]"
        )


@server_app.command("stop")
def server_stop() -> None:
    """Stop the background langgraph dev server started by EvoSci."""
    from ..config import get_effective_config
    from ..langgraph_dev.manager import (
        _DEFAULT_HOST,
        _DEFAULT_PORT,
        is_langgraph_dev_running,
        stop_recorded_server,
    )

    pid = stop_recorded_server()
    if pid is not None:
        console.print(f"[green]✓[/green] Stopped langgraph dev (pid {pid}).")
        return
    config = get_effective_config()
    port = int(getattr(config, "langgraph_dev_port", _DEFAULT_PORT))
    host = (
        str(getattr(config, "langgraph_dev_host", _DEFAULT_HOST) or _DEFAULT_HOST)
    ).strip() or _DEFAULT_HOST
    if is_langgraph_dev_running(port=port, host=host):
        # A server without ownership records (crashed session, deleted state
        # files) can't be verified as ours — refuse to guess, hand the user
        # the manual path instead of a silent no-op.
        console.print(
            f"[yellow]⚠ A langgraph dev is still serving port {port}, but "
            f"EvoSci has no ownership record for it, so it was not "
            f"touched.[/yellow]"
        )
        if sys.platform == "win32":
            manual = (
                f'powershell "Get-NetTCPConnection -LocalPort {port} | '
                f'Select-Object -ExpandProperty OwningProcess | Stop-Process"'
            )
        else:
            manual = f"kill $(lsof -ti :{port})"
        console.print(
            f"[dim]If it is yours, stop it manually: [bold]{manual}[/bold][/dim]"
        )
    else:
        console.print(
            "[dim]No EvoSci-owned langgraph dev server to stop "
            "(stale state, if any, was cleaned up).[/dim]"
        )
