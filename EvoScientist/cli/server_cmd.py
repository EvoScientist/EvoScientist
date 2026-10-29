"""``EvoSci server`` — start, inspect and stop the background langgraph dev
server.

The server outlives the sessions that use it, so it needs an explicit way to
start, see and stop it. ``EvoSci service`` (:mod:`.service_cmd`) covers it
together with ccproxy.
"""

from __future__ import annotations

import sys

import typer  # type: ignore[import-untyped]
from rich.markup import escape

from ..stream.console import console
from ._app import server_app


def manual_stop_command(port: int) -> str:
    """The shell command that stops whatever listens on *port*, for a server
    EvoSci has no ownership record for."""
    if sys.platform == "win32":
        # Feed the PID to Stop-Process via a subexpression, not a $_
        # pipeline: when the user pastes this at a PowerShell prompt, the
        # outer shell expands $_ (empty there) before the child powershell
        # runs, so a $_-based command would stop nothing. (...) is literal
        # inside the double-quoted argument, so it reaches the child intact.
        # -State Listen + -Unique avoids stray/duplicate PIDs from
        # established connections on the same port.
        return (
            f'powershell "Stop-Process -Id (Get-NetTCPConnection '
            f"-LocalPort {port} -State Listen | Select-Object "
            f'-ExpandProperty OwningProcess -Unique) -Force"'
        )
    return f"kill $(lsof -ti :{port})"


@server_app.command("start")
def server_start(
    workdir: str | None = typer.Option(
        None,
        "--workdir",
        help="The project to start it from (default: default_workdir, else "
        "the current folder)",
    ),
) -> None:
    """Start the background langgraph dev server and leave it running, for
    every EvoSci session to share; `EvoSci server stop` stops it. Only the
    server: `EvoSci service start` also starts ccproxy."""
    start_background(workdir, with_ccproxy=False)


def start_background(workdir: str | None, *, with_ccproxy: bool) -> None:
    """``EvoSci server start`` / ``EvoSci service start``: start the server,
    or reuse a running one, for a project, and with *with_ccproxy* also the
    ccproxy it uses with Anthropic OAuth."""
    from ..ccproxy_manager import is_ccproxy_running, uses_oauth
    from ..config import get_effective_config
    from ..langgraph_dev import manager
    from ..langgraph_dev.registration import ServerUnusableError
    from ..langgraph_dev.sdk import langgraph_dev_url
    from ..paths import command_arg, start_workspace_path
    from .commands import (
        _prepare_workdir_or_exit,
        _print_server_notices,
        _require_chatgpt_sign_in,
    )

    config = get_effective_config()
    _require_chatgpt_sign_in(config, interactive=False)
    project = _prepare_workdir_or_exit(
        start_workspace_path(workdir, config.default_workdir)
    )
    oauth = uses_oauth(config)
    try:
        with console.status(
            "[dim]Starting the background agent server...[/dim]", spinner="dots"
        ):
            served = manager.serve_project(config, project, with_ccproxy=with_ccproxy)
    except (ServerUnusableError, RuntimeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc
    if not served:
        console.print(
            "[red]The background agent server did not start or did not register "
            f"{escape(str(project))}; see {escape(str(manager.RUNTIME.log_file))}."
            "[/red]"
        )
        raise typer.Exit(1)
    stop = "EvoSci service stop" if with_ccproxy else "EvoSci server stop"
    start_cmd = (
        f"EvoSci {'service' if with_ccproxy else 'server'} start "
        f"--workdir {escape(command_arg(project))}"
    )
    url = langgraph_dev_url(config)
    # Of two starts at once, the one that lost the race for the port only
    # reused the other's server: this process started it only when it holds it.
    if manager._PROCESS is None:
        console.print(
            f"[green]✓[/green] The background agent server at {url} was already "
            f"running; it serves {escape(str(project))} too."
        )
        if not manager.is_lasting(manager._read_workspace_sidecar()):
            console.print(
                "[yellow]⚠ It is not the background service: an EvoSci deploy, "
                "the desktop app, an older release or another program runs it, "
                "and it may stop with its owner; then the next session or "
                f"[bold]{start_cmd}[/bold] starts the background one.[/yellow]"
            )
    else:
        console.print(
            f"[green]✓[/green] Background agent server running at {url}, started "
            f"from {escape(str(project))}. It keeps running for every EvoSci "
            f"session until [bold]{stop}[/bold]."
        )
    if oauth and not with_ccproxy and not is_ccproxy_running(config.ccproxy_port):
        console.print(
            "[yellow]⚠ Anthropic uses OAuth, and ccproxy is not running: start it "
            "with [bold]EvoSci ccproxy start[/bold] (or use [bold]EvoSci service "
            "start[/bold]).[/yellow]"
        )
    _print_server_notices()


@server_app.command("status")
def server_status() -> None:
    """Show the background langgraph dev server's state."""
    from ..config import get_effective_config
    from ..langgraph_dev.manager import (
        _pid_serves_port,
        _read_workspace_sidecar,
        configured_address,
        is_langgraph_dev_running,
        recorded_address,
    )

    config = get_effective_config()
    port, host = configured_address(config)
    running = is_langgraph_dev_running(port=port, host=host)
    sidecar = _read_workspace_sidecar()

    if not running and sidecar is None:
        console.print("[dim]No background langgraph dev server is running.[/dim]")
        return
    state = "[green]running[/green]" if running else "[red]not responding[/red]"
    console.print(f"[bold]langgraph dev[/bold] on port {port}: {state}")
    if sidecar is not None:
        console.print(f"  started from: {sidecar.get('workspace')}")
        pid = sidecar.get("pid")
        if _pid_serves_port(pid, port):
            console.print(f"  pid:          {pid}")
        elif (address := recorded_address()) is not None:
            # langgraph_dev_port changed while it ran.
            console.print(
                f"  pid:          {pid} [yellow](serves port {address[0]}, not "
                f"port {port}; EvoSci server stop stops it)[/yellow]"
            )
        else:
            console.print(
                f"  pid:          {pid} [yellow](stale record — this pid does "
                f"not serve port {port})[/yellow]"
            )
    elif running:
        console.print(
            "  [yellow]no ownership record — externally managed or older server"
            "[/yellow]"
        )


@server_app.command("stop")
def server_stop(
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Stop it even while it runs work"
    ),
) -> None:
    """Stop the background langgraph dev server started by EvoSci."""
    stop_server(yes=yes)


def _interactive() -> bool:
    return sys.stdin.isatty()


def _busy_work(url: str) -> str | None:
    """What the server at *url* is running that a stop would end, or None
    when it is idle (or cannot be asked)."""
    from ..deploy.busy import (
        backend_has_active_runs,
        has_busy_thread,
        running_bg_process_names,
    )

    if not backend_has_active_runs(url):
        return None
    work = []
    jobs = running_bg_process_names(url)
    if jobs:
        work.append(f"background jobs: {', '.join(jobs)}")
    if has_busy_thread(url):
        work.append("a run in progress")
    return "; ".join(work) or "work in progress"


def _stop_busy_server_anyway() -> bool:
    """Whether to go on with a stop: always when it ends no server or an
    idle one; for a busy one, after asking, since it serves every project and
    its runs and background jobs are not kept. Without a terminal to ask in,
    it is refused (exit 1), as ``EvoSci update`` does; ``--yes`` skips the
    question."""
    from ..langgraph_dev.manager import (
        _base_url,
        is_langgraph_dev_running,
        recorded_address,
    )

    # The server the stop ends is the recorded one, whatever its port; a
    # server without a record is not touched.
    address = recorded_address()
    if address is None:
        return True
    port, host = address
    if not is_langgraph_dev_running(port=port, host=host):
        return True
    work = _busy_work(_base_url(port, host))
    if work is None:
        return True
    message = (
        f"The background agent server is busy ({work}). Stopping it ends that "
        "work, for every project using the server."
    )
    if not _interactive():
        console.print(
            f"[yellow]⚠ {escape(message)} Pass --yes to stop it anyway.[/yellow]"
        )
        raise typer.Exit(1)
    if typer.confirm(f"{message} Stop it?", default=False):
        return True
    console.print("[dim]Left running.[/dim]")
    return False


def stop_server(*, keep_owned: bool = False, yes: bool = False) -> bool:
    """``EvoSci server stop``; with *keep_owned* (``EvoSci service stop``), a
    server that ``EvoSci deploy`` or the desktop app owns, or that another
    process is starting or stopping, is left running. A busy server is
    stopped only after asking, or with *yes*. Whether it was left running."""
    from ..config import get_effective_config
    from ..langgraph_dev.manager import (
        ServerKeptError,
        configured_address,
        is_langgraph_dev_running,
        owned_by_its_starter,
        stop_recorded_server,
    )

    # A server service stop leaves running needs no question about its work.
    kept = keep_owned and owned_by_its_starter()
    if not yes and not kept and not _stop_busy_server_anyway():
        return True
    try:
        pid = stop_recorded_server(keep_owned=keep_owned)
    except ServerKeptError as exc:
        console.print(f"[yellow]⚠ {escape(str(exc))}[/yellow]")
        return True
    if pid is not None:
        console.print(f"[green]✓[/green] Stopped langgraph dev (pid {pid}).")
        return False
    config = get_effective_config()
    port, host = configured_address(config)
    left_running = is_langgraph_dev_running(port=port, host=host)
    if left_running:
        # A server without ownership records (crashed session, deleted state
        # files) can't be verified as ours — refuse to guess, hand the user
        # the manual path instead of a silent no-op.
        console.print(
            f"[yellow]⚠ A langgraph dev is still serving port {port}, but "
            f"EvoSci has no ownership record for it, so it was not "
            f"touched.[/yellow]"
        )
        console.print(
            f"[dim]If it is yours, stop it manually: "
            f"[bold]{manual_stop_command(port)}[/bold][/dim]"
        )
    else:
        console.print(
            "[dim]No EvoSci-owned langgraph dev server to stop "
            "(stale state, if any, was cleaned up).[/dim]"
        )
    return left_running
