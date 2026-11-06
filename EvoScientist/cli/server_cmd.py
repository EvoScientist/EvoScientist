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
    # Listeners only: a client connected to the port (a session) is not it.
    return f"kill $(lsof -ti tcp:{port} -sTCP:LISTEN)"


@server_app.command("start")
def server_start(
    workdir: str | None = typer.Option(
        None,
        "--workdir",
        help="The project it registers (default: default_workdir, else "
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
            result = manager.serve_project(config, project, with_ccproxy=with_ccproxy)
    except (ServerUnusableError, RuntimeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc
    if not result.served:
        console.print(
            "[red]The background agent server did not start or did not register "
            f"{escape(str(project))}; see "
            f"{escape(str(manager.log_file_of(result.record)))}."
            "[/red]"
        )
        if result.reason is not None:
            console.print(f"[dim]{escape(result.reason)}[/dim]")
        raise typer.Exit(1)
    stop = "EvoSci service stop" if with_ccproxy else "EvoSci server stop"
    start_cmd = (
        f"EvoSci {'service' if with_ccproxy else 'server'} start "
        f"--workdir {escape(command_arg(project))}"
    )
    url = result.url
    if not result.started:
        console.print(
            f"[green]✓[/green] The background agent server at {url} was already "
            f"running; it serves {escape(str(project))} too."
        )
        if not manager.is_lasting(result.record):
            console.print(
                "[yellow]⚠ It is not the background service: an EvoSci deploy, "
                "the desktop app, an older release or another program runs it, "
                "and it may stop with its owner; then the next session or "
                f"[bold]{start_cmd}[/bold] starts the background one.[/yellow]"
            )
    else:
        console.print(
            f"[green]✓[/green] Background agent server running at {url}; it "
            f"serves {escape(str(project))} and every other registered project, "
            f"and keeps running for every EvoSci session until [bold]{stop}[/bold]."
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
    """Show the background langgraph dev server's state and the projects it
    serves."""
    from ..config import get_effective_config
    from ..langgraph_dev import server_record
    from ..langgraph_dev.boundary import is_wildcard_host
    from ..langgraph_dev.manager import (
        OLDER_SERVER_MESSAGE,
        configured_address,
        is_langgraph_dev_running,
        lock_problem,
        read_record_and_lock,
    )
    from ..langgraph_dev.sdk import langgraph_dev_url

    config = get_effective_config()
    configured_port, configured_host = configured_address(config)
    record, lock = read_record_and_lock()
    port, host = configured_port, configured_host
    if record is not None:
        port, host = record.port, record.host
    running = is_langgraph_dev_running(port=port, host=host)

    problem = lock_problem(lock) if record is None else None
    if problem is not None:
        console.print(f"[yellow]⚠ {escape(problem)}[/yellow]")
        return
    if not running and record is None:
        console.print("[dim]No background langgraph dev server is running.[/dim]")
        return
    if running:
        state = "[green]running[/green]"
    elif record is not None and record.phase == server_record.STARTING:
        state = "[yellow]starting[/yellow]"
    else:
        state = "[red]not responding[/red]"
    console.print(f"[bold]langgraph dev[/bold] on port {port}: {state}")
    if record is not None:
        console.print(f"  pid:              {record.pid}")
        address = record.url
        if is_wildcard_host(record.host):
            # A wildcard bind is reached over loopback.
            address = f"{address} (listens on {record.host})"
        console.print(f"  address:          {escape(address)}")
        if record.host != configured_host:
            console.print(f"  configured host:  {escape(configured_host)}")
        if record.port != configured_port:
            # langgraph_dev_port changed while it ran.
            console.print(
                f"  [yellow]The configured port is {configured_port}; sessions "
                "do not use this server until it runs there (EvoSci server "
                "stop, and the next session starts it there).[/yellow]"
            )
        if record.older:
            console.print(f"  [yellow]{escape(OLDER_SERVER_MESSAGE)}[/yellow]")
    else:
        console.print(
            "  [yellow]no ownership record — externally managed or older server"
            "[/yellow]"
        )
    if running:
        _print_projects(langgraph_dev_url(port=port, host=host))


def _print_projects(url: str) -> None:
    """List the projects the server at *url* serves, or say why they cannot
    be listed."""
    from ..langgraph_dev.registration import (
        OlderServerError,
        OtherServerError,
        RegistrationError,
        served_projects,
    )

    try:
        projects = served_projects(url)
    except OtherServerError as exc:
        console.print(
            f"  projects:         [yellow]not listed. {escape(str(exc))}[/yellow]"
        )
        return
    except (OlderServerError, RegistrationError) as exc:
        console.print(
            f"  projects:         [yellow]could not be read. {escape(str(exc))}[/yellow]"
        )
        return
    if not projects:
        console.print("  projects:         none")
        return
    console.print("  projects:")
    for project in projects:
        missing = " [yellow](folder missing)[/yellow]" if project.get("missing") else ""
        console.print(f"    {escape(str(project['workspace_dir']))}{missing}")


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


def _stop_busy_server_anyway(record) -> bool:
    """Whether to go on with a stop: always when it ends no server or an
    idle one; for a busy one, after asking, since it serves every project and
    its runs and background jobs are not kept. Without a terminal to ask in,
    it is refused (exit 1), as ``EvoSci update`` does; ``--yes`` skips the
    question."""
    from ..langgraph_dev.manager import _is_port_occupied

    # The server the stop ends is the recorded one, whatever its port; a
    # server without a record is not touched. Its work is asked while it
    # listens, also when /ok misses its second: a busy server answers late.
    if record is None:
        return True
    if not _is_port_occupied(record.port, record.host):
        return True
    work = _busy_work(record.url)
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
    server that ``EvoSci deploy`` or the desktop app owns is left running.
    Either command leaves a server it may not end (another user's or an
    elevated one) or could not open running, says why and exits 1. A busy
    server is stopped only after asking, or with *yes*; a server that took
    the place of the one asked about, or appeared meanwhile, is not stopped.
    Whether it was left running."""
    from ..config import get_effective_config
    from ..langgraph_dev import manager
    from ..langgraph_dev.manager import (
        ServerKeptError,
        ServerStopFailedError,
        configured_address,
        is_langgraph_dev_running,
        owned_by_its_starter,
    )

    record, lock = manager.read_record_and_lock()
    # A server service stop leaves running needs no question about its work.
    kept = keep_owned and owned_by_its_starter(record)
    if not yes and not kept:
        if not _stop_busy_server_anyway(record):
            return True
        # The question may have taken a while: stop only the server it was
        # about, never one that appeared meanwhile.
        current, lock, same = manager.reread_record(record)
        if not same and record is None:
            console.print(
                f"[yellow]⚠ A server (pid {current.pid}) is starting now; nothing "
                "was stopped. Run the stop again once it is up.[/yellow]"
            )
            raise typer.Exit(1)
        if not same:
            console.print(
                f"[yellow]⚠ A server (pid {current.pid}) that was not the one "
                "asked about runs now; nothing was stopped. Run the stop "
                "again.[/yellow]"
            )
            raise typer.Exit(1)
        record = current
    if record is None and manager.starting_without_record():
        console.print(
            "[yellow]⚠ Another EvoScientist process is starting the server; "
            "nothing was stopped. Run the stop again once it is up.[/yellow]"
        )
        raise typer.Exit(1)
    problem = manager.lock_problem(lock) if record is None else None
    if problem is not None:
        console.print(f"[yellow]⚠ {escape(problem)}[/yellow]")
        raise typer.Exit(1)
    config = get_effective_config()
    port, host = configured_address(config)
    try:
        pid = manager.stop_server(keep_owned=keep_owned, record=record)
    except ServerKeptError as exc:
        console.print(f"[yellow]⚠ {escape(str(exc))}[/yellow]")
        if isinstance(exc, ServerStopFailedError):
            raise typer.Exit(1) from None
        return True
    if pid is not None:
        console.print(f"[green]✓[/green] Stopped langgraph dev (pid {pid}).")
        return False
    if record is not None:
        port, host = record.port, record.host
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
        console.print("[dim]No EvoSci-owned langgraph dev server to stop.[/dim]")
    return left_running
