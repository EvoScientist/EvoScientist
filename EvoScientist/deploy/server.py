"""``EvoSci deploy`` — start a standalone langgraph dev server.

Hosts the fully-equipped EvoScientist main agent (MCP + async sub-agents)
for consumption by external LangChain-compatible UIs (deep-agents-ui,
agent-chat-ui, LangSmith Studio) and SDK clients.

Differs from ``EvoSci`` / ``EvoSci serve``: no in-process CLI agent,
no session DB, no channel runtime, no TUI. The terminal only shows
startup progress, the Ready banner, and then blocks until Ctrl+C.

The server is the same one ``EvoSci`` / ``EvoSci serve`` start in the
background.

``--json`` is for a program that starts the server (the desktop app): JSON
event lines on stdout, an attach to a running server, and a stop when stdin
closes. :mod:`.json_mode` documents the events.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time
from collections.abc import Callable, Iterator
from typing import Annotated, Any

import typer  # type: ignore[import-untyped]
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

from ..cli._app import app
from ..stream.console import console


@app.command()
def deploy(
    workdir: str | None = typer.Option(
        None,
        "--workdir",
        help="Workspace directory (default: config.default_workdir or cwd)",
    ),
    port: int | None = typer.Option(
        None,
        "--port",
        help="Port for langgraph dev (default: config.langgraph_dev_port = 6174)",
    ),
    host: str | None = typer.Option(
        None,
        "--host",
        help="Interface to bind (default: config.langgraph_dev_host = "
        "127.0.0.1, i.e. this machine only — pass 0.0.0.0 to reach it from "
        "other machines, but note the server has no auth and serves every "
        "project)",
    ),
    tunnel: bool = typer.Option(
        False,
        "--tunnel",
        help="Expose the server over a public Cloudflare tunnel (no auth — "
        "anyone with the URL can read every project the server serves and run "
        "the agent in any of them; trusted use only)",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug logging",
    ),
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Print one JSON event per line, attach to a running backend, "
            "and stop when stdin closes",
        ),
    ] = False,
):
    """Deploy EvoScientist main agent as a standalone LangGraph dev server.

    Starts ``langgraph dev`` (MCP tools and async sub-agents in the server).
    Connect any LangChain-compatible UI or SDK client to the printed
    endpoint. Press Ctrl+C to stop.
    """
    if json_output:
        from .json_mode import run_json

        code = run_json(
            workdir=workdir, port=port, host=host, tunnel=tunnel, debug=debug
        )
        if code:
            raise typer.Exit(code)
        return

    # 1. Load config (deploy is opinionated about full MCP + async; the only
    # overrides are the bind flags).
    config = _load_config(debug, host, port)

    # Before anything is created or started.
    from ..cli.commands import _require_chatgpt_sign_in

    _require_chatgpt_sign_in(config, interactive=False)

    # 2. Resolve workspace (CLI > config.default_workdir > cwd)
    from ..cli.commands import _prepare_workdir_or_exit
    from ..paths import start_workspace_path

    ws = str(start_workspace_path(workdir, config.default_workdir))
    # Create the folder (or refuse a known project's missing one); do NOT
    # mutate the parent process's paths module state — the deploy parent has
    # no in-process agent.
    _prepare_workdir_or_exit(ws)

    # 3. Resolve port, then validate range so misconfigurations fail fast
    # with a clear message instead of an opaque socket error from langgraph
    # dev later.
    effective_port = _resolve_port(port, config)
    if not _port_in_range(effective_port):
        console.print(f"[red]{_invalid_port_message(effective_port)}[/red]")
        raise typer.Exit(1)

    effective_host = _resolve_host(host, config)

    # 4 to 10: the server, started here and owned by this command (never
    # one already running for this data folder), and stopped here, whatever
    # ends the command.
    with _stop_signals_interrupt():
        proc: subprocess.Popen | None = None
        try:
            proc = _start_server(config, effective_port, effective_host, tunnel, ws)
            _serve(ws, effective_port, effective_host, tunnel)
        finally:
            _stop_own_server(proc)


def _stop_signals() -> list[signal.Signals]:
    """The signals that end ``EvoSci deploy``. On macOS and Linux that
    includes SIGHUP: the terminal closed or the SSH session dropped. (On
    Windows the server shares the terminal's console, so closing it ends the
    server too.)"""
    stops = [signal.SIGINT, signal.SIGTERM]
    hangup = getattr(signal, "SIGHUP", None)
    if hangup is not None:
        stops.append(hangup)
    return stops


def _say(markup: str) -> None:
    """Print a stop message; after a hang-up the terminal is gone, and
    writing to it fails without stopping the stop."""
    with contextlib.suppress(OSError):
        console.print(markup)


def _interrupt(signum: int, frame: Any) -> None:
    """End deploy as Ctrl+C does, with the signals that end it ignored from
    now on, so a second one cannot cut the stop short."""
    for sig in _stop_signals():
        signal.signal(sig, signal.SIG_IGN)
    signal.default_int_handler(signum, frame)


@contextlib.contextmanager
def _stop_signals_interrupt() -> Iterator[None]:
    """Each signal that ends deploy raises ``KeyboardInterrupt``, as Ctrl+C
    does, so the stop in deploy's ``finally`` runs: SIGTERM and SIGHUP would
    otherwise end the process without it, and leave the server, which runs
    in its own session or process group, running. Installed before the
    start, so a signal while the server starts stops it too. A SIGHUP this
    process started with ignored stays ignored: ``nohup EvoSci deploy &``
    keeps deploy and its server up after a logout. (SIGINT is taken over
    even when ignored, as a shell ignores it for ``EvoSci deploy &`` and a
    later ``kill -INT`` stops deploy.)

    The first such signal also ignores the rest from then on: every stop
    that follows it, the one ``ensure_server`` makes of a server it was
    starting as well as deploy's own, runs to the end."""
    hangup = getattr(signal, "SIGHUP", None)
    previous = {
        sig: signal.signal(sig, _interrupt)
        for sig in _stop_signals()
        if sig != hangup or signal.getsignal(sig) is not signal.SIG_IGN
    }
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _stop_own_server(proc: subprocess.Popen | None) -> None:
    """Stop the server this command started: *proc*, or, when the start was
    interrupted before ``ensure_server`` returned its handle, the one the
    manager holds right after spawning. The signals that end deploy are
    ignored meanwhile: a second Ctrl+C must not cut the stop short and
    leave the server running."""
    from ..langgraph_dev import manager as lgm

    server = proc if proc is not None else lgm._PROCESS
    if server is None:
        return
    previous = {sig: signal.signal(sig, signal.SIG_IGN) for sig in _stop_signals()}
    try:
        _say("\n[dim]Stopping langgraph dev...[/dim]")
        # A server that never became ready has no run to put back in the
        # queue, and on Windows would only finish loading first.
        lgm.stop_langgraph_dev(server, clean=proc is not None)
        _say("[dim]Stopped.[/dim]")
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _wait_for_stop() -> None:
    """Block until a signal that ends deploy raises ``KeyboardInterrupt``.
    The short sleeps keep Ctrl+C prompt on Windows, where a blocking wait
    without a timeout does not see it."""
    while True:
        time.sleep(0.5)


def _serve(ws: str, effective_port: int, effective_host: str, tunnel: bool) -> None:
    """Register the workspace with the started server, show the Ready
    banner and serve until a signal ends deploy."""
    from ..langgraph_dev.manager import (
        RUNTIME,
        TunnelFailedError,
        _base_url,
        read_tunnel_url,
    )
    from ..langgraph_dev.registration import (
        RegistrationError,
        ServerUnusableError,
        register_folder,
    )
    from ..paths import Workspace

    # The server serves the projects its clients register, this one too.
    try:
        exposure = register_folder(
            Workspace(ws), url=_base_url(effective_port, effective_host)
        )
    except (ServerUnusableError, RegistrationError) as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc

    # The start waited for the server's first answer; if we got here, the
    # subprocess is up.
    console.print("[green]✓[/green] langgraph dev ready")
    from ..cli.commands import print_exposure_warnings

    print_exposure_warnings(exposure, to=console)
    # The agent is built inside the server, so its own hint lands in the
    # server log; show it here.
    from ..agent_shell import setup_hint

    hint = setup_hint()
    if hint is not None:
        console.print(f"[yellow]⚠ {escape(hint)}[/yellow]")

    # 8b. Cloudflare tunnel URL — the local server is healthy, but cloudflared
    # establishes the public tunnel a few seconds later and prints the random
    # URL into the log. Poll for it so we can surface it in the ready banner.
    public_url: str | None = None
    if tunnel:
        tunnel_error: str | None = None
        with console.status(
            "[dim]Waiting for Cloudflare tunnel URL...[/dim]", spinner="dots"
        ):
            try:
                public_url = read_tunnel_url()
            except TunnelFailedError as exc:
                tunnel_error = str(exc)
        if public_url:
            console.print("[green]✓[/green] tunnel up")
        elif tunnel_error is not None:
            console.print(
                "[yellow]⚠ Cloudflare quick tunnel failed: "
                f"{escape(tunnel_error)}. Details in the log "
                f"({_shorten(str(RUNTIME.log_file))}).[/yellow]"
            )
        else:
            console.print(
                "[yellow]⚠ Tunnel URL not detected within the wait window. "
                f"Check the log ({_shorten(str(RUNTIME.log_file))}) for a "
                "trycloudflare.com URL.[/yellow]"
            )

    # 9. Ready banner
    log_hint = _shorten(str(RUNTIME.log_file))
    public_line = f"[bold]Public URL:[/bold]   {public_url}\n" if public_url else ""
    from ..langgraph_dev.assistants import chat_assistant_id

    assistant_id = chat_assistant_id(Workspace(ws))
    console.print(
        Panel(
            Text.from_markup(
                f"[bold]Endpoint:[/bold]     "
                f"{_base_url(effective_port, effective_host)}\n"
                f"{public_line}"
                f"[bold]Assistant ID:[/bold] {assistant_id}\n"
                f"[bold]Connect via:[/bold]  any LangChain SDK / "
                f"LangGraph-compatible UI\n"
                f"[bold]Logs:[/bold]         {log_hint}\n\n"
                f"[dim]Press Ctrl+C to stop.[/dim]"
            ),
            title="[bold green]✓ Ready[/bold green]",
            border_style="green",
        )
    )

    # 10. Serve until a signal ends deploy; the caller stops the server.
    with contextlib.suppress(KeyboardInterrupt):
        _wait_for_stop()


def _start_server(
    config: Any,
    effective_port: int,
    effective_host: str,
    tunnel: bool,
    ws: str,
) -> subprocess.Popen:
    """Start the server this command owns, printing the startup banner right
    before, and return its handle; exits 1 when one already runs for this
    data folder, or the port or the start fails."""
    from ..ccproxy_manager import uses_oauth
    from ..langgraph_dev.manager import ensure_server, is_lasting, owner_running

    starting = (
        "[dim]Starting langgraph dev and its ccproxy...[/dim]"
        if uses_oauth(config)
        else "[dim]Starting langgraph dev...[/dim]"
    )
    status_box: dict[str, Any] = {}

    def _before_start() -> None:
        # 5. Startup banner, once nothing runs for this data folder.
        console.print(
            Panel(
                Text.from_markup(
                    f"[bold]Workspace:[/bold] {_shorten(ws)}\n"
                    f"[bold]Host:[/bold]      {effective_host}\n"
                    f"[bold]Port:[/bold]      {effective_port}\n"
                    f"[bold]Auth:[/bold]      {_describe_auth(config)}"
                ),
                title="[bold cyan]EvoScientist Deploy[/bold cyan]",
                border_style="cyan",
            )
        )
        if config.dangerous_mode:
            from ..cli._constants import (
                DANGEROUS_BANNER_LABEL,
                DANGEROUS_BANNER_MESSAGE,
            )

            console.print(
                f"[bold white on red] ⚠ {DANGEROUS_BANNER_LABEL} [/bold white on red] "
                f"[bold red]{DANGEROUS_BANNER_MESSAGE}[/bold red]"
            )

    # 6. Start langgraph dev
    try:
        with console.status(starting, spinner="dots") as status:
            status_box["status"] = status
            result = ensure_server(
                config,
                backend="langgraph_server",
                port=effective_port,
                host=effective_host,
                # deploy is the server's foreground owner: it stops it.
                lasting=False,
                tunnel=tunnel,
                reuse=False,
                before_start=_before_start,
                on_wait=lock_wait_status(status_box, starting),
            )
    except OSError as exc:
        console.print(f"[red]langgraph dev startup failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    if result.started:
        return result.proc
    record = result.record
    if result.problem == "running":
        where = f"port {result.port}"
        if record is not None:
            where += f", pid {record.pid}"
        console.print(
            f"[red]This data folder's langgraph dev server already runs "
            f"({where}).[/red]"
        )
        # ``EvoSci server status`` reads the record, so it describes this
        # server whatever its port.
        status = " [bold]EvoSci server status[/bold] lists the projects it serves."
        if record is None:
            console.print("[dim]Stop the existing EvoSci/serve session first.[/dim]")
        elif record.lasting is False and not owner_running(record):
            console.print(
                "[dim]The EvoSci deploy or desktop app that started it no "
                "longer runs. Stop it with [bold]EvoSci server stop[/bold], "
                f"then deploy again.{status}[/dim]"
            )
        elif not is_lasting(record):
            console.print(
                f"[dim]It is not the background service: another "
                f"EvoSci deploy, the desktop app or an older release "
                f"runs it. Stop that one first.{status}[/dim]"
            )
        else:
            console.print(
                f"[dim]It is the background agent server, which your EvoSci "
                f"sessions share.{status} Use it as it is, or stop it with "
                "[bold]EvoSci server stop[/bold] to deploy here (that "
                "ends the background work of the sessions using "
                "it).[/dim]"
            )
    else:
        # What deploy --json and the WebUI report, said here.
        from .launcher import server_error

        error = server_error(result)
        console.print(f"[red]{escape(error.message)}[/red]")
        if error.detail:
            console.print(f"[dim]{escape(error.detail)}[/dim]")
        if "exit_code" in error.fields:
            console.print(f"[dim]Exit code: {error.fields['exit_code']}[/dim]")
        if "log_file" in error.fields:
            console.print(
                f"[dim]Log: {escape(_shorten(error.fields['log_file']))}[/dim]"
            )
        if "lock_file" in error.fields:
            console.print(f"[dim]Lock: {escape(error.fields['lock_file'])}[/dim]")
    raise typer.Exit(1)


def lock_wait_status(
    status_box: dict[str, Any], starting: str
) -> Callable[[bool], None]:
    """An ``on_wait`` for ``ensure_server`` that says on the spinner in
    *status_box* (``"status"``, once shown) when another process is starting
    or stopping the server, and goes back to *starting* after."""

    def show(waiting: bool) -> None:
        status = status_box.get("status")
        if status is None:
            return
        status.update(
            "[dim]Waiting for another EvoSci process to start or stop "
            "langgraph dev...[/dim]"
            if waiting
            else starting
        )

    return show


def _load_config(debug: bool, host: str | None = None, port: int | None = None) -> Any:
    """The effective config, applied to the environment, with ``--debug``
    logging switched on.

    ``--host`` and ``--port`` are overrides, as ``EvoSci --host`` and ``EvoSci
    serve --host`` pass theirs, so the config fingerprint matches a server
    started with the same bind.
    """
    from ..config import apply_config_to_env, get_effective_config

    cli_overrides: dict[str, Any] = {}
    if host is not None and host.strip():
        cli_overrides["langgraph_dev_host"] = host.strip()
    if port is not None:
        cli_overrides["langgraph_dev_port"] = port
    if debug:
        cli_overrides["log_level"] = "DEBUG"
    config = get_effective_config(cli_overrides)
    if debug:
        from ..config.settings import record_env_origin

        os.environ["EVOSCIENTIST_LOG_LEVEL"] = "DEBUG"
        record_env_origin("EVOSCIENTIST_LOG_LEVEL", "cli")
        from ..cli.commands import _configure_logging

        _configure_logging()
    apply_config_to_env(config)
    return config


def _resolve_port(port: int | None, config: Any) -> int:
    """``--port``, else the configured port. An explicit None check, so
    ``--port 0`` is not taken for "unset" and fails the range check."""
    from ..langgraph_dev.manager import _DEFAULT_PORT

    if port is not None:
        return port
    return int(getattr(config, "langgraph_dev_port", _DEFAULT_PORT))


def _port_in_range(port: int) -> bool:
    return 1 <= port <= 65535


def _invalid_port_message(port: int) -> str:
    return f"Invalid port {port}. Use an integer in [1, 65535]."


def _resolve_host(host: str | None, config: Any) -> str:
    """``--host``, else the configured host, else loopback.

    A blank ``--host`` means "not passed" (matching serve), so it can never
    discard the configured bind. Both branches strip: whitespace reaching
    socket.bind() surfaces as an opaque gaierror, and duck-typed configs
    handed to this function never ran ``__post_init__`` normalization.
    """
    from ..langgraph_dev.manager import _DEFAULT_HOST

    cli_host = host.strip() if host is not None else ""
    return (
        cli_host
        or str(getattr(config, "langgraph_dev_host", _DEFAULT_HOST) or "").strip()
        or _DEFAULT_HOST
    )


def _describe_auth(config: Any) -> str:
    """Render a one-line auth summary for the startup banner."""
    anth = getattr(config, "anthropic_auth_mode", "api_key")
    oai = getattr(config, "openai_auth_mode", "api_key")
    if anth == "oauth" and oai == "oauth":
        return "OAuth (Anthropic via ccproxy) + ChatGPT sign-in (OpenAI)"
    if anth == "oauth":
        return "OAuth (Anthropic via ccproxy) + API key (OpenAI)"
    if oai == "oauth":
        return "API key (Anthropic) + ChatGPT sign-in (OpenAI)"
    return "API key"


def _shorten(path: str) -> str:
    """Replace ``$HOME`` prefix with ``~`` for compact display."""
    home = os.path.expanduser("~")
    if path.startswith(home):
        return "~" + path[len(home) :]
    return path
