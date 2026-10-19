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
event lines on stdout, an attach to a running server for the same workspace,
and a stop when stdin closes. :mod:`.json_mode` documents the events.
"""

from __future__ import annotations

import atexit
import os
import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any

import typer  # type: ignore[import-untyped]
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
        "other machines, but note the server has no auth)",
    ),
    tunnel: bool = typer.Option(
        False,
        "--tunnel",
        help="Expose the server over a public Cloudflare tunnel (no auth — "
        "anyone with the URL can drive the agent; trusted use only)",
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
            help="Print one JSON event per line, attach to a running backend "
            "for the same workspace, and stop when stdin closes",
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

    from ..langgraph_dev.manager import (
        RUNTIME,
        _base_url,
        _is_loopback_host,
        _is_port_occupied,
        _pid_serves_port,
        _read_workspace_sidecar,
        _server_config_fingerprint,
        is_langgraph_dev_running,
        read_tunnel_url,
        start_langgraph_dev,
        stop_langgraph_dev,
    )

    # 1. Load config (deploy is opinionated about full MCP + async; the only
    # overrides are the bind flags).
    config = _load_config(debug, host, port)

    # Before anything is created or started.
    from ..cli.commands import _require_chatgpt_sign_in

    _require_chatgpt_sign_in(config, interactive=False)

    # 2. Resolve workspace (CLI > config.default_workdir > cwd)
    from ..paths import start_workspace_path

    ws = str(start_workspace_path(workdir, config.default_workdir))
    # Subprocess inherits this path via EVOSCIENTIST_WORKSPACE_DIR (set inside
    # start_langgraph_dev). Ensure the dir exists; do NOT mutate the parent
    # process's paths module state — the deploy parent has no in-process agent.
    os.makedirs(ws, exist_ok=True)

    # 3. Resolve port, then validate range so misconfigurations fail fast
    # with a clear message instead of an opaque socket error from langgraph
    # dev later.
    effective_port = _resolve_port(port, config)
    if not _port_in_range(effective_port):
        console.print(f"[red]{_invalid_port_message(effective_port)}[/red]")
        raise typer.Exit(1)

    effective_host = _resolve_host(host, config)

    # Steps 4 to 7 hold the server start lock (see _start_lock).
    with _start_lock():
        # 4. Pre-flight port check — refuse to start if a non-EvoSci process is
        # holding the port. If an existing EvoSci langgraph dev is already up,
        # also refuse (deploy is the "primary server" — running multiple on the
        # same port is a configuration error).
        if _is_port_occupied(effective_port, effective_host):
            if is_langgraph_dev_running(port=effective_port, host=effective_host):
                console.print(
                    f"[red]Port {effective_port} is already serving a langgraph dev "
                    f"instance.[/red]"
                )
                sidecar = _read_workspace_sidecar()
                if sidecar is not None and _pid_serves_port(
                    sidecar.get("pid"), effective_port
                ):
                    # Surface what we know about the occupant — with keepalive it
                    # may be an ownerless leftover rather than a live session.
                    # Only when the recorded pid verifiably serves THIS port, so a
                    # stale or other-port record is never blamed.
                    console.print(
                        f"[dim]It serves workspace {sidecar.get('workspace')} "
                        f"(pid {sidecar.get('pid')}). Stop it with "
                        f"[bold]EvoSci server stop[/bold], or use "
                        f"[bold]--port[/bold] to deploy on a different port.[/dim]"
                    )
                else:
                    console.print(
                        "[dim]Stop the existing EvoSci/serve session first, or use "
                        "[bold]--port[/bold] to deploy on a different port.[/dim]"
                    )
            else:
                console.print(
                    f"[red]Port {effective_port} is occupied by another process.[/red]"
                )
                console.print(
                    f"[dim]Run [bold]lsof -i :{effective_port}[/bold] to inspect, "
                    f"or use [bold]--port[/bold] to pick a different port.[/dim]"
                )
            raise typer.Exit(1)

        # 5. Startup banner
        _auth_label = _describe_auth(config)
        console.print(
            Panel(
                Text.from_markup(
                    f"[bold]Workspace:[/bold] {_shorten(ws)}\n"
                    f"[bold]Host:[/bold]      {effective_host}\n"
                    f"[bold]Port:[/bold]      {effective_port}\n"
                    f"[bold]Auth:[/bold]      {_auth_label}"
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

        if not _is_loopback_host(effective_host):
            console.print(
                "[bold white on red] ⚠ PUBLIC BIND [/bold white on red] "
                f"[bold red]Listening on {effective_host} — no auth, and the agent "
                f"can run shell. Trusted networks only.[/bold red]"
            )

        if tunnel:
            console.print(
                "[bold white on red] ⚠ PUBLIC TUNNEL [/bold white on red] "
                "[bold red]Public URL, no auth — share only with people you "
                "trust.[/bold red]"
            )

        # 6. ccproxy lifecycle (Anthropic OAuth only)
        _ccproxy_proc = None
        if config.anthropic_auth_mode == "oauth":
            try:
                from ..ccproxy_manager import maybe_start_ccproxy, stop_ccproxy

                with console.status(
                    "[dim]Starting ccproxy (OAuth proxy)...[/dim]", spinner="dots"
                ):
                    _ccproxy_proc = maybe_start_ccproxy(config)
                if _ccproxy_proc:
                    atexit.register(stop_ccproxy, _ccproxy_proc)
                    console.print("[green]✓[/green] ccproxy started")
            except RuntimeError as exc:
                console.print(f"[red]ccproxy startup failed:[/red] {exc}")
                raise typer.Exit(1) from exc

        # 7. Start langgraph dev
        jobs_per_worker = int(getattr(config, "langgraph_dev_jobs_per_worker", 10))
        file_persistence = bool(getattr(config, "langgraph_dev_file_persistence", True))
        try:
            with console.status(
                "[dim]Starting langgraph dev...[/dim]",
                spinner="dots",
            ):
                proc = start_langgraph_dev(
                    workspace_dir=Path(ws),
                    port=effective_port,
                    host=effective_host,
                    file_persistence=file_persistence,
                    jobs_per_worker=jobs_per_worker,
                    tunnel=tunnel,
                    config_fingerprint=_server_config_fingerprint(config),
                    allowed_origins=getattr(config, "backend_allowed_origins", ""),
                )
            atexit.register(stop_langgraph_dev, proc)
        except Exception as exc:
            console.print(f"[red]langgraph dev startup failed:[/red] {exc}")
            raise typer.Exit(1) from exc

    # start_langgraph_dev already health-polled before returning; if we got
    # here, the subprocess is up.
    console.print("[green]✓[/green] langgraph dev ready")
    # The agent is built inside the server, so its own hint lands in the
    # server log; show it here.
    from rich.markup import escape

    from ..agent_shell import setup_hint

    hint = setup_hint()
    if hint is not None:
        console.print(f"[yellow]⚠ {escape(hint)}[/yellow]")

    # 8b. Cloudflare tunnel URL — the local server is healthy, but cloudflared
    # establishes the public tunnel a few seconds later and prints the random
    # URL into the log. Poll for it so we can surface it in the ready banner.
    public_url: str | None = None
    if tunnel:
        with console.status(
            "[dim]Waiting for Cloudflare tunnel URL...[/dim]", spinner="dots"
        ):
            public_url = read_tunnel_url()
        if public_url:
            console.print("[green]✓[/green] tunnel up")
        else:
            console.print(
                "[yellow]⚠ Tunnel URL not detected within the wait window. "
                f"Check the log ({_shorten(str(RUNTIME.log_file))}) for a "
                "trycloudflare.com URL.[/yellow]"
            )

    # 9. Ready banner
    log_hint = _shorten(str(RUNTIME.log_file))
    public_line = f"[bold]Public URL:[/bold]   {public_url}\n" if public_url else ""
    console.print(
        Panel(
            Text.from_markup(
                f"[bold]Endpoint:[/bold]     "
                f"{_base_url(effective_port, effective_host)}\n"
                f"{public_line}"
                f"[bold]Assistant ID:[/bold] EvoScientist\n"
                f"[bold]Connect via:[/bold]  any LangChain SDK / "
                f"LangGraph-compatible UI\n"
                f"[bold]Logs:[/bold]         {log_hint}\n\n"
                f"[dim]Press Ctrl+C to stop.[/dim]"
            ),
            title="[bold green]✓ Ready[/bold green]",
            border_style="green",
        )
    )

    # 10. Block on signal — mirror serve's dual-gate (threading.Event +
    # explicit SIGINT/SIGTERM handlers) so SIGTERM (no default raise) also
    # triggers clean shutdown.
    shutdown_event = threading.Event()

    def _handle_shutdown(signum: int, _frame: Any) -> None:
        shutdown_event.set()
        if signum == signal.SIGINT:
            signal.default_int_handler(signum, _frame)

    _orig_sigint = signal.signal(signal.SIGINT, _handle_shutdown)
    _orig_sigterm = signal.signal(signal.SIGTERM, _handle_shutdown)

    try:
        while not shutdown_event.is_set():
            shutdown_event.wait(timeout=0.5)
    except KeyboardInterrupt:
        shutdown_event.set()
    finally:
        signal.signal(signal.SIGINT, _orig_sigint)
        signal.signal(signal.SIGTERM, _orig_sigterm)
        # stop_langgraph_dev + stop_ccproxy run via atexit during interpreter
        # shutdown, so subprocess teardown happens AFTER this print returns.
        # Don't claim "Stopped." here — that would be a lie until atexit fires.
        console.print(
            "\n[dim]Shutting down (background cleanup may take a few seconds)...[/dim]"
        )


@contextmanager
def _start_lock() -> Iterator[None]:
    """Hold the server start lock, the one ``ensure_langgraph_dev()`` and
    ``deploy --json`` take, across the port check and the start.

    Without it, a second starter for the same port can see a backend that is
    bound but not answering yet and kill it as stale, or overwrite its
    ownership record. With it, ``deploy`` waits for that backend and then
    refuses it as a running server.
    """
    from filelock import FileLock, Timeout

    from ..langgraph_dev.manager import _FILE_LOCK_TIMEOUT, RUNTIME

    RUNTIME.pid_dir.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(RUNTIME.lock_file))
    try:
        lock.acquire(timeout=0)
    except Timeout:
        try:
            with console.status(
                "[dim]Waiting for another EvoSci process to start its server...[/dim]",
                spinner="dots",
            ):
                lock.acquire(timeout=_FILE_LOCK_TIMEOUT)
        except Timeout:
            console.print(
                f"[red]Another EvoScientist process has been starting a backend "
                f"for over {_FILE_LOCK_TIMEOUT:.0f} s.[/red]"
            )
            console.print(
                f"[dim]It holds {_shorten(str(lock.lock_file))}. Run deploy "
                f"again once it is done.[/dim]"
            )
            raise typer.Exit(1) from None
    try:
        yield
    finally:
        lock.release()


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
        os.environ["EVOSCIENTIST_LOG_LEVEL"] = "DEBUG"
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
