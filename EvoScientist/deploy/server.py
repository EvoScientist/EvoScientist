"""``EvoSci deploy`` — start a standalone langgraph dev server.

Hosts the fully-equipped EvoScientist main agent (MCP + async sub-agents)
for consumption by external LangChain-compatible UIs (deep-agents-ui,
agent-chat-ui, LangSmith Studio) and SDK clients.

Differs from ``EvoSci`` / ``EvoSci serve``: no in-process CLI agent,
no session DB, no channel runtime, no TUI. The terminal only shows
startup progress, the Ready banner, and then blocks until Ctrl+C.

Mode dispatch happens via ``EVOSCIENTIST_DEPLOY_MODE=true`` env var
injected by ``start_langgraph_dev(deploy_mode=True)``. The subprocess
reads this at module-load time (``langgraph_dev/manager.py``) to flip
``_ASYNC_SUBAGENTS_AVAILABLE`` to True, and the agent build code
(``EvoScientist.py:_get_default_agent``) takes the MCP-loading branch
because ``EVOSCIENTIST_DEPLOYED_NO_MCP`` is not set.
"""

from __future__ import annotations

import atexit
import os
import signal
import threading
from pathlib import Path
from typing import Any

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
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug logging",
    ),
):
    """Deploy EvoScientist main agent as a standalone LangGraph dev server.

    Starts ``langgraph dev`` in deploy mode (full MCP + async sub-agents).
    Connect any LangChain-compatible UI or SDK client to the printed
    endpoint. Press Ctrl+C to stop.
    """
    from ..config import apply_config_to_env, get_effective_config
    from ..langgraph_dev.manager import (
        _DEFAULT_PORT,
        _is_port_occupied,
        is_langgraph_dev_running,
        start_langgraph_dev,
        stop_langgraph_dev,
    )
    from ..paths import ensure_dirs, set_workspace_root

    # 1. Load config (no CLI overrides here — deploy is opinionated about
    # full MCP + async; user-facing flags are workspace/port/debug only).
    cli_overrides: dict[str, Any] = {}
    if debug:
        cli_overrides["log_level"] = "DEBUG"
    config = get_effective_config(cli_overrides)
    if debug:
        os.environ["EVOSCIENTIST_LOG_LEVEL"] = "DEBUG"
        from ..cli.commands import _configure_logging

        _configure_logging()
    apply_config_to_env(config)

    # 2. Resolve workspace (CLI > config.default_workdir > cwd)
    if workdir:
        ws = os.path.abspath(os.path.expanduser(workdir))
    elif config.default_workdir:
        ws = os.path.abspath(os.path.expanduser(config.default_workdir))
    else:
        ws = os.getcwd()
    os.makedirs(ws, exist_ok=True)
    set_workspace_root(ws)
    ensure_dirs()

    # 3. Resolve port
    effective_port = port or int(getattr(config, "langgraph_dev_port", _DEFAULT_PORT))

    # 4. Pre-flight port check — refuse to start if a non-EvoSci process is
    # holding the port. If an existing EvoSci langgraph dev is already up,
    # also refuse (deploy is the "primary server" — running multiple on the
    # same port is a configuration error).
    if _is_port_occupied(effective_port):
        if is_langgraph_dev_running(port=effective_port):
            console.print(
                f"[red]Port {effective_port} is already serving a langgraph dev "
                f"instance.[/red]"
            )
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
                f"[bold]Port:[/bold]      {effective_port}\n"
                f"[bold]Auth:[/bold]      {_auth_label}"
            ),
            title="[bold cyan]EvoScientist Deploy[/bold cyan]",
            border_style="cyan",
        )
    )

    # 6. ccproxy lifecycle (only if any provider uses OAuth)
    _ccproxy_proc = None
    if config.anthropic_auth_mode == "oauth" or config.openai_auth_mode == "oauth":
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

    # 7. Start langgraph dev (deploy mode → full MCP + async)
    jobs_per_worker = int(getattr(config, "langgraph_dev_jobs_per_worker", 10))
    file_persistence = bool(getattr(config, "langgraph_dev_file_persistence", True))
    try:
        with console.status(
            "[dim]Starting langgraph dev (deploy mode: MCP + async)...[/dim]",
            spinner="dots",
        ):
            proc = start_langgraph_dev(
                workspace_dir=Path(ws),
                port=effective_port,
                file_persistence=file_persistence,
                jobs_per_worker=jobs_per_worker,
                deploy_mode=True,
            )
        atexit.register(stop_langgraph_dev, proc)
    except Exception as exc:
        console.print(f"[red]langgraph dev startup failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    # 8. Wait for health (start_langgraph_dev returns after its own health
    # poll, but we double-check to make the Ready banner honest).
    if not is_langgraph_dev_running(port=effective_port):
        console.print("[red]langgraph dev is not responding on /ok endpoint.[/red]")
        console.print(
            "[dim]Check the log file for MCP-spawn errors or import failures.[/dim]"
        )
        raise typer.Exit(1)

    console.print("[green]✓[/green] langgraph dev ready")

    # 9. Ready banner
    log_hint = "~/.config/evoscientist/langgraph_dev.log"
    console.print(
        Panel(
            Text.from_markup(
                f"[bold]Endpoint:[/bold]     "
                f"http://localhost:{effective_port}\n"
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
        console.print("\n[dim]Shutting down...[/dim]")
        # atexit handles stop_langgraph_dev + stop_ccproxy in correct order.
        console.print("[dim]Stopped.[/dim]")


def _describe_auth(config: Any) -> str:
    """Render a one-line auth summary for the startup banner."""
    anth = getattr(config, "anthropic_auth_mode", "api_key")
    oai = getattr(config, "openai_auth_mode", "api_key")
    if anth == "oauth" and oai == "oauth":
        return "OAuth (Anthropic + OpenAI via ccproxy)"
    if anth == "oauth":
        return "OAuth (Anthropic via ccproxy) + API key (OpenAI)"
    if oai == "oauth":
        return "API key (Anthropic) + OAuth (OpenAI via ccproxy)"
    return "API key"


def _shorten(path: str) -> str:
    """Replace ``$HOME`` prefix with ``~`` for compact display."""
    home = os.path.expanduser("~")
    if path.startswith(home):
        return "~" + path[len(home) :]
    return path
