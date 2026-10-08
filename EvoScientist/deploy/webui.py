"""``EvoSci`` WebUI mode — deploy-style LangGraph server + browser front-end.

Selected via ``ui_backend = "webui"`` (onboard → "Select UI mode" → WebUI).
Running ``EvoSci`` then becomes, in ONE terminal:

    EvoSci deploy  +  node <tools>/webui/<version>/dist/server.js

i.e. start a *full* langgraph dev server (MCP + async sub-agents, exactly like
``EvoSci deploy``) AND the ``@evoscientist/webui`` Next.js front-end that
``EvoSci setup`` installed locally, so the user never needs two terminals.

Design boundary: this module is the **terminal front-end** over the
shell-agnostic launcher core in :mod:`EvoScientist.deploy.launcher`. All the
reusable start / health / stop logic lives there; this file only resolves CLI
inputs, renders Rich panels for the launcher's structured results and errors,
and owns the terminal's signal-driven blocking loop. Other front-ends drive the
same :class:`~EvoScientist.deploy.launcher.WebUILauncher` without this module.

``EvoSci deploy`` stays a clean, opinionated standalone server for *external*
consumers (deep-agents-ui, agent-chat-ui, LangSmith Studio, SDK clients); WebUI
mode is a separate, parallel launcher.

The front-end is the newest release inside the range this core supports
(``setup/webui.py``), installed on demand when nothing is installed yet. Once
installed it starts offline. After the UI is up, a background check downloads
a newer release in range, which the next launch uses, so front-end fixes still
reach users without an EvoScientist release.
"""

from __future__ import annotations

import atexit
import os
import signal
import threading
from typing import Any

import typer  # type: ignore[import-untyped]
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

from ..stream.console import console
from .launcher import (
    InstalledWebUIRunner,
    LauncherError,
    WebUILauncher,
    build_launcher_config,
)


def _shorten(path: str) -> str:
    """Replace ``$HOME`` prefix with ``~`` for compact display."""
    home = os.path.expanduser("~")
    if path.startswith(home):
        return "~" + path[len(home) :]
    return path


def _render_launcher_error(exc: LauncherError) -> None:
    """Render a launcher failure as a red panel. The launcher's ``code`` is the
    contract; the CLI just presents ``message`` + ``detail``."""
    title = "WebUI unavailable" if exc.code == "node_missing" else "WebUI failed"
    body = exc.message
    if exc.detail:
        body += f"\n\n{exc.detail}"
    console.print(
        Panel(
            Text(body),
            title=f"[bold red]{title}[/bold red]",
            border_style="red",
        )
    )


def run_webui(config: Any, workspace_dir: str | None = None) -> None:
    """Start the deploy-style backend + the WebUI front-end, then block.

    Thin terminal adapter over :class:`~EvoScientist.deploy.launcher.WebUILauncher`.

    Args:
        config: Effective ``EvoScientistConfig`` (already env-applied upstream,
            but re-applied here so this is safe to call standalone).
        workspace_dir: Resolved workspace path; falls back to
            ``config.default_workdir`` then cwd.

    Blocks until Ctrl+C / SIGTERM, or until the front-end process exits, then
    tears down both subprocesses. Never returns a value.
    """
    from ..config import apply_config_to_env
    from ..langgraph_dev.manager import (
        RUNTIME,
        _base_url,
        _format_hostport,
        _is_loopback_host,
    )

    apply_config_to_env(config)

    cfg = build_launcher_config(config, workspace_dir)

    # Port sanity — a CLI concern, so it lives here rather than in the core.
    for label, p in (("langgraph dev", cfg.backend_port), ("WebUI", cfg.webui_port)):
        if not (1 <= p <= 65535):
            console.print(
                f"[red]Invalid {label} port {p}. Use an integer in [1, 65535].[/red]"
            )
            raise typer.Exit(1)
    if cfg.webui_port == cfg.backend_port:
        # Same port → the backend would claim it first and the WebUI would fail to
        # bind. Catch it here with a clear message instead of a cryptic error.
        console.print(
            f"[red]WebUI port and langgraph dev port must differ "
            f"(both are {cfg.webui_port}).[/red]"
        )
        console.print(
            "[dim]Change one with [bold]EvoSci config set webui_port <port>"
            "[/bold].[/dim]"
        )
        raise typer.Exit(1)

    webui_log = RUNTIME.log_file.parent / "webui.log"
    starting = "[dim]Starting langgraph dev...[/dim]"
    status_box: dict[str, Any] = {}

    def _install_progress(fraction: float, message: str) -> None:
        # An on-demand Node / WebUI install runs inside launcher.start(),
        # before the backend; show its steps on the same spinner.
        status = status_box.get("status")
        if status is None:
            return
        if fraction >= 1.0:
            status.update(starting)
        else:
            status.update(f"[dim]{escape(message)} ({int(fraction * 100)}%)...[/dim]")

    runner = InstalledWebUIRunner(progress=_install_progress, log_path=webui_log)
    launcher = WebUILauncher(config, cfg, runner)
    try:
        with console.status(starting, spinner="dots") as status:
            status_box["status"] = status
            result = launcher.start()
    except LauncherError as exc:
        _render_launcher_error(exc)
        raise typer.Exit(1) from exc
    finally:
        status_box.clear()
    # start() has already spawned the backend (unless reused) and the front-end;
    # register teardown now so a Ctrl+C while the output below renders still
    # stops them. Idempotent, and honours keepalive inside launcher.stop().
    atexit.register(launcher.stop)

    if result.backend_started:
        console.print("[green]✓[/green] langgraph dev ready")
        if cfg.keepalive:
            # Keepalive: the backend outlives this session so the
            # next same-workspace launch reuses it instantly. The front-end
            # below still stops on exit as usual.
            console.print(
                "[dim]keepalive: backend server stays up after exit — "
                "stop it with [bold]EvoSci server stop[/bold].[/dim]"
            )
    else:
        console.print(
            f"[green]✓[/green] Reusing langgraph dev already serving "
            f"port {cfg.backend_port}"
        )
    for warning in result.warnings:
        # Warnings can carry paths; a segment like "[lab]" is not markup.
        console.print(f"[yellow]⚠ {escape(warning)}[/yellow]")

    # The launcher opens the browser once the WebUI answers.
    try:
        with console.status(
            f"[dim]Starting WebUI {escape(runner.version or '')}...[/dim]",
            spinner="dots",
        ):
            ready = launcher.wait_ready()
    except LauncherError as exc:
        if exc.code == "webui_start_failed":
            log_hint = f"See {_shorten(str(webui_log))}."
            exc.detail = f"{exc.detail}\n{log_hint}" if exc.detail else log_hint
        _render_launcher_error(exc)
        raise typer.Exit(1) from exc
    runner.start_update_check()
    browser_note = (
        "opened in your browser" if ready.browser_opened else "open it in your browser"
    )

    # The UI reaches the backend from the BROWSER; when only the front-end is
    # exposed, remote pages load but every request fails — say so.
    remote_backend_hint = ""
    if not _is_loopback_host(cfg.webui_host) and _is_loopback_host(cfg.backend_host):
        remote_backend_hint = (
            f"\n[yellow]Note:[/yellow] the UI connects to the backend from the "
            f"browser. Remote visitors cannot reach a loopback backend — run "
            f"[bold]EvoSci config set langgraph_dev_host 0.0.0.0[/bold] and "
            f"point the UI at [bold]http://<this-machine-ip>:{cfg.backend_port}"
            f"[/bold].\n"
        )
    console.print(
        Panel(
            Text.from_markup(
                f"[bold]Backend:[/bold]  "
                f"{_base_url(cfg.backend_port, cfg.backend_host)}  "
                f"[dim](langgraph dev — Assistant: EvoScientist)[/dim]\n"
                f"[bold]WebUI:[/bold]    "
                f"http://{_format_hostport(cfg.webui_host, cfg.webui_port)}  "
                f"[dim]({browser_note})[/dim]\n"
                f"[bold]Logs:[/bold]     {_shorten(str(RUNTIME.log_file))}\n"
                f"          {_shorten(str(webui_log))}\n"
                f"{remote_backend_hint}\n"
                f"[dim]Press Ctrl+C to stop.[/dim]"
            ),
            title="[bold green]✓ EvoScientist WebUI[/bold green]",
            border_style="green",
        )
    )
    if not _is_loopback_host(cfg.backend_host):
        console.print(
            "[bold white on red] ⚠ PUBLIC BIND [/bold white on red] "
            f"[bold red]Backend listening on {cfg.backend_host} — no auth, and "
            f"the agent can run shell. Trusted networks only.[/bold red]"
        )
    if not _is_loopback_host(cfg.webui_host):
        console.print(
            "[bold white on red] ⚠ PUBLIC BIND [/bold white on red] "
            f"[bold red]WebUI listening on {cfg.webui_host} — its API reads, "
            f"writes and uploads workspace files and installs skills, with no "
            f"auth. Trusted networks only.[/bold red]"
        )

    # Block on signal — exit also if the front-end dies on its own (e.g. the
    # user closes it), so we don't leave the backend orphaned.
    shutdown_event = threading.Event()

    def _handle_shutdown(signum: int, _frame: Any) -> None:
        shutdown_event.set()
        if signum == signal.SIGINT:
            signal.default_int_handler(signum, _frame)

    _orig_sigint = signal.signal(signal.SIGINT, _handle_shutdown)
    _orig_sigterm = signal.signal(signal.SIGTERM, _handle_shutdown)

    try:
        while not shutdown_event.is_set():
            if not launcher.poll():
                console.print("\n[dim]WebUI server exited.[/dim]")
                break
            shutdown_event.wait(timeout=0.5)
    except KeyboardInterrupt:
        shutdown_event.set()
    finally:
        signal.signal(signal.SIGINT, _orig_sigint)
        signal.signal(signal.SIGTERM, _orig_sigterm)
        console.print(
            "\n[dim]Shutting down (background cleanup may take a few seconds)...[/dim]"
        )
        launcher.stop()
