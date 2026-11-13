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

import contextlib
import os
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

    Blocks until Ctrl+C / SIGTERM / SIGHUP, or the front-end exits, then stops
    the front-end. The shared backend keeps running. An inherited ignored
    SIGHUP stays ignored for nohup. Never returns a value.
    """
    from ..config import apply_config_to_env
    from ..langgraph_dev.boundary import is_loopback_host
    from ..langgraph_dev.manager import RUNTIME, _format_hostport

    apply_config_to_env(config)

    try:
        cfg = build_launcher_config(config, workspace_dir)
    except LauncherError as exc:
        _render_launcher_error(exc)
        raise typer.Exit(1) from exc

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

    webui_log = RUNTIME.webui_log
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

    from .server import _stop_signals_interrupt, lock_wait_status

    runner = InstalledWebUIRunner(progress=_install_progress, log_path=webui_log)
    launcher = WebUILauncher(
        config, cfg, runner, on_lock_wait=lock_wait_status(status_box, starting)
    )
    # Install before startup: a hang-up during readiness or banner rendering
    # must reach cleanup too. The deploy guard preserves nohup and shields
    # teardown from repeated stop signals.
    with _stop_signals_interrupt():
        try:
            try:
                with console.status(starting, spinner="dots") as status:
                    status_box["status"] = status
                    result = launcher.start()
            except LauncherError as exc:
                _render_launcher_error(exc)
                raise typer.Exit(1) from exc
            finally:
                status_box.clear()
            if result.backend_started:
                console.print("[green]✓[/green] langgraph dev ready")
                console.print(
                    "[dim]The backend server keeps running after the WebUI closes, for "
                    "every EvoSci session — stop it with [bold]EvoSci server stop"
                    "[/bold].[/dim]"
                )
            else:
                console.print(
                    f"[green]✓[/green] Reusing langgraph dev already serving "
                    f"port {launcher.backend_port}"
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
                "opened in your browser"
                if ready.browser_opened
                else "open it in your browser"
            )

            # The UI reaches the backend from the BROWSER; when only the front-end is
            # exposed, remote pages load but every request fails — say so.
            remote_backend_hint = ""
            backend_public = (
                result.exposure is not None and result.exposure.public_host is not None
            )
            if not is_loopback_host(cfg.webui_host) and not backend_public:
                remote_backend_hint = (
                    f"\n[yellow]Note:[/yellow] the UI connects to the backend from the "
                    f"browser. Remote visitors cannot reach a loopback backend — run "
                    f"[bold]EvoSci config set langgraph_dev_host 0.0.0.0[/bold] (and "
                    f"[bold]EvoSci server stop[/bold] if a server already runs on "
                    f"loopback), then point the UI at "
                    f"[bold]http://<this-machine-ip>:{launcher.backend_port}[/bold].\n"
                )
            from ..langgraph_dev.assistants import chat_assistant_id
            from ..paths import Workspace

            assistant_id = chat_assistant_id(Workspace(cfg.workspace_dir))
            console.print(
                Panel(
                    Text.from_markup(
                        f"[bold]Backend:[/bold]  "
                        f"{result.backend_url}  "
                        f"[dim](langgraph dev — Assistant: {assistant_id})[/dim]\n"
                        f"[bold]WebUI:[/bold]    "
                        f"http://{_format_hostport(cfg.webui_host, cfg.webui_port)}  "
                        f"[dim]({browser_note})[/dim]\n"
                        f"[bold]Logs:[/bold]     {_shorten(str(result.log_file or RUNTIME.log_file))}\n"
                        f"          {_shorten(str(webui_log))}\n"
                        f"{remote_backend_hint}\n"
                        f"[dim]Press Ctrl+C to stop.[/dim]"
                    ),
                    title="[bold green]✓ EvoScientist WebUI[/bold green]",
                    border_style="green",
                )
            )
            from ..cli.commands import print_exposure_warnings

            # The backend as it answered the registration: it may be one another
            # session started.
            print_exposure_warnings(result.exposure, to=console)
            if not is_loopback_host(cfg.webui_host):
                console.print(
                    "[bold white on red] ⚠ PUBLIC BIND [/bold white on red] "
                    f"[bold red]WebUI listening on {cfg.webui_host} — its API reads, "
                    f"writes and uploads workspace files and installs skills, with no "
                    f"auth. Trusted networks only.[/bold red]"
                )

            # Poll the front-end so an exit also ends the terminal adapter.
            shutdown_event = threading.Event()
            while not shutdown_event.is_set():
                if not launcher.poll():
                    console.print("\n[dim]WebUI server exited.[/dim]")
                    break
                shutdown_event.wait(timeout=0.5)
        finally:
            # A closed terminal may reject output; that must not skip stop().
            with contextlib.suppress(OSError):
                console.print(
                    "\n[dim]Shutting down (background cleanup may take a few seconds)...[/dim]"
                )
            launcher.stop()
