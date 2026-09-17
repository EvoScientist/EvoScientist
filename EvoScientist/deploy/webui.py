"""``EvoSci`` WebUI mode — deploy-style LangGraph server + browser front-end.

Selected via ``ui_backend = "webui"`` (onboard → "Select UI mode" → WebUI).
Running ``EvoSci`` then becomes, in ONE terminal:

    EvoSci deploy  +  npx @evoscientist/webui

i.e. start a *full* langgraph dev server (MCP + async sub-agents, exactly like
``EvoSci deploy``) AND launch the published ``@evoscientist/webui`` Next.js
front-end via ``npx``, so the user never needs two terminals.

Design boundary: this module is the **terminal front-end** over the
shell-agnostic launcher core in :mod:`EvoScientist.deploy.launcher`. All the
reusable start / health / stop logic lives there; this file only resolves CLI
inputs, renders Rich panels for the launcher's structured results and errors,
and owns the terminal's signal-driven blocking loop. A desktop shell drives the
same :class:`~EvoScientist.deploy.launcher.WebUILauncher` without this module.

``EvoSci deploy`` stays a clean, opinionated standalone server for *external*
consumers (deep-agents-ui, agent-chat-ui, LangSmith Studio, SDK clients); WebUI
mode is a separate, parallel launcher.

``npx @evoscientist/webui@latest`` is used (not a pinned version) so each launch
transparently pulls the newest published UI — front-end fixes ship to users
without touching the EvoScientist install. The trade-off: the first launch (and
the first launch after a new release) downloads the package and needs network;
subsequent launches reuse the npm cache.
"""

from __future__ import annotations

import atexit
import os
import signal
import threading
from typing import Any

import typer  # type: ignore[import-untyped]
from rich.panel import Panel
from rich.text import Text

from ..stream.console import console
from .launcher import (
    _WEBUI_PACKAGE,
    LauncherError,
    NpxWebUIRunner,
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
        # Same port → the backend would claim it first and npx would fail to
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

    launcher = WebUILauncher(config, cfg, NpxWebUIRunner())
    console.print("[dim]Starting langgraph dev (deploy mode: MCP + async)…[/dim]")
    try:
        result = launcher.start()
    except LauncherError as exc:
        _render_launcher_error(exc)
        raise typer.Exit(1) from exc

    if result.backend_started:
        console.print("[green]✓[/green] langgraph dev ready")
        if cfg.keepalive:
            # Keepalive: the deploy-mode backend outlives this session so the
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
        console.print(f"[yellow]⚠ {warning}[/yellow]")

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
                f"[dim](opens in your browser)[/dim]\n"
                f"[bold]Logs:[/bold]     {_shorten(str(RUNTIME.log_file))}\n"
                f"{remote_backend_hint}\n"
                f"[dim]Fetching {_WEBUI_PACKAGE} via npx (first run may take a "
                f"moment)…  Press Ctrl+C to stop.[/dim]"
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
    # user closes it), so we don't leave the backend orphaned. Teardown is
    # idempotent and honours keepalive inside launcher.stop().
    atexit.register(launcher.stop)
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
        launcher.stop()
        console.print(
            "\n[dim]Shutting down (background cleanup may take a few seconds)...[/dim]"
        )
