"""Typer command registrations — onboard, config, mcp, main callback."""

import json
import logging
import os
import queue
import re
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

import click
import typer
from rich.markup import escape
from rich.table import Table

from ..commands.base import (
    ChannelRuntime,
    Command,
    CommandContext,
    active_teams_configurable_extra,
)
from ..gateway import (
    GraphGateway,
    GraphTarget,
    RunRequest,
)
from ..llm.context_window import DEFAULT_CONTEXT_WINDOW_FALLBACK, resolve_context_window
from ..paths import (
    SessionDirs,
    Workspace,
    ensure_dirs,
    reload_env_dirs,
    start_workspace_path,
)
from ..runtime import AsyncRuntime
from ..stream.console import console
from . import (
    async_notifier,
    server_cmd,  # noqa: F401 — registers `EvoSci server` commands
)
from ._app import app, channel_app, config_app, configure_app, mcp_app, sessions_app
from ._constants import build_metadata
from .agent import (
    _create_run_dir,
    _load_agent,
    _remove_unused_run_dir,
    _shorten_path,
)
from .channel import (
    ChannelMessage,
    _channel_message_cancel_scope,
    _channels_stop,
    _claim_or_complete_channel_request,
    _complete_channel_request,
    _message_queue,
    _set_channel_response,
    _set_channels_media_dir,
    _start_channels_bus_mode,
    channel_ask_user_prompt,
    channel_hitl_prompt,
    dispatch_channel_slash_command,
    forget_channel_origin,
    get_channel_origin,
    publish_to_channel_origin,
    remember_channel_origin,
)
from .channel_sends import PendingChannelSends
from .mcp_ui import (
    _mcp_add_server_from_kwargs,
    _mcp_edit_server_fields,
    _mcp_list_servers,
    _mcp_remove_server,
    _show_mcp_config,
)

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from ..config import EvoScientistConfig
    from ..gateway import RuntimeGateways


_ASYNC_RUNTIME_META_KEY = "evoscientist.async_runtime"


def _close_cli_async_runtime(runtime: AsyncRuntime) -> None:
    """Close the owned runtime or surface a controlled CLI shutdown failure."""
    try:
        runtime.close()
    except TimeoutError as exc:
        click.echo(
            f"Error: Async runtime shutdown did not complete: {exc}",
            err=True,
        )
        raise click.exceptions.Exit(1) from None


def _get_cli_async_runtime(ctx: typer.Context) -> AsyncRuntime:
    """Return the application-scoped runtime owned by this CLI invocation."""
    root = ctx.find_root()
    runtime = root.meta.get(_ASYNC_RUNTIME_META_KEY)
    if runtime is None:
        runtime = AsyncRuntime()
        root.meta[_ASYNC_RUNTIME_META_KEY] = runtime
        root.call_on_close(lambda: _close_cli_async_runtime(runtime))
    if not isinstance(runtime, AsyncRuntime):  # pragma: no cover - defensive
        raise RuntimeError("CLI async runtime context is invalid")
    return runtime


# =============================================================================
# Onboard command
# =============================================================================


@app.command()
def onboard(
    ctx: typer.Context,
    skip_validation: bool = typer.Option(
        False, "--skip-validation", help="Skip API key validation during setup"
    ),
    # ---- Pre-fill answers (any subset; remaining prompts stay interactive)
    provider: str | None = typer.Option(
        None, "--provider", help="Pre-set LLM provider (e.g. anthropic, openai)"
    ),
    model: str | None = typer.Option(None, "--model", help="Pre-set model name"),
    api_key: str | None = typer.Option(
        None, "--api-key", help="Pre-set API key for the chosen --provider"
    ),
    tavily_key: str | None = typer.Option(
        None, "--tavily-key", help="Pre-set Tavily API key"
    ),
    workspace_mode: str | None = typer.Option(
        None,
        "--workspace-mode",
        help="Pre-set workspace mode (daemon | run)",
    ),
    show_thinking: bool | None = typer.Option(
        None,
        "--show-thinking/--no-show-thinking",
        help="Pre-set thinking-panel visibility",
    ),
    ui: str | None = typer.Option(
        None, "--ui", help="Pre-set UI backend (tui | cli | webui)"
    ),
    port: int | None = typer.Option(
        None, "--port", help="Pre-set langgraph dev server port"
    ),
    # ---- Skip flags
    skip_skills: bool = typer.Option(
        False, "--skip-skills", help="Skip skills install"
    ),
    skip_mcp: bool = typer.Option(False, "--skip-mcp", help="Skip MCP server setup"),
    skip_latex: bool = typer.Option(False, "--skip-latex", help="Skip LaTeX setup"),
    skip_channels: bool = typer.Option(
        False, "--skip-channels", help="Skip channels setup"
    ),
    non_interactive: bool = typer.Option(
        False,
        "--non-interactive",
        help="Run without prompts — every required answer must come from a flag",
    ),
):
    """Interactive setup wizard for EvoScientist.

    Guides you through configuring API keys, model selection,
    workspace settings, and agent parameters.

    Any answer can be pre-set via a flag (``--provider anthropic
    --model claude-sonnet-4-6 ...``); prompts for unset answers stay
    interactive unless ``--non-interactive`` is passed, in which case any
    missing required answer aborts the wizard.
    """
    from ..config.onboard.constants import (
        VALID_PROVIDERS,
        VALID_UI_BACKENDS,
        VALID_WORKSPACE_MODES,
    )
    from ..config.onboard.prompter import NonInteractivePrompter

    # Validate constrained string flags up-front so a typo doesn't silently
    # poison the saved config. Allowed-value sets live in
    # ``EvoScientist/config/onboard/constants.py``; a drift test in
    # ``tests/test_onboard.py`` keeps them aligned with the interactive
    # ``Choice(value=...)`` lists in ``steps.py``.
    if ui is not None and ui not in VALID_UI_BACKENDS:
        raise typer.BadParameter(
            f"--ui must be one of {sorted(VALID_UI_BACKENDS)}", param_hint="--ui"
        )
    if workspace_mode is not None and workspace_mode not in VALID_WORKSPACE_MODES:
        raise typer.BadParameter(
            f"--workspace-mode must be one of {sorted(VALID_WORKSPACE_MODES)}",
            param_hint="--workspace-mode",
        )
    if provider is not None and provider not in VALID_PROVIDERS:
        raise typer.BadParameter(
            f"--provider must be one of {sorted(VALID_PROVIDERS)}",
            param_hint="--provider",
        )
    # Match the interactive prompt's range (1024 < port < 65536). Without
    # this check, --port 80 or --port 99999 would land in config and break
    # the langgraph dev server on startup.
    if port is not None and not (1024 < port < 65536):
        raise typer.BadParameter(
            "--port must be in the user-port range (1025 — 65535)",
            param_hint="--port",
        )

    # Collect flag-supplied answers keyed by the prompt_id wizard steps use.
    answers: dict = {}
    if ui is not None:
        answers["ui"] = ui
    if port is not None:
        answers["port"] = str(port)
    if provider is not None:
        answers["provider"] = provider
    if model is not None:
        answers["model"] = model
    if api_key is not None:
        answers["api_key"] = api_key
    if tavily_key is not None:
        answers["tavily_key"] = tavily_key
    if workspace_mode is not None:
        answers["workspace_mode"] = workspace_mode
    if show_thinking is not None:
        answers["show_thinking"] = show_thinking

    skip_set = {
        section
        for section, flag in (
            ("skills", skip_skills),
            ("mcp", skip_mcp),
            ("latex", skip_latex),
            ("channels", skip_channels),
        )
        if flag
    }

    prompter = None
    if answers or skip_set or non_interactive:
        prompter = NonInteractivePrompter(
            answers=answers,
            skip_set=skip_set,
            strict=non_interactive,
        )

    _run_onboard_cli(
        skip_validation=skip_validation,
        prompter=prompter,
        runtime=_get_cli_async_runtime(ctx),
    )


# =============================================================================
# Setup command
# =============================================================================


@app.command()
def setup(
    manifest: bool = typer.Option(
        False, "--manifest", help="Print the setup stages as JSON and exit"
    ),
    stage: str | None = typer.Option(
        None, "--stage", help="Run only this stage (ids from --manifest)"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Print one JSON event per line instead of progress text"
    ),
    cn: bool = typer.Option(
        False, "--cn", help="Download from mainland China mirrors and remember it"
    ),
    skip: Annotated[
        list[str] | None,
        typer.Option(
            "--skip", help="Skip this stage (repeatable; ids from --manifest)"
        ),
    ] = None,
):
    """Install what EvoScientist needs beyond the Python package (Node.js, Git
    for Windows on Windows, a Python for the agent's shell when none is on
    PATH, and the WebUI).

    Runs every stage that applies to this platform, in order. With ``--json``
    stdout carries only the JSON event lines; everything else goes to stderr.
    """
    import json
    import sys

    from ..config import load_config, set_config_value
    from ..setup import STAGES, get_stage, run_stages
    from ..setup import manifest as setup_manifest
    from ..setup.protocol import ConsoleEmitter, JsonEmitter

    if manifest:
        sys.stdout.write(json.dumps(setup_manifest()) + "\n")
        return

    if json_output:
        # The shared console also carries log warnings; stdout belongs to the
        # event lines.
        from ..stream.json_sink import redirect_console_to_stderr

        redirect_console_to_stderr()

    known = ", ".join(s.id for s in STAGES)
    skipped = frozenset(skip or ())
    unknown = sorted(skipped - {s.id for s in STAGES})
    if unknown:
        typer.echo(
            f"Unknown stage {', '.join(map(repr, unknown))}. Known stages: {known}",
            err=True,
        )
        raise typer.Exit(2)
    if skipped and stage is not None:
        typer.echo("--skip cannot be combined with --stage.", err=True)
        raise typer.Exit(2)

    selected = tuple(s for s in STAGES if s.applies())
    if stage is not None:
        found = get_stage(stage)
        if found is None:
            typer.echo(f"Unknown stage {stage!r}. Known stages: {known}", err=True)
            raise typer.Exit(2)
        selected = (found,)

    if cn:
        try:
            set_config_value("mirror", "cn")
        except OSError as exc:
            # Saving the choice is secondary; this run still uses the mirror.
            logging.getLogger(__name__).warning(
                f"Could not save mirror: cn to the config ({exc}); "
                "using the mirror for this run only."
            )
    mirror = "cn" if cn else load_config().mirror

    emit = JsonEmitter() if json_output else ConsoleEmitter(console)
    code = run_stages(selected, emit, mirror, skip=skipped)
    if code:
        raise typer.Exit(code)


# =============================================================================
# `EvoSci configure <section>` — re-run one onboarding section
# =============================================================================


_CONFIGURE_SECTIONS = {
    "ui": "UI backend",
    "port": "LangGraph server port",
    "provider": "LLM provider + auth + API key",
    "model": "Model + reasoning effort",
    "tavily": "Tavily search key",
    "workspace": "Workspace mode",
    "thinking": "Thinking panel",
    "skills": "Skills",
    "mcp": "MCP servers",
    "latex": "LaTeX (TinyTeX)",
    "channels": "Channels",
}


def _run_onboard_cli(**kwargs: Any) -> None:
    """Invoke the wizard, presenting non-interactive errors as a clean
    message + exit code 1 instead of a raw Python traceback.

    The wizard raises ``RuntimeError`` for *expected* non-interactive
    failures: rejected ``--api-key`` / ``--tavily-key`` presets, missing
    required flags under ``--non-interactive``, or a missing base URL.
    Those are user-input problems, not bugs — surface them like any other
    CLI validation error rather than dumping a stack trace.
    """
    from ..config import run_onboard

    try:
        run_onboard(**kwargs)
    except RuntimeError as exc:
        console.print(f"[red]✗ {escape(str(exc))}[/red]")
        raise typer.Exit(code=1) from exc


def _configure_section(
    section: str,
    skip_validation: bool = False,
    *,
    runtime: AsyncRuntime | None = None,
) -> None:
    """Run a single onboarding section, reusing the wizard's step logic."""
    kwargs: dict[str, Any] = {
        "skip_validation": skip_validation,
        "only_sections": {section},
    }
    if runtime is not None:
        kwargs["runtime"] = runtime
    _run_onboard_cli(
        **kwargs,
    )


@configure_app.command("ui")
def configure_ui():
    """Re-run UI backend (TUI / CLI) selection."""
    _configure_section("ui")


@configure_app.command("port")
def configure_port():
    """Re-run langgraph dev server port selection."""
    _configure_section("port")


@configure_app.command("provider")
def configure_provider(
    skip_validation: bool = typer.Option(False, "--skip-validation"),
):
    """Re-run LLM provider, auth mode, and API key prompts.

    Model selection is automatically re-run after provider — the model list
    depends on the provider, and silently leaving e.g. ``model="claude-...""``
    when the provider was switched to ``openai`` would break the first
    request. Press Enter on the model picker to keep the current default.
    """
    _run_onboard_cli(
        skip_validation=skip_validation,
        only_sections={"provider", "model"},
    )


@configure_app.command("model")
def configure_model():
    """Re-run model selection (and reasoning effort for OpenRouter)."""
    _configure_section("model")


@configure_app.command("tavily")
def configure_tavily(
    skip_validation: bool = typer.Option(False, "--skip-validation"),
):
    """Re-run Tavily search-key prompt."""
    _configure_section("tavily", skip_validation=skip_validation)


@configure_app.command("workspace")
def configure_workspace():
    """Re-run workspace mode (daemon/run) selection."""
    _configure_section("workspace")


@configure_app.command("thinking")
def configure_thinking():
    """Re-run thinking-panel visibility selection."""
    _configure_section("thinking")


@configure_app.command("skills")
def configure_skills():
    """Re-run skills install/sync."""
    _configure_section("skills")


@configure_app.command("mcp")
def configure_mcp():
    """Re-run MCP server selection."""
    _configure_section("mcp")


@configure_app.command("latex")
def configure_latex():
    """Re-run LaTeX (TinyTeX) setup."""
    _configure_section("latex")


@configure_app.command("channels")
def configure_channels(ctx: typer.Context):
    """Re-run channels selection and per-channel configuration."""
    _configure_section("channels", runtime=_get_cli_async_runtime(ctx))


# =============================================================================
# Channel setup command
# =============================================================================


@channel_app.command("setup")
def channel_setup(ctx: typer.Context):
    """Interactive channel configuration wizard.

    Guides you through selecting and configuring messaging channels
    (Telegram, Discord, or iMessage).
    """
    from ..config import load_config, save_config
    from ..config.onboard.channels import _step_channels

    config = load_config()
    updates = _step_channels(config, runtime=_get_cli_async_runtime(ctx))
    if updates:
        for key, value in updates.items():
            setattr(config, key, value)
        save_config(config)
        console.print("[green]Channel configuration saved.[/green]")
    else:
        console.print("[dim]No changes made.[/dim]")


# =============================================================================
# Compact helper
# =============================================================================

_COMPACT_CONTEXT_WINDOW_FALLBACK = DEFAULT_CONTEXT_WINDOW_FALLBACK
_MANUAL_COMPACT_MIN_FRACTION = 0.40
_MANUAL_COMPACT_MIN_PERCENT = int(_MANUAL_COMPACT_MIN_FRACTION * 100)


class CompactResult:
    """Structured result from compact_conversation.

    Attributes:
        status: "noop" (nothing to compact), "ok" (compacted), or "error".
        message: Short human-readable message (used as fallback / TUI text).
        messages_compacted: Number of messages summarized (0 for noop/error).
        messages_kept: Number of messages unchanged.
        tokens_before: Total tokens before compaction.
        tokens_after: Total tokens after compaction.
        tokens_summarized: Tokens in the summarized portion (before).
        tokens_summary: Tokens in the summary message (after).
        pct_decrease: Percentage decrease.
        context_window: Model context window used for thresholding.
        context_percent: Effective context utilization percent.
        summary_text: Human-readable compact summary content for UI display.
    """

    __slots__ = (
        "context_percent",
        "context_window",
        "message",
        "messages_compacted",
        "messages_kept",
        "pct_decrease",
        "status",
        "summary_text",
        "tokens_after",
        "tokens_before",
        "tokens_summarized",
        "tokens_summary",
    )

    def __init__(
        self,
        status: str,
        message: str,
        *,
        messages_compacted: int = 0,
        messages_kept: int = 0,
        tokens_before: int = 0,
        tokens_after: int = 0,
        tokens_summarized: int = 0,
        tokens_summary: int = 0,
        pct_decrease: int = 0,
        context_window: int = 0,
        context_percent: int = 0,
        summary_text: str = "",
    ):
        self.status = status
        self.message = message
        self.messages_compacted = messages_compacted
        self.messages_kept = messages_kept
        self.tokens_before = tokens_before
        self.tokens_after = tokens_after
        self.tokens_summarized = tokens_summarized
        self.tokens_summary = tokens_summary
        self.pct_decrease = pct_decrease
        self.context_window = context_window
        self.context_percent = context_percent
        self.summary_text = summary_text

    def __str__(self) -> str:
        return self.message


class CompactSummaryRenderable:
    """Rich renderable payload for the manual compact summary content."""

    __slots__ = ("summary_text",)

    def __init__(self, summary_text: str):
        self.summary_text = (summary_text or "").strip()

    def __rich_console__(self, console, options):
        yield render_compact_summary_panel(self.summary_text)


def _ensure_async_subagent_server(
    config: Any, *, dirs: SessionDirs, backend: str | None = None
) -> None:
    """Start the langgraph dev subprocess for background agent work.

    Shared by both the interactive entry and the serve entry so the
    user-visible status message and workspace-mismatch handling stay in one
    place.

    ``backend`` is the calling surface's resolved gateway backend, forwarded to
    ``ensure_langgraph_dev`` so whether a server is needed follows the
    surface's choice rather than re-reading the global flag. ``None`` keeps the
    global-read behavior.

    Raises ``typer.Exit(1)`` (after surfacing a red error) when an
    externally-managed langgraph dev is already running for a different
    workspace — e.g., ``EvoSci deploy --workdir /A`` is up and the user
    is starting ``EvoSci`` / ``EvoSci serve`` in /B. Continuing in that
    state would route async sub-agent calls to a process pinned to /A
    while the main agent runs in /B.
    """
    from ..langgraph_dev.manager import (
        _DEFAULT_HOST,
        WorkspaceMismatchError,
        _is_loopback_host,
        ensure_langgraph_dev,
        is_async_subagents_available,
    )

    try:
        with console.status(
            "[dim]Starting background agent server (langgraph dev)...[/dim]",
            spinner="dots",
        ):
            ensure_langgraph_dev(
                config,
                workspace_dir=dirs.workspace.root,
                run_dir=dirs.run_dir,
                backend=backend,
            )
            _reconcile_autoskill_schedule(config, workspace=dirs.workspace)
            _adopt_stale_scheduled_tasks(workspace=dirs.workspace)
    except WorkspaceMismatchError as exc:
        console.print(f"[red]{exc}[/red]")
        _remove_unused_run_dir(dirs.run_dir)
        raise typer.Exit(1) from exc

    from ..langgraph_dev import manager as _lg_manager

    if _lg_manager.CONFIG_DRIFT_SINCE_LAUNCH:
        console.print(
            "[yellow]⚠ Configuration or version changed since the background "
            "agent server was launched — it still runs the old settings or "
            "version. Apply them with [bold]EvoSci server stop[/bold], then "
            "restart EvoSci.[/yellow]"
        )
    if _lg_manager.AGENT_PYTHON_DRIFT is not None:
        console.print(f"[yellow]⚠ {escape(_lg_manager.AGENT_PYTHON_DRIFT)}[/yellow]")

    # The backend is shared by every UI mode, so the exposure warning lives
    # here, not just in deploy/WebUI. Gated on the server being up: warning
    # about a bind that never happened would be worse than saying nothing.
    bind_host = str(getattr(config, "langgraph_dev_host", _DEFAULT_HOST) or "").strip()
    if (
        bind_host
        and not _is_loopback_host(bind_host)
        and is_async_subagents_available()
    ):
        console.print(
            "[bold white on red] ⚠ PUBLIC BIND [/bold white on red] "
            f"[bold red]Agent server listening on {bind_host} — no auth, and "
            f"the agent can run shell. Use --host 127.0.0.1 on untrusted "
            f"networks.[/bold red]"
        )


def warn_server_backend_hitl_caveats(
    backend: str | None, *, surface_label: str
) -> None:
    """Warn that the server gateway backend is lossy for a HITL surface.

    The interactive CLI and TUI create a per-session workspace and switch models
    mid-session, but on the server backend neither reaches the run: execution
    stays in the server's import-time workspace, and per-run config (model,
    active teams, HITL suppression) is dropped when a turn resumes after a tool
    approval. Both are gaps the server path has not closed yet, so a surface
    that opts into the server backend surfaces this once at startup rather than
    silently misrouting.
    """
    if backend != "langgraph_server":
        return
    console.print(
        f"[yellow]⚠ Server gateway backend active for the {surface_label}. "
        "Until the remaining gaps close, the per-session workspace is not "
        "applied server-side (execute / write_file run in the server's "
        "workspace), and a model or team switch is dropped for the rest of a "
        "turn that resumes after a tool approval.[/yellow]"
    )


def _reconcile_autoskill_schedule(config: Any, *, workspace: Workspace) -> None:
    """Best-effort reconciliation for EvoMemory's hidden AutoSkills cron.

    The cron belongs to the workspace, whichever run folder the session
    works in. Only for a server known to serve *workspace*, as for
    ``_adopt_stale_scheduled_tasks``: the reconcile replaces crons it counts as
    the workspace's, and in another workspace's store those are not.
    """
    try:
        from ..langgraph_dev import manager
        from ..memory.autoskills.schedule import reconcile_autoskill_schedule

        if manager.SERVED_WORKSPACE != workspace.root:
            return
        reconcile_autoskill_schedule(config, workspace_dir=workspace.root)
    except Exception:
        logging.getLogger(__name__).warning(
            "Failed to reconcile EvoMemory AutoSkills schedule", exc_info=True
        )


def _adopt_stale_scheduled_tasks(*, workspace: Workspace) -> None:
    """Best-effort re-tagging of the served store's untagged and moved tasks.

    Their runs then work in *workspace* and pass the server's folder check.
    Only for a server known to serve *workspace*: a reused server without a
    workspace record, or one another session has since moved, may hold another
    workspace's store, whose crons this must not take over.
    """
    try:
        from ..cron import schedule as crons
        from ..langgraph_dev import manager

        if manager.SERVED_WORKSPACE != workspace.root:
            return
        if crons.is_available():
            crons.adopt_stale_tasks(
                crons._client(), workspace, run_kind=crons.SCHEDULED_RUN_KIND
            )
    except Exception:
        logging.getLogger(__name__).warning(
            "Failed to re-tag stale scheduled tasks", exc_info=True
        )


def _pending_skill_proposals_message(workspace_dir: str | Path) -> str | None:
    """Return a concise review reminder when autoskill proposals are waiting."""
    try:
        from .. import paths
        from ..memory.autoskills.proposals import pending_skill_proposal_count

        count = pending_skill_proposal_count(
            paths.MEMORIES_DIR,
            workspace_dir=workspace_dir,
        )
    except Exception:
        return None
    if not count:
        return None
    return (
        f"EvoMemory has {count} autoskill proposal(s) ready for review. "
        "Run /autoskills review."
    )


# How long a CLI waits for EvoMemory work before it exits or moves the server.
_MEMORY_WAIT_SECONDS = 120.0
_MEMORY_WAIT_POLL_SECONDS = 0.5
_MEMORY_WAIT_OUTPUT_GRACE_SECONDS = 3.0


def _wait_for_memory_workers(*, timeout_action: str = "shutting down") -> None:
    """Wait, bounded, for queued EvoMemory work to finish.

    One-shot runs wait so post-run memory persists before atexit cleanup;
    moving the server waits because a move stops runs in flight.
    ``timeout_action`` says what happens when the wait runs out. Returns at
    once when nothing runs.
    """
    try:
        from ..memory.worker_activity import (
            MemoryActivityPhase,
            MemoryWorkerStatusSnapshot,
            wait_for_memory_pipeline_idle,
        )
    except Exception:
        return

    announced = False

    def print_saved(observed: MemoryWorkerStatusSnapshot) -> None:
        saved = []
        if observed.observations_recorded:
            saved.append(f"{observed.observations_recorded} observation(s)")
        if observed.profile_updates:
            saved.append(f"{observed.profile_updates} profile update(s)")
        if saved:
            console.print(f"[dim]EvoMemory saved {', '.join(saved)}.[/dim]")

    def print_waiting(phase: MemoryActivityPhase) -> None:
        nonlocal announced
        if not announced:
            console.print(f"[dim]Waiting for EvoMemory {phase}...[/dim]")
            announced = True

    def print_timeout(phase: MemoryActivityPhase) -> None:
        console.print(
            f"[dim]EvoMemory {phase} is still running; {timeout_action}.[/dim]"
        )

    wait_for_memory_pipeline_idle(
        timeout_seconds=_MEMORY_WAIT_SECONDS,
        poll_seconds=_MEMORY_WAIT_POLL_SECONDS,
        output_grace_seconds=_MEMORY_WAIT_OUTPUT_GRACE_SECONDS,
        on_saved=print_saved,
        on_waiting=print_waiting,
        on_timeout=print_timeout,
    )


_RESUME_SYNC_STATUS = (
    "[dim]Syncing background agent server to resumed workspace...[/dim]"
)


async def _sync_background_agent_server_workspace(
    config: Any,
    *,
    dirs: SessionDirs,
    backend: str | None = None,
    status_message: str = _RESUME_SYNC_STATUS,
) -> None:
    """Sync langgraph dev to a resumed workspace for background agent work.

    ``ensure_langgraph_dev`` is intentionally always called: EvoMemory
    background workers require the server even when async subagents are disabled.
    WorkspaceMismatchError is left for callers to handle according to their UI
    flow.

    ``backend`` is the calling surface's resolved gateway backend, forwarded so
    whether a server is needed follows the surface's choice; ``None`` keeps
    the global-read behavior.
    """
    import asyncio

    from ..langgraph_dev.manager import (
        ensure_langgraph_dev,
        owned_server_pinned_elsewhere,
    )

    with console.status(status_message, spinner="dots"):
        # Only a move stops runs in flight; a sync to the served folders is a
        # health check and must not wait on memory work.
        if await asyncio.to_thread(
            owned_server_pinned_elsewhere, dirs.workspace.root, dirs.run_dir
        ):
            await asyncio.to_thread(
                _wait_for_memory_workers, timeout_action="moving the server anyway"
            )
        await asyncio.to_thread(
            ensure_langgraph_dev,
            config,
            workspace_dir=dirs.workspace.root,
            run_dir=dirs.run_dir,
            backend=backend,
        )
        await asyncio.to_thread(
            _reconcile_autoskill_schedule,
            config,
            workspace=dirs.workspace,
        )
        await asyncio.to_thread(_adopt_stale_scheduled_tasks, workspace=dirs.workspace)


async def _move_server_or_keep_session(
    config: Any,
    *,
    target: SessionDirs,
    current: SessionDirs | None,
    backend: str | None = None,
    new_run_dir: bool = False,
    status_message: str = _RESUME_SYNC_STATUS,
) -> bool:
    """Move the background agent server to *target* before a session goes there.

    Until the server serves several workspaces it is pinned to one workspace
    and run folder, so background workers and deployed sub-agents would
    otherwise keep working in *current*, the folders the server was last
    synced to (``None`` when it was not started yet). Nothing happens when
    the two are the same.

    Raises ``WorkspaceMismatchError`` when another EvoSci session holds the
    server for other folders: the caller keeps its session as it is, and the
    run folder made for *target* (``new_run_dir``) is removed. Returns False
    when the move failed otherwise: the session can go to *target*, but
    async sub-agents and EvoMemory workers may be unavailable.
    """
    if target == current:
        return True

    from ..langgraph_dev.manager import WorkspaceMismatchError

    try:
        await _sync_background_agent_server_workspace(
            config, dirs=target, backend=backend, status_message=status_message
        )
    except WorkspaceMismatchError:
        if new_run_dir:
            _remove_unused_run_dir(target.run_dir)
        raise
    except Exception:
        logging.getLogger(__name__).warning(
            "Failed to sync background agent server to %s; continuing in degraded mode",
            target.work_dir,
            exc_info=True,
        )
        return False
    return True


async def _restore_thread_dirs(
    thread_id: str,
    *,
    dirs: SessionDirs,
    served: SessionDirs | None,
    graph_gateway: GraphGateway,
    config: Any,
    backend: str | None = None,
) -> SessionDirs:
    """Return the folders to resume *thread_id* in, with the server moved there.

    A thread resumes in the workspace and run folder it was stored with,
    wherever the CLI was started; a thread stored without folders resumes in
    *dirs*, the folders the session started in. *served* are the folders the
    server was started for, ``None`` when it was not started yet. A run
    folder made for *dirs* that the thread does not use is removed. Raises
    ``typer.Exit(1)`` when another session holds the server.
    """
    from ..langgraph_dev.manager import WorkspaceMismatchError

    metadata = await graph_gateway.get_thread_metadata(thread_id) or {}
    restored = (
        SessionDirs.from_stored(metadata.get("workspace_dir"), metadata.get("run_dir"))
        or dirs
    )
    try:
        synced = await _move_server_or_keep_session(
            config, target=restored, current=served, backend=backend
        )
    except WorkspaceMismatchError as exc:
        _remove_unused_run_dir(dirs.run_dir)
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    if not synced:
        console.print(
            "[yellow]Background agent server sync failed; async subagents and "
            "EvoMemory workers may be unavailable.[/yellow]"
        )
    if restored != dirs:
        _remove_unused_run_dir(dirs.run_dir)
    return restored


def _resolve_context_window(
    model: Any, fallback: int = _COMPACT_CONTEXT_WINDOW_FALLBACK
) -> int:
    """Resolve a model context window with a stable fallback."""
    return resolve_context_window(model, fallback=fallback)


def _percent_used(tokens: int, context_window: int) -> int:
    """Return a clamped utilization percent."""
    if context_window <= 0:
        return 0
    return max(0, min(100, round((tokens / context_window) * 100)))


def render_compact_result(result: CompactResult):  # -> rich.text.Text
    """Render a CompactResult as styled Rich Text.

    Uses the same visual language as the token usage display:
    cyan for numbers, green for savings, dim for labels.
    """
    from rich.text import Text

    output = Text()

    if result.status == "noop":
        output.append("○ ", style="dim")
        output.append("Manual compact not needed", style="dim")
        if result.tokens_before > 0:
            output.append("  [", style="dim")
            output.append(f"{result.tokens_before:,}", style="cyan")
            if result.context_window > 0:
                output.append(" / ", style="dim")
                output.append(f"{result.context_window:,}", style="cyan")
                output.append(" tokens", style="dim")
                output.append("  │  ", style="dim")
                output.append(f"{result.context_percent}%", style="cyan")
                output.append(" of window", style="dim")
            else:
                output.append(" tokens", style="dim")
            output.append("]", style="dim")
        if result.message:
            output.append("\n  ", style="")
            output.append(result.message, style="dim")
        return output

    if result.status == "error":
        output.append("✗ ", style="red")
        output.append(result.message, style="red")
        return output

    # status == "ok"
    output.append("✓ ", style="green")
    output.append("Compacted ", style="dim")
    output.append(f"{result.messages_compacted}", style="bold")
    output.append(" messages", style="dim")
    output.append("  [", style="dim")
    output.append(f"{result.tokens_before:,}", style="cyan")
    output.append(" → ", style="dim")
    output.append(f"{result.tokens_after:,}", style="green")
    output.append(" tokens", style="dim")
    output.append(f"  ↓{result.pct_decrease}%", style="green bold")
    output.append("]", style="dim")

    # Second line: detail breakdown
    output.append("\n  ", style="")
    output.append("Summarized: ", style="dim")
    output.append(f"{result.tokens_summarized:,}", style="cyan")
    output.append(" → ", style="dim")
    output.append(f"{result.tokens_summary:,}", style="green")
    output.append("  │  ", style="dim")
    output.append("Kept: ", style="dim")
    output.append(f"{result.messages_kept}", style="cyan")
    output.append(" messages unchanged", style="dim")
    if result.context_window > 0:
        output.append("  │  ", style="dim")
        output.append("Window: ", style="dim")
        output.append(f"{result.context_percent}%", style="cyan")
        output.append(" used", style="dim")

    return output


def render_compact_summary_panel(summary_text: str):
    """Render the compacted summary content as a Rich panel."""
    from rich.panel import Panel
    from rich.text import Text

    content = (summary_text or "").strip()
    body = Text(content or "(empty summary)", style="dim italic")
    return Panel(
        body,
        title="Context Compacted",
        border_style="#f59e0b",
        padding=(0, 1),
    )


def build_compact_summary_renderable(
    result: CompactResult,
) -> CompactSummaryRenderable | None:
    """Build the UI summary payload for a successful compact operation."""
    if result.status != "ok" or not result.summary_text.strip():
        return None
    return CompactSummaryRenderable(result.summary_text)


async def compact_conversation(
    graph_gateway: GraphGateway,
    thread_id: str,
    target: GraphTarget,
    *,
    workspace: Workspace,
    input_tokens_hint: int | None = None,
) -> CompactResult:
    """Compact the conversation by summarizing old messages.

    Reads the graph's checkpointed state, creates a temporary
    ``SummarizationMiddleware``, generates a summary, and writes
    the compacted state back through ``GraphGateway``.

    ``input_tokens_hint`` is the real LLM input token count from the last
    ``usage_metadata`` (includes system prompt + tool schemas).  When
    provided it is used for the display values in ``CompactResult`` so the
    panel stays in sync with the status bar; the internal compact logic
    (cutoff determination) still uses message-level token counts.

    Returns a structured ``CompactResult``.
    """
    from langchain_core.messages.utils import count_tokens_approximately

    try:
        state_values = await graph_gateway.get_state_values(target, thread_id)
    except Exception as exc:
        return CompactResult("error", f"Failed to read state: {exc}")

    messages = state_values.get("messages", [])
    if not messages:
        return CompactResult(
            "noop", "Nothing to compact — no messages in conversation."
        )

    from deepagents.middleware.summarization import (
        SummarizationEvent,
        SummarizationMiddleware,
        compute_summarization_defaults,
    )

    from ..EvoScientist import _ensure_chat_model, _get_default_backend

    try:
        model = _ensure_chat_model()
    except Exception as exc:
        return CompactResult(
            "error", f"Compaction requires a working model configuration: {exc}"
        )

    backend = _get_default_backend(
        workspace, work_dir=target.run_dir or target.workspace_dir
    )
    context_window = _resolve_context_window(model)

    defaults = compute_summarization_defaults(model)
    middleware = SummarizationMiddleware(
        model=model,
        backend=backend,
        keep=defaults["keep"],
        trim_tokens_to_summarize=None,
    )

    # Rebuild effective message list accounting for prior compaction
    event = state_values.get("_summarization_event")
    effective = middleware._apply_event_to_messages(messages, event)
    effective_tokens = count_tokens_approximately(effective)

    # For display and threshold we prefer the real LLM input token count
    # (includes system prompt + tool schemas) so the panel stays in sync with
    # the status bar.  The internal compact logic (cutoff, partition, savings)
    # still uses effective_tokens (message-level) because compact only reduces
    # messages, not the constant system/tool overhead.
    display_tokens = (
        input_tokens_hint
        if input_tokens_hint is not None and input_tokens_hint > 0
        else effective_tokens
    )
    display_percent = _percent_used(display_tokens, context_window)

    if display_percent < _MANUAL_COMPACT_MIN_PERCENT:
        return CompactResult(
            "noop",
            "Conversation is below the manual compact threshold "
            f"({display_percent}% < {_MANUAL_COMPACT_MIN_PERCENT}%).",
            tokens_before=display_tokens,
            context_window=context_window,
            context_percent=display_percent,
        )

    cutoff = middleware._determine_cutoff_index(effective)
    if cutoff == 0:
        return CompactResult(
            "noop",
            f"Conversation (~{display_tokens:,} tokens) is within the retention budget.",
            tokens_before=display_tokens,
            context_window=context_window,
            context_percent=display_percent,
        )

    to_summarize, to_keep = middleware._partition_messages(effective, cutoff)

    tokens_summarized = count_tokens_approximately(to_summarize)
    tokens_kept = count_tokens_approximately(to_keep)
    tokens_before = tokens_summarized + tokens_kept

    # Skip if savings would be negligible — compacting ≤2 messages with
    # <2% of total tokens prevents the infinite 1-message-at-a-time loop
    # that occurs when the conversation sits just above the keep budget.
    _MIN_COMPACT_MESSAGES = 3
    _MIN_COMPACT_TOKEN_FRACTION = 0.02
    if (
        len(to_summarize) < _MIN_COMPACT_MESSAGES
        and tokens_summarized < tokens_before * _MIN_COMPACT_TOKEN_FRACTION
    ):
        return CompactResult(
            "noop",
            f"Nothing to compact — only {len(to_summarize)} message(s) "
            f"({tokens_summarized:,} tokens) would be summarized, "
            f"not worth the overhead.",
            tokens_before=display_tokens,
            context_window=context_window,
            context_percent=display_percent,
        )

    # Generate summary (LLM call)
    summary = await middleware._acreate_summary(to_summarize)

    # Reuse the persisted _summarization_session_id (or generate one) so
    # history keeps appending to a single file; re-persisted below.
    session_id = middleware._get_session_id(state_values)

    # Offload old messages to backend
    file_path: str | None = None
    try:
        file_path = await middleware._aoffload_to_backend(
            backend, to_summarize, session_id
        )
    except Exception:
        pass  # non-fatal — proceed without offloaded history

    from langchain_core.messages import HumanMessage

    summary_msg = cast(
        HumanMessage,
        middleware._build_new_messages_with_path(summary, file_path)[0],
    )

    # Compute token savings (message-level, used for pct calculation)
    tokens_summary = count_tokens_approximately([summary_msg])
    tokens_after = tokens_summary + tokens_kept
    pct = (
        round((tokens_before - tokens_after) / tokens_before * 100)
        if tokens_before > 0
        else 0
    )

    # Adjust display totals: preserve real overhead (system + tools) by
    # offsetting from input_tokens_hint rather than using bare message counts.
    msg_reduction = tokens_before - tokens_after  # how many message tokens saved
    display_before = display_tokens
    display_after = max(0, display_tokens - msg_reduction)
    display_after_percent = _percent_used(display_after, context_window)

    # Append savings note to summary message for model awareness
    savings_note = (
        f"\n\n{len(to_summarize)} messages were compacted "
        f"({tokens_summarized:,} → {tokens_summary:,} tokens). "
        f"Total context: {display_before:,} → {display_after:,} tokens "
        f"({pct}% decrease), "
        f"{len(to_keep)} messages unchanged."
    )
    summary_msg.content += savings_note

    state_cutoff = middleware._compute_state_cutoff(event, cutoff)

    new_event: SummarizationEvent = {
        "cutoff_index": state_cutoff,
        "summary_message": summary_msg,
        "file_path": file_path,
    }

    await graph_gateway.update_state_values(
        target,
        thread_id,
        {"_summarization_event": new_event, "_summarization_session_id": session_id},
    )

    return CompactResult(
        "ok",
        f"Compacted {len(to_summarize)} messages "
        f"({display_before:,} → {display_after:,} tokens, {pct}% decrease)",
        messages_compacted=len(to_summarize),
        messages_kept=len(to_keep),
        tokens_before=display_before,
        tokens_after=display_after,
        tokens_summarized=tokens_summarized,
        tokens_summary=tokens_summary,
        pct_decrease=pct,
        context_window=context_window,
        context_percent=display_after_percent,
        summary_text=summary,
    )


# =============================================================================
# Serve helpers
# =============================================================================

_serve_logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ServeRuntimeState:
    """Mutable serve-mode runtime shared by the poll loop and slash callbacks."""

    agent: "CompiledStateGraph"
    thread_id: str
    # The session's workspace and, after resuming a run-mode thread, its run folder.
    dirs: SessionDirs
    config: "EvoScientistConfig | None"
    runtime_gateways: "RuntimeGateways"
    async_runtime: AsyncRuntime
    resume_warning_thread_id: str | None = None
    gateway_backend: str | None = None

    def set_agent(
        self,
        agent: "CompiledStateGraph",
        channel_runtime: ChannelRuntime | None,
    ) -> None:
        self.agent = agent
        if channel_runtime is not None:
            channel_runtime.agent = agent

    def set_thread_id(
        self,
        thread_id: str,
        channel_runtime: ChannelRuntime | None,
        *,
        forget_previous_origin: bool = True,
    ) -> None:
        old_thread_id = self.thread_id
        if forget_previous_origin:
            forget_channel_origin(old_thread_id)
        self.thread_id = thread_id
        if channel_runtime is not None:
            channel_runtime.thread_id = thread_id


def _make_serve_start_new_session_cb(
    runtime_state: ServeRuntimeState,
    channel_runtime: ChannelRuntime | None = None,
):
    """Build the ``start_new_session_cb`` used by serve mode.

    ``/new`` delegates session rotation entirely to this callback: it
    does not mutate ``ctx.thread_id`` itself, it just calls
    ``ctx.ui.start_new_session()`` and expects the surface to issue a
    fresh thread id.  Without a wired callback the channel user gets
    ``ChannelCommandUI``'s fallback "restart the channel link" message
    and nothing actually rotates.  This helper generates a new thread
    id, updates the shared runtime state, and syncs the channel runtime so
    subsequent messages land on the new thread.

    The new thread works in the workspace root: after a ``/resume`` of a
    run-mode thread, the agent is reloaded and the server synced back
    there, as a resume into other folders does.
    """

    async def _cb() -> None:
        root = SessionDirs(runtime_state.dirs.workspace)
        new_tid = await runtime_state.runtime_gateways.graph_gateway.create_thread(
            GraphTarget(**root.metadata())
        )
        await _apply_serve_resume_state(
            runtime_state, channel_runtime, thread_id=new_tid, dirs=root
        )
        console.print(f"[dim][serve] New thread: {new_tid}[/dim]")

    return _cb


def _serve_resume_config(
    runtime_state: ServeRuntimeState,
    config: "EvoScientistConfig | None",
) -> "EvoScientistConfig | None":
    """Return the effective config to use for serve-mode resume sync."""
    return config if config is not None else runtime_state.config


async def _apply_serve_resume_state(
    runtime_state: ServeRuntimeState,
    channel_runtime: ChannelRuntime | None,
    *,
    thread_id: str,
    dirs: SessionDirs | None,
    config: "EvoScientistConfig | None" = None,
) -> None:
    """Adopt a resumed thread/workspace into serve-mode runtime state.

    Workspace-bound resources are rebuilt and synced before mutating the shared
    state. The agent is loaded before syncing the external server so a load
    failure cannot move the server away from the currently active session.
    When another session holds the server nothing changes; when the move fails
    otherwise, serve switches and continues without background work.
    """
    import asyncio

    new_dirs = dirs if dirs is not None and dirs != runtime_state.dirs else None
    workspace_update: tuple[SessionDirs, CompiledStateGraph] | None = None

    if new_dirs is not None:
        effective_config = _serve_resume_config(runtime_state, config)
        if effective_config is None:
            raise RuntimeError(
                "Cannot resume into a different workspace in serve mode without "
                "the effective configuration."
            )
        new_agent = await asyncio.to_thread(
            _load_agent,
            work_dir=str(new_dirs.work_dir),
            workspace=new_dirs.workspace,
            config=effective_config,
            runtime=runtime_state.async_runtime,
        )
        if not await _move_server_or_keep_session(
            effective_config,
            target=new_dirs,
            current=runtime_state.dirs,
            backend=runtime_state.gateway_backend,
        ):
            console.print(
                "[yellow]Background agent server sync failed; async "
                "subagents and EvoMemory workers may be unavailable.[/yellow]"
            )
        workspace_update = (new_dirs, new_agent)

    old_thread_id = runtime_state.thread_id
    thread_changed = thread_id != old_thread_id
    if thread_changed:
        runtime_state.set_thread_id(thread_id, channel_runtime)

    if workspace_update is not None:
        updated_dirs, updated_agent = workspace_update
        runtime_state.dirs = updated_dirs
        _set_channels_media_dir(updated_dirs.workspace.media_dir)
        runtime_state.set_agent(updated_agent, channel_runtime)


def _make_serve_handle_session_resume_cb(
    runtime_state: ServeRuntimeState,
    channel_runtime: ChannelRuntime | None = None,
    *,
    config: "EvoScientistConfig | None" = None,
):
    """Build the ChannelCommandUI resume callback for serve mode."""

    async def _cb(thread_id: str, dirs: SessionDirs | None = None) -> None:
        old_thread_id = runtime_state.thread_id
        await _apply_serve_resume_state(
            runtime_state,
            channel_runtime,
            thread_id=thread_id,
            dirs=dirs,
            config=config,
        )
        if thread_id != old_thread_id:
            runtime_state.resume_warning_thread_id = thread_id

    return _cb


def _make_serve_cmd_completed_hook(
    runtime_state: ServeRuntimeState,
    channel_runtime: ChannelRuntime | None = None,
    *,
    config: "EvoScientistConfig | None" = None,
):
    """Build the ``on_cmd_completed`` hook used by serve mode.

    Adopts ``/model`` agent swaps and ``/resume`` thread/workspace
    swaps back into ``runtime_state`` so the outer poll loop picks up
    the new handles on subsequent messages.  Also keeps
    ``channel_runtime`` in sync so the bus sees the new values.

    For ``/resume`` specifically, surface a user-visible warning via
    ``ctx.ui``: serve uses ``InMemorySaver`` (not the SQLite
    checkpointer the interactive CLI uses), so historical state for
    any persisted thread is not available — the resumed thread will
    start fresh.  Without this the ``/resume`` command appears to
    succeed silently from the channel user's POV.

    Extracted from ``_serve_process_message`` so it can be unit tested
    without spinning up the whole serve loop.
    """

    async def _hook(
        ctx: CommandContext,
        original_agent: "CompiledStateGraph",
        cmd: Command,
    ) -> None:
        if ctx.agent is not None and ctx.agent is not original_agent:
            runtime_state.set_agent(ctx.agent, channel_runtime)

        old_thread_id = runtime_state.thread_id
        resume_warning_thread_id = runtime_state.resume_warning_thread_id
        runtime_state.resume_warning_thread_id = None

        # ``/resume`` mutates ``ctx.thread_id`` and ``ctx.dirs`` directly.
        # Adopt them here (a no-op when the resume callback already has) so
        # subsequent messages run on the resumed thread.  A bare ``/resume``
        # with no argument just prints usage and leaves ``ctx.thread_id``
        # unchanged — ``thread_changed`` keeps the user-facing warning from
        # firing in that case.  Other commands leave the thread to
        # ``runtime_state``: ``/new`` rotates it through its callback, and
        # ``ctx.thread_id`` still holds the old id afterwards.
        new_tid = ctx.thread_id
        if cmd.name == "/resume":
            await _apply_serve_resume_state(
                runtime_state,
                channel_runtime,
                thread_id=new_tid,
                dirs=ctx.dirs,
                config=config,
            )

        thread_changed = new_tid != old_thread_id

        # Surface the in-memory-state limitation to the channel user
        # for ``/resume`` so the missing history isn't silent.  Flush
        # is required because ``cmd_manager.execute`` already flushed
        # the command's own output before calling this hook.
        if cmd.name == "/resume" and (
            thread_changed or resume_warning_thread_id == new_tid
        ):
            try:
                ctx.ui.append_system(
                    "Note: serve mode uses in-memory state — "
                    f"thread {new_tid[:8]} starts without prior history.",
                    style="yellow",
                )
                await ctx.ui.flush()
            except Exception:  # pragma: no cover — defensive
                pass

    return _hook


def _serve_process_message(
    msg: ChannelMessage,
    *,
    runtime_state: ServeRuntimeState,
    model: str | None,
    show_thinking: bool,
    on_cmd_completed: Callable[..., Awaitable[None]] | None = None,
    handle_session_resume_cb: Callable[..., Awaitable[None]] | None = None,
    start_new_session_cb: Callable[[], Awaitable[None]] | None = None,
    channel_runtime: ChannelRuntime | None = None,
) -> None:
    """Process a single channel message in headless serve mode.

    Headless equivalent of interactive.py's ``_process_channel_message``.
    No CLI prompt manipulation — just log lines for monitoring.

    ``runtime_state`` is shared with the outer ``serve()`` loop.
    ``on_cmd_completed`` (the agent-swap / session-adoption hook) and
    ``start_new_session_cb`` (thread rotation for ``/new``) are
    constructed once in ``serve()`` — if omitted, they're rebuilt per
    message (backward compat for existing tests).  ``/resume`` lands
    via the ``on_cmd_completed`` hook because the command mutates
    ``ctx.thread_id`` / ``ctx.workspace`` / ``ctx.run_dir`` directly.
    """
    from .channel import _bus_loop
    from .tui_runtime import run_streaming

    runtime_gateways = runtime_state.runtime_gateways

    if not _claim_or_complete_channel_request(msg):
        return

    remember_channel_origin(runtime_state.thread_id, msg)

    dirs = runtime_state.dirs

    console.print(
        f"[dim][{msg.channel_type}] {msg.sender}: {escape(msg.content[:80])}[/dim]"
    )

    # -- channel callback helpers (same pattern as interactive.py) --

    pending_channel_sends = PendingChannelSends(_bus_loop, _serve_logger)

    def _send_to_channel(coro, label: str, timeout: int = 15) -> None:
        pending_channel_sends.submit(coro, label, timeout)

    def _send_thinking(thinking: str) -> None:
        ch = msg.channel_ref
        if ch and ch.send_thinking:
            _send_to_channel(
                ch.send_thinking_message(
                    sender=msg.chat_id,
                    thinking=thinking,
                    metadata=msg.metadata,
                ),
                "Thinking",
            )

    def _send_todo(items: list[dict]) -> None:
        from ..channels.consumer import _format_todo_list

        if msg.channel_ref:
            _send_to_channel(
                msg.channel_ref.send_todo_message(
                    sender=msg.chat_id,
                    content=_format_todo_list(items),
                    metadata=msg.metadata,
                ),
                "Todo",
            )

    def _send_media(file_path: str) -> None:
        if msg.channel_ref:
            _send_to_channel(
                msg.channel_ref.send_media(
                    recipient=msg.chat_id,
                    file_path=file_path,
                    metadata=msg.metadata,
                ),
                "Media",
                timeout=30,
            )

    def _hitl_outcome(action_requests: list, human_budget_exhausted: bool):
        return channel_hitl_prompt(
            action_requests, msg, human_budget_exhausted=human_budget_exhausted
        )

    def _ask_user_prompt(ask_user_data: dict) -> dict:
        return channel_ask_user_prompt(ask_user_data, msg)

    # ---- Slash command dispatch (cmd_manager, not the agent) ----
    # Headless equivalent of the Rich CLI / TUI slash branch so channel
    # commands like ``/evoskills`` actually execute in serve mode instead
    # of being fed to the LLM as a plain prompt.  ``await_agent_ready`` is
    # None because the agent is always loaded before the serve loop polls.
    # Slash commands run on the application-owned runtime. The main thread
    # remains the signal owner while command coroutines share one stable loop.
    try:
        _slash_handled = False
        _slash_error: Exception | None = None
        try:
            async_runtime = runtime_state.async_runtime
            _slash_handled = async_runtime.run_sync(
                lambda: dispatch_channel_slash_command(
                    msg,
                    agent=runtime_state.agent,
                    thread_id=runtime_state.thread_id,
                    dirs=dirs,
                    checkpointer=None,
                    append_system=lambda t, s="dim": console.print(t, style=s),
                    start_new_session_cb=start_new_session_cb
                    or _make_serve_start_new_session_cb(
                        runtime_state,
                        channel_runtime,
                    ),
                    handle_session_resume_cb=handle_session_resume_cb
                    or _make_serve_handle_session_resume_cb(
                        runtime_state,
                        channel_runtime,
                    ),
                    on_cmd_completed=on_cmd_completed
                    or _make_serve_cmd_completed_hook(
                        runtime_state,
                        channel_runtime,
                        config=runtime_state.config,
                    ),
                    channel_runtime=channel_runtime,
                    graph_gateway=runtime_gateways.graph_gateway,
                    async_runtime=async_runtime,
                )
            )
        except Exception as exc:
            _slash_error = exc
            _serve_logger.exception("Slash dispatch failed for %s", msg.channel_type)

        if _slash_error is not None:
            _set_channel_response(msg.msg_id, f"Command error: {_slash_error}")
            console.print(
                f"[red]Slash command error: {escape(str(_slash_error))}[/red]"
            )
            return

        if _slash_handled:
            # A channel-issued /new or /resume rotates the thread inside the
            # dispatch above; re-bind the now-current thread to this channel
            # so async-notifier turns on it still forward back here.
            remember_channel_origin(runtime_state.thread_id, msg)
            console.print(f"[dim][{msg.channel_type}] Replied to {msg.sender}[/dim]")
            return

        meta = build_metadata(dirs, model)
        try:
            response = run_streaming(
                ui_backend="cli",
                agent=runtime_state.agent,
                message=msg.content,
                thread_id=runtime_state.thread_id,
                show_thinking=show_thinking,
                interactive=True,
                metadata=meta,
                configurable_extra=active_teams_configurable_extra(channel_runtime),
                on_thinking=_send_thinking,
                on_todo=_send_todo,
                on_file_write=_send_media,
                work_dir=str(dirs.work_dir),
                hitl_outcome_fn=_hitl_outcome,
                ask_user_prompt_fn=_ask_user_prompt,
                cancel_scope=_channel_message_cancel_scope(msg),
                gateway=runtime_gateways.graph_gateway,
                runtime=runtime_state.async_runtime,
            )
        except Exception as e:
            response = f"Error: {e}"
            console.print(f"[red]Serve error: {e}[/red]")

        pending_channel_sends.settle()
        _set_channel_response(msg.msg_id, response)
        console.print(f"[dim][{msg.channel_type}] Replied to {msg.sender}[/dim]")
    finally:
        _complete_channel_request(msg.msg_id)


# =============================================================================
# Serve command (headless mode)
# =============================================================================


def _serve_drain_notifications(
    *,
    runtime_state: ServeRuntimeState,
    model: str | None,
    show_thinking: bool,
    channel_runtime: ChannelRuntime | None = None,
) -> None:
    """Drain the async-task notification queue in headless serve mode.

    Mirrors the Rich CLI's ``_check_channel_queue`` notification path.
    Uses a dedicated event loop (same pattern as serve mode's slash dispatch).
    """
    import asyncio as _aio

    from .tui_runtime import run_streaming

    def _run_notification_message(text: str, notifs: list) -> None:
        """Synchronous wrapper: run the agent on the synthetic notification text."""
        # Render the per-task visual frame (matches CLI/TUI aesthetic).
        from EvoScientist.cli.async_notifier import format_notification_lines

        for line_text, line_style in format_notification_lines(notifs):
            console.print(line_text, style=line_style, markup=False)
        # Use the current folders from runtime_state (updated by /resume's
        # session-rebind callback).
        meta = build_metadata(runtime_state.dirs, model)
        tid = runtime_state.thread_id
        try:
            response = run_streaming(
                ui_backend="cli",
                agent=runtime_state.agent,
                message=text,
                thread_id=tid,
                show_thinking=show_thinking,
                interactive=True,
                metadata=meta,
                configurable_extra=active_teams_configurable_extra(channel_runtime),
                gateway=runtime_state.runtime_gateways.graph_gateway,
                runtime=runtime_state.async_runtime,
            )
        except Exception as exc:
            _serve_logger.warning("Notification agent turn failed: %s", exc)
            return
        if publish_to_channel_origin(tid, response or ""):
            # Mirror a normal channel turn's closing "Replied to" line so the
            # forwarded notification reads as terminated in the serve log.
            origin = get_channel_origin(tid)
            if origin is not None:
                console.print(
                    f"[dim][{origin.channel_type}] Replied to "
                    f"{origin.sender or origin.chat_id}[/dim]"
                )

    async def _run_notification_message_async(text: str, notifs: list) -> None:
        await _aio.to_thread(_run_notification_message, text, notifs)

    async def _read_async_tasks() -> async_notifier.AsyncTasksState:
        thread_id = runtime_state.thread_id
        if not thread_id:
            return {}
        registry = await async_notifier.read_async_tasks_from_gateway(
            runtime_state.runtime_gateways.graph_gateway,
            GraphTarget(
                local_graph=runtime_state.agent,
                **runtime_state.dirs.metadata(),
            ),
            thread_id,
        )
        return registry or {}

    async def _consume() -> None:
        await async_notifier.consume_notifications(
            run_message=_run_notification_message_async,
            read_async_tasks_state=_read_async_tasks,
            current_thread_id=runtime_state.thread_id,
        )

    try:
        runtime_state.async_runtime.run_sync(_consume)
    except Exception as exc:
        _serve_logger.warning("Notification drain failed: %s", exc)


@app.command()
def serve(
    ctx: typer.Context,
    no_thinking: bool = typer.Option(
        False, "--no-thinking", help="Disable thinking relay to channels"
    ),
    workdir: str | None = typer.Option(
        None, "--workdir", help="Override workspace directory"
    ),
    host: str | None = typer.Option(
        None,
        "--host",
        help="Interface to bind the langgraph dev backend to (default: "
        "langgraph_dev_host = 127.0.0.1). Pass 0.0.0.0 to reach it from "
        "another machine — the backend has no auth.",
    ),
    auto_approve: bool = typer.Option(
        False,
        "--auto-approve",
        help="Skip tool approval prompts for HITL actions",
    ),
    auto_mode: bool = typer.Option(
        False,
        "--auto-mode",
        help="Run unattended: skip ask_user and tool approval prompts",
    ),
    ask_user: bool = typer.Option(
        False,
        "--ask-user",
        help="Enable agent to ask clarifying questions about your research preferences",
    ),
    dangerous: bool = typer.Option(
        False,
        "--dangerous",
        help="DANGEROUS: real-filesystem access (no workspace confinement); implies --auto-approve",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug logging and channel trace output in serve mode",
    ),
):
    """Run EvoScientist in headless mode -- channels only, no interactive prompt.

    Starts all configured channels and processes messages via the agent.
    Press Ctrl+C to shut down.
    """
    from ..config import apply_config_to_env, get_effective_config

    cli_overrides = {}
    # serve starts no front-end, so only the backend bind applies here.
    if host is not None and host.strip():
        cli_overrides["langgraph_dev_host"] = host.strip()
    if auto_approve:
        cli_overrides["auto_approve"] = True
    if auto_mode:
        cli_overrides["auto_mode"] = True
        cli_overrides["auto_approve"] = True
        cli_overrides["enable_ask_user"] = False
    elif ask_user:
        cli_overrides["enable_ask_user"] = True
    if dangerous:
        cli_overrides["dangerous_mode"] = True
    if debug:
        cli_overrides["log_level"] = "DEBUG"
        cli_overrides["channel_debug_tracing"] = True
    config = get_effective_config(cli_overrides)
    async_runtime = _get_cli_async_runtime(ctx)
    if debug:
        os.environ["EVOSCIENTIST_LOG_LEVEL"] = "DEBUG"
        os.environ["EVOSCIENTIST_CHANNEL_DEBUG_TRACING"] = "true"
    apply_config_to_env(config)
    if debug:
        _configure_logging()

    # Auto-start ccproxy if any provider uses OAuth mode
    _ccproxy_proc_serve = None
    if config.anthropic_auth_mode == "oauth" or config.openai_auth_mode == "oauth":
        try:
            from ..ccproxy_manager import maybe_start_ccproxy, stop_ccproxy

            _ccproxy_proc_serve = maybe_start_ccproxy(config)
            if _ccproxy_proc_serve:
                import atexit

                atexit.register(stop_ccproxy, _ccproxy_proc_serve)
        except RuntimeError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc

    if not config.channel_enabled:
        console.print("[red]No channels configured.[/red]")
        console.print("[dim]Run [bold]evosci channel setup[/bold] first.[/dim]")
        raise typer.Exit(1)

    effective_channel_thinking = config.channel_send_thinking and (not no_thinking)
    ws_path = start_workspace_path(workdir, config.default_workdir)
    os.makedirs(ws_path, exist_ok=True)
    dirs = SessionDirs(Workspace(ws_path))
    reload_env_dirs()
    ensure_dirs()

    from ..config import GatewaySurface, resolve_gateway_backend

    gateway_backend = resolve_gateway_backend(config, GatewaySurface.SERVE)

    # Auto-start langgraph dev (after workspace resolution, so deployed
    # async sub-agents inherit the CLI's workspace via EVOSCIENTIST_WORKSPACE_DIR).
    _ensure_async_subagent_server(config, dirs=dirs, backend=gateway_backend)

    if config.dangerous_mode:
        from ._constants import DANGEROUS_BANNER_LABEL, DANGEROUS_BANNER_MESSAGE

        console.print(
            f"[bold white on red] ⚠ {DANGEROUS_BANNER_LABEL} [/bold white on red] "
            f"[bold red]{DANGEROUS_BANNER_MESSAGE}[/bold red]"
        )
    console.print("[dim]Loading agent...[/dim]")
    agent = _load_agent(workspace=dirs.workspace, config=config, runtime=async_runtime)

    from ..gateway import create_runtime_gateways_for_config

    runtime_gateways = create_runtime_gateways_for_config(
        config, backend=gateway_backend
    )
    tid = async_runtime.run_sync(
        lambda: runtime_gateways.graph_gateway.create_thread(
            GraphTarget(**dirs.metadata())
        )
    )

    # Mutable runtime shared with _serve_process_message so channel slash
    # commands can update the active agent/thread/workspace for subsequent
    # messages.
    runtime_state = ServeRuntimeState(
        agent=agent,
        thread_id=tid,
        dirs=dirs,
        config=config,
        runtime_gateways=runtime_gateways,
        async_runtime=async_runtime,
        gateway_backend=gateway_backend,
    )

    channel_runtime = ChannelRuntime(agent=agent, thread_id=tid)

    # Build the slash-dispatch callbacks once; the poll loop reuses
    # them for every inbound message.  Without this hoist each message
    # would allocate a fresh closure pair.
    _serve_on_cmd_completed = _make_serve_cmd_completed_hook(
        runtime_state, channel_runtime, config=config
    )
    _serve_handle_session_resume_cb = _make_serve_handle_session_resume_cb(
        runtime_state, channel_runtime, config=config
    )
    _serve_start_new_session_cb = _make_serve_start_new_session_cb(
        runtime_state, channel_runtime
    )

    _start_channels_bus_mode(
        config,
        agent,
        tid,
        media_dir=dirs.workspace.media_dir,
        send_thinking=effective_channel_thinking,
    )
    console.print("[green]Serve mode started (bus mode).[/green]")

    console.print(f"[dim]Thread: {tid}[/dim]")
    console.print(f"[dim]Workspace: {_shorten_path(str(dirs.workspace.root))}[/dim]")
    console.print("[dim]Press Ctrl+C to stop.[/dim]\n")

    # Explicit SIGINT/SIGTERM handlers.  Python's default SIGINT raises
    # KeyboardInterrupt in the main thread, which ought to unblock
    # ``_message_queue.get(timeout=...)`` and land in the ``except``
    # below — but edge cases (e.g. an asyncio ``set_wakeup_fd`` left
    # dangling by a nested ``asyncio.run``) can silently swallow the
    # signal.  Setting a ``threading.Event`` in addition gives us a
    # second gate that the poll loop always observes.
    import signal
    import threading

    shutdown_event = threading.Event()
    no_active_cancel_scope = object()
    active_cancel_scope: str | object | None = no_active_cancel_scope

    def _handle_shutdown(signum: int, _frame: Any) -> None:
        shutdown_event.set()
        # Cancelling the owned asyncio task is not enough when it is awaiting a
        # blocking execute call: the executor thread and its isolated process
        # group keep running until the matching stream event is set.  Request
        # scope cancellation before KeyboardInterrupt unwinds message cleanup
        # (which discards that scope).  SIGTERM also needs this to unblock the
        # synchronous serve call so the poll loop can observe shutdown_event.
        scope = active_cancel_scope
        if scope is not no_active_cancel_scope:
            from ..stream.display import request_stream_cancel

            request_stream_cancel(cast(str | None, scope))
        # Fall back to Python's default SIGINT behavior (raises
        # KeyboardInterrupt) so blocking I/O inside ``run_streaming``
        # is still interrupted.  For SIGTERM there's no default that
        # raises, so the event check below is the only gate.
        if signum == signal.SIGINT:
            signal.default_int_handler(signum, _frame)

    _orig_sigint = signal.signal(signal.SIGINT, _handle_shutdown)
    _orig_sigterm = signal.signal(signal.SIGTERM, _handle_shutdown)

    def _serve_reader_target() -> GraphTarget:
        return GraphTarget(
            local_graph=runtime_state.agent,
            **runtime_state.dirs.metadata(),
        )

    async def _serve_enqueue_completions() -> None:
        # On turn close, read async_tasks off thread state and enqueue any
        # completions not yet surfaced; the idle drain below injects them.
        thread_id = runtime_state.thread_id
        if not thread_id:
            return
        await async_notifier.enqueue_completions_from_state(
            runtime_state.runtime_gateways.graph_gateway,
            _serve_reader_target(),
            thread_id,
        )
        await async_notifier.enqueue_bg_process_completions_from_state(
            runtime_state.runtime_gateways.graph_gateway,
            _serve_reader_target(),
            thread_id,
        )

    async def _serve_enqueue_completions_idle() -> None:
        # Throttled idle-tick reader: surfaces async-task + bg-process completions
        # while no channel message is being processed, without a state read on
        # every poll tick.
        thread_id = runtime_state.thread_id
        if not thread_id:
            return
        gateway = runtime_state.runtime_gateways.graph_gateway
        target = _serve_reader_target()
        await async_notifier.enqueue_completions_from_state_throttled(
            gateway, target, thread_id
        )
        await async_notifier.enqueue_bg_process_completions_from_state_throttled(
            gateway, target, thread_id
        )

    try:
        while not shutdown_event.is_set():
            try:
                msg = _message_queue.get(timeout=0.5)
            except queue.Empty:
                msg = None
            if shutdown_event.is_set():
                break
            if msg is not None:
                active_cancel_scope = _channel_message_cancel_scope(msg)
                try:
                    _serve_process_message(
                        msg,
                        runtime_state=runtime_state,
                        model=config.model,
                        show_thinking=effective_channel_thinking,
                        on_cmd_completed=_serve_on_cmd_completed,
                        handle_session_resume_cb=_serve_handle_session_resume_cb,
                        start_new_session_cb=_serve_start_new_session_cb,
                        channel_runtime=channel_runtime,
                    )
                except KeyboardInterrupt:
                    shutdown_event.set()
                    break
                finally:
                    active_cancel_scope = no_active_cancel_scope
                runtime_state.async_runtime.run_sync(_serve_enqueue_completions)

            # Detect async-task completions from state (throttled) so they
            # surface while idle, then poll the notification queue.
            runtime_state.async_runtime.run_sync(_serve_enqueue_completions_idle)
            if async_notifier.has_pending_notifications(runtime_state.thread_id):
                # Notification turns use the default stream cancellation scope.
                active_cancel_scope = None
                try:
                    _serve_drain_notifications(
                        runtime_state=runtime_state,
                        model=config.model,
                        show_thinking=effective_channel_thinking,
                        channel_runtime=channel_runtime,
                    )
                finally:
                    active_cancel_scope = no_active_cancel_scope
                # Re-arm the idle reader unconditionally after a notification
                # turn too: it may have launched a chained task (analysis
                # finished -> start writing) that would otherwise sit in state
                # with the reader disarmed until an inbound channel message.
                runtime_state.async_runtime.run_sync(_serve_enqueue_completions)
    except KeyboardInterrupt:
        shutdown_event.set()
    finally:
        signal.signal(signal.SIGINT, _orig_sigint)
        signal.signal(signal.SIGTERM, _orig_sigterm)
        console.print("\n[dim]Shutting down...[/dim]")
        _channels_stop(runtime=channel_runtime)
        console.print("[dim]Stopped.[/dim]")


# =============================================================================
# Config commands
# =============================================================================


@config_app.callback(invoke_without_command=True)
def config_callback(ctx: typer.Context):
    """Configuration management commands"""
    if ctx.invoked_subcommand is None:
        config_list()


@config_app.command("list")
def config_list():
    """List all configuration values"""
    from ..config import get_config_path, list_config

    config_data = list_config()

    table = Table(title="EvoScientist Configuration", show_header=True)
    table.add_column("Setting", style="cyan")
    table.add_column("Value")

    # Mask API keys
    def format_value(key: str, value: Any) -> str:
        if "api_key" in key and value:
            return "***" + str(value)[-4:] if len(str(value)) > 4 else "***"
        if value == "":
            return "[dim](not set)[/dim]"
        return str(value)

    for key, value in config_data.items():
        table.add_row(key, format_value(key, value))

    console.print(table)
    console.print(f"\n[dim]Config file: {get_config_path()}[/dim]")


@config_app.command("get")
def config_get(key: str = typer.Argument(..., help="Configuration key to get")):
    """Get a single configuration value"""
    from ..config import get_config_value

    value = get_config_value(key)
    if value is None:
        console.print(f"[red]Unknown key: {key}[/red]")
        raise typer.Exit(1)

    # Mask API keys
    if "api_key" in key and value:
        display_value = "***" + str(value)[-4:] if len(str(value)) > 4 else "***"
    elif value == "":
        display_value = "(not set)"
    else:
        display_value = str(value)

    console.print(f"[cyan]{key}[/cyan]: {display_value}")


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Configuration key to set"),
    value: str = typer.Argument(..., help="New value"),
):
    """Set a single configuration value"""
    from ..config import set_config_value

    if set_config_value(key, value):
        console.print(f"[green]Set {escape(key)}[/green]")
    else:
        console.print(f"[red]Could not set {escape(key)}: invalid key or value[/red]")
        raise typer.Exit(1)


@config_app.command("reset")
def config_reset(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Reset configuration to defaults"""
    from ..config import get_config_path, reset_config

    config_path = get_config_path()

    if not config_path.exists():
        console.print("[yellow]No config file to reset.[/yellow]")
        return

    if not yes:
        confirm = typer.confirm("Reset configuration to defaults?")
        if not confirm:
            console.print("[dim]Cancelled.[/dim]")
            return

    reset_config()
    console.print("[green]Configuration reset to defaults.[/green]")


@config_app.command("path")
def config_path():
    """Show the configuration file path"""
    from ..config import get_config_path

    path = get_config_path()
    exists = path.exists()
    status = "[green]exists[/green]" if exists else "[dim]not created yet[/dim]"
    console.print(f"{path} ({status})")


# =============================================================================
# MCP commands
# =============================================================================


@mcp_app.callback(invoke_without_command=True)
def mcp_callback(ctx: typer.Context):
    """MCP server management commands"""
    if ctx.invoked_subcommand is None:
        mcp_list()


@mcp_app.command("list")
def mcp_list():
    """List configured MCP servers"""
    _mcp_list_servers()


@mcp_app.command("config")
def mcp_config(
    name: str | None = typer.Argument(None, help="Server name (omit to show all)"),
):
    """Show detailed configuration for MCP servers

    \b
    Examples:
      evosci mcp config             # Show all servers in detail
      evosci mcp config filesystem  # Show one server
    """
    status = _show_mcp_config(name or "", show_blank_line=False)
    if status == "empty":
        console.print(
            "[dim]Add one with:[/dim] EvoSci mcp add <name> <transport> <command-or-url> [args...]"
        )
        return
    if status == "missing":
        raise typer.Exit(1)


@mcp_app.command("add")
def mcp_add(
    name: Annotated[str, typer.Argument(help="Server name")],
    target: Annotated[str, typer.Argument(help="Command (stdio) or URL (http/sse)")],
    args: Annotated[
        list[str] | None, typer.Argument(help="Extra args for stdio command")
    ] = None,
    transport: Annotated[
        str | None,
        typer.Option("--transport", "-T", help="Transport type (default: auto-detect)"),
    ] = None,
    tools: Annotated[
        str | None,
        typer.Option(
            "--tools",
            "-t",
            help="Comma-separated tool allowlist (supports wildcards: *_exa, read_*)",
        ),
    ] = None,
    expose_to: Annotated[
        str | None,
        typer.Option("--expose-to", "-e", help="Comma-separated target agents"),
    ] = None,
    header: Annotated[
        list[str] | None,
        typer.Option("--header", "-H", help="HTTP header as Key:Value (repeatable)"),
    ] = None,
    env: Annotated[
        list[str] | None,
        typer.Option("--env", help="Env var as KEY=VALUE for stdio (repeatable)"),
    ] = None,
    env_ref: Annotated[
        list[str] | None,
        typer.Option(
            "--env-ref", help="Env var name as ${NAME} runtime ref (repeatable)"
        ),
    ] = None,
):
    """Add an MCP server to user config

    \b
    Transport is auto-detected: URLs default to http, commands default to stdio.

    \b
    Examples:
      evosci mcp add sequential-thinking npx -- -y @modelcontextprotocol/server-sequential-thinking
      evosci mcp add docs-langchain https://docs.langchain.com/mcp
      evosci mcp add my-sse https://example.com/sse --transport sse -e research-agent
      evosci mcp add brave-search npx --env-ref BRAVE_API_KEY -- -y @modelcontextprotocol/server-brave-search
    """
    from ..mcp import build_mcp_add_kwargs

    # Merge env and env_ref into a single dict
    env_dict: dict[str, str] = {}
    for e in env or []:
        if "=" in e:
            k, v = e.split("=", 1)
            env_dict[k.strip()] = v.strip()
    for ref in env_ref or []:
        env_dict[ref] = "${" + ref + "}"

    kwargs = build_mcp_add_kwargs(
        name=name,
        target=target,
        extra_args=list(args) if args else None,
        transport=transport,
        tools=[t.strip() for t in tools.split(",") if t.strip()] if tools else None,
        expose_to=[a.strip() for a in expose_to.split(",") if a.strip()]
        if expose_to
        else None,
        headers={
            k.strip(): v.strip()
            for h in (header or [])
            for k, v in [h.split(":", 1)]
            if ":" in h
        }
        or None,
        env=env_dict or None,
    )

    if not _mcp_add_server_from_kwargs(kwargs, show_reload_hint=False):
        raise typer.Exit(1)


@mcp_app.command("edit")
def mcp_edit(
    name: Annotated[str, typer.Argument(help="Server name to edit")],
    transport: Annotated[
        str | None, typer.Option("--transport", help="New transport type")
    ] = None,
    command: Annotated[
        str | None, typer.Option("--command", help="New command (stdio)")
    ] = None,
    url: Annotated[
        str | None, typer.Option("--url", help="New URL (http/sse/websocket)")
    ] = None,
    tools: Annotated[
        str | None,
        typer.Option(
            "--tools",
            "-t",
            help="Comma-separated tool allowlist, supports wildcards ('none' to clear)",
        ),
    ] = None,
    expose_to: Annotated[
        str | None,
        typer.Option(
            "--expose-to",
            "-e",
            help="Comma-separated target agents ('none' to clear)",
        ),
    ] = None,
    header: Annotated[
        list[str] | None,
        typer.Option("--header", "-H", help="HTTP header as Key:Value (repeatable)"),
    ] = None,
    env: Annotated[
        list[str] | None,
        typer.Option("--env", help="Env var as KEY=VALUE for stdio (repeatable)"),
    ] = None,
):
    """Edit an existing MCP server in user config

    \b
    Examples:
      evosci mcp edit filesystem --expose-to main,code-agent
      evosci mcp edit filesystem -t read_file,write_file
      evosci mcp edit my-api --url http://new-host:9090/mcp
      evosci mcp edit my-api --tools none
    """
    from ..mcp import build_mcp_edit_fields

    fields = build_mcp_edit_fields(
        transport=transport,
        command=command,
        url=url,
        tools=tools,
        expose_to=expose_to,
        headers=header,
        env=env,
    )

    if not _mcp_edit_server_fields(name, fields, show_reload_hint=False):
        raise typer.Exit(1)


@mcp_app.command("remove")
def mcp_remove(
    name: str = typer.Argument(..., help="Server name to remove"),
):
    """Remove an MCP server from user config"""
    if not _mcp_remove_server(name, show_reload_hint=False):
        raise typer.Exit(1)


@mcp_app.command("install")
def mcp_install(
    source: Annotated[
        str | None, typer.Argument(help="Server name or tag filter")
    ] = None,
):
    """Browse and install MCP servers from the registry and marketplace

    \b
    Examples:
      evosci mcp install                       # Interactive browser
      evosci mcp install search                # Filter by 'search' tag
      evosci mcp install sequential-thinking   # Install by name
    """
    from .mcp_install_cmd import _cmd_install_mcp

    _cmd_install_mcp(source or "")


# =============================================================================
# Sessions commands — read-only diagnostics for ~/.evoscientist/sessions.db
# =============================================================================


def _format_bytes(n: int) -> str:
    """Render a byte count as a human-readable string (KB / MB / GB)."""
    if n < 1024:
        return f"{n} B"
    units = ["KB", "MB", "GB", "TB"]
    size = float(n) / 1024.0
    for unit in units:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


@sessions_app.callback(invoke_without_command=True)
def sessions_callback(ctx: typer.Context):
    """Inspect and manage the sessions DB.

    Running ``EvoSci sessions`` with no subcommand defaults to ``stats``
    so the bare command is informative rather than silent.
    """
    if ctx.invoked_subcommand is None:
        sessions_stats(ctx)


@sessions_app.command("stats")
def sessions_stats(ctx: typer.Context):
    """Show DB size, thread count, total checkpoints, top heaviest threads."""
    from ..sessions import db_stats

    runtime = _get_cli_async_runtime(ctx)
    stats = runtime.run_sync(db_stats)

    table = Table(title="EvoScientist sessions DB", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value")
    table.add_row("Path", stats["db_path"])
    table.add_row("Size", _format_bytes(int(stats["size_bytes"])))
    table.add_row("Threads", str(stats["thread_count"]))
    table.add_row("Checkpoints", str(stats["checkpoint_count"]))
    table.add_row("Writes", str(stats["write_count"]))
    console.print(table)

    if stats["top_threads"]:
        top = Table(title="Heaviest threads (checkpoints per thread)")
        top.add_column("thread_id", style="yellow")
        top.add_column("checkpoints", justify="right")
        for row in stats["top_threads"]:
            top.add_row(str(row["thread_id"]), str(row["count"]))
        console.print(top)


# =============================================================================
# Main callback (default behavior)
# =============================================================================


def _version_callback(value: bool):
    if value:
        # ``--version`` is eager, so it runs before ``--json`` is parsed (Click
        # runs eager callbacks in command-line order); read the raw arguments
        # so both flag orders print the same JSON.
        if "--json" in sys.argv[1:]:
            from ..setup.protocol import PROTOCOL
            from ..setup.webui import compat_range
            from ..stream.json_sink import redirect_console_to_stderr

            # stdout carries only the JSON line; a warning (e.g. the range
            # override) goes to stderr.
            redirect_console_to_stderr()

            typer.echo(
                json.dumps(
                    {
                        "version": _pkg_version("EvoScientist"),
                        "protocol": PROTOCOL,
                        "webui_compat": compat_range(),
                    }
                )
            )
        else:
            typer.echo(f"EvoScientist {_pkg_version('EvoScientist')}")
        raise typer.Exit()


def _is_fresh_interactive_session(prompt: str | None, thread_id: str | None) -> bool:
    """True for a brand-new interactive session — no one-shot ``-p`` prompt and
    no ``--resume`` / ``--thread-id`` to continue.

    This is the only case where a WebUI-configured ``EvoSci`` opens the browser
    app: a one-shot or a resume has a concrete conversation to render in the
    terminal, so it falls back to the Rich CLI instead.
    """
    return not prompt and not thread_id


def _resolve_stream_json_auto_mode(auto_mode: bool | None, output_format: str) -> bool:
    """Resolve the effective ``--auto-mode`` for a single-shot run.

    stream-json is headless, so auto-mode defaults on there when the caller did
    not pass the flag (``None``); an explicit ``--auto-mode`` / ``--no-auto-mode``
    always wins, and non-stream-json runs keep the historical off-by-default.
    """
    if auto_mode is not None:
        return auto_mode
    return output_format == "stream-json"


@app.callback(invoke_without_command=True)
def _main_callback(
    ctx: typer.Context,
    version: bool | None = typer.Option(
        None,
        "-V",
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
    mode: str | None = typer.Option(
        None,
        "-m",
        "--mode",
        help="Workspace mode: 'daemon' (persistent, default) or 'run' (isolated per-session)",
    ),
    name: str | None = typer.Option(
        None,
        "-n",
        "--name",
        help="Name for this run (used as directory name instead of timestamp; requires --mode run)",
    ),
    prompt: str | None = typer.Option(
        None, "-p", "--prompt", help="Query to execute (single-shot mode)"
    ),
    thread_id: str | None = typer.Option(
        None,
        "--resume",
        "--thread-id",
        help="Thread ID (or prefix) to resume a previous session.",
    ),
    workdir: str | None = typer.Option(
        None, "--workdir", help="Override workspace directory for this session"
    ),
    use_cwd: bool = typer.Option(
        False, "--use-cwd", help="Use current working directory as workspace"
    ),
    no_thinking: bool = typer.Option(
        False, "--no-thinking", help="Disable thinking display"
    ),
    auto_approve: bool = typer.Option(
        False,
        "--auto-approve",
        help="Skip tool approval prompts for HITL actions",
    ),
    auto_mode: bool | None = typer.Option(
        None,
        "--auto-mode/--no-auto-mode",
        help="Run unattended: skip ask_user and tool approval prompts "
        "(default: on when --output-format stream-json)",
    ),
    ask_user: bool = typer.Option(
        False,
        "--ask-user",
        help="Enable agent to ask clarifying questions about your research preferences",
    ),
    dangerous: bool = typer.Option(
        False,
        "--dangerous",
        help="DANGEROUS: real-filesystem access (no workspace confinement); implies --auto-approve",
    ),
    auth_mode: str | None = typer.Option(
        None,
        "--auth-mode",
        help="Auth mode for Anthropic/OpenAI: api_key (default) or oauth (ccproxy).",
    ),
    ui: str | None = typer.Option(
        None,
        "--ui",
        help="UI backend: tui (default), cli, or webui.",
    ),
    host: str | None = typer.Option(
        None,
        "--host",
        help="Interface to bind servers to (default: 127.0.0.1 for both). "
        "Sets langgraph_dev_host — the backend shared by every UI mode — and "
        "webui_host (WebUI mode only). Applies to the default entry; the "
        "serve and deploy subcommands take their own --host. Pass 0.0.0.0 to "
        "reach both from another machine (the backend has no auth).",
    ),
    output_format: str | None = typer.Option(
        None,
        "--output-format",
        help=(
            "Output format for single-shot (-p) mode: 'text' (default) or "
            "'stream-json' (line-delimited JSON events to stdout)."
        ),
    ),
    version_json: bool = typer.Option(
        False,
        "--json",
        hidden=True,
        help="With --version: print the version, protocol and WebUI range as JSON.",
    ),
):
    """EvoScientist Agent - AI-powered research & code execution CLI"""
    if version_json:
        # ``--version --json`` exits in the version callback before this runs.
        typer.echo("--json is only valid with --version.", err=True)
        raise typer.Exit(2)
    # If a subcommand was invoked, don't run the default behavior
    if ctx.invoked_subcommand is not None:
        return

    async_runtime = _get_cli_async_runtime(ctx)

    # Load and apply configuration
    from ..config import apply_config_to_env, get_effective_config

    # Resolve the output format first. In stream-json mode stdout must carry
    # only JSONL, so establish the mode and redirect the console to stderr
    # BEFORE anything below (ccproxy startup, validation) can print a
    # human-readable line to stdout.
    effective_output_format = (output_format or "text").lower()
    if effective_output_format not in ("text", "stream-json"):
        raise typer.BadParameter("--output-format must be 'text' or 'stream-json'")
    if effective_output_format == "stream-json":
        if not prompt:
            raise typer.BadParameter(
                "--output-format stream-json requires -p/--prompt (single-shot mode)"
            )
        from ..stream.json_sink import redirect_console_to_stderr

        redirect_console_to_stderr()

    # stream-json is headless, so auto-mode defaults on (auto-handle approval and
    # ask_user gates) unless the caller explicitly passed --no-auto-mode. Without
    # it the run would stall at the first gate and end without doing the work;
    # --no-auto-mode is the (experimental) opt-in to receiving interrupt/ask_user
    # events and driving resume yourself.
    effective_auto_mode = _resolve_stream_json_auto_mode(
        auto_mode, effective_output_format
    )
    if effective_output_format == "stream-json" and auto_mode is False:
        console.print(
            "[yellow]--no-auto-mode with stream-json is experimental: an "
            "interrupt/ask_user event ends the run early and is not yet "
            "resumable.[/yellow]"
        )

    # Build CLI overrides dict
    cli_overrides = {}
    if mode:
        cli_overrides["default_mode"] = mode
    if workdir:
        cli_overrides["default_workdir"] = workdir
    if no_thinking:
        cli_overrides["show_thinking"] = False
    if ui:
        cli_overrides["ui_backend"] = ui
    if host is not None and host.strip():
        # One flag drives both servers; the backend applies in EVERY UI mode
        # (auto-started for tui/cli/serve too), webui_host only in WebUI mode.
        cli_overrides["webui_host"] = host.strip()
        cli_overrides["langgraph_dev_host"] = host.strip()
    if auto_approve:
        cli_overrides["auto_approve"] = True
    if effective_auto_mode:
        cli_overrides["auto_mode"] = True
        cli_overrides["auto_approve"] = True
        cli_overrides["enable_ask_user"] = False
    else:
        # An explicit --no-auto-mode (auto_mode is False, not None) must win over
        # a config file that enables auto-mode; without this the resolved-off
        # value writes nothing and silently falls back to the config default.
        if auto_mode is False:
            cli_overrides["auto_mode"] = False
        if ask_user:
            cli_overrides["enable_ask_user"] = True
    if dangerous:
        cli_overrides["dangerous_mode"] = True
    if auth_mode:
        if auth_mode not in ("api_key", "oauth"):
            raise typer.BadParameter("--auth-mode must be 'api_key' or 'oauth'")
        cli_overrides["anthropic_auth_mode"] = auth_mode
        cli_overrides["openai_auth_mode"] = auth_mode

    config = get_effective_config(cli_overrides)
    apply_config_to_env(config)

    # Auto-start ccproxy if any provider uses OAuth mode
    _ccproxy_proc = None
    if config.anthropic_auth_mode == "oauth" or config.openai_auth_mode == "oauth":
        try:
            from ..ccproxy_manager import maybe_start_ccproxy, stop_ccproxy

            _ccproxy_proc = maybe_start_ccproxy(config)
            if _ccproxy_proc:
                import atexit

                atexit.register(stop_ccproxy, _ccproxy_proc)
        except RuntimeError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc

    show_thinking = config.show_thinking if not no_thinking else False
    effective_channel_thinking = config.channel_send_thinking and (not no_thinking)

    # Validate mutually exclusive options
    if workdir and use_cwd:
        raise typer.BadParameter("Use either --workdir or --use-cwd, not both.")

    if mode and (workdir or use_cwd):
        raise typer.BadParameter(
            "--mode cannot be combined with --workdir or --use-cwd"
        )

    if mode and mode not in ("run", "daemon"):
        raise typer.BadParameter("--mode must be 'run' or 'daemon'")
    if ui and ui.lower() not in ("cli", "tui", "webui"):
        raise typer.BadParameter("--ui must be 'tui', 'cli', or 'webui'")

    # --name only makes sense in run mode
    if name and not (
        mode == "run"
        or (not mode and not workdir and not use_cwd and config.default_mode == "run")
    ):
        raise typer.BadParameter("--name can only be used with --mode run")

    # Sanitize run name: allow alphanumeric, hyphens, underscores
    if name:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise typer.BadParameter(
                "--name may only contain letters, digits, hyphens, and underscores"
            )

    # Resolve the session's workspace and the folder the agent works in.
    # Priority: --use-cwd / --workdir > default_workdir > cwd. ``--mode``
    # (or ``default_mode``) only decides whether the agent works in the
    # workspace root (daemon) or in a fresh ``runs/<name>`` folder (run).
    effective_mode: str | None = None  # None means explicit --workdir/--use-cwd
    if use_cwd:
        workspace_root = start_workspace_path()
    elif workdir:
        workspace_root = start_workspace_path(workdir)
        os.makedirs(workspace_root, exist_ok=True)
    else:
        workspace_root = start_workspace_path(default_workdir=config.default_workdir)
        effective_mode = mode or config.default_mode
    workspace = Workspace(workspace_root)

    # The project .env was merged into os.environ by get_effective_config().
    reload_env_dirs()
    # Ensure memory and skills subdirs exist in workspace
    ensure_dirs()

    # WebUI mode: instead of the in-terminal CLI/TUI, run a deploy-style
    # langgraph server (full MCP + async) + the locally installed
    # @evoscientist/webui front-end in THIS terminal, then block. Reuses start_langgraph_dev
    # but leaves `EvoSci deploy` untouched (it stays a clean server for external
    # UIs / SDK clients).
    #
    # The browser app is only launched for a FRESH interactive session. With
    # `-p` (one-shot) or `--resume`/`--thread-id` (continue a specific
    # conversation), there is concrete terminal output to render, so fall back
    # to the Rich CLI instead of opening the browser UI.
    from .tui_runtime import normalize_ui_backend

    if normalize_ui_backend(config.ui_backend) == "webui":
        if _is_fresh_interactive_session(prompt, thread_id):
            from ..deploy.webui import run_webui

            # Run mode is a CLI and TUI feature: the WebUI works in the workspace root.
            if effective_mode == "run":
                console.print(
                    "[dim]The WebUI works in the workspace root; --mode run "
                    "applies to the CLI and TUI.[/dim]"
                )
            run_webui(config, workspace_dir=str(workspace.root))
            return
        config.ui_backend = "cli"

    # ``--mode=run`` works in a fresh ``runs/<name>`` folder of the workspace;
    # ``/new`` then moves to another one.
    run_dir = _create_run_dir(workspace, name) if effective_mode == "run" else None
    dirs = SessionDirs(workspace, run_dir)

    # Resolve the gateway backend for whichever surface this callback launches:
    # single-shot when a prompt is given, else the interactive CLI / TUI (the
    # same value each inner entry re-resolves for its own factory call). Drives
    # both the server start and the single-shot factory below.
    from ..config import GatewaySurface, resolve_gateway_backend

    if prompt:
        _gateway_surface = GatewaySurface.SINGLE_SHOT
    elif normalize_ui_backend(config.ui_backend) == "tui":
        _gateway_surface = GatewaySurface.TUI
    else:
        _gateway_surface = GatewaySurface.INTERACTIVE
    gateway_backend = resolve_gateway_backend(config, _gateway_surface)

    # Auto-start langgraph dev (after workspace resolution, so deployed
    # async sub-agents inherit the CLI's workspace via EVOSCIENTIST_WORKSPACE_DIR).
    # A one-shot resume starts it for the thread's own folders instead, unless
    # the server gateway backend needs a running server to be built.
    prespawn = not (prompt and thread_id) or gateway_backend == "langgraph_server"
    if prespawn:
        _ensure_async_subagent_server(config, dirs=dirs, backend=gateway_backend)

    if prompt:
        # Single-shot mode: wrap in persistent checkpointer
        import asyncio

        from ..gateway import create_runtime_gateways_for_config
        from ..sessions import get_checkpointer
        from ..stream.json_sink import stream_json
        from .interactive import cmd_run
        from .resume_hint import print_resume_hint

        runtime_gateways = create_runtime_gateways_for_config(
            config, backend=gateway_backend
        )
        graph_gateway = runtime_gateways.graph_gateway

        async def _single_shot():
            async with get_checkpointer() as checkpointer:
                # Resolve resume target first so a bad --resume/--thread-id
                # exits before the slow _load_agent() provider setup.
                if thread_id:
                    resolution = await graph_gateway.resolve_thread(thread_id)
                    if resolution.thread_id:
                        tid = resolution.thread_id
                        session_dirs = await _restore_thread_dirs(
                            tid,
                            dirs=dirs,
                            served=dirs if prespawn else None,
                            graph_gateway=graph_gateway,
                            config=config,
                            backend=gateway_backend,
                        )
                    elif resolution.matches:
                        console.print(
                            f"[yellow]Ambiguous thread ID '{escape(thread_id)}'. Matches:[/yellow]"
                        )
                        for s in resolution.matches:
                            console.print(f"  [cyan]{escape(s)}[/cyan]")
                        raise typer.Exit(1)
                    else:
                        console.print(
                            f"[red]Thread '{escape(thread_id)}' not found.[/red]"
                        )
                        raise typer.Exit(1)
                else:
                    tid = await graph_gateway.create_thread()
                    session_dirs = dirs
                console.print("[dim]Loading agent...[/dim]")
                agent = await asyncio.to_thread(
                    _load_agent,
                    work_dir=str(session_dirs.work_dir),
                    workspace=session_dirs.workspace,
                    checkpointer=checkpointer,
                    config=config,
                    runtime=async_runtime,
                )
                try:
                    if effective_output_format == "stream-json":
                        # Headless JSONL path: drive the sink through the gateway
                        # directly. We are already inside the async single-shot
                        # loop, so this is a plain await — no nested-loop juggling,
                        # and the gateway seam keeps it execution-backend agnostic.
                        request = RunRequest(
                            message=prompt,
                            thread_id=tid,
                            metadata=build_metadata(session_dirs, config.model),
                            target=GraphTarget(
                                local_graph=agent, **session_dirs.metadata()
                            ),
                        )
                        try:
                            await stream_json(graph_gateway, request)
                        except Exception as exc:
                            # stream_events already emitted a terminal `error`
                            # event onto the JSON stream before re-raising; exit
                            # cleanly so the stream ends with that event instead
                            # of a raw traceback.
                            raise typer.Exit(1) from exc
                        finally:
                            # Let post-run memory workers persist before exit,
                            # matching the text path (cmd_run does this itself).
                            _wait_for_memory_workers()
                    else:
                        stream_worker = asyncio.create_task(
                            asyncio.to_thread(
                                cmd_run,
                                agent,
                                prompt,
                                thread_id=tid,
                                show_thinking=show_thinking,
                                dirs=session_dirs,
                                model=config.model,
                                ui_backend=config.ui_backend,
                                runtime_gateways=runtime_gateways,
                                async_runtime=async_runtime,
                            )
                        )
                        try:
                            await asyncio.shield(stream_worker)
                        except asyncio.CancelledError:
                            from ..stream.display import request_stream_cancel
                            from .tui_runtime import settle_cancelled_worker

                            await settle_cancelled_worker(
                                stream_worker,
                                on_cancel=request_stream_cancel,
                            )
                            raise
                finally:
                    # Model failures can bypass middleware ``after_agent``
                    # hooks. Close any remaining QuickJS workers while this
                    # event loop is still available; their synchronous GC
                    # fallback can deadlock during interpreter shutdown.
                    from ..middleware.code_interpreter import (
                        aclose_code_interpreters,
                    )

                    await aclose_code_interpreters()
                    try:
                        print_resume_hint(tid, console=console)
                    except Exception:
                        pass

        async_runtime.run_sync(_single_shot)
    else:
        from .interactive import cmd_interactive

        # Interactive mode (default) — checkpointer managed inside cmd_interactive
        cmd_interactive(
            show_thinking=show_thinking,
            channel_send_thinking=effective_channel_thinking,
            dirs=dirs,
            mode=effective_mode,
            model=config.model,
            provider=config.provider,
            run_name=name,
            thread_id=thread_id,
            ui_backend=config.ui_backend,
            config=config,
            async_runtime=async_runtime,
        )


def _configure_logging():
    """Configure logging with warning symbols for better visibility."""
    from rich.logging import RichHandler

    from ..config import get_effective_config

    def _resolve_log_level() -> int:
        """Resolve the root log level from config/env with a safe fallback."""
        try:
            raw = (get_effective_config().log_level or "").strip().upper()
        except Exception:
            raw = ""
        if raw == "WARN":
            raw = "WARNING"
        return getattr(logging, raw, logging.WARNING)

    resolved_level = _resolve_log_level()
    verbose_logging = resolved_level <= logging.DEBUG

    class DimWarningHandler(RichHandler):
        """Custom handler that renders warnings in dim style."""

        def emit(self, record: logging.LogRecord) -> None:
            if record.levelno == logging.WARNING:
                # Use Rich console to print dim warning
                msg = record.getMessage()
                console.print(
                    f"[dim yellow]\u26a0\ufe0f  Warning:[/dim yellow] [dim]{escape(msg)}[/dim]"
                )
            else:
                super().emit(record)

    # Configure root logger to use our handler for WARNING and above
    handler = DimWarningHandler(
        console=console,
        show_time=verbose_logging,
        show_path=verbose_logging,
        show_level=verbose_logging,
    )
    handler.setLevel(resolved_level)

    # Apply to root logger (catches all loggers including deepagents)
    root_logger = logging.getLogger()
    # Remove existing handlers to avoid duplicate output
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    root_logger.addHandler(handler)
    root_logger.setLevel(resolved_level)

    # Suppress noisy schema warnings from langchain_google_genai
    # (e.g. "Key '$schema' is not supported in schema, ignoring")
    logging.getLogger("langchain_google_genai._function_utils").setLevel(logging.ERROR)
