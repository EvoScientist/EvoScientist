"""Interactive CLI mode and single-shot execution."""

import asyncio
import logging
import queue
import random
import signal
import sys
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import TYPE_CHECKING, Any

import typer  # type: ignore[import-untyped]
from prompt_toolkit import PromptSession  # type: ignore[import-untyped]
from prompt_toolkit.auto_suggest import (
    AutoSuggestFromHistory,  # type: ignore[import-untyped]
)
from prompt_toolkit.completion import (  # type: ignore[import-untyped]
    Completer,
    Completion,
)
from prompt_toolkit.formatted_text import HTML  # type: ignore[import-untyped]
from prompt_toolkit.history import FileHistory  # type: ignore[import-untyped]
from prompt_toolkit.key_binding import KeyBindings  # type: ignore[import-untyped]
from prompt_toolkit.patch_stdout import patch_stdout  # type: ignore[import-untyped]
from prompt_toolkit.shortcuts import CompleteStyle  # type: ignore[import-untyped]
from prompt_toolkit.styles import Style as PtStyle  # type: ignore[import-untyped]
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

import EvoScientist.cli.channel as _ch_mod

from ..commands.base import Command, CommandContext
from ..commands.manager import manager as cmd_manager
from ..gateway import (
    GraphGateway,
    GraphTarget,
    RuntimeGateways,
    create_runtime_gateways_for_config,
)
from ..paths import SessionDirs
from ..sessions import get_checkpointer, short_thread_id
from ..stream.console import console
from ..stream.display import _fix_markdown_heading_spacing
from . import async_notifier
from ._agent_loader import BackgroundAgentLoader, MCPProgressTracker
from ._constants import (
    DANGEROUS_BANNER_LABEL,
    DANGEROUS_BANNER_MESSAGE,
    LOGO_GRADIENT,
    LOGO_LINES,
    WELCOME_SLOGANS,
    build_metadata,
)
from .agent import (
    _create_run_dir,
    _load_agent,
    _shorten_path,
)
from .channel import (
    ChannelMessage,
    _auto_start_channel,
    _channels_is_running,
    _message_queue,
    _set_channel_response,
    dispatch_channel_slash_command,
)
from .channel_sends import PendingChannelSends
from .file_mentions import complete_file_mention, resolve_file_mentions
from .rich_command_ui import RichCLICommandUI
from .status_bar import (
    SPINNER_FRAMES,
    STATUS_BAD,
    STATUS_BAR_BG,
    STATUS_CRITICAL,
    STATUS_DIM,
    STATUS_GOOD,
    STATUS_STRONG,
    STATUS_TEXT,
    STATUS_WARN,
    apply_assistant_text_to_snapshot,
    apply_user_text_to_snapshot,
    build_session_status_snapshot,
    build_status_fragments,
    build_status_text,
    make_empty_status_snapshot,
    make_usage_status_snapshot,
)
from .tui_interactive import run_textual_interactive
from .tui_runtime import (
    StreamCancellationTimeout,
    resolve_ui_backend,
    run_streaming,
    run_streaming_async,
)

_channel_logger = logging.getLogger(__name__)

# Keeps references to fire-and-forget coroutines so they aren't GC'd mid-flight.
_background_tasks: set[asyncio.Task] = set()

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from ..runtime import AsyncRuntime


@dataclass(frozen=True, slots=True)
class _StartupSession:
    """Resolved interactive startup session with a concrete active thread."""

    thread_id: str
    dirs: SessionDirs
    resumed: bool


async def _run_serialized_turn(
    turn_lock: asyncio.Lock,
    operation: Callable[[], Awaitable[Any]],
) -> Any:
    """Run one session turn without overlapping another frontend source."""
    async with turn_lock:
        return await operation()


# =============================================================================
# Banner
# =============================================================================


def print_banner(
    thread_id: str,
    workspace_dir: str | None = None,
    memory_dir: str | None = None,
    mode: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    ui_backend: str | None = None,
):
    """Print welcome banner with ASCII art logo, info line, and hint."""
    for line, color in zip(LOGO_LINES, LOGO_GRADIENT, strict=False):
        console.print(Text(line, style=f"{color} bold"))
    info = Text()
    info.append("  ", style="dim")
    parts: list[tuple[str, str]] = []
    if model:
        parts.append(("Model: ", model))
    if provider:
        parts.append(("Provider: ", provider))
    if mode:
        parts.append(("Mode: ", mode))
    if ui_backend:
        parts.append(("UI: ", ui_backend))
    for i, (label, value) in enumerate(parts):
        if i > 0:
            info.append("  ", style="dim")
        info.append(label, style="dim")
        info.append(value, style="magenta")
    # Directory line
    import os

    effective_dir = workspace_dir or os.getcwd()
    home = os.path.expanduser("~")
    dir_display = (
        effective_dir.replace(home, "~", 1)
        if effective_dir.startswith(home)
        else effective_dir
    )
    info.append("\n  ", style="dim")
    info.append("Directory: ", style="dim")
    info.append(dir_display, style="magenta")
    _nl_key = "Option+Enter" if sys.platform == "darwin" else "Ctrl+J"
    info.append("\n  Enter ", style="#ffe082")
    info.append("send", style="#ffe082 bold")
    info.append(f" \u2022 {_nl_key} ", style="#ffe082")
    info.append("newline", style="#ffe082 bold")
    info.append(" \u2022 Type ", style="#ffe082")
    info.append("/", style="#ffe082 bold")
    info.append(" for commands", style="#ffe082")
    info.append(" \u2022 ", style="#ffe082")
    info.append("@ files", style="#ffe082 bold")
    info.append(" \u2022 Ctrl+C ", style="#ffe082")
    info.append("interrupt", style="#ffe082 bold")
    console.print(info)
    print_dangerous_warning()


def print_dangerous_warning() -> None:
    """Print an unmissable warning when dangerous (real-filesystem) mode is on."""
    try:
        from ..config import get_effective_config

        if not get_effective_config().dangerous_mode:
            return
    except Exception:
        return
    warn = Text()
    warn.append(f"\n  \u26a0 {DANGEROUS_BANNER_LABEL}", style="bold white on red")
    warn.append(f"  {DANGEROUS_BANNER_MESSAGE}", style="bold red")
    console.print(warn)


# =============================================================================
# Slash-command completer
# =============================================================================

_COMPLETION_STYLE = PtStyle.from_dict(
    {
        "completion-menu": "bg:default noreverse nounderline noitalic",
        "completion-menu.completion": "bg:default #888888 noreverse",
        "completion-menu.completion.current": "bg:default default bold noreverse",
        "completion-menu.meta.completion": "bg:default #888888 noreverse",
        "completion-menu.meta.completion.current": "bg:default default bold noreverse",
        "scrollbar.background": "bg:default",
        "scrollbar.button": "bg:default",
        "status-bar": f"bg:{STATUS_BAR_BG} {STATUS_TEXT}",
        "status-bar-strong": f"bg:{STATUS_BAR_BG} {STATUS_STRONG} bold",
        "status-bar-dim": f"bg:{STATUS_BAR_BG} {STATUS_DIM}",
        "status-bar-good": f"bg:{STATUS_BAR_BG} {STATUS_GOOD} bold",
        "status-bar-warn": f"bg:{STATUS_BAR_BG} {STATUS_WARN} bold",
        "status-bar-bad": f"bg:{STATUS_BAR_BG} {STATUS_BAD} bold",
        "status-bar-critical": f"bg:{STATUS_BAR_BG} {STATUS_CRITICAL} bold",
    }
)


class SlashCommandCompleter(Completer):
    """Autocomplete for slash commands and ``@file`` mentions.

    ``dirs_getter`` is invoked on every keystroke so suggestions
    automatically follow ``/new`` / ``/resume`` changes without having to
    poke the completer from the callbacks.
    """

    def __init__(
        self,
        dirs_getter: Callable[[], SessionDirs | None] | None = None,
    ) -> None:
        """Initialise the completer.

        Args:
            dirs_getter: Callable returning the session's folders: the
                workspace for slash-command completions that list installed
                skills or experts, and the work folder for ``@file``
                completions.
        """
        self._dirs_getter = dirs_getter or (lambda: None)

    def get_completions(self, document, complete_event):
        """Yield prompt_toolkit completions for slash commands and ``@file``."""
        text = document.text_before_cursor
        dirs = self._dirs_getter()
        workspace_dir = str(dirs.work_dir) if dirs is not None else None

        # Slash command / subcommand completions take priority
        if text.startswith("/"):
            from ..commands._completion_engine import compute_completions

            result = compute_completions(
                text,
                len(text),
                workspace=dirs.workspace if dirs is not None else None,
            )
            if result.kind != "empty" and result.candidates:
                for c in result.candidates:
                    start_pos = c.replace_start - len(text)
                    yield Completion(
                        c.text,
                        start_position=start_pos,
                        display_meta=c.description,
                    )
            return

        # @file mention completion (only for non-command input)
        if "@" in text:
            candidates = complete_file_mention(text, workspace_dir)
            if candidates:
                import re as _re

                m = _re.search(r'@"[^"\n]*$|@[^\s"\']*$', text)
                start = -len(m.group(0)) if m else 0
                for path, type_hint in candidates:
                    yield Completion(path, start_position=start, display_meta=type_hint)
                return


async def _resolve_startup_session(
    requested_thread_id: str | None,
    *,
    dirs: SessionDirs,
    graph_gateway: GraphGateway,
    config: Any,
    backend: str | None = None,
) -> _StartupSession:
    """Resolve/create the initial CLI session before shared REPL state exists."""
    if not requested_thread_id:
        return _StartupSession(
            thread_id=await graph_gateway.create_thread(GraphTarget(**dirs.metadata())),
            dirs=dirs,
            resumed=False,
        )

    resolution = await graph_gateway.resolve_thread(requested_thread_id)
    if resolution.thread_id is None:
        if resolution.matches:
            console.print(
                f"[yellow]Ambiguous thread ID '{escape(requested_thread_id)}'. "
                "Matches:[/yellow]"
            )
            for match in resolution.matches:
                console.print(f"  [cyan]{match}[/cyan]")
        else:
            console.print(
                f"[red]Thread '{escape(requested_thread_id)}' not found.[/red]"
            )
        return _StartupSession(
            thread_id=await graph_gateway.create_thread(GraphTarget(**dirs.metadata())),
            dirs=dirs,
            resumed=False,
        )

    from .commands import _restore_thread_dirs

    return _StartupSession(
        thread_id=resolution.thread_id,
        dirs=await _restore_thread_dirs(
            resolution.thread_id,
            dirs=dirs,
            served=dirs,
            graph_gateway=graph_gateway,
            config=config,
            backend=backend,
        ),
        resumed=True,
    )


async def _new_session_dirs(
    current: SessionDirs,
    *,
    config: Any,
    backend: str | None,
    mode: str | None,
    run_name: str | None,
) -> SessionDirs:
    """The folders ``/new`` starts in, with the background agent server moved there.

    ``--mode=run`` starts every session in a fresh run folder of the current
    workspace; daemon mode works in its root. The server moves first, so the
    new session's background work lands in its folder. Raises
    ``WorkspaceMismatchError`` when another session holds the server: the
    session stays as it is and the fresh run folder is removed.
    """
    from .commands import _move_server_or_keep_session

    workspace = current.workspace
    new_dirs = SessionDirs(
        workspace,
        _create_run_dir(workspace, run_name) if mode == "run" else None,
    )
    if not await _move_server_or_keep_session(
        config,
        target=new_dirs,
        current=current,
        backend=backend,
        new_run_dir=True,
        status_message="[dim]Moving background agent server to the new session...[/dim]",
    ):
        console.print(
            "[yellow]Background agent server sync failed; started the new "
            "session, but async subagents and EvoMemory workers may be "
            "unavailable.[/yellow]"
        )
    return new_dirs


# =============================================================================
# Interactive & single-shot modes
# =============================================================================


async def _run_rich_cli_streaming_turn(**kwargs: Any) -> str:
    """Run one Rich CLI turn with a fresh, turn-local SIGINT policy.

    ``asyncio.run`` installs a SIGINT handler whose interrupt count lasts for
    the lifetime of the runner.  The Rich CLI intentionally recovers after a
    cancelled turn, so relying on that handler makes Ctrl+C on a later turn
    look like the runner's second interrupt and raises ``KeyboardInterrupt``.

    While a model turn is active, route the first Ctrl+C to a child task
    instead.  Restoring the runner's handler after every turn keeps Ctrl+C at
    the prompt unchanged and resets the force-quit boundary for the next turn.
    A second Ctrl+C before the current turn settles remains a force quit.
    """
    stream_task = asyncio.create_task(
        run_streaming_async(**kwargs, recover_on_cancel=True)
    )

    # Interactive CLI execution belongs on the main thread, but retaining the
    # ordinary await makes this helper safe in embedded/test environments where
    # Python does not permit installing process signal handlers.
    if threading.current_thread() is not threading.main_thread():
        return await stream_task

    previous_sigint = signal.getsignal(signal.SIGINT)
    interrupted = False

    def _cancel_turn(signum: int, frame: Any) -> None:
        nonlocal interrupted
        if interrupted or stream_task.done():
            signal.default_int_handler(signum, frame)
            return
        interrupted = True
        stream_task.cancel()

    signal.signal(signal.SIGINT, _cancel_turn)
    try:
        return await stream_task
    finally:
        signal.signal(signal.SIGINT, previous_sigint)


def cmd_interactive(
    show_thinking: bool = True,
    channel_send_thinking: bool = True,
    mode: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    run_name: str | None = None,
    thread_id: str | None = None,
    ui_backend: str = "cli",
    config=None,
    async_runtime: "AsyncRuntime | None" = None,
    *,
    dirs: SessionDirs,
) -> None:
    """Interactive conversation mode with streaming output.

    The persistent ``AsyncSqliteSaver`` checkpointer is opened here and
    shared for the entire interactive session lifetime.

    Args:
        show_thinking: Whether to display thinking panels
        channel_send_thinking: Whether channels should receive thinking messages
        mode: Workspace mode ('daemon' or 'run'), displayed in banner
        model: Model name to display in banner
        provider: LLM provider name to display in banner
        run_name: Optional run name for /new session deduplication
        thread_id: Optional thread ID to resume a previous session
        ui_backend: UI backend ('cli' or 'tui')
        dirs: The session's workspace, and its run folder under
            ``--mode=run`` (``/new`` then moves to a fresh run folder)
    """
    resolved_ui_backend = resolve_ui_backend(ui_backend, warn_fallback=True)
    if resolved_ui_backend == "tui":
        load_agent = partial(
            _load_agent,
            config=config,
            runtime=async_runtime,
        )
        run_textual_interactive(
            show_thinking=show_thinking,
            channel_send_thinking=channel_send_thinking,
            dirs=dirs,
            mode=mode,
            model=model,
            provider=provider,
            run_name=run_name,
            thread_id=thread_id,
            load_agent=load_agent,
            create_run_dir=_create_run_dir,
            config=config,
            async_runtime=async_runtime,
        )
        return

    from ..config import GatewaySurface, resolve_gateway_backend
    from .commands import warn_server_backend_hitl_caveats

    gateway_backend = resolve_gateway_backend(config, GatewaySurface.INTERACTIVE)
    warn_server_backend_hitl_caveats(gateway_backend, surface_label="interactive CLI")

    from .. import paths

    memory_dir = str(paths.MEMORIES_DIR)

    paths.DATA_DIR.mkdir(parents=True, exist_ok=True)
    history_file = str(paths.DATA_DIR / "history")

    # Key bindings: Enter submits, Alt+Enter (Option+Enter) inserts newline
    _kb = KeyBindings()

    @_kb.add("escape", "enter")  # Alt+Enter / Option+Enter on macOS
    def _insert_newline(event):
        event.current_buffer.insert_text("\n")

    @_kb.add("enter")
    def _submit(event):
        event.current_buffer.validate_and_handle()

    session = PromptSession(
        history=FileHistory(history_file),
        auto_suggest=AutoSuggestFromHistory(),
        completer=SlashCommandCompleter(dirs_getter=lambda: state["dirs"]),
        complete_style=CompleteStyle.COLUMN,
        complete_while_typing=True,
        style=_COMPLETION_STYLE,
        multiline=True,
        key_bindings=_kb,
    )

    def _print_separator():
        """Print a horizontal separator line spanning the terminal width."""
        width = console.size.width
        console.print(Text("\u2500" * width, style="dim"))

    from ..commands.base import ChannelRuntime, active_teams_configurable_extra

    channel_runtime = ChannelRuntime()

    progress_tracker = MCPProgressTracker()

    def _on_mcp_progress(event: str, server: str, detail: str) -> None:
        """Record progress + print the inline ✓/✗ line.

        Runs on the MCP worker thread; ``console.print`` while the main
        loop is inside ``patch_stdout`` lands above the prompt safely.
        """
        new_state = progress_tracker.record(event, server, detail)
        if new_state == "ok":
            console.print(
                f"[green]\u2713[/green] [dim]MCP[/dim] [bold]{server}[/bold] "
                f"[dim]({detail} tools)[/dim]"
            )
        elif new_state == "error":
            console.print(
                f"[red]\u2717[/red] [dim]MCP[/dim] [bold]{server}[/bold] "
                f"[red]failed:[/red] {escape(detail)}"
            )

    agent_loader = BackgroundAgentLoader(
        _load_agent,
        on_progress=_on_mcp_progress,
    )

    # One frontend event sink for the whole session — injected into the agent's
    # middleware (write side) and the local gateway's streaming path (read side)
    # so both share one owner. It survives agent rebuilds (/model, /new, MCP
    # reload) because the session, not the agent, holds it.
    from ..stream.sink import SessionEventSink

    event_sink = SessionEventSink(
        fallback_display=lambda text, style: console.print(text, style=style)
    )

    runtime_gateways = create_runtime_gateways_for_config(
        config, backend=gateway_backend, events=event_sink
    )
    graph_gateway = runtime_gateways.graph_gateway
    requested_thread_id = thread_id

    # Mutable state for async loop
    state: dict[str, Any] = {
        "dirs": dirs,
        "running": True,
        "resumed": False,
        "ui_backend": resolved_ui_backend,
        "status_started_at": datetime.now(),
        "status_base_snapshot": make_empty_status_snapshot(model),
        "status_snapshot": make_empty_status_snapshot(model),
        "status_streaming_text": "",
        "status_last_input_tokens": None,
    }

    def _on_status_after_compact(input_tokens: int) -> None:
        """Mirror inline /compact post-update: refresh both fields so the
        next status render reflects the reduced context immediately.
        ``_refresh_status_snapshot`` is invoked by the dispatch block once
        the command finishes (since it's async)."""
        state["status_last_input_tokens"] = input_tokens
        state["status_base_snapshot"] = make_usage_status_snapshot(
            input_tokens,
            model_name=model,
        )

    # ``rich_ui`` is constructed inside ``_async_main_loop`` so the
    # lifecycle callbacks can close over ``checkpointer`` from
    # ``get_checkpointer()``.

    def _start_agent_load(checkpointer) -> None:
        progress_tracker.prime()
        agent_loader.start(
            workspace=state["dirs"].workspace,
            work_dir=str(state["dirs"].work_dir),
            checkpointer=checkpointer,
            config=config,
            events=event_sink,
            runtime=async_runtime,
        )

    async def _await_agent_ready() -> "CompiledStateGraph":
        """Await the agent load and apply CLI-side post-load side effects.

        Raises when called before ``_start_agent_load``: reloading here
        would drop the SQLite checkpointer and silently lose persistence.
        """
        try:
            agent = await agent_loader.await_ready()
        except RuntimeError as exc:
            if "before start()" in str(exc):
                raise RuntimeError(
                    "_await_agent_ready called before _start_agent_load — "
                    "the checkpointer reference is not available here."
                ) from exc
            raise
        await _refresh_status_snapshot(reset_streaming_text=True)
        if _channels_is_running():
            channel_runtime.bind(agent, state["thread_id"])
        return agent

    def _rebuild_status_snapshot() -> None:
        """Compose the visible snapshot from thread state + live output."""
        state["status_snapshot"] = apply_assistant_text_to_snapshot(
            state["status_base_snapshot"],
            state["status_streaming_text"],
        )

    def _set_status_streaming_text(text: str | None) -> None:
        """Update the in-flight assistant overlay used by the status bar."""
        new_text = text or ""
        if new_text == state["status_streaming_text"]:
            return
        state["status_streaming_text"] = new_text
        _rebuild_status_snapshot()

    async def _refresh_status_snapshot(
        pending_user_text: str | None = None,
        *,
        reset_streaming_text: bool = True,
    ) -> None:
        """Recompute the persistent status-bar snapshot for the active thread."""
        pending = (pending_user_text or "").strip()
        if pending:
            if state["status_last_input_tokens"] is not None:
                state["status_base_snapshot"] = apply_user_text_to_snapshot(
                    make_usage_status_snapshot(
                        state["status_last_input_tokens"],
                        model_name=model,
                    ),
                    pending,
                )
            else:
                state["status_base_snapshot"] = await build_session_status_snapshot(
                    state["thread_id"],
                    model_name=model,
                    pending_user_text=pending,
                    graph_gateway=graph_gateway,
                )
        elif state["status_last_input_tokens"] is not None:
            state["status_base_snapshot"] = make_usage_status_snapshot(
                state["status_last_input_tokens"],
                model_name=model,
            )
        else:
            state["status_base_snapshot"] = await build_session_status_snapshot(
                state["thread_id"],
                model_name=model,
                graph_gateway=graph_gateway,
            )
        if reset_streaming_text:
            state["status_streaming_text"] = ""
        _rebuild_status_snapshot()

    def _bottom_toolbar():
        """Render the persistent bottom status bar for prompt_toolkit."""
        try:
            from prompt_toolkit.application import get_app

            width = get_app().output.get_size().columns
        except Exception:
            width = console.size.width
        fragments = build_status_fragments(
            state["status_snapshot"],
            state["status_started_at"],
            width,
        )
        if agent_loader.is_pending:
            # Per-server ✓/✗ lines are printed above the prompt by
            # `_on_mcp_progress`; this just shows the animated summary.
            done, total = progress_tracker.totals()
            frame = SPINNER_FRAMES[
                int(datetime.now().timestamp() * 10) % len(SPINNER_FRAMES)
            ]
            label = (
                f"{frame} Loading MCP tools {done}/{total} "
                if total and width >= 60
                else f"{frame} Loading MCP tools "
            )
            fragments = [
                ("class:status-bar-warn", label),
                ("class:status-bar-dim", "│ "),
                *fragments,
            ]
        return fragments

    def _stream_status_footer():
        """Render the live Rich footer used during streaming output."""
        return build_status_text(
            state["status_snapshot"],
            state["status_started_at"],
            console.size.width,
        )

    async def _handle_stream_status_event(event_type: str, stream_state) -> None:
        """Keep the CLI status bar aligned with live stream progress."""
        if event_type == "usage_stats":
            last_input_tokens = getattr(stream_state, "last_input_tokens", 0)
            if last_input_tokens > 0:
                state["status_last_input_tokens"] = last_input_tokens
                state["status_base_snapshot"] = make_usage_status_snapshot(
                    last_input_tokens,
                    model_name=model,
                )
                _rebuild_status_snapshot()
        elif event_type == "text":
            _set_status_streaming_text(stream_state.response_text)
        elif event_type in ("done", "error"):
            _set_status_streaming_text("")

    async def _render_history(thread_id: str):
        """Display conversation history for a resumed session."""
        messages = await graph_gateway.get_thread_messages(thread_id)
        if not messages:
            return

        from ..message_meta import is_pinned_skill, pinned_skill_name, skill_label

        HISTORY_WINDOW = 50

        # Only human and ai messages; skip tool/system
        display = [m for m in messages if getattr(m, "type", None) in ("human", "ai")]

        if len(display) > HISTORY_WINDOW:
            skipped = len(display) - HISTORY_WINDOW
            display = display[-HISTORY_WINDOW:]
            console.print(f"[dim]── ... {skipped} earlier messages ──[/dim]")
        else:
            console.print("[dim]── Conversation history ──[/dim]")

        for msg in display:
            msg_type = getattr(msg, "type", None)
            content = getattr(msg, "content", "") or ""

            if msg_type == "human":
                if is_pinned_skill(msg):
                    name = pinned_skill_name(msg) or "unknown"
                    console.print(f"[dim]{escape(skill_label(name))}[/dim]")
                    continue
                # Extract text from multimodal list
                if isinstance(content, list):
                    parts = [
                        b.get("text", "")
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    ]
                    content = " ".join(parts) if parts else ""
                content = content.strip()
                if content:
                    console.print(
                        Text.assemble(("\u276f ", "bold blue"), (content, ""))
                    )

            elif msg_type == "ai":
                thinking_text = ""
                text_content = ""

                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        if block.get("type") == "thinking":
                            thinking_text += block.get("thinking", "")
                        elif block.get("type") == "text":
                            text_content += block.get("text", "")
                else:
                    text_content = content or ""

                text_content = text_content.strip()

                # Thinking panel (only when show_thinking is enabled)
                if thinking_text.strip() and show_thinking:
                    console.print(
                        Panel(
                            thinking_text.strip(),
                            title="[bold blue]\U0001f4ad Thinking[/bold blue]",
                            border_style="blue",
                            expand=False,
                        )
                    )

                # AI response — full Markdown rendering
                if text_content:
                    console.print(Markdown(_fix_markdown_heading_spacing(text_content)))

            # Skip tool messages — verbose and not useful in replay

        console.print("[dim]── End of history ──[/dim]")
        console.print()

    async def _async_main_loop():
        """Async main loop with prompt_async and channel queue checking."""
        nonlocal model
        async with get_checkpointer() as checkpointer:
            startup = await _resolve_startup_session(
                requested_thread_id,
                dirs=state["dirs"],
                graph_gateway=graph_gateway,
                config=config,
                backend=gateway_backend,
            )
            state["thread_id"] = startup.thread_id
            state["dirs"] = startup.dirs
            state["resumed"] = startup.resumed
            if startup.resumed:
                state["status_started_at"] = datetime.now()
                state["status_last_input_tokens"] = None
            # Lifecycle callbacks (new / resume) need ``checkpointer``
            # in scope — define the ``rich_ui`` adapter here rather than
            # at the outer function level.

            def _print_pending_skill_proposals_notice() -> None:
                from .commands import _pending_skill_proposals_message

                message = _pending_skill_proposals_message(state["dirs"].workspace.root)
                if message:
                    console.print(message, style="yellow")

            async def _on_start_new_session() -> None:
                """NewCommand callback — move to a fresh run folder (run mode),
                issue a new thread id, reset session-scoped status fields,
                and kick off background agent reload. The dispatch block
                refreshes the status bar post-execute (symmetric with
                /compact)."""
                new_dirs = await _new_session_dirs(
                    state["dirs"],
                    config=config,
                    backend=gateway_backend,
                    mode=mode,
                    run_name=run_name,
                )
                _ch_mod.forget_channel_origin(state.get("thread_id"))
                state["dirs"] = new_dirs
                state["thread_id"] = await graph_gateway.create_thread(
                    GraphTarget(**state["dirs"].metadata())
                )
                state["resumed"] = False
                state["status_started_at"] = datetime.now()
                state["status_last_input_tokens"] = None
                _start_agent_load(checkpointer)
                console.print(
                    f"[green]New session:[/green] [yellow]{state['thread_id']}[/yellow]"
                )
                console.print(
                    f"[dim]Workspace:[/dim] [cyan]"
                    f"{_shorten_path(str(state['dirs'].work_dir))}[/cyan]\n"
                )
                _print_pending_skill_proposals_notice()

            async def _on_handle_session_resume(
                thread_id: str, dirs: SessionDirs | None
            ) -> None:
                """ResumeCommand callback — after the command resolves
                the thread id + restores workspace from metadata, this
                callback mutates REPL state, reloads the agent, and
                renders conversation history."""
                if dirs is not None:
                    # Move the background agent server to the resumed folders
                    # so background workers and deployed sub-agents don't work
                    # in the previous ones. Restart can take 10-15s behind a
                    # spinner, in a worker thread so the event loop keeps
                    # serving channel polls / MCP heartbeats. State changes
                    # only after the move, so a WorkspaceMismatchError leaves
                    # the session's folders and thread as they are, and command
                    # UIs (including channels) report the failure.
                    from .commands import _move_server_or_keep_session

                    if not await _move_server_or_keep_session(
                        config,
                        target=dirs,
                        current=state["dirs"],
                        backend=gateway_backend,
                    ):
                        console.print(
                            "[yellow]Background agent server sync failed; "
                            "resumed the session, but async subagents and "
                            "EvoMemory workers may be unavailable.[/yellow]"
                        )
                    state["dirs"] = dirs
                    _ch_mod._set_channels_media_dir(dirs.workspace.media_dir)
                if thread_id != state.get("thread_id"):
                    # Only drop the origin on a real thread change — resuming
                    # the already-active thread must keep its live origin so a
                    # later async-notifier turn still forwards to the channel.
                    _ch_mod.forget_channel_origin(state.get("thread_id"))
                state["thread_id"] = thread_id
                state["resumed"] = True
                state["status_started_at"] = datetime.now()
                state["status_last_input_tokens"] = None
                _start_agent_load(checkpointer)
                await _refresh_status_snapshot(reset_streaming_text=True)
                console.print(
                    f"[green]Resumed session:[/green] [yellow]{thread_id}[/yellow]"
                )
                console.print(
                    f"[dim]Workspace:[/dim] [cyan]"
                    f"{_shorten_path(str(state['dirs'].work_dir))}[/cyan]"
                )
                console.print()
                await _render_history(thread_id)
                _print_pending_skill_proposals_notice()

            # Rich CLI collapses ``request_quit`` / ``force_quit`` into the
            # same "break the prompt loop" effect — there's no equivalent
            # of the TUI's double-press-to-confirm quit distinction.  A
            # shared ``_stop`` helper makes the intentional symmetry
            # explicit instead of silently duplicating a lambda.
            def _stop() -> None:
                state["running"] = False

            rich_ui = RichCLICommandUI(
                console,
                on_request_quit=_stop,
                on_force_quit=_stop,
                on_clear_chat=lambda: console.clear(),
                on_status_after_compact=_on_status_after_compact,
                on_start_new_session=_on_start_new_session,
                on_handle_session_resume=_on_handle_session_resume,
            )

            # Kick off agent construction (MCP tool enumeration is the
            # slow part) in the background so the banner and prompt can
            # appear immediately.  The status bar shows a spinner while
            # this is in flight; submitting a message awaits the result.
            _start_agent_load(checkpointer)
            await _refresh_status_snapshot(reset_streaming_text=True)

            # Print banner
            if state["resumed"]:
                print_banner(
                    state["thread_id"],
                    str(state["dirs"].work_dir),
                    memory_dir,
                    mode,
                    model,
                    provider,
                    state["ui_backend"],
                )
                console.print(
                    f"[green]Resumed session [yellow]{state['thread_id']}[/yellow][/green]\n"
                )
                await _render_history(state["thread_id"])
            else:
                print_banner(
                    state["thread_id"],
                    str(state["dirs"].work_dir),
                    memory_dir,
                    mode,
                    model,
                    provider,
                    state["ui_backend"],
                )
            _print_pending_skill_proposals_notice()

            # ---- Channel queue processing (bus → main thread) ----

            turn_lock = asyncio.Lock()

            async def _process_channel_message(msg: ChannelMessage) -> None:
                """Process a single channel message with real-time streaming.

                Clears the waiting prompt line and reprints the message as if
                the user typed it after ❯, then streams the agent response
                with Rich Live display.

                Display:
                  ❯ message content
                  [channel: Received from sender]
                  ─────────────────
                  (real-time streaming output)
                  [channel: Replied to sender]
                  ─────────────────
                """
                if not _ch_mod._claim_or_complete_channel_request(msg):
                    return

                _ch_mod.remember_channel_origin(state["thread_id"], msg)

                try:
                    # Clear the waiting ❯ prompt line
                    sys.stdout.write("\r\033[2K")
                    sys.stdout.flush()

                    # Reprint as if user typed it after ❯
                    prompt_line = Text()
                    prompt_line.append("\u276f ", style="bold blue")
                    prompt_line.append(msg.content)
                    console.print(prompt_line)
                    rx = Text()
                    rx.append(f"[{msg.channel_type}: Received from ", style="dim")
                    rx.append(msg.sender, style="cyan")
                    rx.append("]", style="dim")
                    console.print(rx)
                    _print_separator()

                    pending_channel_sends = PendingChannelSends(
                        _ch_mod._bus_loop, _channel_logger
                    )

                    def _send_to_channel(coro, label: str, timeout: int = 15) -> None:
                        pending_channel_sends.submit(coro, label, timeout)

                    def _send_thinking_to_channel(thinking: str) -> None:
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

                    def _send_todo_to_channel(items: list[dict]) -> None:
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

                    def _send_media_to_channel(file_path: str) -> None:
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

                    def _channel_hitl_outcome(
                        action_requests: list, human_budget_exhausted: bool
                    ):
                        """Send HITL approval prompt to channel user and wait for reply."""
                        return _ch_mod.channel_hitl_prompt(
                            action_requests,
                            msg,
                            human_budget_exhausted=human_budget_exhausted,
                        )

                    def _channel_ask_user(ask_user_data: dict) -> dict:
                        """Send ask_user questions to channel user and wait for reply."""
                        return _ch_mod.channel_ask_user_prompt(ask_user_data, msg)

                    # ---- Slash command dispatch (cmd_manager, not the agent) ----
                    # Mirrors the TUI's behavior so ``/evoskills``, ``/mcp list``
                    # etc. sent via iMessage actually execute instead of being
                    # fed to the LLM as a plain prompt.
                    async def _on_channel_cmd_completed(
                        ctx: CommandContext, original_agent: Any, cmd: Command
                    ) -> None:
                        """Mirror the REPL adoption block at
                        ``interactive.py:1005-1030`` so ``/model`` and similar
                        state-mutating commands invoked via a channel actually
                        rebind the running session and keep the status bar
                        in sync."""
                        nonlocal model
                        agent_swapped = (
                            ctx.agent is not None and ctx.agent is not original_agent
                        )
                        if agent_swapped:
                            from ..EvoScientist import _ensure_config

                            agent_loader.adopt(ctx.agent)
                            cfg = _ensure_config()
                            model = cfg.model
                            state["status_base_snapshot"] = make_empty_status_snapshot(
                                model
                            )

                        # Rebind the runtime whenever the agent OR
                        # thread_id may have moved — ``/new`` and
                        # ``/resume`` rotate ``state["thread_id"]``
                        # without swapping the agent, and the bus
                        # expects both to stay in sync (matches the
                        # serve-mode hook contract).
                        if _channels_is_running():
                            runtime_agent = (
                                ctx.agent
                                if ctx.agent is not None
                                else agent_loader.agent
                            )
                            if runtime_agent is not None:
                                channel_runtime.bind(runtime_agent, state["thread_id"])

                        # ``/new`` rotates ``state["thread_id"]`` / workspace,
                        # ``/compact`` reduces token usage — both need the
                        # status snapshot re-rendered even when the agent
                        # didn't swap.  ``/resume`` refreshes inline in its
                        # own async callback.
                        if agent_swapped or cmd.name in (
                            "/compact",
                            "/new",
                        ):
                            await _refresh_status_snapshot(reset_streaming_text=True)

                    _slash_handled = await dispatch_channel_slash_command(
                        msg,
                        agent=agent_loader.agent,
                        thread_id=state["thread_id"],
                        dirs=state["dirs"],
                        checkpointer=checkpointer,
                        append_system=lambda t, s="dim": console.print(t, style=s),
                        start_new_session_cb=_on_start_new_session,
                        handle_session_resume_cb=_on_handle_session_resume,
                        await_agent_ready=_await_agent_ready,
                        on_cmd_completed=_on_channel_cmd_completed,
                        channel_runtime=channel_runtime,
                        graph_gateway=runtime_gateways.graph_gateway,
                        async_runtime=async_runtime,
                    )
                    if _slash_handled:
                        # A channel-issued /new or /resume rotates the thread
                        # inside the dispatch above; re-bind the now-current
                        # thread to this channel so async-notifier turns on it
                        # still forward back here.
                        _ch_mod.remember_channel_origin(state["thread_id"], msg)
                        _print_separator()
                        sys.stdout.write("\033[34;1m❯\033[0m ")
                        sys.stdout.flush()
                        return

                    try:
                        ready_agent = await _await_agent_ready()
                        meta = build_metadata(state["dirs"], model)
                        await _refresh_status_snapshot(
                            msg.content, reset_streaming_text=True
                        )
                        response = await run_streaming_async(
                            ui_backend=state["ui_backend"],
                            agent=ready_agent,
                            message=msg.content,
                            thread_id=state["thread_id"],
                            show_thinking=show_thinking,
                            interactive=True,
                            metadata=meta,
                            configurable_extra=active_teams_configurable_extra(
                                channel_runtime
                            ),
                            on_thinking=_send_thinking_to_channel,
                            on_todo=_send_todo_to_channel,
                            on_file_write=_send_media_to_channel,
                            work_dir=str(state["dirs"].work_dir),
                            hitl_outcome_fn=_channel_hitl_outcome,
                            ask_user_prompt_fn=_channel_ask_user,
                            on_stream_event=_handle_stream_status_event,
                            status_footer_builder=_stream_status_footer,
                            cancel_scope=_ch_mod._channel_message_cancel_scope(msg),
                            gateway=runtime_gateways.graph_gateway,
                            runtime=async_runtime,
                        )
                        # Channel turns run on the same thread as local prompts
                        # and can launch async tasks and background processes;
                        # without these reads their completions surface only on
                        # the next local prompt. Same calls + args as the
                        # local-path post-stream reads; best-effort, never
                        # blocks the reply.
                        _channel_target = GraphTarget(
                            local_graph=ready_agent,
                            **state["dirs"].metadata(),
                        )
                        await async_notifier.enqueue_completions_from_state(
                            runtime_gateways.graph_gateway,
                            _channel_target,
                            state["thread_id"],
                        )
                        await async_notifier.enqueue_bg_process_completions_from_state(
                            runtime_gateways.graph_gateway,
                            _channel_target,
                            state["thread_id"],
                        )
                    except Exception as e:
                        response = f"Error: {e}"
                        console.print(f"[red]Channel error: {e}[/red]")

                    await pending_channel_sends.settle_async()
                    _set_channel_response(msg.msg_id, response)
                    await _refresh_status_snapshot(reset_streaming_text=True)

                    tx = Text()
                    tx.append(f"[{msg.channel_type}: Replied to ", style="dim")
                    tx.append(msg.sender, style="cyan")
                    tx.append("]", style="dim")
                    console.print(tx)
                    _print_separator()

                    # Redraw the ❯ prompt on a new line after separator
                    sys.stdout.write("\033[34;1m\u276f\033[0m ")
                    sys.stdout.flush()
                finally:
                    _ch_mod._complete_channel_request(msg.msg_id)

            async def _inject_notification_message(
                text: str,
                notifs: list,
                *,
                target_thread_id: str | None,
            ) -> None:
                """Inject a batched async-task notification as a synthetic user message.

                Renders one compact tool-result-style line per task (matching the
                TaskList spinner aesthetic) for the human operator.  The LLM
                receives the full structured ``text`` from ``format_batch_message``
                unchanged — only the screen visual is simplified.
                """
                from EvoScientist.cli.async_notifier import format_notification_lines

                for line_text, line_style in format_notification_lines(notifs):
                    console.print(line_text, style=line_style, markup=False)
                meta = build_metadata(state["dirs"], model)
                await _refresh_status_snapshot(text, reset_streaming_text=True)
                ready_agent = await _await_agent_ready()
                response = await run_streaming_async(
                    ui_backend=state["ui_backend"],
                    agent=ready_agent,
                    message=text,
                    # Falls back to live state["thread_id"] if no override is
                    # passed (legacy / direct-call paths). Dedup reader has no
                    # fallback and returns {} for a falsey id; the asymmetry
                    # is intentional — we'd rather inject into the live thread
                    # than drop the notification entirely.
                    thread_id=target_thread_id or state["thread_id"],
                    show_thinking=show_thinking,
                    interactive=True,
                    metadata=meta,
                    configurable_extra=active_teams_configurable_extra(channel_runtime),
                    on_stream_event=_handle_stream_status_event,
                    status_footer_builder=_stream_status_footer,
                    gateway=runtime_gateways.graph_gateway,
                    runtime=async_runtime,
                )
                _notif_tid = target_thread_id or state["thread_id"]
                if _ch_mod.publish_to_channel_origin(_notif_tid, response):
                    # Forwarded to a channel — print the same closing
                    # "Replied to" line a normal channel turn shows, so the
                    # forwarded block reads as terminated on screen.
                    _origin = _ch_mod.get_channel_origin(_notif_tid)
                    if _origin is not None:
                        tx = Text()
                        tx.append(f"[{_origin.channel_type}: Replied to ", style="dim")
                        tx.append(_origin.sender or _origin.chat_id, style="cyan")
                        tx.append("]", style="dim")
                        console.print(tx)
                # Re-arm the completion reader after the notification turn: it
                # may have launched a chained task (e.g. analysis finished ->
                # start writing) that would otherwise sit in state with the idle
                # reader disarmed until the next local prompt. Best-effort.
                await async_notifier.enqueue_completions_from_state(
                    runtime_gateways.graph_gateway,
                    GraphTarget(
                        local_graph=ready_agent,
                        **state["dirs"].metadata(),
                    ),
                    _notif_tid,
                )
                await _refresh_status_snapshot(reset_streaming_text=True)
                console.print()
                _print_separator()
                sys.stdout.write("\033[34;1m❯\033[0m ")
                sys.stdout.flush()

            async def _empty_async_tasks() -> async_notifier.AsyncTasksState:
                return {}

            async def _read_current_async_tasks(
                target_thread_id: str,
            ) -> async_notifier.AsyncTasksState:
                """Snapshot async_tasks from the active agent state for dedup.

                Uses ``agent_loader.agent`` (the currently loaded agent) and
                ``target_thread_id`` (the thread id captured at the start of
                ``consume_notifications`` — frozen so a mid-consume ``/new``
                cannot make us read the wrong thread's state).
                """
                agent = agent_loader.agent
                if agent is None:
                    return {}
                try:
                    registry = await async_notifier.read_async_tasks_from_gateway(
                        runtime_gateways.graph_gateway,
                        GraphTarget(
                            local_graph=agent,
                            **state["dirs"].metadata(),
                        ),
                        target_thread_id,
                    )
                    return registry or {}
                except Exception:
                    return {}

            async def _check_channel_queue() -> None:
                """Poll the channel + notification queues and dispatch."""
                while True:
                    try:
                        msg = _message_queue.get_nowait()
                    except queue.Empty:
                        msg = None
                    if msg is not None:
                        await _run_serialized_turn(
                            turn_lock,
                            lambda _msg=msg: _process_channel_message(_msg),
                        )
                        continue  # check queues again immediately

                    # Notification path (only when no channel message was pending).
                    # Wrap in try/except so an exception in dedup/inject can't
                    # kill the poller task — channel + notification dispatch
                    # would silently die otherwise (Fix #4).
                    current_tid = state.get("thread_id")
                    # Detect async-task + bg-process completions from state
                    # (throttled) so a completion surfaces while the prompt sits
                    # idle; the drain below injects it. Best-effort — never kill the
                    # poller task.
                    _reader_agent = agent_loader.agent
                    if current_tid and _reader_agent is not None:
                        _reader_target = GraphTarget(
                            local_graph=_reader_agent,
                            **state["dirs"].metadata(),
                        )
                        try:
                            await (
                                async_notifier.enqueue_completions_from_state_throttled(
                                    runtime_gateways.graph_gateway,
                                    _reader_target,
                                    current_tid,
                                )
                            )
                            await async_notifier.enqueue_bg_process_completions_from_state_throttled(
                                runtime_gateways.graph_gateway,
                                _reader_target,
                                current_tid,
                            )
                        except Exception:
                            _channel_logger.warning(
                                "async-notifier idle reader failed", exc_info=True
                            )
                    if async_notifier.has_pending_notifications(current_tid):
                        read_async_tasks_state = (
                            (lambda _tid=current_tid: _read_current_async_tasks(_tid))
                            if current_tid
                            else _empty_async_tasks
                        )
                        try:
                            await async_notifier.consume_notifications(
                                run_message=lambda text, notifs, _tid=current_tid: (
                                    _run_serialized_turn(
                                        turn_lock,
                                        lambda: _inject_notification_message(
                                            text,
                                            notifs,
                                            target_thread_id=_tid,
                                        ),
                                    )
                                ),
                                read_async_tasks_state=read_async_tasks_state,
                                current_thread_id=current_tid,
                            )
                        except Exception:
                            _channel_logger.warning(
                                "async-notifier consume failed", exc_info=True
                            )
                        continue

                    await asyncio.sleep(0.1)

            queue_task = asyncio.create_task(_check_channel_queue())

            # Startup hint
            console.print(
                Text(
                    "  EvoScientist is your research buddy.\n"
                    "  Tell it about your taste before cooking some meal!",
                    style="yellow",
                )
            )

            # Auto-start channel if enabled in config.  Needs the agent
            # bound before the bus starts polling, so schedule it as a
            # background coroutine that waits for the loader first.
            from ..config import load_config

            _channel_cfg = load_config()
            if (
                _channel_cfg
                and _channel_cfg.channel_enabled
                and not _channels_is_running()
            ):

                async def _deferred_auto_start_channel(cfg):
                    try:
                        agent = await _await_agent_ready()
                    except Exception as e:
                        console.print(
                            f"[red]Channel auto-start skipped: agent load failed:[/red] "
                            f"{escape(str(e))}"
                        )
                        return
                    if not _channels_is_running():
                        _auto_start_channel(
                            agent,
                            state["thread_id"],
                            cfg,
                            media_dir=state["dirs"].workspace.media_dir,
                            send_thinking=channel_send_thinking,
                            runtime=channel_runtime,
                        )

                _auto_start_task = asyncio.create_task(
                    _deferred_auto_start_channel(_channel_cfg)
                )
                _background_tasks.add(_auto_start_task)
                _auto_start_task.add_done_callback(_background_tasks.discard)

            # Update check — non-blocking, runs in background thread
            import concurrent.futures

            _update_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

            def _show_update_hint() -> None:
                try:
                    from ..update_check import _installed_version, is_update_available

                    available, latest = is_update_available()
                    if available:
                        current = _installed_version()
                        console.print(
                            Text(
                                f"  Update available: v{latest} (current: v{current}).\n"
                                "  Run: uv tool upgrade EvoScientist",
                                style="yellow",
                            )
                        )
                except Exception:
                    pass

            _update_executor.submit(_show_update_hint)

            # Slogan — after channels, right before user input
            console.print(
                Text(f"  {random.choice(WELCOME_SLOGANS)}", style="dim italic")
            )
            console.print()

            try:
                _print_separator()
                while state["running"]:
                    try:
                        # ``patch_stdout`` routes stray ``print`` /
                        # ``console.print`` calls — including the MCP
                        # progress callback firing from a worker thread —
                        # above the live prompt instead of over it.
                        with patch_stdout(raw=True):
                            user_input = await session.prompt_async(
                                HTML("<ansiblue><b>\u276f</b></ansiblue> "),
                                bottom_toolbar=_bottom_toolbar,
                                refresh_interval=1.0,
                            )
                        user_input = user_input.strip()

                        if not user_input:
                            # Erase the empty prompt line so it looks like nothing happened
                            sys.stdout.write("\033[A\033[2K\r")
                            sys.stdout.flush()
                            continue

                        _print_separator()

                        # ==== Shared CommandManager dispatch ====
                        # Every registered slash command routes through the
                        # manager.  Unresolved input (non-slash, typo) falls
                        # through to the agent message path below.
                        _parsed = cmd_manager.resolve(user_input)
                        if _parsed is not None:
                            _cmd, _cmd_args = _parsed
                            _agent_for_ctx: Any = agent_loader.agent
                            if _cmd.needs_agent(_cmd_args):
                                _agent_for_ctx = await _await_agent_ready()
                            ctx = CommandContext(
                                agent=_agent_for_ctx,
                                thread_id=state["thread_id"],
                                ui=rich_ui,
                                dirs=state["dirs"],
                                checkpointer=checkpointer,
                                config=config,
                                input_tokens_hint=state.get("status_last_input_tokens"),
                                channel_runtime=channel_runtime,
                                graph_gateway=runtime_gateways.graph_gateway,
                                async_runtime=async_runtime,
                            )
                            await cmd_manager.execute(user_input, ctx)

                            # ExitCommand signals quit via ``force_quit`` →
                            # callback flips ``state["running"]`` to False.
                            if not state["running"]:
                                break

                            # Agent swap (e.g. /model successfully built a
                            # new agent): adopt into loader + reset status
                            # snapshot + sync channel runtime.
                            agent_swapped = (
                                ctx.agent is not None
                                and ctx.agent is not _agent_for_ctx
                            )
                            if agent_swapped:
                                from ..EvoScientist import _ensure_config

                                agent_loader.adopt(ctx.agent)
                                cfg = _ensure_config()
                                model = cfg.model
                                state["status_base_snapshot"] = (
                                    make_empty_status_snapshot(model)
                                )

                            # Rebind the runtime whenever the agent OR
                            # thread_id may have moved — ``/new`` /
                            # ``/resume`` rotate ``state["thread_id"]``
                            # without swapping the agent, and the bus
                            # expects both to stay in sync.
                            if _channels_is_running():
                                runtime_agent = (
                                    ctx.agent
                                    if ctx.agent is not None
                                    else agent_loader.agent
                                )
                                if runtime_agent is not None:
                                    channel_runtime.bind(
                                        runtime_agent, state["thread_id"]
                                    )

                            # Commands that mutate status fields need an
                            # async refresh here (/compact + /new use sync
                            # callbacks; /model swaps the agent).  /resume
                            # awaits its own refresh inline inside the
                            # async callback.
                            if agent_swapped or _cmd.name in ("/compact", "/new"):
                                await _refresh_status_snapshot(
                                    reset_streaming_text=True,
                                )
                            continue

                        # Unknown slash command (typo) — short-circuit so
                        # it doesn't get forwarded to the agent, which
                        # would waste tokens interpreting the nonsense.
                        # ``/skill-name`` naming an installed skill goes on
                        # to the agent, which pins that skill.
                        if user_input.lstrip().startswith("/"):
                            from ..commands.skill_slash import skill_slash_names
                            from ..message_meta import skill_label

                            pinned = skill_slash_names(
                                user_input, state["dirs"].workspace
                            )
                            if not pinned:
                                bad_cmd = user_input.split(None, 1)[0]
                                console.print(
                                    f"[red]Unknown command:[/red] {escape(bad_cmd)}"
                                )
                                console.print(
                                    "[dim]Type /help to see available commands.[/dim]"
                                )
                                console.print()
                                continue
                            for name in pinned:
                                console.print(f"[dim]{escape(skill_label(name))}[/dim]")

                        # Resolve @file mentions — inject file contents inline
                        _, message_to_send, file_warnings = resolve_file_mentions(
                            user_input, str(state["dirs"].work_dir)
                        )

                        # Stream agent response with metadata for persistence
                        # Warnings printed here so they appear just before the
                        # model response, not before the user input echo.
                        for w in file_warnings:
                            console.print(f"[yellow]⚠ {escape(w)}[/yellow]")
                        console.print()
                        ready_agent = await _await_agent_ready()
                        meta = build_metadata(state["dirs"], model)
                        await _refresh_status_snapshot(
                            message_to_send, reset_streaming_text=True
                        )
                        try:
                            await _run_serialized_turn(
                                turn_lock,
                                lambda _agent=ready_agent, _message=message_to_send, _thread_id=state["thread_id"], _meta=meta: (
                                    _run_rich_cli_streaming_turn(
                                        ui_backend=state["ui_backend"],
                                        agent=_agent,
                                        message=_message,
                                        thread_id=_thread_id,
                                        show_thinking=show_thinking,
                                        interactive=True,
                                        metadata=_meta,
                                        configurable_extra=active_teams_configurable_extra(
                                            channel_runtime
                                        ),
                                        on_stream_event=_handle_stream_status_event,
                                        status_footer_builder=_stream_status_footer,
                                        gateway=runtime_gateways.graph_gateway,
                                        runtime=async_runtime,
                                    )
                                ),
                            )
                        finally:
                            # On stream close — or a raised turn — read async_tasks
                            # + bg_processes off thread state and enqueue any
                            # completions not yet surfaced, so a task launched
                            # mid-turn still surfaces and keeps idle polling armed.
                            # Best-effort — never blocks the prompt on a read
                            # failure.
                            _close_target = GraphTarget(
                                local_graph=ready_agent,
                                **state["dirs"].metadata(),
                            )
                            await async_notifier.enqueue_completions_from_state(
                                runtime_gateways.graph_gateway,
                                _close_target,
                                state["thread_id"],
                            )
                            await async_notifier.enqueue_bg_process_completions_from_state(
                                runtime_gateways.graph_gateway,
                                _close_target,
                                state["thread_id"],
                            )
                        await _refresh_status_snapshot(reset_streaming_text=True)
                        console.print()
                        _print_separator()

                    except KeyboardInterrupt:
                        console.print()
                        state["running"] = False
                        break
                    except EOFError:
                        # Handle Ctrl+D
                        console.print()
                        state["running"] = False
                        break
                    except StreamCancellationTimeout as e:
                        console.print(f"[red]{escape(str(e))}[/red]")
                        console.print(
                            "[dim]Exiting because the active turn could not be "
                            "stopped safely.[/dim]"
                        )
                        state["running"] = False
                        break
                    except Exception as e:
                        error_msg = str(e)
                        if (
                            "authentication" in error_msg.lower()
                            or "api_key" in error_msg.lower()
                        ):
                            console.print("[red]Error: API key not configured.[/red]")
                            console.print(
                                "[dim]Run [bold]EvoSci onboard[/bold] to set up your API key.[/dim]"
                            )
                            state["running"] = False
                            break
                        else:
                            console.print(f"[red]Error: {escape(str(e))}[/red]")
            finally:
                queue_task.cancel()
                try:
                    await queue_task
                except asyncio.CancelledError:
                    pass
                try:
                    from ..middleware.code_interpreter import (
                        aclose_code_interpreters,
                    )

                    await aclose_code_interpreters()
                except Exception:
                    _channel_logger.debug(
                        "code interpreter cleanup failed",
                        exc_info=True,
                    )
                # Best-effort: guard so a DB lookup failure here can't
                # shadow the original exception exiting _async_main_loop.
                current_tid = state.get("thread_id")
                if current_tid:
                    try:
                        if await graph_gateway.thread_exists(current_tid):
                            state["resume_hint_thread_id"] = current_tid
                    except Exception:
                        _channel_logger.debug(
                            "resume-hint thread_exists lookup failed",
                            exc_info=True,
                        )

    # Run the async main loop
    from .resume_hint import print_resume_hint

    try:
        asyncio.run(_async_main_loop())
    except KeyboardInterrupt:
        console.print()
    finally:
        try:
            print_resume_hint(state.get("resume_hint_thread_id"), console=console)
        except Exception:
            _channel_logger.debug("print_resume_hint failed", exc_info=True)


def cmd_run(
    agent: "CompiledStateGraph",
    prompt: str,
    thread_id: str,
    show_thinking: bool = True,
    model: str | None = None,
    ui_backend: str = "cli",
    *,
    dirs: SessionDirs,
    runtime_gateways: RuntimeGateways,
    async_runtime: "AsyncRuntime | None" = None,
) -> None:
    """Single-shot execution with streaming display.

    Args:
        agent: Compiled agent graph
        prompt: User prompt
        thread_id: Thread ID for conversation persistence.
        show_thinking: Whether to display thinking panels
        dirs: The session's workspace and run folder
        model: Model name for checkpoint metadata
        ui_backend: UI backend ('cli' or 'tui')
    """
    width = console.size.width
    sep = Text("\u2500" * width, style="dim")
    console.print(sep)
    console.print(Text(f"> {prompt}"))
    console.print(sep)
    console.print(f"[dim]Thread: {short_thread_id(thread_id)}[/dim]")
    console.print(f"[dim]Workspace: {_shorten_path(str(dirs.work_dir))}[/dim]")
    console.print()

    meta = build_metadata(dirs, model)
    try:
        run_streaming(
            ui_backend=resolve_ui_backend(ui_backend, warn_fallback=True),
            agent=agent,
            message=prompt,
            thread_id=thread_id,
            show_thinking=show_thinking,
            interactive=False,
            metadata=meta,
            gateway=runtime_gateways.graph_gateway,
            runtime=async_runtime,
        )
        from .commands import _wait_for_memory_workers

        _wait_for_memory_workers()
    except Exception as e:
        error_msg = str(e)
        if "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
            console.print("[red]Error: API key not configured.[/red]")
            console.print(
                "[dim]Run [bold]EvoSci onboard[/bold] to set up your API key.[/dim]"
            )
            raise typer.Exit(1) from e
        else:
            console.print(f"[red]Error: {e}[/red]")
            # This is the process boundary for single-shot text mode.  Letting
            # provider exceptions escape makes Typer/Rich render the complete
            # async exception chain after we already printed a concise error;
            # large OpenAI/httpx chains can keep the CLI busy well after the
            # resume hint is shown.  Convert the failure to Click's controlled
            # exit signal while preserving the cause for programmatic callers.
            raise typer.Exit(1) from e
