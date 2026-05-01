"""Interactive CLI mode and single-shot execution."""

import asyncio
import logging
import queue
import random
import sys
from collections.abc import Callable
from datetime import datetime
from typing import Any

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

from ..commands.base import CommandContext
from ..commands.manager import manager as cmd_manager
from ..sessions import (
    generate_thread_id,
    get_checkpointer,
    get_thread_messages,
    get_thread_metadata,
    resolve_thread_id_prefix,
    thread_exists,
)
from ..stream.console import console
from ._agent_loader import BackgroundAgentLoader, MCPProgressTracker
from ._constants import LOGO_GRADIENT, LOGO_LINES, WELCOME_SLOGANS, build_metadata
from .agent import _create_session_workspace, _load_agent, _shorten_path
from .channel import (
    ChannelMessage,
    _auto_start_channel,
    _channels_is_running,
    _message_queue,
    _set_channel_response,
    dispatch_channel_slash_command,
)
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
from .tui_runtime import resolve_ui_backend, run_streaming

_channel_logger = logging.getLogger(__name__)

# Keeps references to fire-and-forget coroutines so they aren't GC'd mid-flight.
_background_tasks: set[asyncio.Task] = set()

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

    ``workspace_getter`` is invoked on every keystroke so ``@file``
    suggestions automatically follow ``/new`` / ``/resume`` workspace
    changes without having to poke the completer from the callbacks.
    """

    def __init__(
        self,
        workspace_getter: Callable[[], str | None] | None = None,
    ) -> None:
        self._workspace_getter = workspace_getter or (lambda: None)

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        workspace_dir = self._workspace_getter()

        # @file mention completion
        if "@" in text:
            candidates = complete_file_mention(text, workspace_dir)
            if candidates:
                # Replace from the last '@' token
                import re as _re

                m = _re.search(r"@[^\s]*$", text)
                start = -len(m.group(0)) if m else 0
                for path, type_hint in candidates:
                    yield Completion(path, start_position=start, display_meta=type_hint)
                return

        # Slash command completion
        if not text.startswith("/"):
            return
        # ``list_commands`` is dedup'd on the Command instance so aliases
        # (e.g. /quit, /q for /exit) don't appear as separate rows.
        for cmd, desc in sorted(cmd_manager.list_commands()):
            if cmd.startswith(text):
                yield Completion(
                    cmd,
                    start_position=-len(text),
                    display=f"{cmd:<40}",
                    display_meta=desc,
                )


# =============================================================================
# Interactive & single-shot modes
# =============================================================================


def cmd_interactive(
    show_thinking: bool = True,
    channel_send_thinking: bool = True,
    workspace_dir: str | None = None,
    workspace_fixed: bool = False,
    mode: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    run_name: str | None = None,
    thread_id: str | None = None,
    ui_backend: str = "cli",
    config=None,
) -> None:
    """Interactive conversation mode with streaming output.

    The persistent ``AsyncSqliteSaver`` checkpointer is opened here and
    shared for the entire interactive session lifetime.

    Args:
        show_thinking: Whether to display thinking panels
        channel_send_thinking: Whether channels should receive thinking messages
        workspace_dir: Per-session workspace directory path
        workspace_fixed: If True, /new keeps the same workspace directory
        mode: Workspace mode ('daemon' or 'run'), displayed in banner
        model: Model name to display in banner
        provider: LLM provider name to display in banner
        run_name: Optional run name for /new session deduplication
        thread_id: Optional thread ID to resume a previous session
        ui_backend: UI backend ('cli' or 'tui')
    """
    import nest_asyncio

    nest_asyncio.apply()

    resolved_ui_backend = resolve_ui_backend(ui_backend, warn_fallback=True)
    if resolved_ui_backend == "tui":
        from functools import partial

        load_agent = partial(_load_agent, config=config)
        run_textual_interactive(
            show_thinking=show_thinking,
            channel_send_thinking=channel_send_thinking,
            workspace_dir=workspace_dir,
            workspace_fixed=workspace_fixed,
            mode=mode,
            model=model,
            provider=provider,
            run_name=run_name,
            thread_id=thread_id,
            load_agent=load_agent,
            create_session_workspace=_create_session_workspace,
        )
        return

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
        completer=SlashCommandCompleter(
            workspace_getter=lambda: state["workspace_dir"],
        ),
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

    # Mutable state for async loop
    state: dict[str, Any] = {
        "thread_id": thread_id or generate_thread_id(),
        "workspace_dir": workspace_dir,
        "running": True,
        "resumed": False,
        "ui_backend": resolved_ui_backend,
        "status_started_at": datetime.now(),
        "status_base_snapshot": make_empty_status_snapshot(model),
        "status_snapshot": make_empty_status_snapshot(model),
        "status_streaming_text": "",
        "status_last_input_tokens": None,
    }

    from ..commands.base import ChannelRuntime

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
            workspace_dir=state["workspace_dir"],
            checkpointer=checkpointer,
            config=config,
        )

    async def _await_agent_ready() -> Any:
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

    async def _resolve_thread_id(tid: str) -> str | None:
        """Resolve a (possibly partial) thread ID. Returns full ID or None."""
        resolved, matches = await resolve_thread_id_prefix(tid)
        if resolved:
            return resolved
        if matches:
            console.print(
                f"[yellow]Ambiguous thread ID '{escape(tid)}'. Matches:[/yellow]"
            )
            for s in matches:
                console.print(f"  [cyan]{s}[/cyan]")
            return None
        console.print(f"[red]Thread '{escape(tid)}' not found.[/red]")
        return None

    async def _render_history(thread_id: str):
        """Display conversation history for a resumed session."""
        messages = await get_thread_messages(thread_id)
        if not messages:
            return

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
                    console.print(Markdown(text_content))

            # Skip tool messages — verbose and not useful in replay

        console.print("[dim]── End of history ──[/dim]")
        console.print()

    async def _async_main_loop():
        """Async main loop with prompt_async and channel queue checking."""
        nonlocal model
        async with get_checkpointer() as checkpointer:
            # Lifecycle callbacks (new / resume) need ``checkpointer``
            # in scope — define the ``rich_ui`` adapter here rather than
            # at the outer function level.

            def _on_start_new_session() -> None:
                """NewCommand callback — rotate workspace (if not fixed),
                issue a new thread id, reset session-scoped status fields,
                and kick off background agent reload. The dispatch block
                refreshes the status bar post-execute (symmetric with
                /compact)."""
                if not workspace_fixed:
                    state["workspace_dir"] = _create_session_workspace(run_name)
                state["thread_id"] = generate_thread_id()
                state["resumed"] = False
                state["status_started_at"] = datetime.now()
                state["status_last_input_tokens"] = None
                _start_agent_load(checkpointer)
                console.print(
                    f"[green]New session:[/green] [yellow]{state['thread_id']}[/yellow]"
                )
                if state["workspace_dir"]:
                    console.print(
                        f"[dim]Workspace:[/dim] [cyan]"
                        f"{_shorten_path(state['workspace_dir'])}[/cyan]\n"
                    )

            async def _on_handle_session_resume(
                thread_id: str, workspace_dir: str | None
            ) -> None:
                """ResumeCommand callback — after the command resolves
                the thread id + restores workspace from metadata, this
                callback mutates REPL state, reloads the agent, and
                renders conversation history."""
                if workspace_dir:
                    state["workspace_dir"] = workspace_dir
                state["thread_id"] = thread_id
                state["resumed"] = True
                state["status_started_at"] = datetime.now()
                state["status_last_input_tokens"] = None
                _start_agent_load(checkpointer)
                await _refresh_status_snapshot(reset_streaming_text=True)
                console.print(
                    f"[green]Resumed session:[/green] [yellow]{thread_id}[/yellow]"
                )
                if state["workspace_dir"]:
                    console.print(
                        f"[dim]Workspace:[/dim] [cyan]"
                        f"{_shorten_path(state['workspace_dir'])}[/cyan]"
                    )
                console.print()
                await _render_history(thread_id)

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

            # Handle --thread-id resume
            if thread_id:
                resolved = await _resolve_thread_id(thread_id)
                if resolved:
                    meta = await get_thread_metadata(resolved)
                    ws = (meta or {}).get("workspace_dir", "") or state["workspace_dir"]
                    state["thread_id"] = resolved
                    state["resumed"] = True
                    state["status_started_at"] = datetime.now()
                    state["status_last_input_tokens"] = None
                    if ws:
                        state["workspace_dir"] = ws
                else:
                    # Resolution failed (ambiguous/not-found); the user's raw
                    # input is still seeded in state["thread_id"] from init.
                    # Replace with a fresh ID so a new session isn't
                    # checkpointed under the bad prefix.
                    state["thread_id"] = generate_thread_id()

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
                    state["workspace_dir"],
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
                    state["workspace_dir"],
                    memory_dir,
                    mode,
                    model,
                    provider,
                    state["ui_backend"],
                )

            # ---- Channel queue processing (bus → main thread) ----

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

                    def _send_to_channel(coro, label: str, timeout: int = 15) -> None:
                        """Schedule an async channel send on the bus loop."""
                        loop = _ch_mod._bus_loop
                        if not loop:
                            return
                        try:
                            asyncio.run_coroutine_threadsafe(coro, loop).result(
                                timeout=timeout
                            )
                        except Exception as e:
                            _channel_logger.debug(f"{label} send failed: {e}")

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

                    def _channel_hitl_prompt(
                        action_requests: list,
                    ) -> list[dict] | None:
                        """Send HITL approval prompt to channel user and wait for reply."""
                        return _ch_mod.channel_hitl_prompt(action_requests, msg)

                    def _channel_ask_user(ask_user_data: dict) -> dict:
                        """Send ask_user questions to channel user and wait for reply."""
                        return _ch_mod.channel_ask_user_prompt(ask_user_data, msg)

                    # ---- Slash command dispatch (cmd_manager, not the agent) ----
                    # Mirrors the TUI's behavior so ``/evoskills``, ``/mcp list``
                    # etc. sent via iMessage actually execute instead of being
                    # fed to the LLM as a plain prompt.
                    async def _on_channel_cmd_completed(
                        ctx: Any, original_agent: Any, cmd: Any
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
                        if agent_swapped or getattr(cmd, "name", None) in (
                            "/compact",
                            "/new",
                        ):
                            await _refresh_status_snapshot(reset_streaming_text=True)

                    _slash_handled = await dispatch_channel_slash_command(
                        msg,
                        agent=agent_loader.agent,
                        thread_id=state["thread_id"],
                        workspace_dir=state["workspace_dir"],
                        checkpointer=checkpointer,
                        append_system=lambda t, s="dim": console.print(t, style=s),
                        start_new_session_cb=_on_start_new_session,
                        handle_session_resume_cb=_on_handle_session_resume,
                        await_agent_ready=_await_agent_ready,
                        on_cmd_completed=_on_channel_cmd_completed,
                        channel_runtime=channel_runtime,
                    )
                    if _slash_handled:
                        _print_separator()
                        sys.stdout.write("\033[34;1m❯\033[0m ")
                        sys.stdout.flush()
                        return

                    try:
                        ready_agent = await _await_agent_ready()
                        meta = build_metadata(state["workspace_dir"], model)
                        await _refresh_status_snapshot(
                            msg.content, reset_streaming_text=True
                        )
                        response = run_streaming(
                            ui_backend=state["ui_backend"],
                            agent=ready_agent,
                            message=msg.content,
                            thread_id=state["thread_id"],
                            show_thinking=show_thinking,
                            interactive=True,
                            metadata=meta,
                            on_thinking=_send_thinking_to_channel,
                            on_todo=_send_todo_to_channel,
                            on_file_write=_send_media_to_channel,
                            hitl_prompt_fn=_channel_hitl_prompt,
                            ask_user_prompt_fn=_channel_ask_user,
                            on_stream_event=_handle_stream_status_event,
                            status_footer_builder=_stream_status_footer,
                            cancel_scope=_ch_mod._channel_message_cancel_scope(msg),
                        )
                    except Exception as e:
                        response = f"Error: {e}"
                        console.print(f"[red]Channel error: {e}[/red]")

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

            async def _check_channel_queue() -> None:
                """Poll the channel message queue and dispatch to the agent."""
                while True:
                    try:
                        msg = _message_queue.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(0.1)
                        continue
                    await _process_channel_message(msg)

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
                                workspace_dir=state["workspace_dir"],
                                checkpointer=checkpointer,
                                config=config,
                                input_tokens_hint=state.get("status_last_input_tokens"),
                                channel_runtime=channel_runtime,
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
                        if user_input.lstrip().startswith("/"):
                            bad_cmd = user_input.split(None, 1)[0]
                            console.print(
                                f"[red]Unknown command:[/red] {escape(bad_cmd)}"
                            )
                            console.print(
                                "[dim]Type /help to see available commands.[/dim]"
                            )
                            console.print()
                            continue

                        # Resolve @file mentions — inject file contents inline
                        _, message_to_send, file_warnings = resolve_file_mentions(
                            user_input, state["workspace_dir"]
                        )

                        # Stream agent response with metadata for persistence
                        # Warnings printed here so they appear just before the
                        # model response, not before the user input echo.
                        for w in file_warnings:
                            console.print(f"[yellow]⚠ {escape(w)}[/yellow]")
                        console.print()
                        ready_agent = await _await_agent_ready()
                        meta = build_metadata(state["workspace_dir"], model)
                        await _refresh_status_snapshot(
                            message_to_send, reset_streaming_text=True
                        )
                        run_streaming(
                            ui_backend=state["ui_backend"],
                            agent=ready_agent,
                            message=message_to_send,
                            thread_id=state["thread_id"],
                            show_thinking=show_thinking,
                            interactive=True,
                            metadata=meta,
                            on_stream_event=_handle_stream_status_event,
                            status_footer_builder=_stream_status_footer,
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
                # Best-effort: guard so a DB lookup failure here can't
                # shadow the original exception exiting _async_main_loop.
                current_tid = state.get("thread_id")
                if current_tid:
                    try:
                        if await thread_exists(current_tid):
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
    agent: Any,
    prompt: str,
    thread_id: str | None = None,
    show_thinking: bool = True,
    workspace_dir: str | None = None,
    model: str | None = None,
    ui_backend: str = "cli",
) -> None:
    """Single-shot execution with streaming display.

    Args:
        agent: Compiled agent graph
        prompt: User prompt
        thread_id: Optional thread ID (generates new one if None)
        show_thinking: Whether to display thinking panels
        workspace_dir: Per-session workspace directory path
        model: Model name for checkpoint metadata
        ui_backend: UI backend ('cli' or 'tui')
    """
    thread_id = thread_id or generate_thread_id()

    width = console.size.width
    sep = Text("\u2500" * width, style="dim")
    console.print(sep)
    console.print(Text(f"> {prompt}"))
    console.print(sep)
    console.print(f"[dim]Thread: {thread_id}[/dim]")
    if workspace_dir:
        console.print(f"[dim]Workspace: {_shorten_path(workspace_dir)}[/dim]")
    console.print()

    meta = build_metadata(workspace_dir, model)
    try:
        run_streaming(
            ui_backend=resolve_ui_backend(ui_backend, warn_fallback=True),
            agent=agent,
            message=prompt,
            thread_id=thread_id,
            show_thinking=show_thinking,
            interactive=False,
            metadata=meta,
        )
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
            raise
