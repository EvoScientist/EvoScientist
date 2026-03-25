"""Interactive CLI mode and single-shot execution."""

import asyncio
import logging
import queue
import random
import sys
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
from prompt_toolkit.shortcuts import CompleteStyle  # type: ignore[import-untyped]
from prompt_toolkit.styles import Style as PtStyle  # type: ignore[import-untyped]
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import EvoScientist.cli.channel as _ch_mod

from ..sessions import (
    _format_relative_time,
    delete_thread,
    find_similar_threads,
    generate_thread_id,
    get_checkpointer,
    get_thread_messages,
    get_thread_metadata,
    list_threads,
    thread_exists,
)
from ..stream.display import console
from ._constants import LOGO_GRADIENT, LOGO_LINES, WELCOME_SLOGANS, build_metadata
from .agent import _create_session_workspace, _load_agent, _shorten_path
from .channel import (
    ChannelMessage,
    _auto_start_channel,
    _channels_is_running,
    _cmd_channel,
    _cmd_channel_stop,
    _message_queue,
    _set_channel_response,
)
from .file_mentions import complete_file_mention, resolve_file_mentions
from .mcp_ui import _cmd_mcp
from .skills_cmd import (
    _cmd_install_skill,
    _cmd_install_skills,
    _cmd_list_skills,
    _cmd_uninstall_skill,
)
from .tui_interactive import run_textual_interactive
from .tui_runtime import resolve_ui_backend, run_streaming

_channel_logger = logging.getLogger(__name__)


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
    info.append(" \u2022 Ctrl+C ", style="#ffe082")
    info.append("interrupt", style="#ffe082 bold")
    console.print(info)


# =============================================================================
# Slash-command completer
# =============================================================================

_SLASH_COMMANDS = [
    ("/current", "Show current session info"),
    ("/threads", "List recent sessions"),
    ("/resume", "Resume a previous session (prefix match)"),
    ("/delete", "Delete a saved session"),
    ("/new", "Start a new session"),
    ("/skills", "List installed skills"),
    ("/install-skill", "Add a skill from path or GitHub"),
    ("/uninstall-skill", "Remove an installed skill"),
    ("/evoskills", "Browse and install EvoSkills (optional: /evoskills <tag>)"),
    ("/mcp", "Manage MCP servers"),
    ("/channel", "Configure messaging channels"),
    ("/compact", "Compact conversation to free context"),
    ("/exit", "Quit EvoScientist"),
]

_COMPLETION_STYLE = PtStyle.from_dict(
    {
        "completion-menu": "bg:default noreverse nounderline noitalic",
        "completion-menu.completion": "bg:default #888888 noreverse",
        "completion-menu.completion.current": "bg:default default bold noreverse",
        "completion-menu.meta.completion": "bg:default #888888 noreverse",
        "completion-menu.meta.completion.current": "bg:default default bold noreverse",
        "scrollbar.background": "bg:default",
        "scrollbar.button": "bg:default",
    }
)

# Style for questionary pickers — matches _COMPLETION_STYLE visual language:
# gray (#888888) for non-selected, bold for selected, no background changes.
_PICKER_STYLE = PtStyle.from_dict(
    {
        "questionmark": "#888888",
        "question": "",
        "pointer": "bold",
        "highlighted": "bold",
        "text": "#888888",
        "answer": "bold",
    }
)


class SlashCommandCompleter(Completer):
    """Autocomplete for slash commands and ``@file`` mentions."""

    def __init__(self, workspace_dir: str | None = None) -> None:
        self._workspace_dir = workspace_dir

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        # @file mention completion
        if "@" in text:
            candidates = complete_file_mention(text, self._workspace_dir)
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
        for cmd, desc in _SLASH_COMMANDS:
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

    memory_dir = str(paths.MEMORY_DIR)

    from ..config.settings import get_config_dir

    config_dir = get_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    history_file = str(config_dir / "history")

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
        completer=SlashCommandCompleter(workspace_dir=workspace_dir),
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
        "agent": None,
        "thread_id": thread_id or generate_thread_id(),
        "workspace_dir": workspace_dir,
        "running": True,
        "resumed": False,
        "ui_backend": resolved_ui_backend,
    }

    async def _resolve_thread_id(tid: str) -> str | None:
        """Resolve a (possibly partial) thread ID. Returns full ID or None."""
        if await thread_exists(tid):
            return tid
        similar = await find_similar_threads(tid)
        if len(similar) == 1:
            return similar[0]
        if len(similar) > 1:
            console.print(
                f"[yellow]Ambiguous thread ID '{escape(tid)}'. Matches:[/yellow]"
            )
            for s in similar:
                console.print(f"  [cyan]{s}[/cyan]")
            return None
        console.print(f"[red]Thread '{escape(tid)}' not found.[/red]")
        return None

    async def _cmd_threads():
        """Handle /threads command — show recent sessions."""
        threads = await list_threads(
            limit=0,
            include_message_count=True,
            include_preview=True,
        )
        if not threads:
            console.print("[yellow]No saved sessions.[/yellow]")
            return
        table = Table(title="Sessions", show_header=True, header_style="bold cyan")
        table.add_column("ID", style="bold")
        table.add_column("Preview", style="dim", max_width=50, no_wrap=True)
        table.add_column("Messages", justify="right")
        table.add_column("Model", style="dim")
        table.add_column("Last Used", style="dim")
        for t in threads:
            tid = t["thread_id"]
            marker = " *" if tid == state["thread_id"] else ""
            table.add_row(
                f"{tid}{marker}",
                t.get("preview", "") or "",
                str(t.get("message_count", 0)),
                t.get("model", "") or "",
                _format_relative_time(t.get("updated_at")),
            )
        console.print()
        console.print(table)
        console.print(
            "[dim]  /resume[/dim] to continue a session  [dim]/delete <id>[/dim] to remove  [dim]/new[/dim] to start fresh"
        )
        console.print()

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

    async def _cmd_resume(arg: str, checkpointer):
        """Handle /resume [id] — resume a previous session."""
        if not arg:
            # Show interactive session picker with conversation previews
            threads = await list_threads(
                limit=0,
                include_message_count=True,
                include_preview=True,
            )
            if not threads:
                console.print("[yellow]No sessions to resume.[/yellow]")
                return

            import questionary

            from .widgets.thread_selector import _build_items

            choices = []
            items = _build_items(threads)
            for item in items:
                if item["type"] == "header":
                    choices.append(
                        questionary.Separator(
                            f"\u2500\u2500 \U0001f4c2 {item['label']}"
                        )
                    )
                elif item["type"] == "subheader":
                    choices.append(questionary.Separator(f"   {item['label']}"))
                else:
                    t = item["thread"]
                    tid = t["thread_id"]
                    preview = t.get("preview", "") or ""
                    msgs = t.get("message_count", 0)
                    model = t.get("model", "") or ""
                    when = _format_relative_time(t.get("updated_at"))
                    indent = "    " if item.get("indented") else "  "
                    parts = [f"{indent}{tid}"]
                    if preview:
                        parts.append(
                            preview[:40] + "\u2026" if len(preview) > 40 else preview
                        )
                    parts.append(f"({msgs} msgs)")
                    if model:
                        parts.append(model)
                    if when:
                        parts.append(when)
                    label = "  ".join(parts)
                    choices.append(questionary.Choice(title=label, value=tid))

            from prompt_toolkit.layout.dimension import Dimension
            from questionary.prompts.common import InquirerControl

            prompt = questionary.select(
                "Select session to resume:",
                choices=choices,
                style=_PICKER_STYLE,
            )
            # Limit visible list to 10 rows with scrolling
            for window in prompt.application.layout.find_all_windows():
                if isinstance(window.content, InquirerControl):
                    window.height = Dimension(max=10)
                    break
            selected = prompt.ask()

            if selected is None:
                return
            arg = selected

        resolved = await _resolve_thread_id(arg)
        if not resolved:
            return

        meta = await get_thread_metadata(resolved)
        ws = (meta or {}).get("workspace_dir", "") or state["workspace_dir"]

        state["thread_id"] = resolved
        state["resumed"] = True
        if ws:
            state["workspace_dir"] = ws
        console.print("[dim]Loading session...[/dim]")
        state["agent"] = _load_agent(
            workspace_dir=state["workspace_dir"],
            checkpointer=checkpointer,
            config=config,
        )
        # Sync shared refs if channel is running
        if _channels_is_running():
            _ch_mod._cli_agent = state["agent"]
            _ch_mod._cli_thread_id = state["thread_id"]
        console.print(f"[green]Resumed session:[/green] [yellow]{resolved}[/yellow]")
        if state["workspace_dir"]:
            console.print(
                f"[dim]Workspace:[/dim] [cyan]{_shorten_path(state['workspace_dir'])}[/cyan]"
            )
        console.print()
        await _render_history(resolved)

    async def _cmd_delete(arg: str):
        """Handle /delete <id> — delete a saved session."""
        if not arg:
            console.print("[red]Usage: /delete <thread-id>[/red]")
            return
        resolved = await _resolve_thread_id(arg)
        if not resolved:
            return
        if resolved == state["thread_id"]:
            console.print("[red]Cannot delete the current session.[/red]")
            return
        deleted = await delete_thread(resolved)
        if deleted:
            console.print(f"[green]Deleted session {resolved}.[/green]")
        else:
            console.print(f"[red]Session {resolved} not found.[/red]")

    async def _async_main_loop():
        """Async main loop with prompt_async and channel queue checking."""
        async with get_checkpointer() as checkpointer:
            # Handle --thread-id resume
            if thread_id:
                resolved = await _resolve_thread_id(thread_id)
                if resolved:
                    meta = await get_thread_metadata(resolved)
                    ws = (meta or {}).get("workspace_dir", "") or state["workspace_dir"]
                    state["thread_id"] = resolved
                    state["resumed"] = True
                    if ws:
                        state["workspace_dir"] = ws

            console.print("[dim]Loading agent...[/dim]")
            state["agent"] = _load_agent(
                workspace_dir=state["workspace_dir"],
                checkpointer=checkpointer,
                config=config,
            )

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

                def _channel_hitl_prompt(action_requests: list) -> list[dict] | None:
                    """Send HITL approval prompt to channel user and wait for reply."""
                    return _ch_mod.channel_hitl_prompt(action_requests, msg)

                def _channel_ask_user(ask_user_data: dict) -> dict:
                    """Send ask_user questions to channel user and wait for reply."""
                    return _ch_mod.channel_ask_user_prompt(ask_user_data, msg)

                meta = build_metadata(state["workspace_dir"], model)
                try:
                    response = run_streaming(
                        ui_backend=state["ui_backend"],
                        agent=state["agent"],
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
                    )
                except Exception as e:
                    response = f"Error: {e}"
                    console.print(f"[red]Channel error: {e}[/red]")

                _set_channel_response(msg.msg_id, response)

                tx = Text()
                tx.append(f"[{msg.channel_type}: Replied to ", style="dim")
                tx.append(msg.sender, style="cyan")
                tx.append("]", style="dim")
                console.print(tx)
                _print_separator()

                # Redraw the ❯ prompt on a new line after separator
                sys.stdout.write("\033[34;1m\u276f\033[0m ")
                sys.stdout.flush()

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

            # Auto-start channel if enabled in config
            from ..config import load_config

            _channel_cfg = load_config()
            if (
                _channel_cfg
                and _channel_cfg.channel_enabled
                and not _channels_is_running()
            ):
                _auto_start_channel(
                    state["agent"],
                    state["thread_id"],
                    _channel_cfg,
                    send_thinking=channel_send_thinking,
                )

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
                        user_input = await session.prompt_async(
                            HTML("<ansiblue><b>\u276f</b></ansiblue> ")
                        )
                        user_input = user_input.strip()

                        if not user_input:
                            # Erase the empty prompt line so it looks like nothing happened
                            sys.stdout.write("\033[A\033[2K\r")
                            sys.stdout.flush()
                            continue

                        _print_separator()

                        # Special commands
                        if user_input.lower() in ("/exit", "/quit", "/q"):
                            console.print("[dim]Goodbye![/dim]")
                            state["running"] = False
                            break

                        if user_input.lower() == "/threads":
                            await _cmd_threads()
                            continue

                        if user_input.lower().startswith("/resume"):
                            arg = user_input[len("/resume") :].strip()
                            await _cmd_resume(arg, checkpointer)
                            continue

                        if user_input.lower().startswith("/delete"):
                            arg = user_input[len("/delete") :].strip()
                            await _cmd_delete(arg)
                            continue

                        if user_input.lower() == "/new":
                            # New session: new thread; workspace only changes if not fixed
                            if not workspace_fixed:
                                state["workspace_dir"] = _create_session_workspace(
                                    run_name
                                )
                            console.print("[dim]Loading new session...[/dim]")
                            state["agent"] = _load_agent(
                                workspace_dir=state["workspace_dir"],
                                checkpointer=checkpointer,
                                config=config,
                            )
                            state["thread_id"] = generate_thread_id()
                            state["resumed"] = False
                            # Sync channel refs so the queue checker uses the new agent
                            if _channels_is_running():
                                _ch_mod._cli_agent = state["agent"]
                                _ch_mod._cli_thread_id = state["thread_id"]
                            console.print(
                                f"[green]New session:[/green] [yellow]{state['thread_id']}[/yellow]"
                            )
                            if state["workspace_dir"]:
                                console.print(
                                    f"[dim]Workspace:[/dim] [cyan]{_shorten_path(state['workspace_dir'])}[/cyan]\n"
                                )
                            continue

                        if user_input.lower() == "/current":
                            console.print(
                                f"[dim]Thread:[/dim] [yellow]{state['thread_id']}[/yellow]"
                            )
                            if state["workspace_dir"]:
                                console.print(
                                    f"[dim]Workspace:[/dim] [cyan]{_shorten_path(state['workspace_dir'])}[/cyan]"
                                )
                            if memory_dir:
                                console.print(
                                    f"[dim]Memory dir:[/dim] [cyan]{_shorten_path(memory_dir)}[/cyan]"
                                )
                            console.print(
                                f"[dim]UI:[/dim] [cyan]{state['ui_backend']}[/cyan]"
                            )
                            console.print()
                            continue

                        if user_input.lower() == "/skills":
                            _cmd_list_skills()
                            continue

                        if user_input.lower().startswith("/install-skill"):
                            source = user_input[len("/install-skill") :].strip()
                            _cmd_install_skill(source)
                            continue

                        if user_input.lower().startswith("/uninstall-skill"):
                            name = user_input[len("/uninstall-skill") :].strip()
                            _cmd_uninstall_skill(name)
                            continue

                        if user_input.lower().startswith("/evoskills"):
                            browse_args = user_input[len("/evoskills") :].strip()
                            _cmd_install_skills(browse_args)
                            continue

                        if user_input.lower().startswith("/mcp"):
                            _cmd_mcp(user_input[len("/mcp") :])
                            continue

                        if user_input.lower().startswith("/channel"):
                            args = user_input[len("/channel") :].strip()
                            if args.lower().startswith("stop"):
                                stop_arg = args[len("stop") :].strip()
                                _cmd_channel_stop(stop_arg or None)
                            else:
                                _cmd_channel(
                                    args,
                                    state["agent"],
                                    state["thread_id"],
                                    send_thinking=channel_send_thinking,
                                )
                            continue

                        if user_input.lower() == "/compact":
                            from .commands import (
                                compact_conversation,
                                render_compact_result,
                            )

                            with console.status(
                                "[cyan]Compacting conversation...[/cyan]"
                            ):
                                result = await compact_conversation(
                                    agent=state["agent"],
                                    thread_id=state["thread_id"],
                                )
                            console.print(render_compact_result(result))
                            continue

                        # Resolve @file mentions — inject file contents inline
                        _, message_to_send, file_warnings = resolve_file_mentions(
                            user_input, state["workspace_dir"]
                        )

                        # Stream agent response with metadata for persistence
                        # Warnings printed here so they appear just before the
                        # model response, not before the user input echo.
                        for w in file_warnings:
                            console.print(f"[yellow]⚠ {w}[/yellow]")
                        console.print()
                        meta = build_metadata(state["workspace_dir"], model)
                        run_streaming(
                            ui_backend=state["ui_backend"],
                            agent=state["agent"],
                            message=message_to_send,
                            thread_id=state["thread_id"],
                            show_thinking=show_thinking,
                            interactive=True,
                            metadata=meta,
                        )
                        console.print()
                        _print_separator()

                    except KeyboardInterrupt:
                        console.print("\n[dim]Goodbye![/dim]")
                        state["running"] = False
                        break
                    except EOFError:
                        # Handle Ctrl+D
                        console.print("\n[dim]Goodbye![/dim]")
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

    # Run the async main loop
    try:
        asyncio.run(_async_main_loop())
    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye![/dim]")


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
