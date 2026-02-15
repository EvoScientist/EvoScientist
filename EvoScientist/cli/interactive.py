"""Interactive CLI mode and single-shot execution."""

import asyncio
import os
import sys
from datetime import datetime, timezone
from typing import Any

import typer  # type: ignore[import-untyped]
from prompt_toolkit import PromptSession  # type: ignore[import-untyped]
from prompt_toolkit.completion import Completer, Completion  # type: ignore[import-untyped]
from prompt_toolkit.history import FileHistory  # type: ignore[import-untyped]
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory  # type: ignore[import-untyped]
from prompt_toolkit.formatted_text import HTML  # type: ignore[import-untyped]
from prompt_toolkit.shortcuts import CompleteStyle  # type: ignore[import-untyped]
from prompt_toolkit.styles import Style as PtStyle  # type: ignore[import-untyped]
from rich.table import Table
from rich.text import Text

from ..sessions import (
    generate_thread_id,
    get_checkpointer,
    list_threads,
    thread_exists,
    find_similar_threads,
    delete_thread,
    get_thread_metadata,
    get_thread_messages,
    _format_relative_time,
    AGENT_NAME,
)
from ..stream.display import console, _run_streaming
from .agent import _shorten_path, _create_session_workspace, _load_agent
from .channel import (
    _channels_is_running,
    _cmd_channel,
    _cmd_channel_stop,
    _auto_start_channel,
)
import EvoScientist.cli.channel as _ch_mod
from .mcp_ui import _cmd_mcp
from .skills_cmd import _cmd_list_skills, _cmd_install_skill, _cmd_uninstall_skill


# =============================================================================
# Banner
# =============================================================================

EVOSCIENTIST_ASCII_LINES = [
    r" \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557 \u2588\u2588\u2557   \u2588\u2588\u2557  \u2588\u2588\u2588\u2588\u2588\u2588\u2557  \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557  \u2588\u2588\u2588\u2588\u2588\u2588\u2557 \u2588\u2588\u2557 \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557 \u2588\u2588\u2588\u2557   \u2588\u2588\u2557 \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557 \u2588\u2588\u2557 \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557 \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557",
]

# Blue gradient: deep navy -> royal blue -> sky blue -> cyan
_GRADIENT_COLORS = ["#1a237e", "#1565c0", "#1e88e5", "#42a5f5", "#64b5f6", "#90caf9"]

# Keep the real ASCII art lines (raw strings) rather than the escaped version above
_REAL_ASCII_LINES = [
    r" ███████╗ ██╗   ██╗  ██████╗  ███████╗  ██████╗ ██╗ ███████╗ ███╗   ██╗ ████████╗ ██╗ ███████╗ ████████╗",
    r" ██╔════╝ ██║   ██║ ██╔═══██╗ ██╔════╝ ██╔════╝ ██║ ██╔════╝ ████╗  ██║ ╚══██╔══╝ ██║ ██╔════╝ ╚══██╔══╝",
    r" █████╗   ██║   ██║ ██║   ██║ ███████╗ ██║      ██║ █████╗   ██╔██╗ ██║    ██║    ██║ ███████╗    ██║   ",
    r" ██╔══╝   ╚██╗ ██╔╝ ██║   ██║ ╚════██║ ██║      ██║ ██╔══╝   ██║╚██╗██║    ██║    ██║ ╚════██║    ██║   ",
    r" ███████╗  ╚████╔╝  ╚██████╔╝ ███████║ ╚██████╗ ██║ ███████╗ ██║ ╚████║    ██║    ██║ ███████║    ██║   ",
    r" ╚══════╝   ╚═══╝    ╚═════╝  ╚══════╝  ╚═════╝ ╚═╝ ╚══════╝ ╚═╝  ╚═══╝    ╚═╝    ╚═╝ ╚══════╝    ╚═╝   ",
]


def print_banner(
    thread_id: str,
    workspace_dir: str | None = None,
    memory_dir: str | None = None,
    mode: str | None = None,
    model: str | None = None,
    provider: str | None = None,
):
    """Print welcome banner with ASCII art logo, thread ID, workspace path, and mode."""
    for line, color in zip(_REAL_ASCII_LINES, _GRADIENT_COLORS):
        console.print(Text(line, style=f"{color} bold"))
    info = Text()
    if model or provider or mode:
        info.append("  ", style="dim")
        parts = []
        if model:
            parts.append(("Model: ", model))
        if provider:
            parts.append(("Provider: ", provider))
        if mode:
            parts.append(("Mode: ", mode))
        for i, (label, value) in enumerate(parts):
            if i > 0:
                info.append("  ", style="dim")
            info.append(label, style="dim")
            info.append(value, style="magenta")
    info.append("\n  Type ", style="#ffe082")
    info.append("/", style="#ffe082 bold")
    info.append(" for commands", style="#ffe082")
    console.print(info)
    console.print()


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
    ("/mcp", "Manage MCP servers"),
    ("/channel", "Configure messaging channels"),
    ("/exit", "Quit EvoScientist"),
]

_COMPLETION_STYLE = PtStyle.from_dict({
    "completion-menu": "bg:default noreverse nounderline noitalic",
    "completion-menu.completion": "bg:default #888888 noreverse",
    "completion-menu.completion.current": "bg:default default bold noreverse",
    "completion-menu.meta.completion": "bg:default #888888 noreverse",
    "completion-menu.meta.completion.current": "bg:default default bold noreverse",
    "scrollbar.background": "bg:default",
    "scrollbar.button": "bg:default",
})

# Style for questionary pickers — matches _COMPLETION_STYLE visual language:
# gray (#888888) for non-selected, bold for selected, no background changes.
_PICKER_STYLE = PtStyle.from_dict({
    "questionmark": "#888888",
    "question": "",
    "pointer": "bold",
    "highlighted": "bold",
    "text": "#888888",
    "answer": "bold",
})


class SlashCommandCompleter(Completer):
    """Autocomplete for slash commands — triggers when input starts with '/'."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
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


def _build_metadata(workspace_dir: str | None, model: str | None) -> dict:
    """Build metadata dict for LangGraph checkpoint persistence."""
    return {
        "agent_name": AGENT_NAME,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "workspace_dir": workspace_dir or "",
        "model": model or "",
    }


def cmd_interactive(
    show_thinking: bool = True,
    workspace_dir: str | None = None,
    workspace_fixed: bool = False,
    mode: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    run_name: str | None = None,
    thread_id: str | None = None,
) -> None:
    """Interactive conversation mode with streaming output.

    The persistent ``AsyncSqliteSaver`` checkpointer is opened here and
    shared for the entire interactive session lifetime.

    Args:
        show_thinking: Whether to display thinking panels
        workspace_dir: Per-session workspace directory path
        workspace_fixed: If True, /new keeps the same workspace directory
        mode: Workspace mode ('daemon' or 'run'), displayed in banner
        model: Model name to display in banner
        provider: LLM provider name to display in banner
        run_name: Optional run name for /new session deduplication
        thread_id: Optional thread ID to resume a previous session
    """
    import nest_asyncio
    nest_asyncio.apply()

    from .. import paths
    memory_dir = str(paths.MEMORY_DIR)

    history_file = str(os.path.expanduser("~/.EvoScientist_history"))
    session = PromptSession(
        history=FileHistory(history_file),
        auto_suggest=AutoSuggestFromHistory(),
        completer=SlashCommandCompleter(),
        complete_style=CompleteStyle.COLUMN,
        complete_while_typing=True,
        style=_COMPLETION_STYLE,
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
    }

    async def _resolve_thread_id(tid: str) -> str | None:
        """Resolve a (possibly partial) thread ID. Returns full ID or None."""
        if await thread_exists(tid):
            return tid
        similar = await find_similar_threads(tid)
        if len(similar) == 1:
            return similar[0]
        if len(similar) > 1:
            console.print(f"[yellow]Ambiguous thread ID '{tid}'. Matches:[/yellow]")
            for s in similar:
                console.print(f"  [cyan]{s}[/cyan]")
            return None
        console.print(f"[red]Thread '{tid}' not found.[/red]")
        return None

    async def _cmd_threads():
        """Handle /threads command — show recent sessions."""
        threads = await list_threads(
            limit=0, include_message_count=True, include_preview=True,
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
        console.print("[dim]  /resume[/dim] to continue a session  [dim]/delete <id>[/dim] to remove  [dim]/new[/dim] to start fresh")
        console.print()

    async def _render_history(thread_id: str):
        """Display a compact conversation history for a resumed session."""
        messages = await get_thread_messages(thread_id)
        if not messages:
            return

        MAX_CONTENT_LEN = 200  # truncate long messages

        def _truncate(text: str) -> str:
            text = text.strip()
            if len(text) <= MAX_CONTENT_LEN:
                return text
            return text[:MAX_CONTENT_LEN] + "..."

        console.print("[dim]── Conversation history ──[/dim]")
        for msg in messages:
            msg_type = getattr(msg, "type", None)
            content = getattr(msg, "content", "") or ""
            # content can be a list of blocks (multimodal) — extract text
            if isinstance(content, list):
                parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
                content = " ".join(parts) if parts else ""

            if msg_type == "human":
                console.print(Text.assemble(
                    ("\u276f ", "bold blue"),
                    (_truncate(content), ""),
                ))
            elif msg_type == "ai":
                tool_calls = getattr(msg, "tool_calls", None) or []
                if content:
                    console.print(Text(_truncate(content), style="dim"))
                if tool_calls:
                    names = [tc.get("name", "?") for tc in tool_calls]
                    console.print(Text(
                        f"  \u25b6 {', '.join(names)}",
                        style="dim italic",
                    ))
            # Skip tool messages — they are verbose and not useful in replay

        console.print("[dim]── End of history ──[/dim]")
        console.print()

    async def _cmd_resume(arg: str, checkpointer):
        """Handle /resume [id] — resume a previous session."""
        if not arg:
            # Show interactive session picker with conversation previews
            threads = await list_threads(
                limit=0, include_message_count=True, include_preview=True,
            )
            if not threads:
                console.print("[yellow]No sessions to resume.[/yellow]")
                return

            import questionary

            choices = []
            # Display-width-aware padding (CJK chars take 2 columns)
            import unicodedata

            def _display_width(s: str) -> int:
                w = 0
                for ch in s:
                    w += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
                return w

            def _pad_to_width(s: str, target: int) -> str:
                pad = target - _display_width(s)
                return s + " " * max(pad, 2)

            lefts = [t.get("preview", "") or t["thread_id"] for t in threads]
            col_width = max(_display_width(s) for s in lefts) + 4
            for t, left_text in zip(threads, lefts):
                tid = t["thread_id"]
                when = _format_relative_time(t.get("updated_at"))
                label = f"{_pad_to_width(left_text, col_width)}({tid} {when})"
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
        state["agent"] = _load_agent(workspace_dir=state["workspace_dir"], checkpointer=checkpointer)
        # Sync shared refs if channel is running
        if _channels_is_running():
            _ch_mod._cli_agent = state["agent"]
            _ch_mod._cli_thread_id = state["thread_id"]
        console.print(f"[green]Resumed session:[/green] [yellow]{resolved}[/yellow]")
        if state["workspace_dir"]:
            console.print(f"[dim]Workspace:[/dim] [cyan]{_shorten_path(state['workspace_dir'])}[/cyan]")
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
            state["agent"] = _load_agent(workspace_dir=state["workspace_dir"], checkpointer=checkpointer)

            # Print banner
            if state["resumed"]:
                print_banner(state["thread_id"], state["workspace_dir"], memory_dir, mode, model, provider)
                console.print(f"[green]Resumed session [yellow]{state['thread_id']}[/yellow][/green]\n")
            else:
                print_banner(state["thread_id"], state["workspace_dir"], memory_dir, mode, model, provider)

            # Start background queue checker
            # (no longer needed — bus mode handles messages internally)

            # Auto-start channel if enabled in config
            from ..config import load_config
            config = load_config()
            if config and config.channel_enabled and not _channels_is_running():
                _auto_start_channel(state["agent"], state["thread_id"], config)

            try:
                _print_separator()
                while state["running"]:
                    try:
                        user_input = await session.prompt_async(
                            HTML('<ansiblue><b>\u276f</b></ansiblue> ')
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
                            arg = user_input[len("/resume"):].strip()
                            await _cmd_resume(arg, checkpointer)
                            continue

                        if user_input.lower().startswith("/delete"):
                            arg = user_input[len("/delete"):].strip()
                            await _cmd_delete(arg)
                            continue

                        if user_input.lower() == "/new":
                            # New session: new thread; workspace only changes if not fixed
                            if not workspace_fixed:
                                state["workspace_dir"] = _create_session_workspace(run_name)
                            console.print("[dim]Loading new session...[/dim]")
                            state["agent"] = _load_agent(workspace_dir=state["workspace_dir"], checkpointer=checkpointer)
                            state["thread_id"] = generate_thread_id()
                            state["resumed"] = False
                            console.print(f"[green]New session:[/green] [yellow]{state['thread_id']}[/yellow]")
                            if state["workspace_dir"]:
                                console.print(f"[dim]Workspace:[/dim] [cyan]{_shorten_path(state['workspace_dir'])}[/cyan]\n")
                            continue

                        if user_input.lower() == "/current":
                            console.print(f"[dim]Thread:[/dim] [yellow]{state['thread_id']}[/yellow]")
                            if state["workspace_dir"]:
                                console.print(f"[dim]Workspace:[/dim] [cyan]{_shorten_path(state['workspace_dir'])}[/cyan]")
                            if memory_dir:
                                console.print(f"[dim]Memory dir:[/dim] [cyan]{_shorten_path(memory_dir)}[/cyan]")
                            console.print()
                            continue

                        if user_input.lower() == "/skills":
                            _cmd_list_skills()
                            continue

                        if user_input.lower().startswith("/install-skill"):
                            source = user_input[len("/install-skill"):].strip()
                            _cmd_install_skill(source)
                            continue

                        if user_input.lower().startswith("/uninstall-skill"):
                            name = user_input[len("/uninstall-skill"):].strip()
                            _cmd_uninstall_skill(name)
                            continue

                        if user_input.lower().startswith("/mcp"):
                            _cmd_mcp(user_input[4:])
                            continue

                        if user_input.lower().startswith("/channel"):
                            args = user_input[len("/channel"):].strip()
                            if args.lower().startswith("stop"):
                                stop_arg = args[len("stop"):].strip()
                                _cmd_channel_stop(stop_arg or None)
                            else:
                                _cmd_channel(args, state["agent"], state["thread_id"])
                            continue

                        # Stream agent response with metadata for persistence
                        console.print()
                        meta = _build_metadata(state["workspace_dir"], model)
                        _run_streaming(
                            state["agent"], user_input, state["thread_id"],
                            show_thinking, interactive=True, metadata=meta,
                        )
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
                        if "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
                            console.print("[red]Error: API key not configured.[/red]")
                            console.print("[dim]Run [bold]EvoSci onboard[/bold] to set up your API key.[/dim]")
                            state["running"] = False
                            break
                        else:
                            console.print(f"[red]Error: {e}[/red]")
            finally:
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
) -> None:
    """Single-shot execution with streaming display.

    Args:
        agent: Compiled agent graph
        prompt: User prompt
        thread_id: Optional thread ID (generates new one if None)
        show_thinking: Whether to display thinking panels
        workspace_dir: Per-session workspace directory path
        model: Model name for checkpoint metadata
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

    meta = _build_metadata(workspace_dir, model)
    try:
        _run_streaming(agent, prompt, thread_id, show_thinking, interactive=False, metadata=meta)
    except Exception as e:
        error_msg = str(e)
        if "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
            console.print("[red]Error: API key not configured.[/red]")
            console.print("[dim]Run [bold]EvoSci onboard[/bold] to set up your API key.[/dim]")
            raise typer.Exit(1)
        else:
            console.print(f"[red]Error: {e}[/red]")
            raise
