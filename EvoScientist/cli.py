"""
EvoScientist Agent CLI

Command-line interface with streaming output for the EvoScientist research agent.

Features:
- Thinking panel (blue) - shows model reasoning
- Tool calls with status indicators (green/yellow/red dots)
- Tool results in tree format with folding
- Response panel (green) - shows final response
- Thread ID support for multi-turn conversations
- Interactive mode with prompt_toolkit
"""

import os
import sys
import uuid
from datetime import datetime
from typing import Any, Optional

import typer  # type: ignore[import-untyped]
from prompt_toolkit import PromptSession  # type: ignore[import-untyped]
from prompt_toolkit.history import FileHistory  # type: ignore[import-untyped]
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory  # type: ignore[import-untyped]
from prompt_toolkit.formatted_text import HTML  # type: ignore[import-untyped]
from rich.text import Text  # type: ignore[import-untyped]

# Backward-compat re-exports (tests import these from EvoScientist.cli)
from .stream.state import SubAgentState, StreamState, _parse_todo_items, _build_todo_stats  # noqa: F401
from .stream.display import console, _run_streaming


# =============================================================================
# Banner
# =============================================================================

EVOSCIENTIST_ASCII_LINES = [
    r" ███████╗ ██╗   ██╗  ██████╗  ███████╗  ██████╗ ██╗ ███████╗ ███╗   ██╗ ████████╗ ██╗ ███████╗ ████████╗",
    r" ██╔════╝ ██║   ██║ ██╔═══██╗ ██╔════╝ ██╔════╝ ██║ ██╔════╝ ████╗  ██║ ╚══██╔══╝ ██║ ██╔════╝ ╚══██╔══╝",
    r" █████╗   ██║   ██║ ██║   ██║ ███████╗ ██║      ██║ █████╗   ██╔██╗ ██║    ██║    ██║ ███████╗    ██║   ",
    r" ██╔══╝   ╚██╗ ██╔╝ ██║   ██║ ╚════██║ ██║      ██║ ██╔══╝   ██║╚██╗██║    ██║    ██║ ╚════██║    ██║   ",
    r" ███████╗  ╚████╔╝  ╚██████╔╝ ███████║ ╚██████╗ ██║ ███████╗ ██║ ╚████║    ██║    ██║ ███████║    ██║   ",
    r" ╚══════╝   ╚═══╝    ╚═════╝  ╚══════╝  ╚═════╝ ╚═╝ ╚══════╝ ╚═╝  ╚═══╝    ╚═╝    ╚═╝ ╚══════╝    ╚═╝   ",
]

# Blue gradient: deep navy -> royal blue -> sky blue -> cyan
_GRADIENT_COLORS = ["#1a237e", "#1565c0", "#1e88e5", "#42a5f5", "#64b5f6", "#90caf9"]


def print_banner(thread_id: str, workspace_dir: str | None = None):
    """Print welcome banner with ASCII art logo, thread ID, and workspace path."""
    for line, color in zip(EVOSCIENTIST_ASCII_LINES, _GRADIENT_COLORS):
        console.print(Text(line, style=f"{color} bold"))
    info = Text()
    info.append("  Thread: ", style="dim")
    info.append(thread_id, style="yellow")
    if workspace_dir:
        info.append("\n  Workspace: ", style="dim")
        info.append(workspace_dir, style="cyan")
    info.append("\n  Commands: ", style="dim")
    info.append("/exit", style="bold")
    info.append(", ", style="dim")
    info.append("/new", style="bold")
    info.append(" (new session), ", style="dim")
    info.append("/thread", style="bold")
    info.append(" (show thread ID)", style="dim")
    console.print(info)
    console.print()


# =============================================================================
# CLI commands
# =============================================================================

def cmd_interactive(agent: Any, show_thinking: bool = True, workspace_dir: str | None = None) -> None:
    """Interactive conversation mode with streaming output.

    Args:
        agent: Compiled agent graph
        show_thinking: Whether to display thinking panels
        workspace_dir: Per-session workspace directory path
    """
    thread_id = str(uuid.uuid4())
    print_banner(thread_id, workspace_dir)

    history_file = str(os.path.expanduser("~/.EvoScientist_history"))
    session = PromptSession(
        history=FileHistory(history_file),
        auto_suggest=AutoSuggestFromHistory(),
        enable_history_search=True,
    )

    def _print_separator():
        """Print a horizontal separator line spanning the terminal width."""
        width = console.size.width
        console.print(Text("\u2500" * width, style="dim"))

    _print_separator()
    while True:
        try:
            user_input = session.prompt(
                HTML('<ansiblue><b>&gt;</b></ansiblue> ')
            ).strip()

            if not user_input:
                # Erase the empty prompt line so it looks like nothing happened
                sys.stdout.write("\033[A\033[2K\r")
                sys.stdout.flush()
                continue

            _print_separator()

            # Special commands
            if user_input.lower() in ("/exit", "/quit", "/q"):
                console.print("[dim]Goodbye![/dim]")
                break

            if user_input.lower() == "/new":
                # New session: new workspace, new agent, new thread
                workspace_dir = _create_session_workspace()
                console.print("[dim]Loading new session...[/dim]")
                agent = _load_agent(workspace_dir=workspace_dir)
                thread_id = str(uuid.uuid4())
                console.print(f"[green]New session:[/green] [yellow]{thread_id}[/yellow]")
                console.print(f"[dim]Workspace:[/dim] [cyan]{workspace_dir}[/cyan]\n")
                continue

            if user_input.lower() == "/thread":
                console.print(f"[dim]Thread:[/dim] [yellow]{thread_id}[/yellow]")
                if workspace_dir:
                    console.print(f"[dim]Workspace:[/dim] [cyan]{workspace_dir}[/cyan]")
                console.print()
                continue

            # Stream agent response
            console.print()
            _run_streaming(agent, user_input, thread_id, show_thinking, interactive=True)
            _print_separator()

        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye![/dim]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def cmd_run(agent: Any, prompt: str, thread_id: str | None = None, show_thinking: bool = True, workspace_dir: str | None = None) -> None:
    """Single-shot execution with streaming display.

    Args:
        agent: Compiled agent graph
        prompt: User prompt
        thread_id: Optional thread ID (generates new one if None)
        show_thinking: Whether to display thinking panels
        workspace_dir: Per-session workspace directory path
    """
    thread_id = thread_id or str(uuid.uuid4())

    width = console.size.width
    sep = Text("\u2500" * width, style="dim")
    console.print(sep)
    console.print(Text(f"> {prompt}"))
    console.print(sep)
    console.print(f"[dim]Thread: {thread_id}[/dim]")
    if workspace_dir:
        console.print(f"[dim]Workspace: {workspace_dir}[/dim]")
    console.print()

    try:
        _run_streaming(agent, prompt, thread_id, show_thinking, interactive=False)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise


# =============================================================================
# Agent loading helpers
# =============================================================================

def _create_session_workspace() -> str:
    """Create a per-session workspace directory and return its path."""
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace_dir = os.path.join(".", "workspace", session_id)
    os.makedirs(workspace_dir, exist_ok=True)
    return workspace_dir


def _load_agent(workspace_dir: str | None = None):
    """Load the CLI agent (with InMemorySaver checkpointer for multi-turn).

    Args:
        workspace_dir: Optional per-session workspace directory.
    """
    from .EvoScientist import create_cli_agent
    return create_cli_agent(workspace_dir=workspace_dir)


# =============================================================================
# Typer app
# =============================================================================

app = typer.Typer(no_args_is_help=False, add_completion=False)


@app.callback(invoke_without_command=True)
def _main_callback(
    ctx: typer.Context,
    prompt: Optional[str] = typer.Argument(None, help="Query to execute (single-shot mode)"),
    interactive: bool = typer.Option(False, "-i", "--interactive", help="Interactive conversation mode"),
    thread_id: Optional[str] = typer.Option(None, "--thread-id", help="Thread ID for conversation persistence"),
    no_thinking: bool = typer.Option(False, "--no-thinking", help="Disable thinking display"),
):
    """EvoScientist Agent - AI-powered research & code execution CLI."""
    from dotenv import load_dotenv  # type: ignore[import-untyped]
    load_dotenv(override=True)

    show_thinking = not no_thinking

    # Create per-session workspace
    workspace_dir = _create_session_workspace()

    # Load agent with session workspace
    console.print("[dim]Loading agent...[/dim]")
    agent = _load_agent(workspace_dir=workspace_dir)

    if interactive:
        cmd_interactive(agent, show_thinking=show_thinking, workspace_dir=workspace_dir)
    elif prompt:
        cmd_run(agent, prompt, thread_id=thread_id, show_thinking=show_thinking, workspace_dir=workspace_dir)
    else:
        # Default: interactive mode
        cmd_interactive(agent, show_thinking=show_thinking, workspace_dir=workspace_dir)


def main():
    """CLI entry point — delegates to the Typer app."""
    app()


if __name__ == "__main__":
    main()
