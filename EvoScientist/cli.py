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
- Configuration management (onboard, config commands)
"""

import logging
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
from rich.table import Table  # type: ignore[import-untyped]

# Backward-compat re-exports (tests import these from EvoScientist.cli)
from .stream.state import SubAgentState, StreamState, _parse_todo_items, _build_todo_stats  # noqa: F401
from .stream.display import console, _run_streaming
from .paths import ensure_dirs, new_run_dir, default_workspace_dir


def _shorten_path(path: str) -> str:
    """Shorten absolute path to relative path from current directory."""
    if not path:
        return path
    try:
        cwd = os.getcwd()
        if path.startswith(cwd):
            # Remove cwd prefix, keep the relative part
            rel = path[len(cwd):].lstrip(os.sep)
            # Add current dir name for context
            return os.path.join(os.path.basename(cwd), rel) if rel else os.path.basename(cwd)
        return path
    except Exception:
        return path


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


def print_banner(
    thread_id: str,
    workspace_dir: str | None = None,
    memory_dir: str | None = None,
    mode: str | None = None,
    model: str | None = None,
    provider: str | None = None,
):
    """Print welcome banner with ASCII art logo, thread ID, workspace path, and mode."""
    for line, color in zip(EVOSCIENTIST_ASCII_LINES, _GRADIENT_COLORS):
        console.print(Text(line, style=f"{color} bold"))
    info = Text()
    info.append("  Thread: ", style="dim")
    info.append(thread_id, style="yellow")
    if workspace_dir:
        info.append("\n  Workspace: ", style="dim")
        info.append(_shorten_path(workspace_dir), style="cyan")
    if memory_dir:
        trimmed = memory_dir.rstrip("/").rstrip("\\")
        info.append("\n  Memory dir: ", style="dim")
        info.append(_shorten_path(trimmed), style="cyan")
    if model or provider or mode:
        info.append("\n  ", style="dim")
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
    info.append("\n  Commands: ", style="dim")
    info.append("/exit", style="bold")
    info.append(", ", style="dim")
    info.append("/new", style="bold")
    info.append(", ", style="dim")
    info.append("/thread", style="bold")
    info.append(", ", style="dim")
    info.append("/skills", style="bold")
    info.append(", ", style="dim")
    info.append("/install-skill", style="bold")
    info.append(", ", style="dim")
    info.append("/uninstall-skill", style="bold")
    console.print(info)
    console.print()


# =============================================================================
# Skill management commands
# =============================================================================


def _cmd_list_skills() -> None:
    """List installed user skills."""
    from .skills_manager import list_skills
    from .paths import USER_SKILLS_DIR

    skills = list_skills(include_system=False)

    if not skills:
        console.print("[dim]No user skills installed.[/dim]")
        console.print("[dim]Install with:[/dim] /install-skill <path-or-url>")
        console.print(f"[dim]Skills directory:[/dim] [cyan]{_shorten_path(str(USER_SKILLS_DIR))}[/cyan]")
        console.print()
        return

    console.print(f"[bold]Installed Skills[/bold] ({len(skills)}):")
    for skill in skills:
        console.print(f"  [green]{skill.name}[/green] - {skill.description}")
    console.print(f"\n[dim]Location:[/dim] [cyan]{_shorten_path(str(USER_SKILLS_DIR))}[/cyan]")
    console.print()


def _cmd_install_skill(source: str) -> None:
    """Install a skill from local path or GitHub URL."""
    from .skills_manager import install_skill

    if not source:
        console.print("[red]Usage:[/red] /install-skill <path-or-url>")
        console.print("[dim]Examples:[/dim]")
        console.print("  /install-skill ./my-skill")
        console.print("  /install-skill https://github.com/user/repo/tree/main/skill-name")
        console.print("  /install-skill user/repo@skill-name")
        console.print()
        return

    console.print(f"[dim]Installing skill from:[/dim] {source}")

    result = install_skill(source)

    if result["success"]:
        console.print(f"[green]Installed:[/green] {result['name']}")
        console.print(f"[dim]Description:[/dim] {result.get('description', '(none)')}")
        console.print(f"[dim]Path:[/dim] [cyan]{_shorten_path(result['path'])}[/cyan]")
        console.print()
        console.print("[dim]Reload the agent with /new to use the skill.[/dim]")
    else:
        console.print(f"[red]Failed:[/red] {result['error']}")
    console.print()


def _cmd_uninstall_skill(name: str) -> None:
    """Uninstall a user-installed skill."""
    from .skills_manager import uninstall_skill

    if not name:
        console.print("[red]Usage:[/red] /uninstall-skill <skill-name>")
        console.print("[dim]Use /skills to see installed skills.[/dim]")
        console.print()
        return

    result = uninstall_skill(name)

    if result["success"]:
        console.print(f"[green]Uninstalled:[/green] {name}")
        console.print("[dim]Reload the agent with /new to apply changes.[/dim]")
    else:
        console.print(f"[red]Failed:[/red] {result['error']}")
    console.print()


# =============================================================================
# CLI commands
# =============================================================================

def cmd_interactive(
    agent: Any,
    show_thinking: bool = True,
    workspace_dir: str | None = None,
    workspace_fixed: bool = False,
    mode: str | None = None,
    model: str | None = None,
    provider: str | None = None,
) -> None:
    """Interactive conversation mode with streaming output.

    Args:
        agent: Compiled agent graph
        show_thinking: Whether to display thinking panels
        workspace_dir: Per-session workspace directory path
        workspace_fixed: If True, /new keeps the same workspace directory
        mode: Workspace mode ('daemon' or 'run'), displayed in banner
        model: Model name to display in banner
        provider: LLM provider name to display in banner
    """
    thread_id = str(uuid.uuid4())
    from .EvoScientist import MEMORY_DIR
    memory_dir = MEMORY_DIR
    print_banner(thread_id, workspace_dir, memory_dir, mode, model, provider)

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
                # New session: new thread; workspace only changes if not fixed
                if not workspace_fixed:
                    workspace_dir = _create_session_workspace()
                console.print("[dim]Loading new session...[/dim]")
                agent = _load_agent(workspace_dir=workspace_dir)
                thread_id = str(uuid.uuid4())
                console.print(f"[green]New session:[/green] [yellow]{thread_id}[/yellow]")
                if workspace_dir:
                    console.print(f"[dim]Workspace:[/dim] [cyan]{_shorten_path(workspace_dir)}[/cyan]\n")
                continue

            if user_input.lower() == "/thread":
                console.print(f"[dim]Thread:[/dim] [yellow]{thread_id}[/yellow]")
                if workspace_dir:
                    console.print(f"[dim]Workspace:[/dim] [cyan]{_shorten_path(workspace_dir)}[/cyan]")
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

            # Stream agent response
            console.print()
            _run_streaming(agent, user_input, thread_id, show_thinking, interactive=True)
            _print_separator()

        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye![/dim]")
            break
        except Exception as e:
            error_msg = str(e)
            if "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
                console.print("[red]Error: API key not configured.[/red]")
                console.print("[dim]Run [bold]EvoSci onboard[/bold] to set up your API key.[/dim]")
                break
            else:
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
        console.print(f"[dim]Workspace: {_shorten_path(workspace_dir)}[/dim]")
    console.print()

    try:
        _run_streaming(agent, prompt, thread_id, show_thinking, interactive=False)
    except Exception as e:
        error_msg = str(e)
        if "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
            console.print("[red]Error: API key not configured.[/red]")
            console.print("[dim]Run [bold]EvoSci onboard[/bold] to set up your API key.[/dim]")
            raise typer.Exit(1)
        else:
            console.print(f"[red]Error: {e}[/red]")
            raise


# =============================================================================
# Agent loading helpers
# =============================================================================

def _create_session_workspace() -> str:
    """Create a per-session workspace directory and return its path."""
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace_dir = str(new_run_dir(session_id))
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

# Config subcommand group
config_app = typer.Typer(help="Configuration management commands", invoke_without_command=True)
app.add_typer(config_app, name="config")


# =============================================================================
# Onboard command
# =============================================================================

@app.command()
def onboard(
    skip_validation: bool = typer.Option(
        False,
        "--skip-validation",
        help="Skip API key validation during setup"
    ),
):
    """Interactive setup wizard for EvoScientist.

    Guides you through configuring API keys, model selection,
    workspace settings, and agent parameters.
    """
    from .onboard import run_onboard
    run_onboard(skip_validation=skip_validation)


# =============================================================================
# Config commands
# =============================================================================

@config_app.callback(invoke_without_command=True)
def config_callback(ctx: typer.Context):
    """Configuration management commands."""
    if ctx.invoked_subcommand is None:
        config_list()


@config_app.command("list")
def config_list():
    """List all configuration values."""
    from .config import list_config, get_config_path

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
    """Get a single configuration value."""
    from .config import get_config_value

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
    """Set a single configuration value."""
    from .config import set_config_value

    if set_config_value(key, value):
        console.print(f"[green]Set {key}[/green]")
    else:
        console.print(f"[red]Invalid key: {key}[/red]")
        raise typer.Exit(1)


@config_app.command("reset")
def config_reset(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Reset configuration to defaults."""
    from .config import reset_config, get_config_path

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
    """Show the configuration file path."""
    from .config import get_config_path

    path = get_config_path()
    exists = path.exists()
    status = "[green]exists[/green]" if exists else "[dim]not created yet[/dim]"
    console.print(f"{path} ({status})")


# =============================================================================
# Main callback (default behavior)
# =============================================================================

@app.callback(invoke_without_command=True)
def _main_callback(
    ctx: typer.Context,
    prompt: Optional[str] = typer.Option(None, "-p", "--prompt", help="Query to execute (single-shot mode)"),
    thread_id: Optional[str] = typer.Option(None, "--thread-id", help="Thread ID for conversation persistence"),
    no_thinking: bool = typer.Option(False, "--no-thinking", help="Disable thinking display"),
    workdir: Optional[str] = typer.Option(None, "--workdir", help="Override workspace directory for this session"),
    use_cwd: bool = typer.Option(False, "--use-cwd", help="Use current working directory as workspace"),
    mode: Optional[str] = typer.Option(
        None,
        "--mode",
        help="Workspace mode: 'daemon' (persistent, default) or 'run' (isolated per-session)"
    ),
):
    """EvoScientist Agent - AI-powered research & code execution CLI."""
    # If a subcommand was invoked, don't run the default behavior
    if ctx.invoked_subcommand is not None:
        return

    from dotenv import load_dotenv, find_dotenv  # type: ignore[import-untyped]
    # find_dotenv() traverses up the directory tree to locate .env
    load_dotenv(find_dotenv(), override=True)

    # Load and apply configuration
    from .config import get_effective_config, apply_config_to_env

    # Build CLI overrides dict
    cli_overrides = {}
    if mode:
        cli_overrides["default_mode"] = mode
    if workdir:
        cli_overrides["default_workdir"] = workdir
    if no_thinking:
        cli_overrides["show_thinking"] = False

    config = get_effective_config(cli_overrides)
    apply_config_to_env(config)

    show_thinking = config.show_thinking if not no_thinking else False

    # Validate mutually exclusive options
    if workdir and use_cwd:
        raise typer.BadParameter("Use either --workdir or --use-cwd, not both.")

    if mode and (workdir or use_cwd):
        raise typer.BadParameter("--mode cannot be combined with --workdir or --use-cwd")

    if mode and mode not in ("run", "daemon"):
        raise typer.BadParameter("--mode must be 'run' or 'daemon'")

    ensure_dirs()

    # Resolve effective mode from config (CLI mode already applied via overrides)
    effective_mode: str | None = None  # None means explicit --workdir/--use-cwd was used

    # Resolve workspace directory for this session
    # Priority: --use-cwd > --workdir > --mode (explicit) > default_workdir > default_mode
    if use_cwd:
        workspace_dir = os.getcwd()
        workspace_fixed = True
    elif workdir:
        workspace_dir = os.path.abspath(os.path.expanduser(workdir))
        os.makedirs(workspace_dir, exist_ok=True)
        workspace_fixed = True
    elif mode:
        # Explicit --mode overrides default_workdir
        effective_mode = mode
        workspace_root = config.default_workdir or str(default_workspace_dir())
        workspace_root = os.path.abspath(os.path.expanduser(workspace_root))
        if effective_mode == "run":
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            workspace_dir = os.path.join(workspace_root, "runs", session_id)
            os.makedirs(workspace_dir, exist_ok=True)
            workspace_fixed = False
        else:  # daemon
            workspace_dir = workspace_root
            os.makedirs(workspace_dir, exist_ok=True)
            workspace_fixed = True
    elif config.default_workdir:
        # Use configured default workdir with configured mode
        workspace_root = os.path.abspath(os.path.expanduser(config.default_workdir))
        effective_mode = config.default_mode
        if effective_mode == "run":
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            workspace_dir = os.path.join(workspace_root, "runs", session_id)
            os.makedirs(workspace_dir, exist_ok=True)
            workspace_fixed = False
        else:  # daemon
            workspace_dir = workspace_root
            os.makedirs(workspace_dir, exist_ok=True)
            workspace_fixed = True
    else:
        effective_mode = config.default_mode
        if effective_mode == "run":
            workspace_dir = _create_session_workspace()
            workspace_fixed = False
        else:  # daemon mode (default)
            workspace_dir = str(default_workspace_dir())
            os.makedirs(workspace_dir, exist_ok=True)
            workspace_fixed = True

    # Load agent with session workspace
    console.print("[dim]Loading agent...[/dim]")
    agent = _load_agent(workspace_dir=workspace_dir)

    if prompt:
        # Single-shot mode: execute query and exit
        cmd_run(agent, prompt, thread_id=thread_id, show_thinking=show_thinking, workspace_dir=workspace_dir)
    else:
        # Interactive mode (default)
        cmd_interactive(
            agent,
            show_thinking=show_thinking,
            workspace_dir=workspace_dir,
            workspace_fixed=workspace_fixed,
            mode=effective_mode,
            model=config.model,
            provider=config.provider,
        )


def _configure_logging():
    """Configure logging with warning symbols for better visibility."""
    from rich.logging import RichHandler

    class DimWarningHandler(RichHandler):
        """Custom handler that renders warnings in dim style."""

        def emit(self, record: logging.LogRecord) -> None:
            if record.levelno == logging.WARNING:
                # Use Rich console to print dim warning
                msg = record.getMessage()
                console.print(f"[dim yellow]\u26a0\ufe0f  Warning:[/dim yellow] [dim]{msg}[/dim]")
            else:
                super().emit(record)

    # Configure root logger to use our handler for WARNING and above
    handler = DimWarningHandler(console=console, show_time=False, show_path=False, show_level=False)
    handler.setLevel(logging.WARNING)

    # Apply to root logger (catches all loggers including deepagents)
    root_logger = logging.getLogger()
    # Remove existing handlers to avoid duplicate output
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.WARNING)


def main():
    """CLI entry point — delegates to the Typer app."""
    import warnings
    warnings.filterwarnings("ignore", message=".*not known to support tools.*")
    warnings.filterwarnings("ignore", message=".*type is unknown and inference may fail.*")
    _configure_logging()
    app()


if __name__ == "__main__":
    main()
