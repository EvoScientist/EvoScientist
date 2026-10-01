"""Agent loading and workspace helpers."""

import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ..paths import RUN_NAME_FORMAT, Workspace

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from ..runtime import AsyncRuntime


def _shorten_path(path: str) -> str:
    """Shorten absolute path to relative path from current directory."""
    if not path:
        return path
    try:
        cwd = os.getcwd()
        if path.startswith(cwd):
            rel = path[len(cwd) :].lstrip(os.sep)
            return (
                os.path.join(os.path.basename(cwd), rel)
                if rel
                else os.path.basename(cwd)
            )
        return path
    except Exception:
        return path


def _deduplicate_run_name(name: str, runs_dir: Path) -> str:
    """Return *name* if available, otherwise *name_1*, *name_2*, etc."""
    if not (runs_dir / name).exists():
        return name
    i = 1
    while (runs_dir / f"{name}_{i}").exists():
        i += 1
    return f"{name}_{i}"


def _create_run_dir(workspace: Workspace, name: str | None = None) -> Path:
    """Create a ``--mode=run`` session folder under ``workspace.runs_dir``.

    Args:
        workspace: The workspace the run folder belongs to.
        name: Optional human-friendly run name.  Duplicates are resolved
              by appending ``_1``, ``_2``, etc.  Falls back to a timestamp
              if *name* is None.
    """
    if name:
        session_id = _deduplicate_run_name(name, workspace.runs_dir)
    else:
        session_id = datetime.now().strftime(RUN_NAME_FORMAT)
    run_dir = workspace.runs_dir / session_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _remove_unused_run_dir(run_dir: Path | None) -> None:
    """Remove a run folder created for a session that did not start."""
    if run_dir is None:
        return
    try:
        run_dir.rmdir()
    except OSError:
        pass


def _load_agent(
    work_dir: str | None = None,
    checkpointer=None,
    config=None,
    chat_model=None,
    *,
    workspace: Workspace,
    on_mcp_progress=None,
    events=None,
    runtime: "AsyncRuntime | None" = None,
) -> "CompiledStateGraph":
    """Load the CLI agent with optional persistent checkpointer.

    Args:
        work_dir: The folder the agent works in (defaults to the
            workspace root).
        workspace: The session's workspace.
        checkpointer: Optional LangGraph checkpointer (e.g. ``AsyncSqliteSaver``).
            Falls back to ``InMemorySaver`` when ``None``.
        config: Optional pre-loaded ``EvoScientistConfig``.  Forwarded to
            ``create_cli_agent`` to avoid double config loading.
        chat_model: Optional pre-built chat model.  Forwarded to
            ``create_cli_agent``; combined with an explicit ``config`` it
            selects the pure (no module-global write) build path.
        on_mcp_progress: Optional per-server MCP progress callback.
            Signature ``(event, server_name, detail) -> None``.
        runtime: Optional application-scoped runtime used for MCP discovery.
    """
    from ..EvoScientist import create_cli_agent

    return create_cli_agent(
        work_dir=work_dir,
        workspace=workspace,
        checkpointer=checkpointer,
        config=config,
        chat_model=chat_model,
        on_mcp_progress=on_mcp_progress,
        events=events,
        runtime=runtime,
    )
