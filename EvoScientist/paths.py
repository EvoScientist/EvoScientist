"""Path resolution utilities for EvoScientist runtime directories."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path


def _expand(path: str) -> Path:
    return Path(path).expanduser()


def _env_path(key: str) -> Path | None:
    value = os.getenv(key)
    if not value:
        return None
    return _expand(value)


# Workspace root: current working directory by default (user's project dir)
WORKSPACE_ROOT = _env_path("EVOSCIENTIST_WORKSPACE_DIR") or Path.cwd()

RUNS_DIR = _env_path("EVOSCIENTIST_RUNS_DIR") or (WORKSPACE_ROOT / "runs")
MEMORY_DIR = _env_path("EVOSCIENTIST_MEMORY_DIR") or (WORKSPACE_ROOT / "memory")
USER_SKILLS_DIR = _env_path("EVOSCIENTIST_SKILLS_DIR") or (WORKSPACE_ROOT / "skills")
MEDIA_DIR = _env_path("EVOSCIENTIST_MEDIA_DIR") or (WORKSPACE_ROOT / "media")


def set_workspace_root(path: str | Path) -> None:
    """Update workspace root and re-derive dependent directories.

    Directories with an explicit environment-variable override keep their
    env-var value; all others are re-derived from the new root.
    Also resets ``_active_workspace`` to the new root as a safe default.
    """
    global WORKSPACE_ROOT, RUNS_DIR, MEMORY_DIR, USER_SKILLS_DIR, MEDIA_DIR, _active_workspace
    WORKSPACE_ROOT = Path(path).resolve()
    _active_workspace = WORKSPACE_ROOT
    RUNS_DIR = _env_path("EVOSCIENTIST_RUNS_DIR") or (WORKSPACE_ROOT / "runs")
    MEMORY_DIR = _env_path("EVOSCIENTIST_MEMORY_DIR") or (WORKSPACE_ROOT / "memory")
    USER_SKILLS_DIR = _env_path("EVOSCIENTIST_SKILLS_DIR") or (WORKSPACE_ROOT / "skills")
    MEDIA_DIR = _env_path("EVOSCIENTIST_MEDIA_DIR") or (WORKSPACE_ROOT / "media")


def ensure_dirs() -> None:
    """Create runtime subdirectories (memory, skills) if they do not exist.

    Does NOT create the workspace root itself — it should already exist
    (either the user's cwd or a directory they specified).
    """
    for path in (MEMORY_DIR, USER_SKILLS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def default_workspace_dir() -> Path:
    """Default workspace for non-CLI usage."""
    return WORKSPACE_ROOT


def new_run_dir(session_id: str | None = None) -> Path:
    """Create a new run directory name under RUNS_DIR (path only)."""
    if session_id is None:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return RUNS_DIR / session_id


# Active workspace (may differ from WORKSPACE_ROOT in per-session modes)
_active_workspace: Path = WORKSPACE_ROOT


def set_active_workspace(path: str | Path) -> None:
    """Update the active workspace root (called on agent creation)."""
    global _active_workspace
    _active_workspace = Path(path).resolve()


def resolve_virtual_path(virtual_path: str) -> Path:
    """Resolve a virtual workspace path (e.g. /image.png) to a real filesystem path."""
    vpath = virtual_path if virtual_path.startswith("/") else "/" + virtual_path
    return (_active_workspace / vpath.lstrip("/")).resolve()
