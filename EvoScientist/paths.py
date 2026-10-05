"""Path resolution utilities for EvoScientist runtime directories."""

from __future__ import annotations

import functools
import logging
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def normalize_path(path: str | Path) -> Path:
    """Return the canonical absolute form of *path*.

    Expands ``~`` and resolves symlinks, so ``/tmp/x`` and ``/private/tmp/x``
    on macOS, or two spellings of a symlinked project folder, compare equal.
    The path does not need to exist. Raises ``ValueError`` for a path with a
    NUL byte, which no platform accepts (Windows on Python 3.13 would
    otherwise resolve it without complaint).
    """
    if "\x00" in str(path):
        raise ValueError(f"Path contains a NUL byte: {path!r}")
    return Path(path).expanduser().resolve()


@dataclass(frozen=True)
class Workspace:
    """A project folder: the unit that owns skills, runs, media and memory.

    Build one where a workspace enters the process (CLI startup, ``/resume``,
    ``serve``, a server graph build) and pass it to whatever needs it.
    Constructing a ``Workspace`` never touches the filesystem beyond resolving
    the path, and never creates the folder.
    """

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", normalize_path(self.root))

    @property
    def skills_dir(self) -> Path:
        """Workspace skills tier (``<root>/skills``)."""
        return self.root / "skills"

    @property
    def runs_dir(self) -> Path:
        """Parent folder of ``--mode=run`` session folders (``<root>/runs``)."""
        return self.root / "runs"

    @property
    def media_dir(self) -> Path:
        """Where channels store inbound attachments (``<root>/media``)."""
        return self.root / "media"

    @property
    def key(self) -> str:
        """Stable string form for storage and comparison (POSIX separators)."""
        return self.root.as_posix()


# Name ``--mode=run`` gives a run folder when the user passes no ``--name``.
RUN_NAME_FORMAT = "%Y%m%d_%H%M%S"
_RUN_NAME_SHAPE = re.compile(r"\d{8}_\d{6}")


def is_generated_run_name(name: str) -> bool:
    """True if *name* is a run folder name ``--mode=run`` generates."""
    if not _RUN_NAME_SHAPE.fullmatch(name):
        return False
    try:
        datetime.strptime(name, RUN_NAME_FORMAT)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class SessionDirs:
    """Where a session lives: its workspace, and the run folder it works in.

    ``run_dir`` is set only in ``--mode=run``, where the session works in a
    folder under ``<root>/runs``. Skills, media, the memory project and crons
    always come from the workspace; the sandbox works in :attr:`work_dir`.
    """

    workspace: Workspace
    run_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.run_dir is None:
            return
        run_dir = normalize_path(self.run_dir)
        object.__setattr__(
            self, "run_dir", None if run_dir == self.workspace.root else run_dir
        )

    @property
    def work_dir(self) -> Path:
        """The folder the agent works in."""
        return self.run_dir or self.workspace.root

    def metadata(self) -> dict[str, str]:
        """Thread metadata and ``configurable`` keys for this session.

        ``run_dir`` is left out, not set to an empty string, when the session
        works in the workspace root.
        """
        fields = {"workspace_dir": self.workspace.key}
        if self.run_dir is not None:
            fields["run_dir"] = self.run_dir.as_posix()
        return fields

    @classmethod
    def from_stored(
        cls, workspace_dir: str | Path | None, run_dir: str | Path | None = None
    ) -> SessionDirs | None:
        """Read the folders a thread was stored with, normalised.

        Returns ``None`` when nothing was stored or the value cannot be
        resolved. A ``run_dir`` outside the workspace is dropped: run folders
        live under ``<root>/runs``. Resolves paths, so call it off the event
        loop on the server.
        """
        if not workspace_dir:
            return None
        try:
            dirs = cls(Workspace(workspace_dir), Path(run_dir) if run_dir else None)
        except (ValueError, OSError, RuntimeError):
            return None
        if dirs.run_dir is not None and not dirs.run_dir.is_relative_to(
            dirs.workspace.root
        ):
            return cls(dirs.workspace)
        return dirs

    @classmethod
    def from_legacy(cls, workspace_dir: str | Path) -> SessionDirs:
        """Read a ``workspace_dir`` stored before ``run_dir`` existed.

        Run-mode sessions stored their run folder as ``workspace_dir``. A
        folder ``<root>/runs/<name>`` whose name has the form ``--mode=run``
        generates is split into the workspace and ``run_dir``. Named runs are
        left as they are, because a project folder can itself live in a
        folder called ``runs``. Only the one-time upgrade of stored values
        uses this; new values carry ``run_dir`` themselves.
        """
        path = normalize_path(workspace_dir)
        if path.parent.name == "runs" and is_generated_run_name(path.name):
            return cls(Workspace(path.parent.parent), path)
        return cls(Workspace(path))


def start_workspace_path(
    workdir: str | Path | None = None,
    default_workdir: str | Path | None = None,
) -> Path:
    """Pick the folder a process starts in: ``workdir``, then ``default_workdir``,
    then the current directory. The result is absolute but not resolved."""
    for candidate in (workdir, default_workdir):
        if candidate:
            return Path(os.path.abspath(os.path.expanduser(candidate)))
    return Path(os.getcwd())


@functools.cache
def process_workspace() -> Workspace:
    """The workspace of a process that was not given one explicitly.

    ``EVOSCIENTIST_WORKSPACE_DIR`` when set (the langgraph dev manager sets it
    for the server subprocess), otherwise the current directory. Read once per
    process, so every graph the server builds and every route it answers agree
    on the workspace even if the environment changes later (a ``.env`` merge).
    """
    return Workspace(_env_path("EVOSCIENTIST_WORKSPACE_DIR") or Path.cwd())


@functools.cache
def process_session_dirs() -> SessionDirs:
    """:func:`process_workspace` plus the run folder the process works in.

    The langgraph dev manager sets ``EVOSCIENTIST_RUN_DIR`` when it starts the
    server for a ``--mode=run`` session. Read once per process, like the
    workspace.
    """
    return SessionDirs(process_workspace(), _env_path("EVOSCIENTIST_RUN_DIR"))


def resolve_virtual_path(work_dir: str | Path, virtual_path: str) -> Path:
    """Resolve a virtual path (e.g. ``/image.png``) against *work_dir*."""
    vpath = virtual_path if virtual_path.startswith("/") else "/" + virtual_path
    return (Path(work_dir) / vpath.lstrip("/")).resolve()


def _expand(path: str) -> Path:
    return Path(path).expanduser()


def _env_path(key: str) -> Path | None:
    value = os.getenv(key)
    if not value:
        return None
    return _expand(value)


def _global_data_dir() -> Path:
    """Global application data directory (~/.evoscientist/ by default).

    This is the base for sessions.db, skills/, memories/, history — things
    that are NOT configuration but application state. Config files (config.yaml,
    mcp.yaml) continue to live in XDG_CONFIG_HOME.
    """
    return Path.home() / ".evoscientist"


# Global data dir: ~/.evoscientist/ by default, overridable via env var.
DATA_DIR: Path = _env_path("EVOSCIENTIST_DATA_DIR") or _global_data_dir()


def _global_skills_dir() -> Path:
    return DATA_DIR / "skills"


def _global_memories_dir() -> Path:
    return DATA_DIR / "memories"


# Global skills: shared across all workspaces (~/.evoscientist/skills/)
GLOBAL_SKILLS_DIR: Path = _global_skills_dir()

# Global memories: shared across all workspaces (~/.evoscientist/memories/)
GLOBAL_MEMORIES_DIR: Path = _global_memories_dir()


# Memories dir: global by default, overridable via env var.
# Supports both new (EVOSCIENTIST_MEMORIES_DIR) and old (EVOSCIENTIST_MEMORY_DIR) env vars.
def _memories_dir_from_env() -> Path:
    return (
        _env_path("EVOSCIENTIST_MEMORIES_DIR")
        or _env_path("EVOSCIENTIST_MEMORY_DIR")
        or GLOBAL_MEMORIES_DIR
    )


MEMORIES_DIR: Path = _memories_dir_from_env()


def reload_env_dirs() -> None:
    """Re-read the memories-folder override from the environment.

    ``paths`` is imported before ``get_effective_config()`` merges the
    project ``.env`` into ``os.environ``, so entry points call this once
    after loading their config to pick up an override set there.
    """
    global MEMORIES_DIR
    MEMORIES_DIR = _memories_dir_from_env()


# DEPRECATED(0.1.0): remove this migration helper and its call site below.
def migrate_legacy_sessions_db() -> None:
    """One-time migration: copy sessions.db (and its WAL/SHM siblings) from
    ~/.config/evoscientist/ to ~/.evoscientist/.

    Scope is intentionally narrow — only the SQLite trio, because users can't
    easily move those by hand. User-facing files (skills/, memories/, history)
    are migrated via an agent prompt documented in the release notes.

    Idempotent via ``.migrated`` marker file. The marker is not written when
    a copy fails, so transient I/O errors don't permanently block retry.
    """
    marker = DATA_DIR / ".migrated"
    if marker.exists():
        return

    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.debug("Could not create %s; skipping legacy migration.", DATA_DIR)
        return

    # Resolve legacy source via XDG_CONFIG_HOME (matches config.settings.get_config_dir).
    # Inlined here to avoid importing config.settings at paths load time.
    xdg = os.environ.get("XDG_CONFIG_HOME")
    legacy = (
        (Path(xdg) / "evoscientist")
        if xdg
        else (Path.home() / ".config" / "evoscientist")
    )
    if not legacy.exists():
        marker.touch()
        return

    migrated: list[str] = []
    failed: list[str] = []
    for name in ("sessions.db", "sessions.db-wal", "sessions.db-shm"):
        src = legacy / name
        dst = DATA_DIR / name
        if not src.exists():
            continue
        if dst.exists():
            continue  # already migrated
        try:
            shutil.copy2(src, dst)
            migrated.append(name)
        except OSError as e:
            logger.warning("Failed to migrate %s: %s", src, e)
            failed.append(name)

    if migrated:
        logger.info(
            "Migrated legacy session DB from %s to %s: %s. "
            "Legacy files are kept as backup; this auto-migration will be "
            "removed in EvoScientist 0.1.0.",
            legacy,
            DATA_DIR,
            ", ".join(migrated),
        )

    # Only write the marker when there were no failures — preserves retry
    # on transient I/O errors.
    if not failed:
        marker.touch()


# DEPRECATED(0.1.0): remove this call together with migrate_legacy_sessions_db().
try:
    migrate_legacy_sessions_db()
except Exception:
    # Never block startup on migration failures
    logger.exception("Legacy session DB migration failed; continuing without it.")


def ensure_dirs() -> None:
    """Create runtime subdirectories if they do not exist.

    Creates DATA_DIR (and MEMORIES_DIR as its subdir). Skills directories
    are created on demand by ``install_skill()`` when the user first
    installs a skill.

    Does NOT create the workspace root itself — it should already exist
    (either the user's cwd or a directory they specified).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MEMORIES_DIR.mkdir(parents=True, exist_ok=True)
