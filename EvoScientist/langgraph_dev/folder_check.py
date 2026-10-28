"""Which folders each access to a served graph is for: a thread's binding.

A thread is bound to the folders its first run worked in: the project and,
for a ``--mode=run`` session, the run folder. The server stamps them on every
checkpoint it writes, and the newest stamped checkpoint is the binding
(``sessions.thread_binding``). Only the server writes the stamp: the graph
factory decides the folders here and hands them to the checkpointer, which
ignores whatever folders the request carried.

- A run is for the project its workspace assistant stands for (any project
  the registry knows); on a graph's own assistant, for its thread's. On a
  bound thread it works in the binding, and is refused when it names another
  project or run folder; on a new thread it works in the run folder it
  forwards, or the project root. A run on a new thread that is not on a
  workspace assistant names no project, and is refused
  (:class:`NoProjectError`); a ``workspace_dir`` it forwards does not count.
- A state update works in its thread's binding and binds nothing.
- A run or update is refused when its project is not served: not
  registered, or its folder is missing.
- A read writes nothing. It works in the project its config names when that
  project is served, and in the neutral graph otherwise.

Graphs outside ``FOLDER_GRAPH_IDS`` work in the workspace root, so only the
workspace counts for them.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

from .. import paths
from ..paths import PATH_ERRORS, SessionDirs, Workspace, normalize_path
from ..sessions import FOLDER_GRAPH_IDS
from .registry import Registry


class RunFolderMismatchError(RuntimeError):
    """An access for folders other than its thread's or this server's."""


class NoProjectError(RunFolderMismatchError):
    """A new conversation that names no project this server knows: it is not
    on a project's workspace assistant."""


class ProjectNotServedError(RunFolderMismatchError):
    """A run or update for a project this server does not serve.

    Not registered and a missing folder give one message; ``missing_folder``
    tells the server to delete the system cron whose run it refused.
    """

    def __init__(self, workspace: Workspace, *, missing_folder: bool) -> None:
        super().__init__(
            f"This EvoScientist server does not serve workspace "
            f"{workspace.root}: it is not registered, or its folder is missing."
        )
        self.workspace = workspace
        self.missing_folder = missing_folder


def neutral_dirs() -> SessionDirs:
    """Where reads that no served project owns are built: an empty folder
    the server owns. No run works there."""
    return _neutral_dirs(paths.SERVER_HOME)


@functools.cache
def _neutral_dirs(server_home: Path) -> SessionDirs:
    root = server_home / "neutral"
    root.mkdir(parents=True, exist_ok=True)
    return SessionDirs(Workspace(root))


def _describe(dirs: SessionDirs) -> str:
    text = f"workspace {dirs.workspace.root}"
    if dirs.run_dir is not None:
        text += f" (run folder {dirs.run_dir})"
    return text


def _read_forwarded(value: Any, what: str) -> Path:
    """A forwarded folder, normalised. Read strictly: a value that is not an
    absolute path (``~`` expanded) or cannot be read is refused."""
    try:
        if not Path(value).expanduser().is_absolute():
            raise ValueError("not an absolute path")
        return normalize_path(value)
    except PATH_ERRORS as exc:
        raise RunFolderMismatchError(
            f"The run forwards a {what} this server cannot read ({value!r}): {exc}"
        ) from exc


def read_workspace_or_none(value: Any) -> Workspace | None:
    """The project a client names, read as :func:`_read_forwarded` does, or
    ``None`` when it names none it can read."""
    if not value:
        return None
    try:
        return Workspace(_read_forwarded(value, "workspace"))
    except RunFolderMismatchError:
        return None


def _forwarded_run_dir(configurable: dict[str, Any], project: Workspace) -> Path | None:
    """The run folder of *project* a run forwards in ``run_dir``, or ``None``
    when it forwards none or the project root. A folder that is not one of
    the project's run folders is refused.
    """
    run_dir = configurable.get("run_dir")
    if not run_dir:
        return None
    path = _read_forwarded(run_dir, "run folder")
    if path == project.root:
        return None
    if not project.is_run_folder(path):
        raise RunFolderMismatchError(
            f"The run's folder {run_dir} (resolved: {path}) is not a run folder "
            f"of workspace {project.root}: run folders are "
            f"{project.runs_dir}/<name>, inside the workspace."
        )
    return path


def _check_served(workspace: Workspace, registry: Registry) -> None:
    if not is_served(workspace, registry):
        missing = not workspace.root.is_dir()
        raise ProjectNotServedError(workspace, missing_folder=missing)


def _project_of(
    graph_id: str,
    configurable: dict[str, Any],
    registry: Registry,
    bound: SessionDirs | None,
) -> Workspace | None:
    """The project a run is for, or ``None`` when it names none.

    A workspace assistant stands for its project, whatever ``workspace_dir``
    the run's ``configurable`` holds: the runtime merges into it the stored
    config of the thread, which keeps the first workspace assistant the
    thread ever ran on. A run on a graph's own assistant is for its thread's
    project.
    """
    project = registry.project_of_assistant(graph_id, configurable.get("assistant_id"))
    if project is not None:
        return project
    return bound.workspace if bound is not None else None


def is_served(workspace: Workspace, registry: Registry) -> bool:
    """Registered, and its folder exists: what this server serves."""
    return registry.is_registered(workspace.key) and workspace.root.is_dir()


def _read_folders(configurable: dict[str, Any], registry: Registry) -> SessionDirs:
    """Where a read works: the served project its config names (a thread's
    stored config, an assistant's), else the neutral graph."""
    workspace = read_workspace_or_none(configurable.get("workspace_dir"))
    if workspace is not None and is_served(workspace, registry):
        return SessionDirs(workspace)
    return neutral_dirs()


def resolve_folders(
    graph_id: str,
    access: str,
    configurable: dict[str, Any],
    registry: Registry,
    bound: SessionDirs | None,
) -> tuple[SessionDirs, SessionDirs | None]:
    """The folders an access to *graph_id* works in, and the ones it stamps
    on its writes (``None`` for an access that writes no stamp).

    *access* is langgraph-api's access context; *bound* is the thread's
    binding (read for writes only). Raises :class:`RunFolderMismatchError`
    for an access this server must not serve: exactly that class for a run
    the thread's binding refuses (the name clients match),
    :class:`ProjectNotServedError` for a project this server does not serve,
    :class:`NoProjectError` for a new conversation that names no project.
    Blocking: it checks folders.
    """
    if access not in ("threads.create_run", "threads.update"):
        return _read_folders(configurable, registry), None
    if access == "threads.update":
        if bound is None:
            return neutral_dirs(), None
        _check_served(bound.workspace, registry)
        return bound, bound

    project = _project_of(graph_id, configurable, registry, bound)
    if project is None:
        raise NoProjectError(
            "The run names no workspace this server knows. Start a new "
            "conversation on the graph's workspace assistant for a registered "
            "project (search assistants by graph_id and metadata.workspace_dir; "
            "opening EvoScientist in a project registers it), or update an "
            "older EvoScientist client."
        )
    if bound is not None and bound.workspace != project:
        raise RunFolderMismatchError(
            f"This conversation works in {_describe(bound)}; it cannot "
            f"continue in workspace {project.root}."
        )
    _check_served(project, registry)
    # Read for every graph, so each refuses a broken value alike; only the
    # graphs that work in a run folder use it.
    forwarded_run_dir = _forwarded_run_dir(configurable, project)
    run_dir = forwarded_run_dir if graph_id in FOLDER_GRAPH_IDS else None
    requested = SessionDirs(project, run_dir)
    if bound is not None and run_dir not in (None, bound.run_dir):
        raise RunFolderMismatchError(
            f"This conversation works in {_describe(bound)}; it cannot "
            f"continue in {_describe(requested)}."
        )
    target = bound or requested
    return target, target
