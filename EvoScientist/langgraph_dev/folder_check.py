"""Refuse runs meant for other folders than the ones this server serves.

Until the server serves several workspaces, it is pinned to one workspace and,
for a ``--mode=run`` session, one run folder. Child runs forward the folders
they belong to in ``configurable``; each registered graph is wrapped in a
factory that compares them with the server's before the run executes, so a
run meant for another folder fails instead of working in this one.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from langgraph_sdk.runtime import ServerRuntime

from ..paths import SessionDirs, process_session_dirs


class RunFolderMismatchError(RuntimeError):
    """A run forwarded folders this server does not serve."""


def _describe(dirs: SessionDirs, *, with_run_dir: bool) -> str:
    text = f"workspace {dirs.workspace.root}"
    if with_run_dir and dirs.run_dir is not None:
        text += f" (run folder {dirs.run_dir})"
    return text


def check_run_folders(config: dict[str, Any], *, works_in_folder: bool) -> None:
    """Raise if the run's forwarded folders differ from the server's.

    A run that forwards no ``workspace_dir`` is not checked. For a graph that
    does not work in a folder, only the workspace counts. Resolves paths, so
    call it off the event loop.
    """
    configurable = config.get("configurable") or {}
    requested = SessionDirs.from_stored(
        configurable.get("workspace_dir"), configurable.get("run_dir")
    )
    if requested is None:
        return
    served = process_session_dirs()
    if requested.workspace == served.workspace and (
        not works_in_folder or requested.run_dir == served.run_dir
    ):
        return
    raise RunFolderMismatchError(
        f"This EvoScientist server serves "
        f"{_describe(served, with_run_dir=works_in_folder)}, but the run is for "
        f"{_describe(requested, with_run_dir=works_in_folder)}. Another EvoSci "
        f"session may have moved the server: restart this session once that "
        f"one is done, or stop it with 'EvoSci server stop'."
    )


def folder_checked(
    graph: Any, *, works_in_folder: bool
) -> Callable[..., Awaitable[Any]]:
    """Register *graph* behind a factory that checks each run's folders.

    langgraph-api calls the factory for every access; only runs are checked
    (``runtime.execution_runtime`` is set), and the prebuilt graph is returned.
    ``works_in_folder`` is False for graphs built for the workspace root only
    (scheduled tasks, memory and AutoSkills workers).
    """

    async def factory(config: dict[str, Any], runtime: ServerRuntime) -> Any:
        if runtime.execution_runtime is not None:
            await asyncio.to_thread(
                check_run_folders, config, works_in_folder=works_in_folder
            )
        return graph

    factory.graph = graph  # type: ignore[attr-defined]
    return factory
