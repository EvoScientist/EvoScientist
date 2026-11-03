"""What the server needs for a project before clients use it: its registration.

Registration (``POST /api/workspaces``) is a setup barrier: the project's
threads are restored and its assistants created, and only then is it
recorded as registered, which is what the factories serve while its folder
exists; what its old runtime store has left is then imported, and its AutoSkills
cron reconciled with the configuration.
Every step can run again; each registration restores the project's threads
again, which adds the threads its CLI sessions made since.
The server's start sets every registered project up again: a fresh runtime store
has none of their assistants. Graphs are built when first needed; the chat
graph, which reads use, starts building in the background. Setup runs inside
the server, through the runtime's own ops.
"""

from __future__ import annotations

import asyncio
import logging

from .. import sessions
from ..paths import SessionDirs, Workspace
from ..sessions import AGENT_NAME, restore_workspace_threads
from . import assistants, registry
from .factories import GRAPHS

logger = logging.getLogger(__name__)

_lock: asyncio.Lock | None = None
# The registered projects set up in this server process: their assistants
# exist.
SET_UP: set[str] = set()
# Background builds, kept until done so they are not garbage-collected, and
# run one at a time: builds take turns anyway (``GraphCache``), and one
# waiting here holds no worker thread.
_warm_ups: set[asyncio.Task] = set()
_warm_lock: asyncio.Lock | None = None


class FolderMissingError(RuntimeError):
    """A project's folder does not exist."""


def registration_lock() -> asyncio.Lock:
    """Held while a project is registered or unregistered, one at a time."""
    global _lock
    if _lock is None:
        _lock = asyncio.Lock()
    return _lock


async def _set_up(
    workspace: Workspace, *, restore: bool, restored_before: bool = False
) -> None:
    if not await asyncio.to_thread(workspace.root.is_dir):
        raise FolderMissingError(f"The folder {workspace.root} does not exist.")
    # A project restored before in this process keeps its threads and its
    # registration when this restore fails; the restore logs it.
    if (
        restore
        and not await restore_workspace_threads(workspace.key)
        and not restored_before
    ):
        raise RuntimeError(f"Could not restore the threads of {workspace.root}.")
    try:
        await assistants.ensure_workspace_assistants(workspace)
    except Exception as exc:
        raise RuntimeError(
            f"Could not create the workspace assistants for {workspace.root}: {exc}"
        ) from exc
    await asyncio.to_thread(registry.REGISTRY.set_registered, workspace.key, True)
    SET_UP.add(workspace.key)
    await _import_old_store(workspace)
    await _reconcile_autoskills(workspace)


async def unregister_workspace(workspace: Workspace) -> None:
    """Record *workspace* as no longer registered; the caller holds
    :func:`registration_lock`."""
    await asyncio.to_thread(registry.REGISTRY.set_registered, workspace.key, False)
    SET_UP.discard(workspace.key)


async def _import_old_store(workspace: Workspace) -> None:
    """Import what *workspace*'s old runtime store has left to import (see
    :mod:`~EvoScientist.langgraph_dev.legacy_import`). Logged on failure: the
    project is served either way, and its next setup tries again."""
    from .legacy_import import import_once

    try:
        await import_once(workspace)
    except Exception:
        logger.warning(
            "Could not import the old store of %s; its next setup tries again.",
            workspace.root,
            exc_info=True,
        )


async def _reconcile_autoskills(workspace: Workspace) -> None:
    """Make *workspace*'s AutoSkills cron match the server's configuration
    (a refused fire into a missing folder deleted it). Logged on failure: the
    project is served either way, and its next setup tries again."""
    from ..config import get_effective_config
    from .server_crons import reconcile_autoskills

    try:
        config = await asyncio.to_thread(get_effective_config)
        await reconcile_autoskills(config, [workspace])
    except Exception:
        logger.warning(
            "Could not reconcile the AutoSkills schedule of %s.",
            workspace.root,
            exc_info=True,
        )


async def reconcile_autoskills_everywhere() -> tuple[int, list[str]]:
    """Reconcile the AutoSkills cron of every served project with the
    server's configuration, after it changed (``/autoskills``, the settings
    API); the caller holds :func:`registration_lock`. Returns how many
    projects were served and why each that failed did."""
    from ..config import get_effective_config
    from .folder_check import is_served
    from .server_crons import reconcile_autoskills

    def served() -> list[Workspace]:
        projects = [Workspace(key) for key in registry.REGISTRY.registered()]
        return [p for p in projects if is_served(p, registry.REGISTRY)]

    config = await asyncio.to_thread(get_effective_config)
    projects = await asyncio.to_thread(served)
    return len(projects), await reconcile_autoskills(config, projects)


def _warmed(task: asyncio.Task) -> None:
    _warm_ups.discard(task)
    if not task.cancelled() and (exc := task.exception()) is not None:
        logger.warning("Could not build the chat graph ahead of use: %s", exc)


async def _warm(workspace: Workspace) -> None:
    global _warm_lock
    if _warm_lock is None:
        _warm_lock = asyncio.Lock()
    async with _warm_lock:
        await asyncio.to_thread(GRAPHS.get, AGENT_NAME, SessionDirs(workspace))


def warm_up(workspace: Workspace) -> None:
    """Start building *workspace*'s chat graph, the one reads use, in the
    background, so a client's first read does not wait for it. A failure is
    logged; the read builds it again."""
    task = asyncio.create_task(_warm(workspace))
    _warm_ups.add(task)
    task.add_done_callback(_warmed)


async def register_workspace(dirs: SessionDirs) -> None:
    """Set *dirs*' project up and record it as registered; the caller holds
    :func:`registration_lock`.

    Clients register their project at every start, and every registration
    runs the whole setup again, each step safe to repeat: it restores the
    project's threads, since the runtime's registry learns of the threads a
    CLI session made only from a restore, checks its assistants, imports what
    its old store has left, reconciles its AutoSkills cron and starts
    building its chat graph. Once restored in this server process, a restore
    that fails again is logged by the restore, not raised.

    Raises :class:`FolderMissingError` when the project's folder does not
    exist, and ``RuntimeError`` when its threads cannot be restored or its
    assistants created; a project that was not registered then stays so.
    """
    restored = dirs.workspace.key in sessions.RESTORED_WORKSPACES
    await _set_up(dirs.workspace, restore=True, restored_before=restored)
    warm_up(dirs.workspace)


async def set_up_at_start() -> None:
    """At the server's start, before it serves requests: set every registered
    project up again. Clients register their own project once the server
    answers, the one that started it too.

    The checkpointer has restored every known project's threads by then. A
    project that cannot be set up is logged: one project never stops a
    server that serves the others.
    """
    for key in await asyncio.to_thread(registry.REGISTRY.registered):
        try:
            await _set_up(await asyncio.to_thread(Workspace, key), restore=False)
        except FolderMissingError:
            logger.warning("Not serving %s: its folder is missing.", key)
        except Exception:
            logger.exception("Could not set %s up.", key)
