"""The server home: the folder the server runs in (``paths.SERVER_HOME``).

It holds what belongs to the server rather than to a project: the runtime's
store (``.langgraph_api/``, which langgraph-api writes relative to its working
folder), the local secret, and the lock that keeps one server per data
folder: two servers on one store would both fire its crons and overwrite each
other's writes. A process working in the server home is the server: the
manager starts it there, and the agent's commands run in project folders.
"""

from __future__ import annotations

import logging
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

from filelock import FileLock
from filelock import Timeout as FileLockTimeout

from .. import paths

logger = logging.getLogger(__name__)

LOCK_FILE = "server.lock"
# A server stopped just before may still be exiting, and hold the lock.
LOCK_WAIT = 5.0
# How long a check waits for the lock: other clients checking hold it for an
# instant, a server for its life.
PROBE_WAIT = 0.2

_held: FileLock | None = None


class ServerRunningError(RuntimeError):
    """Another server uses this data folder."""


def running_server_message() -> str:
    return (
        f"Another EvoScientist server already uses the data folder "
        f"{paths.DATA_DIR}, perhaps on the port you used before changing it. "
        "Stop it first with `EvoSci server stop`: two servers on one data "
        "folder would both run its scheduled tasks."
    )


def ensure_server_home() -> Path:
    """Create the server home, readable by this user only. The local secret
    narrows a folder that already exists, and says so (``local_secret``)."""
    home = paths.SERVER_HOME
    home.mkdir(mode=0o700, parents=True, exist_ok=True)
    return home


def serves_from_home() -> bool:
    """Whether this process works in the server home: it is the server."""
    try:
        return Path.cwd().resolve() == paths.SERVER_HOME.resolve()
    except OSError:
        return False


def lock_path() -> Path:
    """This data folder's server lock."""
    return paths.SERVER_HOME / LOCK_FILE


# What :func:`lock_state` finds.
HELD = "held"
FREE = "free"
UNKNOWN = "unknown"


def lock_state(wait: float = PROBE_WAIT) -> str:
    """Whether a server uses this data folder: :data:`HELD` when the lock
    stays held for *wait* seconds, longer than another client's check holds
    it; :data:`FREE`; :data:`UNKNOWN` when the lock file cannot be opened
    (another user's, after a ``sudo EvoSci``). Blocking."""
    if not paths.SERVER_HOME.is_dir():
        return FREE
    lock = FileLock(str(lock_path()))
    try:
        lock.acquire(timeout=wait)
    except FileLockTimeout:
        return HELD
    except OSError:
        return UNKNOWN
    lock.release()
    return FREE


def hold_server_lock() -> None:
    """Take this data folder's server lock for the life of this process.

    Raises :class:`ServerRunningError` when another server keeps it past
    :data:`LOCK_WAIT`. On a file system without locks, filelock falls back
    to a lock file that a crash leaves behind.
    """
    global _held
    if _held is not None:
        return
    ensure_server_home()
    lock = FileLock(str(lock_path()))
    try:
        lock.acquire(timeout=LOCK_WAIT)
    except FileLockTimeout:
        raise ServerRunningError(running_server_message()) from None
    _held = lock


def store_file() -> Path:
    """The runtime's store, relative to the server's working folder, as the
    runtime names it."""
    from langgraph_runtime_inmem.database import OPS_FILENAME

    return Path(OPS_FILENAME)


def _backup(store: Path) -> Path:
    return store.with_name(store.name + ".backup")


def back_up_store() -> None:
    """Copy the runtime's store aside before the runtime loads it: a load that
    fails deletes the file. A failure is logged only."""
    store = store_file()
    backup = _backup(store)
    if not store.is_file():
        return
    # Whole or absent: a copy that fails leaves no backup to mistake for it.
    tmp = backup.with_name(f"{backup.name}.{os.getpid()}")
    try:
        backup.unlink(missing_ok=True)
        shutil.copy2(store, tmp)
        os.replace(tmp, backup)
    except OSError:
        tmp.unlink(missing_ok=True)
        logger.warning("Could not back the server's store up.", exc_info=True)


def check_store_loaded() -> None:
    """After the runtime loaded its store: drop the backup, or keep it as
    ``.langgraph_ops.pckl.unloadable-<time>`` when the runtime deleted the
    store it could not load, and say so. A failure here is logged only."""
    if not serves_from_home():
        return
    try:
        _check_store_loaded()
    except OSError:
        logger.warning("Could not check the server's store backup.", exc_info=True)


def _check_store_loaded() -> None:
    store = store_file()
    backup = _backup(store)
    if not backup.is_file():
        return
    if store.is_file():
        backup.unlink()
        return
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    kept = backup.with_name(f"{store.name}.unloadable-{stamp}")
    backup.rename(kept)
    logger.error(
        "The server could not load its store (threads' runs, scheduled tasks, "
        "assistants); the file is kept as %s.",
        kept.resolve(),
    )
