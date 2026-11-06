"""The server's record of itself in the server home
(``langgraph_dev.workspace.json``).

The server writes it when :mod:`.http` is imported, right after it took its
data folder's lock, as ``starting``, and exits when it cannot; it marks it
``serving`` just before it serves, and removes it only as the process exits,
just before the lock goes with it. It adds the ccproxy it starts, if any, as
soon as that runs. So a record names the server that holds the lock, also
while it stops. A stop by signal (SIGTERM, Ctrl+Break) ends the process before
that removal runs, as uvicorn raises the signal again once it has shut down;
the stop that ended it then deletes the record
(:func:`~EvoScientist.langgraph_dev.manager.stop_server`,
:func:`~EvoScientist.langgraph_dev.manager.stop_langgraph_dev`). Clients read it
(:func:`~EvoScientist.langgraph_dev.manager.read_record`) and never write it.

What the server cannot know comes from the process that started it, in
variables it removes once read: whether it outlives that process
(:data:`LASTING_VAR`), the configuration it was started for
(:data:`FINGERPRINT_VAR`), and the starter's PID (:data:`SPAWNER_PID_VAR`).
"""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import os
import sys
import time
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path

import psutil

from .. import paths

logger = logging.getLogger(__name__)

RECORD_FILE = "langgraph_dev.workspace.json"

# Set by the process that starts the server, read and removed by the server.
SPAWNER_PID_VAR = "EVOSCIENTIST_SERVER_SPAWNER_PID"
LASTING_VAR = "EVOSCIENTIST_SERVER_LASTING"
FINGERPRINT_VAR = "EVOSCIENTIST_SERVER_CONFIG_FINGERPRINT"

# Record keys, and the phases of a server.
CREATE_TIME_KEY = "create_time"
BREAK_GROUP_KEY = "break_group"
PHASE_KEY = "phase"
STARTING = "starting"
SERVING = "serving"
# The ccproxy the server started (Anthropic OAuth), for the next server to
# end when this one is killed and leaves it running.
CCPROXY_PID_KEY = "ccproxy_pid"
CCPROXY_CREATE_TIME_KEY = "ccproxy_create_time"

# How far up from itself the server looks for the process its starter
# started: on Windows ``langgraph.exe`` is a launcher whose child runs Python,
# and a virtual environment's ``python.exe`` may be one more.
_MAX_PARENT_STEPS = 3

# How long a write or removal of the record tries again while a client
# reads it, and how often.
_REPLACE_WAIT = 5.0
_REPLACE_RETRY = 0.05

# langgraph dev's own defaults (its command line's), for a server started by
# hand without ``--port`` or ``--host``.
DEFAULT_PORT = 2024
DEFAULT_HOST = "127.0.0.1"

# What this process wrote, so it removes only its own record.
_written: dict | None = None


def record_path() -> Path:
    return paths.SERVER_HOME / RECORD_FILE


def _started_process() -> tuple[psutil.Process, bool]:
    """The process the starter started, and whether it was found.

    The starter knows the server by the PID its ``Popen`` got: the
    ``langgraph`` launcher, which on Windows runs this Python as a child. It
    is the ancestor whose parent is the starter; without a starter (a server
    started by hand), or when none is, it is this process.
    """
    me = psutil.Process()
    try:
        spawner = int(os.environ.get(SPAWNER_PID_VAR, ""))
    except ValueError:
        return me, False
    process = me
    for _ in range(_MAX_PARENT_STEPS + 1):
        try:
            if process.ppid() == spawner:
                return process, True
            parent = process.parent()
        except psutil.Error:
            break
        if parent is None:
            break
        process = parent
    return me, False


def _address() -> tuple[int, str]:
    """The port and interface the server binds, from its command line, where
    its starter always puts them; langgraph dev's defaults without them."""
    return address_of(sys.argv)


def option(args: list[str], name: str) -> str | None:
    """The value of the option *name* in *args*, given as ``--name value``
    or ``--name=value``."""
    for arg, value in pairwise(args):
        if arg == name:
            return value
    for arg in args:
        if arg.startswith(f"{name}="):
            return arg.split("=", 1)[1]
    return None


def address_of(args: list[str]) -> tuple[int, str]:
    """The port and interface a langgraph dev command line binds."""
    try:
        port = int(option(args, "--port") or DEFAULT_PORT)
    except ValueError:
        port = DEFAULT_PORT
    return port, option(args, "--host") or DEFAULT_HOST


def restart_hint(lasting: bool | None) -> str:
    """How to restart the server a record describes: the background server
    with ``EvoSci server stop``, which leaves one ``EvoSci deploy`` or the
    desktop app owns (*lasting* False) running."""
    if lasting is False:
        return "restart the EvoSci deploy or the desktop app that runs the server"
    return (
        "restart the server: EvoSci server stop; it starts again with your next session"
    )


def own_restart_hint() -> str:
    """:func:`restart_hint` for the server this process is."""
    lasting = (_written or {}).get("lasting")
    return restart_hint(lasting if isinstance(lasting, bool) else None)


def is_langgraph_dev(args: list[str]) -> bool:
    """Whether a command line is a langgraph dev's."""
    return any("langgraph" in arg for arg in args)


def _build() -> dict:
    """What the server records at once: only what is cheap to know."""
    lasting = os.environ.pop(LASTING_VAR, None)
    fingerprint = os.environ.pop(FINGERPRINT_VAR, None)
    process, found = _started_process()
    os.environ.pop(SPAWNER_PID_VAR, None)
    port, host = _address()
    record: dict = {
        "pid": process.pid,
        CREATE_TIME_KEY: process.create_time(),
        PHASE_KEY: STARTING,
        # Ctrl+Break reaches the process group the starter created, which the
        # started process leads (Windows).
        BREAK_GROUP_KEY: found and os.name == "nt",
        "data_dir": str(paths.DATA_DIR),
        "config_dir": str(paths.CONFIG_DIR),
        "host": host,
        "port": port,
    }
    if lasting is not None:
        record["lasting"] = lasting == "1"
    if fingerprint:
        record["config_fingerprint"] = fingerprint
    return record


def _resolved(resolve: Callable[[], str | None]) -> str | None:
    """What *resolve* gives for the drift checks; None when it fails, which
    only weakens those checks."""
    try:
        return resolve()
    except Exception:
        logger.warning("Could not resolve the agents' shell.", exc_info=True)
        return None


def _write(record: dict) -> None:
    """Write *record* atomically (a temp file, then ``os.replace``), so a
    reader never sees half a record."""
    target = record_path()
    tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(record), encoding="utf-8")
        _retried(lambda: os.replace(tmp, target))
    except OSError:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def load_json(path: Path) -> dict | None:
    """The JSON object in *path*; None when it is missing, unreadable or
    not an object."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_record() -> dict | None:
    """Write this server's record as ``starting``, right after it took the
    lock (so it replaces a record a crashed server left at once); it is
    removed as the process exits (:func:`remove_record`), or after a signal
    by the stop that ended it. Returns the record it replaced, which names
    what a killed server left running.

    Raises when it cannot, after logging why: a server without its record
    would hold the lock unseen, so it must not start.
    """
    global _written
    # What a server that held the lock before this one left, if any.
    left = load_json(record_path())
    try:
        record = _build()
        _write(record)
    except Exception:
        logger.exception("Could not write the server's record; it does not start.")
        raise
    _written = record
    return left


def record_ccproxy(pid: int, created: float | None) -> None:
    """Add the ccproxy this server started to its record, as soon as it
    runs: a server killed while ccproxy starts leaves it for the next one to
    end. A failure is logged only."""
    global _written
    if _written is None:
        return
    record = {**_written, CCPROXY_PID_KEY: pid, CCPROXY_CREATE_TIME_KEY: created}
    try:
        _write(record)
    except OSError:
        logger.warning("Could not record the server's ccproxy.", exc_info=True)
        return
    _written = record


def mark_serving() -> None:
    """Rewrite the record as ``serving``, once the server serves, with the
    ``python`` and shell its agents got, for the drift checks. A failure is
    logged only: a client finds the server answering all the same."""
    from ..agent_shell import SIDECAR_KEY as BASH_KEY
    from ..agent_shell import sidecar_bash
    from ..setup.research_env import SIDECAR_KEY as PYTHON_KEY
    from ..setup.research_env import agent_python

    global _written
    if _written is None:
        return
    record = {
        **_written,
        PHASE_KEY: SERVING,
        PYTHON_KEY: _resolved(agent_python),
        BASH_KEY: _resolved(sidecar_bash),
    }
    try:
        _write(record)
    except OSError:
        logger.warning("Could not mark the server's record serving.", exc_info=True)
        return
    _written = record


def _retried(change: Callable[[], object]) -> None:
    """Run *change* to the record, tried again for up to
    :data:`_REPLACE_WAIT`: on Windows replacing or deleting it fails while a
    client has it open, and clients read it every half second while a server
    starts."""
    deadline = time.monotonic() + _REPLACE_WAIT
    while True:
        try:
            change()
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(_REPLACE_RETRY)


def remove_record() -> None:
    """Remove the record while it is still this process's. A failure is
    logged only."""
    global _written
    if _written is None:
        return
    target = record_path()
    try:
        current = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _written = None
        return
    except (OSError, ValueError):
        current = None
    try:
        if current == _written:
            _retried(target.unlink)
    except OSError:
        logger.warning("Could not remove the server's record.", exc_info=True)
        return
    _written = None


# Registered when this module is imported, before the server's other exit
# handlers: they run in reverse order, so the record goes last, just before
# the lock goes with the process. A process that wrote no record removes
# nothing.
atexit.register(remove_record)
