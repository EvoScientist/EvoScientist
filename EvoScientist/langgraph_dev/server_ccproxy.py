"""The server's ccproxy, the local proxy Anthropic's OAuth routes through.

When Anthropic uses OAuth, the server starts ccproxy as its child first
thing in its lifespan (:func:`start_serving`), before it marks itself
serving: not while langgraph-api imports the app, which it times and flags
when slow. When :mod:`.http` is imported the server only ends the ccproxy a
killed predecessor left (its record names it) and notes that OAuth is on, so
a run that starts meanwhile waits for that start. It stops ccproxy as it
exits. ccproxy writes to ``ccproxy.log`` in the server home. Sessions point at
it on ``ccproxy_port``. A ccproxy that already answers there (one the user
runs) is used, not owned: the server neither records nor ends it.

When a configured model is Anthropic's, a server that cannot start its
ccproxy (not installed, not signed in, the port taken, never serving) leaves
the reason for its starter, tagged with its PID, and fails its start
(:class:`CcproxyStartError`). Otherwise it warns and serves without ccproxy.

A run start starts ccproxy when it is not running (:func:`ensure_serving`),
until :data:`_FAILED_RESTART_LIMIT` tries in a row failed; when it cannot,
the run fails by the same rule as the server's start.
"""

from __future__ import annotations

import atexit
import json
import logging
import subprocess
import sys
import threading

from .. import ccproxy_manager, paths
from ..config import get_effective_config
from ..config.settings import providers_in_use
from . import server_record

logger = logging.getLogger(__name__)


class CcproxyStartError(RuntimeError):
    """The server could not start the ccproxy a configured Anthropic model
    needs; raised out of its lifespan, so it does not serve."""


# How many failed starts in a row end the tries at run start, until the
# server restarts.
_FAILED_RESTART_LIMIT = 3

# ccproxy's port once the server found that Anthropic uses OAuth (None
# without OAuth).
_port: int | None = None
# Failed restarts in a row; a counter of restart attempts, and the reason the
# last one failed, for the run starts that waited on it.
_failed = 0
_attempts = 0
_last_error: str | None = None
# Whether a ccproxy that answers without the process this server started
# (one the user runs, a launcher that exited with its child serving) was
# logged.
_noted_orphan = False
# Whether the server ended a ccproxy its predecessor left, whose port then
# needs a moment to come free for the first start.
_ended_left = False
# Whether no start was tried yet: the one as the server starts.
_first = True

_proc: subprocess.Popen | None = None
# Set as the server exits: a ccproxy that ends then is not started again.
_stopping = False
# Guards _proc and _stopping between a run start's restart and the exit's
# stop; _restart_lock lets concurrent run starts restart ccproxy once.
_lock = threading.Lock()
_restart_lock = threading.Lock()
# Windows: the job that ends ccproxy with the server, however the server ends.
_job = None


def start(left: dict | None) -> None:
    """As :mod:`.http` is imported: end the ccproxy the record *left* by the
    previous server names, and note whether Anthropic uses OAuth; the start
    itself is :func:`start_serving`'s.

    The previous one is ended whatever this server uses: its record is
    replaced by now, so no later server would find it."""
    global _port, _ended_left
    from .manager import RUNTIME

    # Only this start's reason may reach its starter.
    try:
        RUNTIME.ccproxy_failure.unlink(missing_ok=True)
    except OSError:
        logger.warning("Could not remove %s", RUNTIME.ccproxy_failure)
    config = get_effective_config()
    ended = _end_left(left, config.ccproxy_port)
    if not ccproxy_manager.uses_oauth(config):
        return
    _port = config.ccproxy_port
    _ended_left = ended
    # Registered after the record's removal, so it runs before it.
    atexit.register(stop)


def start_serving() -> None:
    """First thing in the server's lifespan: start ccproxy when Anthropic
    uses OAuth, as a run start would (runs that start meanwhile share the
    outcome). Blocking.

    When a configured model is Anthropic's and ccproxy cannot start, leaves
    the reason for the starter and raises :class:`CcproxyStartError`.
    Otherwise it only warns and the server serves without ccproxy: ``/model``
    can still pick an Anthropic model for a run, but a stale OAuth setting
    must not keep the server from starting."""
    error = _ensure()
    if error is None:
        return
    if "anthropic" in providers_in_use(get_effective_config()):
        _tell_starter(error)
        raise CcproxyStartError(error)
    logger.warning(
        "Anthropic OAuth is set, but ccproxy could not start: %s\n"
        "An Anthropic model (/model) will not work until it can; each run "
        "start tries to start it.",
        error,
    )


def _tell_starter(reason: str) -> None:
    """Leave *reason* for the process that started the server, tagged with
    this server's PID, so a failed start of a later server is never read as
    this one's (see :func:`~.manager.ccproxy_failure_of`)."""
    from .manager import RUNTIME

    try:
        RUNTIME.ccproxy_failure.write_text(
            json.dumps({"pid": server_record.recorded_pid(), "reason": reason}),
            encoding="utf-8",
        )
    except OSError:
        logger.warning("Could not write %s", RUNTIME.ccproxy_failure)


def _start(port: int, *, ended: bool, taken_hint: str = "") -> None:
    """Check the port and the sign-in, start ccproxy and point the server's
    clients at it; *taken_hint* follows the message when the port is taken."""
    if not 1 <= port <= 65535:
        raise RuntimeError(
            f"Invalid ccproxy port: {port}. Must be between 1 and 65535."
        )
    # The port first: checking the sign-in can take seconds.
    if not _port_free(port, ended=ended):
        raise RuntimeError(port_taken_message(port) + taken_hint)
    ccproxy_manager.check_ccproxy_ready()
    _launch(port)
    # Not set yet when ccproxy could not start with the server.
    ccproxy_manager.setup_ccproxy_env(port)
    logger.info("Started ccproxy on port %d", port)


def needed() -> bool:
    """Whether Anthropic uses OAuth on this server, so runs need ccproxy."""
    return _port is not None


def ensure_serving() -> None:
    """Start ccproxy before a run when Anthropic uses OAuth and it is not
    running. Blocking.

    When it cannot start, the run gets the reason as a :class:`RuntimeError`
    if a configured model (main, auxiliary, fallback) is Anthropic's, as the
    server's start fails then; otherwise the failure is only logged and the
    run goes on (an Anthropic model it picks with ``/model`` then cannot be
    reached). Run starts that wait on one start share its outcome. After
    :data:`_FAILED_RESTART_LIMIT` failures in a row nothing is tried until
    the server restarts.
    """
    error = _ensure()
    if error is not None and "anthropic" in providers_in_use(get_effective_config()):
        raise RuntimeError(error)


def ensure_for_session() -> str | None:
    """Start ccproxy for a session that routes Anthropic through it (OAuth)
    and found nothing answering, as a run start would; why it does not
    serve, or None. Blocking."""
    hint = server_record.own_restart_hint()
    if _port is None:
        return (
            "The server was started without Anthropic OAuth, so it runs no "
            f"ccproxy; {hint}."
        )
    with _lock:
        stopping = _stopping
    if stopping:
        return "The server is stopping; the next session starts it again."
    error = _ensure()
    if error is None and not ccproxy_manager.is_ccproxy_running(_port):
        return f"ccproxy runs but does not answer on port {_port}; {hint}."
    return error


def _ensure() -> str | None:
    """Start ccproxy when Anthropic uses OAuth and it is not running; why it
    does not serve, or None."""
    port = _port
    if port is None:
        return None
    with _lock:
        state = _state()
        seen = _attempts
        # A restart under way has a process before it serves: wait for it
        # and share its outcome.
        if state == _SKIP and not _stopping and _restart_lock.locked():
            state = _RESTART
    if state == _SKIP:
        return None
    if state == _GAVE_UP:
        return _gave_up_message()
    return _restart_once(port, seen)


# What a run start finds (:func:`_state`).
_SKIP = "skip"
_GAVE_UP = "gave up"
_RESTART = "restart"


def _state() -> str:
    """:data:`_SKIP` while the server stops or ccproxy runs,
    :data:`_GAVE_UP` after too many failures, else :data:`_RESTART`. Called
    with :data:`_lock` held."""
    if _stopping or (_proc is not None and _proc.poll() is None):
        return _SKIP
    if _failed >= _FAILED_RESTART_LIMIT:
        return _GAVE_UP
    return _RESTART


def _restart_once(port: int, seen: int) -> str | None:
    """Start ccproxy, unless a start ran since attempt *seen*; the reason
    ccproxy does not serve, or None."""
    global _attempts, _failed, _last_error, _noted_orphan, _ended_left, _first
    with _restart_lock:
        with _lock:
            state = _state()
            if state == _SKIP:
                return None
            if _attempts != seen:
                return _last_error
            if state == _GAVE_UP:
                return _gave_up_message()
            first, _first = _first, False
            # One that exited, or the predecessor's just ended, frees its
            # port a moment later.
            exited = _proc is not None or _ended_left
            _ended_left = False
        if ccproxy_manager.is_ccproxy_running(port):
            # One the user runs, or on Windows the child of a ccproxy.exe
            # launcher that exited: used, not owned.
            if not _noted_orphan:
                _noted_orphan = True
                logger.warning(
                    "A ccproxy already answers on port %d; using it. The "
                    "server did not start it and does not stop it.",
                    port,
                )
            # Not set yet when ccproxy could not start with the server.
            ccproxy_manager.setup_ccproxy_env(port)
            return None
        try:
            _restart(port, exited=exited, first=first)
        except (RuntimeError, OSError) as exc:
            with _lock:
                if _stopping:
                    return None
                _attempts += 1
                _failed += 1
                _last_error = str(exc)
                failed = _failed
            logger.warning("Could not start ccproxy: %s", exc)
            if failed >= _FAILED_RESTART_LIMIT:
                logger.error("%s", _gave_up_message())
            return str(exc)
        with _lock:
            _attempts += 1
            _failed = 0
            _last_error = None
        _noted_orphan = False
        return None


def _gave_up_message() -> str:
    from .manager import RUNTIME

    return (
        f"ccproxy could not be started after {_FAILED_RESTART_LIMIT} tries. "
        "Fix the cause (the server log has the reason, and "
        f"{RUNTIME.ccproxy_log} if it ran), then "
        f"{server_record.own_restart_hint()}."
    )


def _restart(port: int, *, exited: bool, first: bool = False) -> None:
    """Start ccproxy, *first* as the server starts, else again; *exited*
    when a ccproxy on its port just ended."""
    if first:
        logger.info("Starting ccproxy on port %d.", port)
        _start(port, ended=exited)
        return
    logger.warning("ccproxy is not running; starting it on port %d.", port)
    _start(
        port,
        ended=exited,
        taken_hint=f" Then {server_record.own_restart_hint()}.",
    )


def _launch(port: int) -> None:
    """Start ccproxy, record it and wait until it serves; raises
    :class:`RuntimeError` naming its log when it does not."""
    from .manager import RUNTIME, _create_time

    global _proc
    proc = _spawn(port)
    with _lock:
        stopping = _stopping
        if not stopping:
            _proc = proc
    if stopping:
        ccproxy_manager.stop_ccproxy_tree(proc)
        raise RuntimeError("The server is stopping")
    _join_job(proc)
    server_record.record_ccproxy(proc.pid, _create_time(proc.pid))
    try:
        ccproxy_manager.wait_for_ccproxy(proc, port)
    except RuntimeError as exc:
        # Stopped and reaped by the wait: nothing left for the exit's stop.
        with _lock:
            if _proc is proc:
                _proc = None
        raise RuntimeError(f"{exc}. See {RUNTIME.ccproxy_log}") from exc


def _spawn(port: int) -> subprocess.Popen:
    """Start ccproxy with its output in its own log in the server home,
    rotated first as the server's is."""
    from .manager import RUNTIME, _rotate_log_if_needed

    _rotate_log_if_needed(RUNTIME.ccproxy_log)
    # The child keeps its own copy of the file; this one closes.
    with open(RUNTIME.ccproxy_log, "ab") as log:
        return ccproxy_manager.spawn_ccproxy(port, output=log)


def port_taken_message(port: int) -> str:
    return (
        f"ccproxy_port {port} is in use by another program that is not a "
        "ccproxy answering there, so the server cannot start its ccproxy "
        "(Anthropic OAuth). Pick "
        "another port with `EvoSci config set ccproxy_port <port>`, or stop "
        "that program. An exported EVOSCIENTIST_CCPROXY_PORT overrides the "
        "setting; if you use it, set it for every command of the data folder "
        f"{paths.DATA_DIR}."
    )


def _end_left(left: dict | None, port: int) -> bool:
    """End the ccproxy a killed server left: the one its record names, and
    one an older release started for every session (``ccproxy.pid``);
    whether it ended one. *port* is ``ccproxy_port``."""
    ended = False
    if left:
        pid = end_recorded(left, port)
        if pid is not None:
            logger.info("Ended ccproxy %d, which the previous server left.", pid)
            ended = True
    pid = ccproxy_manager.stop_recorded_ccproxy()
    if pid is not None:
        logger.info("Ended ccproxy %d, which an older release started.", pid)
        ended = True
    return ended


def end_recorded(record: dict, port: int) -> int | None:
    """End the ccproxy a server's *record* names, if it still runs; its pid
    when it did. *port* is ``ccproxy_port``."""
    pid = record.get(server_record.CCPROXY_PID_KEY)
    created = record.get(server_record.CCPROXY_CREATE_TIME_KEY)
    if (
        isinstance(pid, int)
        and (created is None or isinstance(created, int | float))
        and ccproxy_manager.end_ccproxy(pid, created, port)
    ):
        return pid
    return None


def _port_free(port: int, *, ended: bool) -> bool:
    """Whether ccproxy can bind *port* on the loopback; given a moment to
    come free when a ccproxy on it was just *ended*."""
    from .manager import _can_bind_port, _is_port_occupied, _wait_for_port_bindable

    host = "127.0.0.1"
    if ended:
        free = _wait_for_port_bindable(port, host)
    else:
        free = _can_bind_port(port, host)
    # The loopback bind can pass beside another program's listener on the
    # wildcard address (macOS, Windows); that listener accepts a connection.
    if free:
        free = not _is_port_occupied(port, host)
    return free


def _join_job(proc: subprocess.Popen) -> None:
    """On Windows, put ccproxy in a job that ends it when the server ends,
    also when the server is killed; one job for every ccproxy it starts."""
    global _job
    from ..deploy.backend_job import backend_job

    job = _job if _job is not None else backend_job()
    if job is None:
        return
    warned = job.warning is not None
    job.adopt(proc)
    if job.warning is not None and not warned:
        logger.warning(
            "Could not put ccproxy in a Windows job object; if the server is "
            "killed, ccproxy runs on until the next server starts."
        )
    _job = job


def stop() -> None:
    """Stop the server's ccproxy and what it started, as the server exits;
    it is not started again."""
    global _proc, _stopping
    with _lock:
        _stopping = True
        proc, _proc = _proc, None
    ccproxy_manager.stop_ccproxy_tree(proc)


def request_stop() -> None:
    """Ask ccproxy to stop as the server's lifespan ends, and start it no
    more: a plain SIGTERM ends the process before :func:`stop` runs at exit
    (uvicorn raises the signal again after its shutdown). Not waited for:
    ccproxy finishes the calls it serves, for runs cancelled only after the
    lifespan. On Windows ``terminate()`` would end only the ``ccproxy.exe``
    launcher; the job object ends the rest."""
    global _stopping
    with _lock:
        _stopping = True
        proc = _proc
    if proc is not None and sys.platform != "win32":
        proc.terminate()
