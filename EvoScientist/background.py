"""Background OS-process execution for the sandbox.

A *process* here is a single detached OS process launched via ``run_in_background``
(distinct from an async sub-agent *task* and a future cron *schedule* — the word
"job" is intentionally never used).

The registry is **module-global (process-level)**: processes survive ``/new`` and
``/resume`` within the same CLI process, but are not persisted across a CLI restart.
The live ``Popen`` handle is held so ``poll()`` / ``returncode`` stay authoritative
(no PID-reuse risk).

Command validation and cwd resolution happen at the tool layer
(``middleware/background.py``); this module is the pure execution + tracking mechanism
and is safe to unit-test on its own. A future scheduler (cron) would reuse ``launch``.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

logger = logging.getLogger(__name__)

_BG_DIRNAME = ".bg_processes"
_KILL_GRACE_SECONDS = 2.0


@dataclass
class BgProcess:
    """A tracked background OS process."""

    process_id: str
    name: str
    command: str
    popen: subprocess.Popen
    pid: int
    log_path: Path
    started_at: str  # ISO-8601 UTC (record/display)
    started_ts: float  # epoch seconds (elapsed computation)
    origin_thread_id: str | None = None  # CLI thread/session that launched it
    returncode: int | None = None
    finished_at: str | None = None
    finished_ts: float | None = None  # epoch at exit; freezes elapsed once done
    stopped: bool = False  # set by stop(); suppresses the completion notification
    # epoch each thread last checked this process (status/list); keyed by thread_id
    # so a check from one session can't dedup another session's completion ping.
    last_checked_by_thread: dict[str | None, float] = field(default_factory=dict)


_PROCESSES: dict[str, BgProcess] = {}
_LOCK = threading.Lock()


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _record_exit(proc: BgProcess) -> None:
    """Record terminal state on first observed exit. Caller MUST hold ``_LOCK``.

    ``finished_ts`` is set when the exit is first observed. The per-process daemon
    watcher (:func:`_watch`) calls this right after ``popen.wait()`` returns, so in
    practice ``finished_ts`` ≈ the real exit time. Calls from ``status`` / ``list_all`` /
    ``stop`` are a fallback for the brief window before the watcher runs.
    """
    rc = proc.popen.poll()
    if rc is not None and proc.returncode is None:
        proc.returncode = rc
        proc.finished_at = _now_iso()
        proc.finished_ts = time.time()


def _elapsed(proc: BgProcess) -> int:
    """Seconds the process has run — frozen at first-observed exit once it has exited."""
    end = proc.finished_ts if proc.finished_ts is not None else time.time()
    return int(end - proc.started_ts)


def terminal_status(proc: BgProcess) -> str | None:
    """Map a finished process's returncode to a notification status, or ``None`` if running.

    Shared by the ``bg_processes`` state snapshot the tools write, the
    :func:`poll_status` accessor the gateway reads, and the client-side completion
    notification, so all three agree on the same terminal vocabulary
    (``success`` / ``error`` / ``interrupted``, aligned with
    ``async_notifier.TERMINAL_STATUSES``).
    """
    rc = proc.returncode
    if rc is None:
        return None
    if rc == 0:
        return "success"
    if rc < 0:
        return "interrupted"  # terminated by a signal
    return "error"


def poll_status(process_id: str) -> str:
    """Live status of ``process_id`` for the state-driven client reader.

    Records the exit first (like :func:`status`), then returns ``"running"`` or the
    terminal status. Returns ``"unknown"`` when the process is not in the registry
    (e.g. a server restart cleared it) so the reader treats it as not-yet-notifiable.
    """
    with _LOCK:
        proc = _PROCESSES.get(process_id)
        if proc is None:
            return "unknown"
        _record_exit(proc)
        return terminal_status(proc) or "running"


def was_observed_done(process_id: str, origin_thread_id: str | None = None) -> bool:
    """True if ``origin_thread_id`` already saw this process's completion itself.

    i.e. the process has exited AND was checked (``status``/``list_all``) from that thread
    at or after it finished. Used to dedup the completion notification (routed to the
    launching thread), so a check from a *different* session can't suppress it.
    """
    with _LOCK:
        proc = _PROCESSES.get(process_id)
        if proc is None or proc.finished_ts is None:
            return False
        seen_ts = proc.last_checked_by_thread.get(origin_thread_id)
        return seen_ts is not None and seen_ts >= proc.finished_ts


def _record_dict(proc: BgProcess) -> dict[str, Any]:
    """Plain snapshot of a process for the ``bg_processes`` thread-state channel."""
    return {
        "process_id": proc.process_id,
        "name": proc.name,
        "command": proc.command,
        "pid": proc.pid,
        "status": terminal_status(proc) or "running",
        "returncode": proc.returncode,
        "started_at": proc.started_at,
        "origin_thread_id": proc.origin_thread_id,
    }


def state_record(process_id: str) -> dict[str, Any] | None:
    """Structured mirror of a single process for ``bg_processes`` state.

    Records the exit first (so ``status`` reflects a just-finished process), then
    returns the fields the client reader needs to detect completion and build the
    notification. ``None`` when the process is not tracked. When ``status`` is
    terminal, writing this into thread state marks the process as observed — the
    reader skips terminal-in-state records, exactly like the async-task reader.
    """
    with _LOCK:
        proc = _PROCESSES.get(process_id)
        if proc is None:
            return None
        _record_exit(proc)
        return _record_dict(proc)


def list_records(
    thread_id: str | None = None, *, include_all: bool = False
) -> list[dict[str, Any]]:
    """Mirror records for the thread's tracked processes (scoping mirrors :func:`list_all`)."""
    with _LOCK:
        procs = [
            p
            for p in _PROCESSES.values()
            if include_all or p.origin_thread_id == thread_id
        ]
        for p in procs:
            _record_exit(p)
        return [_record_dict(p) for p in procs]


def _read_tail(log_path: Path, tail_bytes: int) -> str:
    # Seek from the end so a huge log isn't fully read into memory on each status check.
    try:
        with log_path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size == 0:
                return "(no output yet)"
            if size > tail_bytes:
                f.seek(-tail_bytes, os.SEEK_END)
                return "...(truncated)...\n" + f.read().decode("utf-8", "replace")
            f.seek(0)
            data = f.read()
    except OSError:
        return "(no output captured yet)"
    return data.decode("utf-8", "replace")


def _watch(proc: BgProcess, on_exit: Callable[[BgProcess], None] | None) -> None:
    """Block until ``proc`` exits, record the exit promptly, then fire ``on_exit``.

    Running in a daemon thread, ``popen.wait()`` lets us record ``finished_ts`` at (very
    close to) the real exit time — fixing the observation-time inflation — and gives a
    hook the CLI layer wires to a completion notification, without ``background.py``
    importing the notifier (kept decoupled via the callback).
    """
    try:
        proc.popen.wait()
    except Exception:
        pass
    with _LOCK:
        _record_exit(proc)
    if on_exit is not None:
        try:
            on_exit(proc)
        except Exception:
            logger.warning("background on_exit callback failed", exc_info=True)


def launch(
    command: str,
    cwd: str,
    name: str | None = None,
    *,
    origin_thread_id: str | None = None,
    on_exit: Callable[[BgProcess], None] | None = None,
) -> str:
    """Launch ``command`` detached in ``cwd``; return a short ``process_id``.

    The command is run via ``shell=True`` with output redirected to a per-process log
    file under ``<cwd>/.bg_processes/`` and ``start_new_session=True`` so the child is a
    process-group leader (survives this call's return and can be killed as a group).
    The caller is responsible for validating ``command`` first.

    ``origin_thread_id`` records the launching CLI session so ``list_all`` can scope to it.
    ``on_exit`` (optional) is called with the ``BgProcess`` from a daemon watcher thread
    once the process exits — used by the CLI layer to emit a completion notification.
    """
    process_id = uuid.uuid4().hex[:8]
    log_dir = Path(cwd) / _BG_DIRNAME
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{process_id}.log"

    log_file = open(log_path, "w")
    try:
        popen = subprocess.Popen(
            command,
            shell=True,
            cwd=cwd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    finally:
        # The child inherited its own dup of the fd during spawn; the parent's copy
        # is no longer needed (and must be closed so the pipe/file isn't held open).
        log_file.close()

    proc = BgProcess(
        process_id=process_id,
        name=name or command[:40],
        command=command,
        popen=popen,
        pid=popen.pid,
        log_path=log_path,
        started_at=_now_iso(),
        started_ts=time.time(),
        origin_thread_id=origin_thread_id,
    )
    with _LOCK:
        _PROCESSES[process_id] = proc
    # Daemon watcher: records the precise exit time and fires on_exit when done.
    threading.Thread(target=_watch, args=(proc, on_exit), daemon=True).start()
    return process_id


def status(
    process_id: str, *, thread_id: str | None = None, tail_bytes: int = 16_000
) -> str:
    """Return a human-readable status + recent output tail for ``process_id``."""
    with _LOCK:
        proc = _PROCESSES.get(process_id)
        if proc is None:
            return (
                f"No such background process: {process_id!r}. "
                "Use list_processes to see tracked processes."
            )
        _record_exit(proc)
        proc.last_checked_by_thread[thread_id] = time.time()  # this thread observed it
        running = proc.returncode is None
        elapsed = _elapsed(proc)
        name, pid, command, returncode, log_path = (
            proc.name,
            proc.pid,
            proc.command,
            proc.returncode,
            proc.log_path,
        )
    if running:
        head = f"Process {process_id} (name={name!r}) RUNNING — {elapsed}s elapsed, pid {pid}."
    else:
        head = f"Process {process_id} (name={name!r}) EXITED code {returncode} after ~{elapsed}s."
    tail = _read_tail(log_path, tail_bytes)  # file IO outside the lock
    return (
        f"{head}\nCommand: {command}\n--- output (last {tail_bytes} bytes) ---\n{tail}"
    )


def _kill_process_tree(popen: subprocess.Popen, *, forceful: bool) -> None:
    """Kill the process group/tree in a cross-platform way.

    On POSIX ``start_new_session=True`` makes the child a process-group
    leader; ``os.killpg`` terminates the entire group (shell + any
    grandchildren).  On Windows ``TerminateProcess`` (used by
    ``Popen.terminate()`` / ``Popen.kill()``) only kills the direct
    child — it does *not* cascade to grandchildren.  We use ``psutil``
    to walk the process tree and signal every descendant.
    """
    if os.name == "nt":
        try:
            proc = psutil.Process(popen.pid)
            targets = [proc, *proc.children(recursive=True)]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return
        for p in targets:
            try:
                if forceful:
                    p.kill()
                else:
                    p.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    else:
        sig = signal.SIGKILL if forceful else signal.SIGTERM
        try:
            os.killpg(os.getpgid(popen.pid), sig)
        except ProcessLookupError:
            pass


def stop(process_id: str) -> str:
    """Terminate ``process_id`` and its process group (SIGTERM, then SIGKILL)."""
    with _LOCK:
        proc = _PROCESSES.get(process_id)
        if proc is None:
            return f"No such background process: {process_id!r}."
        if proc.popen.poll() is not None:
            _record_exit(proc)
            return f"Process {process_id} already finished (code {proc.returncode})."
        # Mark as user-stopped so the watcher's on_exit suppresses the completion
        # notification (the user already knows — no need to ping them).
        proc.stopped = True
        # The watcher's popen.wait() reaps without the lock, so a tiny PID-reuse race
        # remains (getpgid on a recycled pid).  On POSIX ProcessLookupError covers the
        # common case; on Windows ``Popen.terminate()`` is a no-op on a dead handle
        # so we poll after the call instead.
        _kill_process_tree(proc.popen, forceful=False)
        if proc.popen.poll() is not None:
            _record_exit(proc)
            return f"Process {process_id} is no longer running."

    deadline = time.time() + _KILL_GRACE_SECONDS
    while time.time() < deadline:
        with _LOCK:
            if proc.popen.poll() is not None:
                _record_exit(proc)
                break
        time.sleep(0.1)
    else:
        with _LOCK:
            if proc.popen.poll() is None:
                _kill_process_tree(proc.popen, forceful=True)
            _record_exit(proc)

    with _LOCK:
        _record_exit(proc)
        name = proc.name
    return f"Stopped background process {process_id} (name={name!r})."


def list_all(thread_id: str | None = None, *, include_all: bool = False) -> str:
    """List tracked background processes with live statuses.

    Scoped to the launching session (``thread_id``) unless ``include_all`` is set.
    """
    with _LOCK:
        all_procs = list(_PROCESSES.values())
        procs = (
            all_procs
            if include_all
            else [p for p in all_procs if p.origin_thread_id == thread_id]
        )
        if not procs:
            if all_procs and not include_all:
                return (
                    "No background processes in this session "
                    f"({len(all_procs)} in other sessions — pass all_threads=True to see them)."
                )
            return "No background processes tracked."
        lines = []
        now = time.time()
        for p in procs:
            _record_exit(p)
            p.last_checked_by_thread[thread_id] = now  # this thread observed it
            state = "RUNNING" if p.returncode is None else f"exited({p.returncode})"
            lines.append(
                f"  {p.process_id}  {state:12}  {_elapsed(p)}s  name={p.name!r}"
            )
        return f"{len(procs)} background process(es):\n" + "\n".join(lines)
