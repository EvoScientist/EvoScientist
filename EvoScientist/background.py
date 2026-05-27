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

import os
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

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
    returncode: int | None = None
    finished_at: str | None = None
    finished_ts: float | None = None  # epoch at exit; freezes elapsed once done


_PROCESSES: dict[str, BgProcess] = {}
_LOCK = threading.Lock()


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _record_exit(proc: BgProcess) -> None:
    """Record terminal state on first observed exit. Caller MUST hold ``_LOCK``.

    ``finished_ts`` is when we first *observed* the exit (via ``poll()``), not the true
    OS exit time — a process polled long after it ended shows an inflated elapsed.
    Precise exit timing needs a per-process watcher (deferred to the Phase 3 completion
    notifier); for Phase 2 this observation-time approximation is acceptable.
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


def _read_tail(log_path: Path, tail_bytes: int) -> str:
    try:
        data = log_path.read_bytes()
    except OSError:
        return "(no output captured yet)"
    if not data:
        return "(no output yet)"
    if len(data) > tail_bytes:
        return "...(truncated)...\n" + data[-tail_bytes:].decode("utf-8", "replace")
    return data.decode("utf-8", "replace")


def launch(command: str, cwd: str, name: str | None = None) -> str:
    """Launch ``command`` detached in ``cwd``; return a short ``process_id``.

    The command is run via ``shell=True`` with output redirected to a per-process log
    file under ``<cwd>/.bg_processes/`` and ``start_new_session=True`` so the child is a
    process-group leader (survives this call's return and can be killed as a group).
    The caller is responsible for validating ``command`` first.
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
    )
    with _LOCK:
        _PROCESSES[process_id] = proc
    return process_id


def status(process_id: str, *, tail_bytes: int = 16_000) -> str:
    """Return a human-readable status + recent output tail for ``process_id``."""
    with _LOCK:
        proc = _PROCESSES.get(process_id)
        if proc is None:
            return (
                f"No such background process: {process_id!r}. "
                "Use list_processes to see tracked processes."
            )
        _record_exit(proc)
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


def stop(process_id: str) -> str:
    """Terminate ``process_id`` and its process group (SIGTERM, then SIGKILL)."""
    with _LOCK:
        proc = _PROCESSES.get(process_id)
        if proc is None:
            return f"No such background process: {process_id!r}."
        if proc.popen.poll() is not None:
            _record_exit(proc)
            return f"Process {process_id} already finished (code {proc.returncode})."
        # Holding the lock across poll()+killpg() means no concurrent caller can
        # poll()/reap this child in between, so the pid can't be recycled before we
        # signal it — closes the PID-reuse window.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
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
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            _record_exit(proc)

    with _LOCK:
        _record_exit(proc)
        name = proc.name
    return f"Stopped background process {process_id} (name={name!r})."


def list_all() -> str:
    """List all tracked background processes with live statuses."""
    with _LOCK:
        procs = list(_PROCESSES.values())
        if not procs:
            return "No background processes tracked."
        lines = []
        for p in procs:
            _record_exit(p)
            state = "RUNNING" if p.returncode is None else f"exited({p.returncode})"
            lines.append(
                f"  {p.process_id}  {state:12}  {_elapsed(p)}s  name={p.name!r}"
            )
        return f"{len(procs)} background process(es):\n" + "\n".join(lines)
