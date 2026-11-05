"""Stand-ins for the langgraph dev server: its record, its process, its
locks and its spawn."""

from __future__ import annotations

import contextlib
import importlib
import json
import subprocess
import threading
from pathlib import Path

import psutil

from EvoScientist import paths
from EvoScientist.langgraph_dev import manager, server_home, server_record

# The creation time the fake processes report.
CREATED = 1_700_000_000.0


@contextlib.contextmanager
def reloaded(module):
    """*module* imported again, then put back as it was: a reload makes new
    classes and functions, and objects of the old ones, built by other tests
    at collection, would no longer pass checks against them."""
    saved = dict(vars(module))
    try:
        yield importlib.reload(module)
    finally:
        vars(module).clear()
        vars(module).update(saved)


class LockHolder:
    """Another process: holds the start lock, or with ``server`` the server
    lock (a server running or exiting), until released."""

    def __init__(self, *, server: bool = False) -> None:
        from filelock import FileLock

        server_home.ensure_server_home()
        path = (
            paths.SERVER_HOME / server_home.LOCK_FILE
            if server
            else manager.RUNTIME.lock_file
        )
        self.lock = FileLock(str(path))
        self.held = threading.Event()
        self.release = threading.Event()
        self.thread = threading.Thread(target=self._hold, daemon=True)
        self.thread.start()
        assert self.held.wait(5)

    def _hold(self) -> None:
        with self.lock:
            self.held.set()
            self.release.wait(10)

    def done(self) -> None:
        self.release.set()
        self.thread.join(5)


class Processes:
    """The processes psutil sees, by PID: a live langgraph dev on a port,
    created at :data:`CREATED` unless told otherwise. Any other PID is gone,
    and holding it fails as on Windows. The command line of a PID in
    ``denied`` cannot be read."""

    def __init__(self, monkeypatch) -> None:
        self.live: dict[int, tuple[float, list[str]]] = {}
        self.denied: set[int] = set()
        monkeypatch.setattr(manager, "_create_time", self._create_time)
        monkeypatch.setattr(manager, "_process_cmdline", self._process_cmdline)
        monkeypatch.setattr(manager, "_langgraph_cmdline", self._cmdline)
        monkeypatch.setattr(manager, "_hold_pid", self._hold_pid)

    def add(
        self,
        pid: int,
        *,
        port: int = 6174,
        created: float = CREATED,
        cmdline: list[str] | None = None,
    ) -> None:
        self.live[pid] = (
            created,
            cmdline or ["langgraph", "dev", "--port", str(port)],
        )

    def end(self, pid: int) -> None:
        self.live.pop(pid, None)

    @contextlib.contextmanager
    def _hold_pid(self, pid: int):
        if pid not in self.live:
            raise psutil.NoSuchProcess(pid)
        yield

    def _create_time(self, pid: int) -> float | None:
        entry = self.live.get(pid)
        return entry[0] if entry else None

    def _process_cmdline(self, pid: int) -> list[str] | None:
        if pid in self.denied:
            raise psutil.AccessDenied(pid)
        entry = self.live.get(pid)
        return entry[1] if entry else None

    def _cmdline(self, pid: object) -> list[str] | None:
        if pid in self.denied:
            return None
        entry = self.live.get(pid) if isinstance(pid, int) else None
        if entry is None or not any("langgraph" in arg for arg in entry[1]):
            return None
        return entry[1]


def write_record(
    pid: int = 4242,
    *,
    port: int = 6174,
    host: str = "127.0.0.1",
    created: float = CREATED,
    phase: str = server_record.SERVING,
    **fields,
) -> dict:
    """Write the server's record as the server does."""
    record = {
        "pid": pid,
        server_record.CREATE_TIME_KEY: created,
        server_record.PHASE_KEY: phase,
        server_record.BREAK_GROUP_KEY: False,
        "data_dir": str(paths.DATA_DIR),
        "config_dir": str(paths.CONFIG_DIR),
        "host": host,
        "port": port,
        **fields,
    }
    target = manager.RUNTIME.record
    server_home.ensure_server_home()
    target.write_text(json.dumps(record), encoding="utf-8")
    return record


def lock_follows_record(
    monkeypatch, held: set | None = None, dead: set | None = None
) -> None:
    """The server lock is held while a record is written, as a server holds
    it, and while *held* (a set a test fills) says so. Every PID is alive
    but those in *dead*."""
    held = held if held is not None else set()
    dead = dead if dead is not None else set()
    monkeypatch.setattr(
        server_home,
        "lock_state",
        lambda wait=0: (
            server_home.HELD
            if held or manager.RUNTIME.record.exists()
            else server_home.FREE
        ),
    )
    monkeypatch.setattr(
        manager, "_runs_as_recorded", lambda record: record.pid not in dead
    )


def write_older_record(pid: int = 4141, *, port: int = 6174, **fields) -> Path:
    """Write the PID file and record an older release kept for the server,
    for this data folder unless *fields* says otherwise; their folder."""
    folder = manager.OLDER_RUNTIME_DIR
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "langgraph_dev.pid").write_text(str(pid), encoding="utf-8")
    record = {
        "pid": pid,
        "data_dir": str(paths.DATA_DIR),
        "config_dir": str(paths.CONFIG_DIR),
        "port": port,
        **fields,
    }
    (folder / server_record.RECORD_FILE).write_text(
        json.dumps(record), encoding="utf-8"
    )
    return folder


class FakeProc:
    """A started server's ``Popen``: runs until it is ended or exits."""

    def __init__(self, pid: int = 4242, returncode: int | None = None) -> None:
        self.pid = pid
        self.returncode = returncode

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int | None:
        return self.returncode

    def exit(self, code: int) -> None:
        self.returncode = code


class Server:
    """What :func:`manager.ensure_server` sees of servers and ports.

    ``answering`` holds the ports where a server answers ``/ok``, ``whose``
    what ``/api/instance`` says of the server on the port (``own``, ``other``
    or ``older``); ``silent`` the ports that take a connection but give no
    answer, ``occupied`` those another program listens on, and ``locked``
    whether a server holds this data folder's lock without a record (a
    written record holds it too). A start (``_spawn_server``) records a
    :class:`FakeProc` and, unless ``exit_code`` is set or ``records`` is
    False, writes its record and answers on its port; the calls are kept in
    ``spawns``. Waits poll quickly.
    """

    def __init__(self, monkeypatch, *, pid: int = 4242) -> None:
        self.processes = Processes(monkeypatch)
        self.answering: set[int] = set()
        self.silent: set[int] = set()
        self.occupied: set[int] = set()
        # Ports a start cannot bind (a listener still closing).
        self.unbindable: set[int] = set()
        self.whose: dict[int, str] = {}
        self.held: set = set()
        # PIDs no process has.
        self.dead: set[int] = set()
        self.spawns: list[dict] = []
        self.procs: list[FakeProc] = []
        self.stopped: list[int] = []
        self.pid = pid
        self.exit_code: int | None = None
        self.records = True
        monkeypatch.setattr(manager, "_POLL_INTERVAL", 0.001)
        monkeypatch.setattr(manager, "_START_BUDGET", 0.2)
        monkeypatch.setattr(manager, "_DRAIN_WAIT", 0.1)
        monkeypatch.setattr(manager, "_SETTLE_WAIT", 0.05)
        monkeypatch.setattr(manager, "_install_node_for_mcp", lambda waits: None)
        monkeypatch.setattr(
            manager,
            "_wait_for_port_bindable",
            lambda port, host="127.0.0.1": port not in self.unbindable | self.occupied,
        )
        monkeypatch.setattr(manager, "_server_config_fingerprint", lambda _c: "fp")
        monkeypatch.setattr(manager, "is_langgraph_dev_running", self._running)
        monkeypatch.setattr(manager, "_is_port_occupied", self._occupied)
        monkeypatch.setattr(manager, "ask", self._ask)
        lock_follows_record(monkeypatch, self.held, self.dead)
        monkeypatch.setattr(manager, "_spawn_server", self._spawn)
        monkeypatch.setattr(manager, "_end_server_tree", self._end)
        # Called at every pause of a wait, with the pause's number, so a test
        # can change what the next look finds.
        self.on_pause = None
        self.pauses = 0
        pause = manager._Waits.pause

        def _pause(waits, *, for_other=False):
            self.pauses += 1
            if self.on_pause is not None:
                self.on_pause(self.pauses)
            pause(waits, for_other=for_other)

        monkeypatch.setattr(manager._Waits, "pause", _pause)

    @property
    def locked(self) -> bool:
        return bool(self.held)

    @locked.setter
    def locked(self, value: bool) -> None:
        if value:
            self.held.add(True)
        else:
            self.held.clear()

    def run(self, pid: int | None = None, *, port: int = 6174, **fields) -> dict:
        """A server of this data folder, recorded and answering."""
        pid = self.pid if pid is None else pid
        self.processes.add(pid, port=port)
        self.answering.add(port)
        return write_record(pid, port=port, **fields)

    def _running(self, base_url=None, *, port=6174, host="127.0.0.1", timeout=1.0):
        return port in self.answering

    def _occupied(self, port, host="127.0.0.1"):
        return port in self.answering | self.silent | self.occupied

    def _ask(self, url: str) -> str:
        port = int(url.rsplit(":", 1)[1])
        if port in self.answering:
            return self.whose.get(port, "own")
        if port in self.occupied:
            return "foreign"
        if port in self.silent:
            return "silent"
        return "refused"

    def _spawn(self, **kwargs) -> FakeProc:
        self.spawns.append(kwargs)
        proc = FakeProc(self.pid, returncode=self.exit_code)
        self.procs.append(proc)
        manager._PROCESS = proc
        if kwargs.get("job") is not None:
            kwargs["job"].started(proc)
        if self.exit_code is None and self.records:
            self.run(port=kwargs["port"], lasting=kwargs.get("lasting"))
        return proc

    def _end(self, pid, popen=None, *, break_group=False) -> None:
        self.stopped.append(pid)
        self.processes.end(pid)
        if popen is not None and popen.returncode is None:
            popen.returncode = -15


class PopenAbort(Exception):
    """Raised by the fake ``Popen`` to short-circuit ``_spawn_server``
    after the env dict is constructed."""


def patch_start_prereqs(monkeypatch, tmp_path: Path, runtime_paths=None) -> dict:
    """Mock everything ``_spawn_server`` needs before ``subprocess.Popen``
    so it runs up to the point where the env dict is captured. Returns a
    ``captured`` dict that the fake Popen fills."""
    captured: dict = {}

    monkeypatch.setattr(manager, "_langgraph_exe", lambda: "/usr/bin/langgraph")

    fake_config = tmp_path / "langgraph.json"
    fake_config.write_text("{}")
    monkeypatch.setattr(manager, "_packaged_langgraph_config", lambda: fake_config)

    def _fake_popen(args, **kwargs):
        captured["args"] = args
        captured["env"] = kwargs.get("env", {})
        captured["cwd"] = kwargs.get("cwd")
        captured["kwargs"] = kwargs
        raise PopenAbort("env captured")

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)
    return captured


def start_lock_is_held() -> bool:
    from filelock import FileLock, Timeout

    probe = FileLock(str(manager.RUNTIME.lock_file))
    try:
        probe.acquire(timeout=0)
    except Timeout:
        return True
    probe.release()
    return False


def break_start_lock(monkeypatch, tmp_path, step: str) -> None:
    """Make the server start lock fail with an OSError, as a read-only or
    full data folder does: at the creation of the server home, or at the
    acquire."""
    import filelock

    if step == "mkdir":
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("")
        monkeypatch.setattr(paths, "SERVER_HOME", blocker / "server")
    else:

        def _denied(self, *_a, **_k):
            raise PermissionError(13, "Permission denied", str(self.lock_file))

        monkeypatch.setattr(filelock.FileLock, "acquire", _denied)
