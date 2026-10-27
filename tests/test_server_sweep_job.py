"""Tests for the Windows server sweep job (issue #604).

The sweep job is a named job around a langgraph server; stopping the server
terminates it by name, which ends MSYS programs whose parent bash already
exited — the ones the stop paths' parent-PID tree walk misses.

The unit tests run anywhere with a faked job API; the ``windows_bash`` test
runs the real thing on a Windows host with Git for Windows in
``C:\\Program Files\\Git`` (the ``windows-latest`` CI image has one).
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from EvoScientist import agent_shell

_SYSTEM_GIT = Path(r"C:\Program Files\Git")
windows_bash = pytest.mark.skipif(
    sys.platform != "win32" or not (_SYSTEM_GIT / "bin" / "bash.exe").is_file(),
    reason="needs Git for Windows in C:\\Program Files\\Git",
)


class FakeJobApi:
    def __init__(self, *, create_ok: bool = True, assign_ok: bool = True) -> None:
        self.create_ok = create_ok
        self.assign_ok = assign_ok
        self.calls: list[tuple] = []

    def create_named(self, name):
        self.calls.append(("create_named", name))
        return 77 if self.create_ok else None

    def open(self, name):
        self.calls.append(("open", name))
        return 88

    def assign(self, job, process):
        self.calls.append(("assign", job, process))
        return self.assign_ok

    def last_error(self):
        return 5

    def resume(self, process):
        self.calls.append(("resume", process))

    def terminate(self, job):
        self.calls.append(("terminate", job))
        return True

    def close(self, job):
        self.calls.append(("close", job))


def _fake_api(monkeypatch, api):
    monkeypatch.setattr(agent_shell, "_job_api", lambda: api)


def test_sweep_job_name_is_stable_per_pid():
    assert (
        agent_shell.server_sweep_job_name(123)
        == agent_shell.server_sweep_job_name(123)
        != agent_shell.server_sweep_job_name(124)
    )


def test_started_assigns_the_named_job_and_resumes(monkeypatch):
    api = FakeJobApi()
    _fake_api(monkeypatch, api)
    sweep = agent_shell.ServerSweepJob(123)
    assert sweep.creationflags != 0
    process = type("P", (), {"_handle": 9, "pid": 123})()
    sweep.started(process, resume=True)
    assert ("create_named", agent_shell.server_sweep_job_name(123)) in api.calls
    assert ("assign", 77, 9) in api.calls
    assert api.calls[-1] == ("resume", 9)
    sweep.close()
    assert ("close", 77) in api.calls


def test_started_leaves_the_resume_to_a_second_job(monkeypatch):
    api = FakeJobApi()
    _fake_api(monkeypatch, api)
    sweep = agent_shell.ServerSweepJob(123)
    process = type("P", (), {"_handle": 9, "pid": 123})()
    sweep.started(process, resume=False)
    assert ("assign", 77, 9) in api.calls
    assert ("resume", 9) not in api.calls


def test_started_still_resumes_when_no_job_could_be_created(monkeypatch):
    api = FakeJobApi(create_ok=False)
    _fake_api(monkeypatch, api)
    sweep = agent_shell.ServerSweepJob(123)
    assert sweep.creationflags == 0
    process = type("P", (), {"_handle": 9, "pid": 123})()
    sweep.started(process, resume=True)
    assert ("assign", 77, 9) not in api.calls
    assert ("resume", 9) in api.calls


def test_terminate_opens_the_job_by_name_and_ends_it(monkeypatch):
    api = FakeJobApi()
    _fake_api(monkeypatch, api)
    assert agent_shell.terminate_server_sweep_job(123) is True
    assert ("open", agent_shell.server_sweep_job_name(123)) in api.calls
    assert ("terminate", 88) in api.calls
    assert ("close", 88) in api.calls


def test_terminate_is_false_when_the_job_cannot_be_opened(monkeypatch):
    class NoJobApi(FakeJobApi):
        def open(self, name):
            self.calls.append(("open", name))
            return None

    api = NoJobApi()
    _fake_api(monkeypatch, api)
    assert agent_shell.terminate_server_sweep_job(123) is False
    assert ("terminate", 88) not in api.calls


def test_started_resumes_through_psutil_when_the_job_api_cannot_load(monkeypatch):
    """The suspended server must be resumed even when the job API itself
    cannot be loaded (e.g. the tests' os.name="nt" on Linux): ``started``
    falls back to a psutil resume instead of leaving the server frozen or
    raising — the never-raises contract."""

    def _no_api():
        raise AttributeError("kernel32 unavailable")

    monkeypatch.setattr(agent_shell, "_job_api", _no_api)
    resumed = []

    class FakePsutilProcess:
        def __init__(self, pid):
            assert pid == 4242

        def resume(self):
            resumed.append(True)

    fake_psutil = type(sys)("psutil")
    fake_psutil.Process = FakePsutilProcess
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    sweep = agent_shell.ServerSweepJob(123)
    assert sweep._handle is None
    assert sweep.creationflags == 0
    process = type("P", (), {"_handle": 9, "pid": 4242})()
    sweep.started(process, resume=True)
    assert resumed == [True]


@windows_bash
def test_sweep_job_ends_msys_orphans_the_tree_walk_misses(tmp_path):
    bash = _SYSTEM_GIT / "bin" / "bash.exe"
    # A stand-in "server": joins the sweep job like the real one, then runs
    # a Git Bash command that leaves an MSYS ``sleep`` behind (its parent
    # bash exits immediately) and keeps running until ended. The sleep's
    # duration is a unique needle: MSYS ``$!`` is an MSYS pid, not the
    # Windows pid psutil needs, so the orphan is found by its cmdline.
    needle = "12347"
    server_script = tmp_path / "fake_server.py"
    server_script.write_text(
        "import subprocess, time\n"
        f"subprocess.Popen([r'{bash}', '-c', 'sleep {needle} & disown'])\n"
        "print('ready', flush=True)\n"
        "time.sleep(120)\n",
        encoding="utf-8",
    )
    sweep = None
    server = None
    try:
        server = subprocess.Popen(
            [sys.executable, str(server_script)],
            stdout=subprocess.PIPE,
            creationflags=agent_shell._CREATE_SUSPENDED,
        )
        sweep = agent_shell.ServerSweepJob(server.pid)
        sweep.started(server, resume=True)
        assert server.stdout is not None
        assert server.stdout.readline().strip() == b"ready"

        orphan = _wait_for_orphan(needle)

        # The bug itself: the orphan's Windows parent is gone, so the stop
        # paths' parent-PID walk does not reach it. The walk raises on a
        # vanished intermediate process, so guard each step like the real
        # callers do.
        try:
            siblings = psutil.Process(server.pid).children(recursive=True)
        except psutil.NoSuchProcess:
            siblings = []
        assert orphan.pid not in {child.pid for child in siblings}

        assert agent_shell.terminate_server_sweep_job(server.pid) is True
        _wait_until(lambda: _gone(orphan) and server.poll() is not None)
        assert _gone(orphan)
        assert server.poll() is not None
    finally:
        if server is not None and server.poll() is None:
            server.kill()
        _best_effort_kill_orphan(needle)
        if sweep is not None:
            sweep.close()


def _matches_orphan(proc: psutil.Process, needle: str) -> bool:
    """True when ``proc`` is the orphaned sleep: its name is cheap to get,
    the cmdline (expensive on Windows) only for the few ``sleep`` processes."""
    try:
        if "sleep" not in (proc.name() or "").lower():
            return False
        cmdline = proc.cmdline()
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False
    return cmdline and needle in cmdline


def _wait_for_orphan(needle: str) -> psutil.Process:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        for proc in psutil.process_iter():
            if _matches_orphan(proc, needle):
                return proc
        time.sleep(0.1)
    raise AssertionError("the orphaned sleep was never found among processes")


def _best_effort_kill_orphan(needle: str) -> None:
    for proc in psutil.process_iter():
        if _matches_orphan(proc, needle):
            _best_effort_kill(proc.pid)


def _parent_is_gone(process: psutil.Process) -> bool:
    try:
        ppid = process.ppid()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return True
    if ppid <= 0:
        return True
    try:
        psutil.Process(ppid)
    except psutil.NoSuchProcess:
        return True
    return False


def _gone(process: psutil.Process) -> bool:
    """True when ``process`` is definitely no longer running."""
    try:
        return not process.is_running()
    except psutil.NoSuchProcess:
        return True
    except psutil.AccessDenied:
        return False


def _wait_until(predicate) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.1)


def _best_effort_kill(pid: int) -> None:
    try:
        psutil.Process(pid).kill()
    except psutil.NoSuchProcess:
        pass
