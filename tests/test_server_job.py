"""Tests for the Windows server job (issue #604).

The server joins a kill-on-close job whose only handle lives in the server, so
ending the server, however that happens, ends the MSYS programs whose parent
bash already exited: the ones the stop paths' parent-PID tree walk misses.

The unit tests run anywhere with a faked job API; the ``windows_bash`` test
runs the real thing on a Windows host with Git for Windows in
``C:\\Program Files\\Git`` (the ``windows-latest`` CI image has one).
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from EvoScientist import agent_shell
from EvoScientist.langgraph_dev import manager
from tests.fakes.langgraph_dev import PopenAbort, patch_start_prereqs

_SYSTEM_GIT = Path(r"C:\Program Files\Git")
windows_bash = pytest.mark.skipif(
    sys.platform != "win32" or not (_SYSTEM_GIT / "bin" / "bash.exe").is_file(),
    reason="needs Git for Windows in C:\\Program Files\\Git",
)


class FakeJobApi:
    def __init__(
        self, *, create_ok=True, assign_ok=True, share_ok=True, kill_on_close_ok=True
    ) -> None:
        self.create_ok = create_ok
        self.assign_ok = assign_ok
        self.share_ok = share_ok
        self.kill_on_close_ok = kill_on_close_ok
        self.calls: list[tuple] = []

    def create(self):
        self.calls.append(("create",))
        return 77 if self.create_ok else None

    def assign(self, job, process):
        self.calls.append(("assign", job, process))
        if self.assign_ok is None:
            raise OSError("assign failed")
        return self.assign_ok

    def share(self, job, process):
        self.calls.append(("share", job, process))
        return self.share_ok

    def kill_on_close(self, job):
        self.calls.append(("kill_on_close", job))
        return self.kill_on_close_ok

    def resume(self, process):
        self.calls.append(("resume", process))

    def close(self, job):
        self.calls.append(("close", job))

    def last_error(self):
        return 5


def _fake_api(monkeypatch, api):
    monkeypatch.setattr(agent_shell, "_job_api", lambda: api)
    agent_shell._warn_no_server_job.cache_clear()


_PROCESS = SimpleNamespace(_handle=9, pid=123)


def test_the_server_holds_the_only_handle_to_a_kill_on_close_job(monkeypatch):
    api = FakeJobApi()
    _fake_api(monkeypatch, api)
    agent_shell.join_server_job(_PROCESS)
    assert api.calls == [
        ("create",),
        ("assign", 77, 9),
        ("share", 77, 9),
        ("kill_on_close", 77),
        ("close", 77),
        ("resume", 9),
    ]


@pytest.mark.parametrize("failing", ["assign", "share"])
def test_no_kill_on_close_unless_the_server_holds_a_handle(
    monkeypatch, caplog, failing
):
    """Closing our handle of a kill-on-close job nobody else holds would end
    the server at once."""
    api = FakeJobApi(**{f"{failing}_ok": False})
    _fake_api(monkeypatch, api)
    agent_shell.join_server_job(_PROCESS)
    assert ("kill_on_close", 77) not in api.calls
    assert api.calls[-2:] == [("close", 77), ("resume", 9)]
    assert "(error 5)" in caplog.text


def test_a_failed_kill_on_close_is_logged(monkeypatch, caplog):
    api = FakeJobApi(kill_on_close_ok=False)
    _fake_api(monkeypatch, api)
    agent_shell.join_server_job(_PROCESS)
    assert api.calls[-2:] == [("close", 77), ("resume", 9)]
    assert "(error 5)" in caplog.text


def test_no_job_still_resumes_the_server(monkeypatch, caplog):
    api = FakeJobApi(create_ok=False)
    _fake_api(monkeypatch, api)
    agent_shell.join_server_job(_PROCESS)
    assert api.calls == [("create",), ("resume", 9)]
    assert "(error 5)" in caplog.text


def test_a_failing_call_still_closes_the_job_and_resumes(monkeypatch):
    api = FakeJobApi(assign_ok=None)
    _fake_api(monkeypatch, api)
    agent_shell.join_server_job(_PROCESS)
    assert api.calls[-2:] == [("close", 77), ("resume", 9)]


def test_resumes_through_psutil_when_the_job_api_cannot_load(monkeypatch):
    def _no_api():
        raise AttributeError("kernel32 unavailable")

    monkeypatch.setattr(agent_shell, "_job_api", _no_api)
    resumed = []
    monkeypatch.setattr(
        psutil,
        "Process",
        lambda pid: SimpleNamespace(resume=lambda: resumed.append(pid)),
    )
    agent_shell.join_server_job(_PROCESS)
    assert resumed == [123]


def test_a_vanished_server_does_not_raise(monkeypatch):
    def _no_api():
        raise AttributeError("kernel32 unavailable")

    def _gone_process(pid):
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(agent_shell, "_job_api", _no_api)
    monkeypatch.setattr(psutil, "Process", _gone_process)
    agent_shell.join_server_job(_PROCESS)


@pytest.mark.parametrize("with_job", [False, True])
def test_only_a_server_without_a_callers_job_joins_a_server_job(
    monkeypatch, tmp_path, runtime_paths, with_job
):
    """``deploy --json``'s kill-on-close job already ends every member; a job
    assigned first would keep the server out of it."""
    patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
    monkeypatch.setattr(manager, "_PROCESS", None)
    monkeypatch.setattr(manager.os, "name", "nt", raising=False)
    monkeypatch.setattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200, raising=False)
    captured = {}

    def _popen(args, **kwargs):
        captured["flags"] = kwargs["creationflags"]
        return SimpleNamespace(pid=7)

    monkeypatch.setattr(subprocess, "Popen", _popen)
    joined = []

    def _join(process):
        # Recorded first, so a caller's stop reaches it if joining fails.
        assert manager._PROCESS is process
        joined.append(process.pid)
        raise PopenAbort("joined the server job")

    class _Job:
        creationflags = 0x4

        def started(self, process):
            raise PopenAbort("joined the caller's job")

    monkeypatch.setattr(agent_shell, "join_server_job", _join)
    with pytest.raises(PopenAbort):
        manager._spawn_server(
            port=16179, lasting=False, job=_Job() if with_job else None
        )
    assert joined == ([] if with_job else [7])
    assert captured["flags"] == 0x200 | 0x4


@windows_bash
def test_server_job_ends_msys_orphans_the_tree_walk_misses(tmp_path):
    bash = _SYSTEM_GIT / "bin" / "bash.exe"
    # A stand-in "server": joins its server job like the real one, then runs
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
    server = None
    try:
        server = subprocess.Popen(
            [sys.executable, str(server_script)],
            stdout=subprocess.PIPE,
            creationflags=agent_shell._CREATE_SUSPENDED,
        )
        agent_shell.join_server_job(server)
        assert server.stdout is not None
        assert server.stdout.readline().strip() == b"ready"

        orphan = _wait_for_orphan(needle)
        assert _wait_until(lambda: _parent_is_gone(orphan))

        # The bug itself: the orphan's Windows parent is gone, so the stop
        # paths' parent-PID walk does not reach it. The walk raises on a
        # vanished intermediate process, so guard each step like the real
        # callers do.
        try:
            siblings = psutil.Process(server.pid).children(recursive=True)
        except psutil.NoSuchProcess:
            siblings = []
        assert orphan.pid not in {child.pid for child in siblings}

        # This process holds no handle to the job: ending the server, as any
        # stop path does, is what ends the orphan.
        server.kill()
        assert _wait_until(lambda: _gone(orphan))
    finally:
        if server is not None and server.poll() is None:
            server.kill()
        _best_effort_kill_orphan(needle)


def _matches_orphan(proc: psutil.Process, needle: str) -> bool:
    """True when ``proc`` is the orphaned sleep: its name is cheap to get,
    the cmdline (expensive on Windows) only for the few ``sleep`` processes."""
    try:
        if "sleep" not in (proc.name() or "").lower():
            return False
        cmdline = proc.cmdline()
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False
    return bool(cmdline) and needle in cmdline


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
            try:
                proc.kill()
            except psutil.NoSuchProcess:
                pass


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


def _wait_until(predicate) -> bool:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return False
