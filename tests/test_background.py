"""Tests for EvoScientist.background — the background-process manager."""

import time

import pytest

from EvoScientist import background as bg


@pytest.fixture(autouse=True)
def _clean_registry():
    """Isolate each test: clear the module-global registry and reap leftovers."""
    bg._PROCESSES.clear()
    yield
    for proc in list(bg._PROCESSES.values()):
        try:
            proc.popen.kill()
        except Exception:
            pass
    bg._PROCESSES.clear()


def test_launch_returns_id_and_creates_log(tmp_path):
    pid = bg.launch("echo hi", str(tmp_path))
    assert pid in bg._PROCESSES
    assert (tmp_path / ".bg_processes" / f"{pid}.log").exists()


def test_status_running_then_exited(tmp_path):
    pid = bg.launch("sleep 1", str(tmp_path))
    assert "RUNNING" in bg.status(pid)
    time.sleep(1.6)
    out = bg.status(pid)
    assert "EXITED" in out
    assert "code 0" in out


def test_output_captured_in_status(tmp_path):
    pid = bg.launch("echo hello-from-bg", str(tmp_path))
    time.sleep(0.6)
    assert "hello-from-bg" in bg.status(pid)


def test_stop_kills_running_process(tmp_path):
    pid = bg.launch("sleep 600", str(tmp_path))
    assert "RUNNING" in bg.status(pid)
    out = bg.stop(pid)
    assert "Stopped" in out
    assert bg._PROCESSES[pid].popen.poll() is not None  # actually terminated


def test_stop_already_finished_is_graceful(tmp_path):
    pid = bg.launch("true", str(tmp_path))
    time.sleep(0.4)
    assert "already finished" in bg.stop(pid)


def test_exited_elapsed_is_frozen(tmp_path):
    """Elapsed for an exited process freezes at its runtime, it must not keep growing."""
    pid = bg.launch("true", str(tmp_path))
    time.sleep(0.3)
    bg.status(pid)  # observe exit -> records finished_ts
    proc = bg._PROCESSES[pid]
    assert proc.finished_ts is not None
    first = bg._elapsed(proc)
    time.sleep(1.1)
    assert bg._elapsed(proc) == first  # frozen, not ticking up


def test_unknown_id_errors_gracefully():
    assert "No such background process" in bg.status("deadbeef")
    assert "No such background process" in bg.stop("deadbeef")


def test_list_all(tmp_path):
    assert "No background processes" in bg.list_all()
    pid = bg.launch("sleep 1", str(tmp_path))
    listing = bg.list_all()
    assert pid in listing
    assert "RUNNING" in listing
