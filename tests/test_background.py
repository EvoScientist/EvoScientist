"""Tests for EvoScientist.background — the background-process manager."""

import sys
import time

import pytest

from EvoScientist import background as bg


def _sleep_cmd(seconds: int) -> str:
    """Cross-platform command that sleeps for *seconds* and exits 0."""
    if sys.platform == "win32":
        # ``ping -n N+1 127.0.0.1 > nul`` sleeps ~N seconds.
        return f"ping -n {seconds + 1} 127.0.0.1 > nul"
    return f"sleep {seconds}"


def _true_cmd() -> str:
    """Cross-platform command that exits 0 immediately."""
    if sys.platform == "win32":
        return "cmd /c exit /b 0"
    return "true"


def _wait_until(predicate, timeout=4.0, interval=0.05):
    """Poll ``predicate`` until true or ``timeout`` — avoids flaky fixed sleeps on slow CI."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


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
    pid = bg.launch(_sleep_cmd(1), str(tmp_path))
    assert "RUNNING" in bg.status(pid)
    assert _wait_until(lambda: "EXITED" in bg.status(pid))
    out = bg.status(pid)
    assert "EXITED" in out
    assert "code 0" in out


def test_output_captured_in_status(tmp_path):
    pid = bg.launch("echo hello-from-bg", str(tmp_path))
    assert _wait_until(lambda: "hello-from-bg" in bg.status(pid))


def test_large_log_returns_truncated_tail(tmp_path):
    """status() preserves the truncation contract for a large log (output shape, not I/O)."""
    pid = bg.launch(_true_cmd(), str(tmp_path))
    log_path = tmp_path / ".bg_processes" / f"{pid}.log"
    log_path.write_bytes(b"A" * 5000 + b"TAIL_MARKER")
    out = bg.status(pid, tail_bytes=64)
    assert "...(truncated)..." in out
    assert "TAIL_MARKER" in out
    assert "A" * 5000 not in out  # the head was not loaded


def test_stop_kills_running_process(tmp_path):
    pid = bg.launch(_sleep_cmd(600), str(tmp_path))
    assert "RUNNING" in bg.status(pid)
    out = bg.stop(pid)
    assert "Stopped" in out
    assert bg._PROCESSES[pid].popen.poll() is not None  # actually terminated


def test_stop_already_finished_is_graceful(tmp_path):
    pid = bg.launch(_true_cmd(), str(tmp_path))
    assert _wait_until(lambda: bg._PROCESSES[pid].popen.poll() is not None)
    assert "already finished" in bg.stop(pid)


def test_exited_elapsed_is_frozen(tmp_path, monkeypatch):
    """Elapsed for an exited process freezes at its runtime, it must not keep growing."""
    pid = bg.launch(_true_cmd(), str(tmp_path))
    assert _wait_until(lambda: bg._PROCESSES[pid].finished_ts is not None)
    bg.status(pid)  # observe exit -> records finished_ts
    proc = bg._PROCESSES[pid]
    assert proc.finished_ts is not None
    first = bg._elapsed(proc)
    monkeypatch.setattr(bg.time, "time", lambda: proc.finished_ts + 100.0)
    assert bg._elapsed(proc) == first


def test_watcher_records_exit_without_polling(tmp_path):
    """The daemon watcher records exit on its own (no status() call needed)."""
    pid = bg.launch(_true_cmd(), str(tmp_path))
    assert _wait_until(lambda: bg._PROCESSES[pid].finished_ts is not None)
    proc = bg._PROCESSES[pid]
    assert proc.finished_ts is not None
    assert proc.returncode == 0


def test_on_exit_callback_fires(tmp_path):
    """on_exit is invoked with the BgProcess once the process exits."""
    fired = {}

    def cb(proc):
        fired["pid"] = proc.process_id
        fired["rc"] = proc.returncode

    pid = bg.launch(_true_cmd(), str(tmp_path), on_exit=cb)
    assert _wait_until(lambda: fired.get("pid") == pid and fired.get("rc") == 0)
    assert fired.get("pid") == pid
    assert fired.get("rc") == 0


def test_unknown_id_errors_gracefully():
    assert "No such background process" in bg.status("deadbeef")
    assert "No such background process" in bg.stop("deadbeef")


def test_list_all(tmp_path):
    assert "No background processes" in bg.list_all()
    pid = bg.launch(_sleep_cmd(1), str(tmp_path))
    listing = bg.list_all()
    assert pid in listing
    assert "RUNNING" in listing


def test_list_all_scopes_to_origin_thread(tmp_path):
    """list_all defaults to the launching session; include_all sees every session."""
    pid_a = bg.launch(_sleep_cmd(1), str(tmp_path), origin_thread_id="A")
    pid_b = bg.launch(_sleep_cmd(1), str(tmp_path), origin_thread_id="B")
    listing_a = bg.list_all("A")
    assert pid_a in listing_a
    assert pid_b not in listing_a  # B's process is hidden from session A
    everything = bg.list_all("A", include_all=True)
    assert pid_a in everything
    assert pid_b in everything


def test_list_all_hints_at_other_sessions(tmp_path):
    """A session with no processes of its own is told others exist."""
    bg.launch(_sleep_cmd(1), str(tmp_path), origin_thread_id="A")
    out = bg.list_all("B")  # a different session
    assert "other sessions" in out
    assert "all_threads=True" in out


def test_dedup_is_per_thread(tmp_path):
    """A check from one session must not suppress another session's completion ping."""
    pid = bg.launch(_true_cmd(), str(tmp_path), origin_thread_id="A")
    assert _wait_until(lambda: bg._PROCESSES[pid].finished_ts is not None)
    bg.status(pid, thread_id="B")  # a DIFFERENT session inspects it
    assert bg.was_observed_done(pid, "B") is True  # B saw it
    assert bg.was_observed_done(pid, "A") is False  # launcher A did not -> still notify
