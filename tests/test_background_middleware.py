"""Tests for BackgroundExecutionMiddleware and its tools."""

import sys
import time

import pytest

from EvoScientist import background as bg
from EvoScientist.middleware.background import (
    BackgroundExecutionMiddleware,
    check_process,
    list_processes,
    run_in_background,
    stop_process,
)


def _sleep_cmd(seconds: int) -> str:
    """Cross-platform command that sleeps for *seconds* and exits 0."""
    if sys.platform == "win32":
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
    from EvoScientist.cli import async_notifier

    bg._PROCESSES.clear()
    async_notifier.drain_notifications(None)
    yield
    for proc in list(bg._PROCESSES.values()):
        try:
            proc.popen.kill()
        except Exception:
            pass
    bg._PROCESSES.clear()
    async_notifier.drain_notifications(None)


def test_middleware_registers_four_tools():
    mw = BackgroundExecutionMiddleware()
    names = {t.name for t in mw.tools}
    assert names == {
        "run_in_background",
        "check_process",
        "stop_process",
        "list_processes",
    }


def test_no_job_in_tool_names():
    """Naming ADR: the word 'job' must not appear in the tool surface."""
    mw = BackgroundExecutionMiddleware()
    assert not any("job" in t.name.lower() for t in mw.tools)


def test_run_rejects_dangerous_command_without_launching(monkeypatch):
    launched = {"called": False}

    def _spy(*args, **kwargs):
        launched["called"] = True
        return "should-not-happen"

    monkeypatch.setattr(bg, "launch", _spy)
    out = run_in_background.invoke({"command": "sudo rm -rf /"})
    assert launched["called"] is False
    assert "blocked" in out.lower()


def test_run_launches_valid_command(tmp_path, monkeypatch):
    # Pin the workspace cwd to a temp dir so the launch is isolated.
    monkeypatch.setattr("EvoScientist.paths.resolve_virtual_path", lambda _vp: tmp_path)
    out = run_in_background.invoke({"command": "echo ok", "name": "demo"})
    assert "Started background process" in out
    assert "check_process" in out
    assert len(bg._PROCESSES) == 1


def test_run_applies_virtual_path_rewriting(tmp_path, monkeypatch):
    """run_in_background must rewrite virtual paths like execute (shared preprocessing)."""
    monkeypatch.setattr("EvoScientist.paths.resolve_virtual_path", lambda _vp: tmp_path)
    captured = {}

    def _spy(command, cwd, name=None, *, origin_thread_id=None, on_exit=None):
        captured["command"] = command
        return "pidX"

    monkeypatch.setattr(bg, "launch", _spy)
    run_in_background.invoke({"command": "python /train.py"})
    # virtual absolute path -> workspace-relative, same as execute would produce
    assert captured["command"] == "python ./train.py"


def test_run_enqueues_completion_notification(tmp_path, monkeypatch):
    """A finished background process enqueues a shell completion notification."""
    from EvoScientist.cli import async_notifier

    monkeypatch.setattr("EvoScientist.paths.resolve_virtual_path", lambda _vp: tmp_path)
    run_in_background.invoke({"command": _true_cmd(), "name": "quick"})
    # drain consumes, so accumulate across polls until the watcher's on_exit enqueues.
    notifs = []
    deadline = time.time() + 4.0
    while time.time() < deadline:
        notifs.extend(async_notifier.drain_notifications(None))
        if any(n.kind == "bg-process" for n in notifs):
            break
        time.sleep(0.05)
    assert any(n.kind == "bg-process" and n.status == "success" for n in notifs)


def test_origin_thread_id_reads_runtime_config():
    """thread_id is read from runtime.config['configurable'] (graph-injected)."""
    from types import SimpleNamespace

    from EvoScientist.middleware.background import _origin_thread_id

    runtime = SimpleNamespace(config={"configurable": {"thread_id": "T-7"}})
    assert _origin_thread_id(runtime) == "T-7"
    assert _origin_thread_id(None) is None  # direct .invoke() / no runtime


def test_notify_done_routes_to_origin_thread(tmp_path):
    """_notify_done enqueues the completion notification to the launching thread."""
    from EvoScientist.cli import async_notifier
    from EvoScientist.middleware.background import _notify_done

    pid = bg.launch(_true_cmd(), str(tmp_path))  # no on_exit -> no auto-notify here
    assert _wait_until(lambda: bg._PROCESSES[pid].finished_ts is not None)
    _notify_done(bg._PROCESSES[pid], "T-123")
    routed = async_notifier.drain_notifications("T-123")
    assert any(n.task_id == pid and n.origin_cli_thread_id == "T-123" for n in routed)


def test_stopped_process_suppresses_notification(tmp_path, monkeypatch):
    """A user-stopped process must NOT emit a completion notification."""
    from EvoScientist.cli import async_notifier

    monkeypatch.setattr("EvoScientist.paths.resolve_virtual_path", lambda _vp: tmp_path)
    run_in_background.invoke({"command": _sleep_cmd(600)})
    (pid,) = list(bg._PROCESSES.keys())
    stop_process.invoke({"process_id": pid})
    # Wait until the watcher observed the exit — it would have enqueued here if the
    # process weren't user-stopped. _notify_done is a no-op for stopped processes.
    assert _wait_until(lambda: bg._PROCESSES[pid].finished_ts is not None)
    notifs = async_notifier.drain_notifications(None)
    assert not any(n.task_id == pid for n in notifs)


def test_checked_after_exit_dedups_notification(tmp_path):
    """Agent checking a finished process suppresses its completion notification."""
    from EvoScientist.cli.async_notifier import (
        AsyncTaskNotification,
        dedup_notifications,
    )

    pid = bg.launch(_true_cmd(), str(tmp_path))
    assert _wait_until(lambda: bg._PROCESSES[pid].finished_ts is not None)
    bg.status(pid)  # agent checks AFTER exit
    assert bg.was_observed_done(pid) is True
    n = AsyncTaskNotification(
        task_id=pid,
        agent_name="x",
        status="success",
        received_at="t",
        kind="bg-process",
    )
    assert dedup_notifications([n], {}) == []  # deduped


def test_not_checked_after_exit_keeps_notification(tmp_path):
    """A finished process the agent never checked still notifies."""
    from EvoScientist.cli.async_notifier import (
        AsyncTaskNotification,
        dedup_notifications,
    )

    pid = bg.launch(_true_cmd(), str(tmp_path))
    assert _wait_until(
        lambda: bg._PROCESSES[pid].finished_ts is not None
    )  # exit, but do NOT check
    assert bg.was_observed_done(pid) is False
    n = AsyncTaskNotification(
        task_id=pid,
        agent_name="x",
        status="success",
        received_at="t",
        kind="bg-process",
    )
    assert dedup_notifications([n], {}) == [n]  # survives


def test_shell_notification_renders_own_background_frame():
    """Shell notifications render under '✦ Background ✦', not 'Agent Teams'."""
    from EvoScientist.cli.async_notifier import (
        AsyncTaskNotification,
        format_notification_lines,
    )

    n = AsyncTaskNotification(
        task_id="fe60ce9c",
        agent_name="test-20s",
        status="success",
        received_at="",
        prompt="python train.py",
        kind="bg-process",
    )
    lines = format_notification_lines([n])
    top, body = lines[0][0], lines[1][0]
    assert "Background" in top
    assert "Agent Teams" not in top
    assert "test-20s" in body
    assert "Cmd:" in body


def test_mixed_notifications_render_two_frames():
    """A mixed batch shows both an Agent Teams frame and a Background frame."""
    from EvoScientist.cli.async_notifier import (
        AsyncTaskNotification,
        format_notification_lines,
    )

    task = AsyncTaskNotification("t1", "writing-agent", "success", "", "")
    shell = AsyncTaskNotification("p1", "demo", "success", "", "", kind="bg-process")
    blob = "\n".join(t for t, _ in format_notification_lines([task, shell]))
    assert "Agent Teams" in blob
    assert "Background" in blob


def test_shell_notification_hints_check_process():
    """format_batch_message points shell processes to check_process, not check_async_task."""
    from EvoScientist.cli.async_notifier import (
        AsyncTaskNotification,
        format_batch_message,
    )

    n = AsyncTaskNotification(
        task_id="ab12",
        agent_name="demo",
        status="success",
        received_at="x",
        kind="bg-process",
    )
    msg = format_batch_message([n])
    assert "check_process" in msg
    assert "check_async_task" not in msg  # shell-only batch -> no sub-agent hint


def test_check_and_list_route_to_manager(tmp_path, monkeypatch):
    monkeypatch.setattr("EvoScientist.paths.resolve_virtual_path", lambda _vp: tmp_path)
    run_in_background.invoke({"command": _sleep_cmd(1)})
    (pid,) = bg._PROCESSES.keys()
    assert pid in check_process.invoke({"process_id": pid})
    assert pid in list_processes.invoke({})
    assert "Stopped" in stop_process.invoke(
        {"process_id": pid}
    ) or "finished" in stop_process.invoke({"process_id": pid})


def test_list_processes_forwards_all_threads(monkeypatch):
    """The all_threads tool arg is forwarded to background.list_all(include_all=...)."""
    captured = {}

    def _spy(thread_id=None, *, include_all=False):
        captured["include_all"] = include_all
        return "ok"

    monkeypatch.setattr(bg, "list_all", _spy)
    list_processes.invoke({"all_threads": True})
    assert captured["include_all"] is True
    list_processes.invoke({})
    assert captured["include_all"] is False
