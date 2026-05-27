"""Tests for BackgroundExecutionMiddleware and its tools."""

import pytest

from EvoScientist import background as bg
from EvoScientist.middleware.background import (
    BackgroundExecutionMiddleware,
    check_process,
    list_processes,
    run_in_background,
    stop_process,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    bg._PROCESSES.clear()
    yield
    for proc in list(bg._PROCESSES.values()):
        try:
            proc.popen.kill()
        except Exception:
            pass
    bg._PROCESSES.clear()


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

    def _spy(command, cwd, name=None):
        captured["command"] = command
        return "pidX"

    monkeypatch.setattr(bg, "launch", _spy)
    run_in_background.invoke({"command": "python /train.py"})
    # virtual absolute path -> workspace-relative, same as execute would produce
    assert captured["command"] == "python ./train.py"


def test_check_and_list_route_to_manager(tmp_path, monkeypatch):
    monkeypatch.setattr("EvoScientist.paths.resolve_virtual_path", lambda _vp: tmp_path)
    run_in_background.invoke({"command": "sleep 1"})
    (pid,) = bg._PROCESSES.keys()
    assert pid in check_process.invoke({"process_id": pid})
    assert pid in list_processes.invoke({})
    assert "Stopped" in stop_process.invoke(
        {"process_id": pid}
    ) or "finished" in stop_process.invoke({"process_id": pid})
