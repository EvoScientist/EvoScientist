"""Tests for the Ctrl+Break helper that asks a Windows langgraph dev server to
shut down cleanly (``EvoScientist.langgraph_dev.console_break``)."""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from EvoScientist.langgraph_dev import console_break, manager


class _Kernel32:
    """Records the calls in order; each call returns its configured result."""

    def __init__(self, attach: bool = True, send: bool = True) -> None:
        self.calls: list[tuple] = []
        self._attach, self._send = attach, send

    def FreeConsole(self):
        self.calls.append(("FreeConsole",))
        return 1

    def AttachConsole(self, pid):
        self.calls.append(("AttachConsole", pid))
        return int(self._attach)

    def GenerateConsoleCtrlEvent(self, event, group):
        self.calls.append(("GenerateConsoleCtrlEvent", event, group))
        return int(self._send)


def test_send_break_attaches_and_sends_to_the_group():
    k32 = _Kernel32()
    assert console_break.send_break(4242, k32, lambda: 0) == console_break.EXIT_SENT
    assert k32.calls == [
        ("FreeConsole",),
        ("AttachConsole", 4242),
        ("GenerateConsoleCtrlEvent", 1, 4242),
        ("FreeConsole",),
    ]


def test_send_break_reports_a_failed_attach_without_sending(capsys):
    k32 = _Kernel32(attach=False)
    code = console_break.send_break(4242, k32, lambda: 6)
    assert code == console_break.EXIT_ATTACH_FAILED
    assert not any(c[0] == "GenerateConsoleCtrlEvent" for c in k32.calls)
    assert "AttachConsole(4242) failed: error 6" in capsys.readouterr().err


def test_send_break_reports_a_failed_send(capsys):
    k32 = _Kernel32(send=False)
    code = console_break.send_break(4242, k32, lambda: 87)
    assert code == console_break.EXIT_SEND_FAILED
    assert "GenerateConsoleCtrlEvent(4242) failed: error 87" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [[], ["x"], ["0"], ["-5"], ["1", "2"]])
def test_main_rejects_anything_but_one_positive_pid(monkeypatch, argv):
    """Group 0 would be every process on the console: never sent."""
    sent: list = []
    monkeypatch.setattr(console_break, "send_break", lambda pid: sent.append(pid))
    assert console_break.main(argv) == console_break.EXIT_USAGE
    assert sent == []


_CHILD = r"""
import signal, sys, time
def got(*_):
    open(sys.argv[1], "w").write("got")
    sys.exit(0)
signal.signal(signal.SIGBREAK, got)
open(sys.argv[1] + ".ready", "w").write("ready")
end = time.time() + 30
while time.time() < end:
    time.sleep(0.05)
"""


@pytest.mark.skipif(os.name != "nt", reason="Ctrl+Break is Windows-only")
def test_helper_ends_a_group_leader_with_its_own_console(tmp_path):
    """A real spawn shaped like a lasting server: own process group, hidden
    console of its own. The helper's Ctrl+Break reaches its SIGBREAK handler,
    and this process (on another console) keeps running."""
    marker = tmp_path / "got"
    child = subprocess.Popen(
        [sys.executable, "-c", _CHILD, str(marker)],
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW,
    )
    try:
        deadline = time.monotonic() + 15
        while not (tmp_path / "got.ready").exists():
            assert time.monotonic() < deadline, "child never installed its handler"
            time.sleep(0.05)
        assert manager._request_clean_stop(child.pid) is True
        child.wait(timeout=10)
        assert marker.read_text() == "got"
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
