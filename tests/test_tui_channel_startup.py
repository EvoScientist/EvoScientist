from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("textual")

from EvoScientist.cli import tui_interactive as tui_mod
from EvoScientist.commands.base import ChannelRuntime


async def test_channel_startup_worker_keeps_event_loop_responsive(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    worker_thread: list[int] = []
    rows = [("telegram", True, "connected (bus)")]

    def blocking_start(*_args, **_kwargs):
        worker_thread.append(threading.get_ident())
        started.set()
        release.wait(timeout=2.0)
        finished.set()
        return rows

    monkeypatch.setattr(tui_mod, "_auto_start_channel", blocking_start)
    main_thread = threading.get_ident()
    task = asyncio.create_task(
        tui_mod._auto_start_channel_in_worker(
            object(),
            "thread-1",
            SimpleNamespace(channel_enabled="telegram"),
            send_thinking=False,
            runtime=ChannelRuntime(),
            stop_requested=threading.Event(),
        )
    )

    for _ in range(100):
        if started.is_set():
            break
        await asyncio.sleep(0.01)

    try:
        assert started.is_set()
        assert finished.is_set() is False
        assert len(worker_thread) == 1
        assert worker_thread[0] != main_thread
    finally:
        release.set()

    assert await task == rows
    assert finished.is_set()


async def test_channel_startup_worker_stops_channels_after_exit(monkeypatch):
    runtime = ChannelRuntime()
    stop_requested = threading.Event()
    stop_requested.set()
    stopped_with: list[ChannelRuntime | None] = []

    monkeypatch.setattr(
        tui_mod,
        "_auto_start_channel",
        lambda *_args, **_kwargs: [("telegram", False, "starting (bus)")],
    )
    monkeypatch.setattr(
        tui_mod,
        "_channels_stop",
        lambda _channel_type=None, *, runtime=None: stopped_with.append(runtime),
    )

    await tui_mod._auto_start_channel_in_worker(
        object(),
        "thread-1",
        SimpleNamespace(channel_enabled="telegram"),
        send_thinking=False,
        runtime=runtime,
        stop_requested=stop_requested,
    )

    assert stopped_with == [runtime]


@pytest.mark.parametrize(
    ("hint", "drift", "expected"),
    [
        (None, None, []),
        ("run EvoSci setup", None, ["run EvoSci setup"]),
        (None, "server python differs", ["server python differs"]),
        (
            "run EvoSci setup",
            "server python differs",
            ["run EvoSci setup", "server python differs"],
        ),
    ],
)
def test_agent_python_notices(monkeypatch, hint, drift, expected):
    """The TUI hides terminal output while it runs, so the hint and the
    server-python warning become in-app notices."""
    from EvoScientist.langgraph_dev import manager
    from EvoScientist.setup import research_env

    monkeypatch.setattr(research_env, "missing_python_hint", lambda: hint)
    monkeypatch.setattr(manager, "AGENT_PYTHON_DRIFT", drift)
    assert tui_mod._agent_python_notices() == expected
