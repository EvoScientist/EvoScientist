"""A media file the agent writes or reads is sent to the channel chat (Rich loop, TUI).

deepagents' ``write_file`` and ``read_file`` name the file in ``file_path``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from EvoScientist.stream import display as display_mod
from tests.fakes.common import FakeGraphGateway

_END = [
    {"type": "text", "content": "finished"},
    {"type": "done", "response": "finished"},
]


def _call(call_id: str, tool_name: str, args: dict) -> dict:
    return {"type": "tool_call", "id": call_id, "name": tool_name, "args": args}


def _result(call_id: str, tool_name: str, success: bool = True) -> dict:
    return {
        "type": "tool_result",
        "id": call_id,
        "name": tool_name,
        "content": "ok" if success else "[TOOL ERROR] refused",
        "success": success,
    }


def _file_tool_turn(tool_name: str, args: dict, success: bool = True) -> list[dict]:
    return [_call("tc1", tool_name, args), _result("tc1", tool_name, success), *_END]


def _one_step(*calls: tuple[object, bool]) -> list[dict]:
    """A turn whose model step writes every file of *calls* before any result."""
    ids = [f"tc{i}" for i in range(len(calls))]
    return [
        *(
            _call(i, "write_file", {"file_path": path, "content": "x"})
            for i, (path, _) in zip(ids, calls, strict=True)
        ),
        *(_result(i, "write_file", ok) for i, (_, ok) in zip(ids, calls, strict=True)),
        *_END,
    ]


CASES = [
    pytest.param("write_file", {"file_path": "/fig.png", "content": "x"}, id="write"),
    pytest.param("read_file", {"file_path": "/fig.png"}, id="read"),
    pytest.param("write_file", {"path": "/fig.png", "content": "x"}, id="legacy"),
    pytest.param("read_file", {"file_path": "/fig.png", "path": "/a.png"}, id="both"),
]

# Files written in one step -> the files sent, in order.
ONE_STEP_CASES = [
    pytest.param(
        [("/a.png", True), ("/b.png", True)], ["a.png", "b.png"], id="two-media"
    ),
    pytest.param(
        [("/a.png", True), ("/notes.txt", True)], ["a.png"], id="media-then-text"
    ),
    pytest.param(
        [("/a.png", True), ("/b.png", False)], ["a.png"], id="second-call-failed"
    ),
    pytest.param(
        [("/a.png", True), (["/b.png"], False)], ["a.png"], id="second-call-malformed"
    ),
]

NOT_SENT_CASES = [
    pytest.param({"file_path": "/notes.txt", "content": "x"}, True, id="not-media"),
    pytest.param({"file_path": "/fig.png", "content": "x"}, False, id="failed"),
    pytest.param({"content": "x"}, True, id="no-path"),
]


def _make_files(root) -> None:
    for name in ("fig.png", "a.png", "b.png", "notes.txt"):
        (root / name).write_bytes(b"x")


class TestRichLoop:
    @pytest.fixture(autouse=True)
    def _quiet(self, monkeypatch):
        monkeypatch.setattr(display_mod.console, "print", lambda *_a, **_k: None)

    @staticmethod
    def _run(events, work_dir) -> list[str]:
        _make_files(work_dir)
        sent: list[str] = []
        display_mod._run_streaming(
            agent=MagicMock(),
            message="hello",
            thread_id="t1",
            show_thinking=False,
            interactive=False,
            on_file_write=sent.append,
            work_dir=work_dir,
            gateway=FakeGraphGateway(events=events),
            _state=display_mod.StreamState(),
        )
        return sent

    @pytest.mark.parametrize(("tool_name", "args"), CASES)
    def test_a_media_file_is_sent(self, tmp_path, tool_name, args):
        assert self._run(_file_tool_turn(tool_name, args), tmp_path) == [
            str((tmp_path / "fig.png").resolve())
        ]

    @pytest.mark.parametrize(("calls", "expected"), ONE_STEP_CASES)
    def test_each_result_sends_its_own_file(self, tmp_path, calls, expected):
        assert self._run(_one_step(*calls), tmp_path) == [
            str((tmp_path / name).resolve()) for name in expected
        ]

    @pytest.mark.parametrize(("args", "success"), NOT_SENT_CASES)
    def test_nothing_is_sent(self, tmp_path, args, success):
        events = _file_tool_turn("write_file", args, success)
        assert self._run(events, tmp_path) == []


class TestTui:
    @pytest.fixture(autouse=True)
    def _textual(self):
        pytest.importorskip("textual")

    @staticmethod
    async def _run(monkeypatch, workspace, events) -> list[str]:
        from EvoScientist.paths import SessionDirs
        from tests.fakes.tui import run_turn

        _make_files(workspace.root)
        sent: list[str] = []
        await run_turn(
            monkeypatch,
            workspace,
            [events],
            prepare=lambda app: setattr(app, "_dirs", SessionDirs(workspace)),
            on_media_cb=sent.append,
        )
        return sent

    @pytest.mark.parametrize(("tool_name", "args"), CASES)
    async def test_a_media_file_is_sent(self, monkeypatch, workspace, tool_name, args):
        events = _file_tool_turn(tool_name, args)
        assert await self._run(monkeypatch, workspace, events) == [
            str((workspace.root / "fig.png").resolve())
        ]

    @pytest.mark.parametrize(("calls", "expected"), ONE_STEP_CASES)
    async def test_each_result_sends_its_own_file(
        self, monkeypatch, workspace, calls, expected
    ):
        assert await self._run(monkeypatch, workspace, _one_step(*calls)) == [
            str((workspace.root / name).resolve()) for name in expected
        ]

    @pytest.mark.parametrize(("args", "success"), NOT_SENT_CASES)
    async def test_nothing_is_sent(self, monkeypatch, workspace, args, success):
        events = _file_tool_turn("write_file", args, success)
        assert await self._run(monkeypatch, workspace, events) == []
