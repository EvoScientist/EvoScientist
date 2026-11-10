"""A media file the agent writes or reads is sent to the channel chat (Rich loop, TUI).

deepagents' ``write_file`` and ``read_file`` name the file in ``file_path``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from EvoScientist.stream import display as display_mod
from tests.fakes.common import FakeGraphGateway


def _file_tool_turn(tool_name: str, args: dict) -> list[dict]:
    return [
        {"type": "tool_call", "id": "tc1", "name": tool_name, "args": args},
        {
            "type": "tool_result",
            "id": "tc1",
            "name": tool_name,
            "content": "ok",
            "success": True,
        },
        {"type": "text", "content": "finished"},
        {"type": "done", "response": "finished"},
    ]


CASES = [
    pytest.param("write_file", {"file_path": "/fig.png", "content": "x"}, id="write"),
    pytest.param("read_file", {"file_path": "/fig.png"}, id="read"),
    pytest.param("write_file", {"path": "/fig.png", "content": "x"}, id="legacy"),
]


@pytest.mark.parametrize(("tool_name", "args"), CASES)
def test_tool_call_file_path_reads_file_path_then_path(tool_name, args):
    assert display_mod.tool_call_file_path(args) == "/fig.png"


def test_tool_call_file_path_is_empty_without_a_path():
    assert display_mod.tool_call_file_path({"content": "x"}) == ""


class TestRichLoop:
    @pytest.fixture(autouse=True)
    def _quiet(self, monkeypatch):
        monkeypatch.setattr(display_mod.console, "print", lambda *_a, **_k: None)

    @staticmethod
    def _run(events, work_dir) -> list[str]:
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
        figure = tmp_path / "fig.png"
        figure.write_bytes(b"png")
        assert self._run(_file_tool_turn(tool_name, args), tmp_path) == [
            str(figure.resolve())
        ]

    def test_a_non_media_file_is_not_sent(self, tmp_path):
        (tmp_path / "notes.txt").write_text("x")
        args = {"file_path": "/notes.txt", "content": "x"}
        assert self._run(_file_tool_turn("write_file", args), tmp_path) == []


class TestTui:
    @pytest.fixture(autouse=True)
    def _textual(self):
        pytest.importorskip("textual")

    @pytest.mark.parametrize(("tool_name", "args"), CASES)
    async def test_a_media_file_is_sent(self, monkeypatch, workspace, tool_name, args):
        from EvoScientist.paths import SessionDirs
        from tests.fakes.tui import run_turn

        figure = workspace.root / "fig.png"
        figure.write_bytes(b"png")
        sent: list[str] = []
        await run_turn(
            monkeypatch,
            workspace,
            [_file_tool_turn(tool_name, args)],
            prepare=lambda app: setattr(app, "_dirs", SessionDirs(workspace)),
            on_media_cb=sent.append,
        )
        assert sent == [str(figure.resolve())]
