"""A media file the agent writes or reads is sent to the channel chat.

Covers the Rich loop, the TUI and the standalone runner's consumer, which all
forward through ``display.forward_media_result``. deepagents' ``write_file``
and ``read_file`` name the file in ``file_path``.
"""

from __future__ import annotations

import asyncio
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


# Every spelling the file tools accept for ``<work_dir>/fig.png``; ``{root}``
# is the work folder.
PATH_CASES = [
    pytest.param("/fig.png", id="virtual"),
    pytest.param("fig.png", id="relative"),
    pytest.param("{root}/fig.png", id="host-absolute"),
]


def _path_turn(tool_name: str, path: str, root) -> list[dict]:
    return _file_tool_turn(tool_name, {"file_path": path.format(root=root)})


def _subagent_turn(tool_name: str, path: str, root) -> list[dict]:
    """A turn whose sub-agent calls *tool_name* on *path*."""
    sa = {"subagent": "research", "instance_id": "task:sa1"}
    return [
        {
            "type": "subagent_start",
            "name": "research",
            "description": "",
            "instance_id": "task:sa1",
            "tool_call_id": "task1",
        },
        {
            "type": "subagent_tool_call",
            **sa,
            "name": tool_name,
            "args": {"file_path": path.format(root=root)},
            "id": "sc1",
        },
        {
            "type": "subagent_tool_result",
            **sa,
            "name": tool_name,
            "content": "ok",
            "success": True,
            "id": "sc1",
        },
        *_END,
    ]


def _make_files(root) -> None:
    for name in ("fig.png", "a.png", "b.png", "notes.txt"):
        (root / name).write_bytes(b"x")


def _rich_sent(events, work_dir) -> list[str]:
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


async def _tui_sent(monkeypatch, workspace, events) -> list[str]:
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


class TestRichLoop:
    @pytest.fixture(autouse=True)
    def _quiet(self, monkeypatch):
        monkeypatch.setattr(display_mod.console, "print", lambda *_a, **_k: None)

    @staticmethod
    def _run(events, work_dir) -> list[str]:
        return _rich_sent(events, work_dir)

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

    @pytest.mark.parametrize("tool_name", ["write_file", "read_file"])
    @pytest.mark.parametrize("path", PATH_CASES)
    def test_every_path_spelling_is_sent(self, tmp_path, tool_name, path):
        events = _path_turn(tool_name, path, tmp_path)
        assert self._run(events, tmp_path) == [str((tmp_path / "fig.png").resolve())]

    @pytest.mark.parametrize("tool_name", ["write_file", "read_file"])
    def test_a_subagent_file_is_sent(self, tmp_path, tool_name):
        events = _subagent_turn(tool_name, "/fig.png", tmp_path)
        assert self._run(events, tmp_path) == [str((tmp_path / "fig.png").resolve())]


class TestTui:
    @pytest.fixture(autouse=True)
    def _textual(self):
        pytest.importorskip("textual")

    @staticmethod
    async def _run(monkeypatch, workspace, events) -> list[str]:
        return await _tui_sent(monkeypatch, workspace, events)

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

    @pytest.mark.parametrize("tool_name", ["write_file", "read_file"])
    @pytest.mark.parametrize("path", PATH_CASES)
    async def test_every_path_spelling_is_sent(
        self, monkeypatch, workspace, tool_name, path
    ):
        events = _path_turn(tool_name, path, workspace.root)
        assert await self._run(monkeypatch, workspace, events) == [
            str((workspace.root / "fig.png").resolve())
        ]

    @pytest.mark.parametrize("tool_name", ["write_file", "read_file"])
    async def test_a_subagent_file_is_sent(self, monkeypatch, workspace, tool_name):
        events = _subagent_turn(tool_name, "/fig.png", workspace.root)
        assert await self._run(monkeypatch, workspace, events) == [
            str((workspace.root / "fig.png").resolve())
        ]

    @pytest.mark.parametrize("turn", [_path_turn, _subagent_turn])
    @pytest.mark.parametrize("tool_name", ["write_file", "read_file"])
    @pytest.mark.parametrize(
        "path", [*PATH_CASES, pytest.param("/gone.png", id="missing")]
    )
    async def test_the_rich_loop_sends_the_same(
        self, monkeypatch, workspace, turn, tool_name, path
    ):
        events = turn(tool_name, path, workspace.root)
        rich = await asyncio.to_thread(_rich_sent, events, workspace.root)
        assert await self._run(monkeypatch, workspace, events) == rich


class TestStandaloneConsumer:
    """``EvoSci channel --agent`` replies through ``InboundConsumer``."""

    @staticmethod
    async def _run(workspace, events) -> list[str]:
        from EvoScientist.channels.bus.events import InboundMessage
        from EvoScientist.channels.bus.message_bus import MessageBus
        from EvoScientist.channels.channel_manager import ChannelManager
        from EvoScientist.channels.consumer import InboundConsumer
        from tests.fakes.common import StubChannel

        _make_files(workspace.root)
        bus = MessageBus()
        manager = ChannelManager(bus, media_dir=workspace.root / "media")
        manager.register(StubChannel())
        consumer = InboundConsumer(
            bus=bus,
            manager=manager,
            agent=MagicMock(),
            thread_id="",
            workspace=workspace,
            graph_gateway=FakeGraphGateway(events=events),
            inference_timeout=5.0,
            drain_timeout=1.0,
        )
        await bus.publish_inbound(
            InboundMessage(channel="stub", sender_id="u1", chat_id="c1", content="go")
        )
        task = asyncio.create_task(consumer.run())
        try:
            outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=5.0)
        finally:
            await consumer.stop()
            await task
        return outbound.media

    @pytest.mark.parametrize("tool_name", ["write_file", "read_file"])
    @pytest.mark.parametrize("path", PATH_CASES)
    async def test_every_path_spelling_is_sent(self, workspace, tool_name, path):
        events = _path_turn(tool_name, path, workspace.root)
        assert await self._run(workspace, events) == [
            str((workspace.root / "fig.png").resolve())
        ]

    async def test_a_subagent_file_is_sent(self, workspace):
        events = _subagent_turn("write_file", "/fig.png", workspace.root)
        assert await self._run(workspace, events) == [
            str((workspace.root / "fig.png").resolve())
        ]

    @pytest.mark.parametrize(("args", "success"), NOT_SENT_CASES)
    async def test_nothing_is_sent(self, workspace, args, success):
        events = _file_tool_turn("write_file", args, success)
        assert await self._run(workspace, events) == []


def test_the_work_folder_wins_over_the_path_as_given(tmp_path, monkeypatch):
    """A path both folders hold is the work folder's file; the rest fall back."""
    work_dir = tmp_path / "ws"
    work_dir.mkdir()
    (work_dir / "fig.png").write_bytes(b"x")
    (tmp_path / "fig.png").write_bytes(b"x")
    (tmp_path / "host.png").write_bytes(b"x")
    monkeypatch.chdir(tmp_path)

    assert display_mod.resolve_media_path("fig.png", work_dir) == str(
        (work_dir / "fig.png").resolve()
    )
    assert display_mod.resolve_media_path("host.png", work_dir) == "host.png"
    assert display_mod.resolve_media_path("fig.png", None) == "fig.png"
    assert display_mod.resolve_media_path("/gone.png", work_dir) == ""
