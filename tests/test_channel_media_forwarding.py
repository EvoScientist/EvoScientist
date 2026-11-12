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


def _consumer(workspace, gateway, **kwargs):
    from EvoScientist.channels.bus.message_bus import MessageBus
    from EvoScientist.channels.channel_manager import ChannelManager
    from EvoScientist.channels.consumer import InboundConsumer
    from tests.fakes.common import StubChannel

    bus = MessageBus()
    manager = ChannelManager(bus, media_dir=workspace.root / "media")
    manager.register(StubChannel())
    return InboundConsumer(
        bus=bus,
        manager=manager,
        agent=MagicMock(),
        thread_id="",
        workspace=workspace,
        graph_gateway=gateway,
        inference_timeout=5.0,
        drain_timeout=1.0,
        **kwargs,
    )


async def _consumer_turn(consumer) -> list:
    """Run one turn and return everything it queued for the chat."""
    from EvoScientist.channels.bus.events import InboundMessage

    msg = InboundMessage(channel="stub", sender_id="u1", chat_id="c1", content="go")
    await consumer._stream_with_hitl(msg, None, "t1", msg.session_key)
    queued = []
    while not consumer.bus.outbound.empty():
        queued.append(consumer.bus.outbound.get_nowait())
    return queued


def _queued_media(queued) -> list[str]:
    return [path for out in queued for path in out.media]


class TestStandaloneConsumer:
    """``EvoSci channel --agent`` replies through ``InboundConsumer``."""

    @staticmethod
    async def _run(workspace, events) -> list[str]:
        _make_files(workspace.root)
        consumer = _consumer(workspace, FakeGraphGateway(events=events))
        return _queued_media(await _consumer_turn(consumer))

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

    async def test_the_file_goes_out_before_the_reply(self, workspace):
        _make_files(workspace.root)
        events = _file_tool_turn("write_file", {"file_path": "/fig.png"})
        consumer = _consumer(workspace, FakeGraphGateway(events=events))
        queued = await _consumer_turn(consumer)
        assert [(out.content, out.media) for out in queued] == [
            ("", [str((workspace.root / "fig.png").resolve())]),
            ("finished", []),
        ]


class TestConsumerTurnEnds:
    """A file collected before the turn ends early is still sent."""

    @staticmethod
    async def _run(monkeypatch, workspace, outcome=None, error=None) -> list[str]:
        """A turn that writes ``fig.png``, then pauses for an approval answered
        with *outcome*, or fails with *error*."""
        import EvoScientist.channels.consumer as consumer_mod
        from tests.fakes.common import interrupt_event

        _make_files(workspace.root)
        head = _file_tool_turn("write_file", {"file_path": "/fig.png"})[:2]

        async def _stream(_request):
            for event in head:
                yield event
            if error is not None:
                raise error
            yield interrupt_event(1)

        async def _resolve(*_args, **_kwargs):
            return outcome

        monkeypatch.setattr(consumer_mod, "resolve_approval", _resolve)
        consumer = _consumer(workspace, FakeGraphGateway(stream=_stream))
        return _queued_media(await _consumer_turn(consumer))

    @pytest.mark.parametrize(
        "outcome",
        [
            pytest.param({"rejected": True}, id="rejected"),
            pytest.param({}, id="unanswered"),
            pytest.param({"unrecognized_reply": "what?"}, id="unrecognized-reply"),
            pytest.param({"budget_exhausted": True}, id="round-limit"),
        ],
    )
    async def test_an_approval_that_ends_the_turn(
        self, monkeypatch, workspace, outcome
    ):
        from EvoScientist.channels.interaction import ApprovalOutcome

        sent = await self._run(monkeypatch, workspace, ApprovalOutcome(**outcome))
        assert sent == [str((workspace.root / "fig.png").resolve())]

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(TimeoutError(), id="timeout"),
            pytest.param(RuntimeError(), id="error"),
        ],
    )
    async def test_a_failed_stream(self, monkeypatch, workspace, error):
        sent = await self._run(monkeypatch, workspace, error=error)
        assert sent == [str((workspace.root / "fig.png").resolve())]


class TestOnlySandboxFilesAreSent:
    """A path is read as the sandbox reads it, so only a file inside the work
    folder (or the media folder) is ever sent, never a host file elsewhere."""

    @pytest.fixture
    def folders(self, tmp_path, monkeypatch):
        """A work folder, and a host folder the sandbox reads like ``/home/``."""
        import EvoScientist.backends as backends_mod

        work_dir = tmp_path / "ws"
        host = tmp_path / "host"
        work_dir.mkdir()
        host.mkdir()
        (host / "passport.pdf").write_bytes(b"secret")
        monkeypatch.setattr(backends_mod, "_SYSTEM_PATH_PREFIXES", (f"{host}/",))
        return work_dir, host

    def test_a_host_file_the_workspace_lacks_is_not_sent(self, folders):
        work_dir, host = folders
        assert (
            display_mod.resolve_media_path(str(host / "passport.pdf"), work_dir) == ""
        )

    def test_a_host_path_does_not_name_a_workspace_file_of_that_name(self, folders):
        work_dir, host = folders
        (work_dir / "passport.pdf").write_bytes(b"x")
        assert (
            display_mod.resolve_media_path(str(host / "passport.pdf"), work_dir) == ""
        )

    def test_the_rich_loop_does_not_send_the_host_file(self, folders, monkeypatch):
        monkeypatch.setattr(display_mod.console, "print", lambda *_a, **_k: None)
        work_dir, host = folders
        events = _file_tool_turn("write_file", {"file_path": f"{host}/passport.pdf"})
        assert _rich_sent(events, work_dir) == []

    async def test_the_consumer_does_not_send_the_host_file(
        self, workspace, tmp_path, monkeypatch
    ):
        import EvoScientist.backends as backends_mod

        host = tmp_path / "host"
        host.mkdir()
        (host / "passport.pdf").write_bytes(b"secret")
        monkeypatch.setattr(backends_mod, "_SYSTEM_PATH_PREFIXES", (f"{host}/",))
        events = _file_tool_turn("write_file", {"file_path": f"{host}/passport.pdf"})
        consumer = _consumer(workspace, FakeGraphGateway(events=events))
        assert _queued_media(await _consumer_turn(consumer)) == []

    def test_a_file_outside_the_work_folder_is_not_sent(self, tmp_path, monkeypatch):
        """Whatever spelling names it: relative, host-absolute, ``..``."""
        work_dir = tmp_path / "ws"
        work_dir.mkdir()
        (tmp_path / "host.png").write_bytes(b"x")
        monkeypatch.chdir(tmp_path)
        for path in ("host.png", str(tmp_path / "host.png"), "/../host.png"):
            assert display_mod.resolve_media_path(path, work_dir) == "", path

    def test_a_link_out_of_the_work_folder_is_not_sent(self, tmp_path):
        work_dir = tmp_path / "ws"
        work_dir.mkdir()
        (tmp_path / "host.png").write_bytes(b"x")
        (work_dir / "link.png").symlink_to(tmp_path / "host.png")
        assert display_mod.resolve_media_path("/link.png", work_dir) == ""

    def test_without_a_work_folder_nothing_is_sent(self, tmp_path, monkeypatch):
        (tmp_path / "fig.png").write_bytes(b"x")
        monkeypatch.chdir(tmp_path)
        assert display_mod.resolve_media_path("fig.png", None) == ""

    def test_a_file_in_the_media_folder_is_sent(self, tmp_path):
        """In ``--mode=run`` the sandbox reaches attachments by real path."""
        work_dir = tmp_path / "runs" / "r1"
        media_dir = tmp_path / "media"
        work_dir.mkdir(parents=True)
        media_dir.mkdir()
        (media_dir / "photo.png").write_bytes(b"x")
        assert display_mod.resolve_media_path(
            str(media_dir / "photo.png"), work_dir, media_dir=media_dir
        ) == str((media_dir / "photo.png").resolve())

    def test_dangerous_mode_sends_the_host_file(self, folders):
        """There the sandbox writes host paths as given."""
        work_dir, host = folders
        path = str(host / "passport.pdf")
        assert display_mod.resolve_media_path(path, work_dir, dangerous=True) == path


class TestRoutedPaths:
    """A path under one of the agent's routes names that tier's file."""

    LOGO = "/skills/demo/assets/logo.svg"

    @pytest.fixture
    def run_dirs(self, workspace):
        """A ``--mode=run`` folder holding a same-named decoy of each tier file."""
        from EvoScientist.paths import SessionDirs

        dirs = SessionDirs(workspace, workspace.runs_dir / "r1")
        for decoy in ("skills/demo/assets/logo.svg", "memories/chart.png"):
            (dirs.work_dir / decoy).parent.mkdir(parents=True, exist_ok=True)
            (dirs.work_dir / decoy).write_bytes(b"decoy")
        return dirs

    @staticmethod
    def _resolve(dirs, path, **kwargs) -> str:
        from EvoScientist.EvoScientist import file_tool_routes

        return display_mod.resolve_media_path(
            path,
            dirs.work_dir,
            media_dir=dirs.sandbox_media_dir,
            routes=file_tool_routes(dirs.workspace),
            **kwargs,
        )

    @pytest.mark.parametrize("dangerous", [False, True])
    def test_a_skills_write_sends_the_workspace_tier_file(self, run_dirs, dangerous):
        logo = run_dirs.workspace.skills_dir / "demo" / "assets" / "logo.svg"
        logo.parent.mkdir(parents=True)
        logo.write_bytes(b"x")
        assert self._resolve(run_dirs, self.LOGO, dangerous=dangerous) == str(
            logo.resolve()
        )

    def test_a_skills_read_sends_the_tier_it_found(self, run_dirs):
        from EvoScientist import paths

        logo = paths.GLOBAL_SKILLS_DIR / "demo" / "assets" / "logo.svg"
        logo.parent.mkdir(parents=True)
        logo.write_bytes(b"x")
        assert self._resolve(run_dirs, self.LOGO) == str(logo.resolve())

    def test_a_memories_path_sends_the_memory_file(self, run_dirs):
        from EvoScientist import paths

        chart = paths.MEMORIES_DIR / "chart.png"
        chart.parent.mkdir(parents=True, exist_ok=True)
        chart.write_bytes(b"x")
        assert self._resolve(run_dirs, "/memories/chart.png") == str(chart.resolve())

    @pytest.mark.parametrize("path", [LOGO, "/memories/chart.png"])
    def test_no_tier_file_sends_nothing_not_the_work_folder_file(self, run_dirs, path):
        assert self._resolve(run_dirs, path) == ""

    def test_a_tier_without_files_on_disk_sends_nothing(self, run_dirs):
        from deepagents.backends.protocol import BackendProtocol

        assert (
            display_mod.resolve_media_path(
                self.LOGO,
                run_dirs.work_dir,
                routes={"/skills/": MagicMock(spec=BackendProtocol)},
            )
            == ""
        )

    async def test_the_consumer_sends_the_memory_file(self, workspace):
        from EvoScientist import paths

        chart = paths.MEMORIES_DIR / "chart.png"
        chart.parent.mkdir(parents=True, exist_ok=True)
        chart.write_bytes(b"x")
        (workspace.root / "memories").mkdir()
        (workspace.root / "memories" / "chart.png").write_bytes(b"decoy")
        events = _file_tool_turn("read_file", {"file_path": "/memories/chart.png"})
        consumer = _consumer(workspace, FakeGraphGateway(events=events))
        assert _queued_media(await _consumer_turn(consumer)) == [str(chart.resolve())]


class TestSandboxDangerousMode:
    """Host paths are read as given only for a sandbox this session built."""

    @staticmethod
    def _config(dangerous: bool):
        from types import SimpleNamespace

        return SimpleNamespace(dangerous_mode=dangerous)

    @pytest.mark.parametrize("dangerous", [False, True])
    def test_an_in_process_agent_follows_the_config(self, dangerous):
        from EvoScientist.gateway.local import LocalGraphGateway
        from EvoScientist.gateway.runtime import sandbox_dangerous_mode

        gateway = LocalGraphGateway(thread_store=MagicMock())
        assert sandbox_dangerous_mode(gateway, self._config(dangerous)) is dangerous

    def test_a_server_backed_session_is_never_dangerous(self):
        """The server's sandbox keeps its own mode, which the session can't see."""
        from EvoScientist.gateway.composite import CompositeGraphGateway
        from EvoScientist.gateway.local import LocalGraphGateway
        from EvoScientist.gateway.runtime import sandbox_dangerous_mode
        from EvoScientist.gateway.server import LangGraphServerGateway

        gateway = CompositeGraphGateway(
            read=LocalGraphGateway(thread_store=MagicMock()),
            execute=object.__new__(LangGraphServerGateway),
        )
        assert sandbox_dangerous_mode(gateway, self._config(True)) is False

    async def test_a_server_backed_standalone_does_not_send_the_host_file(
        self, workspace, tmp_path, monkeypatch
    ):
        """``channel --agent`` with ``--dangerous`` against a server."""
        import EvoScientist.backends as backends_mod
        from EvoScientist.gateway.runtime import sandbox_dangerous_mode
        from EvoScientist.gateway.server import LangGraphServerGateway

        host = tmp_path / "host"
        host.mkdir()
        (host / "private.pdf").write_bytes(b"secret")
        monkeypatch.setattr(backends_mod, "_SYSTEM_PATH_PREFIXES", (f"{host}/",))
        events = _file_tool_turn("read_file", {"file_path": f"{host}/private.pdf"})
        gateway = FakeGraphGateway(events=events)
        consumer = _consumer(
            workspace,
            gateway,
            dangerous=sandbox_dangerous_mode(
                object.__new__(LangGraphServerGateway), self._config(True)
            ),
        )
        assert _queued_media(await _consumer_turn(consumer)) == []
