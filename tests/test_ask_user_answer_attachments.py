"""Attachments that answer a question belong to the paused channel turn."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from EvoScientist.channels.bus.events import InboundMessage
from tests.fakes.common import FakeGraphGateway, stub_consumer


def _read(path: str, call_id: str):
    yield {
        "type": "tool_call",
        "id": call_id,
        "name": "read_file",
        "args": {"file_path": path},
    }
    yield {
        "type": "tool_result",
        "id": call_id,
        "name": "read_file",
        "content": "paper contents",
        "success": True,
    }


def _answer_stream(root: Path, case: str):
    calls = 0

    async def stream(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            questions = [{"question": "Which paper should I summarise?"}]
            if case == "multiple":
                questions.append({"question": "Which other paper?"})
            yield {
                "type": "ask_user",
                "interrupt_id": "paper-question",
                "questions": questions,
            }
            return

        assert request.message.resume["paper-question"]["status"] == "answered"
        paper = root / "media" / "paper.pdf"
        if case == "modified":
            paper.write_bytes(b"edited paper contents")
        elif case == "replaced":
            replacement = paper.with_name("replacement.pdf")
            replacement.write_bytes(b"replacement paper")
            replacement.replace(paper)
        elif case == "alias":
            paper.with_name("alias.pdf").hardlink_to(paper)
        if case.startswith("initial-"):
            for event in _read("/media/initial.pdf", "initial"):
                yield event
        for event in _read(
            "/media/alias.pdf" if case == "alias" else "/media/paper.pdf", "paper"
        ):
            yield event
        if case == "multiple":
            for event in _read("/media/second.pdf", "second"):
                yield event
        yield {"type": "text", "content": "Summary complete"}
        yield {"type": "done", "content": "Summary complete"}

    return stream


async def _reply(bus, publish, root: Path, case: str):
    prompt = await asyncio.wait_for(bus.consume_outbound(), timeout=5)
    assert "Which paper" in prompt.content
    paper = root / "media" / "paper.pdf"
    paper.write_bytes(b"received paper")
    if case == "initial-modified":
        (root / "media" / "initial.pdf").write_bytes(b"edited initial attachment")
    answer = InboundMessage(
        channel="stub",
        sender_id="user",
        chat_id="chat",
        content="Please read /media/paper.pdf",
        media=[] if case == "text-only" else [str(paper)],
    )
    await publish(answer)
    if case == "multiple":
        prompt = await asyncio.wait_for(bus.consume_outbound(), timeout=5)
        assert "Which other paper" in prompt.content
        second = paper.with_name("second.pdf")
        second.write_bytes(b"second received paper")
        await publish(
            InboundMessage(
                channel="stub",
                sender_id="user",
                chat_id="chat",
                content="Also read /media/second.pdf",
                media=[str(second)],
            )
        )


def _initial_message(root: Path, case: str):
    (root / "media").mkdir(exist_ok=True)
    initial = root / "media" / "initial.pdf"
    initial.write_bytes(b"initial attachment")
    return InboundMessage(
        channel="stub",
        sender_id="user",
        chat_id="chat",
        content="Summarise a paper",
        media=[str(initial)] if case.startswith("initial-") else [],
    )


def _expected(root: Path, case: str):
    if case in ("modified", "replaced", "text-only"):
        return [str((root / "media" / "paper.pdf").resolve())]
    if case == "initial-modified":
        return [str((root / "media" / "initial.pdf").resolve())]
    return []


@pytest.mark.parametrize(
    "case",
    [
        "unchanged",
        "modified",
        "replaced",
        "alias",
        "initial-unchanged",
        "initial-modified",
        "text-only",
        "multiple",
    ],
)
async def test_standalone_question_answer_attachments(workspace, case):
    msg = _initial_message(workspace.root, case)
    consumer = stub_consumer(
        workspace, FakeGraphGateway(stream=_answer_stream(workspace.root, case))
    )
    task = asyncio.create_task(
        consumer._stream_with_hitl(msg, None, "question-thread", msg.session_key)
    )
    try:
        await _reply(consumer.bus, consumer._handle_message, workspace.root, case)
        await asyncio.wait_for(task, timeout=5)
        queued = []
        while not consumer.bus.outbound.empty():
            queued.append(consumer.bus.outbound.get_nowait())
        assert any(out.content == "Summary complete" for out in queued)
        assert [path for out in queued for path in out.media] == _expected(
            workspace.root, case
        )
        assert consumer._reply_registry._pending == {}
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def _rich_handler(monkeypatch, workspace, msg, gateway):
    import queue
    import threading
    from contextlib import asynccontextmanager, nullcontext
    from unittest.mock import AsyncMock, MagicMock

    from EvoScientist.cli import interactive as interactive_mod
    from EvoScientist.config import EvoScientistConfig
    from EvoScientist.gateway import RuntimeGateways
    from EvoScientist.paths import SessionDirs

    messages = queue.Queue()
    messages.put(msg)
    replied = threading.Event()
    responses = []
    agent = object()
    loader = MagicMock(agent=agent, await_ready=AsyncMock(return_value=agent))

    @asynccontextmanager
    async def checkpointer():
        yield None

    async def prompt(*_args, **_kwargs):
        await asyncio.to_thread(replied.wait, 10)
        raise EOFError

    def respond(msg_id, response):
        assert msg_id == msg.msg_id
        responses.append(response)
        replied.set()

    monkeypatch.setattr(interactive_mod, "_message_queue", messages)
    monkeypatch.setattr(interactive_mod, "_set_channel_response", respond)
    monkeypatch.setattr(interactive_mod, "get_checkpointer", checkpointer)
    monkeypatch.setattr(
        interactive_mod,
        "PromptSession",
        MagicMock(return_value=MagicMock(prompt_async=prompt)),
    )
    monkeypatch.setattr(interactive_mod, "patch_stdout", lambda **_kw: nullcontext())
    monkeypatch.setattr(
        interactive_mod, "BackgroundAgentLoader", MagicMock(return_value=loader)
    )
    monkeypatch.setattr(
        interactive_mod,
        "create_runtime_gateways_for_config",
        lambda *_args, **_kwargs: RuntimeGateways(
            thread_store=gateway.thread_store, graph_gateway=gateway
        ),
    )
    monkeypatch.setattr(
        interactive_mod,
        "build_session_status_snapshot",
        AsyncMock(
            return_value=interactive_mod.make_empty_status_snapshot("test-model")
        ),
    )
    monkeypatch.setattr("EvoScientist.config.load_config", lambda: None)
    monkeypatch.setattr(
        "EvoScientist.cli.commands._pending_skill_proposals_message", lambda _root: None
    )
    await asyncio.to_thread(
        interactive_mod.cmd_interactive,
        dirs=SessionDirs(workspace),
        config=EvoScientistConfig(),
        show_thinking=False,
        channel_send_thinking=False,
    )
    assert responses == ["Summary complete"]
    assert messages.empty()


async def _serve_handler(monkeypatch, workspace, msg, gateway):
    from unittest.mock import AsyncMock

    from EvoScientist.cli import commands
    from EvoScientist.gateway import RuntimeGateways
    from EvoScientist.paths import SessionDirs
    from EvoScientist.runtime import AsyncRuntime
    from tests.fakes.serve import runtime_state

    responses = []
    monkeypatch.setattr(
        commands,
        "_set_channel_response",
        lambda _id, response: responses.append(response),
    )
    monkeypatch.setattr(
        commands, "dispatch_channel_slash_command", AsyncMock(return_value=False)
    )

    def run():
        with AsyncRuntime(thread_name="question-attachment-test") as runtime:
            state = runtime_state(
                dirs=SessionDirs(workspace),
                runtime_gateways=RuntimeGateways(
                    thread_store=gateway.thread_store, graph_gateway=gateway
                ),
                async_runtime=runtime,
            )
            commands._serve_process_message(
                msg, runtime_state=state, model="test-model", show_thinking=False
            )

    await asyncio.to_thread(run)
    assert responses == ["Summary complete"]


async def _tui_handler(monkeypatch, workspace, msg, gateway):
    from unittest.mock import AsyncMock, MagicMock

    from EvoScientist.gateway import RuntimeGateways
    from EvoScientist.paths import SessionDirs
    from tests.fakes.tui import capture_app

    app = await capture_app(monkeypatch, workspace)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app._dirs = SessionDirs(workspace)
        app._runtime_gateways = RuntimeGateways(
            thread_store=gateway.thread_store, graph_gateway=gateway
        )
        monkeypatch.setattr(
            app, "_await_agent_ready", AsyncMock(return_value=MagicMock())
        )
        await app._process_channel_message(msg)


@pytest.mark.parametrize("handler", [_serve_handler, _rich_handler, _tui_handler])
@pytest.mark.parametrize(
    "case",
    [
        "unchanged",
        "modified",
        "replaced",
        "alias",
        "initial-unchanged",
        "initial-modified",
        "text-only",
        "multiple",
    ],
)
async def test_channel_question_answer_attachments(
    monkeypatch, workspace, handler, case
):
    import queue

    from EvoScientist.channels.bus.message_bus import MessageBus
    from EvoScientist.channels.channel_manager import ChannelManager
    from EvoScientist.channels.interaction import PendingReplyRegistry
    from EvoScientist.cli import channel as channel_mod
    from tests.fakes.common import StubChannel

    inbound = _initial_message(workspace.root, case)
    bus = MessageBus()
    manager = ChannelManager(bus, media_dir=workspace.root / "media")
    channel = StubChannel()
    manager.register(channel)
    sent = []

    async def send_media(*, file_path, **_kwargs):
        sent.append(file_path)
        return True

    monkeypatch.setattr(channel, "send_media", send_media)
    monkeypatch.setattr(channel_mod, "_reply_registry", PendingReplyRegistry())
    monkeypatch.setattr(channel_mod, "_bus_loop", asyncio.get_running_loop())
    monkeypatch.setattr(channel_mod, "_message_queue", queue.Queue())
    monkeypatch.setattr(channel_mod, "_channel_requests", {})
    monkeypatch.setattr(channel_mod, "_session_requests", {})
    monkeypatch.setattr(channel_mod, "_cancelled_channel_messages", set())
    msg = channel_mod.ChannelMessage(
        msg_id=f"{handler.__name__}-{case}",
        content=inbound.content,
        sender=inbound.sender_id,
        channel_type=inbound.channel,
        chat_id=inbound.chat_id,
        media=inbound.media,
        bus_ref=bus,
        channel_ref=channel,
    )
    channel_mod._register_channel_request(msg)
    gateway = FakeGraphGateway(
        stream=_answer_stream(workspace.root, case),
        generated_thread_ids=["question-attachment-thread"],
    )
    consumer = asyncio.create_task(channel_mod._bus_inbound_consumer(bus, manager))
    reply = asyncio.create_task(_reply(bus, bus.publish_inbound, workspace.root, case))
    try:
        await asyncio.wait_for(
            handler(monkeypatch, workspace, msg, gateway), timeout=15
        )
        await asyncio.wait_for(reply, timeout=5)
        assert len(gateway.requests) == 2
        assert gateway.requests[0].message == msg.content
        assert sent == _expected(workspace.root, case)
        assert channel_mod._message_queue.empty()
        assert channel_mod._reply_registry._pending == {}
    finally:
        consumer.cancel()
        reply.cancel()
        await asyncio.gather(consumer, reply, return_exceptions=True)
        channel_mod.forget_channel_origin("question-attachment-thread")
