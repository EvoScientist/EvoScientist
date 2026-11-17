"""Received attachments reach streaming from the Rich CLI channel queue."""

import asyncio
import queue
from contextlib import asynccontextmanager, nullcontext
from unittest.mock import AsyncMock, MagicMock

from EvoScientist.cli import channel as channel_mod
from EvoScientist.cli import interactive as interactive_mod
from EvoScientist.cli.channel import ChannelMessage, _register_channel_request
from EvoScientist.config import EvoScientistConfig
from EvoScientist.gateway import RuntimeGateways
from EvoScientist.paths import SessionDirs
from tests.fakes.common import FakeGraphGateway


def test_rich_channel_handler_passes_received_media(monkeypatch, workspace):
    msg = ChannelMessage(
        msg_id="rich-received-media",
        content="read these attachments",
        sender="user",
        channel_type="imessage",
        chat_id="chat",
        media=[str(workspace.root / "fig.png"), str(workspace.root / "a.png")],
    )
    messages = queue.Queue()
    messages.put(msg)
    replied = asyncio.Event()
    stream = AsyncMock(return_value="ok")
    agent = object()
    loader = MagicMock(agent=agent, await_ready=AsyncMock(return_value=agent))
    gateway = FakeGraphGateway(generated_thread_ids=["rich-media-thread"])

    @asynccontextmanager
    async def checkpointer():
        yield None

    async def prompt(*_args, **_kwargs):
        try:
            await asyncio.wait_for(replied.wait(), timeout=5)
        except TimeoutError:
            pass
        raise EOFError

    def respond(msg_id, response):
        assert msg_id == msg.msg_id
        assert response == "ok"
        replied.set()

    monkeypatch.setattr(channel_mod, "_channel_requests", {})
    monkeypatch.setattr(channel_mod, "_session_requests", {})
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
    monkeypatch.setattr(interactive_mod, "run_streaming_async", stream)
    monkeypatch.setattr("EvoScientist.config.load_config", lambda: None)
    monkeypatch.setattr(
        "EvoScientist.update_check.is_update_available", lambda: (False, None)
    )
    monkeypatch.setattr(
        "EvoScientist.cli.commands._pending_skill_proposals_message", lambda _root: None
    )
    _register_channel_request(msg)
    try:
        interactive_mod.cmd_interactive(
            dirs=SessionDirs(workspace),
            config=EvoScientistConfig(),
            show_thinking=False,
            channel_send_thinking=False,
        )
    finally:
        channel_mod.forget_channel_origin("rich-media-thread")
    stream.assert_awaited_once()
    assert stream.call_args.kwargs["message"] == msg.content
    assert stream.call_args.kwargs["received_media"] == msg.media
    assert messages.empty()
    assert replied.is_set()
