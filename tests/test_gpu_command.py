from __future__ import annotations

import asyncio
import subprocess
from unittest.mock import MagicMock

import pytest

from EvoScientist.channels.bus import MessageBus
from EvoScientist.channels.bus.events import InboundMessage
from EvoScientist.channels.channel_manager import ChannelManager
from EvoScientist.channels.consumer import InboundConsumer
from EvoScientist.cli import channel as channel_cli
from EvoScientist.commands.channel_dispatch import execute_supported_channel_command
from tests.conftest import run_async as _run


class _CompletedProcess:
    def __init__(self, stdout: str = "", stderr: str = ""):
        self.stdout = stdout
        self.stderr = stderr


def _drain_queue(q) -> None:
    while not q.empty():
        try:
            q.get_nowait()
        except Exception:
            break


def test_execute_supported_channel_command_only_matches_exact_gpu(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: _CompletedProcess(stdout="GPU 0  70%\n"),
    )

    async def _test():
        result = await execute_supported_channel_command(
            " /GPU ",
            agent=None,
            thread_id="thread-1",
        )
        not_a_command = await execute_supported_channel_command(
            "/gpu now",
            agent=None,
            thread_id="thread-1",
        )

        assert result == "```text\nGPU 0  70%\n```"
        assert not_a_command is None

    _run(_test())


def test_execute_supported_channel_command_reports_failures(monkeypatch):
    def _fail(*args, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=["nvidia-smi"],
            stderr="driver not loaded",
        )

    monkeypatch.setattr(subprocess, "run", _fail)

    async def _test():
        result = await execute_supported_channel_command(
            "/gpu",
            agent=None,
            thread_id="thread-1",
        )

        assert result is not None
        assert "GPU check failed" in result
        assert "driver not loaded" in result

    _run(_test())


@pytest.mark.parametrize(
    "channel_name",
    [
        "telegram",
        "discord",
        "slack",
        "wechat",
        "dingtalk",
        "feishu",
        "email",
        "qq",
        "signal",
        "imessage",
    ],
)
def test_gpu_command_bypasses_standalone_agent(monkeypatch, channel_name):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: _CompletedProcess(stdout="GPU shortcut output\n"),
    )

    async def _test():
        bus = MessageBus()
        manager = ChannelManager(bus)
        agent = MagicMock()
        consumer = InboundConsumer(
            bus=bus,
            manager=manager,
            agent=agent,
            thread_id="",
        )

        await consumer._handle_message(
            InboundMessage(
                channel=channel_name,
                sender_id="user-1",
                chat_id="chat-1",
                content="/gpu",
                message_id="msg-1",
            )
        )

        outbound = await bus.consume_outbound()
        assert outbound.channel == channel_name
        assert outbound.chat_id == "chat-1"
        assert outbound.reply_to == "msg-1"
        assert "GPU shortcut output" in outbound.content
        assert consumer.metrics["total_successes"] == 1
        assert agent.mock_calls == []

    _run(_test())


def test_gpu_command_bypasses_bus_mode_queue(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: _CompletedProcess(stdout="GPU shortcut output\n"),
    )
    _drain_queue(channel_cli._message_queue)

    async def _test():
        from tests.test_bus_integration import FakeChannel

        bus = MessageBus()
        manager = ChannelManager(bus)
        channel = FakeChannel()
        manager.register(channel)

        consumer = asyncio.create_task(channel_cli._bus_inbound_consumer(bus, manager))

        await bus.publish_inbound(
            InboundMessage(
                channel="fake",
                sender_id="user-1",
                chat_id="chat-1",
                content="/gpu",
                message_id="msg-1",
            )
        )

        outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=2.0)
        await asyncio.sleep(0.05)

        assert channel_cli._message_queue.empty()
        assert outbound.channel == "fake"
        assert outbound.chat_id == "chat-1"
        assert outbound.reply_to == "msg-1"
        assert "GPU shortcut output" in outbound.content
        assert manager._message_counts["fake"]["received"] == 1
        assert manager._message_counts["fake"]["sent"] == 1

        consumer.cancel()
        try:
            await consumer
        except asyncio.CancelledError:
            pass

    _run(_test())
