"""Tests for Discord channel implementation."""

import pytest

from EvoScientist.channels.base import ChannelError
from EvoScientist.channels.discord.channel import DiscordChannel, DiscordConfig
from tests.conftest import run_async as _run


class TestDiscordChannel:
    def test_init(self):
        config = DiscordConfig(bot_token="test")
        channel = DiscordChannel(config)
        assert channel.config is config
        assert channel._running is False

    def test_start_raises_without_token_or_library(self):
        config = DiscordConfig(bot_token="")
        channel = DiscordChannel(config)
        with pytest.raises(ChannelError):
            _run(channel.start())

    def test_stop_when_not_running(self):
        config = DiscordConfig(bot_token="test")
        channel = DiscordChannel(config)
        _run(channel.stop())

    def test_send_returns_false_without_client(self):
        from EvoScientist.channels.base import OutboundMessage

        config = DiscordConfig(bot_token="test")
        channel = DiscordChannel(config)
        msg = OutboundMessage(
            channel="discord",
            chat_id="123",
            content="hello",
            metadata={"chat_id": "123"},
        )
        result = _run(channel.send(msg))
        assert result is False
