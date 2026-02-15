"""Tests for Discord channel implementation."""

import asyncio

import pytest

from EvoScientist.channels.discord.channel import DiscordChannel, DiscordConfig
from EvoScientist.channels.base import ChannelError


def _run(coro):
    """Run an async coroutine safely, creating a fresh event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestDiscordConfig:
    def test_default_values(self):
        config = DiscordConfig()
        assert config.bot_token == ""
        assert config.allowed_senders is None
        assert config.allowed_channels is None
        assert config.text_chunk_limit == 4096

    def test_custom_values(self):
        config = DiscordConfig(
            bot_token="test-token",
            allowed_senders={"111"},
            allowed_channels={"222"},
            text_chunk_limit=1000,
        )
        assert config.bot_token == "test-token"
        assert config.allowed_senders == {"111"}
        assert config.allowed_channels == {"222"}
        assert config.text_chunk_limit == 1000


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
