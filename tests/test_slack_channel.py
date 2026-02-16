"""Tests for Slack channel implementation."""

import asyncio

import pytest

from EvoScientist.channels.slack.channel import SlackChannel, SlackConfig
from EvoScientist.channels.base import ChannelError


def _run(coro):
    """Run an async coroutine safely, creating a fresh event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestSlackConfig:
    def test_default_values(self):
        config = SlackConfig()
        assert config.bot_token == ""
        assert config.app_token == ""
        assert config.allowed_senders is None
        assert config.allowed_channels is None
        assert config.text_chunk_limit == 4096

    def test_custom_values(self):
        config = SlackConfig(
            bot_token="xoxb-test",
            app_token="xapp-test",
            allowed_senders={"U123"},
            allowed_channels={"C456"},
            text_chunk_limit=2000,
        )
        assert config.bot_token == "xoxb-test"
        assert config.app_token == "xapp-test"
        assert config.allowed_senders == {"U123"}
        assert config.allowed_channels == {"C456"}
        assert config.text_chunk_limit == 2000


class TestSlackChannel:
    def test_init(self):
        config = SlackConfig(bot_token="xoxb-test", app_token="xapp-test")
        channel = SlackChannel(config)
        assert channel.config is config
        assert channel._running is False

    def test_start_raises_without_bot_token(self):
        config = SlackConfig(bot_token="", app_token="xapp-test")
        channel = SlackChannel(config)
        with pytest.raises(ChannelError, match="bot token"):
            _run(channel.start())

    def test_start_raises_without_app_token(self):
        config = SlackConfig(bot_token="xoxb-test", app_token="")
        channel = SlackChannel(config)
        with pytest.raises(ChannelError, match="app token"):
            _run(channel.start())

    def test_stop_when_not_running(self):
        config = SlackConfig(bot_token="xoxb-test", app_token="xapp-test")
        channel = SlackChannel(config)
        _run(channel.stop())

    def test_send_returns_false_without_client(self):
        from EvoScientist.channels.base import OutboundMessage

        config = SlackConfig(bot_token="xoxb-test", app_token="xapp-test")
        channel = SlackChannel(config)
        msg = OutboundMessage(
            channel="slack",
            chat_id="C123",
            content="hello",
            metadata={"chat_id": "C123"},
        )
        result = _run(channel.send(msg))
        assert result is False


class TestSlackChannelRegistration:
    def test_slack_registered(self):
        from EvoScientist.channels.channel_manager import available_channels
        channels = available_channels()
        assert "slack" in channels
