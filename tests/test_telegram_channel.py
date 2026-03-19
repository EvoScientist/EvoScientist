"""Tests for Telegram channel implementation."""

import pytest

from EvoScientist.channels.base import ChannelError
from EvoScientist.channels.telegram.channel import TelegramChannel, TelegramConfig
from tests.conftest import run_async as _run


class TestTelegramConfig:
    def test_default_values(self):
        config = TelegramConfig()
        assert config.bot_token == ""
        assert config.allowed_senders is None
        assert config.text_chunk_limit == 4096

    def test_custom_values(self):
        config = TelegramConfig(
            bot_token="test-token",
            allowed_senders={"123", "456"},
            text_chunk_limit=2000,
        )
        assert config.bot_token == "test-token"
        assert config.allowed_senders == {"123", "456"}
        assert config.text_chunk_limit == 2000


class TestTelegramChannel:
    def test_init(self):
        config = TelegramConfig(bot_token="test")
        channel = TelegramChannel(config)
        assert channel.config is config
        assert channel._running is False

    def test_start_raises_without_token(self):
        config = TelegramConfig(bot_token="")
        channel = TelegramChannel(config)
        with pytest.raises(ChannelError, match="bot token"):
            _run(channel.start())

    def test_stop_when_not_running(self):
        config = TelegramConfig(bot_token="test")
        channel = TelegramChannel(config)
        _run(channel.stop())

    def test_send_returns_false_without_app(self):
        from EvoScientist.channels.base import OutboundMessage

        config = TelegramConfig(bot_token="test")
        channel = TelegramChannel(config)
        msg = OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="hello",
            metadata={"chat_id": "123"},
        )
        result = _run(channel.send(msg))
        assert result is False
