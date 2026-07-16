"""Tests for Telegram channel implementation."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from EvoScientist.channels.base import ChannelError
from EvoScientist.channels.telegram.channel import TelegramChannel, TelegramConfig


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

    async def test_start_raises_without_token(self):
        config = TelegramConfig(bot_token="")
        channel = TelegramChannel(config)
        with pytest.raises(ChannelError, match="bot token"):
            await channel.start()

    async def test_stop_when_not_running(self):
        config = TelegramConfig(bot_token="test")
        channel = TelegramChannel(config)
        await channel.stop()

    async def test_cleanup_is_idempotent(self):
        channel = TelegramChannel(TelegramConfig(bot_token="test"))
        app = SimpleNamespace(
            updater=SimpleNamespace(running=True, stop=AsyncMock()),
            running=True,
            shutdown=AsyncMock(),
        )

        async def stop_once():
            if not app.running:
                raise RuntimeError("This Application is not running!")
            app.running = False

        app.stop = AsyncMock(side_effect=stop_once)
        channel._app = app

        await channel._cleanup()
        await channel._cleanup()

        app.updater.stop.assert_awaited_once()
        app.stop.assert_awaited_once()
        app.shutdown.assert_awaited_once()
        assert channel._app is None

    async def test_cleanup_partially_initialized_application(self):
        channel = TelegramChannel(TelegramConfig(bot_token="test"))
        app = SimpleNamespace(
            updater=SimpleNamespace(running=False, stop=AsyncMock()),
            running=False,
            stop=AsyncMock(),
            shutdown=AsyncMock(),
        )
        channel._app = app

        await channel._cleanup()

        app.updater.stop.assert_not_awaited()
        app.stop.assert_not_awaited()
        app.shutdown.assert_awaited_once()
        assert channel._app is None

    async def test_send_returns_false_without_app(self):
        from EvoScientist.channels.base import OutboundMessage

        config = TelegramConfig(bot_token="test")
        channel = TelegramChannel(config)
        msg = OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="hello",
            metadata={"chat_id": "123"},
        )
        result = await channel.send(msg)
        assert result is False
