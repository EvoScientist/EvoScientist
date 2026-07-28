"""Tests for Telegram channel implementation."""

import sys
from datetime import datetime
from types import ModuleType, SimpleNamespace
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

    async def test_registered_handler_accepts_bot_commands(self, monkeypatch):
        class FakeFilter:
            def __init__(self, predicate):
                self._predicate = predicate

            def __or__(self, other):
                return FakeFilter(
                    lambda update: (
                        self.check_update(update) or other.check_update(update)
                    )
                )

            def __and__(self, other):
                return FakeFilter(
                    lambda update: (
                        self.check_update(update) and other.check_update(update)
                    )
                )

            def __invert__(self):
                return FakeFilter(lambda update: not self.check_update(update))

            def check_update(self, update):
                return self._predicate(update)

        class FakeMessageHandler:
            def __init__(self, message_filter, callback):
                self.filters = message_filter
                self.callback = callback

        app = SimpleNamespace(
            handlers=[],
            bot=SimpleNamespace(
                get_me=AsyncMock(return_value=SimpleNamespace(username="botname"))
            ),
            updater=SimpleNamespace(start_polling=AsyncMock()),
            initialize=AsyncMock(),
            start=AsyncMock(),
        )
        app.add_handler = app.handlers.append

        class FakeApplicationBuilder:
            def token(self, _token):
                return self

            def build(self):
                return app

        text_filter = FakeFilter(lambda update: update.message.text is not None)
        command_filter = FakeFilter(lambda update: update.message.is_command)
        false_filter = FakeFilter(lambda _update: False)
        fake_filters = SimpleNamespace(
            TEXT=text_filter,
            COMMAND=command_filter,
            PHOTO=false_filter,
            VOICE=false_filter,
            AUDIO=false_filter,
            Document=SimpleNamespace(ALL=false_filter),
            VIDEO=false_filter,
            Sticker=SimpleNamespace(ALL=false_filter),
            LOCATION=false_filter,
        )
        telegram_module = ModuleType("telegram")
        ext_module = ModuleType("telegram.ext")
        ext_module.ApplicationBuilder = FakeApplicationBuilder
        ext_module.MessageHandler = FakeMessageHandler
        ext_module.filters = fake_filters
        telegram_module.ext = ext_module
        monkeypatch.setitem(sys.modules, "telegram", telegram_module)
        monkeypatch.setitem(sys.modules, "telegram.ext", ext_module)

        channel = TelegramChannel(
            TelegramConfig(bot_token="test", include_attachments=False)
        )
        await channel.start()

        update = SimpleNamespace(message=SimpleNamespace(text="/help", is_command=True))
        assert app.handlers[0].filters.check_update(update) is True
        assert app.handlers[0].callback == channel._on_message

    async def test_group_command_suffix_is_removed_before_enqueue(self):
        channel = TelegramChannel(
            TelegramConfig(bot_token="test", include_attachments=False)
        )
        channel._bot_username = "botname"
        update = self._text_update("/stop@botname", chat_type="supergroup")

        await channel._on_message(update, None)

        message = await channel._queue.get()
        assert message.content == "/stop"
        assert message.is_group is True
        assert message.was_mentioned is True

    async def test_private_command_suffix_is_removed_before_enqueue(self):
        channel = TelegramChannel(
            TelegramConfig(bot_token="test", include_attachments=False)
        )
        channel._bot_username = "botname"
        update = self._text_update("/stop@botname")

        await channel._on_message(update, None)

        message = await channel._queue.get()
        assert message.content == "/stop"
        assert message.is_group is False

    async def test_bare_group_command_passes_mention_gating(self):
        channel = TelegramChannel(
            TelegramConfig(bot_token="test", include_attachments=False)
        )
        channel._bot_username = "botname"
        update = self._text_update("/stop", chat_type="supergroup")

        await channel._on_message(update, None)

        message = await channel._queue.get()
        assert message.content == "/stop"
        assert message.was_mentioned is True

    async def test_group_command_for_other_bot_is_ignored(self):
        channel = TelegramChannel(
            TelegramConfig(bot_token="test", include_attachments=False)
        )
        channel._bot_username = "botname"
        update = self._text_update("/stop@otherbot", chat_type="supergroup")

        await channel._on_message(update, None)

        assert channel._queue.empty()

    async def test_group_command_bypasses_buffered_history(self):
        channel = TelegramChannel(
            TelegramConfig(bot_token="test", include_attachments=False)
        )
        channel._bot_username = "botname"

        chatter = self._text_update("background chatter", chat_type="supergroup")
        await channel._on_message(chatter, None)
        assert channel._queue.empty()

        command = self._text_update("/help@botname", chat_type="supergroup")
        command.message.message_id = 790
        await channel._on_message(command, None)

        message = await channel._queue.get()
        assert message.content == "/help"
        assert message.is_group is True
        assert message.was_mentioned is True

    async def test_start_command_flows_to_shared_dispatch(self):
        channel = TelegramChannel(
            TelegramConfig(bot_token="test", include_attachments=False)
        )
        update = self._text_update("/start")

        await channel._on_message(update, None)

        message = await channel._queue.get()
        assert message.content == "/start"

    @staticmethod
    def _text_update(text, *, chat_type="private"):
        message = SimpleNamespace(
            from_user=SimpleNamespace(id=123),
            chat_id=456,
            chat=SimpleNamespace(type=chat_type),
            text=text,
            caption=None,
            date=datetime(2026, 1, 1),
            message_id=789,
        )
        return SimpleNamespace(message=message)
