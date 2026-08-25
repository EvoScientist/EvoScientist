"""Tests for Discord channel implementation."""

import pytest

from EvoScientist.channels.base import ChannelError
from EvoScientist.channels.discord.channel import DiscordChannel, DiscordConfig


class TestDiscordChannel:
    def test_init(self):
        config = DiscordConfig(bot_token="test")
        channel = DiscordChannel(config)
        assert channel.config is config
        assert channel._running is False

    async def test_start_raises_without_token_or_library(self):
        config = DiscordConfig(bot_token="")
        channel = DiscordChannel(config)
        with pytest.raises(ChannelError):
            await channel.start()

    async def test_stop_when_not_running(self):
        config = DiscordConfig(bot_token="test")
        channel = DiscordChannel(config)
        await channel.stop()

    async def test_send_returns_false_without_client(self):
        from EvoScientist.channels.base import OutboundMessage

        config = DiscordConfig(bot_token="test")
        channel = DiscordChannel(config)
        msg = OutboundMessage(
            channel="discord",
            chat_id="123",
            content="hello",
            metadata={"chat_id": "123"},
        )
        result = await channel.send(msg)
        assert result is False


class TestDiscordRetryErrorExtraction:
    """Test Discord-specific status code extraction."""

    def test_discord_status_code_401_not_retryable(self):
        from unittest.mock import MagicMock

        import discord

        ch = DiscordChannel(DiscordConfig(bot_token="test"))
        resp = MagicMock()
        resp.status = 401
        resp.reason = "Unauthorized"
        resp.headers = {}
        exc = discord.HTTPException(resp, "401 Unauthorized")
        assert ch._extract_status_code(exc) == 401
        assert ch._extract_retry_after(exc) is None

    def test_discord_status_code_403_not_retryable(self):
        from unittest.mock import MagicMock

        import discord

        ch = DiscordChannel(DiscordConfig(bot_token="test"))
        resp = MagicMock()
        resp.status = 403
        resp.reason = "Forbidden"
        resp.headers = {}
        exc = discord.HTTPException(resp, "50001 Missing Access")
        assert ch._extract_status_code(exc) == 403
        assert ch._extract_retry_after(exc) is None

    def test_discord_status_code_500_is_retryable(self):
        from unittest.mock import MagicMock

        import discord

        ch = DiscordChannel(DiscordConfig(bot_token="test"))
        resp = MagicMock()
        resp.status = 500
        resp.reason = "Internal Server Error"
        resp.headers = {}
        exc = discord.HTTPException(resp, "500 Internal Server Error")
        assert ch._extract_status_code(exc) == 500
        assert ch._extract_retry_after(exc) == 1.0

    def test_discord_fallback_to_httpx(self):
        import httpx

        ch = DiscordChannel(DiscordConfig(bot_token="test"))
        exc = httpx.HTTPStatusError(
            "unauthorized",
            request=httpx.Request("POST", "https://example.invalid"),
            response=httpx.Response(401),
        )
        assert ch._extract_status_code(exc) == 401
        assert ch._extract_retry_after(exc) is None
