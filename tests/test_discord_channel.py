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
    """Test Discord-specific status code and error code extraction."""

    class HTTPException(Exception):
        """Mock discord.HTTPException for test environment."""

        def __init__(self, message: str, status: int, code: int | None = None) -> None:
            super().__init__(message)
            self.status = status
            self.code = code

    def test_discord_status_code_401_not_retryable(self):
        ch = DiscordChannel(DiscordConfig(bot_token="test"))
        exc = self.HTTPException("401 Unauthorized", status=401, code=0)
        assert ch._extract_status_code(exc) == 401
        assert ch._extract_retry_after(exc) is None

    def test_discord_status_code_403_not_retryable(self):
        ch = DiscordChannel(DiscordConfig(bot_token="test"))
        exc = self.HTTPException("50001 Missing Access", status=403, code=50001)
        assert ch._extract_status_code(exc) == 403
        assert ch._extract_sdk_error_code(exc) == "50001"
        assert ch._extract_retry_after(exc) is None

    def test_discord_status_code_500_is_retryable(self):
        ch = DiscordChannel(DiscordConfig(bot_token="test"))
        exc = self.HTTPException("500 Internal Server Error", status=500, code=0)
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
