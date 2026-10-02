"""Tests for Slack channel implementation."""

import importlib.util

import pytest

from EvoScientist.channels.base import ChannelError
from EvoScientist.channels.slack.channel import SlackChannel, SlackConfig


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

    async def test_start_raises_without_bot_token(self):
        config = SlackConfig(bot_token="", app_token="xapp-test")
        channel = SlackChannel(config)
        with pytest.raises(ChannelError, match="bot token"):
            await channel.start()

    async def test_start_raises_without_app_token(self):
        config = SlackConfig(bot_token="xoxb-test", app_token="")
        channel = SlackChannel(config)
        with pytest.raises(ChannelError, match="app token"):
            await channel.start()

    async def test_stop_when_not_running(self):
        config = SlackConfig(bot_token="xoxb-test", app_token="xapp-test")
        channel = SlackChannel(config)
        await channel.stop()

    async def test_send_returns_false_without_client(self):
        from EvoScientist.channels.base import OutboundMessage

        config = SlackConfig(bot_token="xoxb-test", app_token="xapp-test")
        channel = SlackChannel(config)
        msg = OutboundMessage(
            channel="slack",
            chat_id="C123",
            content="hello",
            metadata={"chat_id": "C123"},
        )
        result = await channel.send(msg)
        assert result is False


class TestSlackChannelRegistration:
    def test_slack_registered(self):
        from EvoScientist.channels.channel_manager import available_channels

        channels = available_channels()
        assert "slack" in channels


@pytest.mark.skipif(
    importlib.util.find_spec("slack_sdk") is None,
    reason="slack-sdk not installed",
)
class TestSlackRetryErrorExtraction:
    """Test Slack-specific status code and SDK error code extraction."""

    def test_extract_slack_auth_error_not_retryable(self):
        from slack_sdk.errors import SlackApiError
        from slack_sdk.web.slack_response import SlackResponse

        ch = SlackChannel(SlackConfig(bot_token="xoxb-test", app_token="xapp-test"))
        resp = SlackResponse(
            client=None,
            http_verb="POST",
            api_url="https://slack.com/api/chat.postMessage",
            req_args={},
            data={"ok": False, "error": "invalid_auth"},
            headers={},
            status_code=200,
        )
        exc = SlackApiError("The request to the Slack API failed.", response=resp)
        assert ch._extract_sdk_error_code(exc) == "invalid_auth"
        assert ch._extract_retry_after(exc) is None

    def test_extract_slack_token_expired_not_retryable(self):
        from slack_sdk.errors import SlackApiError
        from slack_sdk.web.slack_response import SlackResponse

        ch = SlackChannel(SlackConfig(bot_token="xoxb-test", app_token="xapp-test"))
        resp = SlackResponse(
            client=None,
            http_verb="POST",
            api_url="https://slack.com/api/chat.postMessage",
            req_args={},
            data={"ok": False, "error": "token_expired"},
            headers={},
            status_code=200,
        )
        exc = SlackApiError("The token has expired.", response=resp)
        assert ch._extract_sdk_error_code(exc) == "token_expired"
        assert ch._extract_retry_after(exc) is None

    def test_extract_slack_status_code_401_not_retryable(self):
        from slack_sdk.errors import SlackApiError
        from slack_sdk.web.slack_response import SlackResponse

        ch = SlackChannel(SlackConfig(bot_token="xoxb-test", app_token="xapp-test"))
        resp = SlackResponse(
            client=None,
            http_verb="POST",
            api_url="https://slack.com/api/chat.postMessage",
            req_args={},
            data={"ok": False, "error": "unknown_custom"},
            headers={},
            status_code=401,
        )
        exc = SlackApiError("Unauthorized", response=resp)
        assert ch._extract_status_code(exc) == 401
        assert ch._extract_retry_after(exc) is None

    def test_extract_slack_status_code_500_is_retryable(self):
        from slack_sdk.errors import SlackApiError
        from slack_sdk.web.slack_response import SlackResponse

        ch = SlackChannel(SlackConfig(bot_token="xoxb-test", app_token="xapp-test"))
        resp = SlackResponse(
            client=None,
            http_verb="POST",
            api_url="https://slack.com/api/chat.postMessage",
            req_args={},
            data={"ok": False, "error": "internal_error"},
            headers={},
            status_code=500,
        )
        exc = SlackApiError("Internal Server Error", response=resp)
        assert ch._extract_status_code(exc) == 500
        assert ch._extract_retry_after(exc) == 1.0

    def test_slack_channel_fallback_to_httpx(self):
        import httpx

        ch = SlackChannel(SlackConfig(bot_token="xoxb-test", app_token="xapp-test"))
        exc = httpx.HTTPStatusError(
            "unauthorized",
            request=httpx.Request("POST", "https://example.invalid"),
            response=httpx.Response(401),
        )
        assert ch._extract_status_code(exc) == 401
        assert ch._extract_retry_after(exc) is None

    def test_slack_ratelimited_uses_retry_after_header(self):
        from slack_sdk.errors import SlackApiError
        from slack_sdk.web.slack_response import SlackResponse

        ch = SlackChannel(SlackConfig(bot_token="xoxb-test", app_token="xapp-test"))
        resp = SlackResponse(
            client=None,
            http_verb="POST",
            api_url="https://slack.com/api/chat.postMessage",
            req_args={},
            data={"ok": False, "error": "ratelimited"},
            headers={"Retry-After": "30"},
            status_code=429,
        )
        exc = SlackApiError("ratelimited", response=resp)
        assert ch._extract_retry_delay(exc) == 30.0
        assert ch._extract_retry_after(exc) == 30.0

    def test_slack_malformed_retry_after_falls_through(self):
        from slack_sdk.errors import SlackApiError
        from slack_sdk.web.slack_response import SlackResponse

        ch = SlackChannel(SlackConfig(bot_token="xoxb-test", app_token="xapp-test"))
        resp = SlackResponse(
            client=None,
            http_verb="POST",
            api_url="https://slack.com/api/chat.postMessage",
            req_args={},
            data={"ok": False, "error": "ratelimited"},
            headers={"Retry-After": "soon"},
            status_code=429,
        )
        exc = SlackApiError("ratelimited", response=resp)
        assert ch._extract_retry_delay(exc) is None
        assert ch._extract_retry_after(exc) == ch._rate_limit_delay


@pytest.mark.skipif(
    importlib.util.find_spec("slack_sdk") is None
    or importlib.util.find_spec("aiohttp") is None,
    reason="slack_sdk or aiohttp not installed",
)
class TestSlackRetryWithRawClientResponse:
    """slack_sdk wraps the raw aiohttp response in SlackApiError when a
    JSON-declared body fails to parse; the retry path must survive that."""

    async def test_malformed_json_body_is_retried_and_surfaces_sdk_error(
        self, monkeypatch
    ):
        import aiohttp
        from aiohttp import web
        from slack_sdk.errors import SlackApiError
        from slack_sdk.web.async_client import AsyncWebClient

        from EvoScientist.channels.retry import RetryConfig

        for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            monkeypatch.delenv(var, raising=False)

        calls = 0

        async def handler(request):
            nonlocal calls
            calls += 1
            return web.Response(
                status=200, text="<<not json>>", content_type="application/json"
            )

        app = web.Application()
        app.router.add_post("/api/chat.postMessage", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        try:
            await web.TCPSite(runner, "127.0.0.1", 0).start()
            port = runner.addresses[0][1]
            client = AsyncWebClient(
                token="xoxb-test",
                base_url=f"http://127.0.0.1:{port}/api/",
                retry_handlers=[],
            )
            ch = SlackChannel(SlackConfig(bot_token="xoxb-test", app_token="xapp-test"))
            ch._retry_config = RetryConfig(
                attempts=3, min_delay_s=0.01, max_delay_s=0.02, jitter=0.0
            )
            with pytest.raises(SlackApiError) as excinfo:
                await ch._send_with_retry(
                    lambda: client.chat_postMessage(channel="C1", text="hi")
                )
            assert isinstance(excinfo.value.response, aiohttp.ClientResponse)
            assert calls == 3
        finally:
            await runner.cleanup()
