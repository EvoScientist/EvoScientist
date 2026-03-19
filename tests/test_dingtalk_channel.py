"""Tests for DingTalk channel implementation."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from EvoScientist.channels.base import ChannelError, OutboundMessage
from EvoScientist.channels.dingtalk.channel import DingTalkChannel, DingTalkConfig
from tests.conftest import run_async as _run


class TestDingTalkConfig:
    def test_default_values(self):
        config = DingTalkConfig()
        assert config.client_id == ""
        assert config.client_secret == ""
        assert config.allowed_senders is None
        assert config.text_chunk_limit == 4096

    def test_custom_values(self):
        config = DingTalkConfig(
            client_id="test-id",
            client_secret="test-secret",
            allowed_senders={"user1"},
            text_chunk_limit=2000,
            proxy="http://proxy:8080",
        )
        assert config.client_id == "test-id"
        assert config.client_secret == "test-secret"
        assert config.allowed_senders == {"user1"}
        assert config.text_chunk_limit == 2000
        assert config.proxy == "http://proxy:8080"


class TestDingTalkChannel:
    def test_init(self):
        config = DingTalkConfig(client_id="test-id", client_secret="test-secret")
        channel = DingTalkChannel(config)
        assert channel.config is config
        assert channel._running is False
        assert channel.name == "dingtalk"

    def test_start_raises_without_credentials(self):
        config = DingTalkConfig(client_id="", client_secret="")
        channel = DingTalkChannel(config)
        with pytest.raises(ChannelError, match="client_id and client_secret"):
            _run(channel.start())

    def test_start_raises_without_client_id(self):
        config = DingTalkConfig(client_id="", client_secret="secret")
        channel = DingTalkChannel(config)
        with pytest.raises(ChannelError, match="client_id and client_secret"):
            _run(channel.start())

    def test_start_raises_without_client_secret(self):
        config = DingTalkConfig(client_id="id", client_secret="")
        channel = DingTalkChannel(config)
        with pytest.raises(ChannelError, match="client_id and client_secret"):
            _run(channel.start())

    def test_stop_when_not_running(self):
        config = DingTalkConfig(client_id="test-id", client_secret="test-secret")
        channel = DingTalkChannel(config)
        _run(channel.stop())

    def test_send_returns_false_without_client(self):
        config = DingTalkConfig(client_id="test-id", client_secret="test-secret")
        channel = DingTalkChannel(config)
        msg = OutboundMessage(
            channel="dingtalk",
            chat_id="user123",
            content="hello",
            metadata={"chat_id": "user123"},
        )
        result = _run(channel.send(msg))
        assert result is False

    def test_capabilities(self):
        from EvoScientist.channels.capabilities import DINGTALK

        config = DingTalkConfig()
        channel = DingTalkChannel(config)
        assert channel.capabilities is DINGTALK
        assert channel.capabilities.format_type == "markdown"
        assert channel.capabilities.groups is True
        assert channel.capabilities.mentions is True
        assert channel.capabilities.media_send is True
        assert channel.capabilities.media_receive is True


class TestDingTalkErrorPatterns:
    """Test non-retryable and rate-limit pattern detection."""

    def test_non_retryable_patterns_defined(self):
        config = DingTalkConfig()
        channel = DingTalkChannel(config)
        assert "invalidauthentication" in channel._non_retryable_patterns
        assert "forbidden" in channel._non_retryable_patterns
        assert "40014" in channel._non_retryable_patterns

    def test_non_retryable_returns_none(self):
        config = DingTalkConfig()
        channel = DingTalkChannel(config)
        exc = Exception("invalidauthentication: bad credentials")
        result = channel._extract_retry_after(exc)
        assert result is None

    def test_rate_limit_returns_delay(self):
        config = DingTalkConfig()
        channel = DingTalkChannel(config)
        # Base class default includes "429" and "ratelimit"
        exc = Exception("HTTP 429 ratelimit exceeded")
        result = channel._extract_retry_after(exc)
        assert result is not None
        assert result > 0


class TestDingTalkWsMessageParsing:
    """Test _on_ws_message parsing logic with mocked bus."""

    def _make_channel(self):
        config = DingTalkConfig(client_id="test-app", client_secret="test-secret")
        channel = DingTalkChannel(config)
        channel._running = True
        channel._ws_session = MagicMock()
        channel._ws_session.send_str = AsyncMock()
        channel._http_client = MagicMock()
        channel._access_token = "fake-token"
        channel._token_expires = 9999999999
        return channel

    def test_system_ping_ack(self):
        channel = self._make_channel()
        data = {
            "type": "SYSTEM",
            "headers": {"topic": "ping", "messageId": "ping-1"},
            "data": "pong-data",
        }
        _run(channel._on_ws_message(data))
        channel._ws_session.send_str.assert_called_once()
        sent = json.loads(channel._ws_session.send_str.call_args[0][0])
        assert sent["code"] == 200
        assert sent["data"] == "pong-data"

    def test_callback_text_message(self):
        channel = self._make_channel()
        channel._enqueue_raw = AsyncMock()

        payload = {
            "text": {"content": "hello bot"},
            "senderStaffId": "staff123",
            "conversationType": "1",
            "createAt": "1700000000000",
        }
        data = {
            "type": "CALLBACK",
            "headers": {"messageId": "msg-1", "contentType": "application/json"},
            "data": json.dumps(payload),
        }
        _run(channel._on_ws_message(data))
        channel._enqueue_raw.assert_called_once()
        raw = channel._enqueue_raw.call_args[0][0]
        assert raw.text == "hello bot"
        assert raw.sender_id == "staff123"
        assert raw.is_group is False

    def test_callback_group_message_mention(self):
        channel = self._make_channel()
        channel._enqueue_raw = AsyncMock()

        payload = {
            "text": {"content": "@bot hello"},
            "senderStaffId": "staff456",
            "conversationType": "2",
            "isInAtList": True,
            "createAt": "1700000000000",
        }
        data = {
            "type": "CALLBACK",
            "headers": {"messageId": "msg-2"},
            "data": json.dumps(payload),
        }
        _run(channel._on_ws_message(data))
        raw = channel._enqueue_raw.call_args[0][0]
        assert raw.is_group is True
        assert raw.was_mentioned is True

    def test_callback_group_no_mention(self):
        channel = self._make_channel()
        channel._enqueue_raw = AsyncMock()

        payload = {
            "text": {"content": "just chatting"},
            "senderStaffId": "staff789",
            "conversationType": "2",
            "createAt": "1700000000000",
        }
        data = {
            "type": "CALLBACK",
            "headers": {"messageId": "msg-3"},
            "data": json.dumps(payload),
        }
        _run(channel._on_ws_message(data))
        raw = channel._enqueue_raw.call_args[0][0]
        assert raw.is_group is True
        assert raw.was_mentioned is False

    def test_ignores_non_callback(self):
        channel = self._make_channel()
        channel._enqueue_raw = AsyncMock()

        data = {
            "type": "EVENT",
            "headers": {"messageId": "msg-x"},
            "data": "{}",
        }
        _run(channel._on_ws_message(data))
        channel._enqueue_raw.assert_not_called()

    def test_ignores_empty_content(self):
        channel = self._make_channel()
        channel._enqueue_raw = AsyncMock()

        payload = {
            "text": {"content": ""},
            "senderStaffId": "staff0",
            "conversationType": "1",
        }
        data = {
            "type": "CALLBACK",
            "headers": {"messageId": "msg-e"},
            "data": json.dumps(payload),
        }
        _run(channel._on_ws_message(data))
        channel._enqueue_raw.assert_not_called()

    def test_non_dict_data_ignored(self):
        channel = self._make_channel()
        channel._enqueue_raw = AsyncMock()
        _run(channel._on_ws_message("not a dict"))
        channel._enqueue_raw.assert_not_called()


class TestDingTalkSendChunk:
    """Test _send_chunk with mocked HTTP client."""

    def test_send_chunk_calls_api(self):
        config = DingTalkConfig(client_id="test-app", client_secret="test-secret")
        channel = DingTalkChannel(config)
        channel._access_token = "fake-token"
        channel._token_expires = 9999999999

        mock_response = MagicMock()
        mock_response.json.return_value = {"processQueryKey": "ok"}
        channel._http_client = MagicMock()
        channel._http_client.post = AsyncMock(return_value=mock_response)

        _run(channel._send_chunk("user1", "formatted", "raw text", None, {}))
        channel._http_client.post.assert_called_once()
        call_args = channel._http_client.post.call_args
        body = call_args.kwargs.get("json") or call_args[1].get("json")
        assert body["robotCode"] == "test-app"
        assert body["userIds"] == ["user1"]


class TestDingTalkChannelRegistration:
    def test_dingtalk_registered(self):
        from EvoScientist.channels.channel_manager import available_channels

        channels = available_channels()
        assert "dingtalk" in channels


class TestDingTalkProbe:
    def test_missing_credentials(self):
        from EvoScientist.channels.dingtalk.probe import validate_dingtalk

        ok, msg = _run(validate_dingtalk("", ""))
        assert ok is False
        assert "required" in msg

    def test_missing_client_id(self):
        from EvoScientist.channels.dingtalk.probe import validate_dingtalk

        ok, _msg = _run(validate_dingtalk("", "secret"))
        assert ok is False

    def test_missing_client_secret(self):
        from EvoScientist.channels.dingtalk.probe import validate_dingtalk

        ok, _msg = _run(validate_dingtalk("id", ""))
        assert ok is False
