"""Tests for Feishu channel implementation."""

import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from EvoScientist.channels.base import ChannelError, OutboundMessage
from EvoScientist.channels.feishu.channel import (
    FeishuChannel,
    FeishuConfig,
    _markdown_to_feishu_post,
    _parse_inline_elements,
    _parse_inline_text,
)
from tests.conftest import run_async as _run


class TestFeishuConfig:
    def test_default_values(self):
        config = FeishuConfig()
        assert config.app_id == ""
        assert config.app_secret == ""
        assert config.verification_token == ""
        assert config.encrypt_key == ""
        assert config.webhook_port == 9000
        assert config.text_chunk_limit == 4096
        assert config.feishu_domain == "https://open.feishu.cn"
        assert config.allowed_senders is None
        assert config.subscription_mode == "webhook"

    def test_custom_values(self):
        config = FeishuConfig(
            app_id="test-id",
            app_secret="test-secret",
            verification_token="token123",
            encrypt_key="key123",
            webhook_port=8080,
            allowed_senders={"user1"},
            feishu_domain="https://open.larksuite.com",
        )
        assert config.app_id == "test-id"
        assert config.app_secret == "test-secret"
        assert config.verification_token == "token123"
        assert config.encrypt_key == "key123"
        assert config.webhook_port == 8080
        assert config.allowed_senders == {"user1"}
        assert config.feishu_domain == "https://open.larksuite.com"


class TestFeishuChannel:
    def test_init(self):
        config = FeishuConfig(app_id="test-id", app_secret="test-secret")
        channel = FeishuChannel(config)
        assert channel.config is config
        assert channel._running is False
        assert channel.name == "feishu"

    def test_start_raises_without_app_id(self):
        config = FeishuConfig(app_id="", app_secret="test-secret")
        channel = FeishuChannel(config)
        with pytest.raises(ChannelError, match="app_id"):
            _run(channel.start())

    def test_start_raises_without_app_secret(self):
        config = FeishuConfig(app_id="test-id", app_secret="")
        channel = FeishuChannel(config)
        with pytest.raises(ChannelError, match="app_secret"):
            _run(channel.start())

    def test_stop_when_not_running(self):
        config = FeishuConfig(app_id="test-id", app_secret="test-secret")
        channel = FeishuChannel(config)
        _run(channel.stop())

    def test_send_returns_false_without_client(self):
        config = FeishuConfig(app_id="test-id", app_secret="test-secret")
        channel = FeishuChannel(config)
        msg = OutboundMessage(
            channel="feishu",
            chat_id="oc_test",
            content="hello",
            metadata={"chat_id": "oc_test"},
        )
        result = _run(channel.send(msg))
        assert result is False

    def test_capabilities(self):
        from EvoScientist.channels.capabilities import FEISHU

        config = FeishuConfig()
        channel = FeishuChannel(config)
        assert channel.capabilities is FEISHU
        assert channel.capabilities.format_type == "markdown"
        assert channel.capabilities.groups is True
        assert channel.capabilities.mentions is True
        assert channel.capabilities.media_send is True
        assert channel.capabilities.media_receive is True
        assert channel.capabilities.reactions is True
        assert channel.capabilities.voice is True
        assert channel.capabilities.stickers is True

    def test_extract_post_text(self):
        content = {
            "zh_cn": {
                "title": "Test Title",
                "content": [
                    [
                        {"tag": "text", "text": "Hello "},
                        {"tag": "a", "text": "world", "href": "http://example.com"},
                    ],
                    [{"tag": "text", "text": "Second line"}],
                ],
            }
        }
        result = FeishuChannel._extract_post_text(content)
        assert "Test Title" in result
        assert "Hello world" in result
        assert "Second line" in result

    def test_extract_post_text_empty(self):
        result = FeishuChannel._extract_post_text({})
        assert result == ""

    def test_extract_post_text_skips_at_mentions(self):
        content = {
            "zh_cn": {
                "content": [
                    [{"tag": "text", "text": "Hi "}, {"tag": "at", "user_id": "bot"}],
                ],
            }
        }
        result = FeishuChannel._extract_post_text(content)
        assert result == "Hi"

    def test_strip_mention(self):
        config = FeishuConfig()
        channel = FeishuChannel(config)
        channel._mention_names = ["@_user_1"]
        result = channel._strip_mention("@_user_1 hello world")
        assert result == "hello world"

    def test_strip_mention_multiple(self):
        config = FeishuConfig()
        channel = FeishuChannel(config)
        channel._mention_names = ["@_user_1", "@_user_2"]
        result = channel._strip_mention("@_user_1 @_user_2 hello")
        assert result == "hello"

    def test_strip_mention_no_match(self):
        config = FeishuConfig()
        channel = FeishuChannel(config)
        channel._mention_names = []
        result = channel._strip_mention("hello world")
        assert result == "hello world"


class TestFeishuErrorPatterns:
    """Test non-retryable and rate-limit pattern detection."""

    def test_non_retryable_patterns_defined(self):
        config = FeishuConfig()
        channel = FeishuChannel(config)
        assert len(channel._non_retryable_patterns) > 0
        assert "10003" in channel._non_retryable_patterns
        assert "99991401" in channel._non_retryable_patterns

    def test_non_retryable_returns_none(self):
        config = FeishuConfig()
        channel = FeishuChannel(config)
        exc = Exception("error code 10003: invalid app_id")
        result = channel._extract_retry_after(exc)
        assert result is None

    def test_rate_limit_patterns_defined(self):
        config = FeishuConfig()
        channel = FeishuChannel(config)
        assert "99991400" in channel._rate_limit_patterns
        assert "频率限制" in channel._rate_limit_patterns

    def test_rate_limit_returns_delay(self):
        config = FeishuConfig()
        channel = FeishuChannel(config)
        exc = Exception("99991400 频率限制")
        result = channel._extract_retry_after(exc)
        assert result is not None
        assert result == channel._rate_limit_delay


class TestFeishuWebhookEvent:
    """Test _on_message parsing with mocked bus."""

    def _make_channel(self):
        config = FeishuConfig(app_id="test-app", app_secret="test-secret")
        channel = FeishuChannel(config)
        channel._running = True
        channel._http_client = MagicMock()
        channel._access_token = "fake-token"
        channel._token_expires = 9999999999
        channel._enqueue_raw = AsyncMock()
        return channel

    def test_text_message_v2(self):
        channel = self._make_channel()
        event = {
            "sender": {
                "sender_id": {"open_id": "ou_test123"},
                "sender_type": "user",
            },
            "message": {
                "chat_id": "oc_chat1",
                "message_type": "text",
                "message_id": "msg_1",
                "chat_type": "p2p",
                "content": json.dumps({"text": "hello feishu"}),
                "create_time": "1700000000000",
            },
        }
        _run(channel._on_message(event))
        channel._enqueue_raw.assert_called_once()
        raw = channel._enqueue_raw.call_args[0][0]
        assert raw.text == "hello feishu"
        assert raw.sender_id == "ou_test123"
        assert raw.chat_id == "oc_chat1"
        assert raw.is_group is False

    def test_group_message_with_mention(self):
        channel = self._make_channel()
        event = {
            "sender": {
                "sender_id": {"open_id": "ou_sender"},
                "sender_type": "user",
            },
            "message": {
                "chat_id": "oc_group1",
                "message_type": "text",
                "message_id": "msg_2",
                "chat_type": "group",
                "content": json.dumps({"text": "@_user_1 do something"}),
                "create_time": "1700000000000",
                "mentions": [{"key": "@_user_1", "id": {}}],
            },
        }
        _run(channel._on_message(event))
        raw = channel._enqueue_raw.call_args[0][0]
        assert raw.is_group is True
        assert raw.was_mentioned is True
        assert channel._mention_names == ["@_user_1"]

    def test_group_message_no_mention(self):
        channel = self._make_channel()
        event = {
            "sender": {
                "sender_id": {"open_id": "ou_sender"},
                "sender_type": "user",
            },
            "message": {
                "chat_id": "oc_group2",
                "message_type": "text",
                "message_id": "msg_3",
                "chat_type": "group",
                "content": json.dumps({"text": "just talking"}),
                "create_time": "1700000000000",
            },
        }
        _run(channel._on_message(event))
        raw = channel._enqueue_raw.call_args[0][0]
        assert raw.is_group is True
        assert raw.was_mentioned is False

    def test_skips_bot_messages(self):
        channel = self._make_channel()
        event = {
            "sender": {
                "sender_id": {"open_id": "ou_bot"},
                "sender_type": "app",
            },
            "message": {
                "chat_id": "oc_chat",
                "message_type": "text",
                "message_id": "msg_bot",
                "content": json.dumps({"text": "bot reply"}),
            },
        }
        _run(channel._on_message(event))
        channel._enqueue_raw.assert_not_called()

    def test_post_message(self):
        channel = self._make_channel()
        post_content = {
            "zh_cn": {
                "title": "Test",
                "content": [[{"tag": "text", "text": "Post body"}]],
            }
        }
        event = {
            "sender": {
                "sender_id": {"open_id": "ou_test"},
                "sender_type": "user",
            },
            "message": {
                "chat_id": "oc_chat",
                "message_type": "post",
                "message_id": "msg_post",
                "chat_type": "p2p",
                "content": json.dumps(post_content),
                "create_time": "1700000000000",
            },
        }
        _run(channel._on_message(event))
        raw = channel._enqueue_raw.call_args[0][0]
        assert "Test" in raw.text
        assert "Post body" in raw.text

    def test_unsupported_msg_type_annotation(self):
        channel = self._make_channel()
        event = {
            "sender": {
                "sender_id": {"open_id": "ou_test"},
                "sender_type": "user",
            },
            "message": {
                "chat_id": "oc_chat",
                "message_type": "share_chat",
                "message_id": "msg_share",
                "chat_type": "p2p",
                "content": "{}",
                "create_time": "1700000000000",
            },
        }
        _run(channel._on_message(event))
        raw = channel._enqueue_raw.call_args[0][0]
        assert "share_chat" in raw.text


class TestFeishuSendChunk:
    """Test _send_chunk with mocked HTTP client."""

    def test_send_chunk_post_format(self):
        config = FeishuConfig(app_id="test-app", app_secret="test-secret")
        channel = FeishuChannel(config)
        channel._access_token = "fake-token"
        channel._token_expires = 9999999999

        mock_response = MagicMock()
        mock_response.json.return_value = {"code": 0}
        channel._http_client = MagicMock()
        channel._http_client.post = AsyncMock(return_value=mock_response)

        _run(channel._send_chunk("oc_chat1", "formatted", "raw **text**", None, {}))
        channel._http_client.post.assert_called()
        # Should try post format first
        call_args = channel._http_client.post.call_args
        body = call_args.kwargs.get("json") or call_args[1].get("json")
        assert body["receive_id"] == "oc_chat1"

    def test_send_chunk_with_reply(self):
        config = FeishuConfig(app_id="test-app", app_secret="test-secret")
        channel = FeishuChannel(config)
        channel._access_token = "fake-token"
        channel._token_expires = 9999999999

        mock_response = MagicMock()
        mock_response.json.return_value = {"code": 0}
        channel._http_client = MagicMock()
        channel._http_client.post = AsyncMock(return_value=mock_response)

        _run(channel._send_chunk("oc_chat1", "reply", "reply text", "om_reply_id", {}))
        # Should call the reply API endpoint
        first_call_url = channel._http_client.post.call_args_list[0][0][0]
        assert "reply" in first_call_url


class TestFeishuMarkdownConversion:
    def test_empty_text(self):
        assert _markdown_to_feishu_post("") is None
        assert _markdown_to_feishu_post("   ") is None

    def test_plain_text(self):
        result = _markdown_to_feishu_post("Hello world")
        assert result is not None
        assert "zh_cn" in result
        content = result["zh_cn"]["content"]
        assert len(content) >= 1

    def test_code_block(self):
        md = "```python\nprint('hello')\n```"
        result = _markdown_to_feishu_post(md)
        assert result is not None
        content = result["zh_cn"]["content"]
        found = False
        for para in content:
            for elem in para:
                if elem.get("tag") == "code_block":
                    found = True
                    assert elem["language"] == "python"
                    assert "print" in elem["text"]
        assert found

    def test_code_block_no_language(self):
        md = "```\nsome code\n```"
        result = _markdown_to_feishu_post(md)
        assert result is not None
        content = result["zh_cn"]["content"]
        for para in content:
            for elem in para:
                if elem.get("tag") == "code_block":
                    assert elem["language"] == "plain"

    def test_bold_text(self):
        elements = _parse_inline_text("**bold text**")
        assert any(
            e.get("style") == ["bold"] and e["text"] == "bold text" for e in elements
        )

    def test_inline_code(self):
        elements = _parse_inline_text("`code`")
        assert any(
            "code_block" in (e.get("style") or []) and e["text"] == "code"
            for e in elements
        )

    def test_link(self):
        elements = _parse_inline_text("[click](http://example.com)")
        assert any(e.get("tag") == "a" and e["text"] == "click" for e in elements)

    def test_strikethrough(self):
        elements = _parse_inline_text("~~deleted~~")
        assert any(
            e.get("style") == ["strikethrough"] and e["text"] == "deleted"
            for e in elements
        )

    def test_italic(self):
        elements = _parse_inline_text("_italic text_")
        assert any(
            e.get("style") == ["italic"] and e["text"] == "italic text"
            for e in elements
        )

    def test_heading(self):
        elements = _parse_inline_elements("## My Heading")
        assert any(
            e.get("style") == ["bold"] and e["text"] == "My Heading" for e in elements
        )

    def test_blockquote(self):
        elements = _parse_inline_elements("> quoted text")
        # Should have ▎ prefix with italic style
        assert any(e.get("text") == "▎" for e in elements)
        assert any(e.get("text") == "quoted text" for e in elements)

    def test_unordered_list(self):
        elements = _parse_inline_elements("- list item")
        assert any(e.get("text") == "• " for e in elements)
        assert any(e.get("text") == "list item" for e in elements)

    def test_ordered_list(self):
        elements = _parse_inline_elements("3. third item")
        assert any(e.get("text") == "3. " for e in elements)
        assert any(e.get("text") == "third item" for e in elements)

    def test_multi_paragraph(self):
        md = "First paragraph\n\nSecond paragraph"
        result = _markdown_to_feishu_post(md)
        content = result["zh_cn"]["content"]
        assert len(content) == 2

    def test_mixed_content(self):
        md = "# Title\n\nSome **bold** text\n\n```python\ncode\n```"
        result = _markdown_to_feishu_post(md)
        assert result is not None
        content = result["zh_cn"]["content"]
        assert len(content) >= 3


class TestFeishuChannelRegistration:
    def test_feishu_registered(self):
        from EvoScientist.channels.channel_manager import available_channels

        channels = available_channels()
        assert "feishu" in channels


class TestFeishuProbe:
    def test_missing_app_id(self):
        from EvoScientist.channels.feishu.probe import validate_feishu_credentials

        ok, msg = _run(validate_feishu_credentials("", "secret"))
        assert ok is False
        assert "app_id" in msg

    def test_missing_app_secret(self):
        from EvoScientist.channels.feishu.probe import validate_feishu_credentials

        ok, msg = _run(validate_feishu_credentials("id", ""))
        assert ok is False
        assert "app_secret" in msg


class TestFeishuWebSocketMode:
    """Tests for WebSocket (长连接) subscription mode."""

    def test_config_subscription_mode_websocket(self):
        config = FeishuConfig(
            app_id="test-id",
            app_secret="test-secret",
            subscription_mode="websocket",
        )
        assert config.subscription_mode == "websocket"

    def test_start_websocket_raises_without_lark_oapi(self):
        config = FeishuConfig(
            app_id="test-id",
            app_secret="test-secret",
            subscription_mode="websocket",
        )
        channel = FeishuChannel(config)
        # Temporarily hide lark_oapi if it's installed
        with patch.dict(sys.modules, {"lark_oapi": None}):
            with pytest.raises(ChannelError, match="lark-oapi"):
                _run(channel.start())

    def test_start_webhook_mode_still_works(self):
        """Ensure subscription_mode='webhook' still validates as before."""
        config = FeishuConfig(
            app_id="",
            app_secret="test-secret",
            subscription_mode="webhook",
        )
        channel = FeishuChannel(config)
        with pytest.raises(ChannelError, match="app_id"):
            _run(channel.start())

    def test_invalid_subscription_mode_raises(self):
        config = FeishuConfig(
            app_id="test-id",
            app_secret="test-secret",
            subscription_mode="websockeet",
        )
        channel = FeishuChannel(config)
        with pytest.raises(ChannelError, match="Invalid feishu_subscription_mode"):
            _run(channel.start())

    def test_on_lark_sdk_message_bridges_to_on_message(self):
        """Test that _on_lark_sdk_message enqueues event dict via queue."""
        import queue as queue_mod

        config = FeishuConfig(app_id="test-app", app_secret="test-secret")
        channel = FeishuChannel(config)
        channel._running = True
        channel._http_client = MagicMock()
        channel._access_token = "fake-token"
        channel._token_expires = 9999999999
        channel._enqueue_raw = AsyncMock()
        channel._ws_event_queue = queue_mod.Queue()

        # Build a mock SDK event object matching lark_oapi structure
        mock_sender_id = MagicMock()
        mock_sender_id.open_id = "ou_test_ws"
        mock_sender_id.user_id = "user_ws"

        mock_sender = MagicMock()
        mock_sender.sender_id = mock_sender_id
        mock_sender.sender_type = "user"

        mock_msg = MagicMock()
        mock_msg.chat_id = "oc_ws_chat"
        mock_msg.message_type = "text"
        mock_msg.message_id = "msg_ws_1"
        mock_msg.chat_type = "p2p"
        mock_msg.content = json.dumps({"text": "hello from websocket"})
        mock_msg.create_time = "1700000000000"
        mock_msg.mentions = None

        mock_event = MagicMock()
        mock_event.message = mock_msg
        mock_event.sender = mock_sender

        mock_data = MagicMock()
        mock_data.event = mock_event

        # Call the SDK callback (sync, puts on queue)
        channel._on_lark_sdk_message(mock_data)

        # Verify event was queued
        assert not channel._ws_event_queue.empty()
        event_dict = channel._ws_event_queue.get_nowait()
        assert event_dict["sender"]["sender_id"]["open_id"] == "ou_test_ws"
        assert event_dict["message"]["chat_id"] == "oc_ws_chat"
        assert event_dict["message"]["content"] == json.dumps(
            {"text": "hello from websocket"}
        )

        # Verify the consumer processes it correctly
        _run(channel._on_message(event_dict))
        channel._enqueue_raw.assert_called_once()
        raw = channel._enqueue_raw.call_args[0][0]
        assert raw.text == "hello from websocket"
        assert raw.sender_id == "ou_test_ws"
        assert raw.is_group is False

    def test_cleanup_websocket_mode(self):
        config = FeishuConfig(
            app_id="test-id",
            app_secret="test-secret",
            subscription_mode="websocket",
        )
        channel = FeishuChannel(config)
        mock_client = MagicMock()
        mock_client.aclose = AsyncMock()
        channel._http_client = mock_client
        channel._lark_ws_thread = MagicMock()
        channel._main_loop = MagicMock()
        channel._ws_event_queue = MagicMock()
        channel._ws_consumer_task = None
        channel._access_token = "fake-token"

        _run(channel._cleanup())

        mock_client.aclose.assert_called_once()
        assert channel._http_client is None
        assert channel._lark_ws_thread is None
        assert channel._main_loop is None
        assert channel._ws_event_queue is None
        assert channel._access_token is None
