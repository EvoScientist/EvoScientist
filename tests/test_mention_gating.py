"""Tests for unified mention gating in base.py and per-channel _strip_mention."""

import asyncio
from dataclasses import dataclass

import pytest

from EvoScientist.channels.base import Channel, RawIncoming


# ── Minimal concrete channel for testing base-class logic ─────────────


@dataclass
class _StubConfig:
    allowed_senders: set[str] | None = None
    require_mention: str = "group"


class _StubChannel(Channel):
    name = "stub"

    async def start(self):
        pass

    async def _send_chunk(self, chat_id, formatted_text, raw_text, reply_to, metadata):
        pass


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── _should_process tests ────────────────────────────────────────────


class TestShouldProcess:
    """Tests for the centralized _should_process gate."""

    def _make(self, require_mention="group"):
        config = _StubConfig(require_mention=require_mention)
        return _StubChannel(config)

    def test_dm_always_passes_with_group_mode(self):
        ch = self._make("group")
        raw = RawIncoming(sender_id="u1", chat_id="c1", is_group=False, was_mentioned=False)
        assert ch._should_process(raw) is True

    def test_dm_always_passes_with_always_mode(self):
        ch = self._make("always")
        raw = RawIncoming(sender_id="u1", chat_id="c1", is_group=False, was_mentioned=False)
        assert ch._should_process(raw) is True

    def test_dm_always_passes_with_off_mode(self):
        ch = self._make("off")
        raw = RawIncoming(sender_id="u1", chat_id="c1", is_group=False, was_mentioned=False)
        assert ch._should_process(raw) is True

    def test_group_mentioned_passes_with_group_mode(self):
        ch = self._make("group")
        raw = RawIncoming(sender_id="u1", chat_id="c1", is_group=True, was_mentioned=True)
        assert ch._should_process(raw) is True

    def test_group_not_mentioned_blocked_with_group_mode(self):
        ch = self._make("group")
        raw = RawIncoming(sender_id="u1", chat_id="c1", is_group=True, was_mentioned=False)
        assert ch._should_process(raw) is False

    def test_group_not_mentioned_passes_with_off_mode(self):
        ch = self._make("off")
        raw = RawIncoming(sender_id="u1", chat_id="c1", is_group=True, was_mentioned=False)
        assert ch._should_process(raw) is True

    def test_group_not_mentioned_blocked_with_always_mode(self):
        ch = self._make("always")
        raw = RawIncoming(sender_id="u1", chat_id="c1", is_group=True, was_mentioned=False)
        assert ch._should_process(raw) is False

    def test_group_mentioned_passes_with_always_mode(self):
        ch = self._make("always")
        raw = RawIncoming(sender_id="u1", chat_id="c1", is_group=True, was_mentioned=True)
        assert ch._should_process(raw) is True


class TestBuildInboundGating:
    """Tests that _build_inbound integrates _should_process and _strip_mention."""

    def test_group_not_mentioned_returns_none(self):
        config = _StubConfig(require_mention="group")
        ch = _StubChannel(config)
        raw = RawIncoming(
            sender_id="u1", chat_id="c1", text="hello",
            is_group=True, was_mentioned=False,
        )
        assert ch._build_inbound(raw) is None

    def test_group_mentioned_returns_message(self):
        config = _StubConfig(require_mention="group")
        ch = _StubChannel(config)
        raw = RawIncoming(
            sender_id="u1", chat_id="c1", text="hello",
            is_group=True, was_mentioned=True,
        )
        msg = ch._build_inbound(raw)
        assert msg is not None
        assert msg.content == "hello"

    def test_dm_returns_message_even_when_not_mentioned(self):
        config = _StubConfig(require_mention="group")
        ch = _StubChannel(config)
        raw = RawIncoming(
            sender_id="u1", chat_id="c1", text="hello",
            is_group=False, was_mentioned=False,
        )
        msg = ch._build_inbound(raw)
        assert msg is not None

    def test_strip_mention_called_for_group(self):
        """When is_group=True, _strip_mention should be applied to text."""
        config = _StubConfig(require_mention="group")
        ch = _StubChannel(config)

        # Override _strip_mention to verify it's called
        ch._strip_mention = lambda text: text.replace("@bot ", "").strip()

        raw = RawIncoming(
            sender_id="u1", chat_id="c1", text="@bot hello",
            is_group=True, was_mentioned=True,
        )
        msg = ch._build_inbound(raw)
        assert msg is not None
        assert msg.content == "hello"

    def test_strip_mention_not_called_for_dm(self):
        """When is_group=False, _strip_mention should NOT be applied."""
        config = _StubConfig(require_mention="group")
        ch = _StubChannel(config)

        ch._strip_mention = lambda text: text.replace("@bot ", "").strip()

        raw = RawIncoming(
            sender_id="u1", chat_id="c1", text="@bot hello",
            is_group=False, was_mentioned=True,
        )
        msg = ch._build_inbound(raw)
        assert msg is not None
        assert msg.content == "@bot hello"


# ── Per-channel _strip_mention tests ─────────────────────────────────


class TestTelegramStripMention:
    def test_strip_username(self):
        from EvoScientist.channels.telegram.channel import TelegramChannel, TelegramConfig

        config = TelegramConfig(bot_token="test")
        ch = TelegramChannel(config)
        ch._bot_username = "mybot"
        assert ch._strip_mention("@mybot hello") == "hello"
        assert ch._strip_mention("@MyBot hello") == "hello"
        assert ch._strip_mention("hello @mybot world") == "hello world"

    def test_no_username(self):
        from EvoScientist.channels.telegram.channel import TelegramChannel, TelegramConfig

        config = TelegramConfig(bot_token="test")
        ch = TelegramChannel(config)
        ch._bot_username = ""
        assert ch._strip_mention("@mybot hello") == "@mybot hello"


class TestDiscordStripMention:
    def test_strip_user_id(self):
        from EvoScientist.channels.discord.channel import DiscordChannel, DiscordConfig

        config = DiscordConfig(bot_token="test")
        ch = DiscordChannel(config)

        # Mock client.user
        class FakeUser:
            id = 123456789
        class FakeClient:
            user = FakeUser()
        ch._client = FakeClient()

        assert ch._strip_mention("<@123456789> hello") == "hello"
        assert ch._strip_mention("<@!123456789> hello") == "hello"
        assert ch._strip_mention("hello <@123456789> world") == "hello world"

    def test_no_client(self):
        from EvoScientist.channels.discord.channel import DiscordChannel, DiscordConfig

        config = DiscordConfig(bot_token="test")
        ch = DiscordChannel(config)
        ch._client = None
        assert ch._strip_mention("<@123> hello") == "<@123> hello"


class TestSlackStripMention:
    def test_strip_user_id(self):
        from EvoScientist.channels.slack.channel import SlackChannel, SlackConfig

        config = SlackConfig(bot_token="test", app_token="xapp-test")
        ch = SlackChannel(config)
        ch._bot_user_id = "U123ABC"
        assert ch._strip_mention("<@U123ABC> hello") == "hello"
        assert ch._strip_mention("hello <@U123ABC> world") == "hello world"

    def test_no_bot_user_id(self):
        from EvoScientist.channels.slack.channel import SlackChannel, SlackConfig

        config = SlackConfig(bot_token="test", app_token="xapp-test")
        ch = SlackChannel(config)
        assert ch._strip_mention("<@U123> hello") == "<@U123> hello"


# ── iMessage pipeline integration tests ──────────────────────────


class TestIMessageConfig:
    def test_default_values(self):
        from EvoScientist.channels.imessage.channel_rpc import IMessageConfig

        config = IMessageConfig()
        assert config.allowed_senders is None
        assert config.include_attachments is True
        assert config.text_chunk_limit == 4096

    def test_allowed_senders_is_set(self):
        from EvoScientist.channels.imessage.channel_rpc import IMessageConfig

        config = IMessageConfig(allowed_senders={"+1234", "chat_id:5"})
        assert isinstance(config.allowed_senders, set)
        assert "+1234" in config.allowed_senders


class TestIMessageChannel:
    def test_is_ready_without_client(self):
        from EvoScientist.channels.imessage.channel_rpc import IMessageChannelRpc

        ch = IMessageChannelRpc()
        assert ch._is_ready() is False

    def test_is_allowed_always_true(self):
        from EvoScientist.channels.imessage.channel_rpc import IMessageChannelRpc

        ch = IMessageChannelRpc()
        assert ch.is_allowed("anyone") is True

    def test_build_inbound_uses_rich_filtering(self):
        from EvoScientist.channels.imessage.channel_rpc import (
            IMessageChannelRpc, IMessageConfig,
        )

        config = IMessageConfig(allowed_senders={"+15551234567"})
        ch = IMessageChannelRpc(config)

        # Allowed sender passes
        raw = RawIncoming(
            sender_id="+15551234567", chat_id="c1", text="hi",
            metadata={"chat_id": 1, "chat_guid": None},
        )
        assert ch._build_inbound(raw) is not None

        # Disallowed sender blocked
        raw2 = RawIncoming(
            sender_id="+19999999999", chat_id="c1", text="hi",
            metadata={"chat_id": 1, "chat_guid": None},
        )
        assert ch._build_inbound(raw2) is None

    def test_add_remove_allowed_senders(self):
        from EvoScientist.channels.imessage.channel_rpc import IMessageChannelRpc

        ch = IMessageChannelRpc()
        assert ch.config.allowed_senders is None

        ch.add_allowed_sender("+15551234567")
        assert ch.config.allowed_senders is not None
        assert len(ch.config.allowed_senders) == 1

        ch.remove_allowed_sender("+15551234567")
        assert len(ch.config.allowed_senders) == 0

        ch.clear_allowed_senders()
        assert ch.config.allowed_senders is None

    def test_list_allowed_senders(self):
        from EvoScientist.channels.imessage.channel_rpc import (
            IMessageChannelRpc, IMessageConfig,
        )

        ch = IMessageChannelRpc()
        assert ch.list_allowed_senders() == []

        config = IMessageConfig(allowed_senders={"a", "b"})
        ch2 = IMessageChannelRpc(config)
        result = ch2.list_allowed_senders()
        assert set(result) == {"a", "b"}

    def test_handle_message_sets_is_group(self):
        import asyncio
        from EvoScientist.channels.imessage.channel_rpc import IMessageChannelRpc

        ch = IMessageChannelRpc()
        # Capture what _build_inbound receives
        captured = []
        original = ch._build_inbound
        def spy(raw):
            captured.append(raw)
            return original(raw)
        ch._build_inbound = spy

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(ch._handle_message({
                "message": {
                    "sender": "+1234",
                    "text": "hello",
                    "is_group": True,
                    "chat_id": 42,
                    "id": "msg1",
                }
            }))
        finally:
            loop.close()

        assert len(captured) == 1
        assert captured[0].is_group is True
        assert captured[0].was_mentioned is True


# ── Feishu config tests ──────────────────────────────────────────


class TestFeishuConfig:
    def test_allowed_channels(self):
        from EvoScientist.channels.feishu.channel import FeishuConfig

        config = FeishuConfig(
            app_id="test", app_secret="test",
            allowed_channels={"oc_abc123"},
        )
        assert config.allowed_channels == {"oc_abc123"}


# ── Slack retry tests ────────────────────────────────────────────


class TestSlackRetry:
    def test_extract_retry_after_from_response(self):
        from EvoScientist.channels.slack.channel import SlackChannel, SlackConfig

        config = SlackConfig(bot_token="test", app_token="xapp-test")
        ch = SlackChannel(config)

        # Simulate a SlackApiError-like exception with response headers
        class FakeResponse:
            headers = {"Retry-After": "30"}
        class FakeError(Exception):
            response = FakeResponse()

        result = ch._extract_retry_after(FakeError())
        assert result == 30.0

    def test_extract_retry_after_fallback(self):
        from EvoScientist.channels.slack.channel import SlackChannel, SlackConfig

        config = SlackConfig(bot_token="test", app_token="xapp-test")
        ch = SlackChannel(config)

        # Plain exception without response attribute
        result = ch._extract_retry_after(Exception("some error"))
        assert result == 1.0  # base class default


# ── Attachment size check tests ──────────────────────────────────


class TestAttachmentSizeCheck:
    """Tests for _check_attachment_size and _download_attachment size guard."""

    def test_check_attachment_size_within_limit(self):
        config = _StubConfig()
        ch = _StubChannel(config)
        assert ch._check_attachment_size(100, "small.txt") is None

    def test_check_attachment_size_exceeds_limit(self):
        from EvoScientist.channels.base import MAX_ATTACHMENT_BYTES

        config = _StubConfig()
        ch = _StubChannel(config)
        result = ch._check_attachment_size(MAX_ATTACHMENT_BYTES + 1, "big.zip")
        assert result is not None
        assert "too large" in result


# ── Feishu _strip_mention tests ──────────────────────────────────


class TestFeishuStripMention:
    def test_strip_bot_mention_placeholder(self):
        from EvoScientist.channels.feishu.channel import FeishuChannel, FeishuConfig

        config = FeishuConfig(app_id="test", app_secret="test")
        ch = FeishuChannel(config)
        ch._mention_names = ["@_user_1"]
        assert ch._strip_mention("@_user_1 hello world") == "hello world"

    def test_strip_multiple_mentions(self):
        from EvoScientist.channels.feishu.channel import FeishuChannel, FeishuConfig

        config = FeishuConfig(app_id="test", app_secret="test")
        ch = FeishuChannel(config)
        ch._mention_names = ["@_user_1", "@_user_2"]
        result = ch._strip_mention("@_user_1 @_user_2 hello")
        assert result == "hello"

    def test_no_mention_names(self):
        from EvoScientist.channels.feishu.channel import FeishuChannel, FeishuConfig

        config = FeishuConfig(app_id="test", app_secret="test")
        ch = FeishuChannel(config)
        assert ch._strip_mention("hello world") == "hello world"


# ── Telegram allowed_channels tests ──────────────────────────────


class TestTelegramConfig:
    def test_allowed_channels_field_exists(self):
        from EvoScientist.channels.telegram.channel import TelegramConfig

        config = TelegramConfig(bot_token="test", allowed_channels={"-100123"})
        assert config.allowed_channels == {"-100123"}

    def test_allowed_channels_default_none(self):
        from EvoScientist.channels.telegram.channel import TelegramConfig

        config = TelegramConfig(bot_token="test")
        assert config.allowed_channels is None

    def test_channel_allow_list_integration(self):
        from EvoScientist.channels.telegram.channel import TelegramChannel, TelegramConfig

        config = TelegramConfig(bot_token="test", allowed_channels={"-100123"})
        ch = TelegramChannel(config)
        assert ch.is_channel_allowed("-100123") is True
        assert ch.is_channel_allowed("-100999") is False

    def test_channel_allow_list_empty_allows_all(self):
        from EvoScientist.channels.telegram.channel import TelegramChannel, TelegramConfig

        config = TelegramConfig(bot_token="test")
        ch = TelegramChannel(config)
        assert ch.is_channel_allowed("-100999") is True


# ── Feishu retry tests ───────────────────────────────────────────


class TestFeishuRetry:
    def test_extract_retry_after_rate_limit(self):
        from EvoScientist.channels.feishu.channel import FeishuChannel, FeishuConfig

        config = FeishuConfig(app_id="test", app_secret="test")
        ch = FeishuChannel(config)
        result = ch._extract_retry_after(Exception("code 99991400: rate limit"))
        assert result == 2.0

    def test_extract_retry_after_generic(self):
        from EvoScientist.channels.feishu.channel import FeishuChannel, FeishuConfig

        config = FeishuConfig(app_id="test", app_secret="test")
        ch = FeishuChannel(config)
        result = ch._extract_retry_after(Exception("some error"))
        assert result == 1.0


# ── iMessage reply_to tests ──────────────────────────────────────


class TestIMessageReplyTo:
    def test_send_chunk_includes_reply_to(self):
        """Verify reply_to is passed through to RPC params."""
        from EvoScientist.channels.imessage.channel_rpc import IMessageChannelRpc

        ch = IMessageChannelRpc()
        # Mock the RPC client
        captured_params = {}

        class FakeClient:
            async def request(self, method, params):
                captured_params.update(params)
                return {}

        ch._client = FakeClient()

        _run(ch._send_chunk("chat123", "hello", "hello", "msg42", {"chat_id": "chat123"}))
        assert captured_params.get("reply_to") == "msg42"

    def test_send_chunk_omits_reply_to_when_none(self):
        from EvoScientist.channels.imessage.channel_rpc import IMessageChannelRpc

        ch = IMessageChannelRpc()
        captured_params = {}

        class FakeClient:
            async def request(self, method, params):
                captured_params.update(params)
                return {}

        ch._client = FakeClient()

        _run(ch._send_chunk("chat123", "hello", "hello", None, {"chat_id": "chat123"}))
        assert "reply_to" not in captured_params
