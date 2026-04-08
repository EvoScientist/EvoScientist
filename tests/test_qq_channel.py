"""Tests for QQ channel markdown send behavior."""

from unittest.mock import AsyncMock, MagicMock

from EvoScientist.channels.base import OutboundMessage
from EvoScientist.channels.qq.channel import QQChannel, QQConfig
from tests.conftest import run_async as _run


class TestQQChannelSend:
    @staticmethod
    def _make_ready_channel() -> QQChannel:
        channel = QQChannel(QQConfig(app_id="id", app_secret="secret"))
        channel._running = True
        channel._client = MagicMock()
        channel._client.api = MagicMock()
        channel._client.api.post_c2c_message = AsyncMock()
        channel._client.api.post_group_message = AsyncMock()
        return channel

    def test_send_prefers_native_markdown_for_c2c(self):
        channel = self._make_ready_channel()
        msg = OutboundMessage(
            channel="qq",
            chat_id="openid",
            content="## Title\n\n- item",
            metadata={
                "chat_id": "openid",
                "event_id": "evt_1",
                "msg_type": "c2c",
            },
        )

        assert _run(channel.send(msg)) is True

        channel._client.api.post_c2c_message.assert_awaited_once()
        sent = channel._client.api.post_c2c_message.await_args.kwargs
        assert sent["openid"] == "openid"
        assert sent["msg_type"] == 2
        assert sent["markdown"] == {"content": "## Title\n\n- item"}
        assert sent["msg_id"] == "evt_1"
        assert sent["msg_seq"] == 1
        assert "content" not in sent

    def test_send_falls_back_to_plain_text_when_markdown_send_fails(self):
        channel = self._make_ready_channel()
        channel._trace_event = MagicMock(side_effect=RuntimeError("trace failed"))
        channel._client.api.post_c2c_message = AsyncMock(
            side_effect=[TypeError("unexpected keyword argument 'markdown'"), None]
        )
        msg = OutboundMessage(
            channel="qq",
            chat_id="openid",
            content="## Title\n\n- item",
            metadata={
                "chat_id": "openid",
                "event_id": "evt_2",
                "msg_type": "c2c",
            },
        )

        assert _run(channel.send(msg)) is True

        assert channel._client.api.post_c2c_message.await_count == 2
        first = channel._client.api.post_c2c_message.await_args_list[0].kwargs
        second = channel._client.api.post_c2c_message.await_args_list[1].kwargs

        assert first["msg_type"] == 2
        assert first["markdown"] == {"content": "## Title\n\n- item"}

        assert second["msg_type"] == 0
        assert second["content"] == "Title\n\n• item"
        assert second["msg_id"] == "evt_2"
        assert second["msg_seq"] == 1

    def test_send_does_not_fallback_on_transport_error(self):
        channel = self._make_ready_channel()

        async def _send_once(coro_factory, max_retries=3):
            return await coro_factory()

        channel._send_with_retry = _send_once
        channel._client.api.post_c2c_message = AsyncMock(
            side_effect=RuntimeError("upstream service unavailable")
        )
        msg = OutboundMessage(
            channel="qq",
            chat_id="openid",
            content="## Title\n\n- item",
            metadata={
                "chat_id": "openid",
                "event_id": "evt_3",
                "msg_type": "c2c",
            },
        )

        assert _run(channel.send(msg)) is False
        channel._client.api.post_c2c_message.assert_awaited_once()
        sent = channel._client.api.post_c2c_message.await_args.kwargs
        assert sent["msg_type"] == 2
        assert "content" not in sent
