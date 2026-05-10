"""Tests for QQ channel markdown send behavior."""

from unittest.mock import AsyncMock, MagicMock

from EvoScientist.channels.base import OutboundMessage
from EvoScientist.channels.qq.channel import (
    QQChannel,
    QQConfig,
    _build_qq_keyboard,
)
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
        # Plain fallback consumes a fresh msg_seq — QQ may have already
        # counted the failed markdown attempt, so re-using seq would
        # trigger "duplicate msg_seq".
        assert second["msg_seq"] == 2

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

    def test_send_does_not_fallback_when_transport_error_mentions_markdown(self):
        """A transport-layer error whose message incidentally contains the word
        "markdown" must NOT be reclassified as a markdown compatibility failure,
        otherwise genuine send failures get silently swallowed as plain-text."""
        channel = self._make_ready_channel()

        async def _send_once(coro_factory, max_retries=3):
            return await coro_factory()

        channel._send_with_retry = _send_once
        channel._client.api.post_c2c_message = AsyncMock(
            side_effect=ConnectionError(
                "failed to post markdown message: ConnectionReset"
            )
        )
        msg = OutboundMessage(
            channel="qq",
            chat_id="openid",
            content="## Title",
            metadata={
                "chat_id": "openid",
                "event_id": "evt_transport",
                "msg_type": "c2c",
            },
        )

        assert _run(channel.send(msg)) is False
        channel._client.api.post_c2c_message.assert_awaited_once()

    def test_send_falls_back_on_qq_server_error_code(self):
        """QQ server-side markdown errors (e.g. 304014 template not configured)
        should trigger plain-text fallback with a fresh msg_seq."""
        channel = self._make_ready_channel()
        channel._client.api.post_c2c_message = AsyncMock(
            side_effect=[
                RuntimeError(
                    '{"code": 304014, "message": "markdown template not configured"}'
                ),
                None,
            ]
        )
        msg = OutboundMessage(
            channel="qq",
            chat_id="openid",
            content="## Title",
            metadata={
                "chat_id": "openid",
                "event_id": "evt_4",
                "msg_type": "c2c",
            },
        )

        assert _run(channel.send(msg)) is True

        assert channel._client.api.post_c2c_message.await_count == 2
        first = channel._client.api.post_c2c_message.await_args_list[0].kwargs
        second = channel._client.api.post_c2c_message.await_args_list[1].kwargs

        assert first["msg_type"] == 2
        assert first["msg_seq"] == 1
        assert second["msg_type"] == 0
        # Fresh seq on fallback — avoids QQ "duplicate msg_seq" rejection.
        assert second["msg_seq"] == 2


class TestQQKeyboardBuilder:
    def test_basic_buttons(self):
        kb = _build_qq_keyboard(
            [
                {"text": "Approve", "value": "1", "type": "primary"},
                {"text": "Reject", "value": "2", "type": "danger"},
            ]
        )
        rows = kb["content"]["rows"]
        assert len(rows) == 2  # one button per row
        approve_btn = rows[0]["buttons"][0]
        assert approve_btn["render_data"]["label"] == "Approve"
        assert approve_btn["render_data"]["style"] == 1  # primary
        assert approve_btn["action"]["type"] == 1  # callback
        assert approve_btn["action"]["data"] == "1"
        # "danger" maps to grey (style 0) — QQ has no danger style
        assert rows[1]["buttons"][0]["render_data"]["style"] == 0

    def test_default_value_uses_label(self):
        kb = _build_qq_keyboard([{"text": "OK"}])
        assert kb["content"]["rows"][0]["buttons"][0]["action"]["data"] == "OK"

    def test_skips_empty_label(self):
        kb = _build_qq_keyboard(
            [
                {"text": "", "value": "skip"},
                {"text": "Keep", "value": "k"},
            ]
        )
        rows = kb["content"]["rows"]
        assert len(rows) == 1
        assert rows[0]["buttons"][0]["render_data"]["label"] == "Keep"

    def test_returns_none_when_no_valid_buttons(self):
        assert _build_qq_keyboard([]) is None
        assert _build_qq_keyboard([{"text": ""}]) is None

    def test_explicit_id_preserved(self):
        kb = _build_qq_keyboard([{"text": "Go", "value": "go", "id": "custom_id"}])
        assert kb["content"]["rows"][0]["buttons"][0]["id"] == "custom_id"

    def test_non_string_value_coerced(self):
        kb = _build_qq_keyboard([{"text": "OK", "value": 42}])
        assert kb["content"]["rows"][0]["buttons"][0]["action"]["data"] == "42"


class TestQQSendWithButtons:
    """Send path attaches `keyboard` to markdown payload for C2C messages."""

    @staticmethod
    def _make_channel() -> QQChannel:
        channel = QQChannel(QQConfig(app_id="id", app_secret="secret"))
        channel._running = True
        channel._client = MagicMock()
        channel._client.api = MagicMock()
        channel._client.api.post_c2c_message = AsyncMock()
        channel._client.api.post_group_message = AsyncMock()
        return channel

    def test_c2c_send_attaches_keyboard(self):
        channel = self._make_channel()
        msg = OutboundMessage(
            channel="qq",
            chat_id="openid",
            content="Pick:",
            metadata={
                "chat_id": "openid",
                "event_id": "evt_btn",
                "msg_type": "c2c",
                "buttons": [
                    {"text": "Approve", "value": "1", "type": "primary"},
                    {"text": "Reject", "value": "2"},
                ],
            },
        )
        assert _run(channel.send(msg)) is True

        sent = channel._client.api.post_c2c_message.await_args.kwargs
        assert sent["msg_type"] == 2
        assert "keyboard" in sent
        rows = sent["keyboard"]["content"]["rows"]
        assert rows[0]["buttons"][0]["action"]["data"] == "1"
        assert rows[1]["buttons"][0]["action"]["data"] == "2"

    def test_group_send_does_not_attach_keyboard(self):
        """Group keyboards are out of scope — silently dropped."""
        channel = self._make_channel()
        msg = OutboundMessage(
            channel="qq",
            chat_id="group_openid",
            content="Pick:",
            metadata={
                "chat_id": "group_openid",
                "event_id": "evt_group",
                "msg_type": "group",
                "buttons": [{"text": "Approve", "value": "1"}],
            },
        )
        assert _run(channel.send(msg)) is True
        sent = channel._client.api.post_group_message.await_args.kwargs
        assert "keyboard" not in sent

    def test_fallback_appends_button_hint_when_keyboard_present(self):
        """If markdown send fails and we fall back to plain text, the
        keyboard is lost — append a textual hint so the user still has
        a way to reply (the values still pass `_parse_approval_reply`).
        """
        channel = self._make_channel()
        channel._client.api.post_c2c_message = AsyncMock(
            side_effect=[
                RuntimeError(
                    '{"code": 304014, "message": "markdown template not configured"}'
                ),
                None,
            ]
        )
        msg = OutboundMessage(
            channel="qq",
            chat_id="openid",
            content="Pick:",
            metadata={
                "chat_id": "openid",
                "event_id": "evt_fb",
                "msg_type": "c2c",
                "buttons": [
                    {"text": "Approve", "value": "1"},
                    {"text": "Reject", "value": "2"},
                ],
            },
        )
        assert _run(channel.send(msg)) is True

        plain_call = channel._client.api.post_c2c_message.await_args_list[1].kwargs
        assert plain_call["msg_type"] == 0
        # Fallback content includes hint mapping value→label so the user
        # can type "1"/"2" without the original button UI.
        assert "1=Approve" in plain_call["content"]
        assert "2=Reject" in plain_call["content"]

    def test_fallback_hint_handles_non_string_button_value(self):
        """Regression: integer/None button values must not crash the
        plain-text fallback (the keyboard builder already coerces them)."""
        channel = self._make_channel()
        channel._client.api.post_c2c_message = AsyncMock(
            side_effect=[
                RuntimeError('{"code": 304014, "message": "template not configured"}'),
                None,
            ]
        )
        msg = OutboundMessage(
            channel="qq",
            chat_id="openid",
            content="Pick:",
            metadata={
                "chat_id": "openid",
                "event_id": "evt_coerce",
                "msg_type": "c2c",
                "buttons": [
                    {"text": "OK", "value": 42},  # int
                    {"text": "Cancel"},  # value omitted → defaults to label
                ],
            },
        )
        assert _run(channel.send(msg)) is True
        plain_call = channel._client.api.post_c2c_message.await_args_list[1].kwargs
        assert "42=OK" in plain_call["content"]
        assert "Cancel=Cancel" in plain_call["content"]


class TestQQInteractionCallback:
    """`_on_interaction` should publish click as InboundMessage to the bus
    (skipping debounce) and ACK the interaction."""

    @staticmethod
    def _make_channel() -> QQChannel:
        from EvoScientist.channels.bus.events import InboundMessage

        channel = QQChannel(QQConfig(app_id="id", app_secret="secret"))
        channel._running = True
        channel._client = MagicMock()
        channel._client.api = MagicMock()
        channel._client.api.on_interaction_result = AsyncMock()

        async def _fake_build(raw):
            channel._captured_raw = raw
            return InboundMessage(
                channel="qq",
                sender_id=raw.sender_id,
                chat_id=raw.chat_id,
                content=raw.text,
                metadata=raw.metadata,
                message_id=raw.message_id,
                is_group=raw.is_group,
                was_mentioned=raw.was_mentioned,
            )

        channel._build_inbound_async = _fake_build
        channel._captured_raw = None
        channel._bus = MagicMock()
        channel._bus.publish_inbound = AsyncMock()
        return channel

    @staticmethod
    def _make_interaction(button_data="1", button_id="btn_0", user_openid="u_xxx"):
        resolved = MagicMock(
            button_id=button_id,
            button_data=button_data,
            message_id="msg_orig",
            user_id=None,
            feature_id=None,
        )
        data = MagicMock(type=None, resolved=resolved)
        interaction = MagicMock(
            id="intr_1",
            user_openid=user_openid,
            group_openid=None,
            data=data,
        )
        return interaction

    def test_click_publishes_to_bus_with_button_data(self):
        channel = self._make_channel()
        _run(channel._on_interaction(self._make_interaction("1")))

        channel._bus.publish_inbound.assert_awaited_once()
        inbound = channel._bus.publish_inbound.await_args[0][0]
        assert inbound.content == "1"
        assert inbound.sender_id == "u_xxx"
        assert inbound.chat_id == "u_xxx"
        assert inbound.metadata["button_click"] is True
        assert inbound.metadata["button_value"] == "1"
        assert inbound.metadata["msg_type"] == "c2c"

    def test_click_acks_interaction(self):
        channel = self._make_channel()
        _run(channel._on_interaction(self._make_interaction("1")))
        channel._client.api.on_interaction_result.assert_awaited_once_with("intr_1", 0)

    def test_click_bypasses_debounce(self):
        """Click never hits queue_message (debounce buffer)."""
        channel = self._make_channel()
        channel.queue_message = AsyncMock()
        _run(channel._on_interaction(self._make_interaction("3")))
        channel.queue_message.assert_not_called()
        channel._bus.publish_inbound.assert_awaited_once()

    def test_group_interaction_ignored(self):
        """No user_openid → group/guild click → don't publish."""
        channel = self._make_channel()
        intr = self._make_interaction(user_openid="")
        intr.group_openid = "group_xxx"
        _run(channel._on_interaction(intr))
        channel._bus.publish_inbound.assert_not_called()
        # ACK still fires — it runs first, before the group-skip return.
        channel._client.api.on_interaction_result.assert_awaited_once_with("intr_1", 0)

    def test_click_dropped_when_middleware_rejects(self):
        channel = self._make_channel()
        channel._build_inbound_async = AsyncMock(return_value=None)
        _run(channel._on_interaction(self._make_interaction("1")))
        channel._bus.publish_inbound.assert_not_called()
        # ACK still fires (we don't want the user staring at a stuck button)
        channel._client.api.on_interaction_result.assert_awaited_once()

    def test_empty_button_data_falls_back_to_button_id(self):
        channel = self._make_channel()
        _run(
            channel._on_interaction(
                self._make_interaction(button_data="", button_id="btn_3")
            )
        )
        inbound = channel._bus.publish_inbound.await_args[0][0]
        assert inbound.content == "btn_3"

    def test_ack_fires_even_when_handler_throws(self):
        """ACK must run before downstream processing so the QQ button UI
        stays responsive even if middleware/bus crashes."""
        channel = self._make_channel()
        channel._build_inbound_async = AsyncMock(side_effect=RuntimeError("boom"))
        # Should not raise — handler swallows downstream errors.
        _run(channel._on_interaction(self._make_interaction("1")))
        channel._client.api.on_interaction_result.assert_awaited_once_with("intr_1", 0)

    def test_button_value_metadata_is_string_coerced(self):
        """Regression: metadata['button_value'] must be a string (was raw)."""
        channel = self._make_channel()
        resolved = MagicMock(button_id="btn_0", button_data=42, message_id="msg_orig")
        data = MagicMock(type=None, resolved=resolved)
        intr = MagicMock(id="intr_1", user_openid="u_x", group_openid=None, data=data)
        _run(channel._on_interaction(intr))
        inbound = channel._bus.publish_inbound.await_args[0][0]
        assert inbound.content == "42"
        assert inbound.metadata["button_value"] == "42"


class TestQQCapabilitiesButtons:
    def test_inline_buttons_enabled(self):
        from EvoScientist.channels.capabilities import QQ

        assert QQ.inline_buttons is True
        assert QQ.supports("inline_buttons") is True
