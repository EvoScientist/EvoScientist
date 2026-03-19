"""Additional smoke tests for channels without dedicated suites."""

import pytest

from EvoScientist.channels.base import ChannelError, OutboundMessage
from EvoScientist.channels.email.channel import EmailChannel, EmailConfig
from EvoScientist.channels.imessage.channel_rpc import (
    IMessageChannelRpc,
    IMessageConfig,
)
from EvoScientist.channels.qq.channel import QQChannel, QQConfig
from EvoScientist.channels.signal.channel import SignalChannel, SignalConfig
from tests.conftest import run_async as _run


class TestEmailChannelSmoke:
    def test_start_raises_without_required_imap_settings(self):
        channel = EmailChannel(EmailConfig())
        with pytest.raises(
            ChannelError, match="imap_host and imap_username are required"
        ):
            _run(channel.start())

    def test_send_returns_false_when_smtp_not_ready(self):
        channel = EmailChannel(EmailConfig())
        msg = OutboundMessage(
            channel="email",
            chat_id="user@example.com",
            content="hello",
            metadata={"chat_id": "user@example.com"},
        )
        assert _run(channel.send(msg)) is False


class TestSignalChannelSmoke:
    def test_start_raises_without_phone_number(self):
        channel = SignalChannel(SignalConfig())
        with pytest.raises(ChannelError, match="phone_number is required"):
            _run(channel.start())

    def test_send_returns_false_when_not_connected(self):
        channel = SignalChannel(SignalConfig(phone_number="+123456789"))
        msg = OutboundMessage(
            channel="signal",
            chat_id="+123456789",
            content="hello",
            metadata={"chat_id": "+123456789"},
        )
        assert _run(channel.send(msg)) is False


class TestQQChannelSmoke:
    def test_start_raises_when_sdk_missing(self, monkeypatch):
        from EvoScientist.channels.qq import channel as qq_module

        monkeypatch.setattr(qq_module, "QQ_AVAILABLE", False)
        channel = QQChannel(QQConfig(app_id="id", app_secret="secret"))
        with pytest.raises(ChannelError, match="SDK not installed"):
            _run(channel.start())

    def test_start_raises_without_credentials_when_sdk_available(self, monkeypatch):
        from EvoScientist.channels.qq import channel as qq_module

        monkeypatch.setattr(qq_module, "QQ_AVAILABLE", True)
        channel = QQChannel(QQConfig(app_id="", app_secret=""))
        with pytest.raises(ChannelError, match="app_id and app_secret are required"):
            _run(channel.start())

    def test_send_returns_false_without_client(self):
        channel = QQChannel(QQConfig(app_id="id", app_secret="secret"))
        msg = OutboundMessage(
            channel="qq",
            chat_id="openid",
            content="hello",
            metadata={"chat_id": "openid"},
        )
        assert _run(channel.send(msg)) is False


class TestIMessageChannelSmoke:
    def test_start_wraps_rpc_bootstrap_error(self, monkeypatch):
        async def _broken_start(self):
            raise RuntimeError("imsg not found")

        from EvoScientist.channels.imessage import channel_rpc as imessage_module

        monkeypatch.setattr(imessage_module.ImsgRpcClient, "start", _broken_start)
        channel = IMessageChannelRpc(IMessageConfig())
        with pytest.raises(ChannelError, match="Failed to start imsg"):
            _run(channel.start())

    def test_send_returns_false_without_rpc_client(self):
        channel = IMessageChannelRpc(IMessageConfig())
        msg = OutboundMessage(
            channel="imessage",
            chat_id="+123456789",
            content="hello",
            metadata={"chat_id": "+123456789"},
        )
        assert _run(channel.send(msg)) is False
