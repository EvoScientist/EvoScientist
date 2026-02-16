"""Tests for WeChat channel implementation."""

import asyncio
import hashlib
import time
import xml.etree.ElementTree as ET

import pytest

from EvoScientist.channels.wechat.channel import (
    WeChatChannel,
    WeComConfig,
    WeChatMPConfig,
    _strip_markdown,
)
from EvoScientist.channels.wechat.crypto import (
    WeChatCrypto,
    parse_xml,
    _pkcs7_pad,
    _pkcs7_unpad,
)
from EvoScientist.channels.base import ChannelError


def _run(coro):
    """Run an async coroutine safely, creating a fresh event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── Config tests ──────────────────────────────────────────────────

class TestWeComConfig:
    def test_default_values(self):
        config = WeComConfig()
        assert config.corp_id == ""
        assert config.agent_id == ""
        assert config.secret == ""
        assert config.webhook_port == 9001
        assert config.allowed_senders is None
        assert config.text_chunk_limit == 4096

    def test_custom_values(self):
        config = WeComConfig(
            corp_id="corp123",
            agent_id="1000001",
            secret="my-secret",
            token="my-token",
            encoding_aes_key="a" * 43,
            webhook_port=8080,
            allowed_senders={"user1", "user2"},
        )
        assert config.corp_id == "corp123"
        assert config.agent_id == "1000001"
        assert config.allowed_senders == {"user1", "user2"}
        assert config.webhook_port == 8080


class TestWeChatMPConfig:
    def test_default_values(self):
        config = WeChatMPConfig()
        assert config.app_id == ""
        assert config.app_secret == ""
        assert config.webhook_port == 9001

    def test_custom_values(self):
        config = WeChatMPConfig(
            app_id="wx1234",
            app_secret="secret",
            token="mp-token",
        )
        assert config.app_id == "wx1234"


# ── Channel init / lifecycle tests ────────────────────────────────

class TestWeChatChannelInit:
    def test_wecom_init(self):
        config = WeComConfig(corp_id="corp", agent_id="1", secret="s")
        channel = WeChatChannel(config, backend="wecom")
        assert channel.name == "wechat"
        assert channel._backend == "wecom"
        assert channel._running is False

    def test_mp_init(self):
        config = WeChatMPConfig(app_id="wx", app_secret="s")
        channel = WeChatChannel(config, backend="wechatmp")
        assert channel._backend == "wechatmp"

    def test_start_raises_without_corp_id(self):
        config = WeComConfig(corp_id="", agent_id="1", secret="s")
        channel = WeChatChannel(config, backend="wecom")
        with pytest.raises(ChannelError, match="corp_id"):
            _run(channel.start())

    def test_start_raises_without_secret(self):
        config = WeComConfig(corp_id="corp", agent_id="1", secret="")
        channel = WeChatChannel(config, backend="wecom")
        with pytest.raises(ChannelError, match="secret"):
            _run(channel.start())

    def test_start_raises_without_agent_id(self):
        config = WeComConfig(corp_id="corp", agent_id="", secret="s")
        channel = WeChatChannel(config, backend="wecom")
        with pytest.raises(ChannelError, match="agent_id"):
            _run(channel.start())

    def test_start_raises_mp_without_app_id(self):
        config = WeChatMPConfig(app_id="", app_secret="s")
        channel = WeChatChannel(config, backend="wechatmp")
        with pytest.raises(ChannelError, match="app_id"):
            _run(channel.start())

    def test_stop_when_not_running(self):
        config = WeComConfig(corp_id="c", agent_id="1", secret="s")
        channel = WeChatChannel(config, backend="wecom")
        _run(channel.stop())  # Should not raise

    def test_send_returns_false_without_client(self):
        from EvoScientist.channels.base import OutboundMessage

        config = WeComConfig(corp_id="c", agent_id="1", secret="s")
        channel = WeChatChannel(config, backend="wecom")
        msg = OutboundMessage(
            channel="wechat",
            chat_id="user1",
            content="hello",
            metadata={"chat_id": "user1"},
        )
        result = _run(channel.send(msg))
        assert result is False


# ── Markdown stripping tests ──────────────────────────────────────

class TestStripMarkdown:
    def test_plain_text(self):
        assert _strip_markdown("hello world") == "hello world"

    def test_bold(self):
        assert _strip_markdown("**bold**") == "bold"

    def test_italic(self):
        assert _strip_markdown("_italic_") == "italic"

    def test_code(self):
        assert _strip_markdown("`code`") == "code"

    def test_link(self):
        result = _strip_markdown("[text](https://example.com)")
        assert "text" in result
        assert "https://example.com" in result

    def test_heading(self):
        assert _strip_markdown("## Title").strip() == "Title"

    def test_list_items(self):
        result = _strip_markdown("- item1\n- item2")
        assert "• item1" in result
        assert "• item2" in result

    def test_strikethrough(self):
        assert _strip_markdown("~~deleted~~") == "deleted"

    def test_code_block(self):
        text = "```python\nprint('hi')\n```"
        result = _strip_markdown(text)
        assert "print('hi')" in result


# ── XML parsing tests ─────────────────────────────────────────────

class TestParseXml:
    def test_basic_text_message(self):
        xml = (
            "<xml>"
            "<MsgType><![CDATA[text]]></MsgType>"
            "<Content><![CDATA[hello]]></Content>"
            "<FromUserName><![CDATA[user123]]></FromUserName>"
            "<ToUserName><![CDATA[bot]]></ToUserName>"
            "<MsgId>1234</MsgId>"
            "<CreateTime>1700000000</CreateTime>"
            "</xml>"
        )
        data = parse_xml(xml)
        assert data["MsgType"] == "text"
        assert data["Content"] == "hello"
        assert data["FromUserName"] == "user123"
        assert data["MsgId"] == "1234"

    def test_image_message(self):
        xml = (
            "<xml>"
            "<MsgType><![CDATA[image]]></MsgType>"
            "<PicUrl><![CDATA[https://example.com/img.jpg]]></PicUrl>"
            "<MediaId><![CDATA[media_123]]></MediaId>"
            "<FromUserName><![CDATA[user1]]></FromUserName>"
            "</xml>"
        )
        data = parse_xml(xml)
        assert data["MsgType"] == "image"
        assert data["PicUrl"] == "https://example.com/img.jpg"

    def test_event_message(self):
        xml = (
            "<xml>"
            "<MsgType><![CDATA[event]]></MsgType>"
            "<Event><![CDATA[subscribe]]></Event>"
            "<FromUserName><![CDATA[user1]]></FromUserName>"
            "</xml>"
        )
        data = parse_xml(xml)
        assert data["MsgType"] == "event"
        assert data["Event"] == "subscribe"


# ── Crypto tests ──────────────────────────────────────────────────

class TestPKCS7:
    def test_pad_unpad_roundtrip(self):
        data = b"hello"
        padded = _pkcs7_pad(data)
        assert len(padded) % 32 == 0
        assert _pkcs7_unpad(padded) == data

    def test_pad_block_aligned(self):
        data = b"x" * 32
        padded = _pkcs7_pad(data)
        assert len(padded) == 64  # full padding block added
        assert _pkcs7_unpad(padded) == data


class TestWeChatCrypto:
    """Test the encryption/decryption roundtrip.

    Uses a deterministic 43-char EncodingAESKey.
    """

    # Skip encryption tests when no crypto backend is available
    _has_crypto = False
    try:
        from Crypto.Cipher import AES as _aes  # noqa: F401
        _has_crypto = True
    except ImportError:
        try:
            import pyaes as _pyaes  # noqa: F401
            _has_crypto = True
        except ImportError:
            pass
    pytestmark = pytest.mark.skipif(
        not _has_crypto,
        reason="pycryptodome or pyaes required for encryption tests",
    )

    @pytest.fixture
    def crypto(self):
        # 43 base64 chars → 32 bytes AES key
        key = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFG"
        return WeChatCrypto(
            token="test_token",
            encoding_aes_key=key,
            app_id="wx_test_app",
        )

    def test_encrypt_decrypt_roundtrip(self, crypto):
        msg = "<xml><Content>Hello WeChat!</Content></xml>"
        encrypted = crypto.encrypt(msg)
        decrypted, app_id = crypto.decrypt(encrypted)
        assert decrypted == msg
        assert app_id == "wx_test_app"

    def test_verify_signature(self, crypto):
        timestamp = "1609459200"
        nonce = "abc123"
        parts = sorted([crypto.token, timestamp, nonce])
        expected = hashlib.sha1("".join(parts).encode()).hexdigest()
        assert crypto.verify_signature(expected, timestamp, nonce)
        assert not crypto.verify_signature("wrong", timestamp, nonce)

    def test_verify_signature_with_encrypt(self, crypto):
        timestamp = "1609459200"
        nonce = "abc123"
        encrypt = "some_encrypted_data"
        parts = sorted([crypto.token, timestamp, nonce, encrypt])
        expected = hashlib.sha1("".join(parts).encode()).hexdigest()
        assert crypto.verify_signature(expected, timestamp, nonce, encrypt)

    def test_generate_signature(self, crypto):
        encrypt = "test_encrypted"
        timestamp = "1609459200"
        nonce = "abc"
        sig = crypto.generate_signature(encrypt, timestamp, nonce)
        parts = sorted([crypto.token, timestamp, nonce, encrypt])
        expected = hashlib.sha1("".join(parts).encode()).hexdigest()
        assert sig == expected

    def test_wrap_encrypted_reply(self, crypto):
        msg = "<xml><Content>Reply</Content></xml>"
        xml_reply = crypto.wrap_encrypted_reply(msg)
        assert "<Encrypt>" in xml_reply
        assert "<MsgSignature>" in xml_reply
        assert "<TimeStamp>" in xml_reply
        assert "<Nonce>" in xml_reply

        # Parse and verify the encrypted content decrypts back
        root = ET.fromstring(xml_reply)
        encrypt = root.find("Encrypt").text
        decrypted, app_id = crypto.decrypt(encrypt)
        assert decrypted == msg


# ── Message processing tests ──────────────────────────────────────

class TestMessageProcessing:
    """Test the _process_message method with various XML payloads."""

    def _make_channel(self):
        config = WeComConfig(
            corp_id="corp", agent_id="1", secret="s",
        )
        return WeChatChannel(config, backend="wecom")

    def test_text_message_queued(self):
        channel = self._make_channel()

        async def _test():
            await channel._process_message({
                "MsgType": "text",
                "Content": "Hello!",
                "FromUserName": "user1",
                "ToUserName": "bot",
                "MsgId": "100",
                "CreateTime": str(int(time.time())),
            })
            # Check message was enqueued
            assert not channel._queue.empty()
            msg = await asyncio.wait_for(channel._queue.get(), timeout=1.0)
            assert msg.content == "Hello!"
            assert msg.sender_id == "user1"
            assert msg.channel == "wechat"

        _run(_test())

    def test_location_message(self):
        channel = self._make_channel()

        async def _test():
            await channel._process_message({
                "MsgType": "location",
                "Location_X": "39.9",
                "Location_Y": "116.4",
                "Label": "Beijing",
                "FromUserName": "user1",
                "ToUserName": "bot",
                "MsgId": "101",
                "CreateTime": str(int(time.time())),
            })
            msg = await asyncio.wait_for(channel._queue.get(), timeout=1.0)
            assert "Beijing" in msg.content
            assert "39.9" in msg.content

        _run(_test())

    def test_voice_recognition(self):
        channel = self._make_channel()

        async def _test():
            await channel._process_message({
                "MsgType": "voice",
                "Recognition": "你好世界",
                "FromUserName": "user1",
                "ToUserName": "bot",
                "MsgId": "102",
                "CreateTime": str(int(time.time())),
            })
            msg = await asyncio.wait_for(channel._queue.get(), timeout=1.0)
            assert "你好世界" in msg.content

        _run(_test())

    def test_link_message(self):
        channel = self._make_channel()

        async def _test():
            await channel._process_message({
                "MsgType": "link",
                "Title": "Test Link",
                "Description": "A description",
                "Url": "https://example.com",
                "FromUserName": "user1",
                "ToUserName": "bot",
                "MsgId": "103",
                "CreateTime": str(int(time.time())),
            })
            msg = await asyncio.wait_for(channel._queue.get(), timeout=1.0)
            assert "Test Link" in msg.content
            assert "https://example.com" in msg.content

        _run(_test())

    def test_subscribe_event(self):
        channel = self._make_channel()

        async def _test():
            await channel._process_message({
                "MsgType": "event",
                "Event": "subscribe",
                "FromUserName": "user1",
                "ToUserName": "bot",
                "MsgId": "",
                "CreateTime": str(int(time.time())),
            })
            msg = await asyncio.wait_for(channel._queue.get(), timeout=1.0)
            assert "关注" in msg.content

        _run(_test())

    def test_unsubscribe_ignored(self):
        channel = self._make_channel()

        async def _test():
            await channel._process_message({
                "MsgType": "event",
                "Event": "unsubscribe",
                "FromUserName": "user1",
                "ToUserName": "bot",
                "MsgId": "",
                "CreateTime": str(int(time.time())),
            })
            assert channel._queue.empty()

        _run(_test())

    def test_empty_message_ignored(self):
        channel = self._make_channel()

        async def _test():
            await channel._process_message({
                "MsgType": "text",
                "Content": "",
                "FromUserName": "",
                "ToUserName": "bot",
            })
            assert channel._queue.empty()

        _run(_test())


# ── Registration test ─────────────────────────────────────────────

class TestChannelRegistration:
    def test_wechat_registered(self):
        from EvoScientist.channels.channel_manager import available_channels
        channels = available_channels()
        assert "wechat" in channels
