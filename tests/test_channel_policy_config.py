"""Top-level ``require_mention`` / ``dm_policy`` reach every channel."""

import pytest

from EvoScientist.channels.base import RawIncoming
from EvoScientist.channels.channel_manager import (
    _ensure_channels_registered,
    available_channels,
    create_channel,
)
from EvoScientist.channels.middleware import (
    AllowListMiddleware,
    MentionGatingMiddleware,
)
from EvoScientist.config.settings import EvoScientistConfig

_ensure_channels_registered()


@pytest.mark.parametrize(
    ("name", "extra"),
    [(name, {}) for name in available_channels()]
    + [("wechat", {"wechat_backend": "wechatmp"})],
)
def test_channel_uses_configured_mention_and_dm_policy(name, extra):
    cfg = EvoScientistConfig(require_mention="off", dm_policy="open", **extra)
    ch = create_channel(name, cfg)

    assert ch.require_mention == "off"
    assert ch.dm_policy == "open"
    for mw in ch._inbound_middlewares:
        if isinstance(mw, MentionGatingMiddleware):
            assert mw.require_mention == "off"
        if isinstance(mw, AllowListMiddleware):
            assert mw.dm_policy == "open"


def test_personal_wechat_keeps_its_own_dm_policy():
    cfg = EvoScientistConfig(
        wechat_backend="personal",
        dm_policy="allowlist",
        wechat_personal_dm_policy="open",
    )
    ch = create_channel("wechat", cfg)

    assert ch.dm_policy == "open"


async def test_require_mention_off_does_not_replay_group_messages():
    cfg = EvoScientistConfig(telegram_bot_token="1:a", require_mention="off")
    ch = create_channel("telegram", cfg)

    for i, text in enumerate(("first", "second", "third")):
        raw = RawIncoming(
            sender_id="u1",
            chat_id="g1",
            text=text,
            message_id=str(i),
            is_group=True,
            was_mentioned=text == "third",
        )
        msg = await ch._build_inbound_async(raw)
        assert msg.content == text


async def test_imessage_open_dm_policy_admits_unlisted_direct_messages():
    cfg = EvoScientistConfig(imessage_allowed_senders="+15550001111", dm_policy="open")
    ch = create_channel("imessage", cfg)

    dm = RawIncoming(sender_id="+15559999999", chat_id="5", text="hi", message_id="1")
    group = RawIncoming(
        sender_id="+15559999999",
        chat_id="6",
        text="hi",
        message_id="2",
        is_group=True,
    )
    assert await ch._build_inbound_async(dm) is not None
    assert await ch._build_inbound_async(group) is None
