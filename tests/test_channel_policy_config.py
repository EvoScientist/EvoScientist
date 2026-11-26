"""Top-level ``require_mention`` / ``dm_policy`` reach every channel."""

import pytest

from EvoScientist.channels.channel_manager import available_channels, create_channel
from EvoScientist.channels.middleware import (
    AllowListMiddleware,
    MentionGatingMiddleware,
)
from EvoScientist.config.settings import EvoScientistConfig


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
