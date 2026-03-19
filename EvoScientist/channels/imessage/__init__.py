"""iMessage channel implementation for EvoScientist.

Uses imsg CLI via JSON-RPC for real-time message streaming.

Requirements:
- macOS only
- imsg CLI: brew install steipete/tap/imsg
- Full Disk Access permission
- Messages.app logged into iCloud
"""

from ..channel_manager import _parse_csv, register_channel
from .channel_rpc import IMessageChannelRpc as IMessageChannel
from .channel_rpc import IMessageConfig
from .probe import ProbeResult, probe_imessage
from .targets import (
    IMessageService,
    IMessageTarget,
    normalize_e164,
    normalize_handle,
    parse_target,
)

__all__ = [
    "IMessageChannel",
    "IMessageConfig",
    "IMessageService",
    "IMessageTarget",
    "ProbeResult",
    "normalize_e164",
    "normalize_handle",
    "parse_target",
    "probe_imessage",
]


def create_from_config(config) -> IMessageChannel:
    allowed = _parse_csv(config.imessage_allowed_senders)
    return IMessageChannel(IMessageConfig(allowed_senders=allowed))


register_channel("imessage", create_from_config)
