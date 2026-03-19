"""Signal channel for EvoScientist.

Uses signal-cli in JSON RPC mode — no public IP needed.

Usage in config:
    channel_enabled = "signal"
    signal_phone_number = "+1234567890"
"""

from ..channel_manager import _parse_csv, register_channel
from .channel import SignalChannel, SignalConfig

__all__ = ["SignalChannel", "SignalConfig"]


def create_from_config(config) -> SignalChannel:
    allowed = _parse_csv(config.signal_allowed_senders)
    return SignalChannel(
        SignalConfig(
            phone_number=config.signal_phone_number,
            cli_path=config.signal_cli_path,
            config_dir=config.signal_config_dir or None,
            rpc_port=config.signal_rpc_port,
            allowed_senders=allowed,
        )
    )


register_channel("signal", create_from_config)
