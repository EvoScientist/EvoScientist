"""Signal channel server.

Standalone script to run the Signal channel with CLI options.

Usage:
    python -m EvoScientist.channels.signal.serve --phone-number NUMBER [OPTIONS]

Examples:
    # Basic usage
    python -m EvoScientist.channels.signal.serve --phone-number +1234567890

    # With custom signal-cli path and allowed senders
    python -m EvoScientist.channels.signal.serve --phone-number +1234567890 --cli-path /usr/local/bin/signal-cli --allow +9876543210

    # With agent and thinking
    python -m EvoScientist.channels.signal.serve --phone-number +1234567890 --agent --thinking
"""

import argparse
import logging

from .channel import SignalChannel, SignalConfig
from ..bus import MessageBus
from ..standalone import run_standalone

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Signal channel server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phone-number",
        required=True,
        help="Signal phone number (e.g. +1234567890)",
    )
    parser.add_argument(
        "--cli-path",
        default="signal-cli",
        help="Path to signal-cli binary (default: signal-cli)",
    )
    parser.add_argument(
        "--config-dir",
        help="signal-cli config directory",
    )
    parser.add_argument(
        "--rpc-port",
        type=int,
        default=7583,
        help="signal-cli JSON RPC port (default: 7583)",
    )
    parser.add_argument(
        "--allow",
        action="append",
        dest="allowed_senders",
        help="Allowed sender (phone number). Can be used multiple times.",
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Use EvoScientist agent as handler (default: echo)",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Send thinking content as intermediate messages (requires --agent)",
    )
    return parser.parse_args()


def main():
    """Entry point."""
    args = parse_args()

    config = SignalConfig(
        phone_number=args.phone_number,
        cli_path=args.cli_path,
        config_dir=args.config_dir,
        rpc_port=args.rpc_port,
        allowed_senders=set(args.allowed_senders) if args.allowed_senders else None,
    )

    send_thinking = args.thinking and args.agent
    bus = MessageBus()
    channel = SignalChannel(config)

    run_standalone(channel, bus, use_agent=args.agent, send_thinking=send_thinking)


if __name__ == "__main__":
    main()
