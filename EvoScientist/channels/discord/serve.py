"""Discord channel server.

Standalone script to run the Discord channel with CLI options.

Usage:
    python -m EvoScientist.channels.discord.serve --bot-token TOKEN [OPTIONS]

Examples:
    # Allow all senders (default)
    python -m EvoScientist.channels.discord.serve --bot-token TOKEN

    # Only allow specific senders and channels
    python -m EvoScientist.channels.discord.serve --bot-token TOKEN --allow 123 --allow-channel 456

    # With proxy, agent and thinking
    python -m EvoScientist.channels.discord.serve --bot-token TOKEN --proxy http://proxy:8080 --agent --thinking
"""

import argparse
import logging

from .channel import DiscordChannel, DiscordConfig
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
        description="Discord channel server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--bot-token",
        required=True,
        help="Discord bot token",
    )
    parser.add_argument(
        "--allow",
        action="append",
        dest="allowed_senders",
        help="Allowed sender (Discord user ID). Can be used multiple times.",
    )
    parser.add_argument(
        "--allow-channel",
        action="append",
        dest="allowed_channels",
        help="Allowed channel ID. Can be used multiple times.",
    )
    parser.add_argument(
        "--proxy",
        help="HTTP proxy URL for Discord API requests",
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

    config = DiscordConfig(
        bot_token=args.bot_token,
        allowed_senders=set(args.allowed_senders) if args.allowed_senders else None,
        allowed_channels=set(args.allowed_channels) if args.allowed_channels else None,
        proxy=args.proxy,
    )

    send_thinking = args.thinking and args.agent
    bus = MessageBus()
    channel = DiscordChannel(config)

    run_standalone(channel, bus, use_agent=args.agent, send_thinking=send_thinking)


if __name__ == "__main__":
    main()
