"""Telegram channel server.

Standalone script to run the Telegram channel with CLI options.

Usage:
    python -m EvoScientist.channels.telegram.serve --bot-token TOKEN [OPTIONS]

Examples:
    # Allow all senders (default)
    python -m EvoScientist.channels.telegram.serve --bot-token TOKEN

    # Only allow specific senders
    python -m EvoScientist.channels.telegram.serve --bot-token TOKEN --allow 123456 --allow 789012

    # With agent and thinking
    python -m EvoScientist.channels.telegram.serve --bot-token TOKEN --agent --thinking
"""

import argparse
import logging

from ..bus import MessageBus
from ..standalone import run_standalone
from .channel import TelegramChannel, TelegramConfig

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Telegram channel server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--bot-token",
        required=True,
        help="Telegram bot token from @BotFather",
    )
    parser.add_argument(
        "--allow",
        action="append",
        dest="allowed_senders",
        help="Allowed sender (Telegram user ID). Can be used multiple times.",
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

    config = TelegramConfig(
        bot_token=args.bot_token,
        allowed_senders=set(args.allowed_senders) if args.allowed_senders else None,
    )

    send_thinking = args.thinking and args.agent
    bus = MessageBus()
    channel = TelegramChannel(config)

    run_standalone(channel, bus, use_agent=args.agent, send_thinking=send_thinking)


if __name__ == "__main__":
    main()
