"""iMessage channel server.

Standalone script to run the iMessage channel with CLI options.

Usage:
    python -m EvoScientist.channels.imessage.serve [OPTIONS]

Examples:
    # Allow all senders (default)
    python -m EvoScientist.channels.imessage.serve

    # Only allow specific senders
    python -m EvoScientist.channels.imessage.serve --allow +1234567890 --allow user@example.com

    # Custom imsg path
    python -m EvoScientist.channels.imessage.serve --cli-path /usr/local/bin/imsg
"""

import argparse
import logging

from . import IMessageChannel, IMessageConfig
from ..bus import MessageBus
from ..standalone import run_standalone

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="iMessage channel server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--allow",
        action="append",
        dest="allowed_senders",
        help="Allowed sender (phone/email). Can be used multiple times.",
    )
    parser.add_argument(
        "--cli-path",
        default="imsg",
        help="Path to imsg CLI (default: imsg)",
    )
    parser.add_argument(
        "--db-path",
        help="Path to Messages database",
    )
    parser.add_argument(
        "--attachments",
        action="store_true",
        help="Include attachments in messages",
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

    config = IMessageConfig(
        cli_path=args.cli_path,
        db_path=args.db_path,
        allowed_senders=list(args.allowed_senders) if args.allowed_senders else [],
        include_attachments=args.attachments,
    )

    send_thinking = args.thinking and args.agent
    bus = MessageBus()
    channel = IMessageChannel(config)

    run_standalone(channel, bus, use_agent=args.agent, send_thinking=send_thinking)


if __name__ == "__main__":
    main()
