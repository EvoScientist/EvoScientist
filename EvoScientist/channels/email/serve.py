"""Email channel server.

Standalone script to run the Email channel with CLI options.

Usage:
    python -m EvoScientist.channels.email.serve --imap-host HOST --imap-username USER --imap-password PASS --smtp-host HOST --smtp-username USER --smtp-password PASS --from-address ADDR [OPTIONS]

Examples:
    # Basic usage
    python -m EvoScientist.channels.email.serve --imap-host imap.gmail.com --imap-username bot@gmail.com --imap-password PASS --smtp-host smtp.gmail.com --smtp-username bot@gmail.com --smtp-password PASS --from-address bot@gmail.com

    # With allowed senders and custom poll interval
    python -m EvoScientist.channels.email.serve --imap-host imap.gmail.com --imap-username bot@gmail.com --imap-password PASS --smtp-host smtp.gmail.com --smtp-username bot@gmail.com --smtp-password PASS --from-address bot@gmail.com --allow user@example.com --poll-interval 60

    # With agent and thinking
    python -m EvoScientist.channels.email.serve --imap-host imap.gmail.com --imap-username bot@gmail.com --imap-password PASS --smtp-host smtp.gmail.com --smtp-username bot@gmail.com --smtp-password PASS --from-address bot@gmail.com --agent --thinking
"""

import argparse
import logging

from .channel import EmailChannel, EmailConfig
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
        description="Email channel server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--imap-host",
        required=True,
        help="IMAP server hostname",
    )
    parser.add_argument(
        "--imap-username",
        required=True,
        help="IMAP username",
    )
    parser.add_argument(
        "--imap-password",
        required=True,
        help="IMAP password",
    )
    parser.add_argument(
        "--smtp-host",
        required=True,
        help="SMTP server hostname",
    )
    parser.add_argument(
        "--smtp-username",
        required=True,
        help="SMTP username",
    )
    parser.add_argument(
        "--smtp-password",
        required=True,
        help="SMTP password",
    )
    parser.add_argument(
        "--from-address",
        required=True,
        help="From email address for outgoing messages",
    )
    parser.add_argument(
        "--allow",
        action="append",
        dest="allowed_senders",
        help="Allowed sender (email address). Can be used multiple times.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="IMAP poll interval in seconds (default: 30)",
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

    config = EmailConfig(
        imap_host=args.imap_host,
        imap_username=args.imap_username,
        imap_password=args.imap_password,
        smtp_host=args.smtp_host,
        smtp_username=args.smtp_username,
        smtp_password=args.smtp_password,
        from_address=args.from_address,
        allowed_senders=set(args.allowed_senders) if args.allowed_senders else None,
        poll_interval=args.poll_interval,
    )

    send_thinking = args.thinking and args.agent
    bus = MessageBus()
    channel = EmailChannel(config)

    run_standalone(channel, bus, use_agent=args.agent, send_thinking=send_thinking)


if __name__ == "__main__":
    main()
