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


class _RedactingFormatter(logging.Formatter):
    """Formats like the wrapped formatter, with the bot token blanked out."""

    def __init__(self, inner: logging.Formatter, token: str) -> None:
        super().__init__()
        self._inner = inner
        self._token = token

    def format(self, record: logging.LogRecord) -> str:
        # the whole formatted text, so URLs inside exception messages and tracebacks are covered too
        return self._inner.format(record).replace(self._token, "<bot-token>")


def protect_bot_token(token: str) -> None:
    """Keep the bot token out of the log (#567).

    The Bot API puts it in the URL path (``/bot<token>/getUpdates``), and httpx logs
    every request URL at INFO. Quiet those lines like ``EvoSci serve`` does, and blank
    the token in whatever else gets logged, such as an error that quotes the URL.
    """
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if not token:
        return
    for handler in logging.getLogger().handlers:
        formatter = handler.formatter or logging.Formatter(logging.BASIC_FORMAT)
        if not isinstance(formatter, _RedactingFormatter):
            handler.setFormatter(_RedactingFormatter(formatter, token))


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
    protect_bot_token(args.bot_token)

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
