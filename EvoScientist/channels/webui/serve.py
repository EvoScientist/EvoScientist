"""Standalone runner for the Web UI channel."""

import argparse
import logging

from ..bus import MessageBus
from ..standalone import run_standalone
from .channel import WebUIChannel, WebUIConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Web UI channel server")
    parser.add_argument(
        "--port",
        type=int,
        default=8010,
        help="Port for the Assistant Transport endpoint",
    )
    parser.add_argument(
        "--host",
        "--bind-host",
        dest="bind_host",
        default="127.0.0.1",
        help="Host/interface to bind (default: 127.0.0.1; use 0.0.0.0 for LAN)",
    )
    parser.add_argument(
        "--base-path",
        default="/webui",
        help="Base route prefix for the channel",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Optional API key accepted via Authorization: Bearer or X-API-Key",
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Use the EvoScientist agent instead of running as a passive channel",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    bus = MessageBus()
    channel = WebUIChannel(
        WebUIConfig(
            bind_host=args.bind_host,
            webhook_port=args.port,
            base_path=args.base_path,
            api_key=args.api_key,
        )
    )
    run_standalone(channel, bus, use_agent=args.agent, send_thinking=False)


if __name__ == "__main__":
    main()
