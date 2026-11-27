"""Local stand-in for the signal-cli daemon: chat with ``EvoSci serve`` from a terminal.

For hosts where no messaging channel can reach an outside service. The Signal
channel talks newline-delimited JSON-RPC to ``localhost:<signal_rpc_port>`` and
uses a daemon that already listens there instead of starting signal-cli. This
script is that daemon: what you type goes to EvoScientist as a Signal direct
message from ``--sender``, and what EvoScientist sends (replies and proactive
pushes) is printed here. No outside host is used. Every message is shown under
a header with the wall-clock time it was sent or received.

The config needs ``channel_enabled: signal``, any ``signal_phone_number``,
``signal_allowed_senders`` set to ``--sender`` and ``signal_rpc_port`` set to
``--port``. Start this script before ``EvoSci serve``.

Run:   uv run python scripts/signal_standin.py --port 17590 --log chat.jsonl
Type:  one line and Enter      -> sent as one message
       /file <path>            -> the file's whole text sent as one message
       /clear                  -> wipe the screen and scrollback
       /quit                   -> stop
``--log`` appends every message, both ways, as one JSON line.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

REPLY_METHODS_QUIET = {"sendTyping", "sendReaction"}


class Daemon:
    def __init__(self, sender: str, log_path: Path | None, stamp_format: str) -> None:
        self.sender = sender
        self.log_path = log_path
        self.stamp_format = stamp_format
        self.writers: set[asyncio.StreamWriter] = set()

    def show(self, who: str, text: str) -> None:
        stamp = datetime.now().strftime(self.stamp_format)
        print(f"\n--- {who} {stamp} ---\n{text}\n---", flush=True)

    def record(self, direction: str, text: str, **extra: object) -> None:
        if self.log_path is None:
            return
        row = {
            "time": datetime.now().astimezone().isoformat(timespec="seconds"),
            "direction": direction,
            "text": text,
            **extra,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    async def handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self.writers.add(writer)
        try:
            while line := await reader.readline():
                try:
                    request = json.loads(line.decode())
                except json.JSONDecodeError:
                    continue
                method = request.get("method", "")
                params = request.get("params", {})
                if method == "send":
                    text = params.get("message", "")
                    self.show("EvoScientist", text)
                    self.record("out", text, attachments=params.get("attachments"))
                elif method not in REPLY_METHODS_QUIET:
                    print(f"[rpc {method}]", flush=True)
                if "id" in request:
                    reply = {
                        "jsonrpc": "2.0",
                        "id": request["id"],
                        "result": {"timestamp": int(time.time() * 1000)},
                    }
                    writer.write((json.dumps(reply) + "\n").encode())
                    await writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            self.writers.discard(writer)
            writer.close()

    async def deliver(self, text: str) -> None:
        now = int(time.time() * 1000)
        note = {
            "jsonrpc": "2.0",
            "method": "receive",
            "params": {
                "envelope": {
                    "source": self.sender,
                    "sourceNumber": self.sender,
                    "sourceName": "researcher",
                    "timestamp": now,
                    "dataMessage": {"message": text, "timestamp": now},
                }
            },
        }
        data = (json.dumps(note) + "\n").encode()
        if not self.writers:
            print("[not sent: EvoScientist is not connected]", flush=True)
            return
        for writer in list(self.writers):
            writer.write(data)
            await writer.drain()
        self.record("in", text)

    async def read_input(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                # stdin closed: keep serving so pushes are still printed
                await asyncio.Event().wait()
            text = line.rstrip("\n")
            if not text.strip():
                continue
            if text.strip() == "/quit":
                return
            if text.strip() == "/clear":
                # Wipe the screen and scrollback before the recording starts.
                print("\x1b[3J\x1b[H\x1b[2J", end="", flush=True)
                continue
            typed = text
            text = text.lstrip()
            if text.startswith("/file "):
                path = Path(text[len("/file ") :].strip()).expanduser()
                try:
                    text = path.read_text(encoding="utf-8").strip()
                except OSError as exc:
                    print(f"[not sent: {exc}]", flush=True)
                    continue
            if sys.stdout.isatty():
                # Replace the typed line by the stamped message block.
                columns = shutil.get_terminal_size().columns
                rows = max(1, -(-len(typed) // columns))
                print(f"\x1b[{rows}A\x1b[J", end="")
            self.show("Researcher", text)
            await self.deliver(text)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int, default=7583)
    parser.add_argument("--sender", default="+15550001")
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument(
        "--stamp-format",
        default="%a %d %b %H:%M:%S",
        help="strftime format of the time shown above every message",
    )
    args = parser.parse_args()

    daemon = Daemon(args.sender, args.log, args.stamp_format)
    server = await asyncio.start_server(daemon.handle, "localhost", args.port)
    print(
        f"Listening on localhost:{args.port} as sender {args.sender}. "
        "Type a message, /file <path>, or /quit.",
        flush=True,
    )
    async with server:
        await daemon.read_input()


if __name__ == "__main__":
    asyncio.run(main())
