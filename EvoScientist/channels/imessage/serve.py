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

import asyncio
import argparse
import logging
import signal
from typing import Callable

from . import IMessageChannel, IMessageConfig
from ..base import OutgoingMessage

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _format_todo_list(todos: list[dict]) -> str:
    """Format todo items as a numbered list."""
    lines = ["\U0001f4cb Todo List\n"]  # 📋
    for i, item in enumerate(todos, 1):
        content = item.get("content", "")
        lines.append(f"{i}. {content}")
    lines.append(f"\n\U0001f680 {len(todos)} tasks")  # 🚀
    return "\n".join(lines)


def create_agent_handler(
    on_thinking: Callable | None = None,
    on_todo: Callable | None = None,
):
    """Create handler that uses EvoScientist agent.

    Args:
        on_thinking: Optional async callback for thinking content.
            Signature: async def on_thinking(sender: str, thinking: str) -> None
        on_todo: Optional async callback for todo list updates.
            Signature: async def on_todo(sender: str, content: str, metadata: dict) -> None
    """
    from langchain_core.messages import HumanMessage
    from ...EvoScientist import create_cli_agent
    from ...stream.events import stream_agent_events

    agent = create_cli_agent()
    sessions: dict[str, str] = {}  # sender -> thread_id

    async def handler(msg) -> str:
        import uuid
        sender = msg.sender
        if sender not in sessions:
            sessions[sender] = str(uuid.uuid4())
        thread_id = sessions[sender]

        if on_thinking:
            final_content = ""
            thinking_buffer = []
            todo_sent = False
            thinking_sent = False
            _MIN_THINKING_LEN = 200  # Skip short thinking (simple conversations)

            async for event in stream_agent_events(agent, msg.content, thread_id):
                event_type = event.get("type")

                if event_type == "thinking":
                    thinking_text = event.get("content", "")
                    if thinking_text:
                        thinking_buffer.append(thinking_text)

                elif event_type == "tool_call":
                    if event.get("name") == "write_todos" and on_todo and not todo_sent:
                        todos = event.get("args", {}).get("todos", [])
                        if todos:
                            # Flush thinking before todo (only if long enough)
                            if thinking_buffer and not thinking_sent:
                                full_thinking = "".join(thinking_buffer)
                                if len(full_thinking) >= _MIN_THINKING_LEN:
                                    await on_thinking(sender, full_thinking, msg.metadata)
                                    thinking_sent = True
                                thinking_buffer.clear()
                            await on_todo(sender, _format_todo_list(todos), msg.metadata)
                            todo_sent = True

                elif event_type == "text":
                    final_content += event.get("content", "")

                elif event_type == "done":
                    final_content = event.get("content", "") or final_content

            if thinking_buffer and not thinking_sent:
                full_thinking = "".join(thinking_buffer)
                if len(full_thinking) >= _MIN_THINKING_LEN:
                    await on_thinking(sender, full_thinking, msg.metadata)
                    thinking_sent = True

            return final_content or "No response"
        else:
            config = {"configurable": {"thread_id": thread_id}}
            result = agent.invoke(
                {"messages": [HumanMessage(content=msg.content)]},
                config=config,
            )
            messages = result.get("messages", [])
            for m in reversed(messages):
                if hasattr(m, "content") and m.type == "ai":
                    content = m.content
                    # Handle structured content (thinking mode)
                    if isinstance(content, list):
                        text_parts = []
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text_parts.append(block.get("text", ""))
                        return "\n".join(text_parts) if text_parts else "No response"
                    # Handle plain string content
                    return content
            return "No response"

    return handler


class IMessageServer:
    """Server that runs the iMessage channel and handles messages."""

    def __init__(
        self,
        config: IMessageConfig,
        handler: Callable | None = None,
        send_thinking: bool = False,
        initial_debounce: float = 2.0,
        debounce_step: float = 0.5,
        max_debounce: float = 5.0,
    ):
        """Initialize iMessage server.

        Args:
            config: iMessage channel configuration.
            handler: Message handler function. If None, uses echo handler.
            send_thinking: If True, send thinking content as intermediate messages.
            initial_debounce: Wait time after first message (seconds).
            debounce_step: Additional wait per subsequent message.
            max_debounce: Maximum debounce window cap.
        """
        self.config = config
        self.channel = IMessageChannel(config)
        self.send_thinking = send_thinking
        self.initial_debounce = initial_debounce
        self.debounce_step = debounce_step
        self.max_debounce = max_debounce
        self._running = False
        self._pending_thinking: dict[str, str] = {}  # sender -> accumulated thinking

        # Message buffering for debounce
        self._message_buffers: dict[str, list[str]] = {}  # sender -> [messages]
        self._message_metadata: dict[str, dict] = {}  # sender -> metadata (from first message)
        self._debounce_tasks: dict[str, asyncio.Task] = {}  # sender -> pending task
        self._processing: set[str] = set()  # senders currently being processed

        if handler:
            self.handler = handler
        else:
            self.handler = self._default_handler

    async def _default_handler(self, msg) -> str:
        """Default echo handler."""
        return f"Echo: {msg.content}"

    async def _process_buffered_messages(self, sender: str) -> None:
        """Process all buffered messages for a sender.

        If the sender is currently being processed, skip — new messages
        stay in the buffer and will be picked up after current processing.
        """
        # Don't start a new handler if one is already running for this sender
        if sender in self._processing:
            logger.debug(f"Agent busy for {sender}, messages stay queued")
            return

        if sender not in self._message_buffers:
            return

        messages = self._message_buffers.pop(sender, [])
        metadata = self._message_metadata.pop(sender, None)
        self._debounce_tasks.pop(sender, None)

        if not messages:
            return

        merged_content = "\n".join(messages)
        logger.info(f"Processing {len(messages)} merged message(s) from {sender}")

        self._processing.add(sender)
        try:
            class MergedMessage:
                def __init__(self, s, c, m):
                    self.sender = s
                    self.content = c
                    self.metadata = m

            merged_msg = MergedMessage(sender, merged_content, metadata)
            response = await self.handler(merged_msg)

            if response:
                await self.channel.send(OutgoingMessage(
                    recipient=sender,
                    content=response,
                    metadata=metadata,
                ))
        except Exception as e:
            logger.error(f"Handler error: {e}")
        finally:
            self._processing.discard(sender)

            # If new messages arrived during processing, restart debounce
            if sender in self._message_buffers and self._message_buffers[sender]:
                msg_count = len(self._message_buffers[sender])
                wait = min(
                    self.initial_debounce + (msg_count - 1) * self.debounce_step,
                    self.max_debounce,
                )
                logger.info(f"New messages queued for {sender}, restarting debounce ({wait:.1f}s)")

                async def restart_debounce(_s=sender, _w=wait):
                    await asyncio.sleep(_w)
                    await self._process_buffered_messages(_s)

                self._debounce_tasks[sender] = asyncio.create_task(restart_debounce())

    async def _queue_message(self, msg) -> None:
        """Queue a message with progressive debounce.

        If agent is busy, just buffer — messages will be picked up
        after current processing finishes. Otherwise, start debounce:
        1st: 2.0s, 2nd: 2.5s, 3rd: 3.0s, ... up to max_debounce.
        """
        sender = msg.sender

        if sender not in self._message_buffers:
            self._message_buffers[sender] = []
            self._message_metadata[sender] = msg.metadata
        self._message_buffers[sender].append(msg.content)

        # Agent is busy — just buffer, no debounce needed
        if sender in self._processing:
            logger.debug(f"Agent busy for {sender}, buffering message #{len(self._message_buffers[sender])}")
            return

        if sender in self._debounce_tasks:
            self._debounce_tasks[sender].cancel()

        msg_count = len(self._message_buffers[sender])
        wait = min(
            self.initial_debounce + (msg_count - 1) * self.debounce_step,
            self.max_debounce,
        )
        logger.debug(f"Debounce for {sender}: {wait:.1f}s (message #{msg_count})")

        async def debounce_callback(_s=sender, _w=wait):
            await asyncio.sleep(_w)
            await self._process_buffered_messages(_s)

        self._debounce_tasks[sender] = asyncio.create_task(debounce_callback())

    async def send_todo_message(self, sender: str, content: str, metadata: dict | None = None) -> None:
        """Send todo list as intermediate message."""
        logger.debug(f"Sending todo list to {sender}")
        await self.channel.send(OutgoingMessage(
            recipient=sender,
            content=content,
            metadata=metadata or {},
        ))

    async def send_thinking_message(self, sender: str, thinking: str, metadata: dict | None = None) -> None:
        """Send thinking content as intermediate message."""
        if not self.send_thinking:
            return

        logger.debug(f"Sending thinking to {sender} with metadata: {metadata}")
        content = f"\U0001f9e0\n{thinking}\n\u23f3"
        await self.channel.send(OutgoingMessage(
            recipient=sender,
            content=content,
            metadata=metadata or {},
        ))
        logger.debug(f"Sent thinking to {sender}: {thinking[:50]}...")

    async def run(self) -> None:
        """Run the server."""
        await self.channel.start()
        self._running = True

        logger.info("iMessage server running. Press Ctrl+C to stop.")
        if self.config.allowed_senders:
            logger.info(f"Allowed senders: {self.config.allowed_senders}")
        else:
            logger.info("Allowing all senders")
        logger.info(f"Debounce: {self.initial_debounce}s + {self.debounce_step}s/msg (max {self.max_debounce}s)")

        try:
            async for msg in self.channel.receive():
                logger.info(f"From {msg.sender}: {msg.content[:50]}...")
                await self._queue_message(msg)
        finally:
            for task in self._debounce_tasks.values():
                task.cancel()
            await self.channel.stop()

    async def stop(self) -> None:
        """Stop the server."""
        self._running = False
        await self.channel.stop()


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


async def async_main():
    """Async entry point."""
    args = parse_args()

    config = IMessageConfig(
        cli_path=args.cli_path,
        db_path=args.db_path,
        allowed_senders=set(args.allowed_senders) if args.allowed_senders else None,
        include_attachments=args.attachments,
    )

    handler = None
    send_thinking = args.thinking and args.agent

    if args.agent:
        logger.info("Loading EvoScientist agent...")
        logger.info("Agent loaded")

    server = IMessageServer(
        config,
        handler=None,
        send_thinking=send_thinking,
    )

    if args.agent:
        on_thinking = server.send_thinking_message if send_thinking else None
        on_todo = server.send_todo_message
        handler = create_agent_handler(on_thinking=on_thinking, on_todo=on_todo)
        server.handler = handler
        if send_thinking:
            logger.info("Thinking messages enabled")

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(server.stop()))

    await server.run()


def main():
    """Entry point."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
