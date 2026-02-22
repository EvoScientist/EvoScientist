"""Unified inbound message consumer.

Provides :class:`InboundConsumer` — a single class that consumes
inbound messages from the :class:`MessageBus`, runs them through
the agent, and publishes outbound responses.  This replaces the
inline consumer loops that were duplicated in ``cli.py`` and
``standalone.py``.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, TypeVar

from .base import Channel
from .bus import MessageBus
from .bus.events import InboundMessage, OutboundMessage

logger = logging.getLogger(__name__)

T = TypeVar("T")

_MAX_CHAT_LOCKS = 10_000
_MAX_SESSIONS = 10_000


@dataclass
class ConsumerMetrics:
    """Cumulative processing counters for the consumer."""

    total_processed: int = 0
    total_successes: int = 0
    total_failures: int = 0
    total_timeouts: int = 0


async def _timeout_aiter(
    agen: AsyncIterator[T],
    idle_timeout: float,
) -> AsyncIterator[T]:
    """Wrap an async iterator with a per-yield idle timeout.

    If ``__anext__()`` does not produce a value within *idle_timeout*
    seconds, :class:`asyncio.TimeoutError` is raised.  Continuous
    yielding resets the timer each time, so only a truly stalled
    generator will trigger the timeout.
    """
    ait = agen.__aiter__()
    try:
        while True:
            try:
                item = await asyncio.wait_for(ait.__anext__(), timeout=idle_timeout)
            except StopAsyncIteration:
                return
            yield item
    finally:
        if hasattr(ait, "aclose"):
            await ait.aclose()


def _format_todo_list(todos: list[dict]) -> str:
    """Format todo items as a numbered list."""
    lines = ["\U0001f4cb Todo List\n"]  # 📋
    for i, item in enumerate(todos, 1):
        content = item.get("content", "")
        lines.append(f"{i}. {content}")
    lines.append(f"\n\U0001f680 {len(todos)} tasks")  # 🚀
    return "\n".join(lines)


class InboundConsumer:
    """Consume inbound messages from the bus, process via agent, publish outbound.

    Parameters
    ----------
    bus:
        The MessageBus to consume from / publish to.
    manager:
        The ChannelManager (used to look up channel instances).
    agent:
        The agent object (must support ``stream_agent_events``).
    thread_id:
        Default thread ID for agent conversations.
    send_thinking:
        Whether to forward thinking messages to the channel.
    on_message_received:
        Optional callback ``(msg: InboundMessage) -> None`` invoked when
        a message is consumed (e.g. for CLI Rich display).
    on_streaming_event:
        Optional callback ``(event: dict) -> None`` invoked for each
        streaming event from the agent.
    on_message_sent:
        Optional callback ``(msg: OutboundMessage) -> None`` invoked when
        the outbound message is published.
    inference_timeout:
        Per-yield idle timeout in seconds for the agent stream.  If the
        agent produces no event for this long, the inference is aborted.
    max_concurrent:
        Number of worker coroutines (= max parallel inferences).
    max_pending:
        Maximum depth of the internal work queue.  When full, the
        consumer loop blocks (back-pressure).
    drain_timeout:
        Seconds to wait for in-flight workers to finish during ``stop()``.
    """

    def __init__(
        self,
        bus: MessageBus,
        manager: Any,
        agent: Any,
        thread_id: str,
        *,
        send_thinking: bool = False,
        on_message_received: Callable[[InboundMessage], None] | None = None,
        on_streaming_event: Callable[[dict], None] | None = None,
        on_message_sent: Callable[[OutboundMessage], None] | None = None,
        inference_timeout: float = 300.0,
        max_concurrent: int = 5,
        max_pending: int = 50,
        drain_timeout: float = 30.0,
    ):
        self.bus = bus
        self.manager = manager
        self.agent = agent
        self.thread_id = thread_id
        self.send_thinking = send_thinking
        self._on_message_received = on_message_received
        self._on_streaming_event = on_streaming_event
        self._on_message_sent = on_message_sent
        self._sessions: OrderedDict[str, str] = OrderedDict()  # sender_id -> thread_id (LRU)

        # Per-chat locks: same chat is processed serially (bounded)
        self._chat_locks: dict[str, asyncio.Lock] = {}

        # Inference timeout
        self._inference_timeout = inference_timeout

        # Worker pool
        self._max_concurrent = max_concurrent
        self._work_queue: asyncio.Queue[InboundMessage | None] = asyncio.Queue(
            maxsize=max_pending,
        )
        self._workers: list[asyncio.Task] = []
        self._stopping = False
        self._drain_timeout = drain_timeout

        # Metrics
        self._metrics = ConsumerMetrics()

    def _get_thread_id(self, sender_id: str) -> str:
        """Get or create a thread ID for the given sender.

        Uses LRU ordering: recently accessed senders are moved to the
        end, so eviction always removes the least-recently-active sender.
        """
        if sender_id in self._sessions:
            self._sessions.move_to_end(sender_id)
            return self._sessions[sender_id]

        if len(self._sessions) >= _MAX_SESSIONS:
            # Evict the least-recently-used entry
            self._sessions.popitem(last=False)
        if self.thread_id:
            self._sessions[sender_id] = f"{self.thread_id}:{sender_id}"
        else:
            self._sessions[sender_id] = str(uuid.uuid4())
        return self._sessions[sender_id]

    def _get_channel(self, channel_name: str) -> Channel | None:
        """Look up the channel by name from the manager."""
        return self.manager.get_channel(channel_name)

    # ── lifecycle ──

    async def run(self) -> None:
        """Main consumer loop — runs until ``stop()`` or cancellation.

        Spawns *max_concurrent* worker coroutines that pull from an
        internal bounded queue.  The loop reads from the bus and feeds
        the queue; when the queue is full the loop blocks (back-pressure).
        """
        self._stopping = False
        self._workers = [
            asyncio.create_task(self._worker(i))
            for i in range(self._max_concurrent)
        ]
        try:
            while not self._stopping:
                try:
                    msg = await asyncio.wait_for(
                        self.bus.consume_inbound(), timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break
                if self._stopping:
                    break
                await self._work_queue.put(msg)  # blocks when full (back-pressure)
        finally:
            if not self._stopping:
                await self.stop()

    async def stop(self) -> None:
        """Gracefully drain in-flight work and shut down workers."""
        self._stopping = True
        logger.info("Consumer stopping: draining in-flight messages...")
        pending_count = self._work_queue.qsize()

        # Send a None sentinel per worker so each exits its loop
        for _ in self._workers:
            try:
                self._work_queue.put_nowait(None)
            except asyncio.QueueFull:
                pass

        # Wait for workers to finish, then force-cancel stragglers
        if self._workers:
            done, still_running = await asyncio.wait(
                self._workers, timeout=self._drain_timeout,
            )
            for task in still_running:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            logger.info(
                f"Consumer drain: {len(done)} finished, "
                f"{len(still_running)} force-cancelled, "
                f"{pending_count} were pending"
            )
        self._workers.clear()

    # ── workers ──

    async def _worker(self, worker_id: int) -> None:
        """Pull messages from the work queue and process them."""
        while True:
            msg = await self._work_queue.get()
            if msg is None:
                break  # shutdown sentinel
            try:
                await self._handle_message(msg)
            except Exception:
                logger.exception(f"Worker {worker_id} unhandled error")
            finally:
                self._work_queue.task_done()

    async def _handle_message(self, msg: InboundMessage) -> None:
        """Process a single inbound message."""
        from ..stream.events import stream_agent_events

        if self._on_message_received:
            try:
                self._on_message_received(msg)
            except Exception:
                pass

        channel = self._get_channel(msg.channel)
        thread_id = self._get_thread_id(msg.sender_id)
        session_key = msg.session_key  # "channel:chat_id"

        # Lazily create per-chat lock; evict stale locks when too many
        if session_key not in self._chat_locks:
            self._chat_locks[session_key] = asyncio.Lock()
            if len(self._chat_locks) > _MAX_CHAT_LOCKS:
                self._evict_chat_locks()

        self._metrics.total_processed += 1

        async with self._chat_locks[session_key]:
            try:
                final_content = ""
                thinking_buffer: list[str] = []
                todo_sent = False
                thinking_sent = False

                if channel:
                    await channel.start_typing(msg.chat_id)

                async for event in _timeout_aiter(
                    stream_agent_events(self.agent, msg.content, thread_id, media=msg.media or None),
                    self._inference_timeout,
                ):
                    event_type = event.get("type")

                    if self._on_streaming_event:
                        try:
                            self._on_streaming_event(event)
                        except Exception:
                            pass

                    if event_type == "thinking":
                        thinking_text = event.get("content", "")
                        if thinking_text:
                            thinking_buffer.append(thinking_text)

                    elif event_type == "tool_call":
                        if event.get("name") == "write_todos" and not todo_sent:
                            todos = event.get("args", {}).get("todos", [])
                            if todos and channel:
                                if thinking_buffer and not thinking_sent:
                                    full_thinking = "".join(thinking_buffer)
                                    if full_thinking:
                                        await channel.send_thinking_message(
                                            msg.sender_id,
                                            full_thinking,
                                            msg.metadata,
                                        )
                                        thinking_sent = True
                                    thinking_buffer.clear()
                                await channel.send_todo_message(
                                    msg.sender_id,
                                    _format_todo_list(todos),
                                    msg.metadata,
                                )
                                todo_sent = True

                    elif event_type == "text":
                        final_content += event.get("content", "")

                    elif event_type == "done":
                        final_content = event.get("content", "") or final_content

                if thinking_buffer and not thinking_sent and channel:
                    full_thinking = "".join(thinking_buffer)
                    if full_thinking:
                        await channel.send_thinking_message(
                            msg.sender_id, full_thinking, msg.metadata,
                        )

                outbound = OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=final_content or "No response",
                    reply_to=msg.message_id or None,
                    metadata=msg.metadata,
                )
                await self.bus.publish_outbound(outbound)

                self._metrics.total_successes += 1

                if self._on_message_sent:
                    try:
                        self._on_message_sent(outbound)
                    except Exception:
                        pass

            except asyncio.TimeoutError:
                self._metrics.total_timeouts += 1
                logger.error(
                    f"Inference timeout ({self._inference_timeout}s idle) "
                    f"for {msg.sender_id} in {session_key}"
                )
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Sorry, the response timed out. Please try again.",
                    metadata=msg.metadata,
                ))

            except Exception as e:
                self._metrics.total_failures += 1
                logger.error(f"Agent error: {e}")
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Sorry, something went wrong. Please try again later.",
                    metadata=msg.metadata,
                ))
            finally:
                if channel:
                    await channel.stop_typing(msg.chat_id)

    # ── observability ──

    @property
    def pending_count(self) -> int:
        """Number of messages waiting in the work queue."""
        return self._work_queue.qsize()

    @property
    def active_workers(self) -> int:
        """Number of worker tasks that are still alive."""
        return sum(1 for w in self._workers if not w.done())

    @property
    def metrics(self) -> dict[str, int]:
        """Cumulative processing counters."""
        m = self._metrics
        return {
            "total_processed": m.total_processed,
            "total_successes": m.total_successes,
            "total_failures": m.total_failures,
            "total_timeouts": m.total_timeouts,
            "pending": self.pending_count,
            "active_workers": self.active_workers,
            "chat_locks": len(self._chat_locks),
            "sessions": len(self._sessions),
        }

    # ── internal ──

    def _evict_chat_locks(self) -> None:
        """Remove chat locks that are not currently held."""
        stale = [k for k, lock in self._chat_locks.items() if not lock.locked()]
        for k in stale[:max(1, len(stale) // 2)]:
            del self._chat_locks[k]
