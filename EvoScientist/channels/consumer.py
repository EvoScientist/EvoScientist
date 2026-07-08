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
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from ..gateway import GraphGateway, GraphRunInput, GraphTarget, RunRequest
from .base import Channel
from .bus import MessageBus
from .bus.events import InboundMessage, OutboundMessage
from .capabilities import ChannelCapabilities
from .interaction import (
    ASK_USER_TIMEOUT,
    HITL_APPROVAL_TIMEOUT,
    REJECTED_FEEDBACK,
    ApprovalPolicy,
    InteractionIO,
    PendingReplyRegistry,
    resolve_approval,
    resolve_ask_user,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

_MAX_CHAT_LOCKS = 10_000
_MAX_SESSIONS = 10_000
_MAX_HITL_ROUNDS = 50


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


def _join_subagent_text(buffers: dict[str, tuple[str, list[str]]]) -> str:
    """Join sub-agent text buffers into a single fallback string.

    *buffers* maps ``instance_id`` → ``(display_name, chunks)``.

    When only one instance produced text, return its content directly.
    When multiple instances share the same display name, number them
    (e.g. ``[research-agent #1]``, ``[research-agent #2]``).
    """
    if not buffers:
        return ""
    if len(buffers) == 1:
        _display_name, chunks = next(iter(buffers.values()))
        return "".join(chunks)

    # Group by display_name to detect same-name instances
    name_groups: dict[str, list[list[str]]] = {}
    for _instance_id, (display_name, chunks) in buffers.items():
        name_groups.setdefault(display_name, []).append(chunks)

    sections: list[str] = []
    for display_name, chunk_lists in name_groups.items():
        if len(chunk_lists) == 1:
            sections.append(f"[{display_name}]: {''.join(chunk_lists[0])}")
        else:
            for i, chs in enumerate(chunk_lists, 1):
                sections.append(f"[{display_name} #{i}]: {''.join(chs)}")
    return "\n\n".join(sections)


class _ConsumerIO(InteractionIO):
    """:class:`InteractionIO` over the consumer's bus + reply registry.

    Publishes prompts through ``bus.publish_outbound`` and blocks for
    replies on the consumer's shared :class:`PendingReplyRegistry` — both
    on the consumer's own event loop, so the engine runs natively async
    here with no thread hand-off.
    """

    def __init__(
        self, consumer: InboundConsumer, msg: InboundMessage, session_key: str
    ) -> None:
        self._consumer = consumer
        self._msg = msg
        self._session_key = session_key
        self._last_reply_message: InboundMessage | None = None
        channel = consumer._get_channel(msg.channel)
        self.capabilities = (
            channel.capabilities if channel is not None else ChannelCapabilities()
        )
        self.base_metadata = msg.metadata

    async def send(self, content: str, *, metadata: dict | None = None) -> bool:
        await self._consumer.bus.publish_outbound(
            OutboundMessage(
                channel=self._msg.channel,
                chat_id=self._msg.chat_id,
                content=content,
                metadata=metadata if metadata is not None else self._msg.metadata,
            )
        )
        return True

    async def wait_reply(self, *, timeout: float) -> str | None:
        reply = await self._consumer._reply_registry.wait_event(
            self._session_key, timeout
        )
        if reply is None:
            self._last_reply_message = None
            return None
        self._last_reply_message = (
            reply.context if isinstance(reply.context, InboundMessage) else None
        )
        return reply.content

    def take_reply_context(self) -> InboundMessage | None:
        """Consume the last inbound reply context captured by ``wait_reply``."""
        msg = self._last_reply_message
        self._last_reply_message = None
        return msg


class InboundConsumer:
    """Consume inbound messages from the bus, process via agent, publish outbound.

    Parameters
    ----------
    bus:
        The MessageBus to consume from / publish to.
    manager:
        The ChannelManager (used to look up channel instances).
    agent:
        The local agent object used by local graph gateway targets.
    thread_id:
        Default thread ID for agent conversations.
    graph_gateway:
        Gateway used for thread creation and graph streaming.
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
        graph_gateway: GraphGateway,
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
        self.graph_gateway = graph_gateway
        self.send_thinking = send_thinking
        self._on_message_received = on_message_received
        self._on_streaming_event = on_streaming_event
        self._on_message_sent = on_message_sent
        self._sessions: OrderedDict[str, str] = (
            OrderedDict()
        )  # sender_id -> thread_id (LRU)

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

        # Interaction engine state: one reply registry (routes the next
        # message from a chat into a waiting prompt) and one approval
        # policy (config rule + session "Approve all" grants), shared by
        # the ask_user and HITL flows via ``channels.interaction``.
        self._reply_registry = PendingReplyRegistry()
        self._approval_policy = ApprovalPolicy()

    async def _get_thread_id(self, sender_id: str) -> str:
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
            self._sessions[sender_id] = await self.graph_gateway.create_thread(
                GraphTarget(local_graph=self.agent)
            )
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
            asyncio.create_task(self._worker(i)) for i in range(self._max_concurrent)
        ]
        try:
            while not self._stopping:
                try:
                    msg = await asyncio.wait_for(
                        self.bus.consume_inbound(),
                        timeout=1.0,
                    )
                except TimeoutError:
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
                self._workers,
                timeout=self._drain_timeout,
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
        if self._on_message_received:
            try:
                self._on_message_received(msg)
            except Exception:
                pass

        session_key = msg.session_key  # "channel:chat_id"

        # Lazily create per-chat lock; evict stale locks when too many
        if session_key not in self._chat_locks:
            self._chat_locks[session_key] = asyncio.Lock()
            if len(self._chat_locks) > _MAX_CHAT_LOCKS:
                self._evict_chat_locks()

        self._metrics.total_processed += 1

        # Reply interception: if a prompt (ask_user question or HITL
        # approval) is waiting on this chat, hand it this message instead
        # of starting a fresh agent turn.  The engine parses it (stop /
        # cancel / choice / approval grammar), so the registry only routes
        # text plus the original inbound context — one path for both flows.
        if self._reply_registry.try_resolve(session_key, msg.content, context=msg):
            return

        # Resolved only for real agent turns — a consumed prompt reply must
        # not create a graph thread or touch the sender-session LRU.
        channel = self._get_channel(msg.channel)
        thread_id = await self._get_thread_id(msg.sender_id)

        async with self._chat_locks[session_key]:
            refeed = await self._stream_with_hitl(msg, channel, thread_id, session_key)

        # An unrecognized reply to a pending approval rejects the action and
        # then becomes a new agent turn. The lock was released above, so the
        # previous turn has fully unwound before the refeed turn acquires it.
        # Loops in case the refeed turn hits another approval that is again
        # answered with unparseable text.
        while refeed is not None:
            channel = self._get_channel(refeed.channel)
            thread_id = await self._get_thread_id(refeed.sender_id)
            session_key = refeed.session_key
            if session_key not in self._chat_locks:
                self._chat_locks[session_key] = asyncio.Lock()
                if len(self._chat_locks) > _MAX_CHAT_LOCKS:
                    self._evict_chat_locks()
            async with self._chat_locks[session_key]:
                refeed = await self._stream_with_hitl(
                    refeed, channel, thread_id, session_key
                )

    async def _stream_with_hitl(
        self,
        msg: InboundMessage,
        channel: Channel | None,
        thread_id: str,
        session_key: str,
    ) -> InboundMessage | None:
        """Stream agent events with HITL interrupt handling.

        Returns ``None`` normally.  When a pending approval is answered
        with unrecognized text, returns the intercepted inbound reply so the
        caller can refeed it as a new agent turn after this one unwinds.
        """
        from langgraph.types import Command

        stream_input: GraphRunInput = msg.content

        try:
            if channel:
                await channel.start_typing(msg.chat_id)

            _last_sent_thinking: str | None = None

            for _hitl_round in range(_MAX_HITL_ROUNDS):
                final_content = ""
                thinking_buffer: list[str] = []
                todo_sent = False
                subagent_text_buffers: dict[str, tuple[str, list[str]]] = {}
                thinking_sent = False
                interrupt_data: dict | None = None

                async def _flush_thinking_buffer(
                    buffer: list[str] = thinking_buffer,
                ) -> bool:
                    """Send the current thinking buffer, dedup by content."""
                    nonlocal thinking_sent, _last_sent_thinking
                    if not channel or thinking_sent or not buffer:
                        return False

                    full_thinking = "".join(buffer).rstrip()
                    buffer.clear()
                    if not full_thinking or full_thinking == _last_sent_thinking:
                        return False

                    await channel.send_thinking_message(
                        msg.sender_id,
                        full_thinking,
                        msg.metadata,
                    )
                    thinking_sent = True
                    _last_sent_thinking = full_thinking
                    return True

                async for event in _timeout_aiter(
                    self.graph_gateway.stream_events(
                        RunRequest(
                            message=stream_input,
                            thread_id=thread_id,
                            media=msg.media or None
                            if isinstance(stream_input, str)
                            else None,
                            target=GraphTarget(local_graph=self.agent),
                        )
                    ),
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
                                await _flush_thinking_buffer()
                                await channel.send_todo_message(
                                    msg.sender_id,
                                    _format_todo_list(todos),
                                    msg.metadata,
                                )
                                todo_sent = True

                    elif event_type == "text":
                        final_content += event.get("content", "")

                    elif event_type == "subagent_text":
                        sa_name = event.get("subagent", "unknown")
                        instance_id = event.get("instance_id")
                        if not instance_id:
                            continue
                        if instance_id not in subagent_text_buffers:
                            subagent_text_buffers[instance_id] = (sa_name, [])
                        subagent_text_buffers[instance_id][1].append(
                            event.get("content", "")
                        )

                    elif event_type == "done":
                        final_content = event.get("content", "") or final_content

                    elif event_type == "interrupt":
                        interrupt_data = event
                        break  # exit async for to handle interrupt

                    elif event_type == "ask_user":
                        interrupt_data = event
                        break  # exit async for to handle ask_user

                # Flush thinking
                await _flush_thinking_buffer()

                # No interrupt — normal completion
                if interrupt_data is None:
                    outbound = OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=final_content
                        or _join_subagent_text(subagent_text_buffers)
                        or "No response",
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
                    return  # done

                # ask_user: send questions to channel user, collect answers
                if interrupt_data.get("type") == "ask_user":
                    result = await self._resolve_ask_user(
                        msg,
                        interrupt_data,
                        session_key,
                    )

                    stream_input = Command(resume=result)
                    continue

                # HITL: resolve the interrupt through the shared engine.
                # ``resolve_approval`` handles session/config auto-approve,
                # the approval prompt (with capability-driven buttons), the
                # reply wait, parsing (incl. /stop), and feedback strings.
                action_reqs = interrupt_data.get("action_requests", [])
                io = _ConsumerIO(self, msg, session_key)
                outcome = await resolve_approval(
                    action_reqs,
                    io,
                    self._approval_policy,
                    session_key,
                    timeout=HITL_APPROVAL_TIMEOUT,
                )
                if outcome.unrecognized_reply is not None:
                    # Serve-mode policy: an unrecognized reply rejects the
                    # pending action, confirms with reject feedback, and is
                    # then processed as a new agent turn. The refeed is
                    # returned to ``_handle_message`` so chat-lock ordering
                    # stays serialized.
                    await io.send(REJECTED_FEEDBACK)
                    # In this flow, the final wait_reply call is exactly the
                    # unrecognized approval reply. ask_user does not read this.
                    refeed_msg = io.take_reply_context()
                    if refeed_msg is None:
                        logger.warning(
                            "Unrecognized approval reply had no inbound context; "
                            "dropping refeed"
                        )
                    return refeed_msg
                if outcome.decisions is None:
                    return None  # reject / timeout / stop — end the turn

                stream_input = Command(resume={"decisions": outcome.decisions})
                # continue to next HITL round

        except TimeoutError:
            self._metrics.total_timeouts += 1
            logger.error(
                f"Inference timeout ({self._inference_timeout}s idle) "
                f"for {msg.sender_id} in {session_key}"
            )
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Sorry, the response timed out. Please try again.",
                    metadata=msg.metadata,
                )
            )

        except Exception as e:
            self._metrics.total_failures += 1
            logger.error(f"Agent error: {e}")
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Sorry, something went wrong. Please try again later.",
                    metadata=msg.metadata,
                )
            )
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

    # ── ask_user helpers ──

    async def _resolve_ask_user(
        self,
        msg: InboundMessage,
        event_data: dict,
        session_key: str,
    ) -> dict:
        """Handle an ask_user interrupt via the shared engine.

        Delegates the whole question/answer flow (prompt formatting, choice
        + "Other" grammar, ``/stop`` handling) to
        :func:`channels.interaction.resolve_ask_user` over a
        :class:`_ConsumerIO` adapter, so serve mode and the CLI bridge
        cannot drift.

        Returns a dict suitable for ``Command(resume=...)``:
        ``{"answers": [...], "status": "answered"}`` or
        ``{"status": "cancelled"}``.
        """
        questions = event_data.get("questions", [])
        io = _ConsumerIO(self, msg, session_key)
        return await resolve_ask_user(questions, io, timeout=ASK_USER_TIMEOUT)

    # ── internal ──

    def _evict_chat_locks(self) -> None:
        """Remove chat locks that are not currently held."""
        stale = [k for k, lock in self._chat_locks.items() if not lock.locked()]
        for k in stale[: max(1, len(stale) // 2)]:
            del self._chat_locks[k]
