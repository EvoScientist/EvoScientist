"""Composable message processing middleware.

Each middleware is a standalone class that can be composed into a pipeline.
They extract logic that was previously baked into the Channel base class,
making it reusable across both legacy and plugin-based channels.

Also contains the supporting data structures (DedupCache, GroupHistoryBuffer,
TypingManager, PairingManager) that were previously in separate files.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import random
import time
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .base import RawIncoming
from .bus.events import InboundMessage, OutboundMessage
from .debug import emit_debug_event_if

_logger = logging.getLogger(__name__)


# ── Task cancellation helper ─────────────────────────────────────────


async def _cancel_task(task: asyncio.Task) -> None:
    """Cancel an asyncio task and await its completion.

    Suppresses ``CancelledError`` from the cancelled *task* but re-raises
    if the **current** task was itself cancelled (to avoid swallowing an
    outer cancellation signal — required for correct behavior on
    Python 3.12+ where ``_must_cancel`` no longer auto-re-delivers).
    """
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        current = asyncio.current_task()
        if current is not None and current.cancelling() > 0:
            raise
    except Exception:
        pass  # Already logged elsewhere; prevent unhandled propagation


# ═══════════════════════════════════════════════════════════════════════
# Supporting data structures
# ═══════════════════════════════════════════════════════════════════════


# ── Dedup cache ──────────────────────────────────────────────────────

_DEDUP_MAX = 1000
_DEDUP_TRIM = 500
_DEDUP_TTL = 3600  # 1 hour


class DedupCache:
    """Bounded ordered cache with TTL for detecting duplicate message IDs.

    Entries expire after *ttl_seconds* and are pruned lazily on each
    lookup.  When the cache exceeds *max_size* entries it is trimmed
    down to *trim_to* by evicting the oldest entries.  Accessed entries
    are moved to the end (LRU behavior).
    """

    def __init__(
        self,
        max_size: int = _DEDUP_MAX,
        trim_to: int = _DEDUP_TRIM,
        ttl_seconds: float = _DEDUP_TTL,
    ) -> None:
        self._seen: OrderedDict[str, float] = OrderedDict()
        self._max = max_size
        self._trim = trim_to
        self._ttl = ttl_seconds

    # ── public API ──────────────────────────────────────────────────

    def is_duplicate(self, msg_id: str) -> bool:
        """Return ``True`` if *msg_id* has been seen before.

        First-time IDs are recorded and ``False`` is returned.
        Empty / falsy IDs are never considered duplicates.
        Expired entries are pruned before the check.
        """
        if not msg_id:
            return False

        self._prune()

        if msg_id in self._seen:
            # LRU: refresh position and timestamp
            self._seen.move_to_end(msg_id)
            self._seen[msg_id] = time.monotonic()
            return True

        self._seen[msg_id] = time.monotonic()
        if len(self._seen) > self._max:
            while len(self._seen) > self._trim:
                self._seen.popitem(last=False)
        return False

    def clear(self) -> None:
        """Remove all entries."""
        self._seen.clear()

    @property
    def size(self) -> int:
        """Number of entries currently in the cache."""
        return len(self._seen)

    # ── internal ────────────────────────────────────────────────────

    def _prune(self) -> None:
        """Remove entries older than *ttl_seconds*."""
        cutoff = time.monotonic() - self._ttl
        # OrderedDict is insertion-ordered; oldest entries are first.
        while self._seen:
            _key, ts = next(iter(self._seen.items()))
            if ts > cutoff:
                break
            self._seen.popitem(last=False)


# ── Group history buffer ─────────────────────────────────────────────


@dataclass
class HistoryEntry:
    sender_id: str
    text: str
    timestamp: float
    message_id: str = ""


class GroupHistoryBuffer:
    """Per-chat circular buffer of recent messages."""

    def __init__(self, max_per_chat: int = 50, max_age_seconds: int = 3600):
        self._buffers: dict[str, deque[HistoryEntry]] = {}
        self._max = max_per_chat
        self._max_age = max_age_seconds

    def add(self, chat_id: str, entry: HistoryEntry) -> None:
        """Add a message to the chat's history buffer."""
        if chat_id not in self._buffers:
            self._buffers[chat_id] = deque(maxlen=self._max)
        self._buffers[chat_id].append(entry)

    def get_recent(self, chat_id: str, limit: int = 20) -> list[HistoryEntry]:
        """Get recent messages for context injection, excluding expired ones."""
        buf = self._buffers.get(chat_id)
        if not buf:
            return []
        now = time.monotonic()
        recent = [e for e in buf if now - e.timestamp < self._max_age]
        return recent[-limit:]

    def format_context(self, chat_id: str, limit: int = 20) -> str:
        """Format recent messages as context block for the agent."""
        entries = self.get_recent(chat_id, limit)
        if not entries:
            return ""
        lines = ["[Chat messages since your last reply - for context]"]
        for e in entries:
            lines.append(f"[from: {e.sender_id}] {e.text}")
        lines.append("[/Chat context]")
        return "\n".join(lines)

    def clear(self, chat_id: str) -> None:
        """Clear history for a chat (e.g., after the bot replies)."""
        self._buffers.pop(chat_id, None)


# ── Typing indicator manager ─────────────────────────────────────────


class TypingManager:
    """Manages background typing-indicator loops per chat_id.

    Args:
        send_action: Async callable that sends a single typing indicator
            for a given chat_id.
        interval: Seconds between typing indicator sends.
    """

    def __init__(
        self,
        send_action: Callable[[str], Awaitable[None]],
        interval: float = 5.0,
        debug_trace: bool = False,
        channel_name: str = "unknown",
    ) -> None:
        self._send_action = send_action
        self._interval = interval
        self._tasks: dict[str, asyncio.Task] = {}
        self._debug_trace = debug_trace
        self._channel_name = channel_name

    async def start(self, chat_id: str) -> None:
        """Start a background typing-indicator loop for *chat_id*."""
        await self.stop(chat_id)

        async def _loop() -> None:
            while True:
                try:
                    await self._send_action(chat_id)
                except Exception as exc:
                    emit_debug_event_if(
                        _logger,
                        "typing_error",
                        self._debug_trace,
                        channel=self._channel_name,
                        chat_id=chat_id,
                        error=str(exc),
                    )
                await asyncio.sleep(self._interval)

        self._tasks[chat_id] = asyncio.create_task(_loop())
        emit_debug_event_if(
            _logger,
            "typing_start",
            self._debug_trace,
            channel=self._channel_name,
            chat_id=chat_id,
        )

    async def stop(self, chat_id: str) -> None:
        """Cancel the typing-indicator loop for *chat_id*."""
        task = self._tasks.pop(chat_id, None)
        if task:
            await _cancel_task(task)
            emit_debug_event_if(
                _logger,
                "typing_stop",
                self._debug_trace,
                channel=self._channel_name,
                chat_id=chat_id,
            )

    async def stop_all(self) -> None:
        """Cancel all active typing-indicator loops."""
        for cid in list(self._tasks):
            await self.stop(cid)

    @property
    def active_chats(self) -> list[str]:
        """Return chat_ids with active typing loops."""
        return list(self._tasks)


# ── Pairing manager ─────────────────────────────────────────────────


@dataclass
class PairingRequest:
    sender_id: str
    channel: str
    code: str
    created_at: float
    approved: bool = False


class PairingManager:
    """Manages DM pairing codes for channel access control."""

    CODE_EXPIRY = 3600  # 1 hour
    MAX_PENDING = 50  # max pending requests

    def __init__(self):
        self._pending: dict[str, PairingRequest] = {}  # code -> request
        self._approved: set[str] = set()  # "channel:sender_id" keys

    def is_approved(self, channel: str, sender_id: str) -> bool:
        """Check if sender is already approved."""
        return f"{channel}:{sender_id}" in self._approved

    def request_pairing(self, channel: str, sender_id: str) -> str:
        """Generate a pairing code for a new sender. Returns the code."""
        # Check if already has pending request
        for code, req in list(self._pending.items()):
            if req.sender_id == sender_id and req.channel == channel:
                if time.monotonic() - req.created_at < self.CODE_EXPIRY:
                    return code  # return existing code
                else:
                    del self._pending[code]
                    break

        # Cleanup expired
        self._cleanup_expired()

        # Generate new code
        code = f"{random.randint(100000, 999999)}"
        while code in self._pending:
            code = f"{random.randint(100000, 999999)}"

        self._pending[code] = PairingRequest(
            sender_id=sender_id,
            channel=channel,
            code=code,
            created_at=time.monotonic(),
        )
        _logger.info(f"Pairing code {code} generated for {channel}:{sender_id}")
        return code

    def approve(self, code: str) -> tuple[bool, str]:
        """Approve a pairing code. Returns (success, message)."""
        req = self._pending.get(code)
        if not req:
            return False, f"Unknown code: {code}"
        if time.monotonic() - req.created_at > self.CODE_EXPIRY:
            del self._pending[code]
            return False, f"Code {code} expired"

        key = f"{req.channel}:{req.sender_id}"
        self._approved.add(key)
        del self._pending[code]
        _logger.info(f"Approved pairing for {key}")
        return True, f"Approved {req.sender_id} on {req.channel}"

    def reject(self, code: str) -> tuple[bool, str]:
        """Reject a pairing code."""
        if code in self._pending:
            del self._pending[code]
            return True, f"Rejected code {code}"
        return False, f"Unknown code: {code}"

    def list_pending(self) -> list[PairingRequest]:
        """List all pending (non-expired) requests."""
        self._cleanup_expired()
        return list(self._pending.values())

    def _cleanup_expired(self):
        now = time.monotonic()
        expired = [
            c for c, r in self._pending.items() if now - r.created_at > self.CODE_EXPIRY
        ]
        for c in expired:
            del self._pending[c]


# ═══════════════════════════════════════════════════════════════════════
# Middleware classes
# ═══════════════════════════════════════════════════════════════════════


# ── Inbound middleware base ──────────────────────────────────────────


class InboundMiddleware:
    """Base class for inbound message processing middleware."""

    async def process_inbound(
        self,
        raw: RawIncoming,
        context: dict[str, Any],
    ) -> RawIncoming | None:
        """Process an inbound raw message.

        Return the (possibly modified) RawIncoming to continue the
        pipeline, or ``None`` to drop the message.
        """
        return raw


class OutboundMiddlewareBase:
    """Base class for outbound message processing middleware."""

    async def process_outbound(
        self,
        message: OutboundMessage,
        context: dict[str, Any],
    ) -> OutboundMessage | None:
        """Process an outbound message.

        Return the (possibly modified) OutboundMessage to continue,
        or ``None`` to drop it.
        """
        return message


def _debug_trace_enabled(context: dict[str, Any]) -> bool:
    """Check whether channel-level debug tracing is enabled for this message."""
    channel = context.get("channel")
    if channel is None:
        return False
    checker = getattr(channel, "is_debug_trace_enabled", None)
    if callable(checker):
        try:
            return bool(checker())
        except Exception:
            return False
    return bool(getattr(getattr(channel, "config", None), "debug_trace", False))


def _ctx_channel_name(context: dict[str, Any]) -> str:
    """Extract the channel name from middleware context."""
    ch = context.get("channel")
    return getattr(ch, "name", "unknown") if ch else "unknown"


# ── Dedup ────────────────────────────────────────────────────────────


class DedupMiddleware(InboundMiddleware):
    """Message deduplication using a bounded TTL cache."""

    def __init__(
        self,
        max_size: int = 1000,
        trim_to: int = 500,
        ttl_seconds: float = 3600.0,
    ) -> None:
        self._cache = DedupCache(
            max_size=max_size,
            trim_to=trim_to,
            ttl_seconds=ttl_seconds,
        )

    async def process_inbound(
        self,
        raw: RawIncoming,
        context: dict[str, Any],
    ) -> RawIncoming | None:
        if raw.message_id and self._cache.is_duplicate(raw.message_id):
            emit_debug_event_if(
                _logger,
                "middleware_dedup_drop",
                _debug_trace_enabled(context),
                channel=_ctx_channel_name(context),
                message_id=raw.message_id,
                sender_id=raw.sender_id,
            )
            return None
        return raw


# ── Debounce ─────────────────────────────────────────────────────────


class DebounceMiddleware:
    """Per-sender message batching with configurable timing.

    This middleware collects messages from the same sender and merges
    them after a debounce delay.  It does not follow the simple
    process_inbound pattern because it needs to buffer across calls.

    Usage: call ``submit()`` for each message; merged results are
    delivered via the ``on_ready`` callback.
    """

    def __init__(
        self,
        *,
        initial_debounce: float = 2.0,
        debounce_step: float = 0.5,
        max_debounce: float = 5.0,
        on_ready: Callable[[InboundMessage], Any] | None = None,
    ) -> None:
        self.initial_debounce = initial_debounce
        self.debounce_step = debounce_step
        self.max_debounce = max_debounce
        self.on_ready = on_ready

        self._buffers: dict[str, list[str]] = {}
        self._metadata: dict[str, dict] = {}
        self._media: dict[str, list[str]] = {}
        self._message_ids: dict[str, str] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._channel_name: str = ""

    def set_channel_name(self, name: str) -> None:
        self._channel_name = name

    async def submit(self, msg: InboundMessage) -> None:
        """Buffer *msg* and schedule flush after debounce delay."""
        sender = msg.sender_id

        if sender not in self._buffers:
            self._buffers[sender] = []
            self._metadata[sender] = msg.metadata
            self._media[sender] = []
        self._buffers[sender].append(msg.content)
        if msg.message_id:
            self._message_ids[sender] = msg.message_id
        if msg.media:
            self._media[sender].extend(msg.media)

        if sender in self._tasks:
            self._tasks[sender].cancel()

        count = len(self._buffers[sender])
        wait = min(
            self.initial_debounce + (count - 1) * self.debounce_step,
            self.max_debounce,
        )

        async def _flush(_s: str = sender, _w: float = wait) -> None:
            await asyncio.sleep(_w)
            await self._flush_sender(_s)

        self._tasks[sender] = asyncio.create_task(_flush())

    async def _flush_sender(self, sender: str) -> None:
        messages = self._buffers.pop(sender, [])
        metadata = self._metadata.pop(sender, None)
        media = self._media.pop(sender, [])
        message_id = self._message_ids.pop(sender, "")
        self._tasks.pop(sender, None)
        if not messages:
            return

        merged = "\n".join(messages)
        chat_id = (metadata or {}).get("chat_id", sender)
        inbound = InboundMessage(
            channel=self._channel_name,
            sender_id=sender,
            chat_id=str(chat_id),
            content=merged,
            media=media,
            metadata=metadata or {},
            message_id=message_id,
        )
        if self.on_ready:
            await self.on_ready(inbound)

    async def cancel_all(self) -> None:
        """Cancel all pending debounce tasks and await their completion."""
        tasks = list(self._tasks.values())
        self._tasks.clear()
        for task in tasks:
            await _cancel_task(task)


# ── Chunking ─────────────────────────────────────────────────────────


class ChunkingMiddleware(OutboundMiddlewareBase):
    """Auto-split messages respecting format expansion.

    Wraps the existing ``chunking.chunk_text`` utility and the
    re-splitting logic from ``Channel._prepare_chunks``.
    """

    def __init__(self, capabilities: Any) -> None:
        from .capabilities import ChannelCapabilities

        self._capabilities: ChannelCapabilities = capabilities

    def prepare_chunks(
        self,
        content: str,
        limit: int,
        format_fn: Callable[[str], str] | None = None,
    ) -> list[tuple[str, str]]:
        """Build ``(formatted, raw)`` pairs, re-splitting when needed.

        If *format_fn* is None, formatted == raw.
        """
        from .base import chunk_text

        if format_fn is None:
            format_fn = lambda t: t  # noqa: E731

        raw_chunks = chunk_text(content, limit)
        pairs: list[tuple[str, str]] = []
        for raw in raw_chunks:
            formatted = format_fn(raw)
            if len(formatted) <= limit:
                pairs.append((formatted, raw))
            else:
                sub_limit = max(limit // 2, 500)
                for sub_raw in chunk_text(raw, sub_limit):
                    sub_fmt = format_fn(sub_raw)
                    if len(sub_fmt) <= limit:
                        pairs.append((sub_fmt, sub_raw))
                    else:
                        pairs.append((sub_raw, sub_raw))
        return pairs


# ── Formatting ───────────────────────────────────────────────────────


class FormattingMiddleware(OutboundMiddlewareBase):
    """Markdown -> channel format conversion.

    Uses ``UnifiedFormatter`` configured from capabilities.
    """

    def __init__(self, capabilities: Any) -> None:
        from .capabilities import ChannelCapabilities
        from .formatter import UnifiedFormatter

        caps: ChannelCapabilities = capabilities
        self._formatter = UnifiedFormatter.for_channel(caps.format_type)

    def format(self, text: str) -> str:
        """Convert text to channel format."""
        return self._formatter.format(text)

    async def process_outbound(
        self,
        message: OutboundMessage,
        context: dict[str, Any],
    ) -> OutboundMessage | None:
        formatted = self._formatter.format(message.content)
        return dataclasses.replace(message, content=formatted)


# ── Retry ────────────────────────────────────────────────────────────


class RetryMiddleware:
    """Exponential backoff send retry.

    Wraps ``retry.retry_async`` with channel-appropriate configuration.
    """

    def __init__(self, channel_name: str = "unknown") -> None:
        from .retry import DEFAULT_RETRY, RETRY_PRESETS

        self._config = RETRY_PRESETS.get(channel_name, DEFAULT_RETRY)
        self._channel_name = channel_name

    async def execute(
        self,
        coro_factory: Callable[[], Any],
        should_retry: Callable[[Exception, int], bool] | None = None,
        retry_after_s: Callable[[Exception], float | None] | None = None,
    ) -> Any:
        """Execute *coro_factory* with retry logic."""
        from .retry import retry_async

        return await retry_async(
            coro_factory,
            config=self._config,
            should_retry=should_retry or (lambda exc, _: True),
            retry_after_s=retry_after_s,
            on_retry=lambda info: _logger.warning(
                f"{self._channel_name} retry {info.attempt}/{info.max_attempts} "
                f"in {info.delay_s:.2f}s: {info.error}"
            ),
            label=f"{self._channel_name}.send",
        )


# ── Typing ───────────────────────────────────────────────────────────


class TypingMiddleware:
    """Typing indicator management.

    Wraps ``TypingManager`` for use as a standalone middleware component.
    """

    def __init__(
        self,
        send_typing_fn: Callable[[str], Any],
        interval: float = 5.0,
        debug_trace: bool = False,
        channel_name: str = "unknown",
    ) -> None:
        self._manager = TypingManager(
            send_typing_fn,
            interval=interval,
            debug_trace=debug_trace,
            channel_name=channel_name,
        )

    async def start(self, chat_id: str) -> None:
        await self._manager.start(chat_id)

    async def stop(self, chat_id: str) -> None:
        await self._manager.stop(chat_id)

    async def stop_all(self) -> None:
        await self._manager.stop_all()


# ── ACK Reaction ─────────────────────────────────────────────────────


class AckReactionMiddleware:
    """ACK emoji reaction with configurable scope.

    Scope controls when reactions are sent:
    - ``"all"``: react to every message
    - ``"direct"``: react only in DMs
    - ``"group-all"``: react in group chats (all messages)
    - ``"group-mentions"``: react in groups only when mentioned
    - ``"off"``: disable reactions
    """

    def __init__(
        self,
        *,
        scope: str = "all",
        emoji: str = "\U0001f440",
        remove_after_reply: bool = False,
        send_fn: Callable[[str, str, str], Any] | None = None,
        remove_fn: Callable[[str, str, str], Any] | None = None,
        debug_trace: bool = False,
        channel_name: str = "unknown",
    ) -> None:
        self.scope = scope
        self.emoji = emoji
        self.remove_after_reply = remove_after_reply
        self._send_fn = send_fn
        self._remove_fn = remove_fn
        self._pending: dict[str, str] = {}  # chat_id -> message_id
        self._debug_trace = debug_trace
        self._channel_name = channel_name

    def should_react(self, *, is_group: bool, was_mentioned: bool) -> bool:
        if self.scope == "off":
            return False
        if self.scope == "all":
            return True
        if self.scope == "direct":
            return not is_group
        if self.scope == "group-all":
            return is_group
        if self.scope == "group-mentions":
            return is_group and was_mentioned
        return False

    async def send_ack(self, chat_id: str, message_id: str) -> None:
        if self._send_fn and message_id:
            try:
                await self._send_fn(chat_id, message_id, self.emoji)
                if self.remove_after_reply:
                    self._pending[chat_id] = message_id
                emit_debug_event_if(
                    _logger,
                    "ack_send_ok",
                    self._debug_trace,
                    channel=self._channel_name,
                    chat_id=chat_id,
                    message_id=message_id,
                )
            except Exception as exc:
                emit_debug_event_if(
                    _logger,
                    "ack_send_error",
                    self._debug_trace,
                    channel=self._channel_name,
                    chat_id=chat_id,
                    message_id=message_id,
                    error=str(exc),
                )

    async def remove_ack(self, chat_id: str) -> None:
        message_id = self._pending.pop(chat_id, None)
        if message_id and self._remove_fn:
            try:
                await self._remove_fn(chat_id, message_id, self.emoji)
                emit_debug_event_if(
                    _logger,
                    "ack_remove_ok",
                    self._debug_trace,
                    channel=self._channel_name,
                    chat_id=chat_id,
                    message_id=message_id,
                )
            except Exception as exc:
                emit_debug_event_if(
                    _logger,
                    "ack_remove_error",
                    self._debug_trace,
                    channel=self._channel_name,
                    chat_id=chat_id,
                    message_id=message_id,
                    error=str(exc),
                )


# ── Mention Gating ───────────────────────────────────────────────────


class MentionGatingMiddleware(InboundMiddleware):
    """Filter messages based on mention policy.

    Policy values:
    - ``"always"``: require mention in all chats
    - ``"group"``: require mention only in groups (default)
    - ``"off"``: never require mention
    """

    def __init__(
        self,
        require_mention: str = "group",
        strip_fn: Callable[[str], str] | None = None,
    ) -> None:
        self.require_mention = require_mention
        self._strip_fn = strip_fn

    async def process_inbound(
        self,
        raw: RawIncoming,
        context: dict[str, Any],
    ) -> RawIncoming | None:
        if not self._should_process(raw):
            emit_debug_event_if(
                _logger,
                "middleware_mention_drop",
                _debug_trace_enabled(context),
                channel=_ctx_channel_name(context),
                chat_id=raw.chat_id,
                policy=self.require_mention,
            )
            return None
        # Strip mentions from group messages
        if raw.is_group and self._strip_fn:
            raw = dataclasses.replace(raw, text=self._strip_fn(raw.text))
        return raw

    def _should_process(self, raw: RawIncoming) -> bool:
        if self.require_mention == "off":
            return True
        if self.require_mention == "always":
            return raw.was_mentioned
        # "group" — require mention only in groups
        if not raw.is_group:
            return True
        return raw.was_mentioned


# ── AllowList ────────────────────────────────────────────────────────


class AllowListMiddleware(InboundMiddleware):
    """Sender and channel allow-list enforcement."""

    def __init__(
        self,
        allowed_senders: set[str] | None = None,
        allowed_channels: set[str] | None = None,
        dm_policy: str = "allowlist",
    ) -> None:
        self.allowed_senders = allowed_senders
        self.allowed_channels = allowed_channels
        self.dm_policy = dm_policy

    async def process_inbound(
        self,
        raw: RawIncoming,
        context: dict[str, Any],
    ) -> RawIncoming | None:
        # Channel allow-list
        if self.allowed_channels and str(raw.chat_id) not in self.allowed_channels:
            emit_debug_event_if(
                _logger,
                "middleware_allowlist_drop",
                _debug_trace_enabled(context),
                channel=_ctx_channel_name(context),
                sender_id=raw.sender_id,
                chat_id=raw.chat_id,
                reason="chat_not_allowed",
            )
            return None

        # Sender allow-list
        if not raw.is_group and self.dm_policy == "open":
            return raw  # open DMs bypass sender checks

        if not self._is_sender_allowed(raw.sender_id):
            emit_debug_event_if(
                _logger,
                "middleware_allowlist_drop",
                _debug_trace_enabled(context),
                channel=_ctx_channel_name(context),
                sender_id=raw.sender_id,
                chat_id=raw.chat_id,
                reason="sender_not_allowed",
            )
            return None

        return raw

    def _is_sender_allowed(self, sender: str) -> bool:
        if not self.allowed_senders:
            return True
        sender_str = str(sender)
        if sender_str in self.allowed_senders:
            return True
        if "|" in sender_str:
            for part in sender_str.split("|"):
                if part and part in self.allowed_senders:
                    return True
        return False


# ── Group History ────────────────────────────────────────────────────


class GroupHistoryMiddleware(InboundMiddleware):
    """Buffer non-mentioned group messages, inject as context when mentioned."""

    def __init__(
        self,
        max_per_chat: int = 50,
        max_age_seconds: int = 3600,
    ) -> None:
        self._buffer = GroupHistoryBuffer(
            max_per_chat=max_per_chat,
            max_age_seconds=max_age_seconds,
        )

    async def process_inbound(
        self,
        raw: RawIncoming,
        context: dict[str, Any],
    ) -> RawIncoming | None:
        if not raw.is_group:
            return raw

        # Use monotonic clock for consistent expiry calculation
        ts = time.monotonic()

        if not raw.was_mentioned:
            self._buffer.add(
                raw.chat_id,
                HistoryEntry(
                    sender_id=raw.sender_id,
                    text=raw.text,
                    timestamp=ts,
                    message_id=raw.message_id,
                ),
            )
            # Don't drop here — let MentionGatingMiddleware handle that
            return raw

        # Mentioned: inject history context
        history_context = self._buffer.format_context(raw.chat_id)
        if history_context:
            emit_debug_event_if(
                _logger,
                "middleware_history_injected",
                _debug_trace_enabled(context),
                channel=_ctx_channel_name(context),
                chat_id=raw.chat_id,
                context_len=len(history_context),
            )
            raw = dataclasses.replace(
                raw,
                text=history_context
                + "\n\n[Current message - respond to this]\n"
                + raw.text,
            )
        self._buffer.clear(raw.chat_id)
        return raw


# ── Pairing ──────────────────────────────────────────────────────────


class PairingMiddleware(InboundMiddleware):
    """DM pairing flow management.

    When dm_policy is "pairing", unapproved DM senders receive a
    pairing code.  Approved senders pass through normally.
    """

    def __init__(
        self,
        channel_name: str,
        send_response_fn: Callable[[str, str], Any] | None = None,
        dm_policy: str = "allowlist",
    ) -> None:
        self._manager = PairingManager()
        self._channel_name = channel_name
        self._send_response_fn = send_response_fn
        self._dm_policy = dm_policy
        self._background_tasks: set[asyncio.Task] = set()

    async def process_inbound(
        self,
        raw: RawIncoming,
        context: dict[str, Any],
    ) -> RawIncoming | None:
        if raw.is_group:
            return raw  # pairing only applies to DMs

        if self._dm_policy != "pairing":
            return raw

        if self._manager.is_approved(self._channel_name, raw.sender_id):
            return raw

        # Request pairing
        code = self._manager.request_pairing(self._channel_name, raw.sender_id)
        if self._send_response_fn:
            text = f"\U0001f510 Pairing required. Your code: {code}\nThis code expires in 1 hour."
            task = asyncio.create_task(self._send_response_fn(raw.chat_id, text))
            # Track the task to prevent GC and handle exceptions
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        emit_debug_event_if(
            _logger,
            "middleware_pairing_required",
            _debug_trace_enabled(context),
            channel=_ctx_channel_name(context),
            sender_id=raw.sender_id,
        )
        _logger.info(f"Pairing required for {raw.sender_id}, code sent")
        return None
