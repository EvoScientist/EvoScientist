"""Abstract base class for communication channels.

This module defines the Channel interface that all messaging channels
(iMessage, WeChat, etc.) must implement.
"""

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable
from collections.abc import Callable as CallableABC
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..paths import MEDIA_DIR
from .bus.events import InboundMessage, OutboundMessage
from .capabilities import ChannelCapabilities
from .debug import TraceMixin, debug_trace_enabled
from .formatter import UnifiedFormatter
from .plugin import ChannelMeta, ChannelPlugin

_logger = logging.getLogger(__name__)


# ── Text chunking ────────────────────────────────────────────────────


def chunk_text(text: str, limit: int) -> list[str]:
    """Split text into chunks that respect logical boundaries and code fences.

    If a code block is split across chunks, each chunk is automatically
    wrapped in its own fences (```...```) to maintain formatting.

    Args:
        text: The text to split.
        limit: Maximum characters per chunk.

    Returns:
        List of text chunks, each <= limit characters.
    """
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    remaining = text
    in_code_block = False
    code_block_lang = ""

    while remaining:
        # Effective limit is reduced if we need to add fences
        # We reserve ~20 chars for fences (```lang\n and \n```)
        effective_limit = limit - (20 if in_code_block else 0)

        if len(remaining) <= effective_limit:
            segment = remaining
            best = len(remaining)
        else:
            segment = remaining[:effective_limit]
            best = -1

            # 1. Paragraph/Line/Word boundaries
            if not in_code_block:
                # Paragraph
                pos = segment.rfind("\n\n")
                if pos > 0:
                    best = pos

                # Line
                if best == -1:
                    pos = segment.rfind("\n")
                    if pos > 0:
                        best = pos

                # Word
                if best == -1:
                    pos = segment.rfind(" ")
                    if pos > 0:
                        best = pos
            else:
                # INSIDE code block: ONLY split at newlines to avoid breaking lines of code
                pos = segment.rfind("\n")
                if pos > 0:
                    best = pos

            if best == -1:
                best = effective_limit

        chunk_raw = remaining[:best].rstrip()

        # Track state transitions within this raw segment
        starts_in_code = in_code_block
        current_lang = code_block_lang

        # We use a simple count of ``` to toggle state.
        # Note: This handles both opening and closing fences.
        fences = list(re.finditer(r"```(\w*)", chunk_raw))
        for f in fences:
            if not in_code_block:
                in_code_block = True
                code_block_lang = f.group(1) or ""
            else:
                in_code_block = False
                code_block_lang = ""

        ends_in_code = in_code_block

        # Build the final chunk with necessary fences
        prefix = f"```{current_lang}\n" if starts_in_code else ""
        suffix = "\n```" if ends_in_code else ""

        final_chunk = prefix + chunk_raw + suffix
        if final_chunk.strip():
            chunks.append(final_chunk)

        remaining = remaining[best:].lstrip("\n")

    return chunks


# ── Attachment / media helpers ───────────────────────────────────────

MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024  # 20 MB

IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"})
VIDEO_EXTS = frozenset({".mp4", ".mov", ".avi", ".webm"})
AUDIO_EXTS = frozenset({".mp3", ".ogg", ".m4a", ".wav"})


def classify_media(ext: str) -> str | None:
    """Classify a file extension into a media type string.

    Returns ``"image"``, ``"video"``, ``"audio"``, or ``None``.
    """
    ext = ext.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    if ext in AUDIO_EXTS:
        return "audio"
    return None


def media_path(filename: str) -> Path:
    """Ensure MEDIA_DIR exists and return a path inside it."""
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    return MEDIA_DIR / filename


def check_attachment_size(file_size: int, filename: str) -> str | None:
    """Return a 'too large' annotation if *file_size* exceeds the limit.

    Returns ``None`` when the file is within the allowed size.
    """
    if file_size > MAX_ATTACHMENT_BYTES:
        return f"[attachment: {filename} - too large ({file_size} bytes)]"
    return None


async def download_attachment(
    url: str,
    filename: str,
    *,
    channel_name: str = "",
    headers: dict[str, str] | None = None,
    file_size: int | None = None,
    proxy: str | None = None,
) -> tuple[str | None, str | None]:
    """Download an attachment via httpx.

    Returns ``(local_path, annotation)``.

    If *file_size* exceeds ``MAX_ATTACHMENT_BYTES``, returns
    ``(None, too-large-annotation)`` without downloading.
    On download failure returns ``(None, failure-annotation)``.
    On success returns ``(local_path_str, success-annotation)``.
    """
    if file_size is not None:
        too_large = check_attachment_size(file_size, filename)
        if too_large:
            return None, too_large

    try:
        import httpx

        safe_name = filename.replace("/", "_")
        prefix = f"{channel_name}_" if channel_name else ""
        local_path = media_path(f"{prefix}{safe_name}")

        async with httpx.AsyncClient(proxy=proxy) as client:
            async with client.stream(
                "GET", url, headers=headers or {}, timeout=30
            ) as resp:
                if resp.status_code != 200:
                    return None, f"[attachment: {filename} - download failed]"

                # Check Content-Length header before downloading body
                if file_size is None:
                    cl = resp.headers.get("content-length")
                    if cl:
                        try:
                            too_large = check_attachment_size(int(cl), filename)
                            if too_large:
                                return None, too_large
                        except (ValueError, TypeError):
                            pass

                # Stream body with incremental size check
                chunks: list[bytes] = []
                total = 0
                async for chunk in resp.aiter_bytes():
                    total += len(chunk)
                    if total > MAX_ATTACHMENT_BYTES:
                        return None, check_attachment_size(total, filename)
                    chunks.append(chunk)

        local_path.write_bytes(b"".join(chunks))
        return str(local_path), f"[attachment: {local_path}]"
    except Exception as e:
        _logger.warning(f"Failed to download attachment: {e}")
        return None, f"[attachment: {filename} - download failed]"


# Deprecated aliases — use InboundMessage / OutboundMessage instead.
IncomingMessage = InboundMessage
OutgoingMessage = OutboundMessage


@dataclass
class RawIncoming:
    """Raw data extracted from a platform-specific message event.

    Each channel's ``_on_message`` populates this with platform data,
    then calls ``_enqueue_raw()`` which handles allow-list checks,
    content merging, and ``InboundMessage`` creation.
    """

    sender_id: str
    chat_id: str
    text: str = ""
    media_files: list[str] = field(default_factory=list)
    content_annotations: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: str = ""
    metadata: dict = field(default_factory=dict)
    is_group: bool = False
    was_mentioned: bool = True  # default True so DMs always pass


class Channel(TraceMixin, ChannelPlugin, ABC):
    """Abstract base class for messaging channels.

    Subclasses must implement:
    - start(): Initialize the channel (connect, authenticate, etc.)
    - _send_chunk(): Send a single text chunk (platform-specific)

    Subclasses may optionally override:
    - _cleanup(): Channel-specific teardown (called by stop())
    - _format_chunk(): Convert Markdown to channel format
    - _is_ready(): Return False if channel cannot send
    - _resolve_chat_id(): Extract chat_id from message
    - receive(): Only if custom exit conditions are needed

    Subclasses should set ``name`` to a unique identifier (e.g. "telegram").
    """

    name: str = "base"
    capabilities: ChannelCapabilities = ChannelCapabilities()
    _typing_interval: float = 5.0
    _ready_attrs: tuple[str, ...] = ()

    def __init__(self, config, *, queue_maxsize: int = 1000):
        ChannelPlugin.__init__(self)
        self.id = self.name
        self.meta = ChannelMeta(id=self.name, label=self.name.title())

        self.config = config

        # Cache STT config at startup to avoid loading it on every message
        from ..config.settings import load_config as _load_cfg

        _global = _load_cfg()
        self._stt_enabled: bool = _global.stt_enabled
        self._stt_language: str = _global.stt_language
        self._stt_model: str = _global.stt_model
        self._stt_device: str = _global.stt_device
        self._stt_compute_type: str = _global.stt_compute_type

        # Auto-configure formatter from capabilities
        self._formatter = UnifiedFormatter.for_channel(self.capabilities.format_type)
        self._queue: asyncio.Queue[InboundMessage] = asyncio.Queue(
            maxsize=queue_maxsize
        )
        self._running = False

        # Global tracing can be enabled via shared config/env even when
        # individual channel factories have not been updated yet.
        self._debug_trace: bool = bool(getattr(config, "debug_trace", False)) or (
            debug_trace_enabled()
        )
        self._trace_logger = _logger

        # Typing indicator — delegated to TypingManager
        from .middleware import TypingManager

        self._typing_manager = TypingManager(
            self._send_typing_action,
            interval=self._typing_interval,
            debug_trace=self._debug_trace,
            channel_name=self.name,
        )
        # Keep legacy dict reference for any subclass that touches it directly
        self._typing_tasks = self._typing_manager._tasks

        # Bus integration (injected by ChannelManager.register / set_bus)
        self._bus: Any = None
        self.send_thinking: bool = False
        self._on_activity: Callable | None = None

        # Debounce settings
        self.initial_debounce: float = 2.0
        self.debounce_step: float = 0.5
        self.max_debounce: float = 5.0

        # Per-sender message buffers for debouncing
        self._message_buffers: dict[str, list[str]] = {}
        self._message_metadata: dict[str, dict] = {}
        self._message_media: dict[str, list[str]] = {}
        self._message_ids: dict[str, str] = {}
        self._debounce_tasks: dict[str, asyncio.Task] = {}

        # Mention gating: "always" | "group" | "off"
        self.require_mention: str = getattr(config, "require_mention", "group")

        # DM policy: "open" | "allowlist" | "pairing"
        self.dm_policy: str = getattr(config, "dm_policy", "allowlist")

        # Per-sender is_group / was_mentioned for debounce merge
        self._message_is_group: dict[str, bool] = {}
        self._message_was_mentioned: dict[str, bool] = {}

        # Retry configuration (auto-resolved from channel name)
        from .retry import DEFAULT_RETRY, RETRY_PRESETS, RetryConfig

        self._retry_config: RetryConfig = RETRY_PRESETS.get(self.name, DEFAULT_RETRY)

        # Per-chat send locks to prevent message reordering.
        # Uses an OrderedDict as a bounded LRU cache to avoid unbounded growth.
        self._send_locks: OrderedDict[str, asyncio.Lock] = OrderedDict()
        self._send_locks_max: int = 1024

        # Build inbound middleware pipeline
        self._inbound_middlewares = self._build_inbound_middlewares()

    def _build_inbound_middlewares(self) -> list:
        """Build the inbound middleware chain from config and capabilities.

        Middleware order:
        1. DedupMiddleware — drop duplicates early
        2. AllowListMiddleware — enforce sender/channel restrictions
        3. PairingMiddleware — handle DM pairing (if applicable)
        4. GroupHistoryMiddleware — buffer/inject group history
        5. MentionGatingMiddleware — filter by mention policy
        """
        from .middleware import (
            AllowListMiddleware,
            DedupMiddleware,
            GroupHistoryMiddleware,
            MentionGatingMiddleware,
            PairingMiddleware,
        )

        middlewares = []
        middlewares.append(DedupMiddleware())
        # AllowList
        allowed_senders = getattr(self.config, "allowed_senders", None)
        allowed_channels = getattr(self.config, "allowed_channels", None)
        if allowed_senders and not isinstance(allowed_senders, set):
            allowed_senders = set(allowed_senders)
        if allowed_channels and not isinstance(allowed_channels, set):
            allowed_channels = set(allowed_channels)
        middlewares.append(
            AllowListMiddleware(
                allowed_senders=allowed_senders,
                allowed_channels=allowed_channels,
                dm_policy=self.dm_policy,
            )
        )
        # Pairing
        if self.dm_policy == "pairing":

            async def _send_pair(chat_id, text):
                await self._send_chunk(chat_id, text, text, None, {})

            middlewares.append(
                PairingMiddleware(
                    channel_name=self.name,
                    send_response_fn=_send_pair,
                    dm_policy=self.dm_policy,
                )
            )
        # GroupHistory
        if self.capabilities.groups:
            middlewares.append(GroupHistoryMiddleware())
        # MentionGating
        if self.capabilities.mentions:
            middlewares.append(
                MentionGatingMiddleware(
                    require_mention=self.require_mention,
                    strip_fn=self._strip_mention,
                )
            )
        return middlewares

    def is_debug_trace_enabled(self) -> bool:
        """Return whether extra per-message diagnostics should be emitted."""
        return self._debug_trace

    @abstractmethod
    async def start(self) -> None:
        """Initialize and start the channel.

        This method should:
        - Establish connections
        - Verify permissions/authentication
        - Start any background tasks needed

        Raises:
            ChannelError: If initialization fails
        """
        pass

    async def stop(self) -> None:
        """Stop the channel and flush pending debounce buffers."""
        self._running = False

        # Cancel pending debounce timers, then flush buffered messages so they
        # are not lost when stopping within the debounce window.
        pending_tasks = list(self._debounce_tasks.values())
        for task in pending_tasks:
            task.cancel()
        for task in pending_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                _logger.debug(f"{self.name} debounce task shutdown error: {e}")
        self._debounce_tasks.clear()

        for sender in list(self._message_buffers.keys()):
            try:
                await self._process_buffered_messages(sender)
            except Exception as e:
                _logger.error(
                    f"{self.name} failed to flush buffered messages for {sender}: {e}"
                )

        await self._typing_manager.stop_all()
        await self._cleanup()

    async def _cleanup(self) -> None:
        """Channel-specific teardown. Override in subclasses."""

    async def receive(self) -> AsyncIterator[InboundMessage]:
        """Yield incoming messages from the queue.

        Default implementation polls ``self._queue``. Override only if
        the channel needs custom exit conditions.
        """
        while self._running:
            try:
                msg = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                yield msg
            except TimeoutError:
                continue

    def _acquire_send_lock(self, chat_id: str) -> asyncio.Lock:
        """Get or create a per-chat send lock with LRU eviction.

        Moves the accessed entry to the end (most-recently-used).
        When the cache exceeds ``_send_locks_max``, the least-recently-used
        entry is evicted — but only if its lock is not currently held.
        """
        if chat_id in self._send_locks:
            self._send_locks.move_to_end(chat_id)
        else:
            self._send_locks[chat_id] = asyncio.Lock()
            # Evict oldest unlocked entries when over capacity.
            # Skip locked entries instead of giving up entirely,
            # to prevent unbounded growth.
            if len(self._send_locks) > self._send_locks_max:
                to_evict = [
                    k
                    for k, lock in self._send_locks.items()
                    if not lock.locked() and k != chat_id
                ]
                for k in to_evict:
                    if len(self._send_locks) <= self._send_locks_max:
                        break
                    del self._send_locks[k]
        return self._send_locks[chat_id]

    async def send(self, message: OutboundMessage) -> bool:
        """Send a message. Handles chunking, retry, and error logging.

        Subclasses override ``_send_chunk()`` for the platform-specific call.
        Override ``_format_chunk()`` to convert Markdown to channel format.

        A per-chat lock ensures messages to the same chat are serialised,
        preventing out-of-order delivery when multiple sends overlap.

        If formatting expands a chunk beyond the platform limit (e.g. Markdown
        → HTML), the chunk is automatically re-split at a smaller size.  Per-
        chunk errors are logged but do not abort delivery of remaining chunks.

        When the channel satisfies ``ThreadingAdapter``, its ``reply_to_mode``
        controls which chunks carry a ``reply_to`` reference.
        """
        if not self._is_ready():
            self._trace_event(
                "outbound_send_skip",
                reason="channel_not_ready",
                chat_id=message.chat_id,
                reply_to=message.reply_to,
            )
            return False
        try:
            chat_id = self._resolve_chat_id(message)
            limit = self._get_chunk_limit()
            with self._trace_span(
                "outbound_send",
                chat_id=chat_id,
                reply_to=message.reply_to,
                content_len=len(message.content or ""),
                media_count=len(message.media or []),
                chunk_limit=limit,
            ) as span:
                async with self._acquire_send_lock(chat_id):
                    pairs = self._prepare_chunks(message.content, limit)
                    span.set(chunk_count=len(pairs))
                    had_error = False
                    for i, (formatted, raw) in enumerate(pairs):
                        reply_to = self._resolve_reply_to(message.reply_to, i)
                        try:
                            await self._send_with_retry(
                                lambda _cid=chat_id, _fmt=formatted, _raw=raw, _reply=reply_to, _meta=message.metadata: (
                                    self._send_chunk(_cid, _fmt, _raw, _reply, _meta)
                                )
                            )
                        except Exception as chunk_err:
                            self._trace_event(
                                "outbound_send_chunk_error",
                                chat_id=chat_id,
                                reply_to=reply_to,
                                chunk_index=i,
                                error_type=type(chunk_err).__name__,
                            )
                            _logger.error(f"{self.name} chunk {i} send error: {chunk_err}")
                            had_error = True
                if had_error:
                    span.set(result="partial")
            return not had_error
        except Exception as e:
            _logger.error(f"{self.name} send error: {e}")
            return False

    def _resolve_reply_to(self, reply_to: str | None, chunk_index: int) -> str | None:
        """Determine the reply_to value for a given chunk index.

        Legacy: reply_to on first chunk only.
        """
        if not reply_to:
            return None
        return reply_to if chunk_index == 0 else None

    def _prepare_chunks(
        self,
        content: str,
        limit: int,
    ) -> list[tuple[str, str]]:
        """Build ``(formatted, raw)`` pairs, re-splitting when formatting
        expands a chunk beyond *limit*.

        Returns a list of ``(formatted_text, raw_text)`` tuples ready
        for ``_send_chunk()``.
        """
        raw_chunks = chunk_text(content, limit)
        pairs: list[tuple[str, str]] = []
        for raw in raw_chunks:
            formatted = self._format_chunk(raw)
            if len(formatted) <= limit:
                pairs.append((formatted, raw))
            else:
                # Re-chunk at half the limit to leave room for format expansion
                sub_limit = max(limit // 2, 500)
                for sub_raw in chunk_text(raw, sub_limit):
                    sub_fmt = self._format_chunk(sub_raw)
                    if len(sub_fmt) <= limit:
                        pairs.append((sub_fmt, sub_raw))
                    else:
                        # Still too long — send raw text (guaranteed to fit)
                        pairs.append((sub_raw, sub_raw))
        return pairs

    def _is_ready(self) -> bool:
        """Return False if the channel cannot send (e.g. client not connected).

        Default checks that every attribute named in ``_ready_attrs`` is truthy.
        Override for channels with more complex readiness logic.
        """
        if not self._ready_attrs:
            return True
        return all(getattr(self, attr, None) for attr in self._ready_attrs)

    def _resolve_chat_id(self, message: OutboundMessage) -> str:
        """Extract chat_id from metadata or recipient. Override if needed."""
        return message.metadata.get("chat_id", message.recipient)

    def _get_chunk_limit(self) -> int:
        config_limit = getattr(self.config, "text_chunk_limit", 0)
        cap_limit = self.capabilities.max_text_length
        return config_limit or cap_limit or 4096

    def _format_chunk(self, text: str) -> str:
        """Convert Markdown to channel format via UnifiedFormatter.

        Uses the formatter auto-configured from ``capabilities.format_type``.
        Subclasses rarely need to override this — set ``capabilities`` instead.
        """
        result = self._formatter.format(text)
        self._trace_event(
            "outbound_format_applied",
            input_len=len(text),
            output_len=len(result),
        )
        return result

    @abstractmethod
    async def _send_chunk(
        self,
        chat_id: str,
        formatted_text: str,
        raw_text: str,
        reply_to: str | None,
        metadata: dict,
    ) -> None:
        """Send a single text chunk. Platform-specific implementation."""
        ...

    _format_fallback_patterns: tuple[str, ...] = ("parse", "invalid")

    async def _send_with_format_fallback(
        self,
        send_fn: CallableABC[[str], Awaitable],
        formatted: str,
        raw: str,
    ) -> None:
        """Try *send_fn(formatted)*; on format-related errors retry with *raw*.

        Channels whose ``_send_chunk`` follows the try-formatted / except-fallback
        pattern can delegate to this helper instead of duplicating the logic.
        """
        try:
            await send_fn(formatted)
        except Exception as e:
            if formatted != raw and any(
                p in str(e).lower() for p in self._format_fallback_patterns
            ):
                self._trace_event(
                    "outbound_format_fallback",
                    error=str(e),
                    formatted_len=len(formatted),
                    raw_len=len(raw),
                )
                await send_fn(raw)
            else:
                raise

    async def send_media(
        self,
        recipient: str,
        file_path: str,
        caption: str = "",
        metadata: dict | None = None,
    ) -> bool:
        """Send a media file through the channel.

        Handles the ready-check guard and error logging.  Subclasses
        override ``_send_media_impl()`` with platform-specific logic.

        Args:
            recipient: Target recipient or chat identifier.
            file_path: Local path to the media file.
            caption: Optional caption text.
            metadata: Optional channel-specific metadata.

        Returns:
            True if sent successfully, False otherwise.
        """
        chat_id = self._resolve_media_chat_id(recipient, metadata)
        if not self._is_ready():
            self._trace_event(
                "outbound_media_skip",
                reason="channel_not_ready",
                chat_id=chat_id,
                file_path=file_path,
            )
            return False
        try:
            with self._trace_span(
                "outbound_media",
                chat_id=chat_id,
                file_path=file_path,
                caption_len=len(caption or ""),
            ):
                result = await self._send_media_impl(
                    recipient, file_path, caption, metadata
                )
            return result
        except Exception as e:
            _logger.error(f"{self.name} send_media error: {e}")
            return False

    async def _send_media_impl(
        self,
        recipient: str,
        file_path: str,
        caption: str = "",
        metadata: dict | None = None,
    ) -> bool:
        """Platform-specific media send.  Override in subclasses."""
        return False

    # ── Attachment / proxy helpers ─────────────────────────────────

    def _media_path(self, filename: str) -> Path:
        """Ensure MEDIA_DIR exists and return a path inside it."""
        return media_path(filename)

    def _resolve_media_chat_id(self, recipient: str, metadata: dict | None) -> str:
        """Extract chat_id from metadata, falling back to recipient."""
        return (metadata or {}).get("chat_id", recipient)

    def _get_proxy(self) -> str | None:
        """Return the configured proxy URL, or ``None`` if unset/empty."""
        return getattr(self.config, "proxy", None) or None

    def _check_attachment_size(self, file_size: int, filename: str) -> str | None:
        """Return a 'too large' annotation string if *file_size* exceeds the limit."""
        return check_attachment_size(file_size, filename)

    async def _download_attachment(
        self,
        url: str,
        filename: str,
        *,
        headers: dict[str, str] | None = None,
        file_size: int | None = None,
    ) -> tuple[str | None, str | None]:
        """Download an attachment via httpx.  Returns ``(local_path, annotation)``.

        Delegates to :func:`download_attachment`.
        """
        return await download_attachment(
            url,
            filename,
            channel_name=self.name,
            headers=headers,
            file_size=file_size,
            proxy=self._get_proxy(),
        )

    # ── Send retry abstraction ──────────────────────────────────────

    _non_retryable_patterns: tuple[str, ...] = ()
    _rate_limit_patterns: tuple[str, ...] = ("429", "ratelimit")
    _rate_limit_delay: float = 1.0

    def _extract_retry_after(self, exc: Exception) -> float | None:
        """Extract retry-wait seconds from an exception.

        Returns ``None`` to signal that the error is **not retryable**.

        Pipeline:
        1. SDK-provided ``retry_after`` attribute (Telegram / Slack SDKs).
        2. HTTP ``Retry-After`` header via :meth:`_parse_retry_after_header`.
        3. Non-retryable pattern match → ``None``.
        4. Rate-limit pattern match → ``_rate_limit_delay``.
        5. Default ``1.0`` s (generic transient-error retry).

        Channels can customize behavior declaratively via class attributes
        ``_non_retryable_patterns``, ``_rate_limit_patterns``, and
        ``_rate_limit_delay``, or override this method entirely.
        """
        # 1. SDK retry_after attribute
        retry = getattr(exc, "retry_after", None)
        if retry is not None:
            return float(retry)

        # 2. HTTP Retry-After header
        header_val = self._parse_retry_after_header(exc)
        if header_val is not None:
            return header_val

        msg = str(exc).lower()

        # 3. Non-retryable patterns
        if self._non_retryable_patterns and any(
            p in msg for p in self._non_retryable_patterns
        ):
            return None

        # 4. Rate-limit patterns
        if self._rate_limit_patterns and any(
            p in msg for p in self._rate_limit_patterns
        ):
            return self._rate_limit_delay

        # 5. Default
        return 1.0

    def _parse_retry_after_header(self, exc: Exception) -> float | None:
        """Try to extract a ``Retry-After`` value from an HTTP response."""
        resp = getattr(exc, "response", None)
        if resp is None:
            return None
        headers = getattr(resp, "headers", None)
        if not headers:
            return None
        raw = headers.get("Retry-After") or headers.get("retry-after")
        if raw is None:
            return None
        try:
            return float(raw)
        except (ValueError, TypeError):
            return None

    async def _send_with_retry(
        self,
        coro_factory: CallableABC[[], Awaitable],
        max_retries: int = 3,
    ) -> Any:
        """Send helper with automatic exponential-backoff retry.

        *coro_factory* is called on every attempt so that the awaitable is
        fresh.  Uses :func:`retry.retry_async` for backoff, jitter, and
        server-supplied ``Retry-After`` support.

        The *max_retries* parameter is accepted for backward compatibility
        but the attempt count is taken from ``self._retry_config``.
        """
        from .retry import retry_async

        def _on_retry(info):
            self._trace_event(
                "outbound_send_retry",
                attempt=info.attempt,
                max_attempts=info.max_attempts,
                backoff_s=round(info.delay_s, 2),
                error_type=type(info.error).__name__,
            )
            _logger.warning(
                f"{self.name} send retry {info.attempt}/{info.max_attempts} "
                f"in {info.delay_s:.2f}s: {info.error}"
            )

        return await retry_async(
            coro_factory,
            config=self._retry_config,
            should_retry=lambda exc, _: self._extract_retry_after(exc) is not None,
            retry_after_s=self._extract_retry_after,
            on_retry=_on_retry,
            label=f"{self.name}.send",
        )

    # ── Typing indicator abstraction ─────────────────────────────────

    async def _send_typing_action(self, chat_id: str) -> None:
        """Send a single typing indicator.  Override in sub-classes."""

    async def start_typing(self, chat_id: str) -> None:
        """Start a background typing-indicator loop for *chat_id*."""
        await self._typing_manager.start(chat_id)

    async def stop_typing(self, chat_id: str) -> None:
        """Cancel the typing-indicator loop for *chat_id*."""
        await self._typing_manager.stop(chat_id)

    # ── Mention gating ──────────────────────────────────────────────

    def _should_process(self, raw: RawIncoming) -> bool:
        """Decide whether to process a message based on mention gating."""
        if self.require_mention == "off":
            return True
        # Both "always" and "group" allow DMs through unconditionally
        if not raw.is_group:
            return True
        if self.require_mention == "always":
            return raw.was_mentioned
        # "group" — require mention only in groups
        return raw.was_mentioned

    _mention_pattern: str | None = None
    _mention_strip_count: int = 0  # 0 = all occurrences, 1 = first only

    def _get_bot_identifier(self) -> str | None:
        """Return the bot's identifier for mention pattern substitution.

        Override in subclasses where ``_mention_pattern`` contains
        ``{bot_id}`` placeholder.
        """
        return None

    def _strip_mention(self, text: str) -> str:
        """Strip bot mention from text using the ``_mention_pattern`` approach."""
        if not self._mention_pattern:
            return text
        pattern = self._mention_pattern
        if "{bot_id}" in pattern:
            bot_id = self._get_bot_identifier()
            if not bot_id:
                return text
            pattern = pattern.replace("{bot_id}", re.escape(bot_id))
        return re.sub(pattern, "", text, count=self._mention_strip_count).strip()

    # ── ACK reaction ─────────────────────────────────────────────────

    async def _send_ack_reaction(
        self, chat_id: str, message_id: str, emoji: str = "👀"
    ) -> None:
        """Send an acknowledgment reaction to a message. Override in subclasses that support reactions."""
        pass  # Default no-op; channels override if they support reactions

    async def _remove_ack_reaction(
        self, chat_id: str, message_id: str, emoji: str = "👀"
    ) -> None:
        """Remove the ack reaction after replying. Override in subclasses."""
        pass

    # ── Inbound message pipeline ──────────────────────────────────────

    async def _build_inbound_async(self, raw: RawIncoming) -> InboundMessage | None:
        """Async version: run *raw* through inbound middlewares and convert."""
        context: dict = {"channel": self}
        current: RawIncoming | None = raw
        for mw in self._inbound_middlewares:
            if current is None:
                return None
            result = await mw.process_inbound(current, context)
            if result is None:
                self._trace_event(
                    "inbound_middleware_drop",
                    middleware=type(mw).__name__,
                    sender_id=current.sender_id,
                    chat_id=current.chat_id,
                    message_id=current.message_id or "-",
                )
                return None
            current = result
        if current is None:
            return None
        return self._raw_to_inbound(current)

    def _build_inbound(self, raw: RawIncoming) -> InboundMessage | None:
        """Run *raw* through inbound middlewares and convert to InboundMessage.

        Synchronous wrapper around :meth:`_build_inbound_async`.  When an
        event loop is already running, the coroutine is scheduled on that
        loop via :func:`asyncio.run_coroutine_threadsafe` to avoid
        thread-safety issues with middleware state (DedupCache,
        GroupHistoryBuffer, etc.).
        """
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            future = asyncio.run_coroutine_threadsafe(
                self._build_inbound_async(raw),
                loop,
            )
            return future.result()
        else:
            new_loop = asyncio.new_event_loop()
            try:
                return new_loop.run_until_complete(self._build_inbound_async(raw))
            finally:
                new_loop.close()

    def _raw_to_inbound(self, raw: RawIncoming) -> InboundMessage | None:
        """Convert a RawIncoming to InboundMessage (pure transformation, no filtering).

        Merges text + annotations into content, sets metadata.
        Returns None only if there is no content and no media.
        """
        parts = []
        if raw.text:
            parts.append(raw.text)
        parts.extend(raw.content_annotations)
        content = "\n".join(p for p in parts if p)
        if not content and not raw.media_files:
            return None
        meta = dict(raw.metadata)
        meta.setdefault("chat_id", raw.chat_id)
        return InboundMessage(
            channel=self.name,
            sender_id=raw.sender_id,
            chat_id=raw.chat_id,
            content=content or "[media only]",
            timestamp=raw.timestamp,
            message_id=raw.message_id,
            media=raw.media_files,
            metadata=meta,
            is_group=raw.is_group,
            was_mentioned=raw.was_mentioned,
        )

    async def _enqueue_raw(self, raw: RawIncoming) -> None:
        """Run *raw* through the inbound middleware pipeline, convert to
        InboundMessage, and put it on the queue.

        Convenience method for subclass ``_on_message`` handlers.
        If STT is enabled and the message contains audio files, each audio
        file is transcribed and the result is prepended to ``raw.text``.
        """
        self._trace_event(
            "inbound_raw",
            sender_id=raw.sender_id,
            chat_id=raw.chat_id,
            message_id=raw.message_id or "-",
            has_text=bool(raw.text),
            media_count=len(raw.media_files),
            is_group=raw.is_group,
        )
        if raw.media_files and self._stt_enabled:
            from ..stt import is_audio_file, transcribe_file

            transcripts: list[str] = []
            transcribed_files: set[str] = set()
            for fp in raw.media_files:
                if is_audio_file(fp):
                    self._trace_event("stt_start", file_path=fp)
                    try:
                        text = await transcribe_file(
                            fp,
                            language=self._stt_language,
                            model=self._stt_model,
                            device=self._stt_device,
                            compute_type=self._stt_compute_type,
                        )
                    except Exception as exc:
                        self._trace_event("stt_error", file_path=fp, error=str(exc))
                        continue
                    if text:
                        transcripts.append(text)
                        transcribed_files.add(fp)
                        self._trace_event(
                            "stt_transcribed",
                            file_path=fp,
                            text_len=len(text),
                        )
                # Non-audio files are silently skipped — no trace event
                # to avoid log noise when messages contain many images.
            if transcripts:
                prefix = "\n".join(transcripts)
                raw.text = (prefix + "\n" + raw.text).strip() if raw.text else prefix
                # Remove annotations for transcribed files (exact path match)
                # so the agent does not attempt to process the audio file itself
                raw.content_annotations = [
                    a
                    for a in raw.content_annotations
                    if not any(
                        fp == a or a.endswith(f": {fp}]") or a == f"[voice: {fp}]"
                        for fp in transcribed_files
                    )
                ]

        msg = await self._build_inbound_async(raw)
        if msg is None:
            return
        if raw.message_id:
            try:
                with self._trace_span(
                    "ack_reaction",
                    chat_id=raw.chat_id,
                    message_id=raw.message_id,
                ):
                    await self._send_ack_reaction(raw.chat_id, raw.message_id)
            except Exception:
                pass
        self._trace_debug(
            "%s inbound queued: sender=%s chat=%s message_id=%s media=%d content_len=%d",
            self.name,
            msg.sender_id,
            msg.chat_id,
            msg.message_id or "-",
            len(msg.media),
            len(msg.content),
        )
        await self._queue.put(msg)

    # ── Bus integration ──────────────────────────────────────────────

    def set_bus(self, bus) -> None:
        """Inject the MessageBus reference (called by ChannelManager)."""
        self._bus = bus

    async def queue_message(self, msg: InboundMessage) -> None:
        """Buffer *msg* with debounce, then publish to bus."""
        sender = msg.sender_id

        if sender not in self._message_buffers:
            self._message_buffers[sender] = []
            self._message_metadata[sender] = msg.metadata
            self._message_media[sender] = []
            self._message_is_group[sender] = msg.is_group
            self._message_was_mentioned[sender] = msg.was_mentioned
        self._message_buffers[sender].append(msg.content)
        if msg.message_id:
            self._message_ids[sender] = msg.message_id
        if msg.media:
            self._message_media[sender].extend(msg.media)

        if self._on_activity:
            try:
                self._on_activity(sender, "received")
            except Exception:
                pass

        if sender in self._debounce_tasks:
            self._debounce_tasks[sender].cancel()

        msg_count = len(self._message_buffers[sender])
        wait = min(
            self.initial_debounce + (msg_count - 1) * self.debounce_step,
            self.max_debounce,
        )
        self._trace_event(
            "inbound_debounce_scheduled",
            sender_id=sender,
            chat_id=msg.chat_id,
            message_id=msg.message_id or "-",
            message_count=msg_count,
            wait_s=round(wait, 2),
            media_count=len(msg.media),
        )
        _logger.debug(f"Debounce for {sender}: {wait:.1f}s (message #{msg_count})")

        async def debounce_callback(_s=sender, _w=wait):
            await asyncio.sleep(_w)
            try:
                await self._process_buffered_messages(_s)
            except Exception as e:
                _logger.error(f"{self.name} debounce flush error for {_s}: {e}")

        self._debounce_tasks[sender] = asyncio.create_task(debounce_callback())

    async def _process_buffered_messages(self, sender: str) -> None:
        """Flush buffered messages for *sender* and publish to bus."""
        if sender not in self._message_buffers:
            return

        messages = self._message_buffers.pop(sender, [])
        metadata = self._message_metadata.pop(sender, None)
        media = self._message_media.pop(sender, [])
        message_id = self._message_ids.pop(sender, "")
        is_group = self._message_is_group.pop(sender, False)
        was_mentioned = self._message_was_mentioned.pop(sender, True)
        self._debounce_tasks.pop(sender, None)
        if not messages:
            return

        merged_content = "\n".join(messages)
        self._trace_event(
            "inbound_debounce_flush",
            sender_id=sender,
            chat_id=(metadata or {}).get("chat_id", sender),
            message_id=message_id or "-",
            message_count=len(messages),
            merged_content_len=len(merged_content),
            media_count=len(media),
            is_group=is_group,
            was_mentioned=was_mentioned,
        )
        _logger.info(f"Processing {len(messages)} merged message(s) from {sender}")

        if self._bus:
            chat_id = (metadata or {}).get("chat_id", sender)
            inbound = InboundMessage(
                channel=self.name,
                sender_id=sender,
                chat_id=str(chat_id),
                content=merged_content,
                media=media,
                metadata=metadata or {},
                message_id=message_id,
                is_group=is_group,
                was_mentioned=was_mentioned,
            )
            self._trace_event(
                "inbound_bus_publish",
                sender_id=sender,
                chat_id=inbound.chat_id,
                message_id=message_id or "-",
                media_count=len(media),
            )
            await self._bus.publish_inbound(inbound)

    async def _send_status_message(
        self,
        sender: str,
        content: str,
        metadata: dict | None = None,
    ) -> None:
        """Send a status/intermediate message to the channel."""
        chat_id = (metadata or {}).get("chat_id", sender)
        await self.send(
            OutboundMessage(
                channel=self.name,
                chat_id=str(chat_id),
                content=content,
                metadata=metadata or {},
            )
        )

    async def send_thinking_message(
        self,
        sender: str,
        thinking: str,
        metadata: dict | None = None,
    ) -> None:
        """Send a thinking intermediate message to the channel."""
        if not self.send_thinking:
            return
        await self._send_status_message(
            sender, f"\U0001f9e0\n{thinking}\n\u23f3", metadata
        )

    async def send_todo_message(
        self,
        sender: str,
        content: str,
        metadata: dict | None = None,
    ) -> None:
        """Send a todo list intermediate message to the channel."""
        await self._send_status_message(sender, content, metadata)

    async def run(self) -> None:
        """Run the channel with auto-reconnect (exponential backoff)."""
        backoff = 1.0
        max_backoff = 60.0
        self._running = True
        self._trace_event("channel_run_start")
        while self._running:
            try:
                await self.start()
                self._trace_event("channel_start_ok")
                backoff = 1.0
                async for msg in self.receive():
                    self._trace_event(
                        "channel_receive",
                        sender_id=msg.sender_id,
                        chat_id=msg.chat_id,
                        message_id=msg.message_id or "-",
                        content_len=len(msg.content),
                        media_count=len(msg.media),
                        is_group=msg.is_group,
                        was_mentioned=msg.was_mentioned,
                    )
                    await self.queue_message(msg)
            except asyncio.CancelledError:
                self._trace_event("channel_run_cancelled")
                break
            except ChannelError as e:
                self._trace_event(
                    "channel_fatal_error",
                    error_type=type(e).__name__,
                )
                _logger.error(f"Channel {self.name} fatal error: {e}")
                self._running = False
                break
            except Exception as e:
                self._trace_event(
                    "channel_runtime_error",
                    error_type=type(e).__name__,
                )
                _logger.error(f"Channel {self.name} error: {e}")
            finally:
                # Preserve reconnect intent across stop()
                should_reconnect = self._running
                try:
                    await self.stop()
                except Exception:
                    pass
                self._running = should_reconnect

            if self._running:
                self._trace_event(
                    "channel_reconnect_scheduled",
                    backoff_s=round(backoff, 2),
                )
                _logger.info(f"Reconnecting {self.name} in {backoff:.1f}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

    # ── Channel allow-list check ─────────────────────────────────────

    def is_channel_allowed(self, channel_id: str) -> bool:
        """Return ``True`` if *channel_id* is permitted by config.

        When the allow-list is empty or absent every channel is allowed.
        """
        allowed = getattr(self.config, "allowed_channels", None)
        return not allowed or str(channel_id) in allowed

    # ── Sender allow-list check ──────────────────────────────────────

    def is_allowed(self, sender: str) -> bool:
        """Check if *sender* is permitted by ``self.config.allowed_senders``.

        Returns ``True`` when the allow-list is empty / None (open access).
        Supports ``|``-separated composite IDs (e.g. ``"uid|gid"``).
        Subclasses with richer filtering (iMessage) may override.
        """
        config = getattr(self, "config", None)
        allowed = getattr(config, "allowed_senders", None) if config else None
        if not allowed:
            return True
        sender_str = str(sender)
        if sender_str in allowed:
            return True
        if "|" in sender_str:
            for part in sender_str.split("|"):
                if part and part in allowed:
                    return True
        return False


class ChannelError(Exception):
    """Base exception for channel-related errors."""

    pass
