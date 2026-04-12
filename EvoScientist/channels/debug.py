"""Shared debug logging helpers for channel integrations.

This module is intentionally channel-agnostic. Future per-channel PRs should
reuse these helpers instead of reintroducing ad-hoc logging formats or
standalone ``basicConfig`` calls.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any

_REDACTED = "***"
_SECRET_TOKENS = (
    "token",
    "secret",
    "password",
    "authorization",
    "cookie",
    "api_key",
    "apikey",
    "access_key",
    "private_key",
    "signature",
)


def _load_debug_trace_flag() -> bool:
    """Load the trace feature switch from config as a fallback to env vars."""
    try:
        from ..config.settings import load_config

        return bool(getattr(load_config(), "channel_debug_tracing", False))
    except Exception:
        return False


def debug_trace_enabled(enabled: bool | None = None) -> bool:
    """Resolve the channel debug tracing switch.

    Explicit ``enabled`` takes precedence; otherwise the helper falls back to
    ``EVOSCIENTIST_CHANNEL_DEBUG_TRACING``.
    """

    if enabled is not None:
        return bool(enabled)
    raw = os.environ.get("EVOSCIENTIST_CHANNEL_DEBUG_TRACING", "")
    if raw.strip():
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return _load_debug_trace_flag()


def _should_redact_key(key: str) -> bool:
    lowered = key.lower()
    return any(token in lowered for token in _SECRET_TOKENS)


def _stringify(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if isinstance(value, str):
        return value.replace("\n", "\\n")
    if isinstance(value, Mapping):
        return f"<map:{len(value)}>"
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return f"<seq:{len(value)}>"
    return str(value).replace("\n", "\\n")


def _format_fields(fields: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key, value in fields.items():
        if value is None:
            continue
        safe_value = _REDACTED if _should_redact_key(key) else _stringify(value)
        parts.append(f"{key}={safe_value}")
    return " ".join(parts)


_warned_debug_level_mismatch = False


def _warn_debug_level_mismatch(logger: logging.Logger) -> None:
    """Emit a one-time warning when tracing is enabled but DEBUG logs are hidden."""
    global _warned_debug_level_mismatch
    if _warned_debug_level_mismatch:
        return
    _warned_debug_level_mismatch = True
    logger.warning(
        "channel debug tracing is enabled but logger level is above DEBUG; "
        "set EVOSCIENTIST_LOG_LEVEL=DEBUG, configure log_level=debug, "
        "or use 'serve --debug' to see trace events"
    )


def emit_debug_event(
    logger: logging.Logger,
    event: str,
    *,
    channel: str,
    enabled: bool,
    **fields: Any,
) -> None:
    """Emit a structured channel debug event.

    Example output:
    ``event=inbound_raw channel=telegram message_id=123 chat_id=-1001``
    """

    if not enabled:
        return
    if not logger.isEnabledFor(logging.DEBUG):
        _warn_debug_level_mismatch(logger)
        return
    base_fields = {"event": event, "channel": channel}
    base_fields.update(fields)
    logger.debug(_format_fields(base_fields))


def emit_debug_event_if(
    logger: logging.Logger,
    event: str,
    enabled: bool,
    **fields: Any,
) -> None:
    """Convenience wrapper for code that lacks a :class:`Channel` instance.

    Unlike :func:`emit_debug_event`, the ``channel`` field is not required —
    pass it via *fields* when available.  This is intended for middleware
    classes, managers, and standalone helpers.
    """

    if not enabled:
        return
    if not logger.isEnabledFor(logging.DEBUG):
        _warn_debug_level_mismatch(logger)
        return
    base_fields: dict[str, Any] = {"event": event}
    base_fields.update(fields)
    logger.debug(_format_fields(base_fields))


class TraceMixin:
    """Mixin providing unified structured trace helpers.

    Classes using this mixin must set ``_debug_trace`` (bool) and
    ``_trace_logger`` attribute returning a :class:`logging.Logger`.

    The trace name defaults to ``self.name`` if present, otherwise
    ``"unknown"``.  Override ``_trace_name`` to customise.
    """

    _debug_trace: bool
    _trace_logger: logging.Logger

    @property
    def _trace_name(self) -> str:
        return getattr(self, "name", "unknown")

    def _trace_event(self, event: str, **fields: Any) -> None:
        """Emit a structured debug event when tracing is enabled."""
        emit_debug_event(
            self._trace_logger,
            event,
            channel=self._trace_name,
            enabled=self._debug_trace,
            **fields,
        )
