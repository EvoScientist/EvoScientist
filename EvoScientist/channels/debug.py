"""Shared debug logging helpers for channel integrations.

This module is intentionally channel-agnostic. Future per-channel PRs should
reuse these helpers instead of reintroducing ad-hoc logging formats or
standalone ``basicConfig`` calls.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"
_REDACTED = "***"
_PAYLOAD_PREVIEW_LIMIT = 512
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


def resolve_log_level(raw: str | None, default: int = logging.WARNING) -> int:
    """Resolve a logging level name to the corresponding ``logging`` constant."""
    value = (raw or "").strip().upper()
    if value == "WARN":
        value = "WARNING"
    return getattr(logging, value, default)


def configure_standalone_logging(log_level: str | None = None) -> int:
    """Configure logging for standalone channel entry points.

    Standalone channel servers historically configured logging independently
    and most of them hard-coded ``DEBUG``. Future channel PRs should call this
    helper from ``channels/<name>/serve.py`` so standalone mode follows the same
    environment/config-level contract as the main CLI.
    """

    resolved = resolve_log_level(log_level or os.environ.get("EVOSCIENTIST_LOG_LEVEL"))
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=resolved,
            format=_LOG_FORMAT,
            datefmt=_DATE_FORMAT,
        )
    else:
        root.setLevel(resolved)
        for handler in root.handlers:
            handler.setLevel(resolved)
    return resolved


@lru_cache(maxsize=1)
def _load_debug_flags() -> tuple[bool, bool]:
    """Load debug feature switches from config as a fallback to env vars."""
    try:
        from ..config.settings import load_config

        cfg = load_config()
        return (
            bool(getattr(cfg, "channel_debug_tracing", False)),
            bool(getattr(cfg, "channel_debug_payloads", False)),
        )
    except Exception:
        return (False, False)


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
    cfg_trace, _cfg_payloads = _load_debug_flags()
    return cfg_trace


def debug_payloads_enabled(enabled: bool | None = None) -> bool:
    """Resolve the optional raw-payload debugging switch."""
    if enabled is not None:
        return bool(enabled)
    raw = os.environ.get("EVOSCIENTIST_CHANNEL_DEBUG_PAYLOADS", "")
    if raw.strip():
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    _cfg_trace, cfg_payloads = _load_debug_flags()
    return cfg_payloads


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
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return f"<seq:{len(value)}>"
    return str(value).replace("\n", "\\n")


def _redact_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in payload.items():
        if _should_redact_key(key):
            redacted[key] = _REDACTED
            continue
        if isinstance(value, Mapping):
            redacted[key] = _redact_mapping(value)
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            redacted[key] = [str(_stringify(item)) for item in value[:10]]
        else:
            redacted[key] = _stringify(value)
    return redacted


def _format_fields(fields: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key, value in fields.items():
        if value is None:
            continue
        safe_value = _REDACTED if _should_redact_key(key) else _stringify(value)
        parts.append(f"{key}={safe_value}")
    return " ".join(parts)


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

    if not enabled or not logger.isEnabledFor(logging.DEBUG):
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

    if not enabled or not logger.isEnabledFor(logging.DEBUG):
        return
    base_fields: dict[str, Any] = {"event": event}
    base_fields.update(fields)
    logger.debug(_format_fields(base_fields))


def emit_debug_payload(
    logger: logging.Logger,
    event: str,
    payload: Any,
    *,
    channel: str,
    enabled: bool,
    preview_limit: int = _PAYLOAD_PREVIEW_LIMIT,
    **fields: Any,
) -> None:
    """Emit a redacted payload preview for deep debugging sessions."""

    if not enabled or not logger.isEnabledFor(logging.DEBUG):
        return

    preview: str
    if isinstance(payload, Mapping):
        preview = str(_redact_mapping(payload))
    else:
        preview = _stringify(payload)

    if len(preview) > preview_limit:
        preview = preview[:preview_limit] + "..."

    emit_debug_event(
        logger,
        event,
        channel=channel,
        enabled=enabled,
        payload_preview=preview,
        **fields,
    )
