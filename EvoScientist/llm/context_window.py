"""Helpers for resolving model context windows across LangChain providers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

DEFAULT_CONTEXT_WINDOW_FALLBACK = 200_000

_DIRECT_WINDOW_ATTRS = (
    "context_window",
    "context_length",
    "num_ctx",
    "max_input_tokens",
)
_CONTAINER_ATTRS = (
    "profile",
    "context_management",
    "model_kwargs",
    "metadata",
)


def _coerce_positive_int(value: Any) -> int | None:
    """Best-effort coercion for positive integer-like values."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        return int(value) if value > 0 else None
    if isinstance(value, str):
        normalized = value.strip().replace(",", "").replace("_", "")
        if normalized.isdigit():
            parsed = int(normalized)
            return parsed if parsed > 0 else None
    return None


def _resolve_from_mapping(mapping: Mapping[str, Any]) -> int | None:
    """Resolve a context window from a metadata mapping."""
    for key in _DIRECT_WINDOW_ATTRS:
        if key in mapping:
            resolved = _coerce_positive_int(mapping.get(key))
            if resolved is not None:
                return resolved
    return None


def get_context_window(model_obj: Any | None) -> int | None:
    """Return the best available context-window value from a model object."""
    if model_obj is None:
        return None

    for attr in _DIRECT_WINDOW_ATTRS:
        resolved = _coerce_positive_int(getattr(model_obj, attr, None))
        if resolved is not None:
            return resolved

    for attr in _CONTAINER_ATTRS:
        candidate = getattr(model_obj, attr, None)
        if isinstance(candidate, Mapping):
            resolved = _resolve_from_mapping(candidate)
            if resolved is not None:
                return resolved

    return None


def resolve_context_window(
    model_obj: Any | None,
    *,
    fallback: int = DEFAULT_CONTEXT_WINDOW_FALLBACK,
) -> int:
    """Resolve a usable context window with a stable fallback."""
    resolved = get_context_window(model_obj)
    if resolved is not None:
        return resolved
    return fallback
