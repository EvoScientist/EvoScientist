"""Shared session status bar helpers for CLI and TUI frontends."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.utils import count_tokens_approximately

from ..llm.context_window import (
    DEFAULT_CONTEXT_WINDOW_FALLBACK,
    resolve_context_window,
)
from ..sessions import get_thread_messages

_FALLBACK_CONTEXT_WINDOW = DEFAULT_CONTEXT_WINDOW_FALLBACK
STATUS_BAR_BG = "#171a20"
STATUS_TEXT = "#cbd5e1"
STATUS_STRONG = "#e5e7eb"
STATUS_DIM = "#7c8594"
STATUS_GOOD = "#5fcf8b"
STATUS_WARN = "#d7b45a"
STATUS_BAD = "#d08c61"
STATUS_CRITICAL = "#d86f6f"
STATUS_HINT_IDLE = "#8b9bb0"
STATUS_HINT_BUSY = "#f0c36a"

# Braille spinner frames used by the CLI bottom toolbar and TUI status bar
# to animate the "Loading MCP tools" indicator.
SPINNER_FRAMES = "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"


@dataclass(slots=True)
class SessionStatusSnapshot:
    """Current session metrics shown in the persistent status bar."""

    model_full: str
    model_short: str
    context_tokens: int
    context_window: int
    context_percent: int
    context_source: str = "estimated"


def _percent_from_context(context_tokens: int, context_window: int) -> int:
    """Convert token counts into a clamped percent value."""
    if context_window <= 0:
        return 0
    return max(0, min(100, round((context_tokens / context_window) * 100)))


def _get_default_chat_model() -> Any:
    """Resolve the default chat model lazily to avoid import cycles."""
    from ..EvoScientist import _ensure_chat_model

    return _ensure_chat_model()


def _resolve_model_name(model_name: str | None, model_obj: Any | None) -> str:
    """Best-effort model name resolution for display."""
    if model_name:
        return str(model_name)
    if model_obj is None:
        model_obj = _get_default_chat_model()
    for attr in ("model_name", "model", "name"):
        value = getattr(model_obj, attr, None)
        if value:
            return str(value)
    return "unknown"


def _resolve_context_window(model_obj: Any | None) -> int:
    """Resolve the model context window with a safe fallback."""
    if model_obj is None:
        model_obj = _get_default_chat_model()
    return resolve_context_window(model_obj, fallback=_FALLBACK_CONTEXT_WINDOW)


def shorten_model_name(model_name: str, max_len: int = 26) -> str:
    """Shorten provider-prefixed model names for compact display."""
    short = (model_name or "unknown").split("/")[-1]
    if len(short) > max_len:
        return f"{short[: max_len - 3]}..."
    return short


def format_token_count_compact(value: int) -> str:
    """Format large token counts into a compact human-readable form."""
    abs_value = abs(int(value))
    if abs_value >= 1_000_000:
        num = value / 1_000_000
        suffix = "M"
    elif abs_value >= 1_000:
        num = value / 1_000
        suffix = "K"
    else:
        return str(value)

    if num.is_integer():
        return f"{int(num)}{suffix}"
    return f"{num:.1f}{suffix}"


def format_duration_compact(started_at: datetime, now: datetime | None = None) -> str:
    """Format elapsed wall time into a compact duration string."""
    current = now or datetime.now()
    seconds = max(0, int((current - started_at).total_seconds()))
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h"
    days = hours // 24
    return f"{days}d"


def status_style_name(percent_used: int) -> str:
    """Map utilization percent to shared status bar style buckets."""
    if percent_used >= 95:
        return "critical"
    if percent_used > 80:
        return "bad"
    if percent_used >= 50:
        return "warn"
    return "good"


def build_context_bar(percent_used: int, width: int = 10) -> str:
    """Build a compact visual context progress bar."""
    safe_percent = max(0, min(100, int(percent_used)))
    filled = round((safe_percent / 100) * width)
    body = ("█" * filled) + ("░" * max(0, width - filled))
    return f"[{body}]"


def _display_width(text: str) -> int:
    try:
        from prompt_toolkit.utils import get_cwidth

        return get_cwidth(text or "")
    except Exception:
        return len(text or "")


def trim_status_text(text: str, max_width: int) -> str:
    """Trim status-bar content to fit a single terminal row."""
    if max_width <= 0:
        return ""
    if _display_width(text) <= max_width:
        return text

    ellipsis = "..."
    ellipsis_width = _display_width(ellipsis)
    if max_width <= ellipsis_width:
        return ellipsis[:max_width]

    try:
        from prompt_toolkit.utils import get_cwidth
    except Exception:
        get_cwidth = None

    out: list[str] = []
    width = 0
    for ch in text:
        ch_width = get_cwidth(ch) if get_cwidth else len(ch)
        if width + ch_width + ellipsis_width > max_width:
            break
        out.append(ch)
        width += ch_width
    return "".join(out).rstrip() + ellipsis


def build_status_fragments(
    snapshot: SessionStatusSnapshot,
    started_at: datetime,
    width: int,
) -> list[tuple[str, str]]:
    """Build prompt_toolkit formatted-text fragments for the status bar."""
    duration_label = format_duration_compact(started_at)
    percent = snapshot.context_percent
    percent_label = f"{percent}%"
    if width < 52:
        frags = [
            ("class:status-bar-strong", snapshot.model_short),
            ("class:status-bar-dim", " · "),
            ("class:status-bar-dim", duration_label),
            ("class:status-bar", " "),
        ]
    elif width < 76:
        frags = [
            ("class:status-bar-strong", snapshot.model_short),
            ("class:status-bar-dim", " · "),
            (f"class:status-bar-{status_style_name(percent)}", percent_label),
            ("class:status-bar-dim", " · "),
            ("class:status-bar-dim", duration_label),
            ("class:status-bar", " "),
        ]
    else:
        context_label = (
            f"{format_token_count_compact(snapshot.context_tokens)}/"
            f"{format_token_count_compact(snapshot.context_window)}"
        )
        bucket = status_style_name(percent)
        frags = [
            ("class:status-bar-strong", snapshot.model_short),
            ("class:status-bar-dim", " │ "),
            ("class:status-bar-dim", context_label),
            ("class:status-bar-dim", " │ "),
            (f"class:status-bar-{bucket}", build_context_bar(percent)),
            ("class:status-bar-dim", " "),
            (f"class:status-bar-{bucket}", percent_label),
            ("class:status-bar-dim", " │ "),
            ("class:status-bar-dim", duration_label),
            ("class:status-bar", " "),
        ]

    total_width = sum(_display_width(text) for _, text in frags)
    if total_width > width:
        plain_text = "".join(text for _, text in frags)
        return [("class:status-bar", trim_status_text(plain_text, width))]
    return frags


def build_status_text(
    snapshot: SessionStatusSnapshot,
    started_at: datetime,
    width: int,
):
    """Build a Rich Text object for the persistent TUI status bar."""
    from rich.text import Text

    rich_styles = {
        "status-bar": f"on {STATUS_BAR_BG} {STATUS_TEXT}",
        "status-bar-strong": f"on {STATUS_BAR_BG} {STATUS_STRONG} bold",
        "status-bar-dim": f"on {STATUS_BAR_BG} {STATUS_DIM}",
        "status-bar-good": f"on {STATUS_BAR_BG} {STATUS_GOOD} bold",
        "status-bar-warn": f"on {STATUS_BAR_BG} {STATUS_WARN} bold",
        "status-bar-bad": f"on {STATUS_BAR_BG} {STATUS_BAD} bold",
        "status-bar-critical": f"on {STATUS_BAR_BG} {STATUS_CRITICAL} bold",
    }
    text = Text(no_wrap=True, overflow="crop")
    for style, content in build_status_fragments(snapshot, started_at, width):
        rich_style = rich_styles.get(
            style.removeprefix("class:"),
            f"on {STATUS_BAR_BG} {STATUS_TEXT}",
        )
        text.append(content, style=rich_style)
    return text


def make_empty_status_snapshot(
    model_name: str | None = None, model_obj: Any | None = None
) -> SessionStatusSnapshot:
    """Build a placeholder snapshot before async context counting completes."""
    resolved_name = _resolve_model_name(model_name, model_obj)
    window = _resolve_context_window(model_obj)
    return SessionStatusSnapshot(
        model_full=resolved_name,
        model_short=shorten_model_name(resolved_name),
        context_tokens=0,
        context_window=window,
        context_percent=0,
        context_source="estimated",
    )


def make_usage_status_snapshot(
    input_tokens: int,
    *,
    model_name: str | None = None,
    model_obj: Any | None = None,
) -> SessionStatusSnapshot:
    """Build a snapshot from the last real model input usage."""
    resolved_name = _resolve_model_name(model_name, model_obj)
    window = _resolve_context_window(model_obj)
    context_tokens = max(0, int(input_tokens))
    return SessionStatusSnapshot(
        model_full=resolved_name,
        model_short=shorten_model_name(resolved_name),
        context_tokens=context_tokens,
        context_window=window,
        context_percent=_percent_from_context(context_tokens, window),
        context_source="usage",
    )


def estimate_message_tokens(
    text: str,
    *,
    message_type: str = "ai",
) -> int:
    """Estimate tokens for a single in-flight message fragment."""
    content = (text or "").strip()
    if not content:
        return 0

    try:
        if message_type == "human":
            messages = [HumanMessage(content=content)]
        else:
            messages = [AIMessage(content=content)]
        return int(count_tokens_approximately(messages))
    except Exception:
        return 0


def apply_assistant_text_to_snapshot(
    snapshot: SessionStatusSnapshot,
    assistant_text: str | None,
) -> SessionStatusSnapshot:
    """Overlay in-flight assistant output on top of a base snapshot."""
    extra_tokens = estimate_message_tokens(assistant_text or "", message_type="ai")
    if extra_tokens <= 0:
        return snapshot

    context_tokens = snapshot.context_tokens + extra_tokens
    return replace(
        snapshot,
        context_tokens=context_tokens,
        context_percent=_percent_from_context(context_tokens, snapshot.context_window),
    )


def apply_user_text_to_snapshot(
    snapshot: SessionStatusSnapshot,
    user_text: str | None,
) -> SessionStatusSnapshot:
    """Overlay pending user input on top of an existing snapshot."""
    extra_tokens = estimate_message_tokens(user_text or "", message_type="human")
    if extra_tokens <= 0:
        return snapshot

    context_tokens = snapshot.context_tokens + extra_tokens
    return replace(
        snapshot,
        context_tokens=context_tokens,
        context_percent=_percent_from_context(context_tokens, snapshot.context_window),
    )


async def build_session_status_snapshot(
    thread_id: str,
    *,
    model_name: str | None = None,
    model_obj: Any | None = None,
    pending_user_text: str | None = None,
) -> SessionStatusSnapshot:
    """Count current thread context and return a display snapshot."""
    resolved_name = _resolve_model_name(model_name, model_obj)
    window = _resolve_context_window(model_obj)
    messages = list(await get_thread_messages(thread_id))

    pending = (pending_user_text or "").strip()
    if pending:
        messages.append(HumanMessage(content=pending))

    try:
        context_tokens = int(count_tokens_approximately(messages)) if messages else 0
    except Exception:
        context_tokens = 0

    percent = _percent_from_context(context_tokens, window)

    return SessionStatusSnapshot(
        model_full=resolved_name,
        model_short=shorten_model_name(resolved_name),
        context_tokens=context_tokens,
        context_window=window,
        context_percent=percent,
        context_source="estimated",
    )
