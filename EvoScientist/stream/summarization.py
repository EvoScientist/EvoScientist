"""Summarization event helpers for stream v3 updates."""

import re
from collections.abc import Mapping

from langchain_core.messages import BaseMessage

from .v3_payloads import RawMap, _as_raw_map

_SUMMARY_TAG_RE = re.compile(
    r"<summary>\s*(.*?)\s*</summary>", re.DOTALL | re.IGNORECASE
)


def _extract_summarization_text(msg: BaseMessage) -> str:
    """Extract plain text from a summarization message or chunk."""
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue
            block_map = _as_raw_map(block)
            if block_map is not None:
                text = block_map.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _extract_summary_message_text(summary_message: BaseMessage | None) -> str:
    """Extract user-facing summary text from a stored summarization event."""
    if summary_message is None:
        return ""
    text = _extract_summarization_text(summary_message)
    if not text:
        return ""

    match = _SUMMARY_TAG_RE.search(text)
    if match:
        return match.group(1).strip()

    prefix = "Here is a summary of the conversation to date:"
    if text.startswith(prefix):
        return text[len(prefix) :].strip()

    return text.strip()


def _find_summarization_event_payload(data: object) -> RawMap | None:
    """Find a `_summarization_event` dict anywhere inside an updates payload."""
    seen: set[int] = set()
    stack: list[object] = [data]

    while stack:
        item = stack.pop()
        item_id = id(item)
        if item_id in seen:
            continue
        seen.add(item_id)

        item_map = _as_raw_map(item)
        if item_map is not None:
            event = _as_raw_map(item_map.get("_summarization_event"))
            if event is not None:
                return event
            stack.extend(item_map.values())
            continue

        if isinstance(item, list | tuple):
            stack.extend(item)

    return None


def _summarization_event_signature(
    event: Mapping[str, object] | None,
) -> tuple[object, ...] | None:
    """Build a stable signature for a persisted summarization event."""
    if event is None:
        return None
    summary_message = event.get("summary_message")
    summary_text = _extract_summary_message_text(
        summary_message if isinstance(summary_message, BaseMessage) else None
    )
    return (
        event.get("cutoff_index"),
        event.get("file_path"),
        summary_text,
    )
