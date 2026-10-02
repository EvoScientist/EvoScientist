"""Small helpers for raw LangGraph v3 stream payloads."""

import re
from collections.abc import Mapping
from typing import cast

RawMap = Mapping[str, object]

_THINKING_TAG_RE = re.compile(r"<thinking>.*?</thinking>", re.DOTALL)


def _as_raw_map(value: object) -> RawMap | None:
    if not isinstance(value, Mapping):
        return None
    if not all(isinstance(key, str) for key in value):
        return None
    return cast("RawMap", value)


def _strip_legacy_thinking_tags(content: str) -> str:
    """Remove ``<thinking>...</thinking>`` tags from content strings."""
    return _THINKING_TAG_RE.sub("", content)


def _event_namespace(event: Mapping[str, object]) -> tuple[str, ...]:
    params = _as_raw_map(event.get("params")) or {}
    namespace = params.get("namespace")
    if not isinstance(namespace, list | tuple):
        return ()
    return tuple(str(part) for part in namespace)


def _event_data(event: Mapping[str, object]) -> object:
    params = _as_raw_map(event.get("params")) or {}
    return params.get("data")


def _split_message_event_data(data: object) -> tuple[object, RawMap]:
    if isinstance(data, tuple) and len(data) >= 2:
        metadata = _as_raw_map(data[1]) or {}
        return data[0], metadata
    return data, {}


def _usage_counts(usage: Mapping[str, object] | None) -> tuple[int, int]:
    if not usage:
        return 0, 0
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    return (
        input_tokens if isinstance(input_tokens, int) else 0,
        output_tokens if isinstance(output_tokens, int) else 0,
    )


def _text_from_content(content: object) -> str:
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


def _reasoning_from_content(content: object) -> str:
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        block_map = _as_raw_map(block)
        if block_map is not None:
            reasoning = block_map.get("reasoning") or block_map.get("thinking")
            if isinstance(reasoning, str):
                parts.append(reasoning)
    return "".join(parts)
