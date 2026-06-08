"""Tool result extraction helpers for stream events."""

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from .v3_payloads import _as_raw_map

_IMAGE_MEDIA_TYPES = {
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
    "image/bmp",
    "image/svg+xml",
}


def _extract_tool_content(msg: ToolMessage) -> tuple[str, bool]:
    """Extract display-safe content from a ToolMessage.

    DeepAgents ``read_file`` returns image content as
    ``ToolMessage(content=[ImageContentBlock])`` with
    ``additional_kwargs["read_file_media_type"]`` set. Stringifying that would
    dump huge base64 data into the display.
    """
    additional = msg.additional_kwargs
    media_type = additional.get("read_file_media_type", "")
    if media_type and media_type in _IMAGE_MEDIA_TYPES:
        file_path = additional.get("read_file_path", "")
        if not file_path:
            file_path = msg.name or "image"
        return f"[OK] Image displayed: {file_path} ({media_type})", True

    content = msg.content
    if isinstance(content, list):
        for block in content:
            block_map = _as_raw_map(block)
            if block_map is not None:
                if block_map.get("type") == "image" or "base64" in block_map:
                    return "[OK] Image displayed", True

        parts: list[str] = []
        for block in content:
            block_map = _as_raw_map(block)
            if block_map is not None:
                text = block_map.get("text")
                if text:
                    parts.append(str(text))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts) if parts else str(content), False

    return str(content), False


def _extract_command_tool_content(output: Command, tool_call_id: str) -> str | None:
    if not tool_call_id:
        return None
    update = _as_raw_map(output.update)
    if update is None:
        return None
    messages = update.get("messages")
    if not isinstance(messages, list):
        return None
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        if msg.tool_call_id != tool_call_id:
            continue
        content, _ = _extract_tool_content(msg)
        return content
    return None
