"""Helpers for harness-added user-side messages and ``/skill-name`` input.

Kept import-light: sessions, the gateway, memory and middleware use it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

PINNED_SKILL_SOURCE = "pinned_skill"

# Agent Skills spec: lowercase letters, digits and single hyphens, 1-64 chars.
_SKILL_SLASH = re.compile(r"/([a-z0-9]+(?:-[a-z0-9]+)*)")
_MAX_SKILL_NAME = 64


def _field(message: Any, key: str) -> Any:
    if isinstance(message, Mapping):
        return message.get(key)
    return getattr(message, key, None)


def _additional_kwargs(message: Any) -> Mapping[str, Any]:
    kwargs = _field(message, "additional_kwargs")
    return kwargs if isinstance(kwargs, Mapping) else {}


def message_source(message: Any) -> str | None:
    """Return the ``lc_source`` tag of a harness-added message, if any."""
    source = _additional_kwargs(message).get("lc_source")
    return source if isinstance(source, str) else None


def is_pinned_skill(message: Any) -> bool:
    return message_source(message) == PINNED_SKILL_SOURCE


def pinned_skill_name(message: Any) -> str | None:
    if not is_pinned_skill(message):
        return None
    skill = _additional_kwargs(message).get("skill")
    name = skill.get("name") if isinstance(skill, Mapping) else None
    return name if isinstance(name, str) and name else None


def message_text(content: Any) -> str:
    """Join the text of a message content (a string or content blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, Mapping) and block.get("type") == "text"
        )
    return ""


def parse_skill_slashes(text: str) -> list[str]:
    """Return the skill names given by the leading ``/name`` words of *text*."""
    names: list[str] = []
    for word in text.split():
        match = _SKILL_SLASH.fullmatch(word)
        if match is None or len(match.group(1)) > _MAX_SKILL_NAME:
            break
        if match.group(1) not in names:
            names.append(match.group(1))
    return names
