"""Unified formatting pipeline for all channels.

Internal representation is Markdown. This module converts Markdown to
each platform's native format: HTML, Slack mrkdwn, Discord Markdown,
or plain text.

Channels no longer need per-file format functions — they just declare
``capabilities.format_type`` and the base class auto-configures a
``UnifiedFormatter`` instance.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import ClassVar

# ═════════════════════════════════════════════════════════════════════
# Markdown conversion engine (formerly markdown_utils.py)
# ═════════════════════════════════════════════════════════════════════

_PLACEHOLDER_PREFIX = "\x00BLOCK"
_INLINE_PREFIX = "\x00INLINE"

# A formatting rule: (regex_pattern, replacement)
InlineRule = tuple[str, str]


def convert_markdown(
    text: str,
    *,
    code_block_formatter: Callable[[str, str], str],
    inline_code_formatter: Callable[[str], str],
    inline_rules: list[InlineRule],
    escape_fn: Callable[[str], str] | None = None,
) -> str:
    """Convert Markdown to a channel-specific format.

    Parameters
    ----------
    text:
        Input Markdown text.
    code_block_formatter:
        ``(language, code) -> str`` — format a fenced code block.
    inline_code_formatter:
        ``(code) -> str`` — format an inline code span.
    inline_rules:
        List of ``(pattern, replacement)`` pairs applied in order to the
        remaining text (after code extraction and optional escaping).
    escape_fn:
        Optional function applied to the non-code text *before* inline
        rules.  Useful for HTML-escaping (Telegram) or other channel-
        specific character escaping.

    Returns
    -------
    str
        The converted text.
    """
    # 1. Extract and protect fenced code blocks (```...```)
    code_blocks: list[str] = []

    def _save_code_block(m: re.Match) -> str:
        lang = m.group(1) or ""
        code = m.group(2)
        formatted = code_block_formatter(lang, code)
        idx = len(code_blocks)
        code_blocks.append(formatted)
        return f"{_PLACEHOLDER_PREFIX}{idx}\x00"

    text = re.sub(r"```(\w*)\n?(.*?)```", _save_code_block, text, flags=re.DOTALL)

    # 2. Extract and protect inline code (`...`)
    inline_codes: list[str] = []

    def _save_inline(m: re.Match) -> str:
        code = m.group(1)
        formatted = inline_code_formatter(code)
        idx = len(inline_codes)
        inline_codes.append(formatted)
        return f"{_INLINE_PREFIX}{idx}\x00"

    text = re.sub(r"`([^`]+)`", _save_inline, text)

    # 3. Optional escaping of remaining text
    if escape_fn is not None:
        text = escape_fn(text)

    # 4. Apply inline formatting rules
    for pattern, replacement in inline_rules:
        text = re.sub(pattern, replacement, text, flags=re.MULTILINE)

    # 5. Restore code blocks and inline code
    for idx, html in enumerate(code_blocks):
        text = text.replace(f"{_PLACEHOLDER_PREFIX}{idx}\x00", html)
    for idx, code in enumerate(inline_codes):
        text = text.replace(f"{_INLINE_PREFIX}{idx}\x00", code)

    return text


# ═════════════════════════════════════════════════════════════════════
# Shared helpers
# ═════════════════════════════════════════════════════════════════════


def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _noop_escape(text: str) -> str:
    return text


# ═════════════════════════════════════════════════════════════════════
# HTML profile (Telegram, Email, Teams)
# ═════════════════════════════════════════════════════════════════════


def _html_code_block(lang: str, code: str) -> str:
    escaped = _escape_html(code)
    if lang:
        return f'<pre><code class="language-{lang}">{escaped}</code></pre>'
    return f"<pre><code>{escaped}</code></pre>"


def _html_inline_code(code: str) -> str:
    return f"<code>{_escape_html(code)}</code>"


_HTML_INLINE_RULES: list[InlineRule] = [
    # Headings → bold
    (r"^#{1,6}\s+(.+)$", r"<b>\1</b>"),
    # Blockquote markers (already escaped to &gt;)
    (r"^&gt;\s?", ""),
    # Links [text](url) → <a>
    (r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>'),
    # Bold **text** → <b>
    (r"\*\*(.+?)\*\*", r"<b>\1</b>"),
    # Italic _text_ → <i>
    (r"(?<!\w)_([^_]+?)_(?!\w)", r"<i>\1</i>"),
    # Strikethrough ~~text~~ → <s>
    (r"~~(.+?)~~", r"<s>\1</s>"),
    # List items
    (r"^[\-\*]\s+", "• "),
]


# ═════════════════════════════════════════════════════════════════════
# Slack mrkdwn profile
# ═════════════════════════════════════════════════════════════════════


def _slack_code_block(lang: str, code: str) -> str:
    return f"```\n{code}```"


def _slack_inline_code(code: str) -> str:
    return f"`{code}`"


_SLACK_INLINE_RULES: list[InlineRule] = [
    (r"^#{1,6}\s+(.+)$", r"*\1*"),
    (r"\[([^\]]+)\]\(([^)]+)\)", r"<\2|\1>"),
    (r"\*\*(.+?)\*\*", r"*\1*"),
    (r"~~(.+?)~~", r"~\1~"),
    (r"^[\-\*]\s+", "• "),
]


# ═════════════════════════════════════════════════════════════════════
# Discord profile (mostly passthrough, headings → bold)
# ═════════════════════════════════════════════════════════════════════


def _discord_code_block(lang: str, code: str) -> str:
    return f"```{lang}\n{code}```"


def _discord_inline_code(code: str) -> str:
    return f"`{code}`"


_DISCORD_INLINE_RULES: list[InlineRule] = [
    (r"^#{1,6}\s+(.+)$", r"**\1**"),
]


# ═════════════════════════════════════════════════════════════════════
# Plain text profile (strip all formatting)
# ═════════════════════════════════════════════════════════════════════


def _plain_code_block(lang: str, code: str) -> str:
    return code


def _plain_inline_code(code: str) -> str:
    return code


_PLAIN_INLINE_RULES: list[InlineRule] = [
    (r"^#{1,6}\s+", ""),
    (r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)"),
    (r"\*\*(.+?)\*\*", r"\1"),
    (r"(?<!\w)_([^_]+?)_(?!\w)", r"\1"),
    (r"~~(.+?)~~", r"\1"),
    (r"^[\-\*]\s+", "• "),
]


# ═════════════════════════════════════════════════════════════════════
# Markdown passthrough profile (Feishu, DingTalk, WeCom)
# ═════════════════════════════════════════════════════════════════════


def _md_code_block(lang: str, code: str) -> str:
    return f"```{lang}\n{code}```"


def _md_inline_code(code: str) -> str:
    return f"`{code}`"


_MD_INLINE_RULES: list[InlineRule] = []  # passthrough — already Markdown


# ═════════════════════════════════════════════════════════════════════
# Unified Formatter
# ═════════════════════════════════════════════════════════════════════


class UnifiedFormatter:
    """Converts internal Markdown to a target platform format.

    Instantiated once per channel based on its ``capabilities.format_type``.
    """

    _PROFILES: ClassVar[dict[str, dict]] = {
        "html": {
            "code_block_formatter": _html_code_block,
            "inline_code_formatter": _html_inline_code,
            "inline_rules": _HTML_INLINE_RULES,
            "escape_fn": _escape_html,
        },
        "slack_mrkdwn": {
            "code_block_formatter": _slack_code_block,
            "inline_code_formatter": _slack_inline_code,
            "inline_rules": _SLACK_INLINE_RULES,
            "escape_fn": None,
        },
        "discord": {
            "code_block_formatter": _discord_code_block,
            "inline_code_formatter": _discord_inline_code,
            "inline_rules": _DISCORD_INLINE_RULES,
            "escape_fn": None,
        },
        "markdown": {
            "code_block_formatter": _md_code_block,
            "inline_code_formatter": _md_inline_code,
            "inline_rules": _MD_INLINE_RULES,
            "escape_fn": None,
        },
        "plain": {
            "code_block_formatter": _plain_code_block,
            "inline_code_formatter": _plain_inline_code,
            "inline_rules": _PLAIN_INLINE_RULES,
            "escape_fn": None,
        },
    }

    def __init__(self, format_type: str = "plain") -> None:
        self._format_type = format_type
        profile = self._PROFILES.get(format_type)
        if profile is None:
            raise ValueError(
                f"Unknown format_type: {format_type!r}. "
                f"Available: {list(self._PROFILES.keys())}"
            )
        self._profile = profile

    @property
    def format_type(self) -> str:
        return self._format_type

    def format(self, text: str) -> str:
        """Convert Markdown *text* to the target format."""
        if not text:
            return text
        return convert_markdown(text, **self._profile)

    @classmethod
    def for_channel(cls, format_type: str) -> UnifiedFormatter:
        """Factory: create a formatter for the given format type."""
        return cls(format_type)
