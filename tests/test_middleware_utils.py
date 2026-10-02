"""Unit tests for EvoScientist.middleware.utils helpers.

Focuses on the ``system_message`` composition primitives shared across
middleware modules. Model-side helpers (``disable_thinking``,
``disable_streaming``) are covered by the middleware suites that use them.
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage

from EvoScientist.middleware.utils import (
    replace_block_by_sentinel,
)


class TestReplaceBlockBySentinel:
    """``replace_block_by_sentinel`` swaps the block containing the sentinel
    for a replacement text block, preserving every other block. Used by
    ``ExpertSkillLoaderMiddleware`` to inject the persona in place of the
    graph's fallback block while keeping the base-stack sections intact."""

    _SENTINEL = "__TEST_PERSONA_SLOT__"

    def test_swaps_matching_block(self):
        original = SystemMessage(
            content=[
                {"type": "text", "text": f"{self._SENTINEL}\n\nfallback"},
                {"type": "text", "text": "## `task` (subagent spawner)"},
                {"type": "text", "text": "## Skills System"},
            ]
        )
        result = replace_block_by_sentinel(original, self._SENTINEL, "persona body")
        assert result is not None
        block_texts = [b.get("text", "") for b in result.content_blocks]
        assert block_texts == [
            "persona body",
            "## `task` (subagent spawner)",
            "## Skills System",
        ]

    def test_preserves_block_count(self):
        original = SystemMessage(
            content=[
                {"type": "text", "text": self._SENTINEL},
                {"type": "text", "text": "witness"},
            ]
        )
        result = replace_block_by_sentinel(original, self._SENTINEL, "persona")
        assert result is not None
        assert len(list(result.content_blocks)) == len(list(original.content_blocks))

    def test_returns_none_when_sentinel_missing(self):
        """Signal path: caller decides fallback policy (typically log + append)
        so a deepagents refactor degrades gracefully instead of hard-failing."""
        original = SystemMessage(
            content=[
                {"type": "text", "text": "## `task` (subagent spawner)"},
                {"type": "text", "text": "## Skills System"},
            ]
        )
        assert replace_block_by_sentinel(original, self._SENTINEL, "persona") is None

    def test_returns_none_when_message_is_none(self):
        assert replace_block_by_sentinel(None, self._SENTINEL, "persona") is None
