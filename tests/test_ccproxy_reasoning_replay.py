"""Tests for ccproxy Codex cross-turn reasoning replay (#507).

The Codex backend returns reasoning items with ``encrypted_content`` on every
Responses API response.  These blocks must survive message flattening on the
ccproxy route so that ``reasoning.context = "all_turns"`` can replay them
across turns.  Plain-text reasoning blocks (no ``encrypted_content``) are
still dropped — they are display-only and cannot be passed back.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from EvoScientist.llm.patches import (
    _flatten_message_content,
    _has_encrypted_reasoning,
    _sanitize_messages,
)


# ---------------------------------------------------------------------------
# _has_encrypted_reasoning
# ---------------------------------------------------------------------------
class TestHasEncryptedReasoning:
    def test_direct_encrypted_content(self):
        block = {"type": "reasoning", "encrypted_content": "abc123"}
        assert _has_encrypted_reasoning(block) is True

    def test_empty_encrypted_content_is_false(self):
        block = {"type": "reasoning", "encrypted_content": ""}
        assert _has_encrypted_reasoning(block) is False

    def test_nested_reasoning_dict(self):
        block = {"type": "reasoning", "reasoning": {"encrypted_content": "xyz"}}
        assert _has_encrypted_reasoning(block) is True

    def test_nested_empty_is_false(self):
        block = {"type": "reasoning", "reasoning": {"encrypted_content": ""}}
        assert _has_encrypted_reasoning(block) is False

    def test_plain_text_reasoning_is_false(self):
        block = {"type": "reasoning", "text": "some thinking"}
        assert _has_encrypted_reasoning(block) is False

    def test_non_dict_is_false(self):
        assert _has_encrypted_reasoning("not a dict") is False  # type: ignore[arg-type]

    def test_thinking_block_is_false(self):
        block = {"type": "thinking", "thinking": "thoughts"}
        assert _has_encrypted_reasoning(block) is False


# ---------------------------------------------------------------------------
# _flatten_message_content — keep_encrypted_reasoning flag
# ---------------------------------------------------------------------------
class TestFlattenKeepsEncryptedReasoning:
    def test_default_drops_all_reasoning(self):
        """Without the flag, encrypted reasoning is still dropped (backward compat)."""
        content = [
            {"type": "text", "text": "hello"},
            {"type": "reasoning", "encrypted_content": "enc1"},
        ]
        result = _flatten_message_content(content)
        assert result == "hello"

    def test_keep_flag_preserves_encrypted_reasoning(self):
        content = [
            {"type": "text", "text": "hello"},
            {"type": "reasoning", "encrypted_content": "enc1"},
        ]
        result = _flatten_message_content(content, keep_encrypted_reasoning=True)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == {"type": "text", "text": "hello"}
        assert result[1] == {"type": "reasoning", "encrypted_content": "enc1"}

    def test_keep_flag_still_drops_plain_reasoning(self):
        content = [
            {"type": "text", "text": "hello"},
            {"type": "reasoning", "text": "display-only thought"},
        ]
        result = _flatten_message_content(content, keep_encrypted_reasoning=True)
        assert result == "hello"

    def test_keep_flag_drops_empty_encrypted(self):
        content = [
            {"type": "text", "text": "hello"},
            {"type": "reasoning", "encrypted_content": ""},
        ]
        result = _flatten_message_content(content, keep_encrypted_reasoning=True)
        assert result == "hello"

    def test_keep_flag_preserves_nested_encrypted(self):
        content = [
            {"type": "text", "text": "hi"},
            {"type": "reasoning", "reasoning": {"encrypted_content": "nested"}},
        ]
        result = _flatten_message_content(content, keep_encrypted_reasoning=True)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[1]["reasoning"]["encrypted_content"] == "nested"

    def test_keep_flag_mixed_text_and_reasoning_order(self):
        """Text before and after reasoning is joined, reasoning stays in position."""
        content = [
            {"type": "text", "text": "first"},
            {"type": "reasoning", "encrypted_content": "enc"},
            {"type": "text", "text": "second"},
        ]
        result = _flatten_message_content(content, keep_encrypted_reasoning=True)
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0] == {"type": "text", "text": "first"}
        assert result[1] == {"type": "reasoning", "encrypted_content": "enc"}
        assert result[2] == {"type": "text", "text": "second"}

    def test_keep_flag_with_image(self):
        """Encrypted reasoning and images both survive, in original order."""
        content = [
            {"type": "text", "text": "look"},
            {"type": "reasoning", "encrypted_content": "enc"},
            {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
        ]
        result = _flatten_message_content(content, keep_encrypted_reasoning=True)
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[1]["type"] == "reasoning"
        assert result[2]["type"] == "image_url"

    def test_string_passthrough(self):
        assert _flatten_message_content("plain", keep_encrypted_reasoning=True) == "plain"

    def test_non_list_passthrough(self):
        assert _flatten_message_content(42, keep_encrypted_reasoning=True) == 42  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _sanitize_messages — propagates keep_encrypted_reasoning
# ---------------------------------------------------------------------------
class TestSanitizeMessagesKeepsEncryptedReasoning:
    def test_default_drops_reasoning_in_ai_message(self):
        messages = [
            HumanMessage(content="hi"),
            AIMessage(
                content=[
                    {"type": "text", "text": "answer"},
                    {"type": "reasoning", "encrypted_content": "enc"},
                ]
            ),
        ]
        result = _sanitize_messages(messages)
        ai = result[1]
        assert ai.content == "answer"

    def test_keep_flag_preserves_reasoning_in_ai_message(self):
        messages = [
            HumanMessage(content="hi"),
            AIMessage(
                content=[
                    {"type": "text", "text": "answer"},
                    {"type": "reasoning", "encrypted_content": "enc"},
                ]
            ),
        ]
        result = _sanitize_messages(messages, keep_encrypted_reasoning=True)
        ai = result[1]
        assert isinstance(ai.content, list)
        assert len(ai.content) == 2
        assert ai.content[1] == {"type": "reasoning", "encrypted_content": "enc"}

    def test_keep_flag_does_not_affect_system_message(self):
        messages = [
            SystemMessage(content="you are helpful"),
            AIMessage(
                content=[
                    {"type": "text", "text": "ok"},
                    {"type": "reasoning", "encrypted_content": "enc"},
                ]
            ),
        ]
        result = _sanitize_messages(messages, keep_encrypted_reasoning=True)
        assert result[0].content == "you are helpful"
        assert isinstance(result[1].content, list)
        assert len(result[1].content) == 2

    def test_keep_flag_plain_reasoning_still_dropped(self):
        messages = [
            AIMessage(
                content=[
                    {"type": "text", "text": "answer"},
                    {"type": "reasoning", "text": "display only"},
                ]
            ),
        ]
        result = _sanitize_messages(messages, keep_encrypted_reasoning=True)
        assert result[0].content == "answer"
