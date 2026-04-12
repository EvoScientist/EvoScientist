"""Tests for EvoScientist/stream/events.py helpers."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessageChunk

from EvoScientist.stream.events import (
    _extract_summary_message_text,
    _extract_tool_content,
    _find_summarization_event_payload,
    _process_chunk_content,
    stream_agent_events,
)


class TestExtractToolContent:
    """Verify _extract_tool_content handles image and text ToolMessages."""

    def test_image_via_additional_kwargs(self):
        """Image ToolMessages with read_file_media_type return summary."""
        msg = SimpleNamespace(
            content=[{"type": "image", "base64": "abc123..."}],
            additional_kwargs={
                "read_file_media_type": "image/png",
                "read_file_path": "/chart.png",
            },
            name="read_file",
        )
        content, is_image = _extract_tool_content(msg)
        assert is_image is True
        assert "chart.png" in content
        assert "image/png" in content
        # Must NOT contain base64 data
        assert "abc123" not in content

    def test_image_via_list_content_blocks(self):
        """Image content blocks without metadata are still detected."""
        msg = SimpleNamespace(
            content=[
                {"type": "text", "text": "Image: chart.png"},
                {"type": "image", "base64": "iVBORw0KGgo..."},
            ],
            additional_kwargs={},
            name="read_file",
        )
        content, is_image = _extract_tool_content(msg)
        assert is_image is True
        assert "iVBORw0KGgo" not in content

    def test_normal_text_passthrough(self):
        """Normal text content passes through unchanged."""
        msg = SimpleNamespace(
            content="File written successfully to /output.txt",
            additional_kwargs={},
            name="write_file",
        )
        content, is_image = _extract_tool_content(msg)
        assert is_image is False
        assert content == "File written successfully to /output.txt"

    def test_empty_content(self):
        """Empty content returns empty string."""
        msg = SimpleNamespace(
            content="",
            additional_kwargs={},
            name="read_file",
        )
        content, is_image = _extract_tool_content(msg)
        assert is_image is False
        assert content == ""

    def test_list_text_blocks(self):
        """List of text blocks are joined."""
        msg = SimpleNamespace(
            content=[
                {"type": "text", "text": "Line 1"},
                {"type": "text", "text": "Line 2"},
            ],
            additional_kwargs={},
            name="read_file",
        )
        content, is_image = _extract_tool_content(msg)
        assert is_image is False
        assert "Line 1" in content
        assert "Line 2" in content

    def test_no_additional_kwargs_attr(self):
        """Messages without additional_kwargs attribute are handled."""
        msg = SimpleNamespace(
            content="some result",
            name="execute",
        )
        content, is_image = _extract_tool_content(msg)
        assert is_image is False
        assert content == "some result"


# =============================================================================
# _process_chunk_content — string content passthrough
# =============================================================================


class TestProcessChunkContentStrings:
    """Verify _process_chunk_content handles string content correctly.

    After removing strip_thinking_tags (ccproxy >=0.2.7 no longer embeds
    <thinking> tags), string content is emitted verbatim.  These tests
    serve as a regression baseline: if a future ccproxy version re-introduces
    tags, the raw tags will be visible and these tests will document that.
    """

    def _emit(self, content: str) -> list:
        from EvoScientist.stream.emitter import StreamEventEmitter
        from EvoScientist.stream.tracker import ToolCallTracker

        emitter = StreamEventEmitter()
        tracker = ToolCallTracker()
        chunk = AIMessageChunk(content=content)
        return list(_process_chunk_content(chunk, emitter, tracker))

    def test_plain_text_passthrough(self):
        events = self._emit("Hello world")
        assert len(events) == 1
        assert events[0].type == "text"
        assert events[0].data["content"] == "Hello world"

    def test_thinking_tags_stripped(self):
        """Legacy <thinking> tags from older ccproxy are stripped."""
        raw = "<thinking>some reasoning</thinking>The answer is 42."
        events = self._emit(raw)
        assert len(events) == 1
        assert "<thinking>" not in events[0].data["content"]
        assert events[0].data["content"] == "The answer is 42."

    def test_thinking_tags_only_yields_nothing(self):
        """Content that is only a thinking block yields no events."""
        events = self._emit("<thinking>just reasoning</thinking>")
        assert events == []

    def test_thinking_tags_preserve_surrounding_whitespace(self):
        """Stripping tags does not swallow adjacent spaces."""
        raw = "before <thinking>x</thinking> after"
        events = self._emit(raw)
        assert len(events) == 1
        assert events[0].data["content"] == "before  after"

    def test_empty_string_no_events(self):
        events = self._emit("")
        assert events == []


# =============================================================================
# Multi-mode streaming chunk unpacking
# =============================================================================


def _make_ai_chunk(content: str = "hello", **kwargs):
    """Create a minimal AIMessageChunk for testing."""
    return AIMessageChunk(content=content, **kwargs)


def _collect_events(agent, message="hi", thread_id="t1"):
    """Collect all events from stream_agent_events synchronously."""

    async def _run():
        events = []
        async for ev in stream_agent_events(agent, message, thread_id):
            events.append(ev)
        return events

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_run())
    finally:
        loop.close()


async def _async_iter(items):
    """Create an async iterator from a list."""
    for item in items:
        yield item


class TestMultiModeChunkUnpacking:
    """Test 3-tuple (multi-mode) and 2-tuple (single-mode) chunk handling."""

    def test_3tuple_chunk_unpacking(self):
        """Multi-mode yields 3-tuples (namespace, mode, data); messages are processed."""
        chunk = _make_ai_chunk("hello world")
        mock_agent = AsyncMock()
        mock_agent.astream = MagicMock(
            return_value=_async_iter(
                [
                    ((), "messages", (chunk, {})),
                ]
            )
        )
        events = _collect_events(mock_agent)
        text_events = [e for e in events if e.get("type") == "text"]
        assert len(text_events) == 1
        assert text_events[0]["content"] == "hello world"

    def test_2tuple_fallback(self):
        """Single-mode yields 2-tuples; should still work."""
        chunk = _make_ai_chunk("fallback")
        mock_agent = AsyncMock()
        mock_agent.astream = MagicMock(
            return_value=_async_iter(
                [
                    ((), (chunk, {})),
                ]
            )
        )
        events = _collect_events(mock_agent)
        text_events = [e for e in events if e.get("type") == "text"]
        assert len(text_events) == 1
        assert text_events[0]["content"] == "fallback"

    def test_updates_mode_graceful_skip(self):
        """Updates mode chunks are skipped without error."""
        chunk = _make_ai_chunk("should appear")
        mock_agent = AsyncMock()
        mock_agent.astream = MagicMock(
            return_value=_async_iter(
                [
                    ((), "updates", {"some": "state"}),
                    ((), "messages", (chunk, {})),
                ]
            )
        )
        events = _collect_events(mock_agent)
        text_events = [e for e in events if e.get("type") == "text"]
        assert len(text_events) == 1
        assert text_events[0]["content"] == "should appear"

    def test_summarization_filtered(self):
        """Chunks with lc_source=summarization metadata are filtered out."""
        chunk_real = _make_ai_chunk("real content")
        chunk_synth = _make_ai_chunk("synthetic summary")
        mock_agent = AsyncMock()
        mock_agent.astream = MagicMock(
            return_value=_async_iter(
                [
                    ((), "messages", (chunk_synth, {"lc_source": "summarization"})),
                    ((), "messages", (chunk_real, {})),
                ]
            )
        )
        events = _collect_events(mock_agent)
        summary_start_events = [
            e for e in events if e.get("type") == "summarization_start"
        ]
        assert len(summary_start_events) == 1
        summary_events = [e for e in events if e.get("type") == "summarization"]
        assert len(summary_events) == 1
        assert summary_events[0]["content"] == "synthetic summary"
        text_events = [e for e in events if e.get("type") == "text"]
        assert len(text_events) == 1
        assert text_events[0]["content"] == "real content"

    def test_updates_mode_summarization_event_emitted(self):
        """_summarization_event updates should emit a summarization event."""
        summary_message = SimpleNamespace(
            content="Here is a summary of the conversation to date:\n\nKey facts",
        )
        chunk_real = _make_ai_chunk("real content")
        mock_agent = AsyncMock()
        mock_agent.astream = MagicMock(
            return_value=_async_iter(
                [
                    (
                        (),
                        "updates",
                        {
                            "agent": {
                                "_summarization_event": {
                                    "summary_message": summary_message,
                                    "cutoff_index": 12,
                                    "file_path": None,
                                }
                            }
                        },
                    ),
                    ((), "messages", (chunk_real, {})),
                ]
            )
        )
        events = _collect_events(mock_agent)
        summary_start_events = [
            e for e in events if e.get("type") == "summarization_start"
        ]
        assert len(summary_start_events) == 1
        summary_events = [e for e in events if e.get("type") == "summarization"]
        assert len(summary_events) == 1
        assert summary_events[0]["content"] == "Key facts"

    def test_updates_mode_does_not_duplicate_streamed_summarization(self):
        """If streamed summarization already emitted, updates fallback should not duplicate it."""
        chunk_synth = _make_ai_chunk("synthetic summary")
        summary_message = SimpleNamespace(
            content="Here is a summary of the conversation to date:\n\nKey facts"
        )
        chunk_real = _make_ai_chunk("real content")
        mock_agent = AsyncMock()
        mock_agent.astream = MagicMock(
            return_value=_async_iter(
                [
                    ((), "messages", (chunk_synth, {"lc_source": "summarization"})),
                    (
                        (),
                        "updates",
                        {
                            "_summarization_event": {
                                "summary_message": summary_message,
                                "cutoff_index": 12,
                                "file_path": None,
                            }
                        },
                    ),
                    ((), "messages", (chunk_real, {})),
                ]
            )
        )
        events = _collect_events(mock_agent)
        summary_start_events = [
            e for e in events if e.get("type") == "summarization_start"
        ]
        assert len(summary_start_events) == 1
        summary_events = [e for e in events if e.get("type") == "summarization"]
        assert len(summary_events) == 1
        assert summary_events[0]["content"] == "synthetic summary"

    def test_updates_mode_does_not_reemit_existing_summarization_event(self):
        """Persisted _summarization_event from a prior turn should not be replayed."""
        summary_message = SimpleNamespace(
            content="Here is a summary of the conversation to date:\n\nKey facts",
        )
        chunk_real = _make_ai_chunk("real content")
        mock_agent = AsyncMock()
        mock_agent.aget_state = AsyncMock(
            return_value=SimpleNamespace(
                values={
                    "_summarization_event": {
                        "summary_message": summary_message,
                        "cutoff_index": 12,
                        "file_path": None,
                    }
                }
            )
        )
        mock_agent.astream = MagicMock(
            return_value=_async_iter(
                [
                    (
                        (),
                        "updates",
                        {
                            "_summarization_event": {
                                "summary_message": summary_message,
                                "cutoff_index": 12,
                                "file_path": None,
                            }
                        },
                    ),
                    ((), "messages", (chunk_real, {})),
                ]
            )
        )
        events = _collect_events(mock_agent)
        summary_start_events = [
            e for e in events if e.get("type") == "summarization_start"
        ]
        assert summary_start_events == []
        summary_events = [e for e in events if e.get("type") == "summarization"]
        assert summary_events == []


class TestUsageStatsExtraction:
    """Test token usage extraction from AIMessageChunk."""

    def test_usage_metadata_emitted(self):
        """AIMessageChunk with usage_metadata emits usage_stats event."""
        chunk = _make_ai_chunk(
            "hi",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            },
        )
        mock_agent = AsyncMock()
        mock_agent.astream = MagicMock(
            return_value=_async_iter(
                [
                    ((), "messages", (chunk, {})),
                ]
            )
        )
        events = _collect_events(mock_agent)
        usage_events = [e for e in events if e.get("type") == "usage_stats"]
        assert len(usage_events) == 1
        assert usage_events[0]["input_tokens"] == 100
        assert usage_events[0]["output_tokens"] == 50

    def test_no_usage_metadata_no_event(self):
        """AIMessageChunk without usage_metadata does not emit usage_stats."""
        chunk = _make_ai_chunk("hi")
        mock_agent = AsyncMock()
        mock_agent.astream = MagicMock(
            return_value=_async_iter(
                [
                    ((), "messages", (chunk, {})),
                ]
            )
        )
        events = _collect_events(mock_agent)
        usage_events = [e for e in events if e.get("type") == "usage_stats"]
        assert len(usage_events) == 0


class TestSummarizationHelpers:
    """Summarization extraction helpers."""

    def test_extract_summary_message_text_from_summary_tag(self):
        message = SimpleNamespace(
            content="Before\n<summary>\nImportant facts\n</summary>\nAfter",
        )
        assert _extract_summary_message_text(message) == "Important facts"

    def test_extract_summary_message_text_accepts_output_text_blocks(self):
        message = SimpleNamespace(
            content=[{"type": "output_text", "text": "Summary body"}],
        )
        assert _extract_summary_message_text(message) == "Summary body"

    def test_find_summarization_event_payload_nested(self):
        payload = {
            "node": {
                "response": {
                    "_summarization_event": {
                        "summary_message": SimpleNamespace(content="Summary body"),
                    }
                }
            }
        }
        event = _find_summarization_event_payload(payload)
        assert event is not None
        assert event["summary_message"].content == "Summary body"

    def test_zero_tokens_not_emitted(self):
        """Zero input and output tokens should not emit usage_stats."""
        chunk = _make_ai_chunk(
            "hi",
            usage_metadata={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )
        mock_agent = AsyncMock()
        mock_agent.astream = MagicMock(
            return_value=_async_iter(
                [
                    ((), "messages", (chunk, {})),
                ]
            )
        )
        events = _collect_events(mock_agent)
        usage_events = [e for e in events if e.get("type") == "usage_stats"]
        assert len(usage_events) == 0
