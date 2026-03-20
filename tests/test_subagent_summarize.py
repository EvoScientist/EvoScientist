"""Tests for sub-agent text fallback (fix/subagent-summarize).

Covers:
    1. StreamEventEmitter.subagent_text — event construction
    2. stream_agent_events — subagent_text emission for sub-agent text chunks
    3. InboundConsumer — subagent_text buffer & fallback priority chain
    4. Prompt — DELEGATION_STRATEGY contains summarize guidance
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessageChunk

from EvoScientist.channels.base import Channel
from EvoScientist.channels.bus.events import (
    InboundMessage as BusInbound,
    OutboundMessage as BusOutbound,
)
from EvoScientist.channels.bus.message_bus import MessageBus
from EvoScientist.channels.channel_manager import ChannelManager
from EvoScientist.channels.consumer import InboundConsumer, _join_subagent_text
from EvoScientist.stream.emitter import StreamEvent, StreamEventEmitter
from EvoScientist.stream.events import stream_agent_events

from tests.conftest import run_async as _run


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


@dataclass
class _FakeConfig:
    text_chunk_limit: int = 4096
    allowed_senders: list | None = None
    allowed_channels: list | None = None
    proxy: str | None = None
    require_mention: str = "group"
    dm_policy: str = "allowlist"


def _make_ai_chunk(content: str = "", **kwargs):
    return AIMessageChunk(content=content, **kwargs)


async def _async_iter(items):
    for item in items:
        yield item


def _collect_events(agent, message="hi", thread_id="t1"):
    async def _run_inner():
        events = []
        async for ev in stream_agent_events(agent, message, thread_id):
            events.append(ev)
        return events

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_run_inner())
    finally:
        loop.close()


# ═══════════════════════════════════════════════════════════════════
# 1. StreamEventEmitter.subagent_text
# ═══════════════════════════════════════════════════════════════════


class TestSubagentTextEmitter:
    def test_creates_correct_event_type(self):
        ev = StreamEventEmitter.subagent_text("research-agent", "Found 3 papers.")
        assert isinstance(ev, StreamEvent)
        assert ev.type == "subagent_text"

    def test_data_contains_subagent_and_content(self):
        ev = StreamEventEmitter.subagent_text("analyst", "Result summary")
        assert ev.data["subagent"] == "analyst"
        assert ev.data["content"] == "Result summary"

    def test_data_contains_type_key(self):
        """Event data dict should include 'type' matching event type (project convention)."""
        ev = StreamEventEmitter.subagent_text("a", "b")
        assert ev.data["type"] == "subagent_text"

    def test_empty_content(self):
        ev = StreamEventEmitter.subagent_text("agent", "")
        assert ev.data["content"] == ""

    def test_included_in_all_events_type_check(self):
        """subagent_text must pass the same invariant as other emitters."""
        ev = StreamEventEmitter.subagent_text("s", "c")
        assert "type" in ev.data
        assert ev.data["type"] == ev.type


# ═══════════════════════════════════════════════════════════════════
# 2. stream_agent_events — subagent_text emission
# ═══════════════════════════════════════════════════════════════════


class TestStreamAgentEventsSubagentText:
    """Verify sub-agent text chunks yield subagent_text events."""

    def test_subagent_text_emitted_for_subagent_chunks(self):
        """When a sub-agent produces text, subagent_text events should appear."""
        # Sub-agent chunks come through LangGraph as 3-tuples with a non-root
        # namespace path indicating a sub-agent.
        subagent_chunk = _make_ai_chunk("Sub-agent finding: X is significant.")
        # 3-tuple: (namespace, mode, (msg, metadata))
        # Non-empty namespace signals a sub-agent
        mock_agent = AsyncMock()
        mock_agent.astream = MagicMock(
            return_value=_async_iter([
                (("sub:research",), "messages", (subagent_chunk, {})),
            ])
        )
        events = _collect_events(mock_agent)
        sa_text = [e for e in events if e.get("type") == "subagent_text"]
        assert len(sa_text) >= 1
        assert "Sub-agent finding" in sa_text[0]["content"]

    def test_subagent_text_not_emitted_for_main_agent(self):
        """Main agent text should produce 'text' events, not 'subagent_text'."""
        chunk = _make_ai_chunk("Main agent reply.")
        mock_agent = AsyncMock()
        mock_agent.astream = MagicMock(
            return_value=_async_iter([
                ((), "messages", (chunk, {})),
            ])
        )
        events = _collect_events(mock_agent)
        sa_text = [e for e in events if e.get("type") == "subagent_text"]
        text_events = [e for e in events if e.get("type") == "text"]
        assert len(sa_text) == 0
        assert len(text_events) == 1

    def test_multiple_subagent_text_chunks_all_emitted(self):
        """Multiple text chunks from a sub-agent all yield subagent_text events."""
        chunks = [
            (("sub:a",), "messages", (_make_ai_chunk("Part 1."), {})),
            (("sub:a",), "messages", (_make_ai_chunk("Part 2."), {})),
            (("sub:a",), "messages", (_make_ai_chunk("Part 3."), {})),
        ]
        mock_agent = AsyncMock()
        mock_agent.astream = MagicMock(return_value=_async_iter(chunks))
        events = _collect_events(mock_agent)
        sa_text = [e for e in events if e.get("type") == "subagent_text"]
        assert len(sa_text) == 3
        combined = "".join(e["content"] for e in sa_text)
        assert "Part 1." in combined
        assert "Part 2." in combined
        assert "Part 3." in combined


# ═══════════════════════════════════════════════════════════════════
# 3. InboundConsumer — subagent_text buffer & fallback priority
# ═══════════════════════════════════════════════════════════════════


class _StubChannel(Channel):
    """Minimal concrete channel for consumer tests."""

    name = "stub"

    def __init__(self, config=None):
        super().__init__(config or _FakeConfig())

    async def start(self):
        self._running = True

    async def _send_chunk(self, chat_id, formatted, raw, reply_to, metadata):
        pass

    async def _send_typing_action(self, chat_id):
        pass


def _make_consumer(stream_events: list[dict], **kw):
    """Create an InboundConsumer whose agent streams the given event dicts.

    ``stream_events`` is a flat list of event data dicts (as produced by
    ``StreamEventEmitter.xxx().data``).
    """
    bus = MessageBus()
    mgr = ChannelManager(bus)
    mgr.register(_StubChannel())

    # Patch stream_agent_events to yield pre-built events
    async def _fake_stream(agent, message, thread_id, **kwargs):
        for ev in stream_events:
            yield ev

    agent = MagicMock()
    consumer = InboundConsumer(
        bus=bus,
        manager=mgr,
        agent=agent,
        thread_id="",
        max_concurrent=2,
        max_pending=10,
        inference_timeout=5.0,
        drain_timeout=1.0,
        **kw,
    )
    return consumer, bus, _fake_stream


class TestConsumerSubagentTextFallback:
    """InboundConsumer should use sub-agent text as fallback when main agent is silent."""

    def test_subagent_text_used_when_no_final_content(self):
        """When the main agent produces no text, sub-agent text becomes the response."""
        events = [
            {"type": "subagent_text", "subagent": "research", "content": "Found 3 relevant papers."},
            {"type": "subagent_text", "subagent": "research", "content": " Key insight: X is Y."},
            {"type": "done", "content": ""},
        ]
        consumer, bus, fake_stream = _make_consumer(events)

        async def _test():
            with patch(
                "EvoScientist.stream.events.stream_agent_events",
                new=fake_stream,
            ):
                msg = BusInbound(
                    channel="stub",
                    sender_id="u1",
                    chat_id="c1",
                    content="analyze papers",
                )
                await bus.publish_inbound(msg)

                task = asyncio.create_task(consumer.run())
                outbound = await asyncio.wait_for(
                    bus.consume_outbound(), timeout=5.0
                )

                assert outbound.content == "Found 3 relevant papers. Key insight: X is Y."
                assert outbound.channel == "stub"

                await consumer.stop()
                await task

        _run(_test())

    def test_final_content_takes_priority_over_subagent_text(self):
        """When the main agent produces text, sub-agent text is ignored."""
        events = [
            {"type": "subagent_text", "subagent": "research", "content": "Sub-agent detail."},
            {"type": "text", "content": "Here is my summary."},
            {"type": "done", "content": ""},
        ]
        consumer, bus, fake_stream = _make_consumer(events)

        async def _test():
            with patch(
                "EvoScientist.stream.events.stream_agent_events",
                new=fake_stream,
            ):
                msg = BusInbound(
                    channel="stub",
                    sender_id="u1",
                    chat_id="c1",
                    content="test",
                )
                await bus.publish_inbound(msg)

                task = asyncio.create_task(consumer.run())
                outbound = await asyncio.wait_for(
                    bus.consume_outbound(), timeout=5.0
                )

                assert outbound.content == "Here is my summary."

                await consumer.stop()
                await task

        _run(_test())

    def test_no_response_fallback_when_both_empty(self):
        """When both final_content and subagent_text are empty, 'No response' is used."""
        events = [
            {"type": "done", "content": ""},
        ]
        consumer, bus, fake_stream = _make_consumer(events)

        async def _test():
            with patch(
                "EvoScientist.stream.events.stream_agent_events",
                new=fake_stream,
            ):
                msg = BusInbound(
                    channel="stub",
                    sender_id="u1",
                    chat_id="c1",
                    content="test",
                )
                await bus.publish_inbound(msg)

                task = asyncio.create_task(consumer.run())
                outbound = await asyncio.wait_for(
                    bus.consume_outbound(), timeout=5.0
                )

                assert outbound.content == "No response"

                await consumer.stop()
                await task

        _run(_test())

    def test_done_content_overrides_subagent_text(self):
        """Done event with content takes priority over sub-agent text buffer."""
        events = [
            {"type": "subagent_text", "subagent": "research", "content": "Sub-agent work."},
            {"type": "done", "content": "Final summary from done event."},
        ]
        consumer, bus, fake_stream = _make_consumer(events)

        async def _test():
            with patch(
                "EvoScientist.stream.events.stream_agent_events",
                new=fake_stream,
            ):
                msg = BusInbound(
                    channel="stub",
                    sender_id="u1",
                    chat_id="c1",
                    content="test",
                )
                await bus.publish_inbound(msg)

                task = asyncio.create_task(consumer.run())
                outbound = await asyncio.wait_for(
                    bus.consume_outbound(), timeout=5.0
                )

                assert outbound.content == "Final summary from done event."

                await consumer.stop()
                await task

        _run(_test())


# ═══════════════════════════════════════════════════════════════════
# 3b. _join_subagent_text helper
# ═══════════════════════════════════════════════════════════════════


class TestJoinSubagentText:
    """Unit tests for the _join_subagent_text helper."""

    def test_empty_dict_returns_empty_string(self):
        assert _join_subagent_text({}) == ""

    def test_single_agent_no_prefix(self):
        """One sub-agent: return raw text without [name]: prefix."""
        buffers = {"research": ["Found papers.", " Key insight."]}
        result = _join_subagent_text(buffers)
        assert result == "Found papers. Key insight."
        assert "[research]" not in result

    def test_multiple_agents_with_prefix(self):
        """Multiple sub-agents: each section gets [name]: prefix."""
        buffers = {
            "research": ["Paper A is relevant."],
            "analysis": ["Metric X is high."],
        }
        result = _join_subagent_text(buffers)
        assert "[research]: Paper A is relevant." in result
        assert "[analysis]: Metric X is high." in result
        assert "\n\n" in result

    def test_multiple_agents_chunk_concatenation(self):
        """Chunks within the same agent are joined without separator."""
        buffers = {
            "agent-a": ["chunk1", "chunk2"],
            "agent-b": ["chunk3"],
        }
        result = _join_subagent_text(buffers)
        assert "[agent-a]: chunk1chunk2" in result
        assert "[agent-b]: chunk3" in result

    def test_single_agent_empty_chunks(self):
        """Single agent with empty chunks returns empty string."""
        buffers = {"agent": [""]}
        assert _join_subagent_text(buffers) == ""

    def test_multiple_agents_preserves_order(self):
        """Agent sections appear in insertion order."""
        buffers = {"beta": ["B"], "alpha": ["A"], "gamma": ["G"]}
        result = _join_subagent_text(buffers)
        beta_pos = result.index("[beta]")
        alpha_pos = result.index("[alpha]")
        gamma_pos = result.index("[gamma]")
        assert beta_pos < alpha_pos < gamma_pos


# ═══════════════════════════════════════════════════════════════════
# 3c. InboundConsumer — parallel sub-agent grouping
# ═══════════════════════════════════════════════════════════════════


class TestConsumerParallelSubagentFallback:
    """Consumer should group parallel sub-agent text by agent name."""

    def test_parallel_agents_grouped_with_attribution(self):
        """Multiple sub-agents produce grouped, attributed output."""
        events = [
            {"type": "subagent_text", "subagent": "research", "content": "Found papers."},
            {"type": "subagent_text", "subagent": "analysis", "content": "Metric is high."},
            {"type": "subagent_text", "subagent": "research", "content": " Key insight."},
            {"type": "done", "content": ""},
        ]
        consumer, bus, fake_stream = _make_consumer(events)

        async def _test():
            with patch(
                "EvoScientist.stream.events.stream_agent_events",
                new=fake_stream,
            ):
                msg = BusInbound(
                    channel="stub",
                    sender_id="u1",
                    chat_id="c1",
                    content="test",
                )
                await bus.publish_inbound(msg)

                task = asyncio.create_task(consumer.run())
                outbound = await asyncio.wait_for(
                    bus.consume_outbound(), timeout=5.0
                )

                assert "[research]: Found papers. Key insight." in outbound.content
                assert "[analysis]: Metric is high." in outbound.content

                await consumer.stop()
                await task

        _run(_test())

    def test_single_agent_no_attribution_prefix(self):
        """Single sub-agent fallback has no [name]: prefix."""
        events = [
            {"type": "subagent_text", "subagent": "research", "content": "Only agent."},
            {"type": "done", "content": ""},
        ]
        consumer, bus, fake_stream = _make_consumer(events)

        async def _test():
            with patch(
                "EvoScientist.stream.events.stream_agent_events",
                new=fake_stream,
            ):
                msg = BusInbound(
                    channel="stub",
                    sender_id="u1",
                    chat_id="c1",
                    content="test",
                )
                await bus.publish_inbound(msg)

                task = asyncio.create_task(consumer.run())
                outbound = await asyncio.wait_for(
                    bus.consume_outbound(), timeout=5.0
                )

                assert outbound.content == "Only agent."
                assert "[research]" not in outbound.content

                await consumer.stop()
                await task

        _run(_test())

    def test_missing_subagent_field_uses_unknown(self):
        """Events without 'subagent' field are grouped under 'unknown'."""
        events = [
            {"type": "subagent_text", "content": "No agent name."},
            {"type": "done", "content": ""},
        ]
        consumer, bus, fake_stream = _make_consumer(events)

        async def _test():
            with patch(
                "EvoScientist.stream.events.stream_agent_events",
                new=fake_stream,
            ):
                msg = BusInbound(
                    channel="stub",
                    sender_id="u1",
                    chat_id="c1",
                    content="test",
                )
                await bus.publish_inbound(msg)

                task = asyncio.create_task(consumer.run())
                outbound = await asyncio.wait_for(
                    bus.consume_outbound(), timeout=5.0
                )

                # Single agent (unknown), no prefix
                assert outbound.content == "No agent name."

                await consumer.stop()
                await task

        _run(_test())


class TestDelegationPromptSummarize:
    def test_delegation_strategy_contains_summarize_guidance(self):
        from EvoScientist.prompts import DELEGATION_STRATEGY

        assert "Always Summarize Sub-Agent Results" in DELEGATION_STRATEGY

    def test_delegation_strategy_requires_user_facing_response(self):
        from EvoScientist.prompts import DELEGATION_STRATEGY

        assert "user-facing text" in DELEGATION_STRATEGY.lower() or \
               "user-facing" in DELEGATION_STRATEGY

    def test_delegation_strategy_forbids_silent_end(self):
        from EvoScientist.prompts import DELEGATION_STRATEGY

        assert "Never end your turn silently" in DELEGATION_STRATEGY
