"""Tests for sub-agent text fallback (fix/subagent-summarize).

Covers:
    1. StreamEventEmitter.subagent_text — event construction
    2. stream_agent_events — subagent_text emission for sub-agent text chunks
    3. InboundConsumer — subagent_text buffer & fallback priority chain
    4. Prompt — DELEGATION_STRATEGY contains summarize guidance
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from EvoScientist.channels.bus.events import InboundMessage as BusInbound
from EvoScientist.channels.bus.message_bus import MessageBus
from EvoScientist.channels.channel_manager import ChannelManager
from EvoScientist.channels.consumer import InboundConsumer, _join_subagent_text
from EvoScientist.stream.emitter import StreamEvent, StreamEventEmitter
from tests.conftest import run_async as _run
from tests.fakes import FakeGraphGateway
from tests.fakes import StubChannel as _StubChannel
from tests.stream_v3_fakes import (
    FakeSubagent,
    FakeV3Agent,
    collect_events,
    message_delta,
)

# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════
# 1. StreamEventEmitter.subagent_text
# ═══════════════════════════════════════════════════════════════════


class TestSubagentTextEmitter:
    def test_creates_correct_event_type(self):
        ev = StreamEventEmitter.subagent_text(
            "research-agent", "Found 3 papers.", "task:research"
        )
        assert isinstance(ev, StreamEvent)
        assert ev.type == "subagent_text"

    def test_data_contains_subagent_and_content(self):
        ev = StreamEventEmitter.subagent_text(
            "analyst", "Result summary", "task:analyst"
        )
        assert ev.data["subagent"] == "analyst"
        assert ev.data["content"] == "Result summary"

    def test_data_contains_type_key(self):
        """Event data dict should include 'type' matching event type (project convention)."""
        ev = StreamEventEmitter.subagent_text("a", "b", "task:a")
        assert ev.data["type"] == "subagent_text"

    def test_empty_content(self):
        ev = StreamEventEmitter.subagent_text("agent", "", "task:agent")
        assert ev.data["content"] == ""

    def test_instance_id_included(self):
        ev = StreamEventEmitter.subagent_text(
            "agent", "content", instance_id="tracker:abc"
        )
        assert ev.data["instance_id"] == "tracker:abc"

    def test_included_in_all_events_type_check(self):
        """subagent_text must pass the same invariant as other emitters."""
        ev = StreamEventEmitter.subagent_text("s", "c", "task:s")
        assert "type" in ev.data
        assert ev.data["type"] == ev.type


# ═══════════════════════════════════════════════════════════════════
# 2. stream_agent_events — subagent_text emission
# ═══════════════════════════════════════════════════════════════════


class TestStreamAgentEventsSubagentText:
    """Verify sub-agent text chunks yield subagent_text events."""

    def test_subagent_text_emitted_for_subagent_chunks(self):
        """When a sub-agent produces text, subagent_text events should appear."""
        namespace = ("sub", "research")
        agent = FakeV3Agent(
            [
                message_delta(
                    "Sub-agent finding: X is significant.", namespace=namespace
                )
            ],
            subagents=[FakeSubagent(namespace, "research-agent")],
        )
        events = collect_events(agent)
        sa_text = [e for e in events if e.get("type") == "subagent_text"]
        assert len(sa_text) == 1
        assert "Sub-agent finding" in sa_text[0]["content"]
        # instance_id must be present and non-empty
        assert sa_text[0].get("instance_id"), "instance_id must be a non-empty string"

    def test_subagent_text_not_emitted_for_main_agent(self):
        """Main agent text should produce 'text' events, not 'subagent_text'."""
        agent = FakeV3Agent([message_delta("Main agent reply.")])
        events = collect_events(agent)
        sa_text = [e for e in events if e.get("type") == "subagent_text"]
        text_events = [e for e in events if e.get("type") == "text"]
        assert len(sa_text) == 0
        assert len(text_events) == 1

    def test_multiple_subagent_text_chunks_all_emitted(self):
        """Multiple text chunks from a sub-agent all yield subagent_text events."""
        namespace = ("sub", "a")
        agent = FakeV3Agent(
            [
                message_delta("Part 1.", namespace=namespace),
                message_delta("Part 2.", namespace=namespace),
                message_delta("Part 3.", namespace=namespace),
            ],
            subagents=[FakeSubagent(namespace, "research-agent")],
        )
        events = collect_events(agent)
        sa_text = [e for e in events if e.get("type") == "subagent_text"]
        assert len(sa_text) == 3
        combined = "".join(e["content"] for e in sa_text)
        assert "Part 1." in combined
        assert "Part 2." in combined
        assert "Part 3." in combined
        # All chunks from the same namespace must share the same instance_id
        ids = {e["instance_id"] for e in sa_text}
        assert len(ids) == 1, f"Expected 1 unique instance_id, got {ids}"
        assert all(e.get("instance_id") for e in sa_text)

    def test_parallel_same_name_agents_get_distinct_instance_ids(self):
        """Two sub-agents with the same display name but different namespaces
        produce subagent_text events with different instance_id values.

        This is the core of the interleaving fix: the consumer can tell
        the two instances apart even though their 'subagent' field is
        identical.
        """
        # Two different v3 namespaces, same projected subagent display name.
        ns1 = ("ns", "task", "id1", "agent")
        ns2 = ("ns", "task", "id2", "agent")
        agent = FakeV3Agent(
            [
                message_delta("Instance-1 text.", namespace=ns1),
                message_delta("Instance-2 text.", namespace=ns2),
                message_delta(" More from 1.", namespace=ns1),
            ],
            subagents=[
                FakeSubagent(ns1, "research-agent"),
                FakeSubagent(ns2, "research-agent"),
            ],
        )
        events = collect_events(agent)
        sa_text = [e for e in events if e.get("type") == "subagent_text"]
        assert len(sa_text) == 3

        # All events have the same subagent display name
        assert all(e["subagent"] == "research-agent" for e in sa_text)

        # But instance_ids differ between the two namespaces
        id1_events = [
            e
            for e in sa_text
            if "Instance-1" in e["content"] or "More from 1" in e["content"]
        ]
        id2_events = [e for e in sa_text if "Instance-2" in e["content"]]
        assert len(id1_events) == 2
        assert len(id2_events) == 1

        # Same namespace → same instance_id
        assert id1_events[0]["instance_id"] == id1_events[1]["instance_id"]
        # Different namespace → different instance_id
        assert id1_events[0]["instance_id"] != id2_events[0]["instance_id"]


# ═══════════════════════════════════════════════════════════════════
# 3. InboundConsumer — subagent_text buffer & fallback priority
# ═══════════════════════════════════════════════════════════════════


def _make_consumer(stream_events: list[dict], **kw):
    """Create an InboundConsumer whose agent streams the given event dicts.

    ``stream_events`` is a flat list of event data dicts (as produced by
    ``StreamEventEmitter.xxx().data``).
    """
    bus = MessageBus()
    mgr = ChannelManager(bus)
    mgr.register(_StubChannel())

    agent = MagicMock()
    consumer = InboundConsumer(
        bus=bus,
        manager=mgr,
        agent=agent,
        thread_id="",
        graph_gateway=FakeGraphGateway(events=stream_events),
        max_concurrent=2,
        max_pending=10,
        inference_timeout=5.0,
        drain_timeout=1.0,
        **kw,
    )
    return consumer, bus


class TestConsumerSubagentTextFallback:
    """InboundConsumer should use sub-agent text as fallback when main agent is silent."""

    def test_subagent_text_used_when_no_final_content(self):
        """When the main agent produces no text, sub-agent text becomes the response."""
        events = [
            {
                "type": "subagent_text",
                "subagent": "research",
                "instance_id": "task:research",
                "content": "Found 3 relevant papers.",
            },
            {
                "type": "subagent_text",
                "subagent": "research",
                "instance_id": "task:research",
                "content": " Key insight: X is Y.",
            },
            {"type": "done", "content": ""},
        ]
        consumer, bus = _make_consumer(events)

        async def _test():
            msg = BusInbound(
                channel="stub",
                sender_id="u1",
                chat_id="c1",
                content="analyze papers",
            )
            await bus.publish_inbound(msg)

            task = asyncio.create_task(consumer.run())
            outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=5.0)

            assert outbound.content == "Found 3 relevant papers. Key insight: X is Y."
            assert outbound.channel == "stub"

            await consumer.stop()
            await task

        _run(_test())

    def test_final_content_takes_priority_over_subagent_text(self):
        """When the main agent produces text, sub-agent text is ignored."""
        events = [
            {
                "type": "subagent_text",
                "subagent": "research",
                "instance_id": "task:research",
                "content": "Sub-agent detail.",
            },
            {"type": "text", "content": "Here is my summary."},
            {"type": "done", "content": ""},
        ]
        consumer, bus = _make_consumer(events)

        async def _test():
            msg = BusInbound(
                channel="stub",
                sender_id="u1",
                chat_id="c1",
                content="test",
            )
            await bus.publish_inbound(msg)

            task = asyncio.create_task(consumer.run())
            outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=5.0)

            assert outbound.content == "Here is my summary."

            await consumer.stop()
            await task

        _run(_test())

    def test_duplicate_thinking_not_relayed_across_resume_rounds(self):
        """Repeated thinking from resumed rounds should only be sent once."""
        bus = MessageBus()
        mgr = ChannelManager(bus)
        mgr.register(_StubChannel())

        channel = mgr.get_channel("stub")
        assert channel is not None
        channel.send_thinking_message = AsyncMock()

        thinking = "Initial plan. " * 20
        stream_calls = 0

        async def _fake_stream(_request):
            nonlocal stream_calls
            stream_calls += 1
            if stream_calls == 1:
                yield {"type": "thinking", "content": thinking}
                yield {
                    "type": "ask_user",
                    "interrupt_id": "ask-1",
                    "tool_call_id": "tc-1",
                    "questions": [{"question": "Continue?"}],
                }
                return

            yield {"type": "thinking", "content": thinking}
            yield {"type": "text", "content": "final answer"}
            yield {"type": "done", "content": "final answer"}

        consumer = InboundConsumer(
            bus=bus,
            manager=mgr,
            agent=MagicMock(),
            thread_id="",
            graph_gateway=FakeGraphGateway(stream=_fake_stream),
            max_concurrent=2,
            max_pending=10,
            inference_timeout=5.0,
            drain_timeout=1.0,
            send_thinking=True,
        )
        consumer._resolve_ask_user = AsyncMock(  # type: ignore[method-assign]
            return_value={"answers": ["yes"], "status": "answered"}
        )

        async def _test():
            await bus.publish_inbound(
                BusInbound(
                    channel="stub",
                    sender_id="u1",
                    chat_id="c1",
                    content="analyze papers",
                )
            )

            task = asyncio.create_task(consumer.run())
            outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=5.0)

            assert outbound.content == "final answer"
            assert channel.send_thinking_message.await_count == 1
            call = channel.send_thinking_message.await_args_list[0]
            assert call.args[1] == thinking.rstrip()

            await consumer.stop()
            await task

        _run(_test())

    def test_new_thinking_relayed_after_resume(self):
        """Genuinely different thinking in round 2 should be sent."""
        bus = MessageBus()
        mgr = ChannelManager(bus)
        mgr.register(_StubChannel())

        channel = mgr.get_channel("stub")
        assert channel is not None
        channel.send_thinking_message = AsyncMock()

        thinking_r1 = "Initial plan. " * 20
        thinking_r2 = "Revised plan. " * 20
        stream_calls = 0

        async def _fake_stream(_request):
            nonlocal stream_calls
            stream_calls += 1
            if stream_calls == 1:
                yield {"type": "thinking", "content": thinking_r1}
                yield {
                    "type": "ask_user",
                    "interrupt_id": "ask-1",
                    "tool_call_id": "tc-1",
                    "questions": [{"question": "Continue?"}],
                }
                return

            yield {"type": "thinking", "content": thinking_r2}
            yield {"type": "text", "content": "final answer"}
            yield {"type": "done", "content": "final answer"}

        consumer = InboundConsumer(
            bus=bus,
            manager=mgr,
            agent=MagicMock(),
            thread_id="",
            graph_gateway=FakeGraphGateway(stream=_fake_stream),
            max_concurrent=2,
            max_pending=10,
            inference_timeout=5.0,
            drain_timeout=1.0,
            send_thinking=True,
        )
        consumer._resolve_ask_user = AsyncMock(  # type: ignore[method-assign]
            return_value={"answers": ["yes"], "status": "answered"}
        )

        async def _test():
            await bus.publish_inbound(
                BusInbound(
                    channel="stub",
                    sender_id="u1",
                    chat_id="c1",
                    content="analyze papers",
                )
            )

            task = asyncio.create_task(consumer.run())
            outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=5.0)

            assert outbound.content == "final answer"
            assert channel.send_thinking_message.await_count == 2
            call1 = channel.send_thinking_message.await_args_list[0]
            call2 = channel.send_thinking_message.await_args_list[1]
            assert call1.args[1] == thinking_r1.rstrip()
            assert call2.args[1] == thinking_r2.rstrip()

            await consumer.stop()
            await task

        _run(_test())

    def test_no_response_fallback_when_both_empty(self):
        """When both final_content and subagent_text are empty, 'No response' is used."""
        events = [
            {"type": "done", "content": ""},
        ]
        consumer, bus = _make_consumer(events)

        async def _test():
            msg = BusInbound(
                channel="stub",
                sender_id="u1",
                chat_id="c1",
                content="test",
            )
            await bus.publish_inbound(msg)

            task = asyncio.create_task(consumer.run())
            outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=5.0)

            assert outbound.content == "No response"

            await consumer.stop()
            await task

        _run(_test())

    def test_done_content_overrides_subagent_text(self):
        """Done event with content takes priority over sub-agent text buffer."""
        events = [
            {
                "type": "subagent_text",
                "subagent": "research",
                "instance_id": "task:research",
                "content": "Sub-agent work.",
            },
            {"type": "done", "content": "Final summary from done event."},
        ]
        consumer, bus = _make_consumer(events)

        async def _test():
            msg = BusInbound(
                channel="stub",
                sender_id="u1",
                chat_id="c1",
                content="test",
            )
            await bus.publish_inbound(msg)

            task = asyncio.create_task(consumer.run())
            outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=5.0)

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
        buffers = {"research": ("research", ["Found papers.", " Key insight."])}
        result = _join_subagent_text(buffers)
        assert result == "Found papers. Key insight."
        assert "[research]" not in result

    def test_multiple_agents_with_prefix(self):
        """Multiple sub-agents: each section gets [name]: prefix."""
        buffers = {
            "research": ("research", ["Paper A is relevant."]),
            "analysis": ("analysis", ["Metric X is high."]),
        }
        result = _join_subagent_text(buffers)
        assert "[research]: Paper A is relevant." in result
        assert "[analysis]: Metric X is high." in result
        assert "\n\n" in result

    def test_multiple_agents_chunk_concatenation(self):
        """Chunks within the same agent are joined without separator."""
        buffers = {
            "agent-a": ("agent-a", ["chunk1", "chunk2"]),
            "agent-b": ("agent-b", ["chunk3"]),
        }
        result = _join_subagent_text(buffers)
        assert "[agent-a]: chunk1chunk2" in result
        assert "[agent-b]: chunk3" in result

    def test_single_agent_empty_chunks(self):
        """Single agent with empty chunks returns empty string."""
        buffers = {"agent": ("agent", [""])}
        assert _join_subagent_text(buffers) == ""

    def test_multiple_agents_preserves_order(self):
        """Agent sections appear in insertion order."""
        buffers = {
            "beta-key": ("beta", ["B"]),
            "alpha-key": ("alpha", ["A"]),
            "gamma-key": ("gamma", ["G"]),
        }
        result = _join_subagent_text(buffers)
        beta_pos = result.index("[beta]")
        alpha_pos = result.index("[alpha]")
        gamma_pos = result.index("[gamma]")
        assert beta_pos < alpha_pos < gamma_pos

    def test_same_name_instances_numbered(self):
        """Multiple instances of the same agent type get numbered labels."""
        buffers = {
            "inst-1": ("research-agent", ["Instance 1 text."]),
            "inst-2": ("research-agent", ["Instance 2 text."]),
        }
        result = _join_subagent_text(buffers)
        assert "[research-agent #1]: Instance 1 text." in result
        assert "[research-agent #2]: Instance 2 text." in result
        assert "\n\n" in result


# ═══════════════════════════════════════════════════════════════════
# 3c. InboundConsumer — parallel sub-agent grouping
# ═══════════════════════════════════════════════════════════════════


class TestConsumerParallelSubagentFallback:
    """Consumer should group parallel sub-agent text by agent name."""

    def test_parallel_agents_grouped_with_attribution(self):
        """Multiple sub-agents produce grouped, attributed output."""
        events = [
            {
                "type": "subagent_text",
                "subagent": "research",
                "instance_id": "task:research",
                "content": "Found papers.",
            },
            {
                "type": "subagent_text",
                "subagent": "analysis",
                "instance_id": "task:analysis",
                "content": "Metric is high.",
            },
            {
                "type": "subagent_text",
                "subagent": "research",
                "instance_id": "task:research",
                "content": " Key insight.",
            },
            {"type": "done", "content": ""},
        ]
        consumer, bus = _make_consumer(events)

        async def _test():
            msg = BusInbound(
                channel="stub",
                sender_id="u1",
                chat_id="c1",
                content="test",
            )
            await bus.publish_inbound(msg)

            task = asyncio.create_task(consumer.run())
            outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=5.0)

            assert "[research]: Found papers. Key insight." in outbound.content
            assert "[analysis]: Metric is high." in outbound.content

            await consumer.stop()
            await task

        _run(_test())

    def test_single_agent_no_attribution_prefix(self):
        """Single sub-agent fallback has no [name]: prefix."""
        events = [
            {
                "type": "subagent_text",
                "subagent": "research",
                "instance_id": "task:research",
                "content": "Only agent.",
            },
            {"type": "done", "content": ""},
        ]
        consumer, bus = _make_consumer(events)

        async def _test():
            msg = BusInbound(
                channel="stub",
                sender_id="u1",
                chat_id="c1",
                content="test",
            )
            await bus.publish_inbound(msg)

            task = asyncio.create_task(consumer.run())
            outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=5.0)

            assert outbound.content == "Only agent."
            assert "[research]" not in outbound.content

            await consumer.stop()
            await task

        _run(_test())


class TestConsumerSameNameInterleaved:
    """Two instances of the same agent type with interleaved chunks."""

    def test_same_name_interleaved_chunks_separated_by_instance_id(self):
        """Two research-agent instances with different instance_ids are properly separated.

        With the instance_id fix, chunks are keyed by instance_id so
        each instance's text is buffered independently and labelled
        with numbered suffixes.
        """
        events = [
            {
                "type": "subagent_text",
                "subagent": "research-agent",
                "instance_id": "inst-1",
                "content": "Instance-1 sentence A.",
            },
            {
                "type": "subagent_text",
                "subagent": "research-agent",
                "instance_id": "inst-2",
                "content": "Instance-2 sentence X.",
            },
            {
                "type": "subagent_text",
                "subagent": "research-agent",
                "instance_id": "inst-1",
                "content": " Instance-1 sentence B.",
            },
            {
                "type": "subagent_text",
                "subagent": "research-agent",
                "instance_id": "inst-2",
                "content": " Instance-2 sentence Y.",
            },
            {"type": "done", "content": ""},
        ]
        consumer, bus = _make_consumer(events)

        async def _test():
            msg = BusInbound(
                channel="stub",
                sender_id="u1",
                chat_id="c1",
                content="test",
            )
            await bus.publish_inbound(msg)

            task = asyncio.create_task(consumer.run())
            outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=5.0)

            # Fixed: instances are now properly separated with numbered labels
            assert (
                "[research-agent #1]: Instance-1 sentence A. Instance-1 sentence B."
                in outbound.content
            )
            assert (
                "[research-agent #2]: Instance-2 sentence X. Instance-2 sentence Y."
                in outbound.content
            )

            await consumer.stop()
            await task

        _run(_test())


class TestDelegationPromptSummarize:
    def test_framework_task_tool_contains_summarize_guidance(self):
        """The upstream TASK_TOOL_DESCRIPTION already instructs the LLM to summarize."""
        from deepagents.middleware.subagents import TASK_TOOL_DESCRIPTION

        assert "not visible to the user" in TASK_TOOL_DESCRIPTION
        assert "summary of the result" in TASK_TOOL_DESCRIPTION

    def test_framework_task_system_prompt_contains_reconcile_step(self):
        """The upstream TASK_SYSTEM_PROMPT includes a reconcile/synthesize step."""
        from deepagents.middleware.subagents import TASK_SYSTEM_PROMPT

        assert "Reconcile" in TASK_SYSTEM_PROMPT
        assert "synthesize" in TASK_SYSTEM_PROMPT.lower()
