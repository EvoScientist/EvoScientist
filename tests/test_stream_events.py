"""Tests for EvoScientist/stream/events.py helpers."""

import asyncio

import pytest
from deepagents import create_deep_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, Interrupt

from EvoScientist.middleware.ask_user import AskUserMiddleware
from EvoScientist.stream.events import stream_agent_events
from EvoScientist.stream.summarization import (
    _extract_summary_message_text,
    _find_summarization_event_payload,
)
from EvoScientist.stream.tool_results import (
    _extract_command_tool_content,
    _extract_tool_content,
)
from tests.conftest import run_async
from tests.stream_v3_fakes import (
    ErroringV3Agent,
    FakeSubagent,
    FakeV3Agent,
    HangingV3Agent,
    SubscriptionSensitiveV3Agent,
    collect_events,
    message_delta,
    message_finish,
    message_tool_call_block,
    protocol_event,
    tool_finished,
    tool_started,
)


class _ToolCallingFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


class TestExtractToolContent:
    """Verify _extract_tool_content handles image and text ToolMessages."""

    def test_image_via_additional_kwargs(self):
        """Image ToolMessages with read_file_media_type return summary."""
        msg = ToolMessage(
            content=[{"type": "image", "base64": "abc123..."}],
            name="read_file",
            tool_call_id="tc-image",
            additional_kwargs={
                "read_file_media_type": "image/png",
                "read_file_path": "/chart.png",
            },
        )
        content, is_image = _extract_tool_content(msg)
        assert is_image is True
        assert "chart.png" in content
        assert "image/png" in content
        # Must NOT contain base64 data
        assert "abc123" not in content

    def test_image_via_list_content_blocks(self):
        """Image content blocks without metadata are still detected."""
        msg = ToolMessage(
            content=[
                {"type": "text", "text": "Image: chart.png"},
                {"type": "image", "base64": "iVBORw0KGgo..."},
            ],
            name="read_file",
            tool_call_id="tc-image",
        )
        content, is_image = _extract_tool_content(msg)
        assert is_image is True
        assert "iVBORw0KGgo" not in content

    def test_normal_text_passthrough(self):
        """Normal text content passes through unchanged."""
        msg = ToolMessage(
            content="File written successfully to /output.txt",
            name="write_file",
            tool_call_id="tc-write",
        )
        content, is_image = _extract_tool_content(msg)
        assert is_image is False
        assert content == "File written successfully to /output.txt"

    def test_empty_content(self):
        """Empty content returns empty string."""
        msg = ToolMessage(
            content="",
            name="read_file",
            tool_call_id="tc-empty",
        )
        content, is_image = _extract_tool_content(msg)
        assert is_image is False
        assert content == ""

    def test_list_text_blocks(self):
        """List of text blocks are joined."""
        msg = ToolMessage(
            content=[
                {"type": "text", "text": "Line 1"},
                {"type": "text", "text": "Line 2"},
            ],
            name="read_file",
            tool_call_id="tc-list",
        )
        content, is_image = _extract_tool_content(msg)
        assert is_image is False
        assert "Line 1" in content
        assert "Line 2" in content

    def test_command_tool_content_scans_multiple_messages(self):
        """Command updates may contain multiple messages; match by tool_call_id."""
        output = Command(
            update={
                "messages": [
                    ToolMessage(
                        content="Ignore me",
                        name="read_file",
                        tool_call_id="other",
                    ),
                    ToolMessage(
                        content=[{"type": "image", "base64": "iVBORw0KGgo..."}],
                        name="read_file",
                        tool_call_id="target",
                    ),
                ]
            }
        )

        assert _extract_command_tool_content(output, "target") == "[OK] Image displayed"


# =============================================================================
# v3 protocol streaming
# =============================================================================


class TestV3ProtocolStreaming:
    """Test stream_agent_events against v3 protocol events."""

    def test_message_delta_emits_text(self):
        """v3 content-block text deltas are processed."""
        agent = FakeV3Agent([message_delta("hello world")])
        events = collect_events(agent)
        text_events = [e for e in events if e.get("type") == "text"]
        assert len(text_events) == 1
        assert text_events[0]["content"] == "hello world"
        _, kwargs = agent.astream_events.call_args
        assert kwargs["version"] == "v3"
        assert "stream_mode" not in kwargs
        assert "subgraphs" not in kwargs

    def test_streamed_non_selector_json_is_replayed(self):
        """Normal JSON answers are not swallowed by selector JSON buffering."""
        agent = FakeV3Agent(
            [
                message_delta("{"),
                message_delta('"answer"'),
                message_delta(": 1}"),
            ]
        )
        events = collect_events(agent)
        text_events = [e for e in events if e.get("type") == "text"]
        assert "".join(e["content"] for e in text_events) == '{"answer": 1}'
        assert events[-1]["type"] == "done"
        assert events[-1]["response"] == '{"answer": 1}'

    def test_incomplete_non_selector_json_flushes_on_message_finish(self):
        """Buffered non-selector text is not lost if the message ends mid-object."""
        agent = FakeV3Agent(
            [
                message_delta("{"),
                message_delta('"answer":'),
                message_finish(),
            ]
        )
        events = collect_events(agent)
        text_events = [e for e in events if e.get("type") == "text"]
        assert "".join(e["content"] for e in text_events) == '{"answer":'
        assert events[-1]["response"] == '{"answer":'

    def test_json_answer_with_tools_key_is_replayed_without_selector_context(self):
        """Normal answers may legitimately contain a top-level tools key."""
        agent = FakeV3Agent(
            [
                message_delta('{"tools":["hammer"],"answer":"use safely"}'),
            ]
        )
        events = collect_events(agent)
        text_events = [e for e in events if e.get("type") == "text"]
        assert len(text_events) == 1
        assert text_events[0]["content"] == '{"tools":["hammer"],"answer":"use safely"}'
        assert events[-1]["response"] == '{"tools":["hammer"],"answer":"use safely"}'

    def test_text_delta_strips_legacy_thinking_tags(self):
        """Legacy <thinking> tags are still removed on the v3 text path."""
        agent = FakeV3Agent(
            [message_delta("<thinking>some reasoning</thinking>The answer is 42.")]
        )
        events = collect_events(agent)
        text_events = [e for e in events if e.get("type") == "text"]
        assert len(text_events) == 1
        assert text_events[0]["content"] == "The answer is 42."

    def test_text_delta_with_only_legacy_thinking_tags_is_skipped(self):
        agent = FakeV3Agent([message_delta("<thinking>just reasoning</thinking>")])
        events = collect_events(agent)
        assert [e for e in events if e.get("type") == "text"] == []

    def test_updates_event_without_summary_is_skipped(self):
        """Non-summary updates are skipped without error."""
        agent = FakeV3Agent(
            [
                protocol_event("updates", {"some": "state"}),
                message_delta("should appear"),
            ]
        )
        events = collect_events(agent)
        text_events = [e for e in events if e.get("type") == "text"]
        assert len(text_events) == 1
        assert text_events[0]["content"] == "should appear"

    def test_user_message_clears_memory_worker_saved_counts(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "EvoScientist.stream.events.clear_memory_worker_saved_counts",
            lambda: calls.append(True),
        )
        agent = FakeV3Agent([])

        collect_events(agent, message="new user turn")

        assert calls == [True]

    def test_command_message_clears_memory_worker_saved_counts(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "EvoScientist.stream.events.clear_memory_worker_saved_counts",
            lambda: calls.append(True),
        )
        agent = FakeV3Agent([])
        resume_command = Command(resume={"decisions": [{"type": "approve"}]})

        collect_events(agent, message=resume_command)

        assert calls == [True]
        assert agent.astream_events.call_args.args[0] is resume_command

    def test_summarization_filtered(self):
        """v3 messages with lc_source=summarization emit summarization events."""
        agent = FakeV3Agent(
            [
                message_delta("synthetic summary", {"lc_source": "summarization"}),
                message_delta("real content"),
            ]
        )
        events = collect_events(agent)
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
        summary_message = HumanMessage(
            content="Here is a summary of the conversation to date:\n\nKey facts",
        )
        agent = FakeV3Agent(
            [
                protocol_event(
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
                message_delta("real content"),
            ]
        )
        events = collect_events(agent)
        summary_start_events = [
            e for e in events if e.get("type") == "summarization_start"
        ]
        assert len(summary_start_events) == 1
        summary_events = [e for e in events if e.get("type") == "summarization"]
        assert len(summary_events) == 1
        assert summary_events[0]["content"] == "Key facts"

    def test_updates_mode_does_not_duplicate_streamed_summarization(self):
        """If streamed summarization already emitted, updates fallback should not duplicate it."""
        summary_message = HumanMessage(
            content="Here is a summary of the conversation to date:\n\nKey facts"
        )
        agent = FakeV3Agent(
            [
                message_delta("synthetic summary", {"lc_source": "summarization"}),
                protocol_event(
                    "updates",
                    {
                        "_summarization_event": {
                            "summary_message": summary_message,
                            "cutoff_index": 12,
                            "file_path": None,
                        }
                    },
                ),
                message_delta("real content"),
            ]
        )
        events = collect_events(agent)
        summary_start_events = [
            e for e in events if e.get("type") == "summarization_start"
        ]
        assert len(summary_start_events) == 1
        summary_events = [e for e in events if e.get("type") == "summarization"]
        assert len(summary_events) == 1
        assert summary_events[0]["content"] == "synthetic summary"

    def test_updates_mode_does_not_reemit_existing_summarization_event(self):
        """Persisted _summarization_event from a prior turn should not be replayed."""
        summary_message = HumanMessage(
            content="Here is a summary of the conversation to date:\n\nKey facts",
        )
        summary_event = {
            "_summarization_event": {
                "summary_message": summary_message,
                "cutoff_index": 12,
                "file_path": None,
            }
        }
        agent = FakeV3Agent(
            [
                protocol_event("updates", summary_event),
                message_delta("real content"),
            ],
            state_values=summary_event,
        )
        events = collect_events(agent)
        summary_start_events = [
            e for e in events if e.get("type") == "summarization_start"
        ]
        assert summary_start_events == []
        summary_events = [e for e in events if e.get("type") == "summarization"]
        assert summary_events == []

    def test_direct_stream_loads_existing_summarization_event_when_omitted(self):
        """Public stream_agent_events() suppresses persisted summary replays."""
        summary_message = HumanMessage(
            content="Here is a summary of the conversation to date:\n\nKey facts",
        )
        summary_event = {
            "_summarization_event": {
                "summary_message": summary_message,
                "cutoff_index": 12,
                "file_path": None,
            }
        }
        agent = FakeV3Agent(
            [
                protocol_event("updates", summary_event),
                message_delta("real content"),
            ],
            state_values=summary_event,
        )

        async def _collect():
            events = []
            async for event in stream_agent_events(agent, "hi", "t1"):
                events.append(event)
            return events

        events = run_async(_collect())

        summary_start_events = [
            e for e in events if e.get("type") == "summarization_start"
        ]
        assert summary_start_events == []
        summary_events = [e for e in events if e.get("type") == "summarization"]
        assert summary_events == []

    def test_whole_message_reasoning_is_not_duplicated(self):
        """Providers can expose the same reasoning in kwargs and content blocks."""
        message = AIMessage(
            additional_kwargs={"reasoning_content": "Think once."},
            content=[{"type": "reasoning", "reasoning": "Think once."}],
        )
        agent = FakeV3Agent([protocol_event("messages", (message, {}))])
        events = collect_events(agent)
        thinking_events = [e for e in events if e.get("type") == "thinking"]
        assert len(thinking_events) == 1
        assert thinking_events[0]["content"] == "Think once."

    def test_tool_events_emit_call_and_result(self):
        """v3 tool projection events become UI tool call/result events."""
        output = ToolMessage(
            name="read_file",
            content="File content",
            tool_call_id="tc1",
        )
        agent = FakeV3Agent(
            [
                tool_started("read_file", {"path": "notes.txt"}),
                tool_finished(output),
            ]
        )
        events = collect_events(agent)
        tool_call = next(e for e in events if e.get("type") == "tool_call")
        tool_result = next(e for e in events if e.get("type") == "tool_result")
        assert tool_call["name"] == "read_file"
        assert tool_call["args"] == {"path": "notes.txt"}
        assert tool_call["id"] == "tc1"
        assert tool_result["name"] == "read_file"
        assert tool_result["content"] == "File content"
        assert tool_result["success"] is True
        assert tool_result["id"] == "tc1"

    @pytest.mark.filterwarnings(
        "ignore:The v3 streaming protocol on Pregel is experimental"
    )
    def test_live_deepagents_v3_tool_result_preserves_tool_call_id(self):
        """DeepAgents v3 emits tool_call_id on started and finished tool events."""

        @tool
        def probe(value: str) -> str:
            """Return a deterministic probe result."""
            return f"probe:{value}"

        model = _ToolCallingFakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "probe",
                            "args": {"value": "ok"},
                            "id": "call_probe_1",
                        }
                    ],
                ),
                AIMessage(content="final answer"),
            ]
        )
        agent = create_deep_agent(
            model=model,
            tools=[probe],
            system_prompt="Use tools when requested.",
        )

        async def _collect_events():
            return [
                event
                async for event in stream_agent_events(
                    agent,
                    "run probe",
                    "live-deepagents-tool-id",
                )
            ]

        events = run_async(_collect_events())

        tool_call = next(e for e in events if e.get("type") == "tool_call")
        tool_result = next(e for e in events if e.get("type") == "tool_result")
        done = next(e for e in events if e.get("type") == "done")
        assert tool_call == {
            "type": "tool_call",
            "name": "probe",
            "args": {"value": "ok"},
            "id": "call_probe_1",
        }
        assert tool_result == {
            "type": "tool_result",
            "name": "probe",
            "content": "probe:ok",
            "success": True,
            "id": "call_probe_1",
        }
        assert done["content"] == "final answer"

    @pytest.mark.filterwarnings(
        "ignore:The v3 streaming protocol on Pregel is experimental"
    )
    def test_live_deepagents_v3_hitl_emits_tool_call_and_single_interrupt(self):
        """Live HITL streams the model tool call once before one interrupt."""

        @tool
        def echo_tool(value: str) -> str:
            """Echo a deterministic value."""
            return f"echo:{value}"

        model = _ToolCallingFakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "echo_tool",
                            "args": {"value": "ok"},
                            "id": "call_echo_1",
                        }
                    ],
                )
            ]
        )
        agent = create_deep_agent(
            model=model,
            tools=[echo_tool],
            system_prompt="Use tools when requested.",
            interrupt_on={"echo_tool": True},
            checkpointer=InMemorySaver(),
        )

        async def _collect_events():
            return [
                event
                async for event in stream_agent_events(
                    agent,
                    "run echo",
                    "live-deepagents-hitl",
                )
            ]

        events = run_async(_collect_events())

        tool_calls = [e for e in events if e.get("type") == "tool_call"]
        interrupts = [e for e in events if e.get("type") == "interrupt"]
        assert tool_calls == [
            {
                "type": "tool_call",
                "name": "echo_tool",
                "args": {"value": "ok"},
                "id": "call_echo_1",
            }
        ]
        assert len(interrupts) == 1
        assert events.index(tool_calls[0]) < events.index(interrupts[0])
        assert interrupts[0]["action_requests"][0]["name"] == "echo_tool"
        assert interrupts[0]["action_requests"][0]["args"] == {"value": "ok"}

    @pytest.mark.filterwarnings(
        "ignore:The v3 streaming protocol on Pregel is experimental"
    )
    def test_live_deepagents_v3_ask_user_suppresses_interrupt_tool_result(self):
        """ask_user pause markers are not displayed as failed tool results."""

        model = _ToolCallingFakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "ask_user",
                            "args": {
                                "questions": [
                                    {
                                        "question": "What dataset?",
                                        "type": "text",
                                    }
                                ]
                            },
                            "id": "call_ask_1",
                        }
                    ],
                ),
                AIMessage(content="final after ask"),
            ]
        )
        agent = create_deep_agent(
            model=model,
            tools=[],
            system_prompt="Use ask_user when requested.",
            middleware=[AskUserMiddleware()],
            checkpointer=InMemorySaver(),
        )

        async def _collect(message):
            return [
                event
                async for event in stream_agent_events(
                    agent,
                    message,
                    "live-deepagents-ask-user",
                )
            ]

        first_events = run_async(_collect("ask"))
        first_types = [event.get("type") for event in first_events]
        assert first_types == ["tool_call", "ask_user", "done"]
        ask_event = next(e for e in first_events if e.get("type") == "ask_user")
        assert ask_event["tool_call_id"] == "call_ask_1"
        assert ask_event["questions"] == [{"question": "What dataset?", "type": "text"}]

        resumed_events = run_async(
            _collect(Command(resume={"answers": ["CIFAR-10"], "status": "answered"}))
        )
        tool_result = next(e for e in resumed_events if e.get("type") == "tool_result")
        assert tool_result == {
            "type": "tool_result",
            "name": "ask_user",
            "content": "Q: What dataset?\nA: CIFAR-10",
            "success": True,
            "id": "call_ask_1",
        }
        done = next(e for e in resumed_events if e.get("type") == "done")
        assert done["content"] == "final after ask"

    @pytest.mark.filterwarnings(
        "ignore:The v3 streaming protocol on Pregel is experimental"
    )
    def test_live_deepagents_v3_task_result_uses_subagent_tool_message_content(self):
        """Live task results should display the subagent ToolMessage content."""

        root_model = _ToolCallingFakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "task",
                            "args": {
                                "subagent_type": "researcher",
                                "description": "find answer",
                            },
                            "id": "call_task_1",
                        }
                    ],
                ),
                AIMessage(content="root final"),
            ]
        )
        subagent_model = _ToolCallingFakeModel(
            responses=[AIMessage(content="subagent final")]
        )
        agent = create_deep_agent(
            model=root_model,
            tools=[],
            system_prompt="Delegate when requested.",
            subagents=[
                {
                    "name": "researcher",
                    "description": "Finds answers",
                    "system_prompt": "Answer directly.",
                    "model": subagent_model,
                    "tools": [],
                }
            ],
        )

        async def _collect_events():
            return [
                event
                async for event in stream_agent_events(
                    agent,
                    "delegate",
                    "live-deepagents-subagent",
                )
            ]

        events = run_async(_collect_events())

        subagent_start = next(e for e in events if e.get("type") == "subagent_start")
        subagent_end = next(e for e in events if e.get("type") == "subagent_end")
        task_result = next(
            e
            for e in events
            if e.get("type") == "tool_result" and e.get("name") == "task"
        )
        assert subagent_start["name"] == "researcher"
        assert subagent_start["description"] == ""
        assert subagent_start["instance_id"]
        assert subagent_start["tool_call_id"] == "call_task_1"
        assert subagent_end["instance_id"] == subagent_start["instance_id"]
        assert task_result["id"] == "call_task_1"
        assert task_result["content"] == "subagent final"
        assert "Command(" not in task_result["content"]

    def test_message_tool_call_block_emits_pre_execution_tool_call(self):
        """Model-declared tool calls remain visible before execution starts."""
        agent = FakeV3Agent(
            [
                message_tool_call_block(
                    "execute",
                    {"command": "ls"},
                    tool_call_id="tc-msg",
                ),
                protocol_event(
                    "updates",
                    {
                        "__interrupt__": [
                            Interrupt(
                                value={
                                    "action_requests": [
                                        {
                                            "name": "execute",
                                            "args": {"command": "ls"},
                                            "id": "tc-msg",
                                        }
                                    ],
                                    "review_configs": [],
                                },
                                id="main",
                            )
                        ]
                    },
                ),
            ]
        )
        events = collect_events(agent)
        event_types = [e["type"] for e in events]
        assert event_types.index("tool_call") < event_types.index("interrupt")
        tool_call = next(e for e in events if e.get("type") == "tool_call")
        assert tool_call["id"] == "tc-msg"
        assert tool_call["args"] == {"command": "ls"}

    def test_tool_selection_flushes_before_tool_only_step(self):
        """Selector UI event is emitted even when selection is followed only by a tool."""
        import EvoScientist.middleware.tool_selector as selector_mod

        original_selected = selector_mod._current_selected_tools
        original_total = selector_mod._total_tools_count
        original_last = selector_mod._last_emitted_tools
        selector_mod._current_selected_tools = ["read_file"]
        selector_mod._total_tools_count = 3
        selector_mod._last_emitted_tools = []
        try:
            output = ToolMessage(
                content="File content",
                name="read_file",
                tool_call_id="tc1",
            )
            agent = FakeV3Agent(
                [
                    message_delta('{"tools":["read_file"]}'),
                    tool_started("read_file", {"path": "notes.txt"}),
                    tool_finished(output),
                ]
            )
            events = collect_events(agent)
        finally:
            selector_mod._current_selected_tools = original_selected
            selector_mod._total_tools_count = original_total
            selector_mod._last_emitted_tools = original_last

        event_types = [e["type"] for e in events]
        assert event_types.index("tool_selection") < event_types.index("tool_call")
        selection = next(e for e in events if e.get("type") == "tool_selection")
        assert selection["tools"] == ["read_file"]

    def test_subagent_projection_routes_namespaced_events(self):
        """DeepAgents subagent projection supplies identity for namespaced events."""
        namespace = ("task", "abc")
        output = ToolMessage(
            content="Found result",
            name="search",
            tool_call_id="sa-tc",
        )
        agent = FakeV3Agent(
            [
                message_delta("Sub-agent finding.", namespace=namespace),
                tool_started(
                    "search",
                    {"query": "papers"},
                    tool_call_id="sa-tc",
                    namespace=namespace,
                ),
                tool_finished(output, tool_call_id="sa-tc", namespace=namespace),
            ],
            subagents=[FakeSubagent(namespace, "research-agent")],
        )
        events = collect_events(agent)
        assert any(e.get("type") == "subagent_start" for e in events)
        assert any(e.get("type") == "subagent_end" for e in events)

        text = next(e for e in events if e.get("type") == "subagent_text")
        tool_call = next(e for e in events if e.get("type") == "subagent_tool_call")
        tool_result = next(e for e in events if e.get("type") == "subagent_tool_result")

        assert text["subagent"] == "research-agent"
        assert text["content"] == "Sub-agent finding."
        assert text["instance_id"] == "task:abc"
        start = next(e for e in events if e.get("type") == "subagent_start")
        assert start["tool_call_id"] == "call_task_abc"
        assert tool_call["instance_id"] == "task:abc"
        assert tool_call["subagent"] == "research-agent"
        assert tool_call["name"] == "search"
        assert tool_call["args"] == {"query": "papers"}
        assert tool_result["instance_id"] == "task:abc"
        assert tool_result["subagent"] == "research-agent"
        assert tool_result["content"] == "Found result"
        event_types = [e["type"] for e in events]
        assert event_types.index("subagent_start") < event_types.index("subagent_text")
        assert event_types.index("subagent_tool_result") < event_types.index(
            "subagent_end"
        )
        assert event_types.index("subagent_end") < event_types.index("done")

    def test_namespaced_events_wait_for_delayed_subagent_registration(self):
        """Subagent events are not dropped if protocol events arrive first."""
        namespace = ("task", "late")

        class DelayedSubagentRun:
            def __init__(self):
                self.subagents = self._subagent_iter()

            async def _subagent_iter(self):
                await asyncio.sleep(0)
                yield FakeSubagent(namespace, "research-agent")

            def __aiter__(self):
                return self._events()

            async def _events(self):
                yield message_delta("Sub-agent finding.", namespace=namespace)

            async def abort(self):
                pass

        class Agent:
            def __init__(self):
                self._run = DelayedSubagentRun()

            def astream_events(self, *_args, **_kwargs):
                return self._run

            async def aget_state(self, _config):
                class Snapshot:
                    def __init__(self):
                        self.values = {}

                return Snapshot()

        events = collect_events(Agent())
        event_types = [e["type"] for e in events]
        text = next(e for e in events if e.get("type") == "subagent_text")

        assert text["content"] == "Sub-agent finding."
        assert text["instance_id"] == "task:late"
        assert event_types.index("subagent_start") < event_types.index("subagent_text")

    def test_subagent_tool_dedupe_uses_resolved_path(self):
        """Tool call/result events can arrive on namespace suffixes for one subagent."""
        subagent_path = ("task", "abc")
        call_namespace = (*subagent_path, "agent")
        tool_namespace = (*subagent_path, "tools")
        output = ToolMessage(
            content="Found result",
            name="search",
            tool_call_id="sa-tc",
        )
        agent = FakeV3Agent(
            [
                message_tool_call_block(
                    "search",
                    {"query": "papers"},
                    tool_call_id="sa-tc",
                    namespace=call_namespace,
                ),
                tool_started(
                    "search",
                    {"query": "papers"},
                    tool_call_id="sa-tc",
                    namespace=tool_namespace,
                ),
                tool_finished(output, tool_call_id="sa-tc", namespace=tool_namespace),
            ],
            subagents=[FakeSubagent(subagent_path, "research-agent")],
        )
        events = collect_events(agent)

        calls = [e for e in events if e.get("type") == "subagent_tool_call"]
        results = [e for e in events if e.get("type") == "subagent_tool_result"]

        assert len(calls) == 1
        assert calls[0]["instance_id"] == "task:abc"
        assert calls[0]["id"] == "sa-tc"
        assert len(results) == 1
        assert results[0]["instance_id"] == "task:abc"
        assert results[0]["id"] == "sa-tc"

    def test_subagent_end_is_emitted_before_later_root_text(self):
        """Finished subagents stop showing as active while root streaming continues."""
        output_returned = asyncio.Event()

        class CompletingSubagent:
            path = ("task", "done-first")
            name = "research-agent"

            @property
            def cause(self) -> dict[str, str]:
                return {"type": "toolCall", "tool_call_id": "call_done_first"}

            async def output(self):
                output_returned.set()
                return {"messages": []}

        class RootTextAfterSubagentDoneRun:
            def __init__(self):
                self.subagents = self._subagent_iter()

            async def _subagent_iter(self):
                yield CompletingSubagent()

            def __aiter__(self):
                return self._events()

            async def _events(self):
                await output_returned.wait()
                await asyncio.sleep(0)
                yield message_delta("root answer")

            async def abort(self):
                pass

        class Agent:
            def __init__(self):
                self._run = RootTextAfterSubagentDoneRun()

            def astream_events(self, *_args, **_kwargs):
                return self._run

            async def aget_state(self, _config):
                class Snapshot:
                    def __init__(self):
                        self.values = {}

                return Snapshot()

        events = collect_events(Agent())
        event_types = [e["type"] for e in events]

        assert event_types.index("subagent_end") < event_types.index("text")

    def test_subagent_projection_is_subscribed_before_protocol_pump(self):
        """Subagent handles are not dropped by lazy projection subscription."""
        namespace = ("task", "early")
        agent = SubscriptionSensitiveV3Agent(
            [message_delta("Sub-agent finding.", namespace=namespace)],
            [FakeSubagent(namespace, "research-agent")],
        )
        events = collect_events(agent)
        assert any(e.get("type") == "subagent_start" for e in events)
        assert any(e.get("type") == "subagent_end" for e in events)
        assert [e for e in events if e.get("type") == "text"] == []

        text = next(e for e in events if e.get("type") == "subagent_text")
        assert text["subagent"] == "research-agent"
        assert text["content"] == "Sub-agent finding."
        assert text["instance_id"] == "task:early"

    def test_parallel_same_name_subagent_events_carry_instance_ids(self):
        """Lifecycle and tool events distinguish same-name parallel subagents."""
        ns1 = ("task", "one")
        ns2 = ("task", "two")
        output1 = ToolMessage(
            content="Found one",
            name="search",
            tool_call_id="tc1",
        )
        output2 = ToolMessage(
            content="Found two",
            name="search",
            tool_call_id="tc2",
        )
        agent = FakeV3Agent(
            [
                tool_started(
                    "search", {"query": "a"}, tool_call_id="tc1", namespace=ns1
                ),
                tool_started(
                    "search", {"query": "b"}, tool_call_id="tc2", namespace=ns2
                ),
                tool_finished(output2, tool_call_id="tc2", namespace=ns2),
                tool_finished(output1, tool_call_id="tc1", namespace=ns1),
            ],
            subagents=[
                FakeSubagent(ns1, "research-agent"),
                FakeSubagent(ns2, "research-agent"),
            ],
        )
        events = collect_events(agent)

        starts = [e for e in events if e.get("type") == "subagent_start"]
        calls = [e for e in events if e.get("type") == "subagent_tool_call"]
        results = [e for e in events if e.get("type") == "subagent_tool_result"]
        ends = [e for e in events if e.get("type") == "subagent_end"]

        assert {e["instance_id"] for e in starts} == {"task:one", "task:two"}
        assert {e["instance_id"] for e in calls} == {"task:one", "task:two"}
        assert {e["instance_id"] for e in results} == {"task:one", "task:two"}
        assert {e["instance_id"] for e in ends} == {"task:one", "task:two"}

    def test_stream_construction_error_emits_error_before_reraising(self):
        """astream_events construction failures preserve the UI error event contract."""
        events = []

        async def collect():
            async for ev in stream_agent_events(
                ErroringV3Agent(RuntimeError("boom")),
                "hi",
                "t1",
            ):
                events.append(ev)

        with pytest.raises(RuntimeError, match="boom"):
            run_async(collect())
        assert events == [{"type": "error", "message": "boom"}]

    def test_generator_close_aborts_underlying_v3_stream(self):
        """Early consumer exit should abort the caller-driven v3 run."""

        async def consume_one_and_close():
            agent = HangingV3Agent([message_delta("hi")])
            stream = stream_agent_events(
                agent,
                "hi",
                "t1",
            )
            first = await stream.__anext__()
            await stream.aclose()
            return first, agent.aborted

        first, aborted = run_async(consume_one_and_close())
        assert first["type"] == "text"
        assert first["content"] == "hi"
        assert aborted is True


class TestUsageStatsExtraction:
    """Test token usage extraction from v3 message-finish events."""

    def test_usage_metadata_emitted(self):
        """v3 message-finish usage emits usage_stats event."""
        agent = FakeV3Agent(
            [
                message_delta("hi"),
                message_finish(
                    {
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "total_tokens": 150,
                    }
                ),
            ]
        )
        events = collect_events(agent)
        usage_events = [e for e in events if e.get("type") == "usage_stats"]
        assert len(usage_events) == 1
        assert usage_events[0]["input_tokens"] == 100
        assert usage_events[0]["output_tokens"] == 50

    def test_no_usage_metadata_no_event(self):
        """message-finish without usage does not emit usage_stats."""
        agent = FakeV3Agent([message_delta("hi"), message_finish()])
        events = collect_events(agent)
        usage_events = [e for e in events if e.get("type") == "usage_stats"]
        assert len(usage_events) == 0


class TestSummarizationHelpers:
    """Summarization extraction helpers."""

    def test_extract_summary_message_text_from_summary_tag(self):
        message = HumanMessage(
            content="Before\n<summary>\nImportant facts\n</summary>\nAfter",
        )
        assert _extract_summary_message_text(message) == "Important facts"

    def test_extract_summary_message_text_accepts_output_text_blocks(self):
        message = HumanMessage(
            content=[{"type": "output_text", "text": "Summary body"}],
        )
        assert _extract_summary_message_text(message) == "Summary body"

    def test_find_summarization_event_payload_nested(self):
        payload = {
            "node": {
                "response": {
                    "_summarization_event": {
                        "summary_message": HumanMessage(content="Summary body"),
                    }
                }
            }
        }
        event = _find_summarization_event_payload(payload)
        assert event is not None
        summary_message = event["summary_message"]
        assert isinstance(summary_message, HumanMessage)
        assert summary_message.content == "Summary body"

    def test_zero_tokens_not_emitted(self):
        """Zero input and output tokens should not emit usage_stats."""
        agent = FakeV3Agent(
            [
                message_delta("hi"),
                message_finish(
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                ),
            ]
        )
        events = collect_events(agent)
        usage_events = [e for e in events if e.get("type") == "usage_stats"]
        assert len(usage_events) == 0
