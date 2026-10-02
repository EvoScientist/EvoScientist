"""Tests for StreamState, SubAgentState, and display helpers from cli.py."""

from EvoScientist.cli import (
    StreamState,
    SubAgentState,
    _build_todo_stats,
    _parse_todo_items,
)

# =============================================================================
# SubAgentState
# =============================================================================


class TestSubAgentState:
    def test_add_tool_call(self):
        sa = SubAgentState("research-agent")
        sa.add_tool_call("tavily_search", {"query": "test"}, "tc1")
        assert len(sa.tool_calls) == 1
        assert sa.tool_calls[0]["name"] == "tavily_search"

    def test_add_tool_call_dedup_by_id(self):
        sa = SubAgentState("research-agent")
        sa.add_tool_call("tavily_search", {"query": "test"}, "tc1")
        sa.add_tool_call("tavily_search", {"query": "updated"}, "tc1")
        assert len(sa.tool_calls) == 1
        assert sa.tool_calls[0]["args"]["query"] == "updated"

    def test_add_tool_call_merge_args_by_id(self):
        sa = SubAgentState("agent")
        sa.add_tool_call("search", {}, "tc1")
        sa.add_tool_call("search", {"query": "test"}, "tc1")
        assert sa.tool_calls[0]["args"] == {"query": "test"}

    def test_add_tool_result_matched(self):
        sa = SubAgentState("agent")
        sa.add_tool_call("execute", {}, "tc1")
        sa.add_tool_result("execute", "output", True, tool_call_id="tc1")
        result = sa.get_result_for(sa.tool_calls[0])
        assert result is not None
        assert result["content"] == "output"

    def test_add_tool_result_matched_by_id(self):
        """Concurrent same-name tools must be paired by tool_call_id, not name order."""
        sa = SubAgentState("agent")
        sa.add_tool_call("execute", {"cmd": "a"}, "tc1")
        sa.add_tool_call("execute", {"cmd": "b"}, "tc2")
        # Result for tc2 arrives first (out-of-order).
        sa.add_tool_result("execute", "out2", True, tool_call_id="tc2")
        assert sa.get_result_for(sa.tool_calls[1])["content"] == "out2"
        assert sa.get_result_for(sa.tool_calls[0]) is None
        # Then tc1's result arrives.
        sa.add_tool_result("execute", "out1", True, tool_call_id="tc1")
        assert sa.get_result_for(sa.tool_calls[0])["content"] == "out1"
        assert sa.get_result_for(sa.tool_calls[1])["content"] == "out2"

    def test_get_result_for_no_match(self):
        sa = SubAgentState("agent")
        tc = {"id": "tc_missing", "name": "x", "args": {}}
        assert sa.get_result_for(tc) is None


# =============================================================================
# StreamState
# =============================================================================


class TestStreamState:
    def test_handle_thinking(self):
        state = StreamState()
        result = state.handle_event({"type": "thinking", "content": "hmm"})
        assert result == "thinking"
        assert state.is_thinking is True
        assert state.thinking_text == "hmm"

    def test_handle_text(self):
        state = StreamState()
        result = state.handle_event({"type": "text", "content": "hello"})
        assert result == "text"
        assert state.is_responding is True
        assert state.response_text == "hello"

    def test_handle_text_accumulates(self):
        state = StreamState()
        state.handle_event({"type": "text", "content": "a"})
        state.handle_event({"type": "text", "content": "b"})
        assert state.response_text == "ab"

    def test_handle_tool_call(self):
        state = StreamState()
        state.handle_event(
            {
                "type": "tool_call",
                "id": "tc1",
                "name": "execute",
                "args": {"command": "ls"},
            }
        )
        assert len(state.tool_calls) == 1
        assert state.tool_calls[0]["name"] == "execute"

    def test_handle_tool_call_update_existing(self):
        state = StreamState()
        state.handle_event(
            {"type": "tool_call", "id": "tc1", "name": "execute", "args": {}}
        )
        state.handle_event(
            {
                "type": "tool_call",
                "id": "tc1",
                "name": "execute",
                "args": {"command": "ls"},
            }
        )
        assert len(state.tool_calls) == 1
        assert state.tool_calls[0]["args"] == {"command": "ls"}

    def test_handle_tool_result(self):
        state = StreamState()
        state.handle_event(
            {
                "type": "tool_result",
                "name": "execute",
                "content": "[OK] done",
                "id": "tc1",
            }
        )
        assert len(state.tool_results) == 1
        assert state.is_processing is True

    def test_handle_subagent_start(self):
        state = StreamState()
        state.handle_event(
            {
                "type": "subagent_start",
                "name": "research-agent",
                "description": "Search",
                "instance_id": "task:research",
                "tool_call_id": "tc_task_research",
            }
        )
        assert len(state.subagents) == 1
        assert state.subagents[0].name == "research-agent"
        assert state.subagents[0].is_active is True

    def test_handle_subagent_tool_call(self):
        state = StreamState()
        state.handle_event(
            {
                "type": "subagent_start",
                "name": "research-agent",
                "description": "Search",
                "instance_id": "task:research",
                "tool_call_id": "tc_task_research",
            }
        )
        state.handle_event(
            {
                "type": "subagent_tool_call",
                "subagent": "research-agent",
                "instance_id": "task:research",
                "name": "tavily_search",
                "args": {"query": "test"},
                "id": "tc_sa1",
            }
        )
        assert len(state.subagents) == 1
        assert len(state.subagents[0].tool_calls) == 1

    def test_handle_subagent_tool_result(self):
        state = StreamState()
        state.handle_event(
            {
                "type": "subagent_start",
                "name": "code-agent",
                "description": "Run code",
                "instance_id": "task:code",
                "tool_call_id": "tc_task_code",
            }
        )
        state.handle_event(
            {
                "type": "subagent_tool_call",
                "subagent": "code-agent",
                "instance_id": "task:code",
                "name": "execute",
                "args": {},
                "id": "tc1",
            }
        )
        state.handle_event(
            {
                "type": "subagent_tool_result",
                "subagent": "code-agent",
                "instance_id": "task:code",
                "name": "execute",
                "content": "output",
                "success": True,
                "id": "tc1",
            }
        )
        sa = state.subagents[0]
        assert len(sa.tool_results) == 1
        assert sa.get_result_for(sa.tool_calls[0])["content"] == "output"

    def test_handle_subagent_end(self):
        state = StreamState()
        state.handle_event(
            {
                "type": "subagent_start",
                "name": "agent-x",
                "description": "",
                "instance_id": "task:x",
                "tool_call_id": "tc_task_x",
            }
        )
        state.handle_event(
            {"type": "subagent_end", "name": "agent-x", "instance_id": "task:x"}
        )
        assert state.subagents[0].is_active is False

    def test_same_name_subagents_are_separated_by_instance_id(self):
        state = StreamState()
        state.handle_event(
            {
                "type": "subagent_start",
                "name": "research-agent",
                "description": "Find A",
                "instance_id": "task:one",
                "tool_call_id": "tc_task_1",
            }
        )
        state.handle_event(
            {
                "type": "subagent_start",
                "name": "research-agent",
                "description": "Find B",
                "instance_id": "task:two",
                "tool_call_id": "tc_task_2",
            }
        )
        state.handle_event(
            {
                "type": "subagent_tool_call",
                "subagent": "research-agent",
                "instance_id": "task:two",
                "name": "search",
                "args": {"query": "b"},
                "id": "tc2",
            }
        )
        state.handle_event(
            {
                "type": "subagent_end",
                "name": "research-agent",
                "instance_id": "task:one",
            }
        )

        assert len(state.subagents) == 2
        first, second = state.subagents
        assert first.name == "research-agent"
        assert second.name == "research-agent"
        assert first.is_active is False
        assert second.is_active is True
        assert second.tool_calls[0]["args"] == {"query": "b"}

    def test_handle_done(self):
        state = StreamState()
        state.handle_event({"type": "done", "response": "Final answer"})
        assert state.is_processing is False
        assert state.response_text == "Final answer"

    def test_handle_done_does_not_overwrite_existing_response(self):
        state = StreamState()
        state.handle_event({"type": "text", "content": "Already here"})
        state.handle_event({"type": "done", "response": "Final"})
        assert state.response_text == "Already here"

    def test_handle_error(self):
        state = StreamState()
        state.handle_event({"type": "error", "message": "boom"})
        assert "[Error] boom" in state.response_text
        assert state.is_processing is False

    def test_full_event_sequence(self, sample_events):
        state = StreamState()
        for event in sample_events:
            state.handle_event(event)
        assert state.thinking_text == "Let me think..."
        assert "Here is the answer." in state.response_text
        assert len(state.tool_calls) == 1
        assert len(state.tool_results) == 1
        assert len(state.subagents) == 1
        assert state.subagents[0].is_active is False


# =============================================================================
# _parse_todo_items
# =============================================================================


class TestParseTodoItems:
    def test_json_input(self):
        import json

        items = [{"status": "todo", "content": "Do X"}]
        result = _parse_todo_items(json.dumps(items))
        assert result is not None
        assert len(result) == 1

    def test_python_literal(self):
        result = _parse_todo_items('[{"status": "done", "content": "Y"}]')
        assert result is not None

    def test_embedded_list(self):
        text = "Updated todos:\n" + '[{"status": "todo", "content": "A"}]'
        result = _parse_todo_items(text)
        assert result is not None

    def test_invalid_input(self):
        assert _parse_todo_items("not a list at all") is None

    def test_empty_string(self):
        assert _parse_todo_items("") is None


# =============================================================================
# _build_todo_stats
# =============================================================================


class TestBuildTodoStats:
    def test_mixed_statuses(self):
        items = [
            {"status": "done"},
            {"status": "active"},
            {"status": "todo"},
            {"status": "completed"},
        ]
        result = _build_todo_stats(items)
        assert "1 active" in result
        assert "2 done" in result
        assert "1 pending" in result

    def test_all_done(self):
        items = [{"status": "done"}, {"status": "complete"}]
        result = _build_todo_stats(items)
        assert "2 done" in result
        assert "active" not in result

    def test_unknown_status_becomes_pending(self):
        items = [{"status": "unknown_status"}]
        result = _build_todo_stats(items)
        assert "1 pending" in result

    def test_empty_items(self):
        result = _build_todo_stats([])
        assert "0 items" in result


# =============================================================================
# Todo capture from write_todos args
# =============================================================================


class TestTodoCaptureFromArgs:
    def test_capture_from_tool_call_args(self):
        """write_todos tool_call args should update todo_items."""
        state = StreamState()
        todos = [
            {"status": "todo", "content": "Task A"},
            {"status": "active", "content": "Task B"},
        ]
        state.handle_event(
            {
                "type": "tool_call",
                "id": "tc1",
                "name": "write_todos",
                "args": {"todos": todos},
            }
        )
        assert state.todo_items == todos

    def test_capture_from_tool_result_fallback(self):
        """write_todos tool_result should also update todo_items."""
        import json

        state = StreamState()
        items = [{"status": "done", "content": "Finished"}]
        state.handle_event(
            {
                "type": "tool_result",
                "id": "tc1",
                "name": "write_todos",
                "content": json.dumps(items),
            }
        )
        assert len(state.todo_items) == 1
        assert state.todo_items[0]["status"] == "done"

    def test_args_capture_takes_priority(self):
        """Args capture (structured) should override result parse."""
        import json

        state = StreamState()
        args_todos = [{"status": "active", "content": "From args"}]
        result_todos = [{"status": "todo", "content": "From result"}]
        state.handle_event(
            {
                "type": "tool_call",
                "id": "tc1",
                "name": "write_todos",
                "args": {"todos": args_todos},
            }
        )
        assert state.todo_items[0]["content"] == "From args"
        # Result arrives but args already captured — result updates too
        state.handle_event(
            {
                "type": "tool_result",
                "id": "tc1",
                "name": "write_todos",
                "content": json.dumps(result_todos),
            }
        )
        # Result overwrites (both sources are valid, last write wins)
        assert state.todo_items[0]["content"] == "From result"

    def test_read_todos_result_updates(self):
        """read_todos tool_result should also update todo_items."""
        import json

        state = StreamState()
        items = [{"status": "todo", "content": "Read item"}]
        state.handle_event(
            {
                "type": "tool_result",
                "id": "tc1",
                "name": "read_todos",
                "content": json.dumps(items),
            }
        )
        assert len(state.todo_items) == 1

    def test_non_todo_tool_does_not_capture(self):
        state = StreamState()
        state.handle_event(
            {
                "type": "tool_call",
                "id": "tc1",
                "name": "execute",
                "args": {"command": "ls"},
            }
        )
        assert state.todo_items == []


# =============================================================================
# latest_text reset on tool_call
# =============================================================================


class TestLatestTextReset:
    def test_latest_text_accumulates(self):
        state = StreamState()
        state.handle_event({"type": "text", "content": "a"})
        state.handle_event({"type": "text", "content": "b"})
        assert state.latest_text == "ab"

    def test_latest_text_resets_on_tool_call(self):
        state = StreamState()
        state.handle_event({"type": "text", "content": "first segment"})
        state.handle_event(
            {"type": "tool_call", "id": "tc1", "name": "execute", "args": {}}
        )
        assert state.latest_text == ""
        state.handle_event({"type": "text", "content": "second segment"})
        assert state.latest_text == "second segment"
        # response_text still has everything
        assert state.response_text == "first segmentsecond segment"

    def test_tool_call_marks_existing_text_as_narration(self):
        state = StreamState()
        state.handle_event({"type": "text", "content": "first segment"})
        state.handle_event(
            {"type": "tool_call", "id": "tc1", "name": "execute", "args": {}}
        )

        assert state.narrated_response_end == len("first segment")
        assert state.narration_segments == [(0, "first segment")]

        state.handle_event({"type": "text", "content": "second segment"})
        assert state.narrated_response_end == len("first segment")
        assert state.narration_segments == [(0, "first segment")]

    def test_later_tool_call_extends_narrated_boundary(self):
        state = StreamState()
        state.handle_event({"type": "text", "content": "first segment"})
        state.handle_event(
            {"type": "tool_call", "id": "tc1", "name": "execute", "args": {}}
        )
        state.handle_event({"type": "text", "content": "second segment"})
        state.handle_event(
            {"type": "tool_call", "id": "tc2", "name": "execute", "args": {}}
        )

        assert state.narrated_response_end == len("first segmentsecond segment")
        assert state.narration_segments == [
            (0, "first segment"),
            (1, "second segment"),
        ]


class TestSubagentInstanceIds:
    def test_multiple_subagents_no_cross_merge(self):
        state = StreamState()
        state.handle_event(
            {
                "type": "subagent_start",
                "name": "code-agent",
                "description": "",
                "instance_id": "task:code",
                "tool_call_id": "tc_task_code",
            }
        )
        state.handle_event(
            {
                "type": "subagent_start",
                "name": "research-agent",
                "description": "",
                "instance_id": "task:research",
                "tool_call_id": "tc_task_research",
            }
        )
        state.handle_event(
            {
                "type": "subagent_tool_call",
                "subagent": "code-agent",
                "instance_id": "task:code",
                "name": "execute",
                "args": {},
                "id": "tc1",
            }
        )
        state.handle_event(
            {
                "type": "subagent_tool_call",
                "subagent": "research-agent",
                "instance_id": "task:research",
                "name": "tavily_search",
                "args": {},
                "id": "tc2",
            }
        )
        assert len(state.subagents) == 2
        code_sa = state._subagent_map["task:code"]
        research_sa = state._subagent_map["task:research"]
        assert len(code_sa.tool_calls) == 1
        assert code_sa.tool_calls[0]["name"] == "execute"
        assert len(research_sa.tool_calls) == 1
        assert research_sa.tool_calls[0]["name"] == "tavily_search"


# =============================================================================
# _parse_todo_items edge cases
# =============================================================================


class TestParseTodoItemsAdvanced:
    def test_prefixed_with_update_message(self):
        """'Updated todo list to [...]' format from write_todos result."""
        text = "Updated todo list to [{'status': 'todo', 'content': 'Do X'}]"
        result = _parse_todo_items(text)
        assert result is not None
        assert len(result) == 1
        assert result[0]["content"] == "Do X"

    def test_non_list_json(self):
        """JSON object (not list) should return None."""
        assert _parse_todo_items('{"status": "todo"}') is None

    def test_list_of_non_dicts(self):
        """List of strings should return None."""
        assert _parse_todo_items('["a", "b"]') is None


# =============================================================================
# Token usage tracking
# =============================================================================


class TestUsageStatsAccumulated:
    def test_single_usage_event(self):
        state = StreamState()
        state.handle_event(
            {"type": "usage_stats", "input_tokens": 100, "output_tokens": 50}
        )
        assert state.total_input_tokens == 100
        assert state.total_output_tokens == 50
        assert state.last_input_tokens == 100
        assert state.last_output_tokens == 50

    def test_multiple_usage_events_accumulate(self):
        state = StreamState()
        state.handle_event(
            {"type": "usage_stats", "input_tokens": 100, "output_tokens": 50}
        )
        state.handle_event(
            {"type": "usage_stats", "input_tokens": 200, "output_tokens": 80}
        )
        assert state.total_input_tokens == 300
        assert state.total_output_tokens == 130
        assert state.last_input_tokens == 200
        assert state.last_output_tokens == 80

    def test_zero_usage_event_does_not_clear_last_seen_values(self):
        state = StreamState()
        state.handle_event(
            {"type": "usage_stats", "input_tokens": 100, "output_tokens": 50}
        )
        state.handle_event(
            {"type": "usage_stats", "input_tokens": 0, "output_tokens": 0}
        )
        assert state.total_input_tokens == 100
        assert state.total_output_tokens == 50
        assert state.last_input_tokens == 100
        assert state.last_output_tokens == 50

    def test_usage_stats_in_display_args(self):
        state = StreamState()
        state.handle_event(
            {"type": "usage_stats", "input_tokens": 500, "output_tokens": 200}
        )
        args = state.get_display_args()
        assert args["total_input_tokens"] == 500
        assert args["total_output_tokens"] == 200

    def test_default_zero_tokens(self):
        state = StreamState()
        args = state.get_display_args()
        assert args["total_input_tokens"] == 0
        assert args["total_output_tokens"] == 0

    def test_usage_stats_returns_event_type(self):
        state = StreamState()
        result = state.handle_event(
            {"type": "usage_stats", "input_tokens": 10, "output_tokens": 5}
        )
        assert result == "usage_stats"


class TestComputePhase:
    """Tests for StreamState.compute_phase() research phase derivation."""

    def test_idle_by_default(self):
        state = StreamState()
        assert state.compute_phase() == "idle"

    def test_thinking_phase(self):
        state = StreamState()
        state.handle_event({"type": "thinking", "content": "hmm"})
        assert state.compute_phase() == "thinking"

    def test_researching_while_tool_pending(self):
        state = StreamState()
        state.handle_event(
            {"type": "tool_call", "id": "tc1", "name": "execute", "args": {}}
        )
        assert state.compute_phase() == "researching"

    def test_researching_after_tool_result(self):
        """is_processing is True right after tool_result, so still researching."""
        state = StreamState()
        state.handle_event(
            {"type": "tool_call", "id": "tc1", "name": "execute", "args": {}}
        )
        state.handle_event(
            {"type": "tool_result", "id": "tc1", "name": "execute", "content": "ok"}
        )
        assert state.compute_phase() == "researching"

    def test_writing_after_tools_done_and_text_starts(self):
        state = StreamState()
        state.handle_event(
            {"type": "tool_call", "id": "tc1", "name": "execute", "args": {}}
        )
        state.handle_event(
            {"type": "tool_result", "id": "tc1", "name": "execute", "content": "ok"}
        )
        state.handle_event({"type": "text", "content": "Final report"})
        assert state.compute_phase() == "writing"

    def test_writing_for_pure_text_response(self):
        state = StreamState()
        state.handle_event({"type": "text", "content": "Hello"})
        assert state.compute_phase() == "writing"

    def test_researching_with_active_subagent(self):
        state = StreamState()
        state.handle_event(
            {
                "type": "subagent_start",
                "name": "research",
                "description": "",
                "instance_id": "task:research",
                "tool_call_id": "tc_task_research",
            }
        )
        assert state.compute_phase() == "researching"

    def test_writing_after_subagent_ends(self):
        state = StreamState()
        state.handle_event(
            {
                "type": "subagent_start",
                "name": "research",
                "description": "",
                "instance_id": "task:research",
                "tool_call_id": "tc_task_research",
            }
        )
        state.handle_event(
            {"type": "subagent_end", "name": "research", "instance_id": "task:research"}
        )
        state.handle_event({"type": "text", "content": "Report"})
        assert state.compute_phase() == "writing"

    def test_researching_before_text_when_tools_finished(self):
        """Tools done but model hasn't emitted text yet — may call more tools."""
        state = StreamState()
        state.handle_event(
            {"type": "tool_call", "id": "tc1", "name": "execute", "args": {}}
        )
        state.handle_event(
            {"type": "tool_result", "id": "tc1", "name": "execute", "content": "ok"}
        )
        state.is_processing = False
        assert state.compute_phase() == "researching"

    def test_thinking_takes_priority_over_pending_tools(self):
        state = StreamState()
        state.handle_event(
            {"type": "tool_call", "id": "tc1", "name": "execute", "args": {}}
        )
        state.handle_event({"type": "thinking", "content": "re-thinking"})
        assert state.compute_phase() == "thinking"

    def test_researching_with_task_only_orchestrator(self):
        """Orchestrator delegates via 'task' tool — subagent drives the phase."""
        state = StreamState()
        state.handle_event(
            {"type": "tool_call", "id": "tc1", "name": "task", "args": {}}
        )
        state.handle_event(
            {
                "type": "subagent_start",
                "name": "code-agent",
                "description": "",
                "instance_id": "task:code",
                "tool_call_id": "tc1",
            }
        )
        assert state.compute_phase() == "researching"


class TestHasPendingWork:
    """Tests for StreamState.has_pending_work()."""

    def test_no_work(self):
        state = StreamState()
        assert state.has_pending_work() is False

    def test_pending_tool(self):
        state = StreamState()
        state.handle_event(
            {"type": "tool_call", "id": "tc1", "name": "execute", "args": {}}
        )
        assert state.has_pending_work() is True

    def test_active_subagent(self):
        state = StreamState()
        state.handle_event(
            {
                "type": "subagent_start",
                "name": "agent",
                "description": "",
                "instance_id": "task:agent",
                "tool_call_id": "tc_task_agent",
            }
        )
        assert state.has_pending_work() is True

    def test_is_processing(self):
        state = StreamState()
        state.handle_event(
            {"type": "tool_call", "id": "tc1", "name": "execute", "args": {}}
        )
        state.handle_event(
            {"type": "tool_result", "id": "tc1", "name": "execute", "content": "ok"}
        )
        assert state.is_processing is True
        assert state.has_pending_work() is True

    def test_all_done(self):
        state = StreamState()
        state.handle_event(
            {"type": "tool_call", "id": "tc1", "name": "execute", "args": {}}
        )
        state.handle_event(
            {"type": "tool_result", "id": "tc1", "name": "execute", "content": "ok"}
        )
        state.handle_event({"type": "text", "content": "done"})
        assert state.has_pending_work() is False


class TestVisibleToolCounts:
    """Tests for StreamState.visible_tool_counts()."""

    def test_empty(self):
        state = StreamState()
        assert state.visible_tool_counts() == (0, 0)

    def test_one_pending(self):
        state = StreamState()
        state.handle_event(
            {"type": "tool_call", "id": "tc1", "name": "execute", "args": {}}
        )
        assert state.visible_tool_counts() == (0, 1)

    def test_one_completed(self):
        state = StreamState()
        state.handle_event(
            {"type": "tool_call", "id": "tc1", "name": "execute", "args": {}}
        )
        state.handle_event(
            {"type": "tool_result", "id": "tc1", "name": "execute", "content": "ok"}
        )
        assert state.visible_tool_counts() == (1, 1)

    def test_mixed(self):
        state = StreamState()
        state.handle_event(
            {"type": "tool_call", "id": "tc1", "name": "execute", "args": {}}
        )
        state.handle_event(
            {"type": "tool_call", "id": "tc3", "name": "search", "args": {}}
        )
        state.handle_event(
            {"type": "tool_result", "id": "tc1", "name": "execute", "content": "ok"}
        )
        # execute done, search pending
        assert state.visible_tool_counts() == (1, 2)


class TestUsageWidgetElapsed:
    """Tests for UsageWidget elapsed time display."""

    def test_without_elapsed(self):
        from EvoScientist.cli.widgets.usage_widget import UsageWidget

        w = UsageWidget(1000, 500)
        plain = w._Static__content.plain
        assert "1,000" in plain
        assert "500" in plain
        assert "Elapsed" not in plain

    def test_with_elapsed(self):
        from EvoScientist.cli.widgets.usage_widget import UsageWidget

        w = UsageWidget(1000, 500, elapsed="12s")
        plain = w._Static__content.plain
        assert "Elapsed: 12s" in plain


class TestIsFinalResponseDelegation:
    """Tests that _is_final_response delegates to StreamState.has_pending_work."""

    def test_final_when_no_work(self):
        from EvoScientist.cli.tui_interactive import _is_final_response

        state = StreamState()
        assert _is_final_response(state) is True

    def test_not_final_during_tool(self):
        from EvoScientist.cli.tui_interactive import _is_final_response

        state = StreamState()
        state.handle_event(
            {"type": "tool_call", "id": "tc1", "name": "execute", "args": {}}
        )
        assert _is_final_response(state) is False

    def test_consistent_with_has_pending_work(self):
        from EvoScientist.cli.tui_interactive import _is_final_response

        state = StreamState()
        state.handle_event(
            {"type": "tool_call", "id": "tc1", "name": "execute", "args": {}}
        )
        state.handle_event(
            {"type": "tool_result", "id": "tc1", "name": "execute", "content": "ok"}
        )
        state.handle_event({"type": "text", "content": "done"})
        assert _is_final_response(state) == (not state.has_pending_work())


# =============================================================================
# ChannelState queue mechanism (removed — replaced by bus mode in channel.py)
# =============================================================================
