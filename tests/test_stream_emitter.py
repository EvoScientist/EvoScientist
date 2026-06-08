"""Tests for EvoScientist/stream/emitter.py."""

from EvoScientist.stream.emitter import StreamEvent, StreamEventEmitter


class TestStreamEventEmitter:
    def test_thinking(self):
        ev = StreamEventEmitter.thinking("deep thought", thinking_id=1)
        assert isinstance(ev, StreamEvent)
        assert ev.type == "thinking"
        assert ev.data["content"] == "deep thought"
        assert ev.data["id"] == 1

    def test_text(self):
        ev = StreamEventEmitter.text("hello")
        assert ev.type == "text"
        assert ev.data["content"] == "hello"

    def test_tool_call(self):
        ev = StreamEventEmitter.tool_call("execute", {"command": "ls"}, tool_id="tc1")
        assert ev.type == "tool_call"
        assert ev.data["name"] == "execute"
        assert ev.data["args"] == {"command": "ls"}
        assert ev.data["id"] == "tc1"

    def test_tool_result(self):
        ev = StreamEventEmitter.tool_result(
            "execute", "[OK] done", success=True, tool_call_id="tc1"
        )
        assert ev.type == "tool_result"
        assert ev.data["name"] == "execute"
        assert ev.data["content"] == "[OK] done"
        assert ev.data["success"] is True
        assert ev.data["id"] == "tc1"

    def test_tool_result_failure(self):
        ev = StreamEventEmitter.tool_result(
            "execute", "Error: fail", success=False, tool_call_id="tc1"
        )
        assert ev.data["success"] is False

    def test_subagent_start(self):
        ev = StreamEventEmitter.subagent_start(
            "research-agent",
            "Find papers",
            instance_id="task:abc",
            tool_call_id="call_task_1",
        )
        assert ev.type == "subagent_start"
        assert ev.data["name"] == "research-agent"
        assert ev.data["description"] == "Find papers"
        assert ev.data["instance_id"] == "task:abc"
        assert ev.data["tool_call_id"] == "call_task_1"

    def test_subagent_tool_call(self):
        ev = StreamEventEmitter.subagent_tool_call(
            "research-agent",
            "tavily_search",
            {"query": "q"},
            "tc2",
            "task:abc",
        )
        assert ev.type == "subagent_tool_call"
        assert ev.data["subagent"] == "research-agent"
        assert ev.data["name"] == "tavily_search"
        assert ev.data["instance_id"] == "task:abc"

    def test_subagent_tool_result(self):
        ev = StreamEventEmitter.subagent_tool_result(
            "research-agent",
            "tavily_search",
            "results",
            True,
            tool_call_id="tc2",
            instance_id="task:abc",
        )
        assert ev.type == "subagent_tool_result"
        assert ev.data["subagent"] == "research-agent"
        assert ev.data["id"] == "tc2"
        assert ev.data["instance_id"] == "task:abc"

    def test_subagent_end(self):
        ev = StreamEventEmitter.subagent_end("research-agent", "task:abc")
        assert ev.type == "subagent_end"
        assert ev.data["name"] == "research-agent"
        assert ev.data["instance_id"] == "task:abc"

    def test_done(self):
        ev = StreamEventEmitter.done("final answer")
        assert ev.type == "done"
        assert ev.data["response"] == "final answer"

    def test_done_empty(self):
        ev = StreamEventEmitter.done()
        assert ev.data["response"] == ""

    def test_error(self):
        ev = StreamEventEmitter.error("something broke")
        assert ev.type == "error"
        assert ev.data["message"] == "something broke"

    def test_interrupt(self):
        ev = StreamEventEmitter.interrupt(
            "main",
            [{"name": "execute", "args": {"command": "ls"}}],
            [{"action_name": "execute", "allowed_decisions": ["approve", "reject"]}],
        )
        assert ev.type == "interrupt"
        assert ev.data["interrupt_id"] == "main"
        assert len(ev.data["action_requests"]) == 1
        assert len(ev.data["review_configs"]) == 1

    def test_interrupt_no_review_configs(self):
        ev = StreamEventEmitter.interrupt("x", [{"name": "execute"}])
        assert ev.data["review_configs"] == []

    def test_all_events_have_type_in_data(self):
        """Every event's data dict should contain a 'type' key matching the event type."""
        events = [
            StreamEventEmitter.thinking("x"),
            StreamEventEmitter.text("x"),
            StreamEventEmitter.tool_call("t", {}, "tc1"),
            StreamEventEmitter.tool_result("t", "x", True, "tc1"),
            StreamEventEmitter.subagent_start("s", "d", "task:1", "call_task_1"),
            StreamEventEmitter.subagent_tool_call("s", "t", {}, "tc2", "task:1"),
            StreamEventEmitter.subagent_tool_result(
                "s", "t", "x", True, "tc2", "task:1"
            ),
            StreamEventEmitter.subagent_text("s", "c", "task:1"),
            StreamEventEmitter.subagent_end("s", "task:1"),
            StreamEventEmitter.interrupt("i", []),
            StreamEventEmitter.done(),
            StreamEventEmitter.error("e"),
        ]
        for ev in events:
            assert "type" in ev.data
            assert ev.data["type"] == ev.type
