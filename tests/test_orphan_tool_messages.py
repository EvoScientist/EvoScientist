"""Tests for OrphanedToolMessageMiddleware."""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from EvoScientist.middleware.orphan_tool_messages import (
    OrphanedToolMessageMiddleware,
    _strip_orphaned_tool_messages,
)


def _ai(tool_calls=None, content=""):
    """Create an AIMessage with optional tool_calls."""
    tcs = [
        {"id": tc_id, "name": name, "args": {}}
        for tc_id, name in (tool_calls or [])
    ]
    return AIMessage(content=content, tool_calls=tcs)


def _tool(call_id, name="tool"):
    return ToolMessage(content="ok", tool_call_id=call_id, name=name)


class TestStripOrphanedToolMessages:
    def test_no_messages(self):
        assert _strip_orphaned_tool_messages([]) == []

    def test_no_tool_messages(self):
        msgs = [HumanMessage(content="hi"), _ai(content="hello")]
        assert _strip_orphaned_tool_messages(msgs) == msgs

    def test_valid_pair_preserved(self):
        msgs = [
            HumanMessage(content="hi"),
            _ai(tool_calls=[("call_1", "read_file")]),
            _tool("call_1", "read_file"),
        ]
        result = _strip_orphaned_tool_messages(msgs)
        assert len(result) == 3
        assert result == msgs

    def test_orphaned_tool_message_stripped(self):
        msgs = [
            HumanMessage(content="summary of previous conversation"),
            _tool("call_orphan", "write_file"),
            _ai(tool_calls=[("call_2", "execute")]),
            _tool("call_2", "execute"),
        ]
        result = _strip_orphaned_tool_messages(msgs)
        assert len(result) == 3
        assert all(
            not (isinstance(m, ToolMessage) and m.tool_call_id == "call_orphan")
            for m in result
        )

    def test_multiple_orphans_stripped(self):
        msgs = [
            HumanMessage(content="summary"),
            _tool("orphan_1"),
            _tool("orphan_2"),
            _ai(tool_calls=[("call_3", "t")]),
            _tool("call_3"),
        ]
        result = _strip_orphaned_tool_messages(msgs)
        assert len(result) == 3

    def test_multiple_valid_pairs(self):
        msgs = [
            _ai(tool_calls=[("c1", "t1"), ("c2", "t2")]),
            _tool("c1", "t1"),
            _tool("c2", "t2"),
            HumanMessage(content="ok"),
            _ai(tool_calls=[("c3", "t3")]),
            _tool("c3", "t3"),
        ]
        result = _strip_orphaned_tool_messages(msgs)
        assert len(result) == 6

    def test_does_not_mutate_original(self):
        msgs = [
            HumanMessage(content="summary"),
            _tool("orphan"),
        ]
        original_len = len(msgs)
        _strip_orphaned_tool_messages(msgs)
        assert len(msgs) == original_len

    def test_invalid_tool_calls_counted(self):
        """invalid_tool_calls IDs should also be recognized as valid."""
        ai = AIMessage(
            content="",
            tool_calls=[],
            invalid_tool_calls=[
                {"id": "call_bad", "name": "t", "args": "{invalid", "error": "parse error"},
            ],
        )
        msgs = [ai, _tool("call_bad")]
        result = _strip_orphaned_tool_messages(msgs)
        assert len(result) == 2

    def test_tool_message_before_ai_message_still_matched(self):
        """tool_call_id matching is global, not positional."""
        msgs = [
            _tool("c1"),
            _ai(tool_calls=[("c1", "t")]),
        ]
        result = _strip_orphaned_tool_messages(msgs)
        assert len(result) == 2

    def test_post_summarization_scenario(self):
        """Simulate the real bug: summary replaces old AI+Tool, but a ToolMessage leaks through."""
        msgs = [
            HumanMessage(content="[Summary of previous 50 messages...]"),
            _tool("call_from_old_ai", "execute"),
            HumanMessage(content="please continue"),
            _ai(tool_calls=[("call_new", "read_file")]),
            _tool("call_new", "read_file"),
        ]
        result = _strip_orphaned_tool_messages(msgs)
        assert len(result) == 4
        assert isinstance(result[0], HumanMessage)
        assert isinstance(result[1], HumanMessage)
        assert isinstance(result[2], AIMessage)
        assert isinstance(result[3], ToolMessage)
        assert result[3].tool_call_id == "call_new"


class TestMiddlewareSync:
    def setup_method(self):
        self.mw = OrphanedToolMessageMiddleware()

    def _make_request(self, messages):
        req = MagicMock()
        req.messages = messages
        req.override = MagicMock(return_value=req)
        return req

    def test_clean_messages_pass_through(self):
        msgs = [
            _ai(tool_calls=[("c1", "t")]),
            _tool("c1"),
        ]
        req = self._make_request(msgs)
        sentinel = MagicMock()
        handler = MagicMock(return_value=sentinel)

        result = self.mw.wrap_model_call(req, handler)

        assert result is sentinel
        handler.assert_called_once_with(req)
        req.override.assert_not_called()

    def test_orphans_removed_before_handler(self):
        msgs = [
            HumanMessage(content="summary"),
            _tool("orphan"),
            _ai(tool_calls=[("c1", "t")]),
            _tool("c1"),
        ]
        req = self._make_request(msgs)
        sentinel = MagicMock()
        handler = MagicMock(return_value=sentinel)

        result = self.mw.wrap_model_call(req, handler)

        assert result is sentinel
        req.override.assert_called_once()
        cleaned = req.override.call_args[1]["messages"]
        assert len(cleaned) == 3
        assert not any(
            isinstance(m, ToolMessage) and m.tool_call_id == "orphan"
            for m in cleaned
        )


class TestMiddlewareAsync:
    def setup_method(self):
        self.mw = OrphanedToolMessageMiddleware()

    @staticmethod
    def _run(coro):
        from tests.conftest import run_async
        return run_async(coro)

    def _make_request(self, messages):
        req = MagicMock()
        req.messages = messages
        req.override = MagicMock(return_value=req)
        return req

    def test_clean_messages_pass_through(self):
        msgs = [
            _ai(tool_calls=[("c1", "t")]),
            _tool("c1"),
        ]
        req = self._make_request(msgs)
        sentinel = MagicMock()

        async def handler(r):
            return sentinel

        result = self._run(self.mw.awrap_model_call(req, handler))

        assert result is sentinel
        req.override.assert_not_called()

    def test_orphans_removed_before_handler(self):
        msgs = [
            HumanMessage(content="summary"),
            _tool("orphan"),
            _ai(tool_calls=[("c1", "t")]),
            _tool("c1"),
        ]
        req = self._make_request(msgs)
        sentinel = MagicMock()

        async def handler(r):
            return sentinel

        result = self._run(self.mw.awrap_model_call(req, handler))

        assert result is sentinel
        req.override.assert_called_once()
        cleaned = req.override.call_args[1]["messages"]
        assert len(cleaned) == 3


class TestMiddlewareMeta:
    def test_name(self):
        assert OrphanedToolMessageMiddleware.name == "orphaned_tool_message_sanitizer"

    def test_instantiation(self):
        mw = OrphanedToolMessageMiddleware()
        assert mw.name == "orphaned_tool_message_sanitizer"
