"""Tests for ToolErrorHandlerMiddleware."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from EvoScientist.middleware.tool_error_handler import (
    ToolErrorHandlerMiddleware,
    _build_error_message,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(tool_name: str = "my_mcp_tool", call_id: str = "tc_001"):
    """Create a minimal ToolCallRequest-like object."""
    req = MagicMock()
    req.tool_call = {"id": call_id, "name": tool_name, "args": {"query": "test"}}
    return req


# ---------------------------------------------------------------------------
# _build_error_message
# ---------------------------------------------------------------------------


class TestBuildErrorMessage:
    def test_returns_tool_message(self):
        req = _make_request("broken_tool", "tc_123")
        try:
            raise RuntimeError("connection refused")
        except RuntimeError:
            msg = _build_error_message(req)

        assert isinstance(msg, ToolMessage)
        assert msg.tool_call_id == "tc_123"
        assert msg.name == "broken_tool"
        assert msg.status == "error"

    def test_content_includes_tool_name(self):
        req = _make_request("broken_tool")
        try:
            raise ValueError("bad value")
        except ValueError:
            msg = _build_error_message(req)

        assert "broken_tool" in msg.content

    def test_content_includes_traceback(self):
        req = _make_request()
        try:
            raise RuntimeError("something went wrong")
        except RuntimeError:
            msg = _build_error_message(req)

        assert "RuntimeError" in msg.content
        assert "something went wrong" in msg.content
        assert "Traceback" in msg.content

    def test_content_includes_retry_guidance(self):
        req = _make_request()
        try:
            raise Exception("fail")
        except Exception:
            msg = _build_error_message(req)

        assert "retry" in msg.content.lower()


# ---------------------------------------------------------------------------
# Sync: wrap_tool_call
# ---------------------------------------------------------------------------


class TestWrapToolCallSync:
    def setup_method(self):
        self.mw = ToolErrorHandlerMiddleware()

    def test_success_passes_through(self):
        expected = ToolMessage(content="ok", tool_call_id="tc_001", name="t")
        handler = MagicMock(return_value=expected)
        req = _make_request()

        result = self.mw.wrap_tool_call(req, handler)

        assert result is expected
        handler.assert_called_once_with(req)

    def test_command_passes_through(self):
        cmd = Command(update={"messages": []})
        handler = MagicMock(return_value=cmd)
        req = _make_request()

        result = self.mw.wrap_tool_call(req, handler)

        assert result is cmd

    def test_exception_returns_error_tool_message(self):
        handler = MagicMock(side_effect=RuntimeError("MCP server crashed"))
        req = _make_request("flaky_tool", "tc_999")

        result = self.mw.wrap_tool_call(req, handler)

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert result.tool_call_id == "tc_999"
        assert result.name == "flaky_tool"
        assert "MCP server crashed" in result.content

    def test_exception_does_not_propagate(self):
        handler = MagicMock(side_effect=ConnectionError("refused"))
        req = _make_request()

        # Should NOT raise
        result = self.mw.wrap_tool_call(req, handler)
        assert isinstance(result, ToolMessage)

    def test_keyboard_interrupt_propagates(self):
        """KeyboardInterrupt should NOT be caught (it's BaseException, not Exception)."""
        handler = MagicMock(side_effect=KeyboardInterrupt())
        req = _make_request()

        with pytest.raises(KeyboardInterrupt):
            self.mw.wrap_tool_call(req, handler)

    def test_various_exception_types(self):
        """Different exception types are all caught and reported."""
        for exc_cls in (ValueError, TypeError, OSError, TimeoutError, ConnectionError):
            handler = MagicMock(side_effect=exc_cls(f"{exc_cls.__name__} happened"))
            req = _make_request()

            result = self.mw.wrap_tool_call(req, handler)

            assert isinstance(result, ToolMessage)
            assert result.status == "error"
            assert exc_cls.__name__ in result.content


# ---------------------------------------------------------------------------
# Async: awrap_tool_call
# ---------------------------------------------------------------------------


class TestWrapToolCallAsync:
    def setup_method(self):
        self.mw = ToolErrorHandlerMiddleware()

    @staticmethod
    def _run(coro):
        from tests.conftest import run_async

        return run_async(coro)

    def test_success_passes_through(self):
        expected = ToolMessage(content="ok", tool_call_id="tc_001", name="t")

        async def handler(req):
            return expected

        req = _make_request()
        result = self._run(self.mw.awrap_tool_call(req, handler))

        assert result is expected

    def test_command_passes_through(self):
        cmd = Command(update={"messages": []})

        async def handler(req):
            return cmd

        req = _make_request()
        result = self._run(self.mw.awrap_tool_call(req, handler))

        assert result is cmd

    def test_exception_returns_error_tool_message(self):
        async def handler(req):
            raise RuntimeError("MCP server timed out")

        req = _make_request("slow_tool", "tc_async")
        result = self._run(self.mw.awrap_tool_call(req, handler))

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert result.tool_call_id == "tc_async"
        assert result.name == "slow_tool"
        assert "MCP server timed out" in result.content

    def test_exception_does_not_propagate(self):
        async def handler(req):
            raise ConnectionError("connection lost")

        req = _make_request()
        result = self._run(self.mw.awrap_tool_call(req, handler))
        assert isinstance(result, ToolMessage)

    def test_keyboard_interrupt_propagates(self):
        async def handler(req):
            raise KeyboardInterrupt()

        req = _make_request()
        with pytest.raises(KeyboardInterrupt):
            self._run(self.mw.awrap_tool_call(req, handler))


# ---------------------------------------------------------------------------
# Middleware metadata
# ---------------------------------------------------------------------------


class TestMiddlewareMeta:
    def test_name(self):
        assert ToolErrorHandlerMiddleware.name == "tool_error_handler"

    def test_instantiation(self):
        mw = ToolErrorHandlerMiddleware()
        assert mw.name == "tool_error_handler"
