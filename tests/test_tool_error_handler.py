"""Tests for ToolErrorHandlerMiddleware."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from EvoScientist.middleware.tool_error_handler import (
    ToolErrorHandlerMiddleware,
    _build_error_message,
    _describe_exception,
)
from EvoScientist.stream.formatter import ContentType, ToolResultFormatter
from EvoScientist.stream.utils import is_success

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
        msg = _build_error_message(req, RuntimeError("connection refused"))

        assert isinstance(msg, ToolMessage)
        assert msg.tool_call_id == "tc_123"
        assert msg.name == "broken_tool"
        assert msg.status == "error"

    def test_content_includes_tool_name(self):
        msg = _build_error_message(_make_request("broken_tool"), ValueError("bad"))

        assert "broken_tool" in msg.content

    def test_content_names_the_exception_and_leaves_the_stack_to_the_log(self, caplog):
        try:
            raise RuntimeError("something went wrong")
        except RuntimeError as exc:
            msg = _build_error_message(_make_request(), exc)

        assert "RuntimeError: something went wrong" in msg.content
        assert "Traceback" not in msg.content
        assert "Traceback (most recent call last)" in caplog.text

    def test_content_lists_the_causes_it_was_raised_from(self):
        try:
            try:
                try:
                    raise KeyError("lookup")
                except KeyError:
                    raise ConnectionError("refused")  # noqa: B904
            except ConnectionError as inner:
                raise RuntimeError("fetch failed") from inner
        except RuntimeError as exc:
            msg = _build_error_message(_make_request(), exc)

        assert (
            "RuntimeError: fetch failed\nCaused by: ConnectionError: refused"
            in msg.content
        )
        # An exception merely being handled at the time is not a cause.
        assert "KeyError" not in msg.content

    def test_content_lists_the_members_of_a_group(self):
        # How anyio reports an MCP server that cannot be reached.
        group = ExceptionGroup(
            "unhandled errors in a TaskGroup",
            [
                ExceptionGroup(
                    "unhandled errors in a TaskGroup",
                    [ConnectionError("All connection attempts failed")],
                )
            ],
        )
        msg = _build_error_message(_make_request(), group)

        assert (
            "Sub-exception: ConnectionError: All connection attempts failed"
            in msg.content
        )

    def test_every_cause_is_listed_once_even_in_a_loop(self):
        first, second, third = (
            RuntimeError("first"),
            ValueError("second"),
            OSError("third"),
        )
        first.__cause__, second.__cause__, third.__cause__ = second, third, first

        assert _describe_exception(first) == (
            "RuntimeError: first\nCaused by: ValueError: second\n"
            "Caused by: OSError: third"
        )

    def test_any_depth_of_groups_is_described(self):
        exc: BaseException = ConnectionError("deepest")
        for _ in range(5_000):
            exc = ExceptionGroup("unhandled errors in a TaskGroup", [exc])

        described = _describe_exception(exc)

        assert described.startswith("ExceptionGroup: unhandled errors")
        assert len(described) == 1_000

    def test_a_long_error_is_cut(self):
        described = _describe_exception(RuntimeError("x" * 20_000))
        msg = _build_error_message(_make_request(), RuntimeError("x" * 20_000))

        assert len(described) == 1_000
        assert described.endswith("x... (truncated)")
        assert described in msg.content

    def test_content_includes_retry_guidance(self):
        msg = _build_error_message(_make_request(), Exception("fail"))

        assert "retry" in msg.content.lower()

    def test_the_terminal_shows_any_tool_error_as_failed(self):
        class ConnectTimeout(Exception):
            pass

        msg = _build_error_message(_make_request(), ConnectTimeout("timed out"))

        assert is_success(msg.content) is False
        assert ToolResultFormatter().detect_type(msg.content) is ContentType.ERROR

    def test_output_that_only_quotes_the_marker_is_not_a_failure(self):
        found = '80:        "[TOOL ERROR]",\n'

        assert is_success(found) is True
        assert ToolResultFormatter().detect_type(found) is not ContentType.ERROR


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
        assert "Traceback" not in result.content

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

    async def test_success_passes_through(self):
        expected = ToolMessage(content="ok", tool_call_id="tc_001", name="t")

        async def handler(req):
            return expected

        req = _make_request()
        result = await self.mw.awrap_tool_call(req, handler)

        assert result is expected

    async def test_command_passes_through(self):
        cmd = Command(update={"messages": []})

        async def handler(req):
            return cmd

        req = _make_request()
        result = await self.mw.awrap_tool_call(req, handler)

        assert result is cmd

    async def test_exception_returns_error_tool_message(self):
        async def handler(req):
            raise RuntimeError("MCP server timed out")

        req = _make_request("slow_tool", "tc_async")
        result = await self.mw.awrap_tool_call(req, handler)

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert result.tool_call_id == "tc_async"
        assert result.name == "slow_tool"
        assert "MCP server timed out" in result.content
        assert "Traceback" not in result.content

    async def test_exception_does_not_propagate(self):
        async def handler(req):
            raise ConnectionError("connection lost")

        req = _make_request()
        result = await self.mw.awrap_tool_call(req, handler)
        assert isinstance(result, ToolMessage)

    async def test_keyboard_interrupt_propagates(self):
        async def handler(req):
            raise KeyboardInterrupt()

        req = _make_request()
        with pytest.raises(KeyboardInterrupt):
            await self.mw.awrap_tool_call(req, handler)


# ---------------------------------------------------------------------------
# Middleware metadata
# ---------------------------------------------------------------------------


class TestMiddlewareMeta:
    def test_name(self):
        assert ToolErrorHandlerMiddleware.name == "tool_error_handler"

    def test_instantiation(self):
        mw = ToolErrorHandlerMiddleware()
        assert mw.name == "tool_error_handler"
