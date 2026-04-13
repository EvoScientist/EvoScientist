"""Tests for event loop management in streaming display."""

import asyncio
from unittest.mock import Mock, patch

from EvoScientist.stream.display import _create_event_loop, _get_event_loop


class TestCreateEventLoop:
    """Tests for _create_event_loop helper."""

    def test_creates_new_loop(self):
        """Should create a new event loop and set it as current."""
        # Get initial loop (if any)
        try:
            initial_loop = asyncio.get_event_loop()
            initial_loop.close()
        except RuntimeError:
            pass

        # Create new loop
        loop = _create_event_loop()

        assert loop is not None
        assert not loop.is_closed()
        assert asyncio.get_event_loop() is loop

        # Cleanup
        loop.close()

    def test_replaces_closed_loop(self):
        """Should replace a closed loop."""
        old_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(old_loop)
        old_loop.close()

        new_loop = _create_event_loop()

        assert new_loop is not old_loop
        assert not new_loop.is_closed()
        assert asyncio.get_event_loop() is new_loop

        # Cleanup
        new_loop.close()


class TestGetEventLoop:
    """Tests for _get_event_loop helper."""

    def test_returns_existing_open_loop(self):
        """Should return existing event loop if it's open."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        result = _get_event_loop()

        assert result is loop
        assert not result.is_closed()

        # Cleanup
        loop.close()

    def test_creates_new_loop_when_closed(self):
        """Should create new event loop if current one is closed."""
        old_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(old_loop)
        old_loop.close()

        result = _get_event_loop()

        assert result is not old_loop
        assert not result.is_closed()

        # Cleanup
        result.close()

    def test_handles_no_event_loop(self):
        """Should handle RuntimeError when no event loop exists (edge case)."""
        # This test simulates what happens in a worker thread
        # In practice, get_event_loop() returns a closed loop, not RuntimeError
        # But we handle the RuntimeError case defensively
        loop = _get_event_loop()
        assert loop is not None
        assert not loop.is_closed()

        # Cleanup
        loop.close()


class TestMultipleStreamingCalls:
    """Tests for the main bug fix: multiple _run_streaming calls."""

    def test_sequential_streaming_calls(self):
        """Multiple sequential calls should work without 'Event loop is closed' error."""
        from EvoScientist.stream.display import _run_streaming

        # Mock agent that returns simple events
        mock_agent = Mock()

        async def mock_stream(*args, **kwargs):
            """Mock event stream."""
            yield {"type": "text", "content": "test response"}
            yield {"type": "done", "response": "test response"}

        # Clean up any existing event loop to start fresh
        try:
            existing_loop = asyncio.get_event_loop()
            if not existing_loop.is_closed():
                existing_loop.close()
        except RuntimeError:
            pass

        # Patch the stream_agent_events function
        with patch(
            "EvoScientist.stream.display.stream_agent_events", side_effect=mock_stream
        ):
            # Patch Live to avoid terminal output during tests
            with patch("EvoScientist.stream.display.Live"):
                # First call
                _run_streaming(
                    agent=mock_agent,
                    message="test message 1",
                    thread_id="thread1",
                    show_thinking=False,
                    interactive=True,
                )

                # Second call - this would fail with "Event loop is closed" before the fix
                _run_streaming(
                    agent=mock_agent,
                    message="test message 2",
                    thread_id="thread1",
                    show_thinking=False,
                    interactive=True,
                )

                # Third call for good measure
                _run_streaming(
                    agent=mock_agent,
                    message="test message 3",
                    thread_id="thread1",
                    show_thinking=False,
                    interactive=True,
                )

    def test_loop_reused_across_calls(self):
        """Event loop should be reused across multiple calls."""
        # Create a fresh loop
        loop = _create_event_loop()

        # Simulate multiple calls
        for _ in range(3):
            current_loop = _get_event_loop()
            assert not current_loop.is_closed()

            # Run a simple coroutine
            async def dummy():
                return "ok"

            result = current_loop.run_until_complete(dummy())
            assert result == "ok"

        # Loop should still be open
        assert not loop.is_closed()

        # Cleanup
        loop.close()

    def test_closed_loop_recovery(self):
        """If loop gets closed, next call should create a new one."""
        # Create and close a loop
        loop1 = _create_event_loop()
        loop1.close()

        # Next call should detect closed loop and create new one
        loop2 = _get_event_loop()

        assert loop2 is not loop1
        assert not loop2.is_closed()

        # Should be able to use the new loop
        async def dummy():
            return "success"

        result = loop2.run_until_complete(dummy())
        assert result == "success"

        # Cleanup
        loop2.close()

    def test_recursive_streaming_does_not_resend_same_thinking(self):
        """Resumed runs should not replay the original thinking to channels."""
        from EvoScientist.stream.display import _run_streaming

        mock_agent = Mock()
        thinking = "Initial plan. " * 20
        stream_calls = 0

        async def mock_stream(*args, **kwargs):
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

            yield {"type": "text", "content": "final answer"}
            yield {"type": "done", "response": "final answer"}

        sent_thinking: list[str] = []

        with patch(
            "EvoScientist.stream.display.stream_agent_events",
            side_effect=mock_stream,
        ):
            with patch("EvoScientist.stream.display.Live"):
                result = _run_streaming(
                    agent=mock_agent,
                    message="test message",
                    thread_id="thread1",
                    show_thinking=False,
                    interactive=True,
                    on_thinking=sent_thinking.append,
                    ask_user_prompt_fn=lambda _data: {
                        "answers": ["yes"],
                        "status": "answered",
                    },
                )

        assert result == "final answer"
        assert sent_thinking == [thinking.rstrip()]


class TestEventLoopThreadSafety:
    """Tests for thread safety edge cases."""

    def test_main_thread_normal_case(self):
        """Normal case in main thread should work."""
        loop = _get_event_loop()
        assert loop is not None
        assert not loop.is_closed()

        # Cleanup
        loop.close()
