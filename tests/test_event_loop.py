"""Tests for runtime-loop streaming in the Rich display."""

import asyncio
import threading
from unittest.mock import Mock, patch

import pytest

from EvoScientist.runtime import AgentRuntime
from EvoScientist.stream import display as display_mod
from tests.fakes import FakeGraphGateway


@pytest.fixture
def display_runtime(monkeypatch):
    runtime = AgentRuntime()
    monkeypatch.setattr(display_mod, "runtime", runtime)
    try:
        yield runtime
    finally:
        runtime.close()


def test_sequential_streaming_calls_reuse_runtime_loop(display_runtime):
    mock_agent = Mock()
    stream_threads: list[str] = []

    async def mock_stream(_request):
        stream_threads.append(threading.current_thread().name)
        yield {"type": "text", "content": "test response"}
        yield {"type": "done", "response": "test response"}

    gateway = FakeGraphGateway(stream=mock_stream)

    with patch("EvoScientist.stream.display.Live"):
        for index in range(3):
            display_mod._run_streaming(
                agent=mock_agent,
                message=f"test message {index}",
                thread_id="thread1",
                show_thinking=False,
                interactive=True,
                gateway=gateway,
            )

    assert stream_threads == ["evosci-runtime"] * 3


def test_recursive_streaming_does_not_resend_same_thinking(display_runtime):
    mock_agent = Mock()
    thinking = "Initial plan. " * 20
    stream_calls = 0

    async def mock_stream(_request):
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

    with patch("EvoScientist.stream.display.Live"):
        result = display_mod._run_streaming(
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
            gateway=FakeGraphGateway(stream=mock_stream),
        )

    assert result == "final answer"
    assert sent_thinking == [thinking.rstrip()]


def test_recursive_streaming_sends_new_thinking_after_resume(display_runtime):
    mock_agent = Mock()
    thinking_r1 = "Initial plan. " * 20
    thinking_r2 = "Revised plan. " * 20
    stream_calls = 0

    async def mock_stream(_request):
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
        yield {"type": "done", "response": "final answer"}

    sent_thinking: list[str] = []

    with patch("EvoScientist.stream.display.Live"):
        result = display_mod._run_streaming(
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
            gateway=FakeGraphGateway(stream=mock_stream),
        )

    assert result == "final answer"
    assert sent_thinking == [thinking_r1.rstrip(), thinking_r2.rstrip()]


def test_keyboard_interrupt_cancels_stream_and_waits_for_cleanup(
    display_runtime, monkeypatch
):
    stream_blocked = threading.Event()
    stream_cleanup = threading.Event()

    async def blocking_stream(_request):
        yield {"type": "text", "content": "partial response"}
        stream_blocked.set()
        try:
            await asyncio.Event().wait()
        finally:
            stream_cleanup.set()

    real_submit = display_runtime.submit

    class _InterruptingFuture:
        def __init__(self, future):
            self._future = future
            self._raised = False

        def result(self, timeout=None):
            if not self._raised:
                self._raised = True
                assert stream_blocked.wait(5)
                raise KeyboardInterrupt
            return self._future.result(timeout)

        def cancel(self):
            return self._future.cancel()

    monkeypatch.setattr(
        display_runtime,
        "submit",
        lambda coro: _InterruptingFuture(real_submit(coro)),
    )

    with (
        patch("EvoScientist.stream.display.Live"),
        pytest.raises(KeyboardInterrupt),
    ):
        display_mod._run_streaming(
            agent=Mock(),
            message="test message",
            thread_id="thread1",
            show_thinking=False,
            interactive=True,
            cancel_scope="serve:message-1",
            gateway=FakeGraphGateway(stream=blocking_stream),
        )

    assert stream_cleanup.is_set()
    assert not display_mod.is_stream_cancel_requested("serve:message-1")
