"""Owned-runtime tests for the synchronous Rich streaming adapter."""

import asyncio
import threading
from unittest.mock import Mock, patch

import pytest

from EvoScientist.runtime import AsyncRuntime, AsyncRuntimeError
from EvoScientist.stream.display import _run_streaming
from tests.fakes import FakeGraphGateway


def _text_stream(loops, response="test response"):
    async def _stream(_request):
        loops.append(asyncio.get_running_loop())
        yield {"type": "text", "content": response}
        yield {"type": "done", "response": response}

    return _stream


def test_sequential_streams_reuse_application_runtime_loop():
    loops: list[asyncio.AbstractEventLoop] = []
    gateway = FakeGraphGateway(stream=_text_stream(loops))

    with (
        AsyncRuntime(thread_name="test-stream-runtime") as runtime,
        patch("EvoScientist.stream.display.Live"),
    ):
        for index in range(3):
            result = _run_streaming(
                agent=Mock(),
                message=f"message {index}",
                thread_id="thread1",
                show_thinking=False,
                interactive=True,
                gateway=gateway,
                runtime=runtime,
            )
            assert result == "test response"

    assert len(loops) == 3
    assert loops[0] is loops[1] is loops[2]


def test_direct_streaming_call_scopes_and_closes_runtime():
    execution: dict[str, object] = {}

    async def stream(_request):
        execution["thread"] = threading.current_thread().name
        execution["loop"] = asyncio.get_running_loop()
        yield {"type": "done", "response": "ok"}

    with patch("EvoScientist.stream.display.Live"):
        result = _run_streaming(
            agent=Mock(),
            message="message",
            thread_id="thread1",
            show_thinking=False,
            interactive=True,
            gateway=FakeGraphGateway(stream=stream),
        )

    assert result == "ok"
    assert execution["thread"] == "evosci-stream-runtime"
    assert isinstance(execution["loop"], asyncio.AbstractEventLoop)
    assert not any(
        thread.name == "evosci-stream-runtime" and thread.is_alive()
        for thread in threading.enumerate()
    )


async def test_async_caller_must_offload_synchronous_renderer():
    gateway = FakeGraphGateway(stream=_text_stream([]))

    with (
        AsyncRuntime(thread_name="test-stream-runtime") as runtime,
        patch("EvoScientist.stream.display.Live"),
        pytest.raises(AsyncRuntimeError, match="running event loop"),
    ):
        _run_streaming(
            agent=Mock(),
            message="message",
            thread_id="thread1",
            show_thinking=False,
            interactive=True,
            gateway=gateway,
            runtime=runtime,
        )


@pytest.mark.parametrize(
    ("second_thinking", "expected_count"),
    [(None, 1), ("Revised plan. " * 20, 2)],
)
def test_recursive_streaming_reuses_runtime_and_deduplicates_thinking(
    second_thinking, expected_count
):
    initial_thinking = "Initial plan. " * 20
    stream_calls = 0
    loops: list[asyncio.AbstractEventLoop] = []

    async def stream(_request):
        nonlocal stream_calls
        loops.append(asyncio.get_running_loop())
        stream_calls += 1
        if stream_calls == 1:
            yield {"type": "thinking", "content": initial_thinking}
            yield {
                "type": "ask_user",
                "interrupt_id": "ask-1",
                "tool_call_id": "tc-1",
                "questions": [{"question": "Continue?"}],
            }
            return
        if second_thinking is not None:
            yield {"type": "thinking", "content": second_thinking}
        else:
            yield {"type": "thinking", "content": initial_thinking}
        yield {"type": "text", "content": "final answer"}
        yield {"type": "done", "response": "final answer"}

    sent_thinking: list[str] = []
    with (
        AsyncRuntime(thread_name="test-stream-runtime") as runtime,
        patch("EvoScientist.stream.display.Live"),
    ):
        result = _run_streaming(
            agent=Mock(),
            message="test message",
            thread_id="thread1",
            show_thinking=False,
            interactive=True,
            on_thinking=sent_thinking.append,
            ask_user_prompt_fn=lambda _data: {
                "answers": ["yes"],
                "status": "answered",
            },
            gateway=FakeGraphGateway(stream=stream),
            runtime=runtime,
        )

    assert result == "final answer"
    assert len(sent_thinking) == expected_count
    assert sent_thinking[0] == initial_thinking.rstrip()
    if second_thinking is not None:
        assert sent_thinking[1] == second_thinking.rstrip()
    assert loops[0] is loops[1]
