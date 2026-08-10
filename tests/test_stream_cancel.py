"""Tests for channel-initiated stream cancellation."""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from EvoScientist.backends import CustomSandboxBackend
from EvoScientist.cancellation import current_cancel_event
from EvoScientist.runtime import AsyncRuntime
from EvoScientist.stream import display as display_mod
from tests.fakes import FakeGraphGateway


@pytest.fixture(autouse=True)
def _clean_cancel_event():
    """Ensure all stream-cancel scopes start clear for every test."""
    with display_mod._stream_cancel_lock:
        display_mod._stream_cancel_event.clear()
        display_mod._stream_cancel_events.clear()
        display_mod._stream_cancel_events[display_mod._DEFAULT_STREAM_CANCEL_SCOPE] = (
            display_mod._stream_cancel_event
        )
        display_mod._stream_cancel_handles.clear()
    yield
    with display_mod._stream_cancel_lock:
        display_mod._stream_cancel_event.clear()
        display_mod._stream_cancel_events.clear()
        display_mod._stream_cancel_events[display_mod._DEFAULT_STREAM_CANCEL_SCOPE] = (
            display_mod._stream_cancel_event
        )
        display_mod._stream_cancel_handles.clear()


# ---------------------------------------------------------------------------
# 1. _consume breaks on cancel event
# ---------------------------------------------------------------------------


def test_consume_breaks_on_cancel_event():
    """After set(), ``_consume`` should stop pulling events and mark
    ``state.response_text`` with the ``[Stopped.]`` suffix."""
    seen_events: list[int] = []
    cancel_scope = "scope:consume"

    async def _fake_stream(_request):
        for i in range(100):
            if i == 3:
                # Set during iteration — next loop iter should bail.
                display_mod.request_stream_cancel(cancel_scope)
            seen_events.append(i)
            yield {"type": "text", "content": f"chunk-{i}"}

    result = display_mod._run_streaming(
        agent=MagicMock(),
        message="hello",
        thread_id="t1",
        show_thinking=False,
        interactive=True,
        cancel_scope=cancel_scope,
        gateway=FakeGraphGateway(stream=_fake_stream),
    )

    # We set the flag during event index 3; the cancel check runs at the
    # top of the NEXT iteration (index 4), so indices 0-3 are pulled from
    # the generator before exit.
    assert len(seen_events) <= 5
    assert "[Stopped.]" in result


def test_cancel_interrupts_stalled_stream_and_closes_it_in_consumer_task():
    """Cancellation must not wait for a stalled gateway to yield again."""
    cancel_scope = "scope:stalled"
    stream_started = threading.Event()
    stream_closed = threading.Event()
    tasks: dict[str, asyncio.Task[object] | None] = {}
    result: dict[str, str] = {}

    async def _stalled_stream(_request):
        tasks["consumer"] = asyncio.current_task()
        stream_started.set()
        try:
            await asyncio.Event().wait()
            if False:
                yield {}
        finally:
            tasks["closer"] = asyncio.current_task()
            stream_closed.set()

    def _run() -> None:
        result["response"] = display_mod._run_streaming(
            agent=MagicMock(),
            message="hello",
            thread_id="t1",
            show_thinking=False,
            interactive=True,
            cancel_scope=cancel_scope,
            gateway=FakeGraphGateway(stream=_stalled_stream),
        )

    worker = threading.Thread(target=_run)
    worker.start()
    assert stream_started.wait(2)

    display_mod.request_stream_cancel(cancel_scope)
    worker.join(2)

    assert not worker.is_alive()
    assert stream_closed.is_set()
    assert tasks["closer"] is tasks["consumer"]
    assert result["response"] == "[Stopped.]"


def test_cancel_interrupts_owned_questionary_prompt():
    """A terminal prompt must settle instead of outliving the frontend turn."""
    cancel_scope = "scope:questionary"
    started = threading.Event()
    closed = threading.Event()
    result: dict[str, object] = {}

    class _BlockingQuestion:
        async def ask_async(self):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()

    def _run(runtime: AsyncRuntime) -> None:
        try:
            display_mod._run_owned_questionary_prompt(
                _BlockingQuestion(),
                runtime=runtime,
                cancel_scope=cancel_scope,
            )
        except display_mod._StreamPromptCancelled:
            result["cancelled"] = True

    with AsyncRuntime(thread_name="test-questionary-runtime") as runtime:
        worker = threading.Thread(target=_run, args=(runtime,))
        worker.start()
        assert started.wait(2)

        display_mod.request_stream_cancel(cancel_scope)
        worker.join(2)

    assert not worker.is_alive()
    assert closed.is_set()
    assert result == {"cancelled": True}


def test_cancelled_stream_does_not_repaint_final_live_frame():
    """Late stream cleanup must not overwrite a newer frontend frame."""
    live = MagicMock()
    handle = display_mod.RuntimeHandle()
    handle.cancel()

    display_mod._update_final_live_frame(live, object(), handle)

    live.update.assert_not_called()
    live.refresh.assert_not_called()


def test_cancel_unwinds_hitl_prompt_and_renderer(monkeypatch):
    """The real Rich HITL branch must release its prompt before returning."""
    cancel_scope = "scope:hitl-questionary"
    started = threading.Event()
    closed = threading.Event()
    result: dict[str, str] = {}

    class _BlockingQuestion:
        async def ask_async(self):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()

        def ask(self):  # pragma: no cover - the owned path must use ask_async
            raise AssertionError("blocking questionary.ask() was used")

    monkeypatch.setitem(
        sys.modules,
        "questionary",
        SimpleNamespace(select=lambda *_args, **_kwargs: _BlockingQuestion()),
    )
    monkeypatch.setattr(
        "EvoScientist.config.settings.load_config",
        lambda: SimpleNamespace(
            auto_approve=False, dangerous_mode=False, shell_allow_list=""
        ),
    )

    async def _empty_stream(_request):
        if False:
            yield {}

    state = display_mod.StreamState()
    state.response_text = "Partial answer"
    state.pending_interrupt = {
        "action_requests": [{"name": "execute", "args": {"command": "echo hi"}}]
    }

    def _run(runtime: AsyncRuntime) -> None:
        result["response"] = display_mod._run_streaming(
            agent=MagicMock(),
            message="hello",
            thread_id="t1",
            show_thinking=False,
            interactive=True,
            cancel_scope=cancel_scope,
            _state=state,
            gateway=FakeGraphGateway(stream=_empty_stream),
            runtime=runtime,
        )

    with AsyncRuntime(thread_name="test-hitl-runtime") as runtime:
        worker = threading.Thread(target=_run, args=(runtime,))
        worker.start()
        assert started.wait(2)

        display_mod.request_stream_cancel(cancel_scope)
        worker.join(2)

    assert not worker.is_alive()
    assert closed.is_set()
    assert result["response"] == "Partial answer\n[Stopped.]"


def test_cancel_terminates_active_shell_process_tree(tmp_path):
    """A cancelled turn must not leave delayed shell side effects running."""
    cancel_scope = "scope:shell"
    backend = CustomSandboxBackend(root_dir=str(tmp_path), virtual_mode=True)
    started = tmp_path / "started.txt"
    forbidden = tmp_path / "forbidden.txt"
    result: dict[str, object] = {}

    if sys.platform == "win32":
        command = (
            "echo started> started.txt & "
            "ping -n 11 127.0.0.1 > nul & "
            "echo late> forbidden.txt"
        )
    else:
        command = "printf started > started.txt; sleep 10; printf late > forbidden.txt"

    async def _events():
        result["response"] = await asyncio.to_thread(backend.execute, command)
        if False:
            yield {}

    async def _consume() -> None:
        async for _ in display_mod.iter_with_stream_cancel(_events(), cancel_scope):
            pass

    worker = threading.Thread(target=lambda: asyncio.run(_consume()))
    worker.start()
    deadline = time.monotonic() + 3
    while not started.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert started.read_text().strip() == "started"

    display_mod.request_stream_cancel(cancel_scope)
    worker.join(3)

    assert not worker.is_alive()
    assert result["response"].exit_code == 130
    assert not forbidden.exists()


async def test_stream_cancel_binding_can_close_in_different_task_context():
    """No ContextVar token may survive across an async-generator yield."""
    closed = False

    async def _events():
        nonlocal closed
        try:
            yield {"type": "text", "content": "one"}
            await asyncio.Event().wait()
        finally:
            closed = True

    wrapped = display_mod.iter_with_stream_cancel(_events(), "scope:cross-context")
    async for _ in wrapped:
        break

    assert current_cancel_event() is None
    await asyncio.create_task(wrapped.aclose())

    assert closed is True
    assert current_cancel_event() is None


# ---------------------------------------------------------------------------
# 2. fresh _run_streaming clears stale set event
# ---------------------------------------------------------------------------


def test_run_streaming_short_circuits_when_scope_already_cancelled():
    """A queued request that is cancelled before start should stop immediately."""
    seen_event = False
    cancel_scope = "scope:queued"

    async def _fake_stream(_request):
        nonlocal seen_event
        seen_event = True
        yield {"type": "text", "content": "ok"}

    display_mod.request_stream_cancel(cancel_scope)

    result = display_mod._run_streaming(
        agent=MagicMock(),
        message="hello",
        thread_id="t1",
        show_thinking=False,
        interactive=True,
        cancel_scope=cancel_scope,
        gateway=FakeGraphGateway(stream=_fake_stream),
    )

    assert result == "[Stopped.]"
    assert seen_event is False
    assert not display_mod.is_stream_cancel_requested(cancel_scope)


def test_run_streaming_ignores_other_scope_cancel():
    """Cancelling one scope must not bleed into a different stream."""
    display_mod.request_stream_cancel("scope:other")

    async def _fake_stream(_request):
        yield {"type": "text", "content": "ok"}

    result = display_mod._run_streaming(
        agent=MagicMock(),
        message="hello",
        thread_id="t1",
        show_thinking=False,
        interactive=True,
        cancel_scope="scope:self",
        gateway=FakeGraphGateway(stream=_fake_stream),
    )

    assert "[Stopped.]" not in result


# ---------------------------------------------------------------------------
# 3. pending HITL/ask_user branches short-circuit when stop is requested
# ---------------------------------------------------------------------------


def test_run_streaming_pending_interrupt_short_circuits_on_cancel():
    """If cancel is already set, pending HITL prompt should not run."""

    async def _empty_stream(_request):
        if False:
            yield {}

    state = display_mod.StreamState()
    state.response_text = "Partial answer"
    state.pending_interrupt = {
        "action_requests": [{"name": "execute", "args": {"command": "echo hi"}}]
    }
    display_mod.request_stream_cancel("scope:hitl")

    prompt_called = False

    def _prompt(_requests):
        nonlocal prompt_called
        prompt_called = True
        return None

    result = display_mod._run_streaming(
        agent=MagicMock(),
        message="hello",
        thread_id="t1",
        show_thinking=False,
        interactive=True,
        hitl_prompt_fn=_prompt,
        cancel_scope="scope:hitl",
        _state=state,
        gateway=FakeGraphGateway(stream=_empty_stream),
    )

    assert result == "Partial answer\n[Stopped.]"
    assert prompt_called is False
