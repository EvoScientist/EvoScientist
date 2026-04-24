"""Tests for channel-initiated stream cancellation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from EvoScientist.stream import display as display_mod


@pytest.fixture(autouse=True)
def _clean_cancel_event():
    """Ensure all stream-cancel scopes start clear for every test."""
    with display_mod._stream_cancel_lock:
        display_mod._stream_cancel_event.clear()
        display_mod._stream_cancel_events.clear()
        display_mod._stream_cancel_events[display_mod._DEFAULT_STREAM_CANCEL_SCOPE] = (
            display_mod._stream_cancel_event
        )
    yield
    with display_mod._stream_cancel_lock:
        display_mod._stream_cancel_event.clear()
        display_mod._stream_cancel_events.clear()
        display_mod._stream_cancel_events[display_mod._DEFAULT_STREAM_CANCEL_SCOPE] = (
            display_mod._stream_cancel_event
        )


# ---------------------------------------------------------------------------
# 1. _consume breaks on cancel event
# ---------------------------------------------------------------------------


def test_consume_breaks_on_cancel_event():
    """After set(), ``_consume`` should stop pulling events and mark
    ``state.response_text`` with the ``[Stopped.]`` suffix."""
    seen_events: list[int] = []
    cancel_scope = "scope:consume"

    async def _fake_stream(agent, message, thread_id, **kwargs):
        for i in range(100):
            if i == 3:
                # Set during iteration — next loop iter should bail.
                display_mod.request_stream_cancel(cancel_scope)
            seen_events.append(i)
            yield {"type": "text", "content": f"chunk-{i}"}

    with patch(
        "EvoScientist.stream.display.stream_agent_events",
        new=_fake_stream,
    ):
        result = display_mod._run_streaming(
            agent=MagicMock(),
            message="hello",
            thread_id="t1",
            show_thinking=False,
            interactive=True,
            cancel_scope=cancel_scope,
        )

    # We set the flag during event index 3; the cancel check runs at the
    # top of the NEXT iteration (index 4), so indices 0-3 are pulled from
    # the generator before exit.
    assert len(seen_events) <= 5
    assert "[Stopped.]" in result


# ---------------------------------------------------------------------------
# 2. fresh _run_streaming clears stale set event
# ---------------------------------------------------------------------------


def test_run_streaming_short_circuits_when_scope_already_cancelled():
    """A queued request that is cancelled before start should stop immediately."""
    seen_event = False
    cancel_scope = "scope:queued"

    async def _fake_stream(agent, message, thread_id, **kwargs):
        nonlocal seen_event
        seen_event = True
        yield {"type": "text", "content": "ok"}

    display_mod.request_stream_cancel(cancel_scope)

    with patch(
        "EvoScientist.stream.display.stream_agent_events",
        new=_fake_stream,
    ):
        result = display_mod._run_streaming(
            agent=MagicMock(),
            message="hello",
            thread_id="t1",
            show_thinking=False,
            interactive=True,
            cancel_scope=cancel_scope,
        )

    assert result == "[Stopped.]"
    assert seen_event is False
    assert not display_mod.is_stream_cancel_requested(cancel_scope)


def test_run_streaming_ignores_other_scope_cancel():
    """Cancelling one scope must not bleed into a different stream."""
    display_mod.request_stream_cancel("scope:other")

    async def _fake_stream(agent, message, thread_id, **kwargs):
        yield {"type": "text", "content": "ok"}

    with patch(
        "EvoScientist.stream.display.stream_agent_events",
        new=_fake_stream,
    ):
        result = display_mod._run_streaming(
            agent=MagicMock(),
            message="hello",
            thread_id="t1",
            show_thinking=False,
            interactive=True,
            cancel_scope="scope:self",
        )

    assert "[Stopped.]" not in result


# ---------------------------------------------------------------------------
# 3. pending HITL/ask_user branches short-circuit when stop is requested
# ---------------------------------------------------------------------------


def test_run_streaming_pending_interrupt_short_circuits_on_cancel():
    """If cancel is already set, pending HITL prompt should not run."""

    async def _empty_stream(agent, message, thread_id, **kwargs):
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

    with patch("EvoScientist.stream.display.stream_agent_events", new=_empty_stream):
        result = display_mod._run_streaming(
            agent=MagicMock(),
            message="hello",
            thread_id="t1",
            show_thinking=False,
            interactive=True,
            hitl_prompt_fn=_prompt,
            cancel_scope="scope:hitl",
            _state=state,
        )

    assert result == "Partial answer\n[Stopped.]"
    assert prompt_called is False
