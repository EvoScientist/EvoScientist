"""The ask_user resume paths of issue #762 outside the Rich CLI loop.

``build_ask_user_resume`` keys every resume by the interrupt id of the event
the client actually displayed. The Rich CLI loop is covered by
``test_ask_user_multi_interrupt_graph``; this file covers the rest:

- the helper itself, including the unkeyed fallback it owns: a missing id
  must resume FLAT, because ``{"default": payload}`` is not an id map
  LangGraph recognises (only xxh3-128 hex digest keys are) and would be
  handed to the node as the answer itself, losing the user's answer;
- the channels consumer (``InboundConsumer._stream_with_hitl``): the
  ask_user events of one turn are drained one ``_resolve_ask_user`` round
  each, every resume keyed by the displayed event's interrupt id — and a
  keyless event resumes flat, not keyed ``"default"``;
- the TUI branch (``tui_interactive`` ask_user handling): the same drain
  through the interactive widget wait, in display order.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("textual")

from langgraph.types import Command  # type: ignore[import-untyped]

from tests.fakes.common import FakeGraphGateway

ANSWER_RED = {"answers": ["red"], "status": "answered"}
ANSWER_LARGE = {"answers": ["large"], "status": "answered"}


def _ask(interrupt_id: str, question: str) -> dict:
    return {
        "type": "ask_user",
        "interrupt_id": interrupt_id,
        "questions": [{"question": question, "type": "text"}],
    }


def _ask_without_id(question: str) -> dict:
    """A keyless ask_user event: the case the helper's fallback owns."""
    return {"type": "ask_user", "questions": [{"question": question, "type": "text"}]}


class TestBuildAskUserResumeFallback:
    """The helper owns the "no id" fallback the call sites used to spell."""

    def test_a_keyless_event_resumes_flat_and_a_real_id_keys(self):
        from EvoScientist.backends import build_ask_user_resume

        # {"default": payload} would be passed into the node as the answer
        # itself (LangGraph only unwraps xxh3-128 digest keys), so the
        # user's answer would be lost. A flat resume is what worked before
        # keying and is still right for a single unkeyed interrupt.
        for missing in (None, "", "default"):
            cmd = build_ask_user_resume(missing, ANSWER_RED)
            assert isinstance(cmd, Command)
            assert cmd.resume == ANSWER_RED, missing
        # A real id still keys the resume: that is what closes #762.
        cmd = build_ask_user_resume("19f1cd5c0a7b4f66", ANSWER_RED)
        assert cmd.resume == {"19f1cd5c0a7b4f66": ANSWER_RED}


def _consumer(workspace, gateway):
    from EvoScientist.channels.bus.message_bus import MessageBus
    from EvoScientist.channels.channel_manager import ChannelManager
    from EvoScientist.channels.consumer import InboundConsumer
    from tests.fakes.common import StubChannel

    bus = MessageBus()
    manager = ChannelManager(bus, media_dir=workspace.root / "media")
    manager.register(StubChannel())
    return InboundConsumer(
        bus=bus,
        manager=manager,
        agent=MagicMock(),
        thread_id="",
        workspace=workspace,
        graph_gateway=gateway,
        inference_timeout=5.0,
        drain_timeout=1.0,
    )


def _scripted_stream(script: list[list[dict]]):
    """One list of events per stream call; a call that ends on an ask_user
    event simply ends (the run parked), like the real stream."""
    calls = 0

    async def _fake_stream(_request):
        nonlocal calls
        events = script[calls]
        calls += 1
        for event in events:
            yield event

    return _fake_stream


def _pausing_stream(script: list[list[dict]]):
    """Like the TUI fake of the same name: a round that pauses holds its
    stream open until the resumed round closes it."""
    import asyncio

    calls = 0

    async def _fake_stream(_request):
        nonlocal calls
        events = script[calls]
        calls += 1
        for event in events:
            yield event
        if events and events[-1]["type"] in ("interrupt", "ask_user"):
            await asyncio.Event().wait()

    return _fake_stream


async def _consumer_turn(consumer) -> list:
    from EvoScientist.channels.bus.events import InboundMessage

    msg = InboundMessage(
        channel="stub", sender_id="u1", chat_id="c1", content="go", media=[]
    )
    await consumer._stream_with_hitl(msg, None, "t1", msg.session_key)
    queued = []
    while not consumer.bus.outbound.empty():
        queued.append(consumer.bus.outbound.get_nowait())
    return queued


class TestChannelsConsumerAskUserResume:
    """Two ask_user events in one step, drained through the consumer."""

    async def test_each_round_keys_its_resume_by_the_displayed_interrupt_id(
        self, workspace
    ):
        script = [
            # One step parks BOTH questions; the consumer stops at the
            # first ask_user event it sees. The sibling is keyless.
            [_ask("id-1", "Colour?"), _ask_without_id("Size?")],
            # The resume stream re-emits the still-pending sibling.
            [_ask_without_id("Size?")],
            [
                {"type": "text", "content": "Both answers received."},
                {"type": "done", "content": "Both answers received."},
            ],
        ]
        gateway = FakeGraphGateway(stream=_scripted_stream(script))
        consumer = _consumer(workspace, gateway)

        resolved: list[dict] = []

        async def _fake_resolve(_msg, interrupt_data, _session_key):
            resolved.append(interrupt_data)
            question = interrupt_data["questions"][0]["question"]
            return ANSWER_RED if question == "Colour?" else ANSWER_LARGE

        consumer._resolve_ask_user = AsyncMock(side_effect=_fake_resolve)

        queued = await _consumer_turn(consumer)

        # One _resolve_ask_user per round, each seeing the event displayed,
        # in first-emitted order.
        assert consumer._resolve_ask_user.await_count == 2
        assert [event["questions"][0]["question"] for event in resolved] == [
            "Colour?",
            "Size?",
        ]
        # The keyed sibling resumes by its interrupt id; the keyless one
        # resumes FLAT — not keyed by a made-up "default" id.
        resumes = [r.message for r in gateway.requests[1:]]
        assert len(resumes) == 2
        assert resumes[0].resume == {"id-1": ANSWER_RED}
        assert resumes[1].resume == ANSWER_LARGE
        # The turn produced its response on the channel.
        assert any("Both answers received." in out.content for out in queued)


class TestTuiAskUserResume:
    """The TUI branch: the widget wait feeds the resume, one round each."""

    async def test_each_round_keys_its_resume_by_the_displayed_interrupt_id(
        self, monkeypatch, workspace
    ):
        from EvoScientist.gateway import RuntimeGateways
        from tests.fakes.tui import FINISHED_REPLY, capture_app

        script = [
            # One step parks BOTH questions; the TUI stops at the first
            # ask_user event it sees. The sibling is keyless.
            [_ask("id-1", "Colour?"), _ask_without_id("Size?")],
            # The resume stream re-emits the still-pending sibling.
            [_ask_without_id("Size?")],
            FINISHED_REPLY,
        ]
        app = await capture_app(monkeypatch, workspace)
        gateway = FakeGraphGateway(stream=_pausing_stream(script))
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            app._runtime_gateways = RuntimeGateways(
                thread_store=gateway.thread_store, graph_gateway=gateway
            )
            monkeypatch.setattr(
                app, "_await_agent_ready", AsyncMock(return_value=MagicMock())
            )

            asked: list[str] = []

            async def _fake_wait(ask_w) -> dict:
                # The widget the loop mounted for THIS round's question.
                asked.append(ask_w._questions[0]["question"])
                return ANSWER_RED if asked[-1] == "Colour?" else ANSWER_LARGE

            monkeypatch.setattr(app, "_wait_for_ask_user", _fake_wait)

            response = await app._stream_with_widgets("pick for me")

        assert response == "finished"
        # The questions were asked in first-emitted order, one per round.
        assert asked == ["Colour?", "Size?"]
        # The keyed sibling resumes by its interrupt id; the keyless one
        # resumes FLAT — not keyed by a made-up "default" id.
        resumes = [r.message for r in gateway.requests[1:]]
        assert len(resumes) == 2
        assert resumes[0].resume == {"id-1": ANSWER_RED}
        assert resumes[1].resume == ANSWER_LARGE
