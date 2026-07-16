"""Tests for the CLI channel bridge.

The bridge runs the shared interaction engine on the bus loop via
``run_coroutine_threadsafe`` while the calling thread blocks. Covered here:

* **Ordering** — the reply-interception point sits *ahead* of normal
  enqueue, so a reply to a pending prompt is delivered into the engine's
  wait and never becomes a fresh agent turn.
* **Bridge round-trip** — ``channel_hitl_prompt`` drives ``resolve_approval``
  end-to-end over a real bus loop and returns today's decision payloads.
"""

import asyncio
import threading

import pytest

from EvoScientist.channels import interaction as interaction_mod
from EvoScientist.channels.bus.events import InboundMessage
from EvoScientist.channels.bus.message_bus import MessageBus
from EvoScientist.channels.channel_manager import ChannelManager
from EvoScientist.cli import channel as channel_mod
from EvoScientist.cli.channel import ChannelMessage
from tests.fakes import QueueFakeChannel


def _reset_channel_state():
    channel_mod._reply_registry.clear()
    channel_mod._approval_policy.clear_sessions()
    while not channel_mod._message_queue.empty():
        channel_mod._message_queue.get_nowait()
    with channel_mod._response_lock:
        channel_mod._pending_responses.clear()
    with channel_mod._channel_request_lock:
        channel_mod._channel_requests.clear()
        channel_mod._session_requests.clear()
        channel_mod._cancelled_channel_messages.clear()


@pytest.fixture(autouse=True)
def _clean_bridge_state():
    _reset_channel_state()
    yield
    _reset_channel_state()


# ═══════════════════════════════════════════════════════════════════════
# Reply interception sits ahead of normal enqueue
# ═══════════════════════════════════════════════════════════════════════


class TestReplyInterceptionOrdering:
    async def test_pending_reply_intercepted_not_enqueued(self):
        """A reply to a pending prompt resolves the wait and is NOT enqueued."""
        from EvoScientist.cli.channel import _bus_inbound_consumer, _message_queue

        bus = MessageBus()
        manager = ChannelManager(bus)
        manager.register(QueueFakeChannel())

        # A prompt is waiting on fake:chat1 (as the engine's wait_reply would).
        reply_fut = asyncio.ensure_future(
            channel_mod._reply_registry.wait("fake:chat1", timeout=5.0)
        )
        await asyncio.sleep(0.01)

        consumer = asyncio.create_task(_bus_inbound_consumer(bus, manager))
        await bus.publish_inbound(
            InboundMessage(
                channel="fake",
                sender_id="user1",
                chat_id="chat1",
                content="1",
                message_id="m-reply",
            )
        )

        # The reply is delivered into the pending wait...
        got = await asyncio.wait_for(reply_fut, timeout=2.0)
        assert got == "1"
        # ...and did NOT become a queued agent turn.
        await asyncio.sleep(0.05)
        assert _message_queue.empty()

        consumer.cancel()
        try:
            await consumer
        except asyncio.CancelledError:
            pass

    async def test_message_without_pending_wait_is_enqueued(self):
        """With no pending prompt, a normal message flows to enqueue as before."""
        from EvoScientist.cli.channel import _bus_inbound_consumer, _message_queue

        bus = MessageBus()
        manager = ChannelManager(bus)
        manager.register(QueueFakeChannel())

        consumer = asyncio.create_task(_bus_inbound_consumer(bus, manager))
        await bus.publish_inbound(
            InboundMessage(
                channel="fake",
                sender_id="user1",
                chat_id="chat1",
                content="hello there",
                message_id="m-normal",
            )
        )

        queued = None
        for _ in range(40):
            with _message_queue.mutex:
                queued = _message_queue.queue[0] if _message_queue.queue else None
            if queued is not None:
                break
            await asyncio.sleep(0.02)
        assert queued is not None
        assert queued.content == "hello there"

        consumer.cancel()
        try:
            await consumer
        except asyncio.CancelledError:
            pass
        # Drain the pending waiter created by _handle_bus_message.
        channel_mod._pop_channel_response(queued.msg_id, cancel_pending=True)


# ═══════════════════════════════════════════════════════════════════════
# Bridge timeout budgeting and cancellation
# ═══════════════════════════════════════════════════════════════════════


class TestBridgeTimeouts:
    def test_ask_user_outer_timeout_includes_waits_and_sends(self):
        question_count = 2
        per_question_worst_case = (
            channel_mod.ASK_USER_TIMEOUT * channel_mod._ASK_USER_WAITS_PER_QUESTION
            + channel_mod._BRIDGE_SEND_TIMEOUT
            * channel_mod._ASK_USER_SENDS_PER_QUESTION
        )

        assert channel_mod._ask_user_result_timeout(question_count) == (
            per_question_worst_case * question_count + channel_mod._ENGINE_RESULT_SLACK
        )

    def test_hitl_outer_timeout_exceeds_engine_worst_case(self):
        engine_worst_case = (
            channel_mod.HITL_APPROVAL_TIMEOUT
            + channel_mod._BRIDGE_SEND_TIMEOUT * channel_mod._HITL_SENDS_PER_APPROVAL
        )

        assert channel_mod._hitl_result_timeout() == (
            engine_worst_case + channel_mod._ENGINE_RESULT_SLACK
        )
        assert channel_mod._hitl_result_timeout() > engine_worst_case

    def test_outer_timeout_cancels_engine_and_releases_reply_slot(self, monkeypatch):
        session_key = "fake:timeout"
        registered = threading.Event()
        cancelled = threading.Event()

        async def _wait_forever():
            channel_mod._reply_registry.register(session_key)
            registered.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            finally:
                channel_mod._reply_registry.discard(session_key)

        with _BusLoopThread() as loop:
            monkeypatch.setattr(channel_mod, "_bus_loop", loop)
            result = channel_mod._run_engine_on_bus(
                _wait_forever(),
                result_timeout=0.2,
                on_error=lambda: "cancelled",
            )

        assert result == "cancelled"
        assert registered.wait(timeout=1.0)
        assert cancelled.wait(timeout=1.0)
        assert channel_mod._reply_registry.try_resolve(session_key, "late") is False


# ═══════════════════════════════════════════════════════════════════════
# Bridge round-trip: channel_hitl_prompt over a real bus loop
# ═══════════════════════════════════════════════════════════════════════


class _BusLoopThread:
    """A dedicated event loop running in a background thread (like the bus)."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self.loop

    def __exit__(self, *exc):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=2)
        self.loop.close()


def _feed_reply_when_ready(loop, session_key, reply, *, tries=200):
    """Schedule a coroutine on *loop* that resolves the pending wait."""

    async def _feeder():
        for _ in range(tries):
            if session_key in channel_mod._reply_registry:
                channel_mod._reply_registry.try_resolve(session_key, reply)
                return
            await asyncio.sleep(0.01)

    asyncio.run_coroutine_threadsafe(_feeder(), loop)


class TestHitlPromptBridge:
    def test_no_bus_loop_rejects(self, monkeypatch):
        monkeypatch.setattr(interaction_mod, "config_auto_approve", lambda reqs: False)
        monkeypatch.setattr(channel_mod, "_bus_loop", None)
        msg = ChannelMessage(
            msg_id="m1",
            content="",
            sender="u1",
            channel_type="fake",
            chat_id="chat1",
            bus_ref=object(),
        )
        assert channel_mod.channel_hitl_prompt([{"name": "execute"}], msg) is None

    def test_session_grant_approves_when_bus_loop_down(self, monkeypatch):
        monkeypatch.setattr(interaction_mod, "config_auto_approve", lambda reqs: False)
        monkeypatch.setattr(channel_mod, "_bus_loop", None)
        msg = ChannelMessage(
            msg_id="m1",
            content="",
            sender="u1",
            channel_type="fake",
            chat_id="chat1",
            bus_ref=object(),
        )

        channel_mod._approval_policy.grant_session("fake:chat1")

        assert channel_mod.channel_hitl_prompt([{"name": "execute"}], msg) == [
            {"type": "approve"}
        ]

    def test_approve_round_trip(self, monkeypatch):
        # Force the manual-prompt path (no config auto-approve).
        monkeypatch.setattr(interaction_mod, "config_auto_approve", lambda reqs: False)
        with _BusLoopThread() as loop:
            monkeypatch.setattr(channel_mod, "_bus_loop", loop)
            monkeypatch.setattr(channel_mod, "_manager", None)  # default caps
            bus = MessageBus()
            msg = ChannelMessage(
                msg_id="m1",
                content="",
                sender="u1",
                channel_type="fake",
                chat_id="chat1",
                bus_ref=bus,
                metadata={},
            )
            _feed_reply_when_ready(loop, "fake:chat1", "1")
            result = channel_mod.channel_hitl_prompt(
                [{"name": "execute", "args": {"command": "ls"}}], msg
            )
        assert result == [{"type": "approve"}]

    def test_reject_round_trip(self, monkeypatch):
        monkeypatch.setattr(interaction_mod, "config_auto_approve", lambda reqs: False)
        with _BusLoopThread() as loop:
            monkeypatch.setattr(channel_mod, "_bus_loop", loop)
            monkeypatch.setattr(channel_mod, "_manager", None)
            bus = MessageBus()
            msg = ChannelMessage(
                msg_id="m1",
                content="",
                sender="u1",
                channel_type="fake",
                chat_id="chat1",
                bus_ref=bus,
                metadata={},
            )
            _feed_reply_when_ready(loop, "fake:chat1", "2")
            result = channel_mod.channel_hitl_prompt(
                [{"name": "execute", "args": {"command": "ls"}}], msg
            )
        assert result is None

    def test_unrecognized_reply_declines_without_refeed(self, monkeypatch):
        """CLI-bridge policy: unparseable reply → explicit notice, NO refeed.

        The reply is consumed by the interception registry, never enqueued
        as a new turn, and the user gets the unrecognized-reply notice. Only
        the serve-mode consumer refeeds; see
        TestConsumerUnrecognizedRefeed in tests/test_interaction_engine.py.
        """
        monkeypatch.setattr(interaction_mod, "config_auto_approve", lambda reqs: False)
        with _BusLoopThread() as loop:
            monkeypatch.setattr(channel_mod, "_bus_loop", loop)
            monkeypatch.setattr(channel_mod, "_manager", None)
            bus = MessageBus()
            msg = ChannelMessage(
                msg_id="m1",
                content="",
                sender="u1",
                channel_type="fake",
                chat_id="chat1",
                bus_ref=bus,
                metadata={},
            )
            _feed_reply_when_ready(loop, "fake:chat1", "do something else instead")
            result = channel_mod.channel_hitl_prompt(
                [{"name": "execute", "args": {"command": "ls"}}], msg
            )
            assert result is None

            # Outbound: prompt, then the exact old unrecognized notice.
            async def _drain():
                out = []
                while True:
                    try:
                        m = await asyncio.wait_for(bus.consume_outbound(), timeout=0.2)
                    except TimeoutError:
                        return out
                    out.append(m.content)

            contents = asyncio.run_coroutine_threadsafe(_drain(), loop).result(
                timeout=5
            )
        assert contents[-1] == interaction_mod.UNRECOGNIZED_FEEDBACK
        # No refeed: nothing was enqueued for the main thread.
        assert channel_mod._message_queue.empty()

    def test_approve_all_grants_channel_session(self, monkeypatch):
        monkeypatch.setattr(interaction_mod, "config_auto_approve", lambda reqs: False)
        with _BusLoopThread() as loop:
            monkeypatch.setattr(channel_mod, "_bus_loop", loop)
            monkeypatch.setattr(channel_mod, "_manager", None)
            bus = MessageBus()
            msg = ChannelMessage(
                msg_id="m1",
                content="",
                sender="u1",
                channel_type="fake",
                chat_id="chat1",
                bus_ref=bus,
                metadata={},
            )
            _feed_reply_when_ready(loop, "fake:chat1", "3")
            result = channel_mod.channel_hitl_prompt(
                [{"name": "execute", "args": {"command": "ls"}}], msg
            )
            assert result == [{"type": "approve"}]
            # "Approve all" grant persists: a second prompt auto-approves with
            # no reply fed at all.
            result2 = channel_mod.channel_hitl_prompt(
                [{"name": "execute", "args": {"command": "rm"}}], msg
            )
        assert result2 == [{"type": "approve"}]


class TestAskUserPromptBridge:
    def test_no_msg_cancels(self):
        assert channel_mod.channel_ask_user_prompt(
            {"questions": [{"question": "Q?", "type": "text"}]}, None
        ) == {"status": "cancelled"}

    def test_empty_questions_answered(self):
        assert channel_mod.channel_ask_user_prompt({"questions": []}, None) == {
            "answers": [],
            "status": "answered",
        }

    def test_text_answer_round_trip(self, monkeypatch):
        with _BusLoopThread() as loop:
            monkeypatch.setattr(channel_mod, "_bus_loop", loop)
            bus = MessageBus()
            msg = ChannelMessage(
                msg_id="m1",
                content="",
                sender="u1",
                channel_type="fake",
                chat_id="chat1",
                bus_ref=bus,
                metadata={},
            )
            _feed_reply_when_ready(loop, "fake:chat1", "CIFAR-10")
            result = channel_mod.channel_ask_user_prompt(
                {"questions": [{"question": "Which dataset?", "type": "text"}]}, msg
            )
        assert result == {"answers": ["CIFAR-10"], "status": "answered"}


class TestBridgeClosesUnscheduledCoroutine:
    def test_no_bus_loop_closes_coro(self, monkeypatch):
        """The bridge must close an engine coroutine it never scheduled —
        otherwise GC emits a "was never awaited" RuntimeWarning. (cr_frame
        is not a reliable observable for close() on unstarted coroutines.)"""
        import gc
        import warnings

        monkeypatch.setattr(channel_mod, "_bus_loop", None)

        async def _engine():
            return "never"

        coro = _engine()
        result = channel_mod._run_engine_on_bus(
            coro, result_timeout=1.0, on_error=lambda: "fallback"
        )
        assert result == "fallback"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            del coro
            gc.collect()
        assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
