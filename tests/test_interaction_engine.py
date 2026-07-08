"""Tests for the interaction engine coroutines + reply registry.

The engine (:func:`resolve_ask_user`, :func:`resolve_approval`) is pure
async with an injected :class:`InteractionIO`, so it is exercised here with
a scripted ``FakeIO``: assert the prompts it emits and feed it the replies
a user would send.  Covers the whole grammar — single/multi question,
optional, choice letters, "Other", timeout, ``/stop``, approve / reject /
approve-all, and capability-driven button formatting.
"""

import asyncio

import pytest

from EvoScientist.channels import interaction as I
from EvoScientist.channels.capabilities import ChannelCapabilities

# ═══════════════════════════════════════════════════════════════════════
# Scripted fake IO
# ═══════════════════════════════════════════════════════════════════════


class FakeIO(I.InteractionIO):
    """A scripted :class:`InteractionIO`.

    *replies* is the queue of reply strings ``wait_reply`` hands back in
    order; a ``None`` entry (or exhausting the queue) simulates a timeout.
    Every ``send`` is recorded as ``(content, metadata)`` in ``sent``.
    """

    def __init__(self, replies=None, *, capabilities=None, base_metadata=None):
        self.capabilities = capabilities or ChannelCapabilities()
        self.base_metadata = base_metadata
        self._replies = list(replies or [])
        self.sent: list[tuple[str, dict | None]] = []
        self.send_ok = True

    async def send(self, content, *, metadata=None):
        self.sent.append((content, metadata))
        return self.send_ok

    async def wait_reply(self, *, timeout):
        if not self._replies:
            return None
        return self._replies.pop(0)

    @property
    def contents(self):
        return [c for c, _ in self.sent]


QQ_CAPS = ChannelCapabilities(inline_buttons=True)


# ═══════════════════════════════════════════════════════════════════════
# resolve_ask_user
# ═══════════════════════════════════════════════════════════════════════


class TestResolveAskUser:
    async def test_empty_questions(self):
        io = FakeIO()
        result = await I.resolve_ask_user([], io)
        assert result == {"answers": [], "status": "answered"}
        assert io.sent == []

    async def test_single_text_answered(self):
        io = FakeIO(["CIFAR-10"])
        result = await I.resolve_ask_user(
            [{"question": "Which dataset?", "type": "text"}], io
        )
        assert result == {"answers": ["CIFAR-10"], "status": "answered"}
        assert "Quick check-in" in io.contents[0]

    async def test_multi_question_answered(self):
        io = FakeIO(["ans1", "ans2"])
        result = await I.resolve_ask_user(
            [
                {"question": "Q1?", "type": "text"},
                {"question": "Q2?", "type": "text"},
            ],
            io,
        )
        assert result == {"answers": ["ans1", "ans2"], "status": "answered"}
        assert io.contents[0].startswith("❓ Question 1/2")
        assert io.contents[1].startswith("❓ Question 2/2")

    async def test_choice_letter(self):
        io = FakeIO(["B"])
        q = {
            "question": "Which?",
            "type": "multiple_choice",
            "choices": [{"value": "CIFAR-10"}, {"value": "ImageNet"}],
        }
        result = await I.resolve_ask_user([q], io)
        assert result == {"answers": ["ImageNet"], "status": "answered"}

    async def test_choice_other_subflow(self):
        # "C" is the Other letter for two choices; then a free-form answer.
        io = FakeIO(["C", "my custom dataset"])
        q = {
            "question": "Which?",
            "type": "multiple_choice",
            "choices": [{"value": "CIFAR-10"}, {"value": "ImageNet"}],
        }
        result = await I.resolve_ask_user([q], io)
        assert result == {"answers": ["my custom dataset"], "status": "answered"}
        assert io.contents[1] == I.OTHER_PROMPT

    async def test_optional_suffix_in_prompt(self):
        io = FakeIO(["ans"])
        await I.resolve_ask_user(
            [{"question": "Notes?", "type": "text", "required": False}], io
        )
        assert "(optional)" in io.contents[0]
        assert "Leave empty to skip." in io.contents[0]

    async def test_optional_empty_reply_skips_and_continues(self):
        io = FakeIO(["", "next"])
        result = await I.resolve_ask_user(
            [
                {"question": "Notes?", "type": "text", "required": False},
                {"question": "Next?", "type": "text"},
            ],
            io,
        )
        assert result == {"answers": ["", "next"], "status": "answered"}
        assert io.contents[1].startswith("❓ Question 2/2")
        assert I.ASK_USER_TIMEOUT_FEEDBACK not in io.contents

    async def test_required_empty_reply_cancels_without_timeout_notice(self):
        io = FakeIO([""])
        result = await I.resolve_ask_user(
            [{"question": "Required?", "type": "text"}],
            io,
        )
        assert result == {"status": "cancelled"}
        assert I.ASK_USER_TIMEOUT_FEEDBACK not in io.contents

    async def test_timeout_first_question(self):
        io = FakeIO([])  # no replies -> timeout
        result = await I.resolve_ask_user([{"question": "Q?", "type": "text"}], io)
        assert result == {"status": "cancelled"}
        assert io.contents[-1] == I.ASK_USER_TIMEOUT_FEEDBACK

    async def test_timeout_in_other_subflow(self):
        io = FakeIO(["C"])  # picks Other, then times out on free-form
        q = {
            "question": "Which?",
            "type": "multiple_choice",
            "choices": [{"value": "A"}, {"value": "B"}],
        }
        result = await I.resolve_ask_user([q], io)
        assert result == {"status": "cancelled"}
        assert io.contents[-1] == I.ASK_USER_TIMEOUT_FEEDBACK

    async def test_stop_command_cancels(self):
        io = FakeIO(["/stop"])
        result = await I.resolve_ask_user([{"question": "Q?", "type": "text"}], io)
        assert result == {"status": "cancelled"}
        # /stop is a pure cancel — no timeout notice sent.
        assert I.ASK_USER_TIMEOUT_FEEDBACK not in io.contents

    async def test_stop_command_in_other_subflow(self):
        io = FakeIO(["C", "/stop"])
        q = {
            "question": "Which?",
            "type": "multiple_choice",
            "choices": [{"value": "A"}, {"value": "B"}],
        }
        result = await I.resolve_ask_user([q], io)
        assert result == {"status": "cancelled"}

    async def test_cancel_reply(self):
        io = FakeIO(["cancel"])
        result = await I.resolve_ask_user([{"question": "Q?", "type": "text"}], io)
        assert result == {"status": "cancelled"}

    async def test_send_failure_cancels(self):
        io = FakeIO(["ans"])
        io.send_ok = False
        result = await I.resolve_ask_user([{"question": "Q?", "type": "text"}], io)
        assert result == {"status": "cancelled"}


# ═══════════════════════════════════════════════════════════════════════
# resolve_approval
# ═══════════════════════════════════════════════════════════════════════


REQS = [{"name": "execute", "args": {"command": "rm -rf /tmp/x"}}]


class TestResolveApproval:
    async def test_session_granted_short_circuits(self, monkeypatch):
        monkeypatch.setattr(I, "config_auto_approve", lambda reqs: False)
        p = I.ApprovalPolicy()
        p.grant_session("tg:c1")
        io = FakeIO()
        result = await I.resolve_approval(REQS, io, p, "tg:c1")
        assert result == I.ApprovalOutcome(decisions=[{"type": "approve"}])
        assert io.sent == []  # no prompt, silent

    async def test_config_auto_approve_short_circuits(self, monkeypatch):
        monkeypatch.setattr(I, "config_auto_approve", lambda reqs: True)
        io = FakeIO()
        result = await I.resolve_approval(REQS, io, I.ApprovalPolicy(), "tg:c1")
        assert result == I.ApprovalOutcome(decisions=[{"type": "approve"}])
        assert io.sent == []

    async def test_approve(self, monkeypatch):
        monkeypatch.setattr(I, "config_auto_approve", lambda reqs: False)
        io = FakeIO(["1"])
        result = await I.resolve_approval(REQS, io, I.ApprovalPolicy(), "tg:c1")
        assert result == I.ApprovalOutcome(decisions=[{"type": "approve"}])
        assert io.contents[0].startswith("⚠️ Approval Required")
        assert io.contents[-1] == I.APPROVED_FEEDBACK

    async def test_reject(self, monkeypatch):
        monkeypatch.setattr(I, "config_auto_approve", lambda reqs: False)
        io = FakeIO(["2"])
        result = await I.resolve_approval(REQS, io, I.ApprovalPolicy(), "tg:c1")
        assert result == I.ApprovalOutcome()
        assert io.contents[-1] == I.REJECTED_FEEDBACK

    async def test_approve_all_grants_session(self, monkeypatch):
        monkeypatch.setattr(I, "config_auto_approve", lambda reqs: False)
        io = FakeIO(["3"])
        p = I.ApprovalPolicy()
        result = await I.resolve_approval(REQS, io, p, "tg:c1")
        assert result == I.ApprovalOutcome(decisions=[{"type": "approve"}])
        assert io.contents[-1] == I.APPROVED_AUTO_FEEDBACK
        assert p.is_session_granted("tg:c1")  # future prompts auto-approve

    async def test_multi_request_approve_length(self, monkeypatch):
        monkeypatch.setattr(I, "config_auto_approve", lambda reqs: False)
        reqs = [
            {"name": "execute", "args": {"command": "a"}},
            {"name": "execute", "args": {"command": "b"}},
        ]
        io = FakeIO(["1"])
        result = await I.resolve_approval(reqs, io, I.ApprovalPolicy(), "tg:c1")
        assert result.decisions == [{"type": "approve"}, {"type": "approve"}]

    async def test_unrecognized_reply_reported_not_judged(self, monkeypatch):
        # The engine declines but hands the raw text back — the *driver*
        # decides the feedback / refeed policy (consumer refeeds as a new
        # turn; CLI bridge sends the unrecognized notice).
        monkeypatch.setattr(I, "config_auto_approve", lambda reqs: False)
        io = FakeIO(["huh?"])
        result = await I.resolve_approval(REQS, io, I.ApprovalPolicy(), "tg:c1")
        assert result.decisions is None
        assert result.unrecognized_reply == "huh?"
        # No feedback sent by the engine itself on the unrecognized path.
        assert I.UNRECOGNIZED_FEEDBACK not in io.contents
        assert I.REJECTED_FEEDBACK not in io.contents

    async def test_timeout(self, monkeypatch):
        monkeypatch.setattr(I, "config_auto_approve", lambda reqs: False)
        io = FakeIO([])  # times out
        result = await I.resolve_approval(REQS, io, I.ApprovalPolicy(), "tg:c1")
        assert result == I.ApprovalOutcome()
        assert io.contents[-1] == I.APPROVAL_TIMEOUT_FEEDBACK

    async def test_stop_command_silent_cancel(self, monkeypatch):
        monkeypatch.setattr(I, "config_auto_approve", lambda reqs: False)
        io = FakeIO(["/stop"])
        result = await I.resolve_approval(REQS, io, I.ApprovalPolicy(), "tg:c1")
        assert result == I.ApprovalOutcome()
        # /stop already got its own ack; no reject/unrecognized feedback here.
        assert I.REJECTED_FEEDBACK not in io.contents
        assert I.UNRECOGNIZED_FEEDBACK not in io.contents

    async def test_send_failure_declines(self, monkeypatch):
        monkeypatch.setattr(I, "config_auto_approve", lambda reqs: False)
        io = FakeIO(["1"])
        io.send_ok = False
        result = await I.resolve_approval(REQS, io, I.ApprovalPolicy(), "tg:c1")
        assert result == I.ApprovalOutcome()

    # ── R3: button-capability formatting + payload normalization ──

    async def test_buttons_attached_when_capable(self, monkeypatch):
        monkeypatch.setattr(I, "config_auto_approve", lambda reqs: False)
        io = FakeIO(["1"], capabilities=QQ_CAPS, base_metadata={"chat": "x"})
        await I.resolve_approval(REQS, io, I.ApprovalPolicy(), "tg:c1")
        prompt, metadata = io.sent[0]
        # Button channels drop the textual "Reply: 1=..." cue.
        assert "Reply: 1=Approve" not in prompt
        assert metadata["buttons"] == [
            {"text": "Approve", "value": "1", "type": "primary"},
            {"text": "Reject", "value": "2", "type": "danger"},
            {"text": "Approve all", "value": "3"},
        ]
        assert metadata["chat"] == "x"  # base metadata preserved

    async def test_no_buttons_when_incapable(self, monkeypatch):
        monkeypatch.setattr(I, "config_auto_approve", lambda reqs: False)
        io = FakeIO(["1"])  # default caps: no inline_buttons
        await I.resolve_approval(REQS, io, I.ApprovalPolicy(), "tg:c1")
        prompt, metadata = io.sent[0]
        assert "Reply: 1=Approve, 2=Reject, 3=Approve all" in prompt
        assert "buttons" not in (metadata or {})

    async def test_button_press_payload_normalizes(self, monkeypatch):
        # A button click delivers its `value` ("3") through the same reply
        # path; the engine must treat it exactly like a typed "3".
        monkeypatch.setattr(I, "config_auto_approve", lambda reqs: False)
        io = FakeIO(["3"], capabilities=QQ_CAPS)
        p = I.ApprovalPolicy()
        result = await I.resolve_approval(REQS, io, p, "tg:c1")
        assert result == I.ApprovalOutcome(decisions=[{"type": "approve"}])
        assert p.is_session_granted("tg:c1")


# ═══════════════════════════════════════════════════════════════════════
# PendingReplyRegistry
# ═══════════════════════════════════════════════════════════════════════


class TestPendingReplyRegistry:
    async def test_register_wait_resolve(self):
        reg = I.PendingReplyRegistry()

        async def _resolver():
            # Give wait() a tick to register before resolving.
            await asyncio.sleep(0.01)
            assert reg.try_resolve("s1", "hello") is True

        got, _ = await asyncio.gather(reg.wait("s1", timeout=1.0), _resolver())
        assert got == "hello"
        assert "s1" not in reg  # cleaned up after wait

    async def test_wait_event_returns_reply_context(self):
        reg = I.PendingReplyRegistry()
        context = object()

        async def _resolver():
            await asyncio.sleep(0.01)
            assert reg.try_resolve("s1", "hello", context=context) is True

        got, _ = await asyncio.gather(reg.wait_event("s1", timeout=1.0), _resolver())
        assert got is not None
        assert got.content == "hello"
        assert got.context is context
        assert "s1" not in reg

    async def test_wait_timeout_returns_none(self):
        reg = I.PendingReplyRegistry()
        got = await reg.wait("s1", timeout=0.02)
        assert got is None
        assert "s1" not in reg

    def test_try_resolve_no_pending(self):
        reg = I.PendingReplyRegistry()
        assert reg.try_resolve("nope", "x") is False

    async def test_reregister_cancels_stale(self):
        reg = I.PendingReplyRegistry()
        first = asyncio.ensure_future(reg.wait("s1", timeout=1.0))
        await asyncio.sleep(0.01)  # let first register
        # A second interaction on the same chat re-registers, cancelling the
        # first waiter so it unwinds promptly (returns None) instead of
        # hanging until its own timeout.
        new_fut = reg.register("s1")
        assert await first is None
        # The stale waiter's cleanup must not evict the newer registration.
        assert reg._pending.get("s1") is new_fut
        reg.discard("s1")

    async def test_clear_cancels_all(self):
        reg = I.PendingReplyRegistry()
        fut = asyncio.ensure_future(reg.wait("s1", timeout=1.0))
        await asyncio.sleep(0.01)
        reg.clear()
        got = await fut
        assert got is None

    async def test_task_cancellation_propagates(self):
        reg = I.PendingReplyRegistry()
        task = asyncio.create_task(reg.wait("s1", timeout=1.0))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert "s1" not in reg


# ═══════════════════════════════════════════════════════════════════════
# Consumer driver: unrecognized-reply refeed (serve-mode policy)
# ═══════════════════════════════════════════════════════════════════════
# Pre-engine semantics that must survive the extraction: an unrecognized
# reply while a HITL approval is pending REJECTS the pending action, sends
# the rejection feedback, and the user's text is then processed as a NEW
# agent turn — a user who ignores the prompt and types a fresh instruction
# must not lose it. (The CLI bridge deliberately does NOT refeed; see
# tests/test_cli_channel_bridge.py)


class TestConsumerUnrecognizedRefeed:
    async def test_unrecognized_reply_rejects_and_refeeds(self, monkeypatch):
        from unittest.mock import MagicMock

        from EvoScientist.channels.bus.events import (
            InboundMessage as BusInbound,
        )
        from EvoScientist.channels.bus.message_bus import MessageBus
        from EvoScientist.channels.channel_manager import ChannelManager
        from EvoScientist.channels.consumer import InboundConsumer
        from tests.fakes import FakeGraphGateway, StubChannel

        # Force the manual-prompt path (no config auto-approve).
        monkeypatch.setattr(I, "config_auto_approve", lambda reqs: False)

        bus = MessageBus()
        mgr = ChannelManager(bus)
        mgr.register(StubChannel())

        stream_calls = 0

        async def _fake_stream(request):
            nonlocal stream_calls
            stream_calls += 1
            if stream_calls == 1:
                # First turn hits a HITL interrupt.
                yield {
                    "type": "interrupt",
                    "interrupt_id": "main",
                    "action_requests": [
                        {"name": "execute", "args": {"command": "rm -rf /x"}}
                    ],
                    "review_configs": [],
                }
                return
            # The refeed turn: echo what we were given.
            yield {"type": "text", "content": f"handled: {request.message}"}
            yield {"type": "done", "content": f"handled: {request.message}"}

        gateway = FakeGraphGateway(
            stream=_fake_stream,
            generated_thread_ids=["thread-original", "thread-reply"],
        )
        consumer = InboundConsumer(
            bus=bus,
            manager=mgr,
            agent=MagicMock(),
            thread_id="",
            graph_gateway=gateway,
            max_concurrent=2,
            max_pending=10,
            inference_timeout=5.0,
            drain_timeout=1.0,
        )

        task = asyncio.create_task(consumer.run())
        try:
            await bus.publish_inbound(
                BusInbound(
                    channel="stub",
                    sender_id="u1",
                    chat_id="c1",
                    content="do the thing",
                    message_id="msg-original",
                    metadata={"origin": "original"},
                )
            )

            # 1. Approval prompt goes out.
            prompt = await asyncio.wait_for(bus.consume_outbound(), timeout=5.0)
            assert prompt.content.startswith("⚠️ Approval Required")
            assert prompt.metadata == {"origin": "original"}

            # 2. User ignores the prompt and types a fresh instruction.
            await bus.publish_inbound(
                BusInbound(
                    channel="stub",
                    sender_id="u2",
                    chat_id="c1",
                    content="actually, summarize the report",
                    message_id="msg-reply",
                    media=["file-report.pdf"],
                    metadata={"origin": "reply"},
                )
            )

            # 3. Pending action is rejected with the old serve feedback...
            feedback = await asyncio.wait_for(bus.consume_outbound(), timeout=5.0)
            assert feedback.content == I.REJECTED_FEEDBACK
            assert feedback.metadata == {"origin": "original"}

            # 4. ...and the text is processed as a NEW agent turn.
            response = await asyncio.wait_for(bus.consume_outbound(), timeout=5.0)
            assert response.content == "handled: actually, summarize the report"
            assert response.reply_to == "msg-reply"
            assert response.metadata == {"origin": "reply"}

            # The refeed reached the stream path as its own request.
            assert stream_calls == 2
            assert gateway.requests[0].thread_id == "thread-original"
            assert gateway.requests[-1].message == "actually, summarize the report"
            assert gateway.requests[-1].thread_id == "thread-reply"
            assert gateway.requests[-1].media == ["file-report.pdf"]
        finally:
            await consumer.stop()
            await task


class TestApprovalEmptyReply:
    async def test_empty_reply_is_unrecognized_not_timeout(self, monkeypatch):
        """A media-only/empty reply must reach the refeed path, not timeout."""
        monkeypatch.setattr(I, "config_auto_approve", lambda reqs: False)
        io = FakeIO(replies=[""])
        policy = I.ApprovalPolicy()
        outcome = await I.resolve_approval(
            [{"name": "execute", "args": {"command": "ls"}}],
            io,
            policy,
            "stub:c1",
            timeout=1.0,
        )
        assert outcome.decisions is None
        assert outcome.unrecognized_reply == ""
        sent = [content for content, _ in io.sent]
        assert I.APPROVAL_TIMEOUT_FEEDBACK not in sent


class TestReplyInterceptionSkipsThreadCreation:
    async def test_consumed_reply_creates_no_thread(self):
        """A registry-consumed reply must not create a graph thread or touch
        the sender-session LRU."""
        from unittest.mock import AsyncMock, MagicMock

        from EvoScientist.channels.bus.events import InboundMessage as BusInbound
        from EvoScientist.channels.consumer import InboundConsumer

        gateway = MagicMock()
        gateway.create_thread = AsyncMock(return_value="t-should-not-exist")
        consumer = InboundConsumer(
            bus=MagicMock(),
            manager=MagicMock(),
            agent=MagicMock(),
            thread_id="",
            graph_gateway=gateway,
            max_concurrent=1,
            max_pending=5,
            inference_timeout=1.0,
            drain_timeout=0.5,
        )
        msg = BusInbound(
            channel="stub",
            sender_id="u1",
            chat_id="c1",
            content="1",
            message_id="m1",
        )
        fut = consumer._reply_registry.register(msg.session_key)

        await consumer._handle_message(msg)

        assert fut.done()
        assert fut.result().content == "1"
        gateway.create_thread.assert_not_awaited()
        assert consumer._sessions == {}
