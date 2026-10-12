"""A turn that ends at a declined or unanswered approval closes the parked checkpoint (Rich loop, consumer)."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from EvoScientist.channels.interaction import ApprovalOutcome
from EvoScientist.stream import display as display_mod
from tests.fakes import FakeCheckpointAgent, FakeGraphGateway
from tests.test_hitl_loop import _attended_config, _interrupt_event

BEFORE_TOOLS = ("HumanInTheLoopMiddleware.after_model",)
REJECTED = "User rejected the tool call for `execute` with id call-1. The tool was not executed."
UNANSWERED = "The approval request for the tool call `execute` with id call-1 was left unanswered."


_REQUESTS = {"action_requests": [{"name": "execute", "args": {"command": "ls"}}]}


def _parked_agent(**kwargs) -> FakeCheckpointAgent:
    call = {
        "name": "execute",
        "args": {"command": "echo 1"},
        "id": "call-1",
        "type": "tool_call",
    }
    return FakeCheckpointAgent(
        next=BEFORE_TOOLS,
        values={"messages": [AIMessage(content="", tool_calls=[call])]},
        **kwargs,
    )


def _one_approval_then_done():
    calls = 0

    async def _fake_stream(_request):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield {"type": "text", "content": "partial"}
            yield _interrupt_event(1)
            return
        yield {"type": "text", "content": "must not run"}
        yield {"type": "done", "content": "must not run"}

    return _fake_stream


def _written(agent: FakeCheckpointAgent) -> list[str]:
    return [
        message.content
        for values, _node in agent.updates
        if isinstance(values, dict)
        for message in values["messages"]
    ]


class TestRichLoopClosesAtADecline:
    @staticmethod
    def _run(gateway, **kwargs):
        return display_mod._run_streaming(
            agent=MagicMock(),
            message="hello",
            thread_id="t1",
            show_thinking=False,
            interactive=True,
            gateway=gateway,
            _state=display_mod.StreamState(),
            **kwargs,
        )

    @pytest.fixture(autouse=True)
    def _attended(self, monkeypatch):
        monkeypatch.setattr(display_mod, "_session_auto_approve", False)
        monkeypatch.setattr(display_mod.console, "print", lambda *_a, **_k: None)
        _attended_config(monkeypatch)

    def test_a_reject_at_the_prompt_closes_as_rejected(self):
        agent = _parked_agent()
        gateway = FakeGraphGateway(stream=_one_approval_then_done(), checkpoint=agent)
        result = self._run(gateway, hitl_prompt_fn=lambda _requests: None)
        assert result == "partial"
        assert len(gateway.requests) == 1
        assert [text.startswith(REJECTED) for text in _written(agent)] == [True]
        assert agent.next == ()

    def test_a_reject_on_the_channel_closes_as_rejected_without_a_limit_notice(self):
        from EvoScientist.channels.hitl_budget import HITL_BUDGET_STOP_NOTICE

        agent = _parked_agent()
        gateway = FakeGraphGateway(stream=_one_approval_then_done(), checkpoint=agent)
        result = self._run(
            gateway,
            hitl_outcome_fn=lambda _reqs, _exhausted: ApprovalOutcome(
                prompted=True, rejected=True
            ),
        )
        assert result == "partial"
        assert HITL_BUDGET_STOP_NOTICE not in result
        assert [text.startswith(REJECTED) for text in _written(agent)] == [True]

    def test_an_approval_nobody_answered_closes_as_unanswered(self):
        """A channel timeout, a stop reply or a prompt that could not be sent."""
        agent = _parked_agent()
        gateway = FakeGraphGateway(stream=_one_approval_then_done(), checkpoint=agent)
        result = self._run(
            gateway,
            hitl_outcome_fn=lambda _reqs, _exhausted: ApprovalOutcome(prompted=True),
        )
        assert result == "partial"
        assert len(gateway.requests) == 1
        assert [text.startswith(UNANSWERED) for text in _written(agent)] == [True]

    def test_a_stop_requested_at_the_prompt_closes_nothing(self):
        agent = _parked_agent()
        gateway = FakeGraphGateway(stream=_one_approval_then_done(), checkpoint=agent)

        def _stop_then_decline(_requests):
            display_mod.request_stream_cancel("scope-stop")
            return None

        try:
            self._run(
                gateway, hitl_prompt_fn=_stop_then_decline, cancel_scope="scope-stop"
            )
        finally:
            display_mod.discard_stream_cancel("scope-stop")
        assert agent.updates == []
        assert agent.next == BEFORE_TOOLS

    def test_a_close_that_fails_ends_the_turn_with_one_short_log_line(self, caplog):
        agent = _parked_agent(
            update_error=RuntimeError("409 conflict\nrun kwargs:\n" + "x" * 5000)
        )
        gateway = FakeGraphGateway(stream=_one_approval_then_done(), checkpoint=agent)
        with caplog.at_level(logging.DEBUG, logger="EvoScientist"):
            result = self._run(gateway, hitl_prompt_fn=lambda _requests: None)
        assert result == "partial"
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1
        assert "Could not close the parked checkpoint" in warnings[0].getMessage()
        assert len(warnings[0].getMessage()) < 400
        assert "\n" not in warnings[0].getMessage()
        assert warnings[0].exc_info is None


class TestRichResolverSaysWhoDeclined:
    """``rejected`` separates a refusal from an approval nobody answered."""

    @pytest.fixture(autouse=True)
    def _attended(self, monkeypatch):
        import EvoScientist.channels.interaction as interaction_mod

        monkeypatch.setattr(display_mod, "_session_auto_approve", False)
        monkeypatch.setattr(display_mod.console, "print", lambda *_a, **_k: None)
        monkeypatch.setattr(
            interaction_mod, "config_policy_snapshot", lambda reqs: (None, {})
        )
        # The runner answers; building the real prompt needs a console.
        monkeypatch.setattr("questionary.select", lambda *_a, **_k: object())

    def _resolve(self, **kwargs):
        return display_mod._resolve_hitl_approval(_REQUESTS, **kwargs)

    def test_picking_reject_is_a_rejection(self):
        resolution = self._resolve(question_runner=lambda _question: "Reject")
        assert (resolution.decisions, resolution.rejected) == (None, True)

    def test_ctrl_c_at_the_prompt_is_a_rejection(self):
        resolution = self._resolve(question_runner=lambda _question: None)
        assert (resolution.decisions, resolution.rejected) == (None, True)

    def test_ctrl_c_raised_by_the_prompt_is_a_rejection(self):
        def _interrupted(_question):
            raise KeyboardInterrupt

        resolution = self._resolve(question_runner=_interrupted)
        assert (resolution.decisions, resolution.rejected) == (None, True)

    def test_no_terminal_to_read_from_is_not_a_rejection(self):
        def _eof(_question):
            raise EOFError

        resolution = self._resolve(question_runner=_eof)
        assert (resolution.decisions, resolution.rejected) == (None, False)
        assert resolution.human_prompted is True

    def test_a_custom_prompt_that_returns_nothing_is_a_rejection(self):
        resolution = self._resolve(prompt_fn=lambda _requests: None)
        assert (resolution.decisions, resolution.rejected) == (None, True)

    def test_an_approval_is_not_a_rejection(self):
        resolution = self._resolve(
            prompt_fn=lambda reqs: [{"type": "approve"} for _ in reqs]
        )
        assert resolution.rejected is False

    @pytest.mark.parametrize("rejected", [True, False])
    def test_the_bridge_reports_it(self, rejected):
        resolution = self._resolve(
            outcome_fn=lambda _reqs, _exhausted: ApprovalOutcome(
                prompted=True, rejected=rejected
            )
        )
        assert (resolution.decisions, resolution.rejected) == (None, rejected)


class TestConsumerClosesAtADecline:
    """The standalone consumer: reject, timeout, stop and an unrecognized reply."""

    @staticmethod
    def _consumer(stream, media_dir, agent):
        from EvoScientist.channels.bus.message_bus import MessageBus
        from EvoScientist.channels.channel_manager import ChannelManager
        from EvoScientist.channels.consumer import InboundConsumer
        from tests.fakes import StubChannel

        bus = MessageBus()
        manager = ChannelManager(bus, media_dir=media_dir)
        manager.register(StubChannel())
        gateway = FakeGraphGateway(stream=stream, checkpoint=agent)
        consumer = InboundConsumer(
            bus=bus,
            manager=manager,
            agent=agent,
            thread_id="",
            graph_gateway=gateway,
            max_concurrent=2,
            max_pending=10,
            inference_timeout=5.0,
            drain_timeout=1.0,
        )
        return consumer, bus, gateway

    @staticmethod
    async def _send(bus, content="go"):
        from EvoScientist.channels.bus.events import InboundMessage

        await bus.publish_inbound(
            InboundMessage(
                channel="stub", sender_id="u1", chat_id="c1", content=content
            )
        )

    @pytest.fixture(autouse=True)
    def _no_config_rule(self, monkeypatch):
        import EvoScientist.channels.interaction as interaction_mod

        monkeypatch.setattr(
            interaction_mod, "config_policy_snapshot", lambda reqs: (None, {})
        )

    async def _decline(self, tmp_path, agent, reply, *, stream=None):
        """Park one approval, send ``reply`` (``None`` = say nothing), return what was sent back."""
        import asyncio

        consumer, bus, gateway = self._consumer(
            stream or _one_approval_then_done(), tmp_path, agent
        )
        await self._send(bus)
        task = asyncio.create_task(consumer.run())
        sent = [(await asyncio.wait_for(bus.consume_outbound(), timeout=10.0)).content]
        assert "Approval Required" in sent[0]
        if reply is not None:
            await self._send(bus, content=reply)
        while True:
            try:
                sent.append(
                    (
                        await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
                    ).content
                )
            except TimeoutError:
                break
        await consumer.stop()
        await task
        return sent, gateway

    async def test_a_reject_reply_closes_as_rejected(self, tmp_path):
        agent = _parked_agent()
        sent, gateway = await self._decline(tmp_path, agent, "2")
        assert sent[-1] == "❌ Rejected"
        assert len(gateway.requests) == 1
        assert [text.startswith(REJECTED) for text in _written(agent)] == [True]
        assert agent.next == ()

    async def test_the_round_limit_exit_records_the_limit(self, tmp_path, monkeypatch):
        import asyncio

        import EvoScientist.channels.consumer as consumer_mod

        async def _budget_spent(*_args, **_kwargs):
            return ApprovalOutcome(budget_exhausted=True)

        monkeypatch.setattr(consumer_mod, "resolve_approval", _budget_spent)
        agent = _parked_agent()
        consumer, bus, gateway = self._consumer(
            _one_approval_then_done(), tmp_path, agent
        )
        await self._send(bus)
        task = asyncio.create_task(consumer.run())
        sent = [
            (await asyncio.wait_for(bus.consume_outbound(), timeout=10.0)).content
            for _ in range(2)
        ]
        await consumer.stop()
        await task
        assert sent == ["partial", "Approval round limit reached; stopping this turn."]
        assert len(gateway.requests) == 1
        (text,) = _written(agent)
        assert "(approval round limit reached)" in text

    async def test_a_timeout_closes_as_unanswered(self, tmp_path, monkeypatch):
        import EvoScientist.channels.consumer as consumer_mod

        monkeypatch.setattr(consumer_mod, "HITL_APPROVAL_TIMEOUT", 0.05)
        agent = _parked_agent()
        _sent, gateway = await self._decline(tmp_path, agent, None)
        assert len(gateway.requests) == 1
        assert [text.startswith(UNANSWERED) for text in _written(agent)] == [True]

    async def test_a_stop_reply_closes_as_unanswered(self, tmp_path):
        agent = _parked_agent()
        _sent, gateway = await self._decline(tmp_path, agent, "/stop")
        assert len(gateway.requests) == 1
        assert [text.startswith(UNANSWERED) for text in _written(agent)] == [True]

    async def test_an_unrecognized_reply_closes_before_it_starts_the_next_turn(
        self, tmp_path
    ):
        agent = _parked_agent()
        closed_at_call: list[bool] = []

        async def _fake_stream(_request):
            closed_at_call.append(bool(_written(agent)))
            if len(closed_at_call) == 1:
                yield _interrupt_event(1)
                return
            yield {"type": "text", "content": "finished"}
            yield {"type": "done", "content": "finished"}

        sent, gateway = await self._decline(
            tmp_path, agent, "do something else instead", stream=_fake_stream
        )
        assert sent[-1] == "finished"
        assert len(gateway.requests) == 2
        assert closed_at_call == [False, True]
        assert [text.startswith(REJECTED) for text in _written(agent)] == [True]

    async def test_a_close_that_fails_ends_the_turn_with_one_short_log_line(
        self, tmp_path, caplog
    ):
        agent = _parked_agent(
            update_error=RuntimeError("409 conflict\nrun kwargs:\n" + "x" * 5000)
        )
        with caplog.at_level(logging.DEBUG, logger="EvoScientist"):
            sent, gateway = await self._decline(tmp_path, agent, "2")
        assert sent[-1] == "❌ Rejected"
        assert len(gateway.requests) == 1
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1
        assert "Could not close the parked checkpoint" in warnings[0].getMessage()
        assert "\n" not in warnings[0].getMessage()
        assert warnings[0].exc_info is None
