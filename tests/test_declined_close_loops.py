"""A turn that ends at a declined or unanswered approval closes the parked checkpoint (Rich loop, consumer)."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from EvoScientist.channels.interaction import ApprovalOutcome
from EvoScientist.paths import Workspace
from EvoScientist.stream import display as display_mod
from tests.fakes.common import (
    FakeCheckpointAgent,
    FakeGraphGateway,
    attended_config,
    close_result,
    interrupt_event,
)

BEFORE_TOOLS = ("HumanInTheLoopMiddleware.after_model",)
# What the close wrote for the one call the user was asked about.
REJECTED = ("asked", "rejected", True)
UNANSWERED = ("asked", "unanswered", True)
ROUND_LIMIT = ("asked", "round_limit", True)


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
            yield interrupt_event(1)
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


def _closed(agent: FakeCheckpointAgent) -> list[tuple[str, str, bool]]:
    """What the close wrote, per call, as facts (``close_result``)."""
    return [close_result(text) for text in _written(agent)]


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
        attended_config(monkeypatch)

    def test_a_reject_at_the_prompt_closes_as_rejected(self):
        agent = _parked_agent()
        gateway = FakeGraphGateway(stream=_one_approval_then_done(), checkpoint=agent)
        result = self._run(gateway, hitl_prompt_fn=lambda _requests: None)
        assert result == "partial"
        assert len(gateway.requests) == 1
        assert _closed(agent) == [REJECTED]
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
        assert _closed(agent) == [REJECTED]

    def test_an_approval_nobody_answered_closes_as_unanswered(self):
        """A channel timeout or a prompt that could not be sent."""
        agent = _parked_agent()
        gateway = FakeGraphGateway(stream=_one_approval_then_done(), checkpoint=agent)
        result = self._run(
            gateway,
            hitl_outcome_fn=lambda _reqs, _exhausted: ApprovalOutcome(prompted=True),
        )
        assert result == "partial"
        assert len(gateway.requests) == 1
        assert _closed(agent) == [UNANSWERED]

    def _stopped_run(self, agent, stream=None, **kwargs):
        """A turn on its own cancel scope; the callbacks may request the stop."""
        gateway = FakeGraphGateway(
            stream=stream or _one_approval_then_done(), checkpoint=agent
        )
        try:
            self._run(gateway, cancel_scope="scope-stop", **kwargs)
        finally:
            display_mod.discard_stream_cancel("scope-stop")
        return gateway

    def _assert_closed_as_rejected(self, agent, gateway):
        assert len(gateway.requests) == 1
        assert _closed(agent) == [REJECTED]
        assert agent.next == ()

    def test_a_stop_requested_at_the_prompt_closes_as_rejected(self):
        """A stop while an approval is open is a refusal, as Ctrl+C on it is."""
        agent = _parked_agent()

        def _stop_then_decline(_requests):
            display_mod.request_stream_cancel("scope-stop")
            return None

        gateway = self._stopped_run(agent, hitl_prompt_fn=_stop_then_decline)
        self._assert_closed_as_rejected(agent, gateway)

    @pytest.mark.parametrize(
        "outcome",
        [
            ApprovalOutcome(prompted=True, rejected=True),
            ApprovalOutcome(prompted=True),
            ApprovalOutcome(decisions=[{"type": "approve"}], prompted=True),
        ],
        ids=["a stop reply", "an unanswered prompt", "an approval that came too late"],
    )
    def test_a_stop_while_the_channel_is_asked_closes_as_rejected(self, outcome):
        """Whatever the bridge brings back, a stopped turn does not resume: the
        approval it leaves is closed, not parked."""
        agent = _parked_agent()

        def _stopped(_requests, _exhausted):
            display_mod.request_stream_cancel("scope-stop")
            return outcome

        gateway = self._stopped_run(agent, hitl_outcome_fn=_stopped)
        self._assert_closed_as_rejected(agent, gateway)

    def test_a_stop_that_cuts_the_terminal_prompt_closes_as_rejected(self, monkeypatch):
        agent = _parked_agent()

        def _cut(*_args, **_kwargs):
            display_mod.request_stream_cancel("scope-stop")
            raise display_mod._StreamPromptCancelled

        monkeypatch.setattr(display_mod, "_resolve_hitl_approval", _cut)
        gateway = self._stopped_run(agent)
        self._assert_closed_as_rejected(agent, gateway)

    def test_a_stop_before_the_approval_is_shown_closes_as_rejected(self):
        agent = _parked_agent()

        async def _stream(_request):
            yield interrupt_event(1)
            display_mod.request_stream_cancel("scope-stop")

        def _never_asked(_requests):
            raise AssertionError("a stopped turn must not prompt")

        gateway = self._stopped_run(agent, _stream, hitl_prompt_fn=_never_asked)
        self._assert_closed_as_rejected(agent, gateway)

    def test_a_stop_at_an_open_question_closes_nothing(self):
        """Only an approval is refused by a stop; a question keeps its pause."""
        question = {"name": "ask_user", "args": {}, "id": "q-1", "type": "tool_call"}
        agent = FakeCheckpointAgent(
            next=("tools",),
            values={"messages": [AIMessage(content="", tool_calls=[question])]},
        )

        async def _stream(_request):
            yield {"type": "ask_user", "interrupt_id": "q-1", "questions": ["?"]}
            display_mod.request_stream_cancel("scope-stop")

        self._stopped_run(agent, _stream)
        assert agent.updates == []

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

    def test_no_terminal_to_read_from_is_not_a_rejection(self, monkeypatch):
        def _eof(_question):
            raise EOFError

        printed: list[str] = []
        monkeypatch.setattr(
            display_mod.console, "print", lambda *a, **_k: printed.extend(map(str, a))
        )
        resolution = self._resolve(question_runner=_eof)
        assert (resolution.decisions, resolution.rejected) == (None, False)
        assert resolution.human_prompted is True
        assert any("Left unanswered." in line for line in printed)
        assert not any("Rejected." in line for line in printed)

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
        from tests.fakes.common import StubChannel

        bus = MessageBus()
        manager = ChannelManager(bus, media_dir=media_dir)
        manager.register(StubChannel())
        gateway = FakeGraphGateway(stream=stream, checkpoint=agent)
        consumer = InboundConsumer(
            bus=bus,
            manager=manager,
            agent=agent,
            thread_id="",
            workspace=Workspace("/tmp/channel-project"),
            graph_gateway=gateway,
            max_concurrent=2,
            max_pending=10,
            inference_timeout=5.0,
            drain_timeout=10.0,
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

    @staticmethod
    async def _stop(consumer, task):
        await consumer.stop()
        # The run loop only checks for a stop once a second.
        task.cancel()
        await task

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
            # A worker must hold the reply before stop() ends the run loop.
            async with asyncio.timeout(10.0):
                while consumer.metrics["total_processed"] < 2:
                    await asyncio.sleep(0.01)
        # Each worker finishes its turn before it takes stop()'s sentinel.
        await self._stop(consumer, task)
        while bus.outbound_size:
            sent.append((await bus.consume_outbound()).content)
        return sent, gateway

    async def test_a_reject_reply_closes_as_rejected(self, tmp_path):
        agent = _parked_agent()
        sent, gateway = await self._decline(tmp_path, agent, "2")
        assert sent[-1] == "❌ Rejected"
        assert len(gateway.requests) == 1
        assert _closed(agent) == [REJECTED]
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
        await self._stop(consumer, task)
        assert sent == ["partial", "Approval round limit reached; stopping this turn."]
        assert len(gateway.requests) == 1
        assert _closed(agent) == [ROUND_LIMIT]

    async def test_a_timeout_closes_as_unanswered(self, tmp_path, monkeypatch):
        import EvoScientist.channels.consumer as consumer_mod

        monkeypatch.setattr(consumer_mod, "HITL_APPROVAL_TIMEOUT", 0.05)
        agent = _parked_agent()
        _sent, gateway = await self._decline(tmp_path, agent, None)
        assert len(gateway.requests) == 1
        assert _closed(agent) == [UNANSWERED]

    async def test_a_stop_reply_closes_as_rejected(self, tmp_path):
        agent = _parked_agent()
        _sent, gateway = await self._decline(tmp_path, agent, "/stop")
        assert len(gateway.requests) == 1
        assert _closed(agent) == [REJECTED]

    async def test_an_unrecognized_reply_closes_before_it_starts_the_next_turn(
        self, tmp_path
    ):
        agent = _parked_agent()
        closed_at_call: list[bool] = []

        async def _fake_stream(_request):
            closed_at_call.append(bool(_written(agent)))
            if len(closed_at_call) == 1:
                yield interrupt_event(1)
                return
            yield {"type": "text", "content": "finished"}
            yield {"type": "done", "content": "finished"}

        sent, gateway = await self._decline(
            tmp_path, agent, "do something else instead", stream=_fake_stream
        )
        assert sent[-1] == "finished"
        assert len(gateway.requests) == 2
        assert closed_at_call == [False, True]
        assert _closed(agent) == [REJECTED]

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
