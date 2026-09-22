"""Tests for HITL resume-round budgeting (issue #469).

The per-turn HITL resume loops must count only human-prompted rounds
against their budget. A session "approve all" grant or a config
auto-approving allow-list resumes guarded tool calls with no human
involved, so a long unattended turn must run to completion instead of
halting mid-work after 50 rounds. When the budget IS spent, the loop must
stop visibly — a message for the user plus a WARNING — and, on the Rich
CLI, close the parked interrupt with a rejecting resume so the checkpoint
does not poison the next turn (the #464 recovery preserves genuine
interrupt pauses on purpose).
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import EvoScientist.channels.consumer as consumer_mod
from EvoScientist.channels.bus.events import InboundMessage as BusInbound
from EvoScientist.channels.bus.message_bus import MessageBus
from EvoScientist.channels.channel_manager import ChannelManager
from EvoScientist.channels.consumer import InboundConsumer
from EvoScientist.stream import display as display_mod
from tests.fakes import FakeGraphGateway
from tests.fakes import StubChannel as _StubChannel


def _interrupt_event(n: int) -> dict:
    return {
        "type": "interrupt",
        "interrupt_id": f"i{n}",
        "action_requests": [{"name": "execute", "args": {"command": f"echo step-{n}"}}],
    }


class TestRichCliHitlRoundBudget:
    """Rich CLI ``_run_streaming`` resume-loop budgeting."""

    def test_session_auto_approve_resumes_past_50_rounds(self, monkeypatch, caplog):
        """An unattended "approve all (session)" turn with 60 guarded tool
        calls must run to completion — auto-resolved rounds do not count
        against the human-prompted budget (issue #469)."""
        from langgraph.types import Command  # type: ignore[import-untyped]

        monkeypatch.setattr(display_mod, "_session_auto_approve", True)
        stream_calls = 0

        async def _fake_stream(_request):
            nonlocal stream_calls
            stream_calls += 1
            if stream_calls <= 60:
                yield _interrupt_event(stream_calls)
                return
            yield {"type": "text", "content": "All sixty rounds done"}
            yield {"type": "done", "content": "All sixty rounds done"}

        gateway = FakeGraphGateway(stream=_fake_stream)
        state = display_mod.StreamState()

        with caplog.at_level(logging.WARNING, logger="EvoScientist.stream.display"):
            result = display_mod._run_streaming(
                agent=MagicMock(),
                message="hello",
                thread_id="t1",
                show_thinking=False,
                interactive=True,
                gateway=gateway,
                _state=state,
            )

        resumes = [r for r in gateway.requests if isinstance(r.message, Command)]
        assert stream_calls == 61  # initial stream + 60 auto-resumed rounds
        assert len(resumes) == 60
        assert result == "All sixty rounds done"
        assert state.pending_interrupt is None  # nothing left parked
        assert not any("max rounds" in r.getMessage() for r in caplog.records)

    def test_human_round_budget_exhaustion_stops_visibly_and_drains(
        self, monkeypatch, caplog
    ):
        """51 human-prompted interrupts: 50 prompts are served, then the
        loop stops with a visible notice, keeps the partial response, and
        closes the parked interrupt with a rejecting resume (issue #469)."""
        from langgraph.types import Command  # type: ignore[import-untyped]

        monkeypatch.setattr(display_mod, "_session_auto_approve", False)
        monkeypatch.setattr(
            "EvoScientist.EvoScientist._ensure_config",
            lambda: SimpleNamespace(
                auto_approve=False, dangerous_mode=False, shell_allow_list=""
            ),
        )
        printed: list[str] = []

        def _record_print(*args, **_kwargs):
            printed.append(str(args[0]) if args else "")

        monkeypatch.setattr(display_mod.console, "print", _record_print)

        prompt_calls = 0

        def _human_approves(_requests):
            nonlocal prompt_calls
            prompt_calls += 1
            return [{"type": "approve"}]

        stream_calls = 0

        async def _fake_stream(_request):
            nonlocal stream_calls
            stream_calls += 1
            if stream_calls == 1:
                yield {"type": "text", "content": "Partial work so far"}
            if stream_calls <= 51:
                yield _interrupt_event(stream_calls)
                return
            # Drain round after exhaustion: its output is discarded.
            yield {"type": "text", "content": "post-rejection wrap-up"}
            yield {"type": "done", "content": "post-rejection wrap-up"}

        gateway = FakeGraphGateway(stream=_fake_stream)
        state = display_mod.StreamState()

        with caplog.at_level(logging.WARNING, logger="EvoScientist.stream.display"):
            result = display_mod._run_streaming(
                agent=MagicMock(),
                message="hello",
                thread_id="t1",
                show_thinking=False,
                interactive=True,
                hitl_prompt_fn=_human_approves,
                gateway=gateway,
                _state=state,
            )

        assert prompt_calls == 50  # the 51st round hits the budget, not a prompt
        assert result == "Partial work so far"  # partial-response semantics kept
        assert stream_calls == 52  # + 50 approve resumes + 1 reject drain
        assert state.pending_interrupt is None  # parked interrupt was closed
        assert any("Approval round limit reached" in p for p in printed)
        assert any("max rounds" in r.getMessage() for r in caplog.records)

        drain = gateway.requests[-1]
        assert isinstance(drain.message, Command)
        decisions = drain.message.resume["i51"]["decisions"]
        assert decisions[0]["type"] == "reject"

    def test_ask_user_rounds_share_the_human_budget(self, monkeypatch):
        """ask_user rounds are always human-prompted and count toward the
        same budget; exhaustion closes the parked ask_user checkpoint with
        a cancelled resume instead of leaving it abandoned (issue #469)."""
        from langgraph.types import Command  # type: ignore[import-untyped]

        stream_calls = 0

        async def _fake_stream(_request):
            nonlocal stream_calls
            stream_calls += 1
            if stream_calls <= 51:
                yield {
                    "type": "ask_user",
                    "interrupt_id": f"ask-{stream_calls}",
                    "questions": [{"question": "Continue?"}],
                }
                return
            yield {"type": "text", "content": "never reached"}
            yield {"type": "done", "content": "never reached"}

        gateway = FakeGraphGateway(stream=_fake_stream)
        state = display_mod.StreamState()

        def _ask_user_fn(_pending):
            return {"answers": ["yes"], "status": "answered"}

        result = display_mod._run_streaming(
            agent=MagicMock(),
            message="hello",
            thread_id="t1",
            show_thinking=False,
            interactive=True,
            ask_user_prompt_fn=_ask_user_fn,
            gateway=gateway,
            _state=state,
        )

        assert stream_calls == 52  # 50 ask_user rounds + parked 51st + drain
        assert state.pending_ask_user is None
        drain = gateway.requests[-1]
        assert isinstance(drain.message, Command)
        assert drain.message.resume == {"status": "cancelled"}
        assert result == ""


class TestConsumerHitlRoundBudget:
    """Channel consumer ``_stream_with_hitl`` resume-loop budgeting."""

    def _consumer(self, stream) -> tuple[InboundConsumer, MessageBus]:
        bus = MessageBus()
        mgr = ChannelManager(bus)
        mgr.register(_StubChannel())
        return (
            InboundConsumer(
                bus=bus,
                manager=mgr,
                agent=MagicMock(),
                thread_id="",
                graph_gateway=FakeGraphGateway(stream=stream),
                max_concurrent=2,
                max_pending=10,
                inference_timeout=5.0,
                drain_timeout=1.0,
            ),
            bus,
        )

    async def test_session_grant_resumes_past_50_rounds(self):
        """An auto-approving session grant drives 60 guarded calls to
        completion instead of falling out of the loop at 50 (issue #469)."""
        stream_calls = 0

        async def _fake_stream(_request):
            nonlocal stream_calls
            stream_calls += 1
            if stream_calls <= 60:
                yield _interrupt_event(stream_calls)
                return
            yield {"type": "text", "content": "final answer"}
            yield {"type": "done", "content": "final answer"}

        consumer, bus = self._consumer(_fake_stream)
        consumer._approval_policy.grant_session("stub:c1")

        await bus.publish_inbound(
            BusInbound(channel="stub", sender_id="u1", chat_id="c1", content="go")
        )
        task = asyncio.create_task(consumer.run())
        outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=10.0)

        assert outbound.content == "final answer"
        assert stream_calls == 61  # initial stream + 60 auto-resumed rounds

        await consumer.stop()
        await task

    async def test_human_round_budget_exhausted_notifies_channel(
        self, monkeypatch, caplog
    ):
        """Rounds that prompted a human count toward the budget; when it is
        spent the turn ends with a visible message instead of silently
        falling out of the loop (issue #469)."""
        from EvoScientist.channels.interaction import ApprovalOutcome

        stream_calls = 0

        async def _fake_stream(_request):
            nonlocal stream_calls
            stream_calls += 1
            yield _interrupt_event(stream_calls)

        async def _human_approves(
            _action_reqs, _io, _policy, _session_key, *, timeout=0
        ):
            return ApprovalOutcome(decisions=[{"type": "approve"}], prompted=True)

        consumer, bus = self._consumer(_fake_stream)
        monkeypatch.setattr(consumer_mod, "resolve_approval", _human_approves)

        await bus.publish_inbound(
            BusInbound(channel="stub", sender_id="u1", chat_id="c1", content="go")
        )
        with caplog.at_level(logging.WARNING, logger="EvoScientist.channels.consumer"):
            task = asyncio.create_task(consumer.run())
            outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=10.0)

        assert stream_calls == 50  # 50 human rounds served, then the budget
        assert outbound.content == "Approval round limit reached; stopping this turn."
        assert any("HITL round limit" in r.getMessage() for r in caplog.records)

        await consumer.stop()
        await task
