"""Tests for HITL resume-round budgeting (issue #469).

The per-turn HITL resume loops must count only human-prompted rounds
against their budget. A session "approve all" grant or a config
auto-approving allow-list resumes guarded tool calls with no human
involved, so a long unattended turn must run to completion instead of
halting mid-work after 50 rounds. When the budget IS spent, the loop must
stop visibly — a message for the user plus a WARNING — refuse the NEXT
pending BEFORE prompting for it (the 50th decision is always resumed),
and close the parked checkpoint without resuming the agent. A rejecting
resume would run another model step; the close writes tool results and
clears ``next`` instead, so the checkpoint cannot poison the next turn.
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
        closes the parked interrupt without another model step (issue #469)."""
        from langchain_core.messages import AIMessage
        from langgraph.types import Command  # type: ignore[import-untyped]

        from EvoScientist.backends import HITL_ROUND_LIMIT_REJECT_MESSAGE

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
            yield {"type": "text", "content": "should not stream a drain"}
            yield {"type": "done", "content": "should not stream a drain"}

        gateway = FakeGraphGateway(
            stream=_fake_stream,
            state_values={
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "execute",
                                "args": {"command": "echo step"},
                                "id": "call-1",
                                "type": "tool_call",
                            }
                        ],
                    )
                ]
            },
        )
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
        assert stream_calls == 51  # initial + 50 approve resumes, no drain stream
        assert state.pending_interrupt is None  # parked interrupt was closed
        assert any("Approval round limit reached" in p for p in printed)
        assert any("max rounds" in r.getMessage() for r in caplog.records)

        resumes = [r for r in gateway.requests if isinstance(r.message, Command)]
        assert len(resumes) == 50
        assert "i51" not in resumes[-1].message.resume
        tool_update = next(
            update for update in gateway.updated_states if update[3] == "tools"
        )
        content = tool_update[2]["messages"][0].content
        assert HITL_ROUND_LIMIT_REJECT_MESSAGE in content
        assert "Do not retry this tool call" in content
        assert gateway.updated_states[-1][2] is None
        assert gateway.updated_states[-1][3] == "__end__"

    def test_ask_user_rounds_share_the_human_budget(self, monkeypatch):
        """ask_user rounds are always human-prompted and count toward the
        same budget; exhaustion closes the parked ask_user checkpoint
        without a cancelled resume (issue #469)."""

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

        assert stream_calls == 51  # 50 ask_user rounds + parked 51st, no drain
        assert state.pending_ask_user is None
        assert gateway.updated_states[-1][2] is None
        assert gateway.updated_states[-1][3] == "__end__"
        assert result == ""

    def test_total_cap_closes_without_streaming_another_step(self, monkeypatch, caplog):
        """The runaway guard stops an auto-approved turn without a rejecting
        resume, which would otherwise run another model step (issue #469)."""
        from langgraph.types import Command  # type: ignore[import-untyped]

        import EvoScientist.channels.hitl_budget as budget_mod

        monkeypatch.setattr(budget_mod, "MAX_HITL_TOTAL_ROUNDS", 2)
        monkeypatch.setattr(display_mod, "_session_auto_approve", True)
        stream_calls = 0

        async def _fake_stream(_request):
            nonlocal stream_calls
            stream_calls += 1
            yield _interrupt_event(stream_calls)

        gateway = FakeGraphGateway(stream=_fake_stream)
        state = display_mod.StreamState()
        printed: list[str] = []
        monkeypatch.setattr(
            display_mod.console,
            "print",
            lambda *args, **_kwargs: printed.append(str(args[0]) if args else ""),
        )

        with caplog.at_level(logging.WARNING, logger="EvoScientist.stream.display"):
            display_mod._run_streaming(
                agent=MagicMock(),
                message="hello",
                thread_id="t1",
                show_thinking=False,
                interactive=True,
                gateway=gateway,
                _state=state,
            )

        resumes = [r for r in gateway.requests if isinstance(r.message, Command)]
        assert stream_calls == 2
        assert len(resumes) == 1  # the second interrupt is closed, not resumed
        assert gateway.updated_states[-1][3] == "__end__"
        assert any("Approval round limit reached" in line for line in printed)

    def test_session_grant_after_human_budget_still_auto_resolves(self, monkeypatch):
        """The human budget is checked only when a prompt would be shown."""
        import EvoScientist.channels.hitl_budget as budget_mod

        monkeypatch.setattr(budget_mod, "MAX_HUMAN_HITL_ROUNDS", 1)
        monkeypatch.setattr(display_mod, "_session_auto_approve", False)
        monkeypatch.setattr(
            "EvoScientist.EvoScientist._ensure_config",
            lambda: SimpleNamespace(
                auto_approve=False, dangerous_mode=False, shell_allow_list=""
            ),
        )
        prompts = 0

        def _grant_on_first(_requests):
            nonlocal prompts
            prompts += 1
            display_mod._session_auto_approve = True
            return [{"type": "approve"}]

        stream_calls = 0

        async def _fake_stream(_request):
            nonlocal stream_calls
            stream_calls += 1
            if stream_calls <= 3:
                yield _interrupt_event(stream_calls)
                return
            yield {"type": "text", "content": "kept going"}
            yield {"type": "done", "content": "kept going"}

        gateway = FakeGraphGateway(stream=_fake_stream)
        result = display_mod._run_streaming(
            agent=MagicMock(),
            message="hello",
            thread_id="t1",
            show_thinking=False,
            interactive=True,
            hitl_prompt_fn=_grant_on_first,
            gateway=gateway,
            _state=display_mod.StreamState(),
        )

        assert prompts == 1
        assert result == "kept going"
        assert stream_calls == 4

    def test_close_failure_keeps_the_partial_response(self, monkeypatch, caplog):
        import EvoScientist.channels.hitl_budget as budget_mod

        monkeypatch.setattr(budget_mod, "MAX_HUMAN_HITL_ROUNDS", 0)
        monkeypatch.setattr(display_mod, "_session_auto_approve", False)
        monkeypatch.setattr(
            "EvoScientist.EvoScientist._ensure_config",
            lambda: SimpleNamespace(
                auto_approve=False, dangerous_mode=False, shell_allow_list=""
            ),
        )

        async def _fake_stream(_request):
            yield {"type": "text", "content": "partial"}
            yield _interrupt_event(1)

        gateway = FakeGraphGateway(stream=_fake_stream)
        gateway.update_error = RuntimeError("checkpoint store down")
        monkeypatch.setattr(display_mod.console, "print", lambda *_a, **_k: None)

        with caplog.at_level(logging.WARNING, logger="EvoScientist.stream.display"):
            result = display_mod._run_streaming(
                agent=MagicMock(),
                message="hello",
                thread_id="t1",
                show_thinking=False,
                interactive=True,
                hitl_prompt_fn=lambda _requests: [{"type": "approve"}],
                gateway=gateway,
                _state=display_mod.StreamState(),
            )

        assert result == "partial"
        assert len(gateway.requests) == 1
        assert any(
            "Failed to close parked HITL" in r.getMessage() for r in caplog.records
        )


def test_hitl_budget_stop_ignores_human_cap_for_auto_rounds():
    import EvoScientist.channels.hitl_budget as budget_mod

    assert not budget_mod.hitl_budget_stop(
        human_rounds=50, total_rounds=51, needs_human=False
    )
    assert budget_mod.hitl_budget_stop(
        human_rounds=50, total_rounds=51, needs_human=True
    )
    assert budget_mod.hitl_budget_stop(
        human_rounds=0, total_rounds=1000, needs_human=False
    )


class TestConsumerHitlRoundBudget:
    """Channel consumer ``_stream_with_hitl`` resume-loop budgeting."""

    def _consumer(self, stream) -> tuple[InboundConsumer, MessageBus, FakeGraphGateway]:
        bus = MessageBus()
        mgr = ChannelManager(bus)
        mgr.register(_StubChannel())
        gateway = FakeGraphGateway(stream=stream)
        return (
            InboundConsumer(
                bus=bus,
                manager=mgr,
                agent=MagicMock(),
                thread_id="",
                graph_gateway=gateway,
                max_concurrent=2,
                max_pending=10,
                inference_timeout=5.0,
                drain_timeout=1.0,
            ),
            bus,
            gateway,
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

        consumer, bus, _gateway = self._consumer(_fake_stream)
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

    async def test_session_grant_runs_long_unattended_turns_past_200_rounds(self):
        """The runaway guard is 1000 total rounds (CLI parity), not 200: an
        unattended session-grant turn with 205 guarded calls must run to
        completion — no stop, no truncation at the old cap (issue #469)."""
        stream_calls = 0

        async def _fake_stream(_request):
            nonlocal stream_calls
            stream_calls += 1
            if stream_calls <= 205:
                yield _interrupt_event(stream_calls)
                return
            yield {"type": "text", "content": "all done"}
            yield {"type": "done", "content": "all done"}

        consumer, bus, _gateway = self._consumer(_fake_stream)
        consumer._approval_policy.grant_session("stub:c1")

        await bus.publish_inbound(
            BusInbound(channel="stub", sender_id="u1", chat_id="c1", content="go")
        )
        task = asyncio.create_task(consumer.run())
        outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=30.0)

        assert outbound.content == "all done"
        assert stream_calls == 206  # initial stream + 205 auto-resumed rounds

        await consumer.stop()
        await task

    async def test_human_round_budget_exhausted_notifies_channel(
        self, monkeypatch, caplog
    ):
        """50 human decisions are all resumed (the 50th approval IS sent);
        the 51st pending is refused BEFORE prompting, the parked checkpoint
        is closed without a rejecting resume, and the user sees the stop
        message (issue #469)."""
        from langgraph.types import Command  # type: ignore[import-untyped]

        from EvoScientist.channels.interaction import ApprovalOutcome

        stream_calls = 0
        prompt_calls = 0

        async def _fake_stream(_request):
            nonlocal stream_calls
            stream_calls += 1
            yield _interrupt_event(stream_calls)

        async def _human_approves(
            _action_reqs,
            _io,
            _policy,
            _session_key,
            *,
            timeout=0,
            human_budget_exhausted=False,
        ):
            nonlocal prompt_calls
            if human_budget_exhausted:
                return ApprovalOutcome(budget_exhausted=True)
            prompt_calls += 1
            return ApprovalOutcome(decisions=[{"type": "approve"}], prompted=True)

        consumer, bus, gateway = self._consumer(_fake_stream)
        monkeypatch.setattr(consumer_mod, "resolve_approval", _human_approves)

        await bus.publish_inbound(
            BusInbound(channel="stub", sender_id="u1", chat_id="c1", content="go")
        )
        with caplog.at_level(logging.WARNING, logger="EvoScientist.channels.consumer"):
            task = asyncio.create_task(consumer.run())
            outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=10.0)

        assert prompt_calls == 50  # the 51st pending never prompts a human
        assert stream_calls == 51  # initial + 50 approve resumes, no drain stream
        assert outbound.content == "Approval round limit reached; stopping this turn."
        assert any("HITL round limit" in r.getMessage() for r in caplog.records)

        fiftieth = gateway.requests[-1]
        assert isinstance(fiftieth.message, Command)
        assert fiftieth.message.resume["i50"]["decisions"] == [{"type": "approve"}]
        assert gateway.updated_states[-1][2] is None
        assert gateway.updated_states[-1][3] == "__end__"

        await consumer.stop()
        await task

    async def test_ask_user_rounds_share_budget_and_drain_cancelled(
        self, monkeypatch, caplog
    ):
        """ask_user rounds count toward the same human budget; the 51st
        question is never asked and its checkpoint is closed without a
        cancelled resume (issue #469)."""

        stream_calls = 0
        asked = 0

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

        async def _fake_ask_user(_questions, _io, timeout=0):
            nonlocal asked
            asked += 1
            return {"answers": ["yes"], "status": "answered"}

        consumer, bus, gateway = self._consumer(_fake_stream)
        monkeypatch.setattr(consumer_mod, "resolve_ask_user", _fake_ask_user)

        await bus.publish_inbound(
            BusInbound(channel="stub", sender_id="u1", chat_id="c1", content="go")
        )
        with caplog.at_level(logging.WARNING, logger="EvoScientist.channels.consumer"):
            task = asyncio.create_task(consumer.run())
            outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=10.0)

        assert asked == 50  # the 51st question is never asked
        assert stream_calls == 51  # initial + 50 answer resumes, no drain stream
        assert outbound.content == "Approval round limit reached; stopping this turn."
        assert any("HITL round limit" in r.getMessage() for r in caplog.records)
        assert gateway.updated_states[-1][3] == "__end__"

        await consumer.stop()
        await task

    async def test_human_budget_does_not_block_a_following_session_grant(
        self, monkeypatch
    ):
        """Approving one-by-one and then granting the session must not stop
        the next auto-resolved pending (issue #469 review)."""
        import EvoScientist.channels.hitl_budget as budget_mod
        from EvoScientist.channels.interaction import ApprovalOutcome

        monkeypatch.setattr(budget_mod, "MAX_HUMAN_HITL_ROUNDS", 1)
        stream_calls = 0

        async def _fake_stream(_request):
            nonlocal stream_calls
            stream_calls += 1
            if stream_calls <= 3:
                yield _interrupt_event(stream_calls)
                return
            yield {"type": "text", "content": "continued"}
            yield {"type": "done", "content": "continued"}

        prompts = 0

        async def _human_then_grant(
            _action_reqs,
            _io,
            policy,
            session_key,
            *,
            timeout=0,
            human_budget_exhausted=False,
        ):
            nonlocal prompts
            if policy.is_session_granted(session_key):
                return ApprovalOutcome(decisions=[{"type": "approve"}])
            if human_budget_exhausted:
                return ApprovalOutcome(budget_exhausted=True)
            prompts += 1
            policy.grant_session(session_key)
            return ApprovalOutcome(decisions=[{"type": "approve"}], prompted=True)

        consumer, bus, _gateway = self._consumer(_fake_stream)
        monkeypatch.setattr(consumer_mod, "resolve_approval", _human_then_grant)

        await bus.publish_inbound(
            BusInbound(channel="stub", sender_id="u1", chat_id="c1", content="go")
        )
        task = asyncio.create_task(consumer.run())
        outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=10.0)

        assert prompts == 1
        assert outbound.content == "continued"
        assert stream_calls == 4

        await consumer.stop()
        await task

    async def test_close_failure_still_sends_partial_and_stop(
        self, monkeypatch, caplog
    ):
        import EvoScientist.channels.hitl_budget as budget_mod

        monkeypatch.setattr(budget_mod, "MAX_HUMAN_HITL_ROUNDS", 0)

        async def _fake_stream(_request):
            yield {"type": "text", "content": "partial answer"}
            yield _interrupt_event(1)

        consumer, bus, gateway = self._consumer(_fake_stream)
        gateway.update_error = RuntimeError("checkpoint store down")

        await bus.publish_inbound(
            BusInbound(channel="stub", sender_id="u1", chat_id="c1", content="go")
        )
        with caplog.at_level(logging.WARNING, logger="EvoScientist.channels.consumer"):
            task = asyncio.create_task(consumer.run())
            first = await asyncio.wait_for(bus.consume_outbound(), timeout=10.0)
            second = await asyncio.wait_for(bus.consume_outbound(), timeout=10.0)

        assert first.content == "partial answer"
        assert second.content == "Approval round limit reached; stopping this turn."
        assert any(
            "Failed to close parked HITL" in r.getMessage() for r in caplog.records
        )

        await consumer.stop()
        await task

    async def test_total_cap_closes_without_another_resume(self, monkeypatch):
        """The 1000-round runaway guard stops an auto-approved consumer turn
        and closes the checkpoint. No further resume is streamed."""
        from langgraph.types import Command

        import EvoScientist.channels.hitl_budget as budget_mod

        monkeypatch.setattr(budget_mod, "MAX_HITL_TOTAL_ROUNDS", 2)
        stream_calls = 0

        async def _fake_stream(_request):
            nonlocal stream_calls
            stream_calls += 1
            yield _interrupt_event(stream_calls)

        consumer, bus, gateway = self._consumer(_fake_stream)
        consumer._approval_policy.grant_session("stub:c1")

        await bus.publish_inbound(
            BusInbound(channel="stub", sender_id="u1", chat_id="c1", content="go")
        )
        task = asyncio.create_task(consumer.run())
        outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=10.0)

        resumes = [r for r in gateway.requests if isinstance(r.message, Command)]
        assert stream_calls == 2
        assert len(resumes) == 1
        assert gateway.updated_states[-1][3] == "__end__"
        assert outbound.content == "Approval round limit reached; stopping this turn."

        await consumer.stop()
        await task


def test_tui_loop_does_not_build_a_resume_past_the_total_cap(monkeypatch):
    """TUI HITL loop, without a Pilot harness.

    ``tui_hitl_loop_stop`` is what each TUI branch calls before it prompts or
    builds a resume. Auto rounds ignore the human cap; at the total cap the
    branch stops with no resume left unsent.
    """
    import EvoScientist.channels.hitl_budget as budget_mod
    from EvoScientist.cli.tui_interactive import tui_hitl_loop_stop

    monkeypatch.setattr(budget_mod, "MAX_HITL_TOTAL_ROUNDS", 2)
    resumes_built = 0
    unsent_resume = None
    for total_rounds in range(1, 6):
        # The previous iteration's resume, if any, is what this round streams.
        unsent_resume = None
        if tui_hitl_loop_stop(
            human_rounds=0, total_rounds=total_rounds, needs_human=False
        ):
            break
        unsent_resume = {"type": "approve", "round": total_rounds}
        resumes_built += 1

    assert resumes_built == 1
    assert unsent_resume is None


def test_tui_loop_human_cap_does_not_stop_an_auto_branch():
    """A session-grant branch keeps resuming after the human cap; a widget
    branch does not."""
    from EvoScientist.cli.tui_interactive import tui_hitl_loop_stop

    assert not tui_hitl_loop_stop(human_rounds=50, total_rounds=51, needs_human=False)
    assert tui_hitl_loop_stop(human_rounds=50, total_rounds=51, needs_human=True)
