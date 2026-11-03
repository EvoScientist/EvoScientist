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
from unittest.mock import AsyncMock, MagicMock

import pytest

import EvoScientist.channels.consumer as consumer_mod
from EvoScientist.channels.bus.events import InboundMessage as BusInbound
from EvoScientist.channels.bus.message_bus import MessageBus
from EvoScientist.channels.channel_manager import ChannelManager
from EvoScientist.channels.consumer import InboundConsumer
from EvoScientist.middleware import events as mw_events
from EvoScientist.paths import Workspace
from EvoScientist.stream import display as display_mod
from tests.fakes.common import FakeCheckpointAgent, FakeGraphGateway
from tests.fakes.common import StubChannel as _StubChannel
from tests.fakes.common import attended_config as _attended_config
from tests.fakes.common import interrupt_event as _interrupt_event


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

        dangling = AIMessage(
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
        # A main-agent approval parks before any tool of the batch runs.
        parked_at = "HumanInTheLoopMiddleware.after_model"
        agent = FakeCheckpointAgent(values={"messages": [dangling]}, next=(parked_at,))
        gateway = FakeGraphGateway(stream=_fake_stream, checkpoint=agent)
        state = display_mod.StreamState()

        with caplog.at_level(logging.WARNING, logger="EvoScientist.stream.display"):
            result = display_mod._run_streaming(
                agent=agent,
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
        assert [as_node for _values, as_node in agent.updates] == [
            "__end__",
            parked_at,
            "__end__",
        ]
        tool_update = next(update for update in agent.updates if update[1] == parked_at)
        content = tool_update[0]["messages"][0].content
        assert HITL_ROUND_LIMIT_REJECT_MESSAGE in content
        assert "Do not retry this tool call" in content
        assert agent.updates[-1][0] is None
        assert agent.updates[-1][1] == "__end__"

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

        agent = FakeCheckpointAgent()
        gateway = FakeGraphGateway(stream=_fake_stream, checkpoint=agent)
        state = display_mod.StreamState()

        def _ask_user_fn(_pending):
            return {"answers": ["yes"], "status": "answered"}

        result = display_mod._run_streaming(
            agent=agent,
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
        assert [as_node for _values, as_node in agent.updates] == ["__end__", "__end__"]
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

        agent = FakeCheckpointAgent()
        gateway = FakeGraphGateway(stream=_fake_stream, checkpoint=agent)
        state = display_mod.StreamState()
        printed: list[str] = []
        monkeypatch.setattr(
            display_mod.console,
            "print",
            lambda *args, **_kwargs: printed.append(str(args[0]) if args else ""),
        )

        with caplog.at_level(logging.WARNING, logger="EvoScientist.stream.display"):
            display_mod._run_streaming(
                agent=agent,
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
        assert agent.updates[-1][1] == "__end__"
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

        agent = FakeCheckpointAgent(update_error=RuntimeError("checkpoint store down"))
        gateway = FakeGraphGateway(stream=_fake_stream, checkpoint=agent)
        monkeypatch.setattr(display_mod.console, "print", lambda *_a, **_k: None)

        with caplog.at_level(logging.WARNING, logger="EvoScientist"):
            result = display_mod._run_streaming(
                agent=agent,
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
        failures = [
            r
            for r in caplog.records
            if "Could not close the parked checkpoint" in r.getMessage()
        ]
        assert len(failures) == 1
        assert failures[0].exc_info is None

    def test_channel_session_grant_resumes_past_50_rounds(self, monkeypatch):
        from langgraph.types import Command  # type: ignore[import-untyped]

        from EvoScientist.cli import channel as channel_mod

        msg = channel_mod.ChannelMessage(
            msg_id="m1", content="run", sender="u", channel_type="telegram"
        )
        policy = channel_mod.ApprovalPolicy()
        policy.grant_session(channel_mod._channel_message_session_key(msg))
        monkeypatch.setattr(channel_mod, "_approval_policy", policy)
        monkeypatch.setattr(display_mod, "_session_auto_approve", False)
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
        result = display_mod._run_streaming(
            agent=MagicMock(),
            message="hello",
            thread_id="t1",
            show_thinking=False,
            interactive=True,
            gateway=gateway,
            _state=display_mod.StreamState(),
            hitl_outcome_fn=lambda reqs, exhausted: channel_mod.channel_hitl_prompt(
                reqs, msg, human_budget_exhausted=exhausted
            ),
        )

        resumes = [r for r in gateway.requests if isinstance(r.message, Command)]
        assert stream_calls == 61
        assert len(resumes) == 60
        assert result == "All sixty rounds done"

    def test_channel_budget_stop_reaches_the_channel_reply(self, monkeypatch):
        from EvoScientist.channels.hitl_budget import HITL_BUDGET_STOP_NOTICE
        from EvoScientist.channels.interaction import ApprovalOutcome

        monkeypatch.setattr(display_mod, "_session_auto_approve", False)
        monkeypatch.setattr(
            "EvoScientist.EvoScientist._ensure_config",
            lambda: SimpleNamespace(
                auto_approve=False, dangerous_mode=False, shell_allow_list=""
            ),
        )
        monkeypatch.setattr(display_mod.console, "print", lambda *a, **k: None)
        stream_calls = 0

        async def _fake_stream(_request):
            nonlocal stream_calls
            stream_calls += 1
            if stream_calls == 1:
                yield {"type": "text", "content": "Partial work so far"}
            yield _interrupt_event(stream_calls)

        prompted = 0

        def _bridge(_requests, exhausted):
            nonlocal prompted
            if exhausted:
                return ApprovalOutcome(budget_exhausted=True)
            prompted += 1
            return ApprovalOutcome(decisions=[{"type": "approve"}], prompted=True)

        gateway = FakeGraphGateway(
            stream=_fake_stream, checkpoint=FakeCheckpointAgent()
        )
        result = display_mod._run_streaming(
            agent=MagicMock(),
            message="hello",
            thread_id="t1",
            show_thinking=False,
            interactive=True,
            gateway=gateway,
            _state=display_mod.StreamState(),
            hitl_outcome_fn=_bridge,
        )

        assert prompted == 50
        assert stream_calls == 51
        assert result == f"Partial work so far\n\n{HITL_BUDGET_STOP_NOTICE}"


GRANT = {"hitl_approve_all": True}
ANSWER = {"answers": ["yes"], "status": "answered"}


def _interrupts_then_done(interrupts: int):
    """A stream that parks ``interrupts`` approvals, one per call, then answers."""
    calls = 0

    async def _fake_stream(_request):
        nonlocal calls
        calls += 1
        if calls <= interrupts:
            yield _interrupt_event(calls)
            return
        yield {"type": "text", "content": "finished"}
        yield {"type": "done", "content": "finished"}

    return _fake_stream


class TestRichCliSessionGrantReachesTheGraph:
    """Which calls of a turn carry ``hitl_approve_all`` (the graph then skips the pause).

    The fake stream keeps parking approvals whatever it is sent, so these pin
    the client's side only: the real graph stops interrupting once a call
    carries the key (``tests/test_hitl_approve_all_graph.py``).
    """

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

    @staticmethod
    def _extras(gateway):
        return [request.configurable_extra for request in gateway.requests]

    def test_a_turn_under_the_grant_is_keyed_from_its_first_call(self, monkeypatch):
        monkeypatch.setattr(display_mod, "_session_auto_approve", True)
        gateway = FakeGraphGateway(stream=_interrupts_then_done(2))
        assert self._run(gateway) == "finished"
        assert self._extras(gateway) == [GRANT, GRANT, GRANT]

    def test_the_granting_answer_is_sent_unkeyed(self, monkeypatch):
        """The human's answer must be applied as given, so the key rides from
        the call after it: the first auto-answered resume."""
        monkeypatch.setattr(display_mod, "_session_auto_approve", False)
        _attended_config(monkeypatch)

        def _grant(_requests):
            display_mod._session_auto_approve = True
            return [{"type": "approve"}]

        gateway = FakeGraphGateway(stream=_interrupts_then_done(3))
        assert self._run(gateway, hitl_prompt_fn=_grant) == "finished"
        assert self._extras(gateway) == [None, None, GRANT, GRANT]

    def test_an_allow_list_answer_is_not_keyed(self, monkeypatch):
        import EvoScientist.channels.interaction as interaction_mod

        monkeypatch.setattr(display_mod, "_session_auto_approve", False)
        monkeypatch.setattr(
            interaction_mod,
            "config_policy_snapshot",
            lambda reqs: ([{"type": "approve"} for _ in reqs], {}),
        )
        gateway = FakeGraphGateway(stream=_interrupts_then_done(2))
        assert self._run(gateway) == "finished"
        assert self._extras(gateway) == [None, None, None]

    @pytest.mark.parametrize("granted", [True, False])
    def test_a_bridge_answer_is_keyed_only_when_the_chat_is_granted(
        self, monkeypatch, granted
    ):
        from EvoScientist.channels.interaction import ApprovalOutcome

        monkeypatch.setattr(display_mod, "_session_auto_approve", False)
        _attended_config(monkeypatch)
        gateway = FakeGraphGateway(stream=_interrupts_then_done(1))
        self._run(
            gateway,
            hitl_outcome_fn=lambda reqs, _exhausted: ApprovalOutcome(
                decisions=[{"type": "approve"} for _ in reqs], session_granted=granted
            ),
        )
        assert self._extras(gateway) == [None, GRANT if granted else None]

    def test_a_turn_that_starts_granted_is_keyed_from_its_first_call(self, monkeypatch):
        """A channel-origin turn: the caller knows the chat's grant and says so
        with a flag; the loop leaves the caller's extras alone."""
        monkeypatch.setattr(display_mod, "_session_auto_approve", False)
        extras = {"active_teams": ["reviewer"]}
        gateway = FakeGraphGateway(stream=_interrupts_then_done(0))
        self._run(gateway, configurable_extra=extras, approve_all_at_start=True)
        assert self._extras(gateway) == [
            {"active_teams": ["reviewer"], "hitl_approve_all": True}
        ]
        assert extras == {"active_teams": ["reviewer"]}

    def test_a_turn_that_starts_ungranted_is_not_keyed(self, monkeypatch):
        monkeypatch.setattr(display_mod, "_session_auto_approve", False)
        gateway = FakeGraphGateway(stream=_interrupts_then_done(0))
        self._run(gateway, configurable_extra={"active_teams": ["reviewer"]})
        assert self._extras(gateway) == [{"active_teams": ["reviewer"]}]

    def test_a_starting_grant_is_dropped_from_a_call_that_must_not_carry_it(
        self, monkeypatch
    ):
        from EvoScientist.channels.interaction import ApprovalOutcome

        monkeypatch.setattr(display_mod, "_session_auto_approve", False)
        _attended_config(monkeypatch)
        extras = {"active_teams": ["reviewer"]}
        gateway = FakeGraphGateway(stream=_interrupts_then_done(1))
        self._run(
            gateway,
            configurable_extra=extras,
            approve_all_at_start=True,
            hitl_outcome_fn=lambda reqs, _exhausted: ApprovalOutcome(
                decisions=[{"type": "approve"} for _ in reqs], prompted=True
            ),
        )
        assert self._extras(gateway) == [
            {"active_teams": ["reviewer"], "hitl_approve_all": True},
            {"active_teams": ["reviewer"]},
        ]
        assert extras == {"active_teams": ["reviewer"]}

    @pytest.mark.parametrize("granted", [True, False])
    def test_an_ask_user_answer_is_keyed_only_under_a_grant(self, monkeypatch, granted):
        monkeypatch.setattr(display_mod, "_session_auto_approve", granted)
        calls = 0

        async def _fake_stream(_request):
            nonlocal calls
            calls += 1
            if calls == 1:
                yield {
                    "type": "ask_user",
                    "interrupt_id": "ask-1",
                    "questions": [{"question": "Continue?"}],
                }
                return
            yield {"type": "text", "content": "finished"}
            yield {"type": "done", "content": "finished"}

        gateway = FakeGraphGateway(stream=_fake_stream)
        self._run(
            gateway,
            ask_user_prompt_fn=lambda _p: {"answers": ["yes"], "status": "answered"},
        )
        expected = GRANT if granted else None
        assert self._extras(gateway) == [expected, expected]

    def test_a_grant_answer_with_a_non_approve_decision_is_not_keyed(self, monkeypatch):
        """Belt and braces: the key is sent only when every decision approves."""
        monkeypatch.setattr(display_mod, "_session_auto_approve", False)
        monkeypatch.setattr(
            display_mod,
            "_resolve_hitl_approval",
            lambda *_a, **_k: display_mod._HitlResolution(
                [{"type": "reject", "message": "no"}],
                human_prompted=False,
                session_granted=True,
            ),
        )
        gateway = FakeGraphGateway(stream=_interrupts_then_done(1))
        self._run(gateway)
        assert self._extras(gateway) == [None, None]

    @staticmethod
    def _approval_then_question():
        """One approval, then one question, then the answer."""
        calls = 0

        async def _fake_stream(_request):
            nonlocal calls
            calls += 1
            if calls == 1:
                yield _interrupt_event(1)
                return
            if calls == 2:
                yield {
                    "type": "ask_user",
                    "interrupt_id": "ask-1",
                    "questions": [{"question": "Continue?"}],
                }
                return
            yield {"type": "text", "content": "finished"}
            yield {"type": "done", "content": "finished"}

        return _fake_stream

    def test_a_question_after_the_granting_prompt_is_answered_keyed(self, monkeypatch):
        monkeypatch.setattr(display_mod, "_session_auto_approve", False)
        _attended_config(monkeypatch)

        def _grant(_requests):
            display_mod._session_auto_approve = True
            return [{"type": "approve"}]

        gateway = FakeGraphGateway(stream=self._approval_then_question())
        self._run(gateway, hitl_prompt_fn=_grant, ask_user_prompt_fn=lambda _p: ANSWER)
        assert self._extras(gateway) == [None, None, GRANT]

    def test_a_question_after_a_bridge_grant_answer_is_answered_keyed(
        self, monkeypatch
    ):
        """A chat grant seen during the turn holds for the rest of it."""
        from EvoScientist.channels.interaction import ApprovalOutcome

        monkeypatch.setattr(display_mod, "_session_auto_approve", False)
        _attended_config(monkeypatch)
        gateway = FakeGraphGateway(stream=self._approval_then_question())
        self._run(
            gateway,
            hitl_outcome_fn=lambda reqs, _exhausted: ApprovalOutcome(
                decisions=[{"type": "approve"} for _ in reqs],
                session_granted=True,
                grant_in_force=True,
            ),
            ask_user_prompt_fn=lambda _p: ANSWER,
        )
        assert self._extras(gateway) == [None, GRANT, GRANT]

    def test_a_grant_given_on_the_channel_keys_the_next_question_answer(
        self, monkeypatch
    ):
        """The reply that grants is a human answer (unkeyed), but the chat's
        grant is in force afterwards: the answer to a question that comes next
        carries it, so no approval pauses after the question."""
        from EvoScientist.channels.interaction import ApprovalOutcome

        monkeypatch.setattr(display_mod, "_session_auto_approve", False)
        _attended_config(monkeypatch)
        gateway = FakeGraphGateway(stream=self._approval_then_question())
        self._run(
            gateway,
            hitl_outcome_fn=lambda reqs, _exhausted: ApprovalOutcome(
                decisions=[{"type": "approve"} for _ in reqs],
                prompted=True,
                grant_in_force=True,
            ),
            ask_user_prompt_fn=lambda _p: ANSWER,
        )
        assert self._extras(gateway) == [None, None, GRANT]

    def test_a_question_after_a_plain_approval_is_answered_unkeyed(self, monkeypatch):
        """Approving once grants nothing: the answer to a question that follows
        is unkeyed, so the turn's later gated calls still ask."""
        monkeypatch.setattr(display_mod, "_session_auto_approve", False)
        _attended_config(monkeypatch)
        gateway = FakeGraphGateway(stream=self._approval_then_question())
        self._run(
            gateway,
            hitl_prompt_fn=lambda reqs: [{"type": "approve"} for _ in reqs],
            ask_user_prompt_fn=lambda _p: ANSWER,
        )
        assert self._extras(gateway) == [None, None, None]

    @pytest.mark.parametrize("prompted", [True, False])
    def test_a_question_after_an_ungranted_bridge_answer_is_answered_unkeyed(
        self, monkeypatch, prompted
    ):
        """The same for a channel answer with no grant behind it, whether the
        user approved once or the bridge answered on its own."""
        from EvoScientist.channels.interaction import ApprovalOutcome

        monkeypatch.setattr(display_mod, "_session_auto_approve", False)
        _attended_config(monkeypatch)
        gateway = FakeGraphGateway(stream=self._approval_then_question())
        self._run(
            gateway,
            hitl_outcome_fn=lambda reqs, _exhausted: ApprovalOutcome(
                decisions=[{"type": "approve"} for _ in reqs], prompted=prompted
            ),
            ask_user_prompt_fn=lambda _p: ANSWER,
        )
        assert self._extras(gateway) == [None, None, None]

    def test_a_question_after_a_config_rule_answer_is_answered_unkeyed(
        self, monkeypatch
    ):
        """An allow-list answer is not a grant either."""
        import EvoScientist.channels.interaction as interaction_mod

        monkeypatch.setattr(display_mod, "_session_auto_approve", False)
        monkeypatch.setattr(
            interaction_mod,
            "config_policy_snapshot",
            lambda reqs: ([{"type": "approve"} for _ in reqs], {}),
        )
        gateway = FakeGraphGateway(stream=self._approval_then_question())
        self._run(gateway, ask_user_prompt_fn=lambda _p: ANSWER)
        assert self._extras(gateway) == [None, None, None]

    def test_a_config_rule_answer_drops_a_key_the_turn_started_with(self, monkeypatch):
        """A channel-origin turn whose chat is granted (the caller says so), yet
        this batch is answered by the config rules before the bridge is asked:
        not the grant's answer, so the resume goes out unkeyed."""
        import EvoScientist.channels.interaction as interaction_mod

        def _bridge_must_not_be_asked(_reqs, _exhausted):
            raise AssertionError("the config rule answers first")

        monkeypatch.setattr(display_mod, "_session_auto_approve", False)
        monkeypatch.setattr(
            interaction_mod,
            "config_policy_snapshot",
            lambda reqs: ([{"type": "approve"} for _ in reqs], {}),
        )
        gateway = FakeGraphGateway(stream=_interrupts_then_done(1))
        self._run(
            gateway,
            approve_all_at_start=True,
            hitl_outcome_fn=_bridge_must_not_be_asked,
        )
        assert self._extras(gateway) == [GRANT, None]

    def test_a_chats_grant_does_not_key_a_local_turn(self, monkeypatch):
        """Only a channel-origin call site passes a chat's grant in."""
        from EvoScientist.cli import channel as channel_mod

        policy = channel_mod.ApprovalPolicy()
        policy.grant_session("telegram:c1")
        monkeypatch.setattr(channel_mod, "_approval_policy", policy)
        monkeypatch.setattr(display_mod, "_session_auto_approve", False)
        gateway = FakeGraphGateway(stream=_interrupts_then_done(0))
        self._run(gateway)
        assert self._extras(gateway) == [None]

    def test_a_declined_prompt_sends_nothing_more(self, monkeypatch):
        monkeypatch.setattr(display_mod, "_session_auto_approve", False)
        _attended_config(monkeypatch)
        gateway = FakeGraphGateway(stream=_interrupts_then_done(3))
        self._run(gateway, hitl_prompt_fn=lambda _requests: None)
        assert self._extras(gateway) == [None]

    def test_channel_relays_span_the_whole_uninterrupted_stream(self, monkeypatch):
        """Without approval pauses a granted turn is one stream: the thinking
        relay sends the accumulated text each time it grows and only the first
        todo list is relayed - the shape of an ``auto_approve`` turn."""
        monkeypatch.setattr(display_mod, "_session_auto_approve", True)
        first, second = "a" * 220, "b" * 220

        async def _fake_stream(_request):
            yield {"type": "thinking", "content": first}
            yield {
                "type": "tool_call",
                "id": "t1",
                "name": "write_todos",
                "args": {"todos": [{"content": "step one", "status": "pending"}]},
            }
            yield {
                "type": "tool_result",
                "id": "t1",
                "name": "write_todos",
                "content": "ok",
            }
            yield {"type": "thinking", "content": second}
            yield {
                "type": "tool_call",
                "id": "t2",
                "name": "write_todos",
                "args": {
                    "todos": [
                        {"content": "step one", "status": "completed"},
                        {"content": "step two", "status": "pending"},
                    ]
                },
            }
            yield {
                "type": "tool_result",
                "id": "t2",
                "name": "write_todos",
                "content": "ok",
            }
            yield {"type": "text", "content": "finished"}
            yield {"type": "done", "content": "finished"}

        thinking: list[str] = []
        todos: list[list] = []
        gateway = FakeGraphGateway(stream=_fake_stream)
        self._run(
            gateway,
            on_thinking=thinking.append,
            on_todo=lambda items: todos.append(list(items)),
        )
        assert self._extras(gateway) == [GRANT]
        assert thinking == [first, first + second]
        assert [[item["content"] for item in batch] for batch in todos] == [
            ["step one"]
        ]


class TestConsumerSessionGrantReachesTheGraph:
    """The standalone consumer follows the same rule as the Rich loop, per chat."""

    @staticmethod
    def _consumer(stream, media_dir):
        return TestConsumerHitlRoundBudget()._consumer(stream, media_dir)

    @staticmethod
    async def _send(bus, chat_id="c1", content="go"):
        await bus.publish_inbound(
            BusInbound(channel="stub", sender_id="u1", chat_id=chat_id, content=content)
        )

    @staticmethod
    def _keyed(gateway):
        return [
            (request.configurable_extra or {}).get("hitl_approve_all") is True
            for request in gateway.requests
        ]

    async def test_a_granted_chat_is_keyed_from_its_first_call(self, tmp_path):
        consumer, bus, gateway = self._consumer(_interrupts_then_done(2), tmp_path)
        consumer._approval_policy.grant_session("stub:c1")
        await self._send(bus)
        task = asyncio.create_task(consumer.run())
        outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=10.0)
        await consumer.stop()
        await task

        assert outbound.content == "finished"
        assert self._keyed(gateway) == [True, True, True]

    async def test_the_reply_that_grants_is_sent_unkeyed(self, tmp_path, monkeypatch):
        import EvoScientist.channels.interaction as interaction_mod

        monkeypatch.setattr(
            interaction_mod, "config_policy_snapshot", lambda reqs: (None, {})
        )
        consumer, bus, gateway = self._consumer(_interrupts_then_done(2), tmp_path)
        await self._send(bus)
        task = asyncio.create_task(consumer.run())
        prompt = await asyncio.wait_for(bus.consume_outbound(), timeout=10.0)
        assert "Approval Required" in prompt.content
        await self._send(bus, content="3")  # approve all for this chat
        outbound = None
        for _ in range(4):
            outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=10.0)
            if outbound.content == "finished":
                break
        await consumer.stop()
        await task

        assert outbound is not None
        assert outbound.content == "finished"
        # first call, the granting answer, then the auto-answered resume.
        assert self._keyed(gateway) == [False, False, True]

    async def test_another_chats_grant_does_not_key_this_chat(
        self, tmp_path, monkeypatch
    ):
        import EvoScientist.channels.interaction as interaction_mod

        monkeypatch.setattr(
            interaction_mod,
            "config_policy_snapshot",
            lambda reqs: ([{"type": "approve"} for _ in reqs], {}),
        )
        consumer, bus, gateway = self._consumer(_interrupts_then_done(1), tmp_path)
        consumer._approval_policy.grant_session("stub:other")
        await self._send(bus)
        task = asyncio.create_task(consumer.run())
        outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=10.0)
        await consumer.stop()
        await task

        assert outbound.content == "finished"
        # The config rule answered the approval: not the grant, so no key.
        assert self._keyed(gateway) == [False, False]

    async def test_a_declined_reply_sends_no_further_call(self, tmp_path, monkeypatch):
        import EvoScientist.channels.interaction as interaction_mod

        monkeypatch.setattr(
            interaction_mod, "config_policy_snapshot", lambda reqs: (None, {})
        )
        consumer, bus, gateway = self._consumer(_interrupts_then_done(2), tmp_path)
        await self._send(bus)
        task = asyncio.create_task(consumer.run())
        prompt = await asyncio.wait_for(bus.consume_outbound(), timeout=10.0)
        assert "Approval Required" in prompt.content
        await self._send(bus, content="2")  # reject
        await asyncio.wait_for(bus.consume_outbound(), timeout=10.0)
        await consumer.stop()
        await task

        assert self._keyed(gateway) == [False]

    async def test_a_grant_answer_with_a_reject_is_not_keyed(
        self, tmp_path, monkeypatch
    ):
        """Belt and braces: the key is sent only when every decision approves."""
        from EvoScientist.channels.interaction import ApprovalOutcome

        async def _resolve(*_args, **_kwargs):
            return ApprovalOutcome(
                decisions=[{"type": "reject", "message": "no"}], session_granted=True
            )

        monkeypatch.setattr(consumer_mod, "resolve_approval", _resolve)
        consumer, bus, gateway = self._consumer(_interrupts_then_done(1), tmp_path)
        await self._send(bus)
        task = asyncio.create_task(consumer.run())
        outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=10.0)
        await consumer.stop()
        await task

        assert outbound.content == "finished"
        assert self._keyed(gateway) == [False, False]

    async def test_a_question_in_an_ungranted_chat_is_answered_unkeyed(
        self, tmp_path, monkeypatch
    ):
        """A keyed answer here would let every later gated call of the turn run
        with no prompt; another chat's grant must not count."""
        calls = 0

        async def _fake_stream(_request):
            nonlocal calls
            calls += 1
            if calls == 1:
                yield {
                    "type": "ask_user",
                    "interrupt_id": "ask-1",
                    "questions": [{"question": "Continue?"}],
                }
                return
            yield {"type": "text", "content": "finished"}
            yield {"type": "done", "content": "finished"}

        async def _answer(_questions, _io, timeout=0):
            return dict(ANSWER)

        consumer, bus, gateway = self._consumer(_fake_stream, tmp_path)
        consumer._approval_policy.grant_session("stub:other")
        monkeypatch.setattr(consumer_mod, "resolve_ask_user", _answer)
        await self._send(bus)
        task = asyncio.create_task(consumer.run())
        outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=10.0)
        await consumer.stop()
        await task

        assert outbound.content == "finished"
        assert self._keyed(gateway) == [False, False]

    async def test_a_question_after_a_plain_approval_is_answered_unkeyed(
        self, tmp_path, monkeypatch
    ):
        """Approving once grants nothing, so the question's answer is unkeyed."""
        from EvoScientist.channels.interaction import ApprovalOutcome

        calls = 0

        async def _fake_stream(_request):
            nonlocal calls
            calls += 1
            if calls == 1:
                yield _interrupt_event(1)
                return
            if calls == 2:
                yield {
                    "type": "ask_user",
                    "interrupt_id": "ask-1",
                    "questions": [{"question": "Continue?"}],
                }
                return
            yield {"type": "text", "content": "finished"}
            yield {"type": "done", "content": "finished"}

        async def _approve_once(requests, *_args, **_kwargs):
            return ApprovalOutcome(
                decisions=[{"type": "approve"} for _ in requests], prompted=True
            )

        async def _answer(_questions, _io, timeout=0):
            return dict(ANSWER)

        consumer, bus, gateway = self._consumer(_fake_stream, tmp_path)
        monkeypatch.setattr(consumer_mod, "resolve_approval", _approve_once)
        monkeypatch.setattr(consumer_mod, "resolve_ask_user", _answer)
        await self._send(bus)
        task = asyncio.create_task(consumer.run())
        outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=10.0)
        await consumer.stop()
        await task

        assert outbound.content == "finished"
        assert self._keyed(gateway) == [False, False, False]

    async def test_an_approval_that_times_out_sends_no_further_call(
        self, tmp_path, monkeypatch
    ):
        import EvoScientist.channels.interaction as interaction_mod

        monkeypatch.setattr(
            interaction_mod, "config_policy_snapshot", lambda reqs: (None, {})
        )
        monkeypatch.setattr(consumer_mod, "HITL_APPROVAL_TIMEOUT", 0.05)
        consumer, bus, gateway = self._consumer(_interrupts_then_done(2), tmp_path)
        await self._send(bus)
        task = asyncio.create_task(consumer.run())
        prompt = await asyncio.wait_for(bus.consume_outbound(), timeout=10.0)
        assert "Approval Required" in prompt.content
        await asyncio.wait_for(bus.consume_outbound(), timeout=10.0)  # timeout notice
        await consumer.stop()
        await task

        assert self._keyed(gateway) == [False]

    async def test_a_refed_reply_starts_an_unkeyed_turn(self, tmp_path, monkeypatch):
        """An unrecognized reply declines the approval and becomes a new turn."""
        import EvoScientist.channels.interaction as interaction_mod

        monkeypatch.setattr(
            interaction_mod, "config_policy_snapshot", lambda reqs: (None, {})
        )
        consumer, bus, gateway = self._consumer(_interrupts_then_done(1), tmp_path)
        await self._send(bus)
        task = asyncio.create_task(consumer.run())
        prompt = await asyncio.wait_for(bus.consume_outbound(), timeout=10.0)
        assert "Approval Required" in prompt.content
        await self._send(bus, content="do something else instead")
        outbound = None
        for _ in range(4):
            outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=10.0)
            if outbound.content == "finished":
                break
        await consumer.stop()
        await task

        assert outbound is not None
        assert outbound.content == "finished"
        assert self._keyed(gateway) == [False, False]

    async def test_an_ask_user_pause_still_splits_a_granted_turn(
        self, tmp_path, monkeypatch
    ):
        """Approvals no longer pause a granted turn, a question still does: the
        reply holds the text after it and the todo list is relayed per stream."""

        def _todos(call_id, *items):
            return {
                "type": "tool_call",
                "id": call_id,
                "name": "write_todos",
                "args": {"todos": [{"content": i, "status": "pending"} for i in items]},
            }

        calls = 0

        async def _fake_stream(_request):
            nonlocal calls
            calls += 1
            if calls == 1:
                yield {"type": "thinking", "content": "thinking before."}
                yield _todos("t1", "step one")
                yield {"type": "text", "content": "before the question."}
                yield {
                    "type": "ask_user",
                    "interrupt_id": "ask-1",
                    "questions": [{"question": "Continue?"}],
                }
                return
            yield {"type": "thinking", "content": "thinking after."}
            yield _todos("t2", "step one", "step two")
            yield {"type": "text", "content": "after the question."}
            yield {"type": "done", "content": "after the question."}

        async def _answer(_questions, _io, timeout=0):
            return {"answers": ["yes"], "status": "answered"}

        bus = MessageBus()
        manager = ChannelManager(bus, media_dir=tmp_path)
        stub = _StubChannel()
        stub.send_thinking = True
        manager.register(stub)
        agent = FakeCheckpointAgent()
        gateway = FakeGraphGateway(stream=_fake_stream, checkpoint=agent)
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
            drain_timeout=1.0,
        )
        consumer._approval_policy.grant_session("stub:c1")
        monkeypatch.setattr(consumer_mod, "resolve_ask_user", _answer)

        await self._send(bus)
        task = asyncio.create_task(consumer.run())
        outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=10.0)
        await consumer.stop()
        await task

        assert outbound.content == "after the question."
        assert self._keyed(gateway) == [True, True]
        relayed = [chunk[2] for chunk in stub._sent_chunks]
        assert len([text for text in relayed if "tasks" in text]) == 2
        assert [text for text in relayed if "thinking" in text] == [
            "\U0001f9e0\nthinking before.\n\u23f3",
            "\U0001f9e0\nthinking after.\n\u23f3",
        ]


def test_resolver_reports_the_grant_only_on_its_fast_path(monkeypatch):
    import EvoScientist.channels.interaction as interaction_mod
    from EvoScientist.channels.interaction import ApprovalOutcome

    pending = _interrupt_event(1)
    monkeypatch.setattr(
        "EvoScientist.EvoScientist._ensure_config",
        lambda: SimpleNamespace(
            auto_approve=False, dangerous_mode=False, shell_allow_list=""
        ),
    )

    monkeypatch.setattr(display_mod, "_session_auto_approve", True)
    assert display_mod._resolve_hitl_approval(pending).session_granted is True

    monkeypatch.setattr(display_mod, "_session_auto_approve", False)

    def _grant(_requests):
        display_mod._session_auto_approve = True
        return [{"type": "approve"}]

    granting = display_mod._resolve_hitl_approval(pending, prompt_fn=_grant)
    assert granting.human_prompted is True
    assert granting.session_granted is False

    monkeypatch.setattr(display_mod, "_session_auto_approve", False)
    bridged = display_mod._resolve_hitl_approval(
        pending,
        outcome_fn=lambda reqs, _exhausted: ApprovalOutcome(
            decisions=[{"type": "approve"}], session_granted=True
        ),
    )
    assert bridged.session_granted is True

    monkeypatch.setattr(
        interaction_mod,
        "config_policy_snapshot",
        lambda reqs: ([{"type": "approve"}], {}),
    )
    assert display_mod._resolve_hitl_approval(pending).session_granted is False


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

    def _consumer(
        self, stream, media_dir
    ) -> tuple[InboundConsumer, MessageBus, FakeGraphGateway]:
        bus = MessageBus()
        mgr = ChannelManager(bus, media_dir=media_dir)
        mgr.register(_StubChannel())
        agent = FakeCheckpointAgent()
        gateway = FakeGraphGateway(stream=stream, checkpoint=agent)
        return (
            InboundConsumer(
                bus=bus,
                manager=mgr,
                agent=agent,
                thread_id="",
                workspace=Workspace("/tmp/channel-project"),
                graph_gateway=gateway,
                max_concurrent=2,
                max_pending=10,
                inference_timeout=5.0,
                drain_timeout=1.0,
            ),
            bus,
            gateway,
        )

    async def test_session_grant_resumes_past_50_rounds(self, tmp_path):
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

        consumer, bus, _gateway = self._consumer(_fake_stream, tmp_path)
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

    async def test_session_grant_runs_long_unattended_turns_past_200_rounds(
        self, tmp_path
    ):
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

        consumer, bus, _gateway = self._consumer(_fake_stream, tmp_path)
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

    @pytest.mark.parametrize("pause", ["interrupt", "ask_user"])
    async def test_pause_closes_stream_in_consumer_context(
        self, monkeypatch, tmp_path, pause
    ):
        """Leaving the stream at an approval or a question closes it right
        away, in the consumer's own task and contextvars context (issue #568).

        The fake stream binds and resets the run event sink like
        ``stream_agent_events``: ``reset_run_event_sink`` raises
        ``ValueError`` if it runs in another context, e.g. in the event
        loop's generator finalizer, or when ``_timeout_aiter`` advances the
        stream in a child task (``asyncio.wait_for`` before Python 3.12).
        """
        stream_calls = 0
        started_in: list[asyncio.Task | None] = []
        closed_in: list[asyncio.Task | None] = []
        sink_at_start: list[object] = []
        loop_errors: list[dict] = []
        asyncio.get_running_loop().set_exception_handler(
            lambda _loop, context: loop_errors.append(context)
        )

        async def _fake_stream(_request):
            nonlocal stream_calls
            stream_calls += 1
            started_in.append(asyncio.current_task())
            sink_at_start.append(mw_events._current_run_event_sink.get())
            token = mw_events.bind_run_event_sink(MagicMock())
            try:
                if stream_calls == 1:
                    if pause == "interrupt":
                        yield _interrupt_event(stream_calls)
                    else:
                        yield {
                            "type": "ask_user",
                            "interrupt_id": "ask-1",
                            "questions": [{"question": "Continue?"}],
                        }
                    return
                yield {"type": "text", "content": "final answer"}
                yield {"type": "done", "content": "final answer"}
            finally:
                closed_in.append(asyncio.current_task())
                mw_events.reset_run_event_sink(token)

        async def _fake_ask_user(_questions, _io, timeout=0):
            return {"answers": ["yes"], "status": "answered"}

        consumer, bus, _gateway = self._consumer(_fake_stream, tmp_path)
        consumer._approval_policy.grant_session("stub:c1")
        monkeypatch.setattr(consumer_mod, "resolve_ask_user", _fake_ask_user)

        await bus.publish_inbound(
            BusInbound(channel="stub", sender_id="u1", chat_id="c1", content="go")
        )
        task = asyncio.create_task(consumer.run())
        outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=10.0)

        assert outbound.content == "final answer"
        assert stream_calls == 2
        assert closed_in == started_in
        # The first run's sink was unbound before the resumed run started.
        assert sink_at_start == [None, None]
        assert loop_errors == []

        await consumer.stop()
        await task

    async def test_human_round_budget_exhausted_notifies_channel(
        self, monkeypatch, caplog, tmp_path
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

        consumer, bus, gateway = self._consumer(_fake_stream, tmp_path)
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
        assert consumer.agent.updates[-1][0] is None
        assert consumer.agent.updates[-1][1] == "__end__"

        await consumer.stop()
        await task

    async def test_ask_user_rounds_share_budget_and_drain_cancelled(
        self, monkeypatch, caplog, tmp_path
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

        consumer, bus, _gateway = self._consumer(_fake_stream, tmp_path)
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
        assert consumer.agent.updates[-1][1] == "__end__"

        await consumer.stop()
        await task

    async def test_human_budget_does_not_block_a_following_session_grant(
        self, monkeypatch, tmp_path
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

        consumer, bus, _gateway = self._consumer(_fake_stream, tmp_path)
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
        self, monkeypatch, caplog, tmp_path
    ):
        import EvoScientist.channels.hitl_budget as budget_mod

        monkeypatch.setattr(budget_mod, "MAX_HUMAN_HITL_ROUNDS", 0)

        async def _fake_stream(_request):
            yield {"type": "text", "content": "partial answer"}
            yield _interrupt_event(1)

        consumer, bus, _gateway = self._consumer(_fake_stream, tmp_path)
        consumer.agent.update_error = RuntimeError("checkpoint store down")

        await bus.publish_inbound(
            BusInbound(channel="stub", sender_id="u1", chat_id="c1", content="go")
        )
        with caplog.at_level(logging.WARNING, logger="EvoScientist"):
            task = asyncio.create_task(consumer.run())
            first = await asyncio.wait_for(bus.consume_outbound(), timeout=10.0)
            second = await asyncio.wait_for(bus.consume_outbound(), timeout=10.0)

        assert first.content == "partial answer"
        assert second.content == "Approval round limit reached; stopping this turn."
        failures = [
            r
            for r in caplog.records
            if "Could not close the parked checkpoint" in r.getMessage()
        ]
        assert len(failures) == 1
        assert failures[0].exc_info is None

        await consumer.stop()
        await task

    async def test_total_cap_closes_without_another_resume(self, monkeypatch, tmp_path):
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

        consumer, bus, gateway = self._consumer(_fake_stream, tmp_path)
        consumer._approval_policy.grant_session("stub:c1")

        await bus.publish_inbound(
            BusInbound(channel="stub", sender_id="u1", chat_id="c1", content="go")
        )
        task = asyncio.create_task(consumer.run())
        outbound = await asyncio.wait_for(bus.consume_outbound(), timeout=10.0)

        resumes = [r for r in gateway.requests if isinstance(r.message, Command)]
        assert stream_calls == 2
        assert len(resumes) == 1
        assert consumer.agent.updates[-1][1] == "__end__"
        assert outbound.content == "Approval round limit reached; stopping this turn."

        await consumer.stop()
        await task


def test_tui_loop_does_not_build_a_resume_past_the_total_cap(monkeypatch):
    """TUI HITL loop, without a Pilot harness.

    Each pause branch calls ``hitl_budget_stop`` before it prompts or
    builds a resume. Auto rounds ignore the human cap; at the total cap the
    branch stops with no resume left unsent.
    """
    import EvoScientist.channels.hitl_budget as budget_mod
    from EvoScientist.channels.hitl_budget import hitl_budget_stop

    monkeypatch.setattr(budget_mod, "MAX_HITL_TOTAL_ROUNDS", 2)
    resumes_built = 0
    unsent_resume = None
    for total_rounds in range(1, 6):
        # The previous iteration's resume, if any, is what this round streams.
        unsent_resume = None
        if hitl_budget_stop(
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
    from EvoScientist.channels.hitl_budget import hitl_budget_stop

    assert not hitl_budget_stop(human_rounds=50, total_rounds=51, needs_human=False)
    assert hitl_budget_stop(human_rounds=50, total_rounds=51, needs_human=True)


def test_tui_loop_level_total_cap_uses_completed_round_helper(monkeypatch):
    """The TUI loop-top guard is ``hitl_completed_round_cap_reached``,
    not a rewritten copy of the stop condition (issue #469 review)."""
    import EvoScientist.channels.hitl_budget as budget_mod
    from EvoScientist.channels.hitl_budget import hitl_completed_round_cap_reached

    monkeypatch.setattr(budget_mod, "MAX_HITL_TOTAL_ROUNDS", 2)
    assert not hitl_completed_round_cap_reached(0)
    assert not hitl_completed_round_cap_reached(1)
    assert hitl_completed_round_cap_reached(2)
    assert hitl_completed_round_cap_reached(3)


def test_tui_hitl_loop_checks_completed_round_cap_before_clearing_pending():
    """``_stream_with_widgets`` applies the helper before it clears pending,
    so a round that stored a pause without building a resume cannot replay
    unbounded. Reads the TUI source so this fails if the call is moved
    after the pending clear (no Pilot harness)."""
    from pathlib import Path

    import EvoScientist.cli.tui_interactive as tui

    src = Path(tui.__file__).read_text()
    cap = src.index("hitl_completed_round_cap_reached(_hitl_round)")
    pending = src.index("state.pending_interrupt = None", cap)
    assert pending - cap < 400
    between = src[cap:pending]
    assert "while True" not in between


def test_channel_response_with_budget_stop_keeps_partial_then_notice():
    from EvoScientist.channels.hitl_budget import (
        HITL_BUDGET_STOP_NOTICE,
        channel_response_with_budget_stop,
    )

    assert channel_response_with_budget_stop("") == HITL_BUDGET_STOP_NOTICE
    assert channel_response_with_budget_stop("   ") == HITL_BUDGET_STOP_NOTICE
    assert channel_response_with_budget_stop("partial") == (
        f"partial\n\n{HITL_BUDGET_STOP_NOTICE}"
    )
    assert (
        channel_response_with_budget_stop(HITL_BUDGET_STOP_NOTICE)
        == HITL_BUDGET_STOP_NOTICE
    )


def test_abandoned_tool_messages_closes_invalid_tool_calls():
    """HITL close must patch invalid calls too, matching crash recovery."""
    from langchain_core.messages import AIMessage, ToolMessage

    from EvoScientist.backends import (
        HITL_ROUND_LIMIT_REJECT_MESSAGE,
        abandoned_tool_messages,
    )

    patch = abandoned_tool_messages(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "execute",
                        "args": {},
                        "id": "ok-1",
                        "type": "tool_call",
                    }
                ],
                invalid_tool_calls=[
                    {
                        "name": "execute",
                        "args": "not-json",
                        "id": "bad-1",
                        "error": "parse",
                        "type": "invalid_tool_call",
                    }
                ],
            ),
            ToolMessage(content="done", name="execute", tool_call_id="ok-1"),
        ],
        reason="round_limit",
    )
    assert [message.tool_call_id for message in patch] == ["bad-1"]
    assert HITL_ROUND_LIMIT_REJECT_MESSAGE in patch[0].content


def test_abandoned_tool_messages_closes_invalid_tool_calls_in_dicts():
    """The server gateway hands back state messages as dicts (#509).

    ``convert_to_messages`` moves a dict's ``invalid_tool_calls`` into
    ``additional_kwargs``, so they are read from there.
    """
    from EvoScientist.backends import abandoned_tool_messages

    patch = abandoned_tool_messages(
        [
            {
                "type": "ai",
                "content": "",
                "tool_calls": [
                    {
                        "name": "execute",
                        "args": {},
                        "id": "ok-1",
                        "type": "tool_call",
                    }
                ],
                "invalid_tool_calls": [
                    {
                        "name": "execute",
                        "args": "not-json",
                        "id": "bad-1",
                        "error": "parse",
                        "type": "invalid_tool_call",
                    }
                ],
            },
            {
                "type": "tool",
                "content": "done",
                "name": "execute",
                "tool_call_id": "ok-1",
            },
        ],
        reason="round_limit",
    )
    assert [message.tool_call_id for message in patch] == ["bad-1"]
    assert "was not executed" in patch[0].content


def test_hitl_pause_unresolved_does_not_replay_without_a_resume():
    from EvoScientist.channels.hitl_budget import hitl_pause_unresolved

    assert hitl_pause_unresolved(resuming=False, pending=True)
    assert not hitl_pause_unresolved(resuming=True, pending=True)
    assert not hitl_pause_unresolved(resuming=False, pending=False)


def test_tui_unresolved_pause_exits_without_budget_notice():
    """A stored pause with no resume breaks the loop. The round-limit
    notice stays inside the budget-exhausted branch (CodeRabbit)."""
    from pathlib import Path

    import EvoScientist.cli.tui_interactive as tui

    src = Path(tui.__file__).read_text()
    unresolved = src.index("hitl_pause_unresolved(")
    notice = src.index("HITL_BUDGET_STOP_NOTICE", unresolved)
    gate = src[notice - 250 : notice]
    assert "_hitl_budget_exhausted" in gate
    assert "_hitl_unresolved = True" in src


def test_tui_budget_stop_folds_notice_into_channel_response():
    """Channel replies come from ``_stream_with_widgets``'s return value,
    not ``_append_system``. The close path must fold the notice in when
    channel HITL callbacks are set (issue #469 / CodeRabbit)."""
    from pathlib import Path

    import EvoScientist.cli.tui_interactive as tui

    src = Path(tui.__file__).read_text()
    folded = src.index("channel_response_with_budget_stop(response)")
    gate = src[folded - 700 : folded]
    assert "HITL_BUDGET_STOP_NOTICE" in gate
    assert "channel_hitl_fn" in gate
    assert "channel_ask_user_fn" in gate


class _StickyCheckpointAgent:
    """``aupdate_state`` is a no-op so recovery's verify still sees ``next``."""

    async def aget_state(self, _config):
        return SimpleNamespace(
            next=("tools",),
            tasks=(),
            interrupts=(object(),),
            values={},
        )

    async def aupdate_state(self, *_args, **_kwargs):
        return None


async def test_close_parked_checkpoint_keeps_finished_sibling_results():
    """Clear first so an answered sibling (ask_user beside execute) stays
    in history; only the unanswered call is patched (issue #469 review)."""
    from langchain_core.messages import AIMessage, ToolMessage

    from EvoScientist.backends import (
        HITL_ROUND_LIMIT_REJECT_MESSAGE,
        close_parked_checkpoint,
    )
    from EvoScientist.gateway import GraphTarget

    pending_ask = ToolMessage(
        content="user answered from the widget",
        name="ask_user",
        tool_call_id="ask-1",
    )
    agent = FakeCheckpointAgent(
        values={
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "ask_user",
                            "args": {"questions": [{"question": "Continue?"}]},
                            "id": "ask-1",
                            "type": "tool_call",
                        },
                        {
                            "name": "execute",
                            "args": {"command": "echo sibling"},
                            "id": "exec-1",
                            "type": "tool_call",
                        },
                    ],
                )
            ]
        },
        pending_write=pending_ask,
    )

    gateway = FakeGraphGateway(checkpoint=agent)
    await close_parked_checkpoint(
        gateway, GraphTarget(), "t-sibling", reason="round_limit"
    )

    assert [as_node for _values, as_node in agent.updates] == [
        "__end__",
        "tools",
        "__end__",
    ]
    patch = next(values for values, node in agent.updates if node == "tools")[
        "messages"
    ]
    assert [message.tool_call_id for message in patch] == ["exec-1"]
    assert HITL_ROUND_LIMIT_REJECT_MESSAGE in patch[0].content
    seen = {
        getattr(message, "tool_call_id", None): getattr(message, "content", None)
        for message in agent.values["messages"]
        if getattr(message, "type", None) == "tool"
    }
    assert seen["ask-1"] == "user answered from the widget"
    assert HITL_ROUND_LIMIT_REJECT_MESSAGE in seen["exec-1"]
    verify = await agent.aget_state({})
    assert verify.next == ()
    assert not verify.interrupts


async def test_close_parked_checkpoint_trailing_end_even_without_dangling_calls():
    """With no unanswered calls the first END still gets a trailing END,
    matching recovery: the first clear can leave next == ('model',)."""
    from EvoScientist.backends import close_parked_checkpoint
    from EvoScientist.gateway import GraphTarget

    agent = FakeCheckpointAgent()
    gateway = FakeGraphGateway(checkpoint=agent)
    await close_parked_checkpoint(
        gateway, GraphTarget(), "t-empty", reason="round_limit"
    )
    assert [as_node for _values, as_node in agent.updates] == ["__end__", "__end__"]
    assert all(values is None for values, _node in agent.updates)
    verify = await agent.aget_state({})
    assert verify.next == ()
    assert not verify.interrupts


async def test_close_parked_checkpoint_raises_when_verify_still_stuck():
    """Unlike the old parallel close, a failed repair is visible."""
    from EvoScientist.backends import close_parked_checkpoint
    from EvoScientist.gateway import GraphTarget

    with pytest.raises(RuntimeError, match="Could not close parked HITL"):
        await close_parked_checkpoint(
            FakeGraphGateway(checkpoint=_StickyCheckpointAgent()),
            GraphTarget(),
            "t-sticky",
            reason="round_limit",
        )


async def test_close_parked_checkpoint_ignores_local_graph():
    """After #470, close must use the gateway even if local_graph is present."""
    from EvoScientist.backends import close_parked_checkpoint
    from EvoScientist.gateway import GraphTarget

    agent = FakeCheckpointAgent()
    sentinel = SimpleNamespace(
        aget_state=AsyncMock(side_effect=AssertionError("used local_graph")),
        aupdate_state=AsyncMock(side_effect=AssertionError("used local_graph")),
    )
    gateway = FakeGraphGateway(checkpoint=agent)
    await close_parked_checkpoint(
        gateway, GraphTarget(local_graph=sentinel), "t-gateway", reason="round_limit"
    )
    assert [as_node for _values, as_node in agent.updates] == ["__end__", "__end__"]
    assert sentinel.aget_state.await_count == 0
    assert [node for _target, _tid, _values, node in gateway.updated_states] == [
        "__end__",
        "__end__",
    ]


async def test_close_parked_checkpoint_via_local_gateway():
    """The in-process gateway still closes through its snapshot/update seam."""
    from EvoScientist.backends import close_parked_checkpoint
    from EvoScientist.gateway import GraphTarget, LocalGraphGateway

    agent = FakeCheckpointAgent()
    await close_parked_checkpoint(
        LocalGraphGateway(),
        GraphTarget(local_graph=agent),
        "t-local",
        reason="round_limit",
    )
    assert [as_node for _values, as_node in agent.updates] == ["__end__", "__end__"]
