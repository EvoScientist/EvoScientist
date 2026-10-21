"""Goal events through the stream decoder and the Rich display."""

from __future__ import annotations

import asyncio
import threading
from io import StringIO
from unittest.mock import patch

import pytest
from rich.console import Console

from EvoScientist.middleware.goal import new_goal_request
from EvoScientist.stream.display import create_streaming_display, goal_event_lines
from EvoScientist.stream.state import StreamState
from tests.goal_fakes import ACCEPT, CANCEL

pytestmark = pytest.mark.usefixtures("no_mcp")

CRITERIA = "- the report exists\n- it has a title"


def _render(renderable) -> str:
    out = StringIO()
    Console(file=out, width=100, color_system=None).print(renderable)
    return out.getvalue()


class TestLines:
    def test_accepted_shows_the_criteria_that_were_adopted(self):
        event = {"change": "accepted", "objective": "ship it", "criteria": CRITERIA}
        assert goal_event_lines(event) == [
            ("Goal accepted.", "green"),
            ("  - the report exists", "dim"),
            ("  - it has a title", "dim"),
        ]

    def test_a_goal_adopted_without_a_review_is_shown_as_a_warning(self):
        event = {
            "change": "accepted",
            "objective": "ship it",
            "criteria": CRITERIA,
            "unreviewed": True,
        }
        assert goal_event_lines(event) == [
            ("Goal accepted without review (auto mode).", "yellow"),
            ("  - the report exists", "yellow"),
            ("  - it has a title", "yellow"),
        ]
        amended = {**event, "change": "amended", "criteria": "- a"}
        assert goal_event_lines(amended) == [
            ("Goal amended without review (auto mode).", "yellow"),
            ("  ship it", "yellow"),
            ("  - a", "yellow"),
        ]

    def test_the_stream_event_carries_the_mark_only_when_set(self):
        from EvoScientist.stream.emitter import StreamEventEmitter

        plain = StreamEventEmitter.goal_state(change="accepted").data
        assert "unreviewed" not in plain
        marked = StreamEventEmitter.goal_state(change="accepted", unreviewed=True).data
        assert marked["unreviewed"] is True

    def test_amended_also_shows_the_revised_objective(self):
        event = {"change": "amended", "objective": "ship it soon", "criteria": "- a"}
        assert goal_event_lines(event) == [
            ("Goal amended.", "green"),
            ("  ship it soon", "dim"),
            ("  - a", "dim"),
        ]

    def test_a_failure_says_why_on_one_line(self):
        event = {"change": "failed", "note": "503 upstream\nretry later"}
        assert goal_event_lines(event) == [
            ("Goal request failed: 503 upstream retry later", "red")
        ]

    def test_blocked_and_complete_carry_the_note(self):
        assert goal_event_lines({"change": "blocked", "note": "need a key"}) == [
            ("Goal blocked: need a key", "yellow")
        ]
        assert goal_event_lines({"change": "complete", "note": "all tests pass"}) == [
            ("Goal complete. all tests pass", "green")
        ]

    def test_a_long_note_is_cut_on_the_status_line(self):
        note = "evidence " * 100
        ((text, _style),) = goal_event_lines({"change": "complete", "note": note})
        assert text.startswith("Goal complete. evidence evidence")
        assert text.endswith("...")
        assert len(text) == len("Goal complete. ") + 240
        ((blocked, _style),) = goal_event_lines({"change": "blocked", "note": note})
        assert len(blocked) == len("Goal blocked: ") + 240

    def test_the_pass_limit_says_what_to_do_next(self):
        ((text, _style),) = goal_event_lines({"change": "unmet"})
        assert "/goal amend" in text
        assert "/goal clear" in text

    def test_an_ignored_request_gives_its_reason(self):
        event = {"change": "ignored", "note": "The goal is paused; nothing was done."}
        ((text, _style),) = goal_event_lines(event)
        assert text == "Nothing was changed: The goal is paused; nothing was done."

    def test_an_unknown_change_is_still_shown(self):
        assert goal_event_lines({"change": "frozen"}) == [("Goal: frozen", "dim")]


class TestDisplay:
    def test_goal_lines_are_part_of_the_frame(self):
        state = StreamState()
        state.handle_event(
            {"type": "goal_state", "change": "paused", "status": "paused"}
        )
        text = _render(
            create_streaming_display(**state.get_display_args(), is_final=True)
        )
        assert "Goal paused." in text

    def test_the_drafting_line_goes_once_the_draft_has_an_outcome(self):
        state = StreamState()
        state.handle_event({"type": "goal_state", "change": "drafting"})
        drafting = _render(create_streaming_display(**state.get_display_args()))
        assert "Drafting acceptance criteria..." in drafting
        state.handle_event(
            {"type": "goal_state", "change": "accepted", "criteria": "- a"}
        )
        accepted = _render(create_streaming_display(**state.get_display_args()))
        assert "Drafting" not in accepted
        assert "Goal accepted." in accepted

    def test_a_second_draft_shows_its_line_again(self):
        state = StreamState()
        for change in ("drafting", "drafting"):
            state.handle_event({"type": "goal_state", "change": change})
        text = _render(create_streaming_display(**state.get_display_args()))
        assert text.count("Drafting acceptance criteria...") == 1

    def test_a_turn_with_nothing_else_leaves_no_spinner_behind(self):
        """A request that only changed the goal's status has no reply to show."""
        state = StreamState()
        state.handle_event({"type": "goal_state", "change": "cleared"})
        text = _render(
            create_streaming_display(**state.get_display_args(), is_final=True)
        )
        assert "Processing" not in text

    def test_an_empty_final_frame_shows_nothing(self):
        text = _render(
            create_streaming_display(**StreamState().get_display_args(), is_final=True)
        )
        assert "Processing" not in text

    def test_an_empty_frame_still_spins_while_the_turn_runs(self):
        text = _render(create_streaming_display(**StreamState().get_display_args()))
        assert "Processing" in text


class TestDecoder:
    async def _events(self, tmp_path, payloads):
        from EvoScientist.gateway import GraphTarget, RunRequest
        from EvoScientist.gateway.local import LocalGraphGateway
        from tests.goal_fakes import cli_goal_agent

        agent, _ = cli_goal_agent(tmp_path)
        gateway = LocalGraphGateway()
        seen: list[list[dict]] = []
        for message, request in payloads:
            run = RunRequest(
                message=message,
                thread_id="t1",
                target=GraphTarget(local_graph=agent),
                configurable_extra={"rubric_run": {"mode": "thread", "changes": []}},
                goal_request=request,
            )
            seen.append([event async for event in gateway.stream_events(run)])
        return seen

    async def test_a_request_is_shown_as_a_question_and_goal_events(self, tmp_path):
        from langgraph.types import Command

        create = new_goal_request("create", objective="ship the report")
        first, second = await self._events(
            tmp_path, [("", create), (Command(resume=ACCEPT), None)]
        )
        kinds = [event["type"] for event in first]
        assert kinds == ["goal_state", "ask_user", "done"]
        assert first[0]["change"] == "drafting"
        assert "ship the report" in first[1]["questions"][0]["question"]
        accepted = next(e for e in second if e["type"] == "goal_state")
        assert accepted["change"] == "accepted"
        assert accepted["criteria"] == "- the report exists"
        assert accepted["message"] == "ship the report"
        changes = [e["change"] for e in second if e["type"] == "goal_state"]
        assert changes == ["accepted", "complete"]
        assert [e["type"] for e in second][-1] == "done"

    async def test_an_amendment_carries_the_revised_objective(self, tmp_path):
        from langgraph.types import Command

        from EvoScientist.gateway import GraphTarget, RunRequest
        from EvoScientist.gateway.local import LocalGraphGateway
        from tests.goal_fakes import cli_goal_agent

        agent, _ = cli_goal_agent(
            tmp_path,
            drafts=["- a", "OBJECTIVE:\nship the report soon\n\nCRITERIA:\n- a\n- b"],
        )
        gateway = LocalGraphGateway()

        async def _go(message, request=None):
            run = RunRequest(
                message=message,
                thread_id="t1",
                target=GraphTarget(local_graph=agent),
                configurable_extra={"rubric_run": {"mode": "thread", "changes": []}},
                goal_request=request,
            )
            return [event async for event in gateway.stream_events(run)]

        await _go("", new_goal_request("create", objective="ship the report"))
        await _go(Command(resume=ACCEPT))
        state = await agent.aget_state({"configurable": {"thread_id": "t1"}})
        # The scripted grader passes at once; amend needs a goal still in work.
        await agent.aupdate_state(
            {"configurable": {"thread_id": "t1"}},
            {"_goal": {"status": "active"}},
            as_node="GoalMiddleware.before_agent",
        )
        target = state.values["_goal"]["rubric_id"]
        await _go("", new_goal_request("amend", feedback="sooner", target=target))
        events = await _go(Command(resume=ACCEPT))
        amended = next(e for e in events if e.get("change") == "amended")
        assert amended["objective"] == "ship the report soon"
        assert amended["criteria"] == "- a\n- b"
        assert amended["message"] == "sooner"

    async def test_a_cancelled_request_ends_with_one_event_and_no_text(self, tmp_path):
        from langgraph.types import Command

        create = new_goal_request("create", objective="ship the report")
        _, second = await self._events(
            tmp_path, [("", create), (Command(resume=CANCEL), None)]
        )
        assert [(e["type"], e.get("change")) for e in second] == [
            ("goal_state", "cancelled"),
            ("done", None),
        ]
        assert second[-1].get("response") in ("", None)


class TestTallQuestion:
    def test_a_question_that_ends_in_a_newline_stays_one_prompt(self):
        from EvoScientist.stream import display as display_mod

        data = {"questions": [{"question": "Pick a dataset\n", "type": "text"}]}
        out = StringIO()
        prompts: list[str] = []

        def _text(prompt, **_kwargs):
            prompts.append(prompt)
            return "the-question"

        with (
            patch.object(display_mod, "console", Console(file=out, width=100)),
            patch("questionary.text", _text),
        ):
            result = display_mod._resolve_ask_user_prompt(
                data, question_runner=lambda _question: "CIFAR-10"
            )
        assert result == {"answers": ["CIFAR-10"], "status": "answered"}
        assert prompts == ["(1/1) Pick a dataset"]
        assert "Pick a dataset" not in out.getvalue().split("Quick check-in")[-1]

    def test_a_long_question_is_printed_above_a_one_line_prompt(self):
        from EvoScientist.stream import display as display_mod

        question = (
            "Proposed acceptance criteria for the goal:\nship it\n\n- a\n- b\n\n"
            "Accept them, or type what to change."
        )
        data = {
            "questions": [
                {
                    "question": question,
                    "type": "multiple_choice",
                    "choices": [{"value": "Accept"}, {"value": "Cancel"}],
                }
            ]
        }
        out = StringIO()
        prompts: list[str] = []

        def _select(prompt, **_kwargs):
            prompts.append(prompt)
            return "the-question"

        with (
            patch.object(display_mod, "console", Console(file=out, width=100)),
            patch("questionary.select", _select),
        ):
            result = display_mod._resolve_ask_user_prompt(
                data, question_runner=lambda _question: "Accept"
            )
        assert result == {"answers": ["Accept"], "status": "answered"}
        assert prompts == ["(1/1) Accept them, or type what to change."]
        printed = out.getvalue()
        assert "Proposed acceptance criteria for the goal:" in printed
        assert "- b" in printed


class TestEventsArriveWhileWorkGoesOn:
    """The drafter runs in a nested run that is no sub-agent; its events must
    not hold the turn's own events back until the stream ends."""

    @staticmethod
    async def _stream(agent, seen: list) -> None:
        from EvoScientist.gateway import GraphTarget, RunRequest
        from EvoScientist.gateway.local import LocalGraphGateway

        run = RunRequest(
            message="",
            thread_id="t1",
            target=GraphTarget(local_graph=agent),
            configurable_extra={"rubric_run": {"mode": "thread", "changes": []}},
            goal_request=new_goal_request("create", objective="ship the report"),
        )
        async for event in LocalGraphGateway().stream_events(run):
            seen.append(event)

    @staticmethod
    async def _arrives(seen: list, change: str, *, seconds: float = 10.0) -> bool:
        for _ in range(int(seconds / 0.02)):
            if any(e.get("change") == change for e in seen):
                return True
            await asyncio.sleep(0.02)
        return False

    async def test_the_drafting_line_is_there_while_the_draft_is_written(
        self, tmp_path
    ):
        from tests.goal_fakes import cli_goal_agent

        hold = threading.Event()
        agent, _model = cli_goal_agent(tmp_path, holds={"draft": hold})
        seen: list = []
        task = asyncio.ensure_future(self._stream(agent, seen))
        try:
            assert await self._arrives(seen, "drafting")
            assert [e["type"] for e in seen] == ["goal_state"]
        finally:
            hold.set()
            await asyncio.wait_for(task, timeout=20)
        assert [e["type"] for e in seen] == ["goal_state", "ask_user", "done"]

    async def test_an_unattended_goal_shows_its_start_before_the_work_ends(
        self, tmp_path
    ):
        from langgraph.checkpoint.memory import InMemorySaver

        from EvoScientist.config import EvoScientistConfig
        from EvoScientist.EvoScientist import create_cli_agent
        from EvoScientist.paths import Workspace
        from tests.goal_fakes import AllRoles

        hold = threading.Event()
        agent = create_cli_agent(
            workspace=Workspace(tmp_path),
            config=EvoScientistConfig(
                enable_async_subagents=False,
                memory_workers_enabled=False,
                auto_mode=True,
            ),
            chat_model=AllRoles(holds={"main": hold}),
            checkpointer=InMemorySaver(),
        )
        seen: list = []
        task = asyncio.ensure_future(self._stream(agent, seen))
        try:
            # The main model has not replied yet: the goal is already shown.
            assert await self._arrives(seen, "accepted")
            assert not any(e["type"] == "text" for e in seen)
        finally:
            hold.set()
            await asyncio.wait_for(task, timeout=20)
        changes = [e["change"] for e in seen if e["type"] == "goal_state"]
        assert changes == ["drafting", "accepted", "complete"]
        # Through the real decoder: only the adoption carries the mark.
        marked = [e["change"] for e in seen if e.get("unreviewed")]
        assert marked == ["accepted"]

    async def test_the_drafters_own_output_is_never_shown(self, tmp_path):
        """A streaming model: the draft's tokens and usage stay out of the turn."""
        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk
        from langgraph.checkpoint.memory import InMemorySaver

        from EvoScientist.config import EvoScientistConfig
        from EvoScientist.EvoScientist import create_cli_agent
        from EvoScientist.paths import Workspace
        from tests.goal_fakes import AllRoles

        class _Streaming(AllRoles):
            def _stream(self, messages, stop=None, run_manager=None, **kwargs):
                message = self._generate(messages, stop=stop).generations[0].message
                text = message.content if isinstance(message.content, str) else ""
                pieces = [text[i : i + 6] for i in range(0, len(text), 6)] or [""]
                for index, piece in enumerate(pieces):
                    last = index == len(pieces) - 1
                    chunk = AIMessageChunk(
                        content=piece,
                        tool_calls=message.tool_calls if last else [],
                        usage_metadata=(
                            {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18}
                            if last
                            else None
                        ),
                    )
                    if run_manager:
                        run_manager.on_llm_new_token(
                            piece, chunk=ChatGenerationChunk(message=chunk)
                        )
                    yield ChatGenerationChunk(message=chunk)

        agent = create_cli_agent(
            workspace=Workspace(tmp_path),
            config=EvoScientistConfig(
                enable_async_subagents=False,
                memory_workers_enabled=False,
                auto_mode=True,
            ),
            chat_model=_Streaming(drafts=["- DRAFTERTEXT one\n- DRAFTERTEXT two"]),
            checkpointer=InMemorySaver(),
        )
        seen: list = []
        await self._stream(agent, seen)
        shown = "".join(
            str(e.get("content") or "")
            for e in seen
            if e["type"] in ("text", "thinking")
        )
        assert shown == "working on it"
        assert seen[-1]["response"] == "working on it"
        # Only the main agent's call is counted.
        usage = [e for e in seen if e["type"] == "usage_stats"]
        assert [(e["input_tokens"], e["output_tokens"]) for e in usage] == [(11, 7)]
        accepted = next(e for e in seen if e.get("change") == "accepted")
        assert "DRAFTERTEXT" in accepted["criteria"]


class TestOnlyTheDraftersRunIsHidden:
    """A sub-agent may have a task of its own called ``draft_goal``."""

    @staticmethod
    def _processor(registry):
        from EvoScientist.stream.emitter import StreamEventEmitter
        from EvoScientist.stream.events import _V3EventProcessor

        return _V3EventProcessor(StreamEventEmitter(), registry, None)

    @staticmethod
    def _custom(namespace, payload):
        return {
            "method": "custom",
            "params": {"namespace": list(namespace), "data": payload},
        }

    async def test_a_sub_agents_task_of_the_same_name_is_shown(self):
        from EvoScientist.stream.events import _SubagentRegistry

        registry = _SubagentRegistry()
        registry.register(("tools:1",), "research-agent")
        processor = self._processor(registry)
        retry = {
            "type": "model_retry",
            "attempt": 1,
            "wait_seconds": 1.0,
            "reason": "503",
            "call": "c1",
        }
        events = await processor.process(
            self._custom(("tools:1", "draft_goal:9"), retry)
        )
        assert [e["type"] for e in events] == ["model_retry"]
        assert events[0]["subagent"] == "research-agent"

    @pytest.mark.parametrize(
        "namespace",
        [
            ("GoalMiddleware.before_agent:1", "draft_goal:2"),
            ("GoalMiddleware.before_agent:1", "draft_goal:2", "model:3"),
        ],
    )
    async def test_the_drafters_own_run_is_dropped_without_waiting(self, namespace):
        from EvoScientist.stream.events import _SubagentRegistry

        processor = self._processor(_SubagentRegistry())
        retry = {"type": "model_retry", "attempt": 1, "wait_seconds": 1.0, "call": "c1"}
        events = await asyncio.wait_for(
            processor.process(self._custom(namespace, retry)), timeout=2
        )
        assert events == []

    @pytest.mark.parametrize(
        "namespace",
        [
            ("GoalMiddleware.before_agent:1", "draft_goal_helper:2"),
            ("OtherMiddleware.before_agent:1", "draft_goal:2"),
            ("draft_goal:2",),
        ],
    )
    async def test_a_namespace_that_only_looks_like_it_is_not_dropped(self, namespace):
        from EvoScientist.stream.events import _SubagentRegistry

        registry = _SubagentRegistry()
        registry.register(namespace[:1], "some-agent")
        processor = self._processor(registry)
        retry = {"type": "model_retry", "attempt": 1, "wait_seconds": 1.0, "call": "c1"}
        events = await processor.process(self._custom(namespace, retry))
        assert [e["type"] for e in events] == ["model_retry"]
