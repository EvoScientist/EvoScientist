"""Regression tests for issue #791: a sync sub-agent shown as "tools".

The server gateway names a sub-agent from the LIFECYCLE start event's
``graph_name``: LangGraph's lifecycle transformer copies the nested run's
``lc_agent_name`` there when that run is a sub-agent. The raw ``tasks``
events that carry the same name in their metadata are suppressed from the
server's event log, so a fix that reads only ``tasks`` never sees the name.

The shapes here are the ones the real transformer pushes, read off
``langgraph.stream.transformers.LifecycleTransformer``:

* ``_on_started`` pushes ``event``, ``namespace``, ``graph_name`` when the
  nested run has one, the trigger call id, and ``cause`` only when the
  transformer saw the tool call that started the run;
* ``_on_terminal`` pushes ``event`` and ``namespace`` and nothing else --
  no trigger call id, no cause.

Two tests at the bottom run the REAL transformer over a REAL agent run, so
the payload shapes above come from the transformer itself: a sub-agent
started by a ``task`` tool call, and an unnamed agent run inside a middleware
hook (the goal drafter), which inherits its parent's name and must not be
announced as a sub-agent under that name.
"""

from __future__ import annotations

import pytest

from EvoScientist.gateway.server import _ServerSubagentTracker, _SubagentRegistry
from EvoScientist.stream.emitter import StreamEventEmitter


def _subagent_cause(call_id: str = "call-LLM-1") -> dict:
    """The cause the transformer attaches to a run a tool call started."""
    return {"type": "toolCall", "tool_call_id": call_id}


def _lifecycle_started(
    namespace: list[str],
    *,
    graph_name: str | None = None,
    cause: dict | None = None,
) -> dict:
    """A lifecycle start event in the shape the transformer actually pushes.

    The trigger call id is the namespace segment's own id, as the transformer
    sends it; the cause is present only for a run a tool call started.
    """
    data: dict = {"event": "started"}
    if graph_name is not None:
        data["graph_name"] = graph_name
    segment = namespace[-1]
    data["trigger_call_id"] = segment.partition(":")[2] or segment
    if cause is not None:
        data["cause"] = cause
    return {
        "method": "lifecycle",
        "params": {"namespace": namespace, "data": data},
    }


def _lifecycle_ended(namespace: list[str], phase: str = "completed") -> dict:
    """A terminal event: the event and the namespace, and nothing else."""
    return {
        "method": "lifecycle",
        "params": {"namespace": namespace, "data": {"event": phase}},
    }


def _starts(events: list[dict]) -> list[str]:
    return [e["name"] for e in events if e.get("type") == "subagent_start"]


def _ends(events: list[dict]) -> list[str]:
    return [e["name"] for e in events if e.get("type") == "subagent_end"]


def _tool_call_ids(events: list[dict]) -> list[str]:
    return [
        e.get("tool_call_id", "") for e in events if e.get("type") == "subagent_start"
    ]


def _tracker() -> tuple[_ServerSubagentTracker, _SubagentRegistry]:
    registry = _SubagentRegistry()
    return _ServerSubagentTracker(
        emitter=StreamEventEmitter(), registry=registry
    ), registry


def test_a_sync_subagent_is_named_from_the_lifecycle_graph_name():
    """The production stream shape: a lifecycle start that carries a cause."""
    tracker, registry = _tracker()

    events = tracker.process(
        _lifecycle_started(
            ["tools:task-1"], graph_name="code-agent", cause=_subagent_cause()
        )
    )

    assert _starts(events) == ["code-agent"], events
    assert registry.resolve(("tools:task-1",)).name == "code-agent"


def test_the_tool_call_id_is_the_llm_tool_call_that_started_the_child():
    """The child is announced under its own ``task`` line.

    The cause carries the tool call the model made, so the CLI can place the
    sub-agent under the ``task`` call in the transcript, the way the local
    gateway does.
    """
    tracker, _registry = _tracker()

    events = tracker.process(
        _lifecycle_started(
            ["tools:task-1"],
            graph_name="code-agent",
            cause=_subagent_cause("call_LLM_1"),
        )
    )

    assert _tool_call_ids(events) == ["call_LLM_1"], events


def test_a_subagent_without_a_graph_name_falls_back_to_the_namespace_segment():
    """No graph_name: the tool name is the best name available."""
    tracker, registry = _tracker()

    events = tracker.process(
        _lifecycle_started(["tools:task-1"], cause=_subagent_cause())
    )

    assert _starts(events) == ["tools"], events
    assert registry.resolve(("tools:task-1",)).name == "tools"


@pytest.mark.parametrize("value", ["", "   ", None, 42, {}])
def test_a_blank_or_non_string_graph_name_falls_back(value):
    tracker, _registry = _tracker()

    events = tracker.process(
        _lifecycle_started(["tools:task-1"], graph_name=value, cause=_subagent_cause())
    )

    assert _starts(events) == ["tools"], events


def test_a_nested_child_gets_its_own_start_and_its_own_name():
    """The parent and the nested child each get exactly one start."""
    tracker, registry = _tracker()

    parent = tracker.process(
        _lifecycle_started(
            ["tools:task-1"], graph_name="code-agent", cause=_subagent_cause()
        )
    )
    child = tracker.process(
        _lifecycle_started(
            ["tools:task-1", "tools:task-2"],
            graph_name="research-agent",
            cause=_subagent_cause("call-LLM-2"),
        )
    )

    assert _starts(parent) == ["code-agent"], parent
    assert _starts(child) == ["research-agent"], child
    assert registry.resolve(("tools:task-1",)).name == "code-agent"
    assert registry.resolve(("tools:task-1", "tools:task-2")).name == "research-agent"


def test_the_registry_name_never_changes_after_the_start_was_announced():
    """The start event and the registry must agree for the whole run."""
    tracker, registry = _tracker()

    tracker.process(
        _lifecycle_started(
            ["tools:task-1"], graph_name="code-agent", cause=_subagent_cause()
        )
    )
    tracker.process(
        _lifecycle_started(
            ["tools:task-1"], graph_name="late-rename", cause=_subagent_cause()
        )
    )

    assert registry.resolve(("tools:task-1",)).name == "code-agent"


def test_a_lifecycle_name_matching_the_namespace_is_still_used():
    """A real agent may legitimately share the tool node's name."""
    tracker, registry = _tracker()

    events = tracker.process(
        _lifecycle_started(
            ["tools:task-1"], graph_name="tools", cause=_subagent_cause()
        )
    )

    assert _starts(events) == ["tools"], events
    assert registry.resolve(("tools:task-1",)).name == "tools"


def test_a_non_subagent_nested_run_keeps_the_tool_name():
    """A run no tool call started is not a sub-agent.

    The transformer copies any ``lc_agent_name`` the nested run carries into
    ``graph_name``, including one inherited from the parent, so the name
    counts only when the cause says a tool call started the run. Without a
    cause the tracker keeps the namespace segment -- the name it announced
    before this change -- rather than treating the run as a named sub-agent.
    """
    tracker, registry = _tracker()

    events = tracker.process(_lifecycle_started(["tools:task-1"], graph_name="tools"))

    assert _starts(events) == ["tools"], events
    assert registry.resolve(("tools:task-1",)).name == "tools"


def test_a_late_completion_for_a_never_started_path_emits_nothing():
    tracker, _registry = _tracker()

    events = tracker.process(_lifecycle_ended(["tools:task-1"]))

    assert not events, events


def test_nested_depth_three_announces_each_level_once():
    tracker, _registry = _tracker()
    paths = [
        ["tools:task-1"],
        ["tools:task-1", "tools:task-2"],
        ["tools:task-1", "tools:task-2", "tools:task-3"],
    ]
    names = ["one", "two", "three"]

    events = [
        event
        for path, name in zip(paths, names, strict=True)
        for event in tracker.process(
            _lifecycle_started(path, graph_name=name, cause=_subagent_cause())
        )
    ]

    assert _starts(events) == names, events


def _late_output_event(namespace: list[str]) -> dict:
    """An output event that arrives after the child already ended."""
    return {
        "method": "messages",
        "params": {"namespace": namespace, "data": {"event": "messages"}},
    }


def test_a_late_output_event_does_not_restart_an_ended_child():
    """A late event for an ended child must not announce a second start.

    The child already ended, so a late output event must not announce it
    again -- and must not overwrite the registry's name either, or the
    child the user saw would be replaced by a nameless one.
    """
    tracker, registry = _tracker()

    tracker.process(
        _lifecycle_started(
            ["tools:task-1"], graph_name="code-agent", cause=_subagent_cause()
        )
    )
    tracker.process(_lifecycle_ended(["tools:task-1"]))

    late = tracker.process(_late_output_event(["tools:task-1"]))

    assert not _starts(late), late
    assert registry.resolve(("tools:task-1",)).name == "code-agent"


def test_a_start_after_the_child_ended_announces_nothing():
    """The path stays ended: a namespace is per run, so it is never reused.

    The transformer's namespaces carry a fresh id per nested run, so a second
    child never reuses one. A path that ended therefore stays ended, and a
    late start for it is refused.
    """
    tracker, _registry = _tracker()

    tracker.process(
        _lifecycle_started(
            ["tools:task-1"], graph_name="first", cause=_subagent_cause()
        )
    )
    tracker.process(_lifecycle_ended(["tools:task-1"]))

    late = tracker.process(
        _lifecycle_started(
            ["tools:task-1"], graph_name="second", cause=_subagent_cause()
        )
    )

    assert not _starts(late), late


# --------------------------------------------------------------------------
# The real thing: the transformer's own output over a real agent run.
# --------------------------------------------------------------------------


def _real_lifecycle_payloads(agent) -> list[dict]:
    """The lifecycle payloads the REAL transformer pushes for a REAL run."""
    from langgraph.stream.transformers import LifecycleTransformer

    transformer = LifecycleTransformer()
    channel = getattr(transformer, "_channel", None)
    if channel is None:  # pragma: no cover - the internals moved
        pytest.skip("the lifecycle transformer's channel is not reachable")
    pushed: list[dict] = []
    original = channel.push
    channel.push = pushed.append
    try:
        for namespace, _mode, data in agent.stream(
            {"messages": [_human("hi")]}, stream_mode=["tasks"], subgraphs=True
        ):
            transformer.process(
                {
                    "method": "tasks",
                    "params": {"namespace": list(namespace), "data": data},
                }
            )
    finally:
        channel.push = original
    return pushed


def _human(text: str):
    from langchain_core.messages import HumanMessage

    return HumanMessage(text)


def _track(payloads: list[dict]):
    tracker, registry = _tracker()
    events: list[dict] = []
    for payload in payloads:
        events.extend(
            tracker.process(
                {
                    "method": "lifecycle",
                    "params": {"namespace": payload["namespace"], "data": payload},
                }
            )
        )
    return events, registry


def test_a_real_task_call_announces_the_subagent_under_its_own_name():
    """A real deepagents ``task`` call, through the real transformer.

    The tracker announces ``code-agent`` -- the name the main stream shows
    as ``tools`` -- and the tool call id is the one the model made, so the
    CLI can place the child under its ``task`` line.
    """
    from deepagents import create_deep_agent
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    class Fake(GenericFakeChatModel):
        def bind_tools(self, tools, **kw):
            return self

    outer = Fake(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "task",
                            "id": "call_LLM_1",
                            "args": {
                                "description": "do it",
                                "subagent_type": "code-agent",
                            },
                        }
                    ],
                ),
                AIMessage(content="final"),
            ]
        )
    )
    inner = Fake(messages=iter([AIMessage(content="sub done")]))
    agent = create_deep_agent(
        model=outer,
        name="EvoScientist",
        subagents=[
            {
                "name": "code-agent",
                "description": "d",
                "system_prompt": "s",
                "model": inner,
            }
        ],
    )

    events, _registry = _track(_real_lifecycle_payloads(agent))

    assert _starts(events) == ["code-agent"], events
    assert _tool_call_ids(events) == ["call_LLM_1"], events


def test_an_unnamed_agent_in_a_hook_is_not_announced_under_its_parents_name():
    """An unnamed agent inside a middleware hook inherits its parent's name.

    The goal drafter is such a run: it announces itself as ``EvoScientist``,
    the parent's name, which the main stream shows as ``Drafter.before_agent``.
    No tool call started it, so the tracker should announce it under the
    hook's namespace segment.
    """
    from itertools import cycle

    from langchain.agents import create_agent
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    def fake():
        return GenericFakeChatModel(messages=cycle([AIMessage(content="ok")]))

    inner = create_agent(model=fake(), tools=[])

    class Drafter(AgentMiddleware):
        def before_agent(self, state, runtime):
            inner.invoke({"messages": [_human("draft")]})
            return None

    outer = create_agent(
        model=fake(), tools=[], middleware=[Drafter()], name="EvoScientist"
    )

    payloads = _real_lifecycle_payloads(outer)
    starts = [payload for payload in payloads if payload["event"] == "started"]
    assert len(starts) == 1, payloads
    assert starts[0]["graph_name"] == "EvoScientist"
    assert "cause" not in starts[0]

    events, _registry = _track(payloads)

    assert _starts(events) == ["Drafter.before_agent"], events
