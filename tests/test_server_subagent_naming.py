"""Regression tests for issue #791: a sync sub-agent shown as "tools".

The server gateway names a sub-agent from the LIFECYCLE start event's
``graph_name``: LangGraph's lifecycle transformer copies the nested run's
``lc_agent_name`` there when that run is a sub-agent. The raw ``tasks``
events that carry the same name in their metadata are suppressed from the
server's event log, so a fix that reads only ``tasks`` never sees the name.

These tests use the production event shape: a lifecycle start carrying
``graph_name``, with no ``tasks`` event at all.
"""

from __future__ import annotations

import pytest

from EvoScientist.gateway.server import _ServerSubagentTracker, _SubagentRegistry
from EvoScientist.stream.emitter import StreamEventEmitter


def _lifecycle_started(namespace: list[str], *, graph_name: str | None = None) -> dict:
    data: dict = {"event": "started"}
    if graph_name is not None:
        data["graph_name"] = graph_name
    return {
        "method": "lifecycle",
        "params": {"namespace": namespace, "data": data},
    }


def _lifecycle_ended(namespace: list[str], phase: str = "completed") -> dict:
    return {
        "method": "lifecycle",
        "params": {"namespace": namespace, "data": {"event": phase}},
    }


def _task_started(namespace: list[str], *, agent_name: str | None = None) -> dict:
    data: dict = {}
    if agent_name is not None:
        data["metadata"] = {"lc_agent_name": agent_name}
    return {"method": "tasks", "params": {"namespace": namespace, "data": data}}


def _starts(events: list[dict]) -> list[str]:
    return [e["name"] for e in events if e.get("type") == "subagent_start"]


def _ends(events: list[dict]) -> list[str]:
    return [e["name"] for e in events if e.get("type") == "subagent_end"]


def _tracker() -> tuple[_ServerSubagentTracker, _SubagentRegistry]:
    registry = _SubagentRegistry()
    return _ServerSubagentTracker(
        emitter=StreamEventEmitter(), registry=registry
    ), registry


def test_a_sync_subagent_is_named_from_the_lifecycle_graph_name():
    """The production stream shape: lifecycle start with graph_name, no tasks."""
    tracker, registry = _tracker()

    events = tracker.process(
        _lifecycle_started(["tools:task-1"], graph_name="code-agent")
    )

    assert _starts(events) == ["code-agent"], events
    assert registry.resolve(("tools:task-1",)).name == "code-agent"


def test_a_subagent_without_a_graph_name_falls_back_to_the_namespace_segment():
    """No graph_name: the tool name is the best name available."""
    tracker, registry = _tracker()

    events = tracker.process(_lifecycle_started(["tools:task-1"]))

    assert _starts(events) == ["tools"], events
    assert registry.resolve(("tools:task-1",)).name == "tools"


@pytest.mark.parametrize("value", ["", "   ", None, 42, {}])
def test_a_blank_or_non_string_graph_name_falls_back(value):
    tracker, _registry = _tracker()

    events = tracker.process(_lifecycle_started(["tools:task-1"], graph_name=value))

    assert _starts(events) == ["tools"], events


def test_a_task_event_names_the_child_when_there_is_no_lifecycle_name():
    """A task event is a fallback: the server usually suppresses those."""
    tracker, registry = _tracker()

    events = tracker.process(_task_started(["tools:task-1"], agent_name="code-agent"))

    assert _starts(events) == ["code-agent"], events
    assert registry.resolve(("tools:task-1",)).name == "code-agent"


def test_the_lifecycle_name_wins_over_the_task_name():
    """The lifecycle event is what the server actually delivers."""
    tracker, registry = _tracker()

    events = tracker.process(
        _lifecycle_started(["tools:task-1"], graph_name="code-agent")
    )
    later = tracker.process(_task_started(["tools:task-1"], agent_name="other-name"))

    assert _starts(events) == ["code-agent"], events
    assert not _starts(later), later
    assert registry.resolve(("tools:task-1",)).name == "code-agent"


def test_a_late_task_event_after_the_child_ended_does_not_start_it_again():
    """A late event must not announce a start with no matching end."""
    tracker, registry = _tracker()

    tracker.process(_lifecycle_started(["tools:task-1"], graph_name="code-agent"))
    ended = tracker.process(_lifecycle_ended(["tools:task-1"]))
    late = tracker.process(_task_started(["tools:task-1"], agent_name="code-agent"))

    assert _starts(ended) == []
    assert _ends(ended) == ["code-agent"], ended
    assert not _starts(late), late
    assert not _ends(late), late
    assert registry.resolve(("tools:task-1",)).name == "code-agent"


def test_a_new_child_may_reuse_a_namespace_after_the_first_one_ended():
    """A fresh lifecycle start for the same path is a new child."""
    tracker, registry = _tracker()

    tracker.process(_lifecycle_started(["tools:task-1"], graph_name="first"))
    tracker.process(_lifecycle_ended(["tools:task-1"]))
    events = tracker.process(_lifecycle_started(["tools:task-1"], graph_name="second"))

    assert _starts(events) == ["second"], events
    assert registry.resolve(("tools:task-1",)).name == "second"


def test_a_nested_child_gets_its_own_start_and_its_own_name():
    """The parent and the nested child each get exactly one start."""
    tracker, registry = _tracker()

    parent = tracker.process(
        _lifecycle_started(["tools:task-1"], graph_name="code-agent")
    )
    child = tracker.process(
        _lifecycle_started(
            ["tools:task-1", "tools:task-2"], graph_name="research-agent"
        )
    )

    assert _starts(parent) == ["code-agent"], parent
    assert _starts(child) == ["research-agent"], child
    assert registry.resolve(("tools:task-1",)).name == "code-agent"
    assert registry.resolve(("tools:task-1", "tools:task-2")).name == "research-agent"


def test_the_registry_name_never_changes_after_the_start_was_announced():
    """The start event and the registry must agree for the whole run."""
    tracker, registry = _tracker()

    tracker.process(_lifecycle_started(["tools:task-1"], graph_name="code-agent"))
    tracker.process(_task_started(["tools:task-1"], agent_name="late-rename"))

    assert registry.resolve(("tools:task-1",)).name == "code-agent"


def test_an_ended_child_is_not_started_again_by_a_late_task_result():
    tracker, _registry = _tracker()

    tracker.process(_lifecycle_started(["tools:task-1"], graph_name="code-agent"))
    tracker.process(_lifecycle_ended(["tools:task-1"]))
    late = tracker.process(
        {
            "method": "tasks",
            "params": {"namespace": ["tools:task-1"], "data": {"result": {}}},
        }
    )

    assert not _starts(late), late
    assert not _ends(late), late
