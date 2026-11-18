"""Regression tests for the async sub-agent naming gap on the server path.

A sync sub-agent is named from the lifecycle event's ``graph_name`` (#852),
but an async sub-agent runs on its own thread, so the server stream never
carries its lifecycle events at all. The only launch the parent stream
carries is the ``start_async_task`` tool call and the ``async_tasks`` state
it writes, and the only completion is the terminal state flip and the
completion notification. The tracker names the child from those, through
the same registration rule as the sync children.

These tests use the production event shapes: the tool events as the v3
processor reads them, the async_tasks record as ``_build_task_envelope``
writes it, the CLI "[Async tasks update]" message as ``format_batch_message``
builds it, and the deepagents notification as ``_notification_message``
labels it.
"""

from __future__ import annotations

import json

import pytest

from EvoScientist.gateway import (
    LangGraphServerGateway,
    LangGraphServerThreadStore,
    RunRequest,
)
from EvoScientist.gateway.server import _ServerSubagentTracker, _SubagentRegistry
from EvoScientist.stream.emitter import StreamEventEmitter
from tests.fakes.common import (
    FakeLangGraphClient,
    FakeLangGraphThreadsClient,
    FakeLangGraphThreadStream,
)


@pytest.fixture(autouse=True)
def _fast_thread_replay_skip_timeouts(monkeypatch):
    """Keep ``_ThreadEvents.skip_replay`` off the wall clock in the suite.

    Production defaults stay in ``EvoScientist.gateway.server``; skip_replay
    reads them at call time when the constructor left timeouts unset.
    """
    monkeypatch.setattr("EvoScientist.gateway.server._REPLAY_QUIET_SECONDS", 0.01)
    monkeypatch.setattr("EvoScientist.gateway.server._REPLAY_FIRST_EVENT_SECONDS", 0.05)
    monkeypatch.setattr("EvoScientist.gateway.server._REPLAY_MAX_SECONDS", 0.5)


def _lifecycle_started(
    namespace: list[str],
    *,
    graph_name: str | None = None,
    subagent: bool = True,
    trigger_call_id: str | None = None,
) -> dict:
    data: dict = {"event": "started"}
    if graph_name is not None:
        data["graph_name"] = graph_name
    if subagent:
        data["cause"] = {"type": "toolCall", "tool_call_id": "call-1"}
    data["trigger_call_id"] = trigger_call_id or f"trigger-{len(namespace)}"
    return {
        "method": "lifecycle",
        "params": {"namespace": namespace, "data": data},
    }


def _tool_started(tool_call_id: str, tool_name: str, input_args: dict) -> dict:
    return {
        "method": "tools",
        "params": {
            "namespace": [],
            "data": {
                "event": "tool-started",
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "input": input_args,
            },
        },
    }


def _tool_finished(tool_call_id: str, tool_name: str, output: object) -> dict:
    return {
        "method": "tools",
        "params": {
            "namespace": [],
            "data": {
                "event": "tool-finished",
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "output": output,
            },
        },
    }


def _tool_error(tool_call_id: str, tool_name: str, message: str) -> dict:
    return {
        "method": "tools",
        "params": {
            "namespace": [],
            "data": {
                "event": "tool-error",
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "message": message,
            },
        },
    }


def _updates(data: dict) -> dict:
    return {"method": "updates", "params": {"namespace": [], "data": data}}


def _task_record(
    task_id: str,
    agent_name: str,
    status: str = "running",
    run_id: str = "run-9",
) -> dict:
    return {
        "task_id": task_id,
        "agent_name": agent_name,
        "thread_id": task_id,
        "run_id": run_id,
        "status": status,
        "created_at": "2026-11-18T00:00:00Z",
        "last_checked_at": "2026-11-18T00:00:00Z",
        "last_updated_at": "2026-11-18T00:00:00Z",
        "description": "Implement the experiment",
    }


def _launch(
    tracker: _ServerSubagentTracker,
    *,
    call_id: str = "call-1",
    name: str = "code-agent",
    task_id: str = "async-thread-1",
) -> list[dict]:
    """A successful launch as the parent stream delivers it."""
    events: list[dict] = []
    events += tracker.process(
        _tool_started(
            call_id,
            "start_async_task",
            {"description": "Implement the experiment", "subagent_type": name},
        )
    )
    events += tracker.process(
        _tool_finished(
            call_id,
            "start_async_task",
            f"Launched async subagent. task_id: {task_id}",
        )
    )
    events += tracker.process(
        _updates({"async_tasks": {task_id: _task_record(task_id, name)}})
    )
    return events


def _cli_update_message(
    task_id: str, agent_name: str, status: str, kind: str = "agent"
) -> dict:
    text = (
        "[Async tasks update]\n"
        + json.dumps(
            {"agent": agent_name, "kind": kind, "status": status, "task_id": task_id}
        )
        + "\n(Signal only — fetch full result via check_async_task (sub-agents)"
        " if relevant to the current step, else acknowledge & continue.)"
    )
    return _updates(
        {
            "messages": [
                {"type": "human", "data": {"content": text, "additional_kwargs": {}}}
            ]
        }
    )


def _deepagents_notification_message(
    task_id: str, agent_name: str, status: str, run_id: str = "run-9"
) -> dict:
    event = {
        "task_id": task_id,
        "agent_name": agent_name,
        "status": status,
        "run_id": run_id,
    }
    text = (
        f"<task-notification>\n<task-id>{task_id}</task-id>\n"
        f"<agent-name>{agent_name}</agent-name>\n<status>{status}</status>\n"
    )
    return _updates(
        {
            "messages": [
                {
                    "type": "human",
                    "data": {
                        "content": text,
                        "additional_kwargs": {
                            "lc_source": "async_subagent",
                            "lc_data": dict(event),
                            "deepagents_notification": dict(event),
                        },
                    },
                }
            ]
        }
    )


def _starts(events: list[dict]) -> list[str]:
    return [e["name"] for e in events if e.get("type") == "subagent_start"]


def _ends(events: list[dict]) -> list[str]:
    return [e["name"] for e in events if e.get("type") == "subagent_end"]


def _tracker() -> tuple[_ServerSubagentTracker, _SubagentRegistry]:
    registry = _SubagentRegistry()
    return _ServerSubagentTracker(
        emitter=StreamEventEmitter(), registry=registry
    ), registry


async def _gateway_run(events: list[dict]) -> list[dict]:
    stream = FakeLangGraphThreadStream("abc12345", events=events)
    threads = FakeLangGraphThreadsClient(
        threads=[{"thread_id": "abc12345", "metadata": {"graph_id": "EvoScientist"}}],
        states={"abc12345": {"values": {}}},
        streams={"abc12345": stream},
    )
    gateway = LangGraphServerGateway(
        LangGraphServerThreadStore(client=FakeLangGraphClient(threads))
    )
    return [
        event
        async for event in gateway.stream_events(
            RunRequest(message="hi", thread_id="abc12345")
        )
    ]


def test_an_async_launch_is_named_from_the_tool_finished_result():
    """The launch result names the task; the name comes from the call args."""
    tracker, registry = _tracker()

    events: list[dict] = []
    events += tracker.process(
        _tool_started(
            "call-1",
            "start_async_task",
            {"description": "x", "subagent_type": "code-agent"},
        )
    )
    events += tracker.process(
        _tool_finished(
            "call-1",
            "start_async_task",
            "Launched async subagent. task_id: async-thread-1",
        )
    )

    assert _starts(events) == ["code-agent"], events
    assert events[0].get("tool_call_id") == "call-1"
    assert registry.resolve(("async", "async-thread-1")).name == "code-agent"


def test_an_async_launch_is_named_from_a_serialized_tool_message():
    """The tool result may arrive as a serialized ToolMessage dict."""
    tracker, _registry = _tracker()

    tracker.process(
        _tool_started(
            "call-1",
            "start_async_task",
            {"description": "x", "subagent_type": "code-agent"},
        )
    )
    events = tracker.process(
        _tool_finished(
            "call-1",
            "start_async_task",
            {
                "type": "tool",
                "content": "Launched async subagent. task_id: async-thread-1",
                "tool_call_id": "call-1",
                "name": "start_async_task",
            },
        )
    )

    assert _starts(events) == ["code-agent"], events


def test_an_async_launch_is_named_from_the_async_tasks_record():
    """The state record names the task when the tool result is not the signal."""
    tracker, registry = _tracker()

    tracker.process(
        _tool_started(
            "call-1",
            "start_async_task",
            {"description": "x", "subagent_type": "code-agent"},
        )
    )
    events = tracker.process(
        _updates(
            {
                "async_tasks": {
                    "async-thread-1": _task_record("async-thread-1", "code-agent")
                }
            }
        )
    )

    assert _starts(events) == ["code-agent"], events
    assert events[0].get("tool_call_id") == "call-1"
    assert registry.resolve(("async", "async-thread-1")).name == "code-agent"


def test_a_launch_is_announced_exactly_once_when_both_signals_arrive():
    """Tool result first, record second: the second is a no-op."""
    tracker, _registry = _tracker()

    events = _launch(tracker)

    assert _starts(events) == ["code-agent"], events


def test_a_failed_launch_announces_nothing():
    """A tool error, and an error string as the tool result, both stay silent."""
    tracker, _registry = _tracker()

    events: list[dict] = []
    events += tracker.process(
        _tool_started(
            "call-1",
            "start_async_task",
            {"description": "x", "subagent_type": "code-agent"},
        )
    )
    events += tracker.process(
        _tool_error(
            "call-1",
            "start_async_task",
            "Failed to launch async subagent 'code-agent': boom",
        )
    )
    events += tracker.process(
        _tool_started(
            "call-2",
            "start_async_task",
            {"description": "x", "subagent_type": "no-such-agent"},
        )
    )
    events += tracker.process(
        _tool_finished(
            "call-2", "start_async_task", "Unknown subagent type 'no-such-agent'"
        )
    )

    assert not _starts(events), events
    assert not _ends(events), events


def test_a_terminal_state_flip_ends_the_child():
    tracker, _registry = _tracker()

    _launch(tracker)
    events = tracker.process(
        _updates(
            {
                "async_tasks": {
                    "async-thread-1": _task_record(
                        "async-thread-1", "code-agent", status="success"
                    )
                }
            }
        )
    )

    assert _ends(events) == ["code-agent"], events
    assert not _ends(
        tracker.process(
            _updates(
                {
                    "async_tasks": {
                        "async-thread-1": _task_record(
                            "async-thread-1", "code-agent", status="success"
                        )
                    }
                }
            )
        )
    )


def test_the_cli_update_message_ends_the_child():
    tracker, _registry = _tracker()

    _launch(tracker)
    events = tracker.process(
        _cli_update_message("async-thread-1", "code-agent", "success")
    )

    assert _ends(events) == ["code-agent"], events


def test_the_deepagents_notification_ends_the_child():
    tracker, _registry = _tracker()

    _launch(tracker)
    events = tracker.process(
        _deepagents_notification_message("async-thread-1", "code-agent", "success")
    )

    assert _ends(events) == ["code-agent"], events


def test_two_concurrent_tasks_end_independently():
    tracker, _registry = _tracker()

    events: list[dict] = []
    events += _launch(
        tracker, call_id="call-1", name="code-agent", task_id="async-thread-1"
    )
    events += _launch(
        tracker, call_id="call-2", name="research-agent", task_id="async-thread-2"
    )
    assert _starts(events) == ["code-agent", "research-agent"], events

    ends = tracker.process(
        _updates(
            {
                "async_tasks": {
                    "async-thread-2": _task_record(
                        "async-thread-2", "research-agent", status="error"
                    )
                }
            }
        )
    )

    assert _ends(ends) == ["research-agent"], ends
    assert not _ends(tracker.finish())


def test_a_stale_completion_for_an_older_run_does_not_end_the_child():
    """A completion naming the run the child has moved on from is stale."""
    tracker, _registry = _tracker()

    _launch(tracker)
    tracker.process(
        _updates(
            {
                "async_tasks": {
                    "async-thread-1": _task_record(
                        "async-thread-1",
                        "code-agent",
                        status="running",
                        run_id="run-10",
                    )
                }
            }
        )
    )
    stale = tracker.process(
        _deepagents_notification_message(
            "async-thread-1", "code-agent", "success", run_id="run-9"
        )
    )

    assert not _ends(stale), stale
    current = tracker.process(
        _deepagents_notification_message(
            "async-thread-1", "code-agent", "success", run_id="run-10"
        )
    )
    assert _ends(current) == ["code-agent"], current


def test_a_run_rotation_keeps_the_child_until_its_own_completion():
    tracker, _registry = _tracker()

    _launch(tracker)
    rotated = tracker.process(
        _updates(
            {
                "async_tasks": {
                    "async-thread-1": _task_record(
                        "async-thread-1",
                        "code-agent",
                        status="running",
                        run_id="run-10",
                    )
                }
            }
        )
    )

    assert not _starts(rotated), rotated
    assert not _ends(rotated), rotated
    ends = tracker.process(
        _updates(
            {
                "async_tasks": {
                    "async-thread-1": _task_record(
                        "async-thread-1",
                        "code-agent",
                        status="success",
                        run_id="run-10",
                    )
                }
            }
        )
    )
    assert _ends(ends) == ["code-agent"], ends


def test_finish_ends_sync_children_but_not_running_async_ones():
    """An async child outlives the parent turn; its end comes later."""
    tracker, _registry = _tracker()

    tracker.process(_lifecycle_started(["tools:task-1"], graph_name="code-agent"))
    _launch(tracker)
    events = tracker.finish()

    assert _ends(events) == ["code-agent"], events


def test_a_follow_up_run_revives_a_finished_task():
    tracker, _registry = _tracker()

    _launch(tracker)
    tracker.process(
        _updates(
            {
                "async_tasks": {
                    "async-thread-1": _task_record(
                        "async-thread-1", "code-agent", status="success"
                    )
                }
            }
        )
    )
    events = tracker.process(
        _updates(
            {
                "async_tasks": {
                    "async-thread-1": _task_record(
                        "async-thread-1",
                        "code-agent",
                        status="running",
                        run_id="run-10",
                    )
                }
            }
        )
    )

    assert _starts(events) == ["code-agent"], events


def test_a_plain_nested_graph_is_not_an_async_child():
    """The sync fallback is untouched: a plain nested graph keeps the tool name."""
    tracker, registry = _tracker()

    events = tracker.process(
        _lifecycle_started(["tools:task-1"], graph_name="tools", subagent=False)
    )

    assert _starts(events) == ["tools"], events
    assert registry.resolve(("async", "tools:task-1")) is None


def test_a_completion_for_an_unseen_launch_still_ends_it():
    """The launch was in an earlier turn; the start was announced there."""
    tracker, _registry = _tracker()

    events = tracker.process(
        _updates(
            {
                "async_tasks": {
                    "async-thread-1": _task_record(
                        "async-thread-1", "code-agent", status="success"
                    )
                }
            }
        )
    )

    assert _ends(events) == ["code-agent"], events


def test_a_waiting_task_is_not_ended():
    """Waiting is not terminal: the task is still alive on its own thread."""
    tracker, _registry = _tracker()

    _launch(tracker)
    events = tracker.process(
        _updates(
            {
                "async_tasks": {
                    "async-thread-1": _task_record(
                        "async-thread-1", "code-agent", status="waiting"
                    )
                }
            }
        )
    )

    assert not _ends(events), events


def test_a_background_process_entry_is_not_an_async_child():
    """The update message batches bg-process entries too; they are not children."""
    tracker, _registry = _tracker()

    _launch(tracker)
    events = tracker.process(
        _cli_update_message(
            "async-thread-1", "code-agent", "success", kind="bg-process"
        )
    )

    assert not _ends(events), events


async def test_the_gateway_yields_the_start_and_no_end_for_a_launch():
    events = await _gateway_run(
        [
            _tool_started(
                "call-1",
                "start_async_task",
                {
                    "description": "Implement the experiment",
                    "subagent_type": "code-agent",
                },
            ),
            _tool_finished(
                "call-1",
                "start_async_task",
                "Launched async subagent. task_id: async-thread-1",
            ),
            _updates(
                {
                    "async_tasks": {
                        "async-thread-1": _task_record("async-thread-1", "code-agent")
                    }
                }
            ),
        ]
    )

    assert [e["name"] for e in events if e.get("type") == "subagent_start"] == [
        "code-agent"
    ]
    assert [e["name"] for e in events if e.get("type") == "subagent_end"] == []


async def test_the_gateway_yields_the_end_on_the_completion_turn():
    events = await _gateway_run(
        [
            _deepagents_notification_message("async-thread-1", "code-agent", "success"),
            _updates(
                {
                    "async_tasks": {
                        "async-thread-1": _task_record(
                            "async-thread-1", "code-agent", status="success"
                        )
                    }
                }
            ),
        ]
    )

    assert [e["name"] for e in events if e.get("type") == "subagent_end"] == [
        "code-agent"
    ]
    assert [e["name"] for e in events if e.get("type") == "subagent_start"] == []


async def test_the_gateway_announces_nothing_for_a_failed_launch():
    events = await _gateway_run(
        [
            _tool_started(
                "call-1",
                "start_async_task",
                {"description": "x", "subagent_type": "code-agent"},
            ),
            _tool_error(
                "call-1",
                "start_async_task",
                "Failed to launch async subagent 'code-agent': boom",
            ),
        ]
    )

    assert [e["name"] for e in events if e.get("type") == "subagent_start"] == []
    assert [e["name"] for e in events if e.get("type") == "subagent_end"] == []
