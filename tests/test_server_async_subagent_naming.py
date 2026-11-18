"""Regression tests for the async sub-agent naming gap on the server path.

A sync sub-agent is named from the lifecycle event's ``graph_name`` (#852),
but an async sub-agent runs on its own thread, so the server stream never
carries its lifecycle events at all. The only launch the parent stream
carries is the ``start_async_task`` tool call and the ``async_tasks`` state
it writes, and the only completion is the terminal state flip and the
completion notification. The tracker names the child from those, through
the same registration rule as the sync children.

Every event shape below is built from the PRODUCTION producers:

- Tool events as langgraph emits them (``langgraph/pregel/_tools.py``):
  ``tool-started`` carries a tool_name; ``tool-finished`` and
  ``tool-error`` carry only the event, the tool call id and the result.
  The tool-finished output is a serialized Command built with the REAL
  ``_build_task_envelope`` and the REAL serde, so the wire dict is
  ``{"graph": None, "update": {"messages": [...], "async_tasks": {...}}}``.
- Updates events are built by running the REAL server normalizer
  (``normalize_updates_data`` + ``strip_interrupts_from_values``) over
  realistic upstream payloads. A single-node update arrives top-level;
  parallel tool calls arrive as ``{"value": [...]}``.
- The deepagents notification message is built with the REAL
  ``_notification_message`` and used as the raw dict it returns
  (``role`` / ``content`` / ``additional_kwargs``) -- that is the wire
  form the status pull streams. In production the notification arrives
  together with its terminal record in one update.
- The CLI "[Async tasks update]" message is run INPUT, so it never
  arrives in an updates event. It is used here only in NEGATIVE checks:
  the tracker must not end a child on it. The end signal is the terminal
  async_tasks record, which arrives only when the agent calls
  ``check_async_task``: Evo disables deepagents' automatic status-pull
  hooks, and the CLI message lets the model acknowledge without checking,
  so if the agent never checks, no end is emitted.

A record is state, not a launch event: it announces or re-arms a child
only when a run-causing tool call actually produced that run in this
turn (a launch, which announces itself, or an update that resumed a
finished task onto a new run). A check or a list merely observes the
task, so its record announces nothing -- that is how a check of an
earlier-turn task used to emit a false start. Parallel tool calls
gather their outputs in call order, so a record for an old run can
arrive after a newer run ended; the pending-run gate is what tells a
stale never-seen run from a genuine new one, and the seen set is what
tells a stale seen run from one.

Cross-turn contract (see the gateway test at the end): the gateway builds
a fresh tracker per turn, so a later turn's tracker never saw the launch
and emits only the end. No completion listener runs between turns: the end
is shown when a later turn's live stream observes the terminal record.
Between turns the CLI notifier's "[Async tasks update]" message is the
user-facing completion surface. The TUI finalizes a still-active widget
at the previous turn's end and StreamState ignores an end for an unknown
instance, so a cross-turn end is a stream event that may have no visible
widget effect.
"""

from __future__ import annotations

import json

import pytest
from deepagents.middleware.async_subagents import _notification_message
from langgraph_api.event_streaming.event_normalizers import (
    normalize_updates_data,
    strip_interrupts_from_values,
)
from langgraph_api.serde import json_dumpb, json_loads

from EvoScientist.gateway import (
    LangGraphServerGateway,
    LangGraphServerThreadStore,
    RunRequest,
)
from EvoScientist.gateway.server import _ServerSubagentTracker, _SubagentRegistry
from EvoScientist.middleware.expert_async_subagent import _build_task_envelope
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
) -> dict:
    """A lifecycle start event in the shape the transformer actually pushes.

    The trigger call id is the namespace segment's own id, as the
    transformer sends it; the cause is present only for a run a tool
    call started. (The tracker ignores the trigger call id entirely:
    the terminal payload carries none, so it can never be a guard.)
    """
    data: dict = {"event": "started"}
    if graph_name is not None:
        data["graph_name"] = graph_name
    if subagent:
        data["cause"] = {"type": "toolCall", "tool_call_id": "call-1"}
    segment = namespace[-1]
    data["trigger_call_id"] = segment.partition(":")[2] or segment
    return {
        "method": "lifecycle",
        "params": {"namespace": namespace, "data": data},
    }


def _tool_started(tool_call_id: str, input_args: dict) -> dict:
    # tool-started DOES carry tool_name (langgraph/pregel/_tools.py).
    return {
        "method": "tools",
        "params": {
            "namespace": [],
            "data": {
                "event": "tool-started",
                "tool_call_id": tool_call_id,
                "tool_name": "start_async_task",
                "input": input_args,
            },
        },
    }


def _tool_finished(
    tool_call_id: str,
    task_id: str | None,
    subagent_type: str,
    run_id: str = "run-9",
) -> dict:
    # tool-finished carries NO tool_name. On success the output is a
    # serialized Command built by the real producer; on failure it is a
    # plain string.
    if task_id is None:
        output = f"Failed to launch async subagent '{subagent_type}': boom"
    else:
        command = _build_task_envelope(
            subagent_type, task_id, run_id, tool_call_id, "Implement the experiment"
        )
        output = json_loads(json_dumpb(command))
    return {
        "method": "tools",
        "params": {
            "namespace": [],
            "data": {
                "event": "tool-finished",
                "tool_call_id": tool_call_id,
                "output": output,
            },
        },
    }


def _tool_finished_without_record(
    tool_call_id: str, task_id: str, subagent_type: str
) -> dict:
    """A Command whose update carries only the launch text, no record.

    The envelope always writes both; this is the defensive fallback for a
    launch whose record never streamed.
    """
    command = _build_task_envelope(
        subagent_type, task_id, "run-9", tool_call_id, "Implement the experiment"
    )
    output = json_loads(json_dumpb(command))
    output["update"].pop("async_tasks", None)
    return {
        "method": "tools",
        "params": {
            "namespace": [],
            "data": {
                "event": "tool-finished",
                "tool_call_id": tool_call_id,
                "output": output,
            },
        },
    }


def _tool_error(tool_call_id: str, message: str) -> dict:
    # tool-error carries NO tool_name either.
    return {
        "method": "tools",
        "params": {
            "namespace": [],
            "data": {
                "event": "tool-error",
                "tool_call_id": tool_call_id,
                "message": message,
            },
        },
    }


def _updates(upstream: dict) -> dict:
    """Run a realistic upstream updates payload through the real normalizer."""
    normalized = normalize_updates_data(upstream)
    _requirements, cleaned = strip_interrupts_from_values(normalized.get("values"))
    return {"method": "updates", "params": {"namespace": [], "data": cleaned}}


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


def _record_update(
    task_id: str, agent_name: str, status: str = "running", run_id: str = "run-9"
) -> dict:
    """An updates event carrying one async_tasks record (single node)."""
    return _updates(
        {
            "tools": {
                "async_tasks": {
                    task_id: _task_record(
                        task_id, agent_name, status=status, run_id=run_id
                    )
                }
            }
        }
    )


def _notification_wire(
    task_id: str, agent_name: str, status: str, run_id: str | None = "run-9"
) -> dict:
    """The deepagents notification message, as the producer returns it.

    ``_notification_message`` returns a plain dict (``role`` /
    ``content`` / ``additional_kwargs``); that is the wire form the
    status pull streams, not a serialized message object.
    """
    update: dict[str, object] = {
        "task_id": task_id,
        "agent_name": agent_name,
        "status": status,
    }
    if run_id is not None:
        update["run_id"] = run_id
    return _notification_message(update)  # type: ignore[arg-type]


def _notification_update(
    task_id: str, agent_name: str, status: str, run_id: str | None = "run-9"
) -> dict:
    """An updates event carrying only the deepagents notification message.

    Defensive shape: in production the notification arrives together with
    the terminal record (see ``_notification_and_record_update``); this
    pins the metadata path on its own.
    """
    return _updates(
        {
            "tools": {
                "messages": [
                    _notification_wire(task_id, agent_name, status, run_id=run_id)
                ]
            }
        }
    )


def _notification_and_record_update(
    task_id: str, agent_name: str, status: str, run_id: str = "run-9"
) -> dict:
    """An updates event carrying the notification AND its terminal record.

    This is the production shape: deepagents' status pull returns the
    notification message together with the changed record in one update.
    """
    return _updates(
        {
            "tools": {
                "async_tasks": {
                    task_id: _task_record(
                        task_id, agent_name, status=status, run_id=run_id
                    )
                },
                "messages": [
                    _notification_wire(task_id, agent_name, status, run_id=run_id)
                ],
            }
        }
    )


def _check_command_wire(
    task_id: str, agent_name: str, status: str, run_id: str, tool_call_id: str
) -> dict:
    """A check_async_task result, built with the real command builder.

    The check/update/list tools return Commands whose updates also carry
    async_tasks records, so the tracker must not mistake one for a launch.
    """
    from deepagents.middleware.async_subagents import _build_check_command

    task = _task_record(task_id, agent_name, status=status, run_id=run_id)
    result = {"status": status, "task_id": task_id}
    return json_loads(
        json_dumpb(_build_check_command(result, task, run_id, tool_call_id))
    )


def _tool_finished_raw(tool_call_id: str, output: object) -> dict:
    """A tool-finished event for an arbitrary tool result (no tool_name)."""
    return {
        "method": "tools",
        "params": {
            "namespace": [],
            "data": {
                "event": "tool-finished",
                "tool_call_id": tool_call_id,
                "output": output,
            },
        },
    }


def _tool_started_update(tool_call_id: str) -> dict:
    """A tool-started event for update_async_task (carries a tool_name)."""
    return {
        "method": "tools",
        "params": {
            "namespace": [],
            "data": {
                "event": "tool-started",
                "tool_call_id": tool_call_id,
                "tool_name": "update_async_task",
                "input": {"task_id": "async-thread-1", "message": "keep going"},
            },
        },
    }


def _update_command_wire(
    task_id: str, agent_name: str, run_id: str, tool_call_id: str
) -> dict:
    """An update_async_task result, built with the real command builder.

    The update resumes the task onto ``run_id``; its Command carries the
    record for that run, so the record that follows may re-arm on it.
    """
    from deepagents.middleware.async_subagents import _build_update_command

    tracked = _task_record(task_id, agent_name, status="success", run_id="run-9")
    return json_loads(
        json_dumpb(
            _build_update_command(
                tracked,
                {"run_id": run_id, "status": "running"},
                "interrupt",
                tool_call_id,
            )
        )
    )


def _update_finished(
    tracker: _ServerSubagentTracker,
    *,
    call_id: str = "update-1",
    task_id: str = "async-thread-1",
    agent_name: str = "code-agent",
    run_id: str = "run-10",
) -> list[dict]:
    """An update_async_task call that resumes the task onto ``run_id``."""
    events: list[dict] = []
    events += tracker.process(_tool_started_update(call_id))
    events += tracker.process(
        _tool_finished_raw(
            call_id, _update_command_wire(task_id, agent_name, run_id, call_id)
        )
    )
    return events


def _list_command_wire(
    task_id: str, agent_name: str, status: str, run_id: str, tool_call_id: str
) -> dict:
    """A list_async_tasks result, built with the real command builder.

    The list returns the live status it fetched for the task; a cached
    list can return an old run's status.
    """
    from deepagents.middleware.async_subagents import _build_list_command

    task = _task_record(task_id, agent_name, status="success", run_id="run-8")
    return json_loads(
        json_dumpb(_build_list_command([task], [(status, run_id)], tool_call_id))
    )


def _command_update(command: dict) -> dict:
    """One call's Command update, as the server streams it in a batch item."""
    return command["update"]


def _batch(items: list[dict]) -> dict:
    """One updates event carrying several calls' Command updates, in call order."""
    return _updates({"tools": items})


def _cli_update_message(
    task_id: str, agent_name: str, status: str, kind: str = "agent"
) -> dict:
    """The CLI "[Async tasks update]" message, as a serialized message.

    NEGATIVE-CHECK ONLY: this message is run input in production, so it
    never arrives in an updates event. The tracker must not end a child
    on it.
    """
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
            "tools": {
                "messages": [
                    {
                        "type": "human",
                        "data": {"content": text, "additional_kwargs": {}},
                    }
                ]
            }
        }
    )


def _launch(
    tracker: _ServerSubagentTracker,
    *,
    call_id: str = "call-1",
    name: str = "code-agent",
    task_id: str = "async-thread-1",
) -> list[dict]:
    """A successful launch: tool-started + tool-finished (Command).

    No updates record: the launch signal alone must announce the child.
    """
    events: list[dict] = []
    events += tracker.process(
        _tool_started(
            call_id,
            {"description": "Implement the experiment", "subagent_type": name},
        )
    )
    events += tracker.process(_tool_finished(call_id, task_id, name))
    return events


def _starts(events: list[dict]) -> list[str]:
    return [e["name"] for e in events if e.get("type") == "subagent_start"]


def _ends(events: list[dict]) -> list[str]:
    return [e["name"] for e in events if e.get("type") == "subagent_end"]


def _end_instances(events: list[dict]) -> list[str]:
    return [e["instance_id"] for e in events if e.get("type") == "subagent_end"]


def _start_triggers(events: list[dict]) -> list[str | None]:
    return [e.get("tool_call_id") for e in events if e.get("type") == "subagent_start"]


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
        _tool_started("call-1", {"description": "x", "subagent_type": "code-agent"})
    )
    events += tracker.process(_tool_finished("call-1", "async-thread-1", "code-agent"))

    assert _starts(events) == ["code-agent"], events
    assert events[0].get("tool_call_id") == "call-1"
    assert registry.resolve(("async", "async-thread-1")).name == "code-agent"


def test_an_async_launch_is_named_from_the_launch_text_when_the_record_is_absent():
    """Defensive fallback: the launch text names the task if the record is absent.

    The envelope always writes both, so this shape is not a production
    one; it pins the fallback for a degraded Command.
    """
    tracker, _registry = _tracker()

    tracker.process(
        _tool_started("call-1", {"description": "x", "subagent_type": "code-agent"})
    )
    events = tracker.process(
        _tool_finished_without_record("call-1", "async-thread-1", "code-agent")
    )

    assert _starts(events) == ["code-agent"], events


def test_a_record_alone_announces_nothing():
    """A record is state, not a launch event: no tool call, no child.

    The launch is announced by its tool-finished result; a record that
    arrives with no launch (or update) behind it -- a check of an
    earlier-turn task, a stale record -- announces nothing.
    """
    tracker, registry = _tracker()

    tracker.process(
        _tool_started("call-1", {"description": "x", "subagent_type": "code-agent"})
    )
    events = tracker.process(_record_update("async-thread-1", "code-agent"))

    assert not _starts(events), events
    assert registry.resolve(("async", "async-thread-1")) is None
    events = tracker.process(_tool_finished("call-1", "async-thread-1", "code-agent"))
    assert _starts(events) == ["code-agent"], events
    assert _start_triggers(events) == ["call-1"], events


def test_a_launch_is_announced_exactly_once_when_both_signals_arrive():
    """Tool result first, record second: the second is a no-op."""
    tracker, _registry = _tracker()

    events = _launch(tracker)
    events += tracker.process(_record_update("async-thread-1", "code-agent"))

    assert _starts(events) == ["code-agent"], events


def test_a_failed_launch_announces_nothing():
    """A tool error, and an error string as the tool result, both stay silent."""
    tracker, _registry = _tracker()

    events: list[dict] = []
    events += tracker.process(
        _tool_started("call-1", {"description": "x", "subagent_type": "code-agent"})
    )
    events += tracker.process(
        _tool_error("call-1", "Failed to launch async subagent 'code-agent': boom")
    )
    events += tracker.process(
        _tool_started("call-2", {"description": "x", "subagent_type": "no-such-agent"})
    )
    events += tracker.process(_tool_finished("call-2", None, "no-such-agent"))

    assert not _starts(events), events
    assert not _ends(events), events


def test_a_terminal_state_flip_ends_the_child():
    tracker, _registry = _tracker()

    _launch(tracker)
    events = tracker.process(
        _record_update("async-thread-1", "code-agent", status="success")
    )

    assert _ends(events) == ["code-agent"], events
    assert not _ends(
        tracker.process(
            _record_update("async-thread-1", "code-agent", status="success")
        )
    )


def test_the_cli_update_message_ends_nothing():
    """The CLI message is run input, not an updates payload: it ends nothing.

    The completion it reports is observed through the terminal record the
    task check writes after the message wakes the agent.
    """
    tracker, _registry = _tracker()

    _launch(tracker)
    events = tracker.process(
        _cli_update_message("async-thread-1", "code-agent", "success")
    )

    assert not _ends(events), events
    events = tracker.process(
        _record_update("async-thread-1", "code-agent", status="success")
    )
    assert _ends(events) == ["code-agent"], events


def test_the_deepagents_notification_ends_the_child():
    tracker, _registry = _tracker()

    _launch(tracker)
    events = tracker.process(
        _notification_update("async-thread-1", "code-agent", "success")
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
        _record_update("async-thread-2", "research-agent", status="error")
    )

    assert _ends(ends) == ["research-agent"], ends
    assert not _ends(tracker.finish())


def test_same_name_concurrent_launches_keep_their_own_triggers():
    """Two children of one type, finishing out of order, keep their own call ids.

    The trigger is correlated by task id from the launch Command, never
    guessed by agent name: a FIFO guess would hand call-1's trigger to
    the child call-2 launched.
    """
    tracker, _registry = _tracker()

    events: list[dict] = []
    events += tracker.process(
        _tool_started("call-1", {"description": "x", "subagent_type": "code-agent"})
    )
    events += tracker.process(
        _tool_started("call-2", {"description": "x", "subagent_type": "code-agent"})
    )
    # Out-of-order finishes: call-2's tool returns first.
    events += tracker.process(_tool_finished("call-2", "async-thread-2", "code-agent"))
    events += tracker.process(_tool_finished("call-1", "async-thread-1", "code-agent"))

    assert _starts(events) == ["code-agent", "code-agent"], events
    assert _start_triggers(events) == ["call-2", "call-1"], events

    ends = tracker.process(
        _record_update("async-thread-1", "code-agent", status="success")
    )
    assert _end_instances(ends) == ["async:async-thread-1"], ends
    ends = tracker.process(
        _record_update("async-thread-2", "code-agent", status="success")
    )
    assert _end_instances(ends) == ["async:async-thread-2"], ends


def test_a_check_command_result_is_not_a_launch():
    """check_async_task returns a Command with async_tasks too: it is not a launch.

    Only a start_async_task call the tracker saw start can announce a child.
    """
    tracker, _registry = _tracker()

    events = tracker.process(
        _tool_finished_raw(
            "check-1",
            _check_command_wire(
                "async-thread-1", "code-agent", "success", "run-9", "check-1"
            ),
        )
    )

    assert not _starts(events), events
    assert not _ends(events), events


def test_a_check_of_an_active_child_keeps_its_launch_call_id():
    """A check result must not overwrite the real launch's call id.

    The launch call id is carried on the start event so the CLI places
    the child under its own task line. It is placement, not a guard:
    no async completion signal carries a tool call id to compare
    against, so there is no trigger matching on an async end -- the
    run-id checks are what keep a stale completion from ending the
    child (see the stale-completion tests below).
    """
    tracker, _registry = _tracker()

    _launch(tracker)
    events = tracker.process(
        _tool_finished_raw(
            "check-1",
            _check_command_wire(
                "async-thread-1", "code-agent", "running", "run-9", "check-1"
            ),
        )
    )

    assert not _starts(events), events
    assert not _ends(events), events
    assert tracker._active[("async", "async-thread-1")] == "code-agent"
    assert tracker._async_launch_calls["async-thread-1"] == "call-1"


def test_a_check_update_does_not_announce_a_child_from_an_earlier_turn():
    """A check of a running task from an earlier turn is not a launch.

    The check's tool-finished is ignored, and so is the async_tasks
    update its Command writes: a record announces nothing without a
    launch (or an update that resumed the task) behind it.
    """
    tracker, _registry = _tracker()

    command = _check_command_wire(
        "async-thread-1", "code-agent", "running", "run-9", "check-1"
    )
    finished = tracker.process(_tool_finished_raw("check-1", command))
    assert not _starts(finished), finished
    assert not _ends(finished), finished
    update = normalize_updates_data({"tools": command["update"]})["values"]
    events = tracker.process(
        {"method": "updates", "params": {"namespace": [], "data": update}}
    )

    assert not _starts(events), events
    assert not _ends(events), events


def test_a_stale_record_for_a_never_seen_run_does_not_revive():
    """A check that observed an old run cannot revive an ended child.

    Parallel tool calls gather their outputs in call order, so a slow
    check can stream a record for a run the tracker never saw after a
    newer run already ended. The record has no update behind it, so it
    announces nothing -- only a run a tool call actually produced in
    this turn can revive the child.
    """
    tracker, _registry = _tracker()

    _launch(tracker)
    tracker.process(_record_update("async-thread-1", "code-agent", status="success"))
    command = _check_command_wire(
        "async-thread-1", "code-agent", "running", "run-8", "check-1"
    )
    tracker.process(_tool_finished_raw("check-1", command))
    update = normalize_updates_data({"tools": command["update"]})["values"]
    events = tracker.process(
        {"method": "updates", "params": {"namespace": [], "data": update}}
    )

    assert not _starts(events), events
    assert not _ends(events), events
    assert tracker._ended_async_runs[("async", "async-thread-1")] == "run-9"


def test_a_stale_running_record_does_not_move_the_current_run_backwards():
    """run-9 -> run-10 -> stale run-9 running -> run-9 success: only run-10 ends it."""
    tracker, _registry = _tracker()

    _launch(tracker)
    # The rotation onto run-10 is produced by an update in this turn.
    _update_finished(tracker, run_id="run-10")
    tracker.process(
        _record_update(
            "async-thread-1", "code-agent", status="running", run_id="run-10"
        )
    )
    stale_running = tracker.process(
        _record_update("async-thread-1", "code-agent", status="running", run_id="run-9")
    )
    assert not _starts(stale_running), stale_running
    assert not _ends(stale_running), stale_running
    stale_end = tracker.process(
        _record_update("async-thread-1", "code-agent", status="success", run_id="run-9")
    )
    assert not _ends(stale_end), stale_end
    ends = tracker.process(
        _record_update(
            "async-thread-1", "code-agent", status="success", run_id="run-10"
        )
    )
    assert _ends(ends) == ["code-agent"], ends
    assert not _ends(
        tracker.process(
            _record_update(
                "async-thread-1", "code-agent", status="success", run_id="run-10"
            )
        )
    )


def test_a_late_record_for_a_seen_run_does_not_revive_an_ended_child():
    """A run the tracker already saw cannot revive a child that ended on a later run."""
    tracker, _registry = _tracker()

    _launch(tracker)
    # The rotation onto run-10 is produced by an update in this turn.
    _update_finished(tracker, run_id="run-10")
    tracker.process(
        _record_update(
            "async-thread-1", "code-agent", status="running", run_id="run-10"
        )
    )
    tracker.process(
        _record_update(
            "async-thread-1", "code-agent", status="success", run_id="run-10"
        )
    )
    events = tracker.process(
        _record_update("async-thread-1", "code-agent", status="running", run_id="run-9")
    )

    assert not _starts(events), events
    assert not _ends(events), events


def test_a_seen_old_terminal_record_after_a_newer_run_ends_nothing():
    """run-8 running, run-9 running, run-9 error, run-8 success: one end, not two.

    The terminal record for a run already seen is stale even though it
    names a different run than the tombstone: it must not re-arm the
    child only to end it a second time and rewrite the tombstone to the
    old run.
    """
    tracker, _registry = _tracker()

    _launch(tracker)
    # The rotation onto run-10 is produced by an update in this turn.
    _update_finished(tracker, run_id="run-10")
    tracker.process(
        _record_update(
            "async-thread-1", "code-agent", status="running", run_id="run-10"
        )
    )
    ends = tracker.process(
        _record_update("async-thread-1", "code-agent", status="error", run_id="run-10")
    )
    assert _ends(ends) == ["code-agent"], ends
    stale = tracker.process(
        _record_update("async-thread-1", "code-agent", status="success", run_id="run-9")
    )

    assert not _starts(stale), stale
    assert not _ends(stale), stale
    assert tracker._ended_async_runs[("async", "async-thread-1")] == "run-10"


def test_a_seen_run_does_not_revive_even_with_an_update_pending_run():
    """An update naming a run already seen cannot revive an ended child.

    The pending run lets a record re-arm only for a run never seen
    before; a run the tracker already saw is stale even when a tool
    call produced it in this turn.
    """
    tracker, _registry = _tracker()

    _launch(tracker)
    # The rotation onto run-10 is produced by an update in this turn.
    _update_finished(tracker, run_id="run-10")
    tracker.process(
        _record_update(
            "async-thread-1", "code-agent", status="running", run_id="run-10"
        )
    )
    tracker.process(
        _record_update(
            "async-thread-1", "code-agent", status="success", run_id="run-10"
        )
    )
    _update_finished(tracker, run_id="run-9")
    events = tracker.process(
        _record_update("async-thread-1", "code-agent", status="running", run_id="run-9")
    )

    assert not _starts(events), events
    assert not _ends(events), events


def test_an_unrelated_record_announces_nothing_and_the_launch_keeps_its_trigger():
    """A record for another task of the same name announces nothing on its own.

    Only a launch (or an update that resumed the task) names a run, so
    an unrelated record cannot take a pending launch's call id -- or
    announce a child at all.
    """
    tracker, _registry = _tracker()

    tracker.process(
        _tool_started("launch-new", {"description": "x", "subagent_type": "code-agent"})
    )
    old = tracker.process(_record_update("async-thread-old", "code-agent"))

    assert not _starts(old), old
    new = _launch(tracker, call_id="launch-new", task_id="async-thread-new")
    assert _starts(new) == ["code-agent"], new
    assert _start_triggers(new) == ["launch-new"], new


def test_a_run_id_less_notification_after_a_revival_does_not_end_the_child():
    """A notification with no run id cannot say which run it is for, so it must not
    end a child that ended and was revived onto a newer run."""
    tracker, _registry = _tracker()

    _launch(tracker)
    tracker.process(_record_update("async-thread-1", "code-agent", status="success"))
    _update_finished(tracker, run_id="run-10")
    events = tracker.process(
        _record_update(
            "async-thread-1", "code-agent", status="running", run_id="run-10"
        )
    )
    assert _starts(events) == ["code-agent"], events
    events = tracker.process(
        _notification_update("async-thread-1", "code-agent", "success", run_id=None)
    )

    assert not _ends(events), events
    ends = tracker.process(
        _record_update(
            "async-thread-1", "code-agent", status="success", run_id="run-10"
        )
    )
    assert _ends(ends) == ["code-agent"], ends


def test_a_run_id_less_end_after_an_unknown_run_revival_does_not_end_the_child():
    """A tombstone with no run id is still a tombstone.

    The child ended on a signal with no run id, so the tombstone is
    None; after an update revives it onto run-9, a late run-id-less
    signal must not close it. None must not read as "no tombstone".
    """
    tracker, _registry = _tracker()

    events = tracker.process(
        _notification_update("async-thread-1", "code-agent", "success", run_id=None)
    )
    assert _ends(events) == ["code-agent"], events
    assert tracker._ended_async_runs[("async", "async-thread-1")] is None
    _update_finished(tracker, run_id="run-9")
    revived = tracker.process(
        _record_update("async-thread-1", "code-agent", status="running", run_id="run-9")
    )
    assert _starts(revived) == ["code-agent"], revived
    events = tracker.process(
        _notification_update("async-thread-1", "code-agent", "success", run_id=None)
    )

    assert not _ends(events), events
    ends = tracker.process(
        _record_update("async-thread-1", "code-agent", status="success", run_id="run-9")
    )
    assert _ends(ends) == ["code-agent"], ends


def test_the_notification_arrives_with_its_terminal_record():
    """Production shape: the notification and the terminal record come in one update."""
    tracker, _registry = _tracker()

    _launch(tracker)
    events = tracker.process(
        _notification_and_record_update("async-thread-1", "code-agent", "success")
    )

    assert _ends(events) == ["code-agent"], events
    assert _starts(events) == [], events
    assert not _ends(
        tracker.process(
            _notification_and_record_update("async-thread-1", "code-agent", "success")
        )
    )


def test_batched_updates_name_both_parallel_launches():
    """Parallel tool calls arrive as one updates event with {"value": [...]}.

    The tool-finished results stream first and announce both launches;
    the batch that follows carries each launch's message and its record
    as separate entries (the real ToolNode splits them that way), and
    adds no second start for either. The messages are ToolMessages
    from the real launch envelope, not HumanMessages.
    """
    tracker, _registry = _tracker()

    events: list[dict] = []
    events += tracker.process(
        _tool_started("call-1", {"description": "x", "subagent_type": "code-agent"})
    )
    events += tracker.process(
        _tool_started("call-2", {"description": "x", "subagent_type": "research-agent"})
    )
    events += tracker.process(_tool_finished("call-1", "async-thread-1", "code-agent"))
    events += tracker.process(
        _tool_finished("call-2", "async-thread-2", "research-agent")
    )
    assert _starts(events) == ["code-agent", "research-agent"], events

    def _launch_messages(
        subagent_type: str, task_id: str, tool_call_id: str
    ) -> list[dict]:
        command = _build_task_envelope(
            subagent_type, task_id, "run-9", tool_call_id, "Implement the experiment"
        )
        return json_loads(json_dumpb(command))["update"]["messages"]

    batch = tracker.process(
        _updates(
            {
                "tools": [
                    {
                        "messages": _launch_messages(
                            "code-agent", "async-thread-1", "call-1"
                        )
                    },
                    {
                        "async_tasks": {
                            "async-thread-1": _task_record(
                                "async-thread-1", "code-agent"
                            )
                        }
                    },
                    {
                        "messages": _launch_messages(
                            "research-agent", "async-thread-2", "call-2"
                        )
                    },
                    {
                        "async_tasks": {
                            "async-thread-2": _task_record(
                                "async-thread-2", "research-agent"
                            )
                        }
                    },
                ]
            }
        )
    )

    assert not _starts(batch), batch
    assert not _ends(batch), batch


def test_a_never_seen_stale_run_in_a_batch_does_not_move_the_child_backwards():
    """B2: an update's run-10 record followed by a stale check's never-seen
    run-9 record in one parallel batch must leave the child on run-10.

    Run ids carry no ordering, so the never-seen run-9 looks exactly like
    a genuine new run; only the pending-run gate tells them apart.
    """
    tracker, _registry = _tracker()

    # Launch on run-8.
    events: list[dict] = []
    events += tracker.process(
        _tool_started("call-1", {"description": "x", "subagent_type": "code-agent"})
    )
    events += tracker.process(
        _tool_finished("call-1", "async-thread-1", "code-agent", run_id="run-8")
    )
    assert _starts(events) == ["code-agent"], events
    # An update resumes the task onto run-10.
    _update_finished(tracker, run_id="run-10")
    update_cmd = _update_command_wire(
        "async-thread-1", "code-agent", "run-10", "update-1"
    )
    check_cmd = _check_command_wire(
        "async-thread-1", "code-agent", "running", "run-9", "check-1"
    )
    tracker.process(_tool_finished_raw("check-1", check_cmd))
    # One parallel batch, in call order: the update's record, then the
    # stale check's record for a run this turn never saw.
    batch = tracker.process(
        _batch([_command_update(update_cmd), _command_update(check_cmd)])
    )
    assert not _starts(batch), batch
    assert not _ends(batch), batch
    assert tracker._async_run_ids[("async", "async-thread-1")] == "run-10"
    ends = tracker.process(
        _record_update(
            "async-thread-1", "code-agent", status="success", run_id="run-10"
        )
    )
    assert _ends(ends) == ["code-agent"], ends


def test_a_stale_terminal_record_in_a_batch_does_not_end_the_child_twice():
    """B3b: update run-10/running, check run-10/error, cached list
    run-9/success in one parallel batch: one start, one end, and the
    tombstone stays on run-10.

    The list's terminal record names a run no tool call in this turn
    produced, so it must not re-arm the ended child only to end it a
    second time.
    """
    tracker, _registry = _tracker()

    # Launch on run-8, end it on run-8.
    events: list[dict] = []
    events += tracker.process(
        _tool_started("call-1", {"description": "x", "subagent_type": "code-agent"})
    )
    events += tracker.process(
        _tool_finished("call-1", "async-thread-1", "code-agent", run_id="run-8")
    )
    events += tracker.process(
        _record_update("async-thread-1", "code-agent", status="success", run_id="run-8")
    )
    assert _ends(events) == ["code-agent"], events
    assert tracker._ended_async_runs[("async", "async-thread-1")] == "run-8"
    # An update resumes the task onto run-10.
    _update_finished(tracker, call_id="update-1", run_id="run-10")
    update_cmd = _update_command_wire(
        "async-thread-1", "code-agent", "run-10", "update-1"
    )
    check_cmd = _check_command_wire(
        "async-thread-1", "code-agent", "error", "run-10", "check-1"
    )
    # The list read the state before the update landed, so it returns
    # the cached run-9/success.
    list_cmd = _list_command_wire(
        "async-thread-1", "code-agent", "success", "run-9", "list-1"
    )
    # The tool results stream first; the check and list results are
    # no-ops for the tracker.
    tracker.process(_tool_finished_raw("check-1", check_cmd))
    tracker.process(_tool_finished_raw("list-1", list_cmd))
    # One parallel batch, in call order.
    batch = tracker.process(
        _batch(
            [
                _command_update(update_cmd),
                _command_update(check_cmd),
                _command_update(list_cmd),
            ]
        )
    )
    assert _starts(batch) == ["code-agent"], batch
    assert _ends(batch) == ["code-agent"], batch
    assert tracker._ended_async_runs[("async", "async-thread-1")] == "run-10"


def test_ending_an_older_run_keeps_a_parallel_updates_pending_run():
    """Update-resume: check run-9/success then update run-10/running in one
    parallel batch. Ending run-9 must not clear the pending run-10 the
    parallel update recorded, so the update's record revives the child."""
    tracker, _registry = _tracker()

    _launch(tracker)  # run-9
    # An update resumes the task onto run-10.
    _update_finished(tracker, call_id="update-1", run_id="run-10")
    update_cmd = _update_command_wire(
        "async-thread-1", "code-agent", "run-10", "update-1"
    )
    check_cmd = _check_command_wire(
        "async-thread-1", "code-agent", "success", "run-9", "check-1"
    )
    tracker.process(_tool_finished_raw("check-1", check_cmd))
    # One parallel batch, in call order: the check's record first, then
    # the update's.
    batch = tracker.process(
        _batch([_command_update(check_cmd), _command_update(update_cmd)])
    )
    assert _ends(batch) == ["code-agent"], batch
    assert _starts(batch) == ["code-agent"], batch
    # The revival leaves the child active on run-10: the end below must
    # close that child, not silently re-arm an ended path.
    assert tracker._active.get(("async", "async-thread-1")) == "code-agent"
    ends = tracker.process(
        _record_update(
            "async-thread-1", "code-agent", status="success", run_id="run-10"
        )
    )
    assert _ends(ends) == ["code-agent"], ends


def test_a_stale_completion_for_an_older_run_does_not_end_the_child():
    """A completion naming the run the child has moved on from is stale."""
    tracker, _registry = _tracker()

    _launch(tracker)
    # The rotation onto run-10 is produced by an update in this turn.
    _update_finished(tracker, run_id="run-10")
    tracker.process(
        _record_update(
            "async-thread-1", "code-agent", status="running", run_id="run-10"
        )
    )
    stale = tracker.process(
        _notification_update("async-thread-1", "code-agent", "success", run_id="run-9")
    )
    assert not _ends(stale), stale
    current = tracker.process(
        _notification_update("async-thread-1", "code-agent", "success", run_id="run-10")
    )
    assert _ends(current) == ["code-agent"], current


def test_a_stale_terminal_record_does_not_end_a_newer_run():
    """A terminal record is only the state as last streamed: its run id is checked."""
    tracker, _registry = _tracker()

    _launch(tracker)
    # The rotation onto run-10 is produced by an update in this turn.
    _update_finished(tracker, run_id="run-10")
    tracker.process(
        _record_update(
            "async-thread-1", "code-agent", status="running", run_id="run-10"
        )
    )
    stale = tracker.process(
        _record_update("async-thread-1", "code-agent", status="success", run_id="run-9")
    )

    assert not _ends(stale), stale
    current = tracker.process(
        _record_update(
            "async-thread-1", "code-agent", status="success", run_id="run-10"
        )
    )
    assert _ends(current) == ["code-agent"], current


def test_the_launch_records_its_run_id():
    """A terminal record for an older run than the launch run ends nothing."""
    tracker, _registry = _tracker()

    _launch(tracker)  # the launch run is run-9
    events = tracker.process(
        _record_update("async-thread-1", "code-agent", status="success", run_id="run-8")
    )

    assert not _ends(events), events
    assert not _starts(events), events


def test_a_late_record_for_the_ended_run_does_not_re_announce_it():
    """A late non-terminal record for the run the child ended on is not a revival."""
    tracker, _registry = _tracker()

    _launch(tracker)
    tracker.process(_record_update("async-thread-1", "code-agent", status="success"))
    events = tracker.process(
        _record_update("async-thread-1", "code-agent", status="running", run_id="run-9")
    )

    assert not _starts(events), events
    assert not _ends(events), events


def test_a_cli_message_after_a_revival_does_not_end_the_revived_child():
    """A run-id-less signal cannot say which run it is for, so it must not
    end a child that ended and was revived onto a newer run."""
    tracker, _registry = _tracker()

    _launch(tracker)
    tracker.process(_record_update("async-thread-1", "code-agent", status="success"))
    _update_finished(tracker, run_id="run-10")
    events = tracker.process(
        _record_update(
            "async-thread-1", "code-agent", status="running", run_id="run-10"
        )
    )
    assert _starts(events) == ["code-agent"], events
    events = tracker.process(
        _cli_update_message("async-thread-1", "code-agent", "success")
    )

    assert not _ends(events), events
    ends = tracker.process(
        _record_update(
            "async-thread-1", "code-agent", status="success", run_id="run-10"
        )
    )
    assert _ends(ends) == ["code-agent"], ends


def test_a_run_rotation_keeps_the_child_until_its_own_completion():
    tracker, _registry = _tracker()

    _launch(tracker)
    # The rotation onto run-10 is produced by an update in this turn.
    _update_finished(tracker, run_id="run-10")
    rotated = tracker.process(
        _record_update(
            "async-thread-1", "code-agent", status="running", run_id="run-10"
        )
    )

    assert not _starts(rotated), rotated
    assert not _ends(rotated), rotated
    ends = tracker.process(
        _record_update(
            "async-thread-1", "code-agent", status="success", run_id="run-10"
        )
    )
    assert _ends(ends) == ["code-agent"], ends


def test_finish_ends_sync_children_but_not_running_async_ones():
    """An async child outlives the parent turn; its end comes later."""
    tracker, _registry = _tracker()

    tracker.process(_lifecycle_started(["tools:task-1"], graph_name="code-agent"))
    # The async child is named differently from the sync child on
    # purpose: the assertion must prove WHICH child ended, not just
    # that one did.
    _launch(tracker, name="research-agent")
    events = tracker.finish()

    assert _ends(events) == ["code-agent"], events


def test_a_follow_up_run_revives_a_finished_task():
    """An update that resumes a finished task onto a new run re-arms it.

    The update's result names the new run, so the record that follows
    may re-arm on it. A record alone -- no update behind it -- revives
    nothing.
    """
    tracker, _registry = _tracker()

    _launch(tracker)
    tracker.process(_record_update("async-thread-1", "code-agent", status="success"))
    events = tracker.process(
        _record_update(
            "async-thread-1", "code-agent", status="running", run_id="run-10"
        )
    )
    assert not _starts(events), events
    _update_finished(tracker, run_id="run-10")
    events = tracker.process(
        _record_update(
            "async-thread-1", "code-agent", status="running", run_id="run-10"
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
        _record_update("async-thread-1", "code-agent", status="success")
    )

    assert _ends(events) == ["code-agent"], events


def test_a_waiting_task_is_not_ended():
    """Waiting is not terminal: the task is still alive on its own thread."""
    tracker, _registry = _tracker()

    _launch(tracker)
    events = tracker.process(
        _record_update("async-thread-1", "code-agent", status="waiting")
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
                {
                    "description": "Implement the experiment",
                    "subagent_type": "code-agent",
                },
            ),
            _tool_finished("call-1", "async-thread-1", "code-agent"),
            _record_update("async-thread-1", "code-agent"),
        ]
    )

    assert [e["name"] for e in events if e.get("type") == "subagent_start"] == [
        "code-agent"
    ]
    assert [e["name"] for e in events if e.get("type") == "subagent_end"] == []


async def test_the_gateway_yields_the_end_on_the_completion_turn():
    events = await _gateway_run(
        [
            _notification_update("async-thread-1", "code-agent", "success"),
            _record_update("async-thread-1", "code-agent", status="success"),
        ]
    )

    assert [e["name"] for e in events if e.get("type") == "subagent_end"] == [
        "code-agent"
    ]
    assert [e["name"] for e in events if e.get("type") == "subagent_start"] == []


async def test_the_gateway_yields_the_start_then_the_end_across_turns():
    """Cross-turn contract: turn 1 launches, turn 2 observes the completion.

    The gateway builds a fresh tracker per turn, so turn 2's tracker never
    saw the launch and emits only the end. No completion listener runs
    between turns: the end is shown when a later turn's live stream
    observes the terminal record (the CLI notifier is the user-facing
    completion surface between turns).
    """
    turn_1 = await _gateway_run(
        [
            _tool_started(
                "call-1",
                {
                    "description": "Implement the experiment",
                    "subagent_type": "code-agent",
                },
            ),
            _tool_finished("call-1", "async-thread-1", "code-agent"),
        ]
    )
    assert [e["name"] for e in turn_1 if e.get("type") == "subagent_start"] == [
        "code-agent"
    ]
    assert [e["name"] for e in turn_1 if e.get("type") == "subagent_end"] == []

    turn_2 = await _gateway_run(
        [
            _notification_update("async-thread-1", "code-agent", "success"),
            _record_update("async-thread-1", "code-agent", status="success"),
        ]
    )
    assert [e["name"] for e in turn_2 if e.get("type") == "subagent_end"] == [
        "code-agent"
    ]
    assert [e["name"] for e in turn_2 if e.get("type") == "subagent_start"] == []


async def test_the_gateway_announces_nothing_for_a_failed_launch():
    events = await _gateway_run(
        [
            _tool_started(
                "call-1", {"description": "x", "subagent_type": "code-agent"}
            ),
            _tool_error("call-1", "Failed to launch async subagent 'code-agent': boom"),
        ]
    )

    assert [e["name"] for e in events if e.get("type") == "subagent_start"] == []
    assert [e["name"] for e in events if e.get("type") == "subagent_end"] == []
