"""Tests for async sub-agent auto-notification."""

import queue
from unittest.mock import MagicMock

import pytest

from EvoScientist.cli import async_notifier
from EvoScientist.cli.async_notifier import (
    dedup_notifications,
    drain_notifications,
    format_batch_message,
    format_notification_lines,
)
from EvoScientist.gateway import GraphTarget
from tests.fakes import FakeGraphGateway


def test_notification_dataclass_fields():
    n = async_notifier.AsyncTaskNotification(
        task_id="tid-1",
        agent_name="writing-agent",
        status="success",
        received_at="2026-05-06T12:00:00Z",
    )
    assert n.task_id == "tid-1"
    assert n.status == "success"


def test_notification_queue_is_module_level_fifo():
    n1 = async_notifier.AsyncTaskNotification("a", "x", "success", "")
    n2 = async_notifier.AsyncTaskNotification("b", "x", "success", "")
    async_notifier._notification_queue.put(n1)
    async_notifier._notification_queue.put(n2)
    assert async_notifier._notification_queue.get_nowait().task_id == "a"
    assert async_notifier._notification_queue.get_nowait().task_id == "b"


def _drain_queue(q):
    items = []
    while True:
        try:
            items.append(q.get_nowait())
        except queue.Empty:
            return items


async def test_read_async_tasks_from_gateway_reads_state_values():
    gateway = FakeGraphGateway(
        state_values={
            "async_tasks": {
                "task-1": {"status": "success"},
            }
        }
    )

    tasks = await async_notifier.read_async_tasks_from_gateway(
        gateway,
        GraphTarget(local_graph=MagicMock()),
        "tid",
    )

    assert tasks == {"task-1": {"status": "success"}}


# ============================================================================
# Tests for drain_notifications, dedup_notifications, format_batch_message
# ============================================================================


def test_format_notification_lines_returns_decorated_block():
    """Output is: divider with 'Agent' inset, body lines, plain bottom divider."""
    notifs = [
        async_notifier.AsyncTaskNotification("t1", "writing-agent", "success", "", ""),
        async_notifier.AsyncTaskNotification("t2", "data-agent", "error", "", ""),
        async_notifier.AsyncTaskNotification("t3", "code-agent", "cancelled", "", ""),
    ]
    lines = format_notification_lines(notifs)
    # top divider (with title) + 3 body lines + bottom divider = 5
    assert len(lines) == 5
    # Top divider — open-right frame with ornaments: "╭── ✦ Agent ✦ ─────"
    top_text, top_style = lines[0]
    assert "Agent" in top_text
    assert "✦" in top_text
    assert top_text.startswith("╭")
    assert top_text.endswith("─")  # open right side
    assert top_style == "dim"
    # Body lines (indented, "-agent" suffix stripped)
    text1, style1 = lines[1]
    text2, style2 = lines[2]
    text3, style3 = lines[3]
    assert text1.startswith("     ")
    assert "writing" in text1
    assert "writing-agent" not in text1
    assert "success" in text1
    assert "✔" in text1
    assert style1.startswith("#")
    assert " data " in text2
    assert "data-agent" not in text2
    assert "error" in text2
    assert "✗" in text2
    assert style2 == "red"
    assert " code " in text3
    assert "code-agent" not in text3
    assert "cancelled" in text3
    assert "⚠" in text3
    assert style3 == "yellow"
    # Bottom divider — open-right frame, same width as top
    bot_text, bot_style = lines[4]
    assert "Agent" not in bot_text
    assert bot_text.startswith("╰")
    assert bot_text.endswith("─")
    assert len(bot_text) == len(top_text)
    assert bot_style == "dim"


def test_format_notification_lines_empty_returns_empty():
    """format_notification_lines returns an empty list for no notifications."""
    lines = format_notification_lines([])
    assert lines == []


def test_format_notification_lines_renders_prompt_when_provided():
    """When prompt is set, the body line shows `Task: <prompt preview>`."""
    notifs = [
        async_notifier.AsyncTaskNotification(
            task_id="019dfe2f-aaaa",
            agent_name="writing-agent",
            status="success",
            received_at="",
            prompt="请用中文写一段关于量子叠加的简短介绍",
        ),
    ]
    lines = format_notification_lines(notifs)
    # top divider (with title) + 1 body + bottom divider = 3
    assert len(lines) == 3
    body_text, _style = lines[1]
    assert "Task:" in body_text
    assert "量子叠加" in body_text
    assert "writing" in body_text
    assert "writing-agent" not in body_text
    assert "success" in body_text


def test_format_notification_lines_truncates_long_prompt():
    """Prompts longer than 60 chars get truncated with an ellipsis."""
    long_prompt = "x" * 200
    notifs = [
        async_notifier.AsyncTaskNotification(
            task_id="t1",
            agent_name="agent",
            status="success",
            received_at="",
            prompt=long_prompt,
        ),
    ]
    # Body line is at index 1: [top divider with title, body, bottom divider]
    body_text = format_notification_lines(notifs)[1][0]
    assert "…" in body_text
    assert "x" * 200 not in body_text


def test_format_notification_lines_collapses_newlines_in_prompt():
    """Multi-line prompts collapse to single line for the visual."""
    notifs = [
        async_notifier.AsyncTaskNotification(
            task_id="t1",
            agent_name="agent",
            status="success",
            received_at="",
            prompt="line one\nline two\nline three",
        ),
    ]
    body_text = format_notification_lines(notifs)[1][0]
    assert "\n" not in body_text
    assert "line one line two" in body_text


def test_format_notification_lines_falls_back_to_task_id_when_no_prompt():
    """Without a prompt, fall back to the short task_id."""
    notifs = [
        async_notifier.AsyncTaskNotification(
            "019dfe2f-821a-7d43-ac5b-6bb8781be5cf",
            "writing-agent",
            "success",
            "",
            "",  # no prompt
        ),
    ]
    body_text = format_notification_lines(notifs)[1][0]
    assert "019dfe2f" in body_text
    assert "e5cf" in body_text
    assert "Task:" not in body_text


def test_format_notification_lines_timeout_uses_warning_icon():
    """Timeout and interrupted statuses get the warning icon and yellow style."""
    for status in ("timeout", "interrupted"):
        notifs = [
            async_notifier.AsyncTaskNotification("t", "some-agent", status, "", "")
        ]
        lines = format_notification_lines(notifs)
        # top divider with title + 1 body + bottom divider = 3
        assert len(lines) == 3
        body_text, body_style = lines[1]
        assert "⚠" in body_text
        assert body_style == "yellow"
        assert status in body_text


def test_drain_returns_all_pending_and_empties_queue():
    """drain_notifications pulls every pending notification and empties queue."""
    # Add three notifications
    for tid in ("a", "b", "c"):
        async_notifier._notification_queue.put(
            async_notifier.AsyncTaskNotification(tid, "x", "success", "", "")
        )

    drained = drain_notifications()
    assert [n.task_id for n in drained] == ["a", "b", "c"]
    assert async_notifier._notification_queue.empty()


def test_dedup_skips_tasks_already_checked_after_terminal():
    """dedup_notifications skips tasks with terminal status and last_checked_at >= last_updated_at."""
    async_tasks: async_notifier.AsyncTasksState = {
        "a": {
            "status": "success",
            "last_checked_at": "2026-05-06T12:01:00Z",
            "last_updated_at": "2026-05-06T12:00:00Z",
        },  # already known → skip
        "b": {
            "status": "success",
            "last_checked_at": "2026-05-06T12:00:00Z",
            "last_updated_at": "2026-05-06T12:00:30Z",
        },  # checked stale → keep
        "c": {
            "status": "running",
            "last_checked_at": "",
            "last_updated_at": "",
        },  # not terminal → keep
    }
    notifs = [
        async_notifier.AsyncTaskNotification("a", "x", "success", "", ""),
        async_notifier.AsyncTaskNotification("b", "x", "success", "", ""),
        async_notifier.AsyncTaskNotification(
            "d", "x", "success", "", ""
        ),  # not in map → keep
    ]
    survivors = dedup_notifications(notifs, async_tasks)
    assert {n.task_id for n in survivors} == {"b", "d"}


def test_format_batch_message_single_notification():
    """format_batch_message produces compact JSON for a single notification."""
    notifs = [
        async_notifier.AsyncTaskNotification(
            task_id="tid-1",
            agent_name="writing-agent",
            status="success",
            received_at="2026-05-07T12:00:00Z",
            prompt="Done writing.",
        )
    ]
    msg = format_batch_message(notifs)
    assert msg.startswith("[Async tasks update]")
    # The task line is valid JSON with the expected fields.
    task_line = msg.splitlines()[1]
    obj = __import__("json").loads(task_line)
    assert obj["agent"] == "writing-agent"
    assert obj["task_id"] == "tid-1"
    assert obj["status"] == "success"


def test_format_batch_message_multiple():
    """format_batch_message handles multiple notifications as separate JSON lines."""
    notifs = [
        async_notifier.AsyncTaskNotification(
            task_id="t1",
            agent_name="writing-agent",
            status="success",
            received_at="2026-05-07T12:00:00Z",
            prompt="A",
        ),
        async_notifier.AsyncTaskNotification(
            task_id="t2",
            agent_name="data-analysis-agent",
            status="error",
            received_at="2026-05-07T12:00:01Z",
            prompt="B",
        ),
    ]
    msg = format_batch_message(notifs)
    lines = msg.splitlines()
    assert lines[0] == "[Async tasks update]"
    obj1 = __import__("json").loads(lines[1])
    obj2 = __import__("json").loads(lines[2])
    assert obj1 == {
        "agent": "writing-agent",
        "kind": "agent",
        "status": "success",
        "task_id": "t1",
    }
    assert obj2 == {
        "agent": "data-analysis-agent",
        "kind": "agent",
        "status": "error",
        "task_id": "t2",
    }
    assert "check_async_task" in msg.lower()  # hint to LLM


def test_format_batch_message_unknown_kind_uses_fallback_hint():
    """A batch with only unrecognized kinds must not emit an empty hint join."""
    notifs = [
        async_notifier.AsyncTaskNotification(
            task_id="t1",
            agent_name="x",
            status="success",
            received_at="2026-05-07T12:00:00Z",
            prompt="A",
            kind="weird",
        )
    ]
    msg = format_batch_message(notifs)
    assert "the appropriate status tool" in msg
    assert "via  " not in msg  # no empty " or ".join(hints)


def test_dedup_preserves_order():
    """dedup_notifications preserves the original order of notifications."""
    notifs = [
        async_notifier.AsyncTaskNotification("a", "x", "success", "", ""),
        async_notifier.AsyncTaskNotification("b", "x", "success", "", ""),
    ]
    survivors = dedup_notifications(notifs, async_tasks={})
    assert [n.task_id for n in survivors] == ["a", "b"]


# ============================================================================
# Tests for consume_notifications (integration path)
# ============================================================================


async def test_consume_notifications_calls_runner_with_batched_message():
    """When notifications arrive and agent is idle, consume_notifications fires
    the supplied async runner once with the formatted batch message and notifs list."""
    from EvoScientist.cli import async_notifier as an

    # Set up two pending notifications, no dedup match
    an._notification_queue.put(an.AsyncTaskNotification("t1", "wA", "success", "", ""))
    an._notification_queue.put(an.AsyncTaskNotification("t2", "wB", "success", "", ""))

    captured: dict = {}

    async def fake_runner(text: str, notifs: list) -> None:
        captured["text"] = text
        captured["notifs"] = notifs

    async def fake_state_reader() -> dict:
        return {}  # no dedup info

    await an.consume_notifications(fake_runner, fake_state_reader)
    assert "wA" in captured["text"]
    assert "wB" in captured["text"]
    assert len(captured["notifs"]) == 2


async def test_consume_notifications_no_op_when_queue_empty():
    from EvoScientist.cli import async_notifier as an

    called = False

    async def fake_runner(text: str, notifs: list):
        nonlocal called
        called = True

    async def fake_state_reader():
        return {}

    await an.consume_notifications(fake_runner, fake_state_reader)
    assert called is False


# ============================================================================
# Tests for TUI consumer reentry guard (_notification_consuming flag)
# Exercises Fix 2: the flag prevents two overlapping consume coroutines from
# both eventually calling _inject_notification_tui (and thus _run_turn).
# ============================================================================


async def test_notification_consuming_flag_prevents_reentry():
    """The _notification_consuming guard prevents two overlapping consumers.

    Verifies the flag contract used by _consume_notifications_tui:
    - The flag is checked before scheduling a new consumer.
    - The flag is cleared in a try/finally so exceptions don't freeze it.

    We model the guard using a dict (avoids nonlocal-in-nested-scope issues)
    and run three scenarios sequentially:
    1. Normal: flag cleared after first consume finishes → second can run.
    2. Blocked: flag pre-set to True → guarded_consume bails out immediately.
    3. Exception path: runner raises → flag is still cleared by finally.
    """
    from EvoScientist.cli import async_notifier as an

    state = {"inject_count": 0, "consuming": False}

    async def counting_runner(text: str, notifs: list) -> None:
        state["inject_count"] += 1

    async def fake_state_reader() -> dict:
        return {}

    async def guarded_consume(notif):
        """Mirror the TUI pattern: check flag, set it, run with try/finally."""
        if state["consuming"]:
            return  # blocked
        state["consuming"] = True
        try:
            an._notification_queue.put(notif)
            await an.consume_notifications(counting_runner, fake_state_reader)
        finally:
            state["consuming"] = False

    n1 = an.AsyncTaskNotification("g1", "writing-agent", "success", "", "")
    n2 = an.AsyncTaskNotification("g2", "data-agent", "success", "", "")

    # Scenario 1: normal flow — flag cleared, second consumer runs fine.
    await guarded_consume(n1)
    assert state["inject_count"] == 1
    assert state["consuming"] is False  # finally ran

    state["inject_count"] = 0
    await guarded_consume(n2)
    assert state["inject_count"] == 1
    assert state["consuming"] is False

    # Scenario 2: flag pre-set (first consumer in-flight) → second bails.
    state["inject_count"] = 0
    state["consuming"] = True  # simulate first consumer running
    an._notification_queue.put(n1)
    await guarded_consume(n1)  # should be blocked immediately
    assert state["inject_count"] == 0  # runner never called
    state["consuming"] = False  # cleanup

    # Scenario 3: exception in runner → flag still cleared by finally.
    async def raising_runner(text: str, notifs: list) -> None:
        raise RuntimeError("boom")

    async def guarded_consume_raising(notif):
        if state["consuming"]:
            return
        state["consuming"] = True
        try:
            an._notification_queue.put(notif)
            await an.consume_notifications(raising_runner, fake_state_reader)
        except RuntimeError:
            pass
        finally:
            state["consuming"] = False

    await guarded_consume_raising(n2)
    assert state["consuming"] is False  # cleared despite exception


# ============================================================================
# Tests for Fix #3 — per-thread notification routing
# ============================================================================


def _drain_all(an_mod):
    """Drain every queue (per-thread + unrouted) so tests start clean."""
    while True:
        try:
            an_mod._notification_queue.get_nowait()
        except queue.Empty:
            break
    for q in list(an_mod._notifications_by_thread.values()):
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break
    while True:
        try:
            an_mod._unrouted_queue.get_nowait()
        except queue.Empty:
            break


def _reset_notifier_state(an_mod):
    _drain_all(an_mod)
    an_mod._reader_enqueued_task_ids.clear()
    an_mod._idle_reader_last_poll.clear()
    an_mod._idle_reader_active_seen.clear()


@pytest.fixture(autouse=True)
def _clean_async_notifier_state():
    _reset_notifier_state(async_notifier)
    yield
    _reset_notifier_state(async_notifier)


async def test_consume_only_drains_matching_thread():
    """Notifications tagged with origin_cli_thread_id only drain when the
    consumer is invoked with the matching current_thread_id."""
    from EvoScientist.cli import async_notifier as an

    n_a = an.AsyncTaskNotification(
        "tA", "writing-agent", "success", "", "", origin_cli_thread_id="threadA"
    )
    n_b = an.AsyncTaskNotification(
        "tB", "writing-agent", "success", "", "", origin_cli_thread_id="threadB"
    )
    an._enqueue(n_a)
    an._enqueue(n_b)

    captured: dict = {"runs": []}

    async def runner(text: str, notifs: list) -> None:
        captured["runs"].append([n.task_id for n in notifs])

    async def state_reader() -> dict:
        return {}

    await an.consume_notifications(runner, state_reader, current_thread_id="threadA")
    assert captured["runs"] == [["tA"]]
    # B's notification should still be queued
    assert an.has_pending_notifications("threadB")


async def test_unrouted_notifications_drain_on_any_thread():
    """Notifications without origin_cli_thread_id (legacy / direct put) drain
    regardless of the current_thread_id arg."""
    from EvoScientist.cli import async_notifier as an

    an._notification_queue.put(
        an.AsyncTaskNotification("tU", "writing-agent", "success", "", "")
    )

    captured: dict = {}

    async def runner(text: str, notifs: list) -> None:
        captured["notifs"] = notifs

    async def state_reader() -> dict:
        return {}

    await an.consume_notifications(runner, state_reader, current_thread_id="anything")
    assert [n.task_id for n in captured["notifs"]] == ["tU"]


async def test_thread_switch_drains_pending():
    """Pending notifications for thread B are not delivered while consumer
    asks for thread A; once consumer runs with thread B they drain."""
    from EvoScientist.cli import async_notifier as an

    an._enqueue(
        an.AsyncTaskNotification(
            "tB", "writing-agent", "success", "", "", origin_cli_thread_id="threadB"
        )
    )

    captured: dict = {"runs": []}

    async def runner(text: str, notifs: list) -> None:
        captured["runs"].append([n.task_id for n in notifs])

    async def state_reader() -> dict:
        return {}

    # First consume in thread A → no drain, B's notif still queued
    await an.consume_notifications(runner, state_reader, current_thread_id="threadA")
    assert captured["runs"] == []
    assert an.has_pending_notifications("threadB")

    # Now switch to thread B → drains
    await an.consume_notifications(runner, state_reader, current_thread_id="threadB")
    assert captured["runs"] == [["tB"]]


def test_has_pending_notifications_respects_routing():
    """has_pending_notifications returns true only for matching or unrouted."""
    from EvoScientist.cli import async_notifier as an

    # Unrouted always counts
    an._notification_queue.put(
        an.AsyncTaskNotification("tU", "writing-agent", "success", "", "")
    )
    assert an.has_pending_notifications("threadA") is True
    assert an.has_pending_notifications() is True
    _drain_all(an)

    # Routed only counts for the matching current thread
    an._enqueue(
        an.AsyncTaskNotification(
            "tA", "writing-agent", "success", "", "", origin_cli_thread_id="threadA"
        )
    )
    assert an.has_pending_notifications("threadA") is True
    assert an.has_pending_notifications("threadB") is False
    assert an.has_pending_notifications() is False  # no unrouted, no current_thread


# ============================================================================
# Tests for Fix #1 (v2) — in-band error detection from SSE stream.
#
# We don't poll runs.get after a clean stream close (it had a server-side
# write-back race that returned "error" for successful runs). Instead we
# watch for ``event="error"`` SSE parts which langgraph dev emits when a
# run fails — that signal is authoritative and arrives in-band before the
# stream closes.
# ============================================================================


# ============================================================================
# Tests for Fix #4 — consume_notifications surfaces exceptions to caller
# (callers wrap the await in try/except — verify the inner contract is to
# propagate so the wrapper sees + logs).
# ============================================================================


async def test_consume_notifications_propagates_inject_exception():
    """If the run_message callback raises, consume_notifications propagates
    the exception to the caller — pollers wrap it in try/except so the
    poller task does not die."""
    from EvoScientist.cli import async_notifier as an

    an._notification_queue.put(
        an.AsyncTaskNotification("tX", "writing-agent", "success", "", "")
    )

    async def boom_runner(text: str, notifs: list) -> None:
        raise RuntimeError("kaboom")

    async def state_reader() -> dict:
        return {}

    with pytest.raises(RuntimeError, match="kaboom"):
        await an.consume_notifications(boom_runner, state_reader)


def _drain_one_queue_helper(q):
    items = []
    while True:
        try:
            items.append(q.get_nowait())
        except queue.Empty:
            return items


# ============================================================================
# Slice 2.4a — state-based client reader (enqueue_completions_from_state)
#
# Backend-agnostic: the reader touches only the GraphGateway protocol
# (get_state_values + get_run_status), so FakeGraphGateway exercises the same
# path both real backends implement. The concrete per-backend get_run_status
# reads are pinned in test_graph_gateway.py.
# ============================================================================


def _running_registry(agent_name: str = "writing-agent"):
    return {
        "async_tasks": {
            "task-1": {
                "status": "running",
                "run_id": "run-1",
                "agent_name": agent_name,
            }
        }
    }


async def test_reader_enqueues_completion_from_state():
    gateway = FakeGraphGateway(
        state_values=_running_registry(),
        run_statuses={"run-1": "success"},
    )

    await async_notifier.enqueue_completions_from_state(
        gateway, GraphTarget(local_graph=MagicMock()), "cli-tid"
    )

    drained = drain_notifications("cli-tid")
    assert len(drained) == 1
    n = drained[0]
    assert n.task_id == "task-1"
    assert n.agent_name == "writing-agent"
    assert n.status == "success"
    assert n.origin_cli_thread_id == "cli-tid"


async def test_reader_no_op_while_task_running():
    gateway = FakeGraphGateway(
        state_values=_running_registry(),
        run_statuses={"run-1": "running"},
    )

    await async_notifier.enqueue_completions_from_state(
        gateway, GraphTarget(local_graph=MagicMock()), "cli-tid"
    )

    assert drain_notifications("cli-tid") == []


async def test_reader_dedupes_completion_across_polls():
    gateway = FakeGraphGateway(
        state_values=_running_registry(),
        run_statuses={"run-1": "success"},
    )
    target = GraphTarget(local_graph=MagicMock())

    await async_notifier.enqueue_completions_from_state(gateway, target, "cli-tid")
    await async_notifier.enqueue_completions_from_state(gateway, target, "cli-tid")

    # One enqueue total; the seen-set short-circuits the second poll before it
    # even issues a live status read.
    assert len(drain_notifications("cli-tid")) == 1
    assert gateway.run_status_calls == [("task-1", "run-1")]


async def test_reader_skips_task_already_terminal_in_state():
    gateway = FakeGraphGateway(
        state_values={
            "async_tasks": {
                "task-1": {
                    "status": "success",
                    "run_id": "run-1",
                    "agent_name": "x",
                }
            }
        },
        run_statuses={"run-1": "success"},
    )

    await async_notifier.enqueue_completions_from_state(
        gateway, GraphTarget(local_graph=MagicMock()), "cli-tid"
    )

    assert drain_notifications("cli-tid") == []
    # Terminal-in-state means the agent already saw it — no live read at all.
    assert gateway.run_status_calls == []


async def test_reader_best_effort_on_status_read_error():
    gateway = FakeGraphGateway(
        state_values=_running_registry(),
        run_status_error=RuntimeError("server down"),
    )

    # Must not raise, and nothing is enqueued — the task is retried next poll.
    await async_notifier.enqueue_completions_from_state(
        gateway, GraphTarget(local_graph=MagicMock()), "cli-tid"
    )

    assert drain_notifications("cli-tid") == []


async def test_reader_skips_task_without_run_id():
    gateway = FakeGraphGateway(
        state_values={"async_tasks": {"task-1": {"status": "running"}}},
    )

    await async_notifier.enqueue_completions_from_state(
        gateway, GraphTarget(local_graph=MagicMock()), "cli-tid"
    )

    assert drain_notifications("cli-tid") == []
    assert gateway.run_status_calls == []


async def test_reader_returns_active_task_count():
    gateway = FakeGraphGateway(
        state_values=_running_registry(),
        run_statuses={"run-1": "running"},
    )
    target = GraphTarget(local_graph=MagicMock())

    # One still-running task → active count 1.
    assert (
        await async_notifier.enqueue_completions_from_state(gateway, target, "cli-tid")
        == 1
    )
    # It completes → surfaced once and no longer active.
    gateway.run_statuses["run-1"] = "success"
    assert (
        await async_notifier.enqueue_completions_from_state(gateway, target, "cli-tid")
        == 0
    )
    assert len(drain_notifications("cli-tid")) == 1


async def test_reader_surfaces_one_completion_across_update():
    """update_async_task rotates run_id on the same task_id; the reader keys
    dedup on task_id, so a completion surfaces exactly once across the update —
    the behavior that replaced the watcher's pre_cancel."""
    registry = {
        "async_tasks": {
            "task-1": {
                "status": "running",
                "run_id": "run-1",
                "agent_name": "writing-agent",
            }
        }
    }
    gateway = FakeGraphGateway(
        state_values=registry,
        run_statuses={"run-1": "success", "run-2": "success"},
    )
    target = GraphTarget(local_graph=MagicMock())

    await async_notifier.enqueue_completions_from_state(gateway, target, "cli-tid")
    assert len(drain_notifications("cli-tid")) == 1

    # update_async_task: new run_id on the SAME task_id, back to running.
    registry["async_tasks"]["task-1"]["run_id"] = "run-2"
    registry["async_tasks"]["task-1"]["status"] = "running"
    await async_notifier.enqueue_completions_from_state(gateway, target, "cli-tid")

    # Already surfaced once → not re-enqueued for the rotated run.
    assert drain_notifications("cli-tid") == []


async def test_throttled_reader_rate_limits_within_interval(monkeypatch):
    gateway = FakeGraphGateway(
        state_values=_running_registry(),
        run_statuses={"run-1": "running"},
    )
    target = GraphTarget(local_graph=MagicMock())
    clock = {"t": 1000.0}
    monkeypatch.setattr(async_notifier.time, "monotonic", lambda: clock["t"])

    # First idle tick runs the reader (no prior observation → armed by default).
    await async_notifier.enqueue_completions_from_state_throttled(
        gateway, target, "cli-tid", min_interval_s=3.0
    )
    assert gateway.run_status_calls == [("task-1", "run-1")]

    # Second tick within the interval is throttled — no new state read.
    clock["t"] += 1.0
    await async_notifier.enqueue_completions_from_state_throttled(
        gateway, target, "cli-tid", min_interval_s=3.0
    )
    assert gateway.run_status_calls == [("task-1", "run-1")]

    # After the interval elapses, the reader runs again.
    clock["t"] += 3.0
    await async_notifier.enqueue_completions_from_state_throttled(
        gateway, target, "cli-tid", min_interval_s=3.0
    )
    assert gateway.run_status_calls == [("task-1", "run-1"), ("task-1", "run-1")]


async def test_throttled_reader_disarms_after_task_terminal(monkeypatch):
    gateway = FakeGraphGateway(
        state_values=_running_registry(),
        run_statuses={"run-1": "running"},
    )
    target = GraphTarget(local_graph=MagicMock())
    clock = {"t": 1000.0}
    monkeypatch.setattr(async_notifier.time, "monotonic", lambda: clock["t"])

    # Running task keeps idle polling armed.
    await async_notifier.enqueue_completions_from_state_throttled(
        gateway, target, "cli-tid", min_interval_s=3.0
    )
    assert async_notifier._idle_reader_active_seen["cli-tid"] is True

    # It completes; after the interval the idle tick surfaces it, then disarms.
    gateway.run_statuses["run-1"] = "success"
    clock["t"] += 5.0
    await async_notifier.enqueue_completions_from_state_throttled(
        gateway, target, "cli-tid", min_interval_s=3.0
    )
    assert len(drain_notifications("cli-tid")) == 1
    assert async_notifier._idle_reader_active_seen["cli-tid"] is False
    last_poll = async_notifier._idle_reader_last_poll["cli-tid"]

    # Nothing active now → further idle ticks short-circuit (no state read,
    # last_poll unchanged) even after the interval elapses.
    clock["t"] += 10.0
    await async_notifier.enqueue_completions_from_state_throttled(
        gateway, target, "cli-tid", min_interval_s=3.0
    )
    assert async_notifier._idle_reader_last_poll["cli-tid"] == last_poll
