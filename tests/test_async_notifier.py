"""Tests for async sub-agent auto-notification."""

import asyncio
import queue
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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
    # Drain anything left over from other tests
    while True:
        try:
            async_notifier._notification_queue.get_nowait()
        except queue.Empty:
            break
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


def test_read_async_tasks_from_gateway_reads_state_values(run_async):
    gateway = FakeGraphGateway(
        state_values={
            "async_tasks": {
                "task-1": {"status": "success"},
            }
        }
    )

    tasks = run_async(
        async_notifier.read_async_tasks_from_gateway(
            gateway,
            GraphTarget(local_graph=MagicMock()),
            "tid",
        )
    )

    assert tasks == {"task-1": {"status": "success"}}


def test_watcher_pushes_notification_on_stream_end(run_async):
    # Stream yields one "values" chunk with the final state, then closes
    final_state = {
        "messages": [{"type": "ai", "content": "Quantum superposition is..."}]
    }
    chunks = [SimpleNamespace(event="values", data=final_state)]

    async def fake_stream(thread_id, run_id, stream_mode):
        for c in chunks:
            yield c

    client = MagicMock()
    client.runs.join_stream = fake_stream
    # runs.get is used to fetch terminal status when stream ends
    client.runs.get = AsyncMock(return_value={"status": "success"})

    _drain_all(async_notifier)
    run_async(
        async_notifier.watch_run_and_notify(client, "thr-1", "run-1", "writing-agent")
    )

    notifs = _drain_queue(async_notifier._notification_queue)
    assert len(notifs) == 1
    assert notifs[0].task_id == "thr-1"
    assert notifs[0].agent_name == "writing-agent"
    assert notifs[0].status == "success"


def test_watcher_pushes_error_status_on_stream_exception(run_async):
    async def fake_stream(*a, **kw):
        raise RuntimeError("network broken")
        yield  # unreachable; makes this an async generator

    client = MagicMock()
    client.runs.join_stream = fake_stream
    # On stream failure, watcher falls back to runs.get for terminal status
    client.runs.get = AsyncMock(
        return_value={"status": "error", "error": "network broken"}
    )

    _drain_all(async_notifier)
    run_async(async_notifier.watch_run_and_notify(client, "thr-4", "run-4", "agentZ"))

    notif = async_notifier._notification_queue.get_nowait()
    assert notif.status == "error"


def test_spawn_watcher_replaces_existing_for_same_thread(run_async):
    """A second spawn_watcher with the same thread_id cancels the old watcher
    and registers the new one — supports update_async_task creating a new
    run_id on the same thread_id."""
    spawn_starts = []

    async def fake_stream_long(*a, **kw):
        spawn_starts.append("started")
        # Simulate a long-running stream that gets cancelled
        try:
            while True:
                await asyncio.sleep(0.01)
                yield SimpleNamespace(event="values", data={"messages": []})
        except asyncio.CancelledError:
            raise

    client = MagicMock()
    client.runs.join_stream = fake_stream_long
    client.runs.get = AsyncMock(return_value={"status": "success"})

    async def scenario():
        # Clear all queues and the watcher registries
        async_notifier._active_watchers.clear()
        async_notifier._watcher_by_thread.clear()
        _drain_all(async_notifier)

        # First spawn for thread X, run R1
        t1 = async_notifier.spawn_watcher(client, "thr-X", "R1", "agent")
        assert t1 is not None
        assert async_notifier._watcher_by_thread["thr-X"] is t1
        await asyncio.sleep(0.02)  # let it start streaming

        # Second spawn for SAME thread X, NEW run R2
        t2 = async_notifier.spawn_watcher(client, "thr-X", "R2", "agent")
        assert t2 is not None
        assert t2 is not t1
        assert async_notifier._watcher_by_thread["thr-X"] is t2

        # Old watcher should be cancelled
        await asyncio.sleep(0.02)
        assert t1.cancelled() or t1.done()

        # Cleanup the new task too
        t2.cancel()
        try:
            await t2
        except asyncio.CancelledError:
            pass

        # Cancelled watchers don't push notifications
        assert _drain_one_queue_helper(async_notifier._notification_queue) == []
        assert _drain_one_queue_helper(async_notifier._unrouted_queue) == []
        if hasattr(async_notifier, "_notifications_by_thread"):
            for q in async_notifier._notifications_by_thread.values():
                assert _drain_one_queue_helper(q) == []

    run_async(scenario())


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
    # Clear the queue first
    while True:
        try:
            async_notifier._notification_queue.get_nowait()
        except queue.Empty:
            break

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


def test_consume_notifications_calls_runner_with_batched_message(run_async):
    """When notifications arrive and agent is idle, consume_notifications fires
    the supplied async runner once with the formatted batch message and notifs list."""
    from EvoScientist.cli import async_notifier as an

    # Set up two pending notifications, no dedup match
    while True:
        try:
            an._notification_queue.get_nowait()
        except queue.Empty:
            break
    an._notification_queue.put(an.AsyncTaskNotification("t1", "wA", "success", "", ""))
    an._notification_queue.put(an.AsyncTaskNotification("t2", "wB", "success", "", ""))

    captured: dict = {}

    async def fake_runner(text: str, notifs: list) -> None:
        captured["text"] = text
        captured["notifs"] = notifs

    async def fake_state_reader() -> dict:
        return {}  # no dedup info

    run_async(an.consume_notifications(fake_runner, fake_state_reader))
    assert "wA" in captured["text"]
    assert "wB" in captured["text"]
    assert len(captured["notifs"]) == 2


def test_consume_notifications_no_op_when_queue_empty(run_async):
    from EvoScientist.cli import async_notifier as an

    while True:
        try:
            an._notification_queue.get_nowait()
        except queue.Empty:
            break

    called = False

    async def fake_runner(text: str, notifs: list):
        nonlocal called
        called = True

    async def fake_state_reader():
        return {}

    run_async(an.consume_notifications(fake_runner, fake_state_reader))
    assert called is False


# ============================================================================
# Tests for TUI consumer reentry guard (_notification_consuming flag)
# Exercises Fix 2: the flag prevents two overlapping consume coroutines from
# both eventually calling _inject_notification_tui (and thus _run_turn).
# ============================================================================


def test_notification_consuming_flag_prevents_reentry(run_async):
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

    # Clear the queue
    while True:
        try:
            an._notification_queue.get_nowait()
        except queue.Empty:
            break

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

    async def scenario():
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

    run_async(scenario())


# ============================================================================
# Tests for Fix #3 — per-thread notification routing
# ============================================================================


def _drain_all(an_mod):
    """Drain every queue (per-thread + unrouted) so tests start clean."""
    if hasattr(an_mod, "_notification_queue"):
        while True:
            try:
                an_mod._notification_queue.get_nowait()
            except queue.Empty:
                break
    if hasattr(an_mod, "_notifications_by_thread"):
        for q in list(an_mod._notifications_by_thread.values()):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
    if hasattr(an_mod, "_unrouted_queue"):
        while True:
            try:
                an_mod._unrouted_queue.get_nowait()
            except queue.Empty:
                break


def test_consume_only_drains_matching_thread(run_async):
    """Notifications tagged with origin_cli_thread_id only drain when the
    consumer is invoked with the matching current_thread_id."""
    from EvoScientist.cli import async_notifier as an

    _drain_all(an)
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

    run_async(
        an.consume_notifications(runner, state_reader, current_thread_id="threadA")
    )
    assert captured["runs"] == [["tA"]]
    # B's notification should still be queued
    assert an.has_pending_notifications("threadB")
    _drain_all(an)


def test_unrouted_notifications_drain_on_any_thread(run_async):
    """Notifications without origin_cli_thread_id (legacy / direct put) drain
    regardless of the current_thread_id arg."""
    from EvoScientist.cli import async_notifier as an

    _drain_all(an)
    an._notification_queue.put(
        an.AsyncTaskNotification("tU", "writing-agent", "success", "", "")
    )

    captured: dict = {}

    async def runner(text: str, notifs: list) -> None:
        captured["notifs"] = notifs

    async def state_reader() -> dict:
        return {}

    run_async(
        an.consume_notifications(runner, state_reader, current_thread_id="anything")
    )
    assert [n.task_id for n in captured["notifs"]] == ["tU"]
    _drain_all(an)


def test_thread_switch_drains_pending(run_async):
    """Pending notifications for thread B are not delivered while consumer
    asks for thread A; once consumer runs with thread B they drain."""
    from EvoScientist.cli import async_notifier as an

    _drain_all(an)
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
    run_async(
        an.consume_notifications(runner, state_reader, current_thread_id="threadA")
    )
    assert captured["runs"] == []
    assert an.has_pending_notifications("threadB")

    # Now switch to thread B → drains
    run_async(
        an.consume_notifications(runner, state_reader, current_thread_id="threadB")
    )
    assert captured["runs"] == [["tB"]]
    _drain_all(an)


def test_has_pending_notifications_respects_routing():
    """has_pending_notifications returns true only for matching or unrouted."""
    from EvoScientist.cli import async_notifier as an

    _drain_all(an)
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
    _drain_all(an)


# ============================================================================
# Tests for Fix #1 (v2) — in-band error detection from SSE stream.
#
# We don't poll runs.get after a clean stream close (it had a server-side
# write-back race that returned "error" for successful runs). Instead we
# watch for ``event="error"`` SSE parts which langgraph dev emits when a
# run fails — that signal is authoritative and arrives in-band before the
# stream closes.
# ============================================================================


def test_watcher_reports_error_on_in_band_error_event(run_async):
    """SSE error event in the stream → notification.status == 'error'."""

    async def fake_stream(*a, **kw):
        yield SimpleNamespace(
            event="values", data={"messages": [{"type": "ai", "content": "partial"}]}
        )
        yield SimpleNamespace(event="error", data={"message": "subagent crashed"})

    client = MagicMock()
    client.runs.join_stream = fake_stream
    client.runs.get = AsyncMock(
        return_value={"status": "success"}
    )  # would mislead — should NOT be consulted

    _drain_all(async_notifier)
    run_async(async_notifier.watch_run_and_notify(client, "thrE", "rE", "agentE"))

    notif = async_notifier._notification_queue.get_nowait()
    assert notif.status == "error"
    # We must NOT have polled runs.get — the in-band signal is authoritative.
    client.runs.get.assert_not_awaited()


def test_watcher_clean_exit_with_runs_get_success_is_success(run_async):
    """Clean stream exit + runs.get reports success → status=success."""

    async def fake_stream(*a, **kw):
        yield SimpleNamespace(
            event="values", data={"messages": [{"type": "ai", "content": "ok"}]}
        )

    client = MagicMock()
    client.runs.join_stream = fake_stream
    client.runs.get = AsyncMock(return_value={"status": "success"})

    _drain_all(async_notifier)
    run_async(async_notifier.watch_run_and_notify(client, "thrS", "rS", "agentS"))

    notif = async_notifier._notification_queue.get_nowait()
    assert notif.status == "success"
    client.runs.get.assert_awaited_once()


def test_watcher_clean_exit_with_runs_get_error_is_race_safe(run_async):
    """Clean stream exit + no in-band error event + runs.get returns 'error'
    → status=success (race-safe).

    Server-side state writeback can transiently report 'error' for an
    actually-successful run between SSE close and final-state finalization.
    The absence of an in-band error event is authoritative — the run did
    not actually error. This test guards against re-introducing the race
    we hit when an earlier 'always-poll runs.get' attempt blindly trusted
    the runs.get value.
    """

    async def fake_stream(*a, **kw):
        yield SimpleNamespace(
            event="values", data={"messages": [{"type": "ai", "content": "ok"}]}
        )

    client = MagicMock()
    client.runs.join_stream = fake_stream
    client.runs.get = AsyncMock(return_value={"status": "error"})

    _drain_all(async_notifier)
    run_async(async_notifier.watch_run_and_notify(client, "thrS", "rS", "agentS"))

    notif = async_notifier._notification_queue.get_nowait()
    assert notif.status == "success"


def test_watcher_clean_exit_with_runs_get_running_drops_notification(run_async):
    """Reproduces the production bug: clean SSE close while run is still
    actually running (HTTP keep-alive timeout under concurrency).

    Pre-fix: watcher trusted clean stream exit as 'success' and enqueued a
    false-positive notification for a still-running task.

    Post-fix: watcher verifies via runs.get and re-joins the stream until
    either a terminal status arrives or the reconnect budget is exhausted.
    With a mock that perpetually closes cleanly + reports 'running', the
    watcher exhausts retries and enqueues nothing.
    """

    async def fake_stream(*a, **kw):
        # SSE closes cleanly after one chunk — simulates HTTP keep-alive
        # timeout where the server drops the long-poll without an error.
        yield SimpleNamespace(event="values", data={"messages": []})

    client = MagicMock()
    client.runs.join_stream = fake_stream
    client.runs.get = AsyncMock(return_value={"status": "running"})

    _drain_all(async_notifier)
    run_async(
        async_notifier.watch_run_and_notify(
            client, "thr-bug", "rB", "data-analysis-agent"
        )
    )

    # No notification should have been enqueued anywhere.
    assert _drain_one_queue_helper(async_notifier._unrouted_queue) == []
    assert _drain_one_queue_helper(async_notifier._notification_queue) == []
    for q in async_notifier._notifications_by_thread.values():
        assert _drain_one_queue_helper(q) == []
    # runs.get must have been polled at least once (the verify step).
    assert client.runs.get.await_count >= 1


def test_watcher_unknown_status_treated_as_non_terminal(run_async):
    """Future / unrecognized status values should trigger a re-join, not a
    false-positive notification.

    If the SDK introduces a new non-terminal status (e.g. ``queued``,
    ``scheduled``) the watcher must NOT silently default to ``success`` —
    that would re-introduce the same class of bug we just fixed. The
    safe-default policy: anything outside ``TERMINAL_STATUSES`` is treated
    as ``running``-equivalent and triggers re-join.
    """

    async def fake_stream(*a, **kw):
        yield SimpleNamespace(event="values", data={"messages": []})

    client = MagicMock()
    client.runs.join_stream = fake_stream
    # First call: hypothetical future status. Second call: actual completion.
    client.runs.get = AsyncMock(
        side_effect=[{"status": "queued"}, {"status": "success"}]
    )

    _drain_all(async_notifier)
    run_async(async_notifier.watch_run_and_notify(client, "thrU", "rU", "agentU"))

    notif = async_notifier._notification_queue.get_nowait()
    assert notif.status == "success"
    # Re-joined because the unknown status was not terminal.
    assert client.runs.get.await_count == 2


def test_watcher_runs_get_persistent_failure_drops_notification(run_async, monkeypatch):
    """If ``runs.get`` keeps raising, the watcher cannot verify terminal
    state and MUST drop the notification rather than default to
    ``"success"`` — otherwise a transient server outage reintroduces the
    same false-positive class this watcher exists to prevent."""

    async def fake_stream(*a, **kw):
        yield SimpleNamespace(event="values", data={"messages": []})

    client = MagicMock()
    client.runs.join_stream = fake_stream
    client.runs.get = AsyncMock(side_effect=RuntimeError("server unreachable"))

    # Skip the backoff sleeps to keep this test fast.
    async def _no_sleep(*a, **kw):
        return None

    monkeypatch.setattr(async_notifier.asyncio, "sleep", _no_sleep)

    _drain_all(async_notifier)
    run_async(async_notifier.watch_run_and_notify(client, "thrG", "rG", "agentG"))

    # No notification — watcher exhausted the reconnect budget. Check every
    # queue routing could send to so a future routing change can't make this
    # test silently false-pass.
    assert _drain_one_queue_helper(async_notifier._unrouted_queue) == []
    assert _drain_one_queue_helper(async_notifier._notification_queue) == []
    if hasattr(async_notifier, "_notifications_by_thread"):
        for q in async_notifier._notifications_by_thread.values():
            assert _drain_one_queue_helper(q) == []
    # 1 initial + _MAX_RECONNECT_ATTEMPTS retries = 11 calls total.
    assert client.runs.get.await_count == async_notifier._MAX_RECONNECT_ATTEMPTS + 1


def test_watcher_runs_get_transient_failure_recovers(run_async, monkeypatch):
    """A single ``runs.get`` failure followed by a successful response on
    retry must produce a correct notification — verifies the bounded
    retry path actually recovers from transient outages instead of just
    eating notifications."""

    async def fake_stream(*a, **kw):
        yield SimpleNamespace(event="values", data={"messages": []})

    client = MagicMock()
    client.runs.join_stream = fake_stream
    # First call raises (transient), second call returns terminal status.
    client.runs.get = AsyncMock(
        side_effect=[RuntimeError("blip"), {"status": "success"}]
    )

    async def _no_sleep(*a, **kw):
        return None

    monkeypatch.setattr(async_notifier.asyncio, "sleep", _no_sleep)

    _drain_all(async_notifier)
    run_async(async_notifier.watch_run_and_notify(client, "thrT", "rT", "agentT"))

    notif = async_notifier._notification_queue.get_nowait()
    assert notif.status == "success"
    assert client.runs.get.await_count == 2


def test_watcher_re_joins_stream_until_terminal_status(run_async):
    """When runs.get returns 'running' on attempt N but a terminal status
    on attempt N+1, the watcher re-joins, observes the terminal status,
    and enqueues the notification correctly."""

    async def fake_stream(*a, **kw):
        yield SimpleNamespace(event="values", data={"messages": []})

    client = MagicMock()
    client.runs.join_stream = fake_stream
    # First call: still running. Second call: success.
    client.runs.get = AsyncMock(
        side_effect=[{"status": "running"}, {"status": "success"}]
    )

    _drain_all(async_notifier)
    run_async(async_notifier.watch_run_and_notify(client, "thrR", "rR", "agentR"))

    notif = async_notifier._notification_queue.get_nowait()
    assert notif.status == "success"
    assert client.runs.get.await_count == 2


# ============================================================================
# Tests for Fix #4 — consume_notifications surfaces exceptions to caller
# (callers wrap the await in try/except — verify the inner contract is to
# propagate so the wrapper sees + logs).
# ============================================================================


def test_consume_notifications_propagates_inject_exception(run_async):
    """If the run_message callback raises, consume_notifications propagates
    the exception to the caller — pollers wrap it in try/except so the
    poller task does not die."""
    import pytest

    from EvoScientist.cli import async_notifier as an

    _drain_all(an)
    an._notification_queue.put(
        an.AsyncTaskNotification("tX", "writing-agent", "success", "", "")
    )

    async def boom_runner(text: str, notifs: list) -> None:
        raise RuntimeError("kaboom")

    async def state_reader() -> dict:
        return {}

    with pytest.raises(RuntimeError, match="kaboom"):
        run_async(an.consume_notifications(boom_runner, state_reader))
    _drain_all(an)


def test_watcher_skips_notification_on_stream_fail_with_nonterminal_status(run_async):
    """When the SSE stream errors AND runs.get returns a non-terminal status
    (e.g. ``pending`` because the run is still alive), the watcher must
    NOT enqueue a notification — otherwise the user sees a confusing
    ``⚠ pending`` line for a task that's still working. This is the early-
    return guard added alongside the Fix #2 revert."""

    async def fake_stream(*a, **kw):
        # Simulate transient transport error mid-stream.
        raise RuntimeError("connection reset")
        yield  # unreachable; makes this an async generator

    client = MagicMock()
    client.runs.join_stream = fake_stream
    client.runs.get = AsyncMock(return_value={"status": "pending"})

    _drain_all(async_notifier)
    run_async(async_notifier.watch_run_and_notify(client, "thrP", "rP", "agentP"))

    # No notification should have been enqueued in any queue.
    assert _drain_one_queue_helper(async_notifier._unrouted_queue) == []
    assert _drain_one_queue_helper(async_notifier._notification_queue) == []
    if hasattr(async_notifier, "_notifications_by_thread"):
        for q in async_notifier._notifications_by_thread.values():
            assert _drain_one_queue_helper(q) == []


def _drain_one_queue_helper(q):
    items = []
    while True:
        try:
            items.append(q.get_nowait())
        except queue.Empty:
            return items


def test_active_watchers_grace_filters_by_thread():
    """Verifies _has_relevant_active_watchers ignores sibling-thread watchers
    (otherwise consume_notifications grace period would block thread A by up
    to 3s waiting for thread B's unrelated watchers to finish)."""

    async_notifier._active_watchers.clear()

    # Sentinel handles — only their identity matters here, not their type
    handle_a = object()
    handle_b = object()
    handle_unrouted = object()

    async_notifier._active_watchers[handle_a] = "threadA"
    async_notifier._active_watchers[handle_b] = "threadB"
    async_notifier._active_watchers[handle_unrouted] = None

    # Current thread A → A's own watcher + unrouted are relevant
    assert async_notifier._has_relevant_active_watchers("threadA") is True
    # Current thread C (no active watcher of its own) → only unrouted matters
    assert async_notifier._has_relevant_active_watchers("threadC") is True
    # Drop the unrouted handle → C now has nothing relevant
    del async_notifier._active_watchers[handle_unrouted]
    assert async_notifier._has_relevant_active_watchers("threadC") is False
    # A still has its own watcher
    assert async_notifier._has_relevant_active_watchers("threadA") is True
    # Legacy: None argument falls back to "any active watcher counts"
    assert async_notifier._has_relevant_active_watchers(None) is True

    async_notifier._active_watchers.clear()
    assert async_notifier._has_relevant_active_watchers("threadA") is False
    assert async_notifier._has_relevant_active_watchers(None) is False
