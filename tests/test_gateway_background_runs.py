from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import EvoScientist.gateway.background_runs as background_runs


def _run_payload(thread_id: str) -> background_runs.BackgroundRunPayload:
    return {
        "assistant_id": "graph-1",
        "input": {"messages": []},
        "metadata": {"run_kind": "test"},
        "config": {"configurable": {"thread_id": thread_id}},
    }


def _request(
    *,
    run_payload=_run_payload,
    thread_metadata: dict[str, str] | None = None,
) -> background_runs.BackgroundRunRequest:
    return background_runs.BackgroundRunRequest(
        graph_id="graph-1",
        run_payload=run_payload,
        thread_metadata=thread_metadata,
        url="http://x",
        name="test worker",
    )


def _install_sync_launcher(
    monkeypatch,
    *,
    run_create_result: object | None = None,
    run_create_error: Exception | None = None,
) -> MagicMock:
    monkeypatch.setattr(
        "EvoScientist.langgraph_dev.manager.is_langgraph_dev_running",
        lambda **_kwargs: True,
    )
    fake_client = MagicMock()
    fake_client.threads.create.return_value = {"thread_id": "thread-1"}
    if run_create_error is not None:
        fake_client.runs.create.side_effect = run_create_error
    else:
        fake_client.runs.create.return_value = run_create_result or {
            "run_id": "run-1",
            "status": "pending",
        }
    monkeypatch.setattr("langgraph_sdk.get_sync_client", lambda **_kwargs: fake_client)
    return fake_client


def _install_sync_watcher(monkeypatch, *, status: str | Exception, deleted: list[str]):
    class _Runs:
        def get(self, **_kwargs):
            if isinstance(status, Exception):
                raise status
            return {"status": status}

    class _Threads:
        def delete(self, thread_id: str):
            deleted.append(thread_id)

    monkeypatch.setattr(
        "langgraph_sdk.get_sync_client",
        lambda **_kwargs: SimpleNamespace(runs=_Runs(), threads=_Threads()),
    )


def _watch_sync(
    *,
    hooks: background_runs.BackgroundRunHooks,
    max_poll_failures: int = 3,
) -> None:
    background_runs.watch_background_run_sync(
        url="http://x",
        thread_id="thread-1",
        run_id="run-1",
        name="test worker",
        hooks=hooks,
        watcher_config=background_runs.BackgroundRunWatcherConfig(
            poll_interval_seconds=0,
            max_poll_failures=max_poll_failures,
        ),
    )


def test_launch_background_run_submits_run_and_invokes_hooks(monkeypatch):
    fake_client = _install_sync_launcher(
        monkeypatch,
        run_create_result={
            "run_id": "run-1",
            "status": "pending",
        },
    )

    payload_calls: list[str] = []
    before_calls: list[str] = []
    started: list[background_runs.BackgroundRun] = []
    watchers: list[background_runs.BackgroundRun] = []

    def build_payload(thread_id: str) -> background_runs.BackgroundRunPayload:
        payload_calls.append(thread_id)
        return {
            "assistant_id": "graph-1",
            "input": {"messages": [{"role": "user", "content": "go"}]},
            "metadata": {"run_kind": "test"},
            "config": {"configurable": {"thread_id": thread_id}},
        }

    handle = background_runs.launch_background_run(
        _request(
            run_payload=build_payload,
            thread_metadata={"thread_kind": "test"},
        ),
        hooks=background_runs.BackgroundRunHooks(
            on_before_run=before_calls.append,
            on_started=started.append,
        ),
        spawn_status_watcher=watchers.append,
    )

    assert handle is not None
    assert handle.thread_id == "thread-1"
    assert handle.run_id == "run-1"
    assert payload_calls == ["thread-1"]
    assert before_calls == ["thread-1"]
    assert started == [handle]
    assert watchers == [handle]
    fake_client.threads.create.assert_called_once_with(
        graph_id="graph-1",
        metadata={"thread_kind": "test"},
    )
    fake_client.runs.create.assert_called_once_with(
        thread_id="thread-1",
        assistant_id="graph-1",
        input={"messages": [{"role": "user", "content": "go"}]},
        metadata={"run_kind": "test"},
        config={"configurable": {"thread_id": "thread-1"}},
    )


def test_launch_background_run_routes_watcher_start_failure_to_hook(monkeypatch):
    _install_sync_launcher(monkeypatch)
    watcher_failures: list[background_runs.BackgroundRun] = []
    aborted: list[background_runs.BackgroundRun] = []

    def fail_to_start_watcher(_run: background_runs.BackgroundRun) -> None:
        raise RuntimeError("watcher failed")

    handle = background_runs.launch_background_run(
        _request(),
        hooks=background_runs.BackgroundRunHooks(
            on_watcher_start_failed=watcher_failures.append,
            on_aborted=aborted.append,
        ),
        spawn_status_watcher=fail_to_start_watcher,
    )

    assert handle is not None
    assert watcher_failures == [handle]
    assert aborted == []


def test_launch_background_run_deletes_thread_when_run_creation_fails(monkeypatch):
    fake_client = _install_sync_launcher(
        monkeypatch,
        run_create_error=RuntimeError("run creation failed"),
    )

    with pytest.raises(RuntimeError, match="run creation failed"):
        background_runs.launch_background_run(_request())

    fake_client.threads.delete.assert_called_once_with("thread-1")


async def test_async_launch_background_run_deletes_thread_when_run_creation_fails(
    monkeypatch,
):
    monkeypatch.setattr(
        "EvoScientist.langgraph_dev.manager.is_langgraph_dev_running",
        lambda **_kwargs: True,
    )
    deleted: list[str] = []

    class _Threads:
        async def create(self, **_kwargs):
            return {"thread_id": "thread-1"}

        async def delete(self, thread_id: str):
            deleted.append(thread_id)

    class _Runs:
        async def create(self, **_kwargs):
            raise RuntimeError("run creation failed")

    monkeypatch.setattr(
        "langgraph_sdk.get_client",
        lambda **_kwargs: SimpleNamespace(threads=_Threads(), runs=_Runs()),
    )

    with pytest.raises(RuntimeError, match="run creation failed"):
        await background_runs.alaunch_background_run(_request())

    assert deleted == ["thread-1"]


@pytest.mark.parametrize(
    ("status", "expected_finished", "expected_aborted"),
    [
        ("success", ["run-1"], []),
        ("error", [], ["run-1"]),
    ],
)
def test_sync_status_watcher_handles_terminal_statuses(
    monkeypatch,
    status: str,
    expected_finished: list[str],
    expected_aborted: list[str],
):
    finished: list[background_runs.BackgroundRun] = []
    aborted: list[background_runs.BackgroundRun] = []
    deleted: list[str] = []
    _install_sync_watcher(monkeypatch, status=status, deleted=deleted)

    _watch_sync(
        hooks=background_runs.BackgroundRunHooks(
            on_finished=finished.append,
            on_aborted=aborted.append,
        ),
    )

    assert [run.run_id for run in finished] == expected_finished
    assert [run.run_id for run in aborted] == expected_aborted
    assert deleted == ["thread-1"]


@pytest.mark.parametrize(
    ("use_status_unknown", "expected_unknown", "expected_aborted"),
    [
        (False, [], ["run-1"]),
        (True, ["run-1"], []),
    ],
)
def test_sync_status_watcher_preserves_thread_on_poll_failure(
    monkeypatch,
    use_status_unknown: bool,
    expected_unknown: list[str],
    expected_aborted: list[str],
):
    status_unknown: list[background_runs.BackgroundRun] = []
    aborted: list[background_runs.BackgroundRun] = []
    deleted: list[str] = []
    _install_sync_watcher(
        monkeypatch,
        status=RuntimeError("poll failed"),
        deleted=deleted,
    )

    _watch_sync(
        hooks=background_runs.BackgroundRunHooks(
            on_status_unknown=status_unknown.append if use_status_unknown else None,
            on_aborted=aborted.append,
        ),
        max_poll_failures=1,
    )

    assert [run.run_id for run in status_unknown] == expected_unknown
    assert [run.run_id for run in aborted] == expected_aborted
    assert deleted == []


async def test_async_status_watcher_aborts_and_deletes_thread_on_error_status():
    finished: list[background_runs.BackgroundRun] = []
    aborted: list[background_runs.BackgroundRun] = []
    deleted: list[str] = []

    class _Runs:
        async def get(self, **_kwargs):
            return {"status": "error"}

    class _Threads:
        async def delete(self, thread_id: str):
            deleted.append(thread_id)

    await background_runs.awatch_background_run(
        SimpleNamespace(runs=_Runs(), threads=_Threads()),
        thread_id="thread-1",
        run_id="run-1",
        name="test worker",
        hooks=background_runs.BackgroundRunHooks(
            on_finished=finished.append,
            on_aborted=aborted.append,
        ),
        watcher_config=background_runs.BackgroundRunWatcherConfig(
            poll_interval_seconds=0,
        ),
    )

    assert finished == []
    assert [run.run_id for run in aborted] == ["run-1"]
    assert deleted == ["thread-1"]


def test_delete_thread_bulk_cancels_nonterminal_runs_before_delete():
    events: list[tuple] = []

    class _Runs:
        def list(self, thread_id: str, *, limit: int, offset: int, status: str):
            events.append(("list", thread_id, status, offset))
            if status == "pending":
                return [{"run_id": "run-pending", "status": "pending"}]
            return []

        def cancel_many(self, *, thread_id: str, run_ids):
            events.append(("cancel_many", thread_id, list(run_ids)))

    class _Threads:
        def delete(self, thread_id: str):
            events.append(("delete", thread_id))

    background_runs._delete_thread(
        SimpleNamespace(runs=_Runs(), threads=_Threads()),
        "thread-1",
        name="test worker",
    )

    assert events == [
        ("list", "thread-1", "pending", 0),
        ("list", "thread-1", "running", 0),
        ("cancel_many", "thread-1", ["run-pending"]),
        ("delete", "thread-1"),
    ]


def test_delete_thread_skips_cancel_when_all_runs_terminal():
    events: list[tuple] = []

    class _Runs:
        def list(self, thread_id: str, *, limit: int, offset: int, status: str):
            events.append(("list", thread_id, status, offset))
            return []

        def cancel_many(self, **_kwargs):  # pragma: no cover
            raise AssertionError("cancel_many must not be called")

    class _Threads:
        def delete(self, thread_id: str):
            events.append(("delete", thread_id))

    background_runs._delete_thread(
        SimpleNamespace(runs=_Runs(), threads=_Threads()),
        "thread-1",
        name="test worker",
    )

    assert events == [
        ("list", "thread-1", "pending", 0),
        ("list", "thread-1", "running", 0),
        ("delete", "thread-1"),
    ]


def test_cancel_thread_runs_paginates_past_first_page(monkeypatch):
    monkeypatch.setattr(background_runs, "_RUN_CANCEL_PAGE_SIZE", 2)
    cancelled: list[list[str]] = []
    pages = {
        ("pending", 0): [
            {"run_id": "p1", "status": "pending"},
            {"run_id": "p2", "status": "pending"},
        ],
        ("pending", 2): [{"run_id": "p3", "status": "pending"}],
        ("running", 0): [{"run_id": "r1", "status": "running"}],
    }

    class _Runs:
        def list(self, thread_id: str, *, limit: int, offset: int, status: str):
            assert limit == 2
            return pages.get((status, offset), [])

        def cancel_many(self, *, thread_id: str, run_ids):
            cancelled.append(list(run_ids))

    background_runs._cancel_thread_runs(
        SimpleNamespace(runs=_Runs()),
        "thread-1",
        name="test worker",
    )

    assert cancelled == [["p1", "p2", "p3", "r1"]]


def test_delete_thread_still_deletes_when_run_listing_fails():
    deleted: list[str] = []

    class _Runs:
        def list(self, thread_id: str, *, limit: int, offset: int, status: str):
            raise RuntimeError("listing failed")

        def cancel_many(self, **_kwargs):  # pragma: no cover
            raise AssertionError("cancel_many should not be reached")

    class _Threads:
        def delete(self, thread_id: str):
            deleted.append(thread_id)

    background_runs._delete_thread(
        SimpleNamespace(runs=_Runs(), threads=_Threads()),
        "thread-1",
        name="test worker",
    )

    assert deleted == ["thread-1"]


async def test_adelete_thread_bulk_cancels_nonterminal_runs_before_delete():
    events: list[tuple] = []

    class _Runs:
        async def list(self, thread_id: str, *, limit: int, offset: int, status: str):
            events.append(("list", thread_id, status, offset))
            if status == "running":
                return [{"run_id": "run-running", "status": "running"}]
            return []

        async def cancel_many(self, *, thread_id: str, run_ids):
            events.append(("cancel_many", thread_id, list(run_ids)))

    class _Threads:
        async def delete(self, thread_id: str):
            events.append(("delete", thread_id))

    await background_runs._adelete_thread(
        SimpleNamespace(runs=_Runs(), threads=_Threads()),
        "thread-1",
        name="test worker",
    )

    assert events == [
        ("list", "thread-1", "pending", 0),
        ("list", "thread-1", "running", 0),
        ("cancel_many", "thread-1", ["run-running"]),
        ("delete", "thread-1"),
    ]


async def test_adelete_thread_still_deletes_when_run_listing_fails():
    deleted: list[str] = []

    class _Runs:
        async def list(self, thread_id: str, *, limit: int, offset: int, status: str):
            raise RuntimeError("listing failed")

        async def cancel_many(self, **_kwargs):  # pragma: no cover
            raise AssertionError("cancel_many should not be reached")

    class _Threads:
        async def delete(self, thread_id: str):
            deleted.append(thread_id)

    await background_runs._adelete_thread(
        SimpleNamespace(runs=_Runs(), threads=_Threads()),
        "thread-1",
        name="test worker",
    )

    assert deleted == ["thread-1"]


def test_launch_cancels_stray_run_before_thread_delete_when_run_creation_fails(
    monkeypatch,
):
    fake_client = _install_sync_launcher(
        monkeypatch,
        run_create_error=RuntimeError("run creation failed"),
    )
    fake_client.runs.list.side_effect = lambda _thread_id, **kwargs: (
        [{"run_id": "stray-run", "status": "pending"}]
        if kwargs.get("status") == "pending"
        else []
    )

    with pytest.raises(RuntimeError, match="run creation failed"):
        background_runs.launch_background_run(_request())

    fake_client.runs.cancel_many.assert_called_once_with(
        thread_id="thread-1",
        run_ids=["stray-run"],
    )
    fake_client.threads.delete.assert_called_once_with("thread-1")
    call_names = [name for name, _args, _kwargs in fake_client.mock_calls]
    assert call_names.index("runs.cancel_many") < call_names.index("threads.delete")


async def test_async_status_watcher_preserves_run_url():
    finished: list[background_runs.BackgroundRun] = []

    class _Runs:
        async def get(self, **_kwargs):
            return {"status": "success"}

    class _Threads:
        async def delete(self, _thread_id: str):
            return None

    await background_runs.awatch_background_run(
        SimpleNamespace(runs=_Runs(), threads=_Threads()),
        url="http://worker.example",
        thread_id="thread-1",
        run_id="run-1",
        name="test worker",
        hooks=background_runs.BackgroundRunHooks(
            on_finished=finished.append,
        ),
        watcher_config=background_runs.BackgroundRunWatcherConfig(
            poll_interval_seconds=0,
        ),
    )

    assert [run.url for run in finished] == ["http://worker.example"]
