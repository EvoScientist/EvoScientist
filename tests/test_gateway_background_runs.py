from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import EvoScientist.gateway.background_runs as background_runs


def test_launch_background_run_submits_run_and_invokes_hooks(monkeypatch):
    monkeypatch.setattr(
        "EvoScientist.langgraph_dev.manager.is_langgraph_dev_running",
        lambda **_kwargs: True,
    )
    fake_client = MagicMock()
    fake_client.threads.create.return_value = {"thread_id": "thread-1"}
    fake_client.runs.create.return_value = {"run_id": "run-1", "status": "pending"}
    monkeypatch.setattr("langgraph_sdk.get_sync_client", lambda **_kwargs: fake_client)

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

    request = background_runs.BackgroundRunRequest(
        graph_id="graph-1",
        run_payload=build_payload,
        thread_metadata={"thread_kind": "test"},
        url="http://x",
        name="test worker",
    )

    handle = background_runs.launch_background_run(
        request,
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
    monkeypatch.setattr(
        "EvoScientist.langgraph_dev.manager.is_langgraph_dev_running",
        lambda **_kwargs: True,
    )
    fake_client = MagicMock()
    fake_client.threads.create.return_value = {"thread_id": "thread-1"}
    fake_client.runs.create.return_value = {"run_id": "run-1", "status": "pending"}
    monkeypatch.setattr("langgraph_sdk.get_sync_client", lambda **_kwargs: fake_client)

    request = background_runs.BackgroundRunRequest(
        graph_id="graph-1",
        run_payload=lambda thread_id: {
            "assistant_id": "graph-1",
            "input": {"messages": []},
            "metadata": {},
            "config": {"configurable": {"thread_id": thread_id}},
        },
        url="http://x",
        name="test worker",
    )
    watcher_failures: list[background_runs.BackgroundRun] = []
    aborted: list[background_runs.BackgroundRun] = []

    def fail_to_start_watcher(_run: background_runs.BackgroundRun) -> None:
        raise RuntimeError("watcher failed")

    handle = background_runs.launch_background_run(
        request,
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
    monkeypatch.setattr(
        "EvoScientist.langgraph_dev.manager.is_langgraph_dev_running",
        lambda **_kwargs: True,
    )
    fake_client = MagicMock()
    fake_client.threads.create.return_value = {"thread_id": "thread-1"}
    fake_client.runs.create.side_effect = RuntimeError("run creation failed")
    monkeypatch.setattr("langgraph_sdk.get_sync_client", lambda **_kwargs: fake_client)

    request = background_runs.BackgroundRunRequest(
        graph_id="graph-1",
        run_payload=lambda thread_id: {
            "assistant_id": "graph-1",
            "input": {"messages": []},
            "metadata": {},
            "config": {"configurable": {"thread_id": thread_id}},
        },
        url="http://x",
        name="test worker",
    )

    with pytest.raises(RuntimeError, match="run creation failed"):
        background_runs.launch_background_run(request)

    fake_client.threads.delete.assert_called_once_with("thread-1")


def test_async_launch_background_run_deletes_thread_when_run_creation_fails(
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
    request = background_runs.BackgroundRunRequest(
        graph_id="graph-1",
        run_payload=lambda thread_id: {
            "assistant_id": "graph-1",
            "input": {"messages": []},
            "metadata": {},
            "config": {"configurable": {"thread_id": thread_id}},
        },
        url="http://x",
        name="test worker",
    )

    async def run() -> None:
        with pytest.raises(RuntimeError, match="run creation failed"):
            await background_runs.alaunch_background_run(request)

    asyncio.run(run())

    assert deleted == ["thread-1"]


def test_sync_status_watcher_finishes_and_deletes_thread(monkeypatch):
    finished: list[background_runs.BackgroundRun] = []
    aborted: list[background_runs.BackgroundRun] = []
    deleted: list[str] = []

    class _Runs:
        def get(self, **_kwargs):
            return {"status": "success"}

    class _Threads:
        def delete(self, thread_id: str):
            deleted.append(thread_id)

    monkeypatch.setattr(
        "langgraph_sdk.get_sync_client",
        lambda **_kwargs: SimpleNamespace(runs=_Runs(), threads=_Threads()),
    )

    background_runs.watch_background_run_sync(
        url="http://x",
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

    assert [run.run_id for run in finished] == ["run-1"]
    assert aborted == []
    assert deleted == ["thread-1"]


def test_sync_status_watcher_aborts_and_deletes_thread_on_error_status(monkeypatch):
    finished: list[background_runs.BackgroundRun] = []
    aborted: list[background_runs.BackgroundRun] = []
    deleted: list[str] = []

    class _Runs:
        def get(self, **_kwargs):
            return {"status": "error"}

    class _Threads:
        def delete(self, thread_id: str):
            deleted.append(thread_id)

    monkeypatch.setattr(
        "langgraph_sdk.get_sync_client",
        lambda **_kwargs: SimpleNamespace(runs=_Runs(), threads=_Threads()),
    )

    background_runs.watch_background_run_sync(
        url="http://x",
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


def test_sync_status_watcher_aborts_without_deleting_on_poll_failure(monkeypatch):
    finished: list[background_runs.BackgroundRun] = []
    aborted: list[background_runs.BackgroundRun] = []
    deleted: list[str] = []

    class _Runs:
        def get(self, **_kwargs):
            raise RuntimeError("poll failed")

    class _Threads:
        def delete(self, thread_id: str):
            deleted.append(thread_id)

    monkeypatch.setattr(
        "langgraph_sdk.get_sync_client",
        lambda **_kwargs: SimpleNamespace(runs=_Runs(), threads=_Threads()),
    )

    background_runs.watch_background_run_sync(
        url="http://x",
        thread_id="thread-1",
        run_id="run-1",
        name="test worker",
        hooks=background_runs.BackgroundRunHooks(
            on_finished=finished.append,
            on_aborted=aborted.append,
        ),
        watcher_config=background_runs.BackgroundRunWatcherConfig(
            poll_interval_seconds=0,
            max_poll_failures=1,
        ),
    )

    assert finished == []
    assert [run.run_id for run in aborted] == ["run-1"]
    assert deleted == []


def test_sync_status_watcher_routes_poll_failure_to_status_unknown(monkeypatch):
    status_unknown: list[background_runs.BackgroundRun] = []
    aborted: list[background_runs.BackgroundRun] = []
    deleted: list[str] = []

    class _Runs:
        def get(self, **_kwargs):
            raise RuntimeError("poll failed")

    class _Threads:
        def delete(self, thread_id: str):
            deleted.append(thread_id)

    monkeypatch.setattr(
        "langgraph_sdk.get_sync_client",
        lambda **_kwargs: SimpleNamespace(runs=_Runs(), threads=_Threads()),
    )

    background_runs.watch_background_run_sync(
        url="http://x",
        thread_id="thread-1",
        run_id="run-1",
        name="test worker",
        hooks=background_runs.BackgroundRunHooks(
            on_status_unknown=status_unknown.append,
            on_aborted=aborted.append,
        ),
        watcher_config=background_runs.BackgroundRunWatcherConfig(
            poll_interval_seconds=0,
            max_poll_failures=1,
        ),
    )

    assert [run.run_id for run in status_unknown] == ["run-1"]
    assert aborted == []
    assert deleted == []


def test_async_status_watcher_aborts_and_deletes_thread_on_error_status():
    finished: list[background_runs.BackgroundRun] = []
    aborted: list[background_runs.BackgroundRun] = []
    deleted: list[str] = []

    class _Runs:
        async def get(self, **_kwargs):
            return {"status": "error"}

    class _Threads:
        async def delete(self, thread_id: str):
            deleted.append(thread_id)

    async def run() -> None:
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

    asyncio.run(run())

    assert finished == []
    assert [run.run_id for run in aborted] == ["run-1"]
    assert deleted == ["thread-1"]


def test_async_status_watcher_preserves_run_url():
    finished: list[background_runs.BackgroundRun] = []

    class _Runs:
        async def get(self, **_kwargs):
            return {"status": "success"}

    class _Threads:
        async def delete(self, _thread_id: str):
            return None

    async def run() -> None:
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

    asyncio.run(run())

    assert [run.url for run in finished] == ["http://worker.example"]
