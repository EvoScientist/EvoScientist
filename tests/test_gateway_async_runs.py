from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import EvoScientist.gateway.async_runs as async_runs


def test_launch_background_agent_submits_run_and_invokes_hooks(monkeypatch):
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
    started: list[async_runs.BackgroundAgentRun] = []
    watchers: list[async_runs.BackgroundAgentRun] = []

    def build_payload(thread_id: str) -> async_runs.BackgroundAgentRunPayload:
        payload_calls.append(thread_id)
        return {
            "assistant_id": "graph-1",
            "input": {"messages": [{"role": "user", "content": "go"}]},
            "metadata": {"run_kind": "test"},
            "config": {"configurable": {"thread_id": thread_id}},
        }

    request = async_runs.BackgroundAgentLaunchRequest(
        graph_id="graph-1",
        run_payload=build_payload,
        thread_metadata={"thread_kind": "test"},
        url="http://x",
        name="test worker",
    )

    handle = async_runs.launch_background_agent(
        request,
        hooks=async_runs.BackgroundAgentLaunchHooks(
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


def test_sync_status_watcher_finishes_and_deletes_thread(monkeypatch):
    finished: list[async_runs.BackgroundAgentRun] = []
    aborted: list[async_runs.BackgroundAgentRun] = []
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

    async_runs.watch_background_agent_run_sync(
        url="http://x",
        thread_id="thread-1",
        run_id="run-1",
        name="test worker",
        hooks=async_runs.BackgroundAgentLaunchHooks(
            on_finished=finished.append,
            on_aborted=aborted.append,
        ),
        watcher_config=async_runs.BackgroundAgentStatusWatcherConfig(
            poll_interval_seconds=0,
        ),
    )

    assert [run.run_id for run in finished] == ["run-1"]
    assert aborted == []
    assert deleted == ["thread-1"]


def test_sync_status_watcher_aborts_without_deleting_on_poll_failure(monkeypatch):
    finished: list[async_runs.BackgroundAgentRun] = []
    aborted: list[async_runs.BackgroundAgentRun] = []
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

    async_runs.watch_background_agent_run_sync(
        url="http://x",
        thread_id="thread-1",
        run_id="run-1",
        name="test worker",
        hooks=async_runs.BackgroundAgentLaunchHooks(
            on_finished=finished.append,
            on_aborted=aborted.append,
        ),
        watcher_config=async_runs.BackgroundAgentStatusWatcherConfig(
            poll_interval_seconds=0,
            max_poll_failures=1,
        ),
    )

    assert finished == []
    assert [run.run_id for run in aborted] == ["run-1"]
    assert deleted == []
