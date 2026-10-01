"""Child runs carry their folders, and the server refuses runs meant for others."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from EvoScientist.langgraph_dev.folder_check import (
    RunFolderMismatchError,
    check_run_folders,
    folder_checked,
)
from EvoScientist.paths import SessionDirs, Workspace

RUN = "20260930_120000"


@pytest.fixture
def served(workspace, monkeypatch) -> SessionDirs:
    """A server pinned to a run folder of ``workspace``."""
    dirs = SessionDirs(workspace, workspace.runs_dir / RUN)
    monkeypatch.setattr(
        "EvoScientist.langgraph_dev.folder_check.process_session_dirs", lambda: dirs
    )
    return dirs


def _config(dirs: SessionDirs | None) -> dict:
    return {"configurable": dirs.metadata() if dirs is not None else {}}


# ---------------------------------------------------------------------------
# The server's check
# ---------------------------------------------------------------------------


def test_runs_that_forward_no_folders_are_not_checked(served):
    check_run_folders(_config(None), works_in_folder=True)


def test_runs_for_the_served_folders_run(served, tmp_path):
    check_run_folders(_config(served), works_in_folder=True)

    # The same folders written another way: a symlink and a trailing slash.
    link = tmp_path / "link"
    link.symlink_to(served.workspace.root, target_is_directory=True)
    check_run_folders(
        {
            "configurable": {
                "workspace_dir": f"{link}/",
                "run_dir": str(link / "runs" / RUN),
            }
        },
        works_in_folder=True,
    )


def test_runs_for_another_workspace_are_refused(served, tmp_path):
    other = SessionDirs(Workspace(tmp_path / "other"))
    for works_in_folder in (True, False):
        with pytest.raises(RunFolderMismatchError) as exc:
            check_run_folders(_config(other), works_in_folder=works_in_folder)
        assert str(served.workspace.root) in str(exc.value)
        assert str(other.workspace.root) in str(exc.value)


def test_only_graphs_working_in_a_folder_check_the_run_folder(served):
    other_run = SessionDirs(served.workspace, served.workspace.runs_dir / "other")
    with pytest.raises(RunFolderMismatchError):
        check_run_folders(_config(other_run), works_in_folder=True)
    # Scheduled tasks and memory workers work in the root, whatever run
    # folder a session holds the server for.
    check_run_folders(_config(other_run), works_in_folder=False)
    check_run_folders(_config(SessionDirs(served.workspace)), works_in_folder=False)


def test_factory_checks_runs_only_and_returns_the_prebuilt_graph(served, tmp_path):
    graph = object()
    factory = folder_checked(graph, works_in_folder=True)
    mismatch = _config(SessionDirs(Workspace(tmp_path / "other")))

    reading = SimpleNamespace(execution_runtime=None)
    assert asyncio.run(factory(mismatch, reading)) is graph

    running = SimpleNamespace(execution_runtime=object())
    assert asyncio.run(factory(_config(served), running)) is graph
    with pytest.raises(RunFolderMismatchError):
        asyncio.run(factory(mismatch, running))


# ---------------------------------------------------------------------------
# Child runs forward their folders
# ---------------------------------------------------------------------------


def _launch(workspace: Workspace, work_dir=None) -> dict:
    """Start an async sub-agent and return the ``configurable`` it was sent."""
    from EvoScientist.middleware.expert_async_subagent import (
        EvoAsyncSubAgentMiddleware,
    )

    middleware = EvoAsyncSubAgentMiddleware(
        workspace=workspace,
        work_dir=work_dir,
        async_subagents=[
            {
                "name": "writing-agent",
                "description": "writes",
                "graph_id": "writing-agent",
                "url": "http://127.0.0.1:6174",
            }
        ],
    )
    start = next(t for t in middleware.tools if t.name == "start_async_task")
    client = MagicMock()
    client.threads.create.return_value = {"thread_id": "t1"}
    client.runs.create.return_value = {"run_id": "r1"}
    with patch(
        "EvoScientist.middleware.expert_async_subagent._ClientCache.get_sync",
        return_value=client,
    ):
        start.func(
            description="draft the paper",
            subagent_type="writing-agent",
            runtime=SimpleNamespace(tool_call_id="tc1", config={}),
        )
    return client.runs.create.call_args.kwargs["config"]["configurable"]


def test_async_tasks_forward_the_session_folders(workspace):
    run = workspace.runs_dir / RUN
    configurable = _launch(workspace, work_dir=run)
    assert configurable["workspace_dir"] == workspace.key
    assert configurable["run_dir"] == run.as_posix()


def test_daemon_async_tasks_forward_the_root_only(workspace):
    configurable = _launch(workspace)
    assert configurable["workspace_dir"] == workspace.key
    assert "run_dir" not in configurable


def test_memory_workers_and_the_linker_forward_the_root(workspace, tmp_path):
    from EvoScientist.memory.launch import (
        _memory_worker_run_payload,
        _observation_linker_run_payload,
    )
    from tests.test_observation_memory import _linker_context, _memory_source_context

    worker = _memory_worker_run_payload(
        context=_memory_source_context(
            memory_dir=tmp_path / "memories", workspace_dir=workspace.root
        ),
        thread_id="t1",
    )
    linker = _observation_linker_run_payload(
        context=_linker_context(
            memory_dir=tmp_path / "memories",
            workspace_dir=workspace.root,
            observation_ids=("o1",),
        ),
        thread_id="t2",
    )
    for payload in (worker, linker):
        assert payload["config"]["configurable"]["workspace_dir"] == workspace.key


def test_manual_autoskills_runs_forward_the_root(workspace, monkeypatch):
    import EvoScientist.memory.autoskills.schedule as schedule

    client = MagicMock()
    client.threads.create.return_value = {"thread_id": "t1"}
    client.runs.create.return_value = {"run_id": "r1"}
    monkeypatch.setattr(schedule, "get_langgraph_sync_client", lambda **_k: client)

    config = SimpleNamespace(
        memory_skill_synthesis_mode=SimpleNamespace(value="propose"),
        memory_skill_synthesis_cadence=SimpleNamespace(value="daily"),
        memory_skill_synthesis_time="03:00",
    )
    monkeypatch.setattr(schedule, "langgraph_dev_url", lambda _c: "http://x")
    schedule.run_autoskill_now(config, workspace_dir=workspace.root)

    configurable = client.runs.create.call_args.kwargs["config"]["configurable"]
    assert configurable["workspace_dir"] == workspace.key


def test_registered_graphs_check_the_folders_they_work_in(served, monkeypatch):
    """Graphs that work in the session's folder refuse another run folder;
    graphs built for the workspace root only check the workspace."""
    import importlib
    import json
    import sys
    from pathlib import Path

    import EvoScientist.EvoScientist as evo
    import EvoScientist.memory.agents as memory_agents
    import EvoScientist.subagents._factory as factory_mod
    import EvoScientist.subagents.expert_container_async as expert_async

    for module, name in (
        (factory_mod, "build_async_subagent_graph"),
        (expert_async, "build_expert_container_async_graph"),
        (memory_agents, "build_memory_worker_graph"),
        (memory_agents, "build_observation_linker_graph"),
        (memory_agents, "build_autoskills_graph"),
    ):
        monkeypatch.setattr(module, name, lambda *a, **k: object())
    monkeypatch.setattr(evo, "_get_default_agent", lambda: MagicMock())
    for name in (
        "EvoScientist.langgraph_dev.graphs",
        "EvoScientist.langgraph_dev.main_graph",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)

    registry = json.loads(
        (Path(evo.__file__).parent / "langgraph_dev" / "langgraph.json").read_text()
    )["graphs"]
    other_run = _config(
        SessionDirs(served.workspace, served.workspace.runs_dir / "other")
    )
    running = SimpleNamespace(execution_runtime=object())
    folder_graphs = {
        "EvoScientist",
        "writing-agent",
        "data-analysis-agent",
        "expert-container-async",
    }

    for graph_id, target in registry.items():
        module_path, attr = target.rsplit(":", 1)
        factory = getattr(importlib.import_module(module_path), attr)
        if graph_id in folder_graphs:
            with pytest.raises(RunFolderMismatchError):
                asyncio.run(factory(other_run, running))
        else:
            asyncio.run(factory(other_run, running))


def test_async_task_check_reports_why_a_run_was_refused(workspace):
    """langgraph-api records a failed run's error on its thread; the check
    shows it to the agent instead of a generic failure."""
    import json

    from EvoScientist.middleware.expert_async_subagent import (
        EvoAsyncSubAgentMiddleware,
    )

    middleware = EvoAsyncSubAgentMiddleware(
        workspace=workspace,
        async_subagents=[
            {
                "name": "writing-agent",
                "description": "writes",
                "graph_id": "writing-agent",
                "url": "http://127.0.0.1:6174",
            }
        ],
    )
    check = next(t for t in middleware.tools if t.name == "check_async_task")
    client = MagicMock()
    client.runs.get.return_value = {
        "status": "error",
        "thread_id": "t1",
        "run_id": "r1",
    }
    client.threads.get.return_value = {
        "error": {
            "error": "RunFolderMismatchError",
            "message": "This EvoScientist server serves workspace /a",
        }
    }
    task = {
        "task_id": "t1",
        "agent_name": "writing-agent",
        "thread_id": "t1",
        "run_id": "r1",
        "status": "running",
        "created_at": "2026-10-01T00:00:00Z",
        "last_checked_at": "2026-10-01T00:00:00Z",
        "last_updated_at": "2026-10-01T00:00:00Z",
    }
    with patch(
        "EvoScientist.middleware.expert_async_subagent._ClientCache.get_sync",
        return_value=client,
    ):
        command = check.func(
            task_id="t1",
            runtime=SimpleNamespace(
                state={"async_tasks": {"t1": task}}, tool_call_id="tc1"
            ),
        )

    result = json.loads(command.update["messages"][0].content)
    assert result["status"] == "error"
    assert "serves workspace /a" in result["error"]
