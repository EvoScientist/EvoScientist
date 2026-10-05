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
from EvoScientist.sessions import FOLDER_GRAPH_IDS

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


def test_check_run_folders_refuses_unreadable_folders(served):
    """Present but unreadable is a mismatch, not an unchecked run."""
    for workspace_dir in ("/tmp/evil\x00x", 123):
        with pytest.raises(RunFolderMismatchError, match="cannot read"):
            check_run_folders(
                {"configurable": {"workspace_dir": workspace_dir}},
                works_in_folder=True,
            )


def test_check_run_folders_refuses_run_dir_outside_workspace(served, tmp_path):
    with pytest.raises(RunFolderMismatchError):
        check_run_folders(
            {
                "configurable": {
                    "workspace_dir": served.workspace.key,
                    "run_dir": str(tmp_path / "elsewhere" / "runs" / RUN),
                }
            },
            works_in_folder=True,
        )


def test_folder_checked_factory_matches_server_signature():
    """langgraph-api classifies the factory by its ``runtime`` annotation."""
    import inspect
    import typing

    from langgraph_sdk.runtime import ServerRuntime

    factory = folder_checked(object(), graph_id="writing-agent")
    assert list(inspect.signature(factory).parameters) == ["config", "runtime"]
    assert typing.get_type_hints(factory)["runtime"] is ServerRuntime


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
    factory = folder_checked(graph, graph_id="writing-agent")
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
    assert FOLDER_GRAPH_IDS <= registry.keys()

    for graph_id, target in registry.items():
        module_path, attr = target.rsplit(":", 1)
        factory = getattr(importlib.import_module(module_path), attr)
        if graph_id in FOLDER_GRAPH_IDS:
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
    runtime = SimpleNamespace(state={"async_tasks": {"t1": task}}, tool_call_id="tc1")
    with patch(
        "EvoScientist.middleware.expert_async_subagent._ClientCache.get_sync",
        return_value=client,
    ):
        first = check.func(task_id="t1", runtime=runtime)
        second = check.func(task_id="t1", runtime=runtime)

    for command in (first, second):
        result = json.loads(command.update["messages"][0].content)
        assert result["status"] == "error"
        assert "serves workspace /a" in result["error"]
    # A failed run's reason does not change; it is looked up once.
    client.threads.get.assert_called_once_with("t1")


def test_new_run_folders_never_reuse_an_existing_one(workspace, monkeypatch):
    """Two sessions starting in the same second get different run folders."""
    from datetime import datetime

    import EvoScientist.cli.agent as agent

    class _SameSecond(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 1, 12, 0, 0)

    monkeypatch.setattr(agent, "datetime", _SameSecond)
    first = agent._create_run_dir(workspace)
    second = agent._create_run_dir(workspace)

    assert first != second
    assert first.is_dir()
    assert second.is_dir()


def test_create_run_dir_skips_dangling_symlink(workspace):
    import EvoScientist.cli.agent as agent

    workspace.runs_dir.mkdir(parents=True)
    (workspace.runs_dir / "exp").symlink_to(workspace.root / "does-not-exist")

    assert agent._create_run_dir(workspace, "exp") == workspace.runs_dir / "exp_1"


# ---------------------------------------------------------------------------
# Failed child runs report why
# ---------------------------------------------------------------------------


def test_run_errors_keep_exception_type():
    """langgraph-api stores a placeholder text for most exception types."""
    from EvoScientist.llm.patches import _RunErrors

    failed = {"status": "error", "thread_id": "t1", "run_id": "r1"}
    attached = _RunErrors().attach(
        failed,
        {"error": {"error": "RateLimitError", "message": "An internal error occurred"}},
    )
    assert "RateLimitError" in attached["error"]


def test_run_errors_return_run_without_id_unchanged():
    from EvoScientist.llm.patches import _RunErrors

    run = {"status": "error", "thread_id": "t1"}
    thread = {"error": {"error": "RuntimeError", "message": "boom"}}
    assert _RunErrors().attach(run, thread) == run


async def test_acheck_async_task_reports_failure_reason(workspace):
    """``acheck_async_task`` (langgraph dev, TUI) reads the thread's error too."""
    import json
    from unittest.mock import AsyncMock

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

    def _task(task_id: str) -> dict:
        return {
            "task_id": task_id,
            "agent_name": "writing-agent",
            "thread_id": task_id,
            "run_id": f"r-{task_id}",
            "status": "running",
            "created_at": "2026-10-01T00:00:00Z",
            "last_checked_at": "2026-10-01T00:00:00Z",
            "last_updated_at": "2026-10-01T00:00:00Z",
        }

    def _thread(thread_id: str, **_k) -> dict:
        if thread_id != "t1":
            raise RuntimeError("thread unreadable")
        return {"error": {"error": "RunFolderMismatchError", "message": "no"}}

    client = MagicMock()
    client.runs.get = AsyncMock(
        side_effect=lambda thread_id, run_id, **_k: {
            "status": "error",
            "thread_id": thread_id,
            "run_id": run_id,
        }
    )
    client.threads.get = AsyncMock(side_effect=_thread)
    runtime = SimpleNamespace(
        state={"async_tasks": {"t1": _task("t1"), "t2": _task("t2")}},
        tool_call_id="tc1",
    )
    with patch(
        "EvoScientist.middleware.expert_async_subagent._ClientCache.get_async",
        return_value=client,
    ):
        refused = await check.coroutine(task_id="t1", runtime=runtime)
        unreadable = await check.coroutine(task_id="t2", runtime=runtime)

    result = json.loads(refused.update["messages"][0].content)
    assert "RunFolderMismatchError" in result["error"]
    # When the thread cannot be read, the run is reported as it is.
    result = json.loads(unreadable.update["messages"][0].content)
    assert result["status"] == "error"


def test_merge_runs_config_replaces_caller_folders():
    """The launching graph's folders win over a caller's stale ones."""
    from EvoScientist.llm.patches import _merge_runs_config_kwargs

    caller = {"workspace_dir": "/c", "run_dir": "/c/runs/r", "thread_id": "t1"}
    with patch("EvoScientist.llm.patches._read_cfg_configurable", lambda: {}):
        merged = _merge_runs_config_kwargs(
            {"config": {"configurable": caller}}, {"workspace_dir": "/ws"}
        )
    assert merged["config"]["configurable"] == {
        "workspace_dir": "/ws",
        "thread_id": "t1",
    }


def test_server_gateway_run_config_forwards_target_folders(monkeypatch):
    """The gateway's own main-agent runs carry their folders like child runs,
    over any a caller put in ``configurable_extra``."""
    import EvoScientist.EvoScientist as evo
    from EvoScientist.gateway.server import LangGraphServerGateway
    from EvoScientist.gateway.types import GraphTarget

    monkeypatch.setattr(
        evo,
        "_ensure_config",
        lambda: SimpleNamespace(model="m", provider="p", recursion_limit=None),
    )
    monkeypatch.setattr(
        "EvoScientist.backends.hitl_suppressed_for_run", lambda _cfg: False
    )
    gateway = object.__new__(LangGraphServerGateway)
    stale = {"workspace_dir": "/b", "run_dir": "/b/runs/y", "active_teams": ["t"]}

    run_mode = gateway._resolve_run_config(
        "t1", stale, target=GraphTarget(workspace_dir="/ws", run_dir="/ws/runs/x")
    )["configurable"]
    daemon = gateway._resolve_run_config(
        "t1", stale, target=GraphTarget(workspace_dir="/ws")
    )["configurable"]
    bare = gateway._resolve_run_config("t1", None)["configurable"]

    assert (run_mode["workspace_dir"], run_mode["run_dir"]) == ("/ws", "/ws/runs/x")
    assert run_mode["model"] == "m"
    assert run_mode["active_teams"] == ["t"]
    assert daemon["workspace_dir"] == "/ws"
    assert "run_dir" not in daemon
    assert "workspace_dir" not in bare


# ---------------------------------------------------------------------------
# The server moves with the session
# ---------------------------------------------------------------------------


async def _sync_order(workspace, *, pinned_elsewhere: bool) -> list[str]:
    """Sync the server to ``workspace`` and return what ran, in order."""
    from EvoScientist.cli import commands

    order: list[str] = []
    with (
        patch.object(
            commands, "_wait_for_memory_workers", lambda **_k: order.append("memory")
        ),
        patch(
            "EvoScientist.langgraph_dev.manager.owned_server_pinned_elsewhere",
            lambda *a, **k: pinned_elsewhere,
        ),
        patch(
            "EvoScientist.langgraph_dev.manager.ensure_langgraph_dev",
            lambda *a, **k: order.append("move"),
        ),
        patch.object(commands, "_reconcile_autoskill_schedule", lambda *a, **k: None),
        patch.object(commands, "_adopt_stale_scheduled_tasks", lambda **_k: None),
    ):
        await commands._sync_background_agent_server_workspace(
            MagicMock(), dirs=SessionDirs(workspace)
        )
    return order


async def test_server_move_waits_for_memory_work(workspace):
    """Moving the pinned server stops runs in flight, so the previous
    session's memory work finishes before it moves."""
    assert await _sync_order(workspace, pinned_elsewhere=True) == ["memory", "move"]


async def test_server_sync_to_served_folders_skips_memory_wait(workspace):
    """A daemon-mode /resume syncs to the folders the server already serves."""
    assert await _sync_order(workspace, pinned_elsewhere=False) == ["move"]


def test_owned_server_pinned_elsewhere_compares_folder_pair(workspace, monkeypatch):
    from EvoScientist.langgraph_dev import manager

    alive = SimpleNamespace(poll=lambda: None)
    run = workspace.runs_dir / RUN
    monkeypatch.setattr(manager, "_PROCESS", alive)
    monkeypatch.setattr(manager, "_PROCESS_WORKSPACE", workspace.root)
    monkeypatch.setattr(manager, "_PROCESS_RUN_DIR", run)
    assert manager.owned_server_pinned_elsewhere(workspace.root, run) is False
    assert manager.owned_server_pinned_elsewhere(workspace.root, None) is True
    assert manager.owned_server_pinned_elsewhere(workspace.root / "other", run) is True
    monkeypatch.setattr(manager, "_PROCESS", None)
    assert manager.owned_server_pinned_elsewhere(workspace.root, None) is False


def test_ensure_async_subagent_server_refused_removes_run_dir(workspace):
    """A session refused by another one's server leaves no empty run folder."""
    import typer

    from EvoScientist.cli import commands
    from EvoScientist.langgraph_dev.manager import WorkspaceMismatchError

    run = workspace.runs_dir / RUN
    run.mkdir(parents=True)

    def _refuse(*_a, **_k):
        raise WorkspaceMismatchError("held by another session")

    with (
        patch("EvoScientist.langgraph_dev.manager.ensure_langgraph_dev", _refuse),
        pytest.raises(typer.Exit),
    ):
        commands._ensure_async_subagent_server(
            MagicMock(), dirs=SessionDirs(workspace, run)
        )

    assert not run.exists()
    assert workspace.runs_dir.is_dir()


async def test_restore_thread_dirs_removes_unused_run_dir(workspace, tmp_path):
    """A ``--mode=run`` resume works in the thread's folders, so the run
    folder made at startup goes."""
    from unittest.mock import AsyncMock

    from EvoScientist.cli import commands

    started = SessionDirs(workspace, workspace.runs_dir / RUN)
    started.run_dir.mkdir(parents=True)
    stored = SessionDirs(Workspace(tmp_path / "other"))
    gateway = SimpleNamespace(
        get_thread_metadata=AsyncMock(return_value=stored.metadata())
    )

    with patch.object(commands, "_sync_background_agent_server_workspace", AsyncMock()):
        restored = await commands._restore_thread_dirs(
            "t1",
            dirs=started,
            served=started,
            graph_gateway=gateway,
            config=MagicMock(),
        )

    assert restored == stored
    assert not started.run_dir.exists()


async def _new_session(workspace, sync) -> SessionDirs:
    """Run the Rich CLI ``/new`` folder switch of a ``--mode=run`` session."""
    from EvoScientist.cli import commands
    from EvoScientist.cli.interactive import _new_session_dirs

    with patch.object(commands, "_sync_background_agent_server_workspace", sync):
        return await _new_session_dirs(
            SessionDirs(workspace),
            config=MagicMock(),
            backend=None,
            mode="run",
            run_name="exp",
        )


async def test_new_session_dirs_move_server_to_new_run_dir(workspace):
    from unittest.mock import AsyncMock

    sync = AsyncMock()
    new_dirs = await _new_session(workspace, sync)

    assert new_dirs == SessionDirs(workspace, workspace.runs_dir / "exp")
    assert new_dirs.run_dir.is_dir()
    assert sync.await_args.kwargs["dirs"] == new_dirs


async def test_new_session_dirs_refused_removes_run_dir(workspace):
    """Another session holds the server: ``/new`` fails and the session
    keeps its folders, with no empty run folder left behind."""
    from unittest.mock import AsyncMock

    from EvoScientist.langgraph_dev.manager import WorkspaceMismatchError

    sync = AsyncMock(side_effect=WorkspaceMismatchError("held by another session"))
    with pytest.raises(WorkspaceMismatchError):
        await _new_session(workspace, sync)

    assert not (workspace.runs_dir / "exp").exists()


async def test_new_session_dirs_survive_failed_sync(workspace):
    """Any other sync failure starts the new session without background work."""
    from unittest.mock import AsyncMock

    new_dirs = await _new_session(workspace, AsyncMock(side_effect=OSError("down")))

    assert new_dirs == SessionDirs(workspace, workspace.runs_dir / "exp")
    assert new_dirs.run_dir.is_dir()
