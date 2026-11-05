"""The served graphs are built per project and run folder, and kept."""

from __future__ import annotations

import pytest

from EvoScientist.langgraph_dev import factories
from EvoScientist.langgraph_dev.assistants import served_graph_ids
from EvoScientist.langgraph_dev.factories import GraphCache
from EvoScientist.paths import SessionDirs, Workspace


@pytest.fixture
def builds(monkeypatch):
    """Record each build as ``(graph_id, dirs)`` and return a fresh object."""
    calls: list[tuple[str, SessionDirs]] = []

    def build(graph_id, dirs):
        calls.append((graph_id, dirs))
        return object()

    monkeypatch.setattr(factories, "_build", build)
    return calls


def test_graph_cache_builds_a_project_graph_once(builds, tmp_path):
    graphs = GraphCache()
    dirs = SessionDirs(Workspace(tmp_path))
    assert graphs.get("EvoScientist", dirs) is graphs.get("EvoScientist", dirs)
    assert builds == [("EvoScientist", dirs)]


def test_graph_cache_shares_root_graphs_across_run_folders(builds, tmp_path):
    """Scheduled tasks and memory workers work in the root, whichever run
    folder the access comes from."""
    workspace = Workspace(tmp_path)
    graphs = GraphCache()
    first = graphs.get("scheduler", SessionDirs(workspace, workspace.runs_dir / "a"))
    second = graphs.get("scheduler", SessionDirs(workspace, workspace.runs_dir / "b"))
    assert first is second
    assert builds == [("scheduler", SessionDirs(workspace))]


def test_graph_cache_builds_folder_graphs_per_run_folder(builds, tmp_path):
    workspace = Workspace(tmp_path)
    graphs = GraphCache()
    run_a = SessionDirs(workspace, workspace.runs_dir / "a")
    run_b = SessionDirs(workspace, workspace.runs_dir / "b")
    assert graphs.get("writing-agent", run_a) is not graphs.get("writing-agent", run_b)
    assert builds == [("writing-agent", run_a), ("writing-agent", run_b)]


def test_graph_cache_drops_least_recently_used_run_folder_graphs(
    builds, monkeypatch, tmp_path
):
    """Project graphs stay; run-folder graphs past the cap are rebuilt."""
    monkeypatch.setattr(GraphCache, "RUN_FOLDER_GRAPHS", 2)
    workspace = Workspace(tmp_path)
    graphs = GraphCache()
    run_a, run_b, run_c = (
        SessionDirs(workspace, workspace.runs_dir / name) for name in "abc"
    )
    project = graphs.get("scheduler", run_a)
    graphs.get("writing-agent", run_a)
    graphs.get("writing-agent", run_b)
    graphs.get("writing-agent", run_a)
    graphs.get("writing-agent", run_c)
    builds.clear()

    graphs.get("writing-agent", run_a)
    graphs.get("writing-agent", run_b)
    assert graphs.get("scheduler", run_c) is project
    assert builds == [("writing-agent", run_b)]


def test_graph_cache_builds_one_graph_at_a_time(monkeypatch, tmp_path):
    """Builds share process-wide state, so concurrent accesses never overlap
    them, and the same graph is not built twice."""
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    active, overlapped, calls = [0], [False], []
    guard = threading.Lock()

    def build(graph_id, dirs):
        with guard:
            active[0] += 1
            overlapped[0] |= active[0] > 1
            calls.append(graph_id)
        time.sleep(0.05)
        with guard:
            active[0] -= 1
        return object()

    monkeypatch.setattr(factories, "_build", build)
    graphs = GraphCache()
    dirs = SessionDirs(Workspace(tmp_path))
    with ThreadPoolExecutor(4) as pool:
        list(
            pool.map(
                lambda g: graphs.get(g, dirs), ["EvoScientist"] * 2 + ["scheduler"] * 2
            )
        )
    assert not overlapped[0]
    assert sorted(calls) == ["EvoScientist", "scheduler"]


def test_every_served_graph_has_a_builder():
    assert set(factories.BUILDERS) == set(served_graph_ids())


def test_built_graphs_filter_their_checkpoint_reads(monkeypatch, tmp_path, tiny_graph):
    """Every graph gets the ``__class__`` swap, and the subclass survives
    ``Pregel.copy(update=...)`` - the call langgraph-api makes in ``get_graph``
    before yielding the graph to endpoint handlers."""
    from EvoScientist.langgraph_dev.main_graph import _EvoFilteredGraph

    monkeypatch.setitem(factories.BUILDERS, "EvoScientist", tiny_graph)
    graph = factories._build("EvoScientist", SessionDirs(Workspace(tmp_path)))
    assert isinstance(graph, _EvoFilteredGraph)
    assert isinstance(graph.copy(update={}), _EvoFilteredGraph)


def test_build_refuses_what_is_not_a_compiled_graph(monkeypatch, tmp_path):
    """Its checkpoint reads could not be filtered."""
    monkeypatch.setitem(factories.BUILDERS, "EvoScientist", lambda _dirs: object())
    with pytest.raises(TypeError, match="EvoScientist"):
        factories._build("EvoScientist", SessionDirs(Workspace(tmp_path)))


def test_build_failure_names_the_graph(monkeypatch, tmp_path):
    def fail(_dirs):
        raise ValueError("bad yaml")

    monkeypatch.setitem(factories.BUILDERS, "scheduler", fail)
    with pytest.raises(RuntimeError, match=r"'scheduler' failed to build.*bad yaml"):
        factories._build("scheduler", SessionDirs(Workspace(tmp_path)))


def test_graph_cache_rebuilds_when_the_mcp_generation_changes(
    builds, monkeypatch, tmp_path
):
    """A graph built from a failed MCP load is rebuilt once the failed
    server recovers and the tool cache bumps its generation (#696)."""
    generation = {"value": 3}
    monkeypatch.setattr(factories, "_mcp_tools_generation", lambda: generation["value"])
    graphs = GraphCache()
    dirs = SessionDirs(Workspace(tmp_path))

    first = graphs.get("EvoScientist", dirs)
    assert graphs.get("EvoScientist", dirs) is first

    generation["value"] = 4
    rebuilt = graphs.get("EvoScientist", dirs)

    assert rebuilt is not first
    assert len(builds) == 2
    assert graphs.get("EvoScientist", dirs) is rebuilt
    assert len(builds) == 2


def test_graph_cache_keeps_non_mcp_graphs_when_the_mcp_generation_changes(
    builds, monkeypatch, tmp_path
):
    """Graphs that never read MCP tools are not rebuilt when the MCP
    generation moves (a recovery only invalidates the MCP graphs)."""
    generation = {"value": 3}
    monkeypatch.setattr(factories, "_mcp_tools_generation", lambda: generation["value"])
    graphs = GraphCache()
    dirs = SessionDirs(Workspace(tmp_path))

    first = graphs.get("evomemory-observation-linker", dirs)
    assert graphs.get("evomemory-observation-linker", dirs) is first

    generation["value"] = 4
    assert graphs.get("evomemory-observation-linker", dirs) is first
    assert len(builds) == 1


def _reset_mcp_cache_state() -> None:
    """Reset the process-wide MCP cache state (also used as teardown)."""
    import EvoScientist.EvoScientist as agent_module

    agent_module._MCP_TOOLS_CACHE_KEY = None
    agent_module._MCP_TOOLS_CACHE_VALUE = None
    agent_module._MCP_TOOLS_CACHE_FAILED = set()
    agent_module._MCP_TOOLS_CACHE_LAST_ATTEMPT = {}
    agent_module._MCP_FAILED_SERVER_RETRY_DELAY = {}
    agent_module._MCP_TOOLS_CACHE_GENERATION = 0


def test_graph_for_retries_failed_mcp_servers_on_run_access(monkeypatch, tmp_path):
    """A cache hit in ``GraphCache.get`` never reaches the MCP loader, so
    the retry is triggered from ``_graph_for`` on a run access, using the
    real loader and the real generation (#696).

    First access builds the graph while ``flaky`` is down; a read access
    must not retry; once ``flaky`` recovers, the next run access retries
    it and the graph is rebuilt with its tools.
    """
    import EvoScientist.EvoScientist as agent_module

    _reset_mcp_cache_state()
    monkeypatch.setattr(agent_module, "_MCP_FAILED_SERVER_RETRY_SECONDS", 0.0)

    behavior = {"good": "ok", "flaky": "fail"}
    config = {
        name: {"transport": "stdio", "command": f"demo-{name}"} for name in behavior
    }
    monkeypatch.setattr("EvoScientist.mcp.client.load_mcp_config", lambda: dict(config))
    calls: list[dict] = []

    def fake_load_mcp_tools(config=None, on_progress=None, **_kwargs):
        calls.append(dict(config or {}))
        routed: dict[str, list] = {}
        for name in config or {}:
            if behavior[name] == "fail":
                if on_progress is not None:
                    on_progress("error", name, "timed out")
                continue
            if on_progress is not None:
                on_progress("success", name, "1")
            routed.setdefault("main", []).append(f"tool-from-{name}")
        return routed

    monkeypatch.setattr("EvoScientist.mcp.load_mcp_tools", fake_load_mcp_tools)

    builds: list[tuple[str, SessionDirs]] = []

    def build(graph_id, dirs):
        # Like the real main-graph build, loading MCP tools is part of
        # building the graph.
        agent_module._load_mcp_tools_cached()
        builds.append((graph_id, dirs))
        return object()

    monkeypatch.setattr(factories, "_build", build)
    graphs = GraphCache()
    monkeypatch.setattr(factories, "GRAPHS", graphs)
    dirs = SessionDirs(Workspace(tmp_path))
    monkeypatch.setattr(factories, "resolve_folders", lambda *args: (dirs, None))

    first, _stamp = factories._graph_for(
        "EvoScientist", {"configurable": {}}, "threads.create_run"
    )
    assert agent_module._MCP_TOOLS_CACHE_FAILED == {"flaky"}
    assert len(builds) == 1

    # A read access must not wait on the network.
    read, _stamp = factories._graph_for(
        "EvoScientist", {"configurable": {}}, "threads.read"
    )
    assert read is first
    assert agent_module._MCP_TOOLS_CACHE_FAILED == {"flaky"}
    assert len(builds) == 1
    assert len(calls) == 1

    # The server recovers; the next run access retries it and rebuilds.
    behavior["flaky"] = "ok"
    second, _stamp = factories._graph_for(
        "EvoScientist", {"configurable": {}}, "threads.create_run"
    )
    assert second is not first
    assert len(builds) == 2
    assert agent_module._MCP_TOOLS_CACHE_FAILED == set()
    assert [sorted(c) for c in calls] == [["flaky", "good"], ["flaky"]]
    _reset_mcp_cache_state()
