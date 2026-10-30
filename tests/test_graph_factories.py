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
