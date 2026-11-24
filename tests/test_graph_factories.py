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


# ---------------------------------------------------------------------------
# mcp.yaml edits drop the cached graphs
# ---------------------------------------------------------------------------


def _mcp_yaml():
    """The mcp.yaml the server reads (its config folder, isolated per test)."""
    from EvoScientist.mcp import client as mcp_client

    return mcp_client.USER_MCP_CONFIG


def _write_mcp_yaml(text: str, tick: int):
    """Write *text* with an mtime of *tick* seconds, so consecutive writes
    differ in modification time however coarse the filesystem's clock is."""
    import os

    path = _mcp_yaml()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    os.utime(path, ns=(tick * 10**9, tick * 10**9))
    return path


def test_graph_cache_rebuilds_after_an_mcp_yaml_edit(builds, tmp_path):
    """An edit applies to the next access: the cached graph is dropped and
    the next ``get`` builds again with the new file's tools."""
    _write_mcp_yaml("servers: {}\n", 1)
    graphs = GraphCache()
    dirs = SessionDirs(Workspace(tmp_path))
    first = graphs.get("EvoScientist", dirs)
    _write_mcp_yaml("servers: {fs: {}}\n", 2)
    assert graphs.get("EvoScientist", dirs) is not first
    assert builds == [("EvoScientist", dirs), ("EvoScientist", dirs)]


def test_graph_cache_keeps_graphs_while_mcp_yaml_is_unchanged(builds, tmp_path):
    _write_mcp_yaml("servers: {}\n", 1)
    graphs = GraphCache()
    dirs = SessionDirs(Workspace(tmp_path))
    graph = graphs.get("EvoScientist", dirs)
    assert graphs.get("EvoScientist", dirs) is graph
    assert builds == [("EvoScientist", dirs)]


def test_graph_cache_rebuilds_lazily_after_an_edit(builds, tmp_path):
    """A project not accessed since the edit is not rebuilt."""
    _write_mcp_yaml("servers: {}\n", 1)
    graphs = GraphCache()
    a = SessionDirs(Workspace(tmp_path / "a"))
    b = SessionDirs(Workspace(tmp_path / "b"))
    graphs.get("EvoScientist", a)
    graphs.get("EvoScientist", b)
    _write_mcp_yaml("servers: {fs: {}}\n", 2)
    builds.clear()
    graphs.get("EvoScientist", a)
    assert builds == [("EvoScientist", a)]


def test_graph_cache_drops_run_folder_graphs_on_an_edit(builds, tmp_path):
    workspace = Workspace(tmp_path)
    _write_mcp_yaml("servers: {}\n", 1)
    graphs = GraphCache()
    run_a = SessionDirs(workspace, workspace.runs_dir / "a")
    graphs.get("writing-agent", run_a)
    _write_mcp_yaml("servers: {fs: {}}\n", 2)
    builds.clear()
    graphs.get("writing-agent", run_a)
    assert builds == [("writing-agent", run_a)]


def test_graph_cache_catches_a_same_size_edit(builds, tmp_path):
    """An edit that keeps the file's size is caught by the modification
    time, and the content digest then tells it from a bare touch."""
    _write_mcp_yaml("servers: {fs: 1}\n", 1)
    graphs = GraphCache()
    dirs = SessionDirs(Workspace(tmp_path))
    first = graphs.get("EvoScientist", dirs)
    _write_mcp_yaml("servers: {fs: 2}\n", 2)
    assert graphs.get("EvoScientist", dirs) is not first


def test_graph_cache_keeps_graphs_while_mcp_yaml_does_not_parse(builds, tmp_path):
    """A half-saved mcp.yaml keeps the graphs the server has (rebuilding
    would read it as ``{}`` and drop every MCP tool); once the file parses,
    the next access rebuilds."""
    _write_mcp_yaml("servers: {}\n", 1)
    graphs = GraphCache()
    dirs = SessionDirs(Workspace(tmp_path))
    first = graphs.get("EvoScientist", dirs)
    _write_mcp_yaml("servers: [unclosed\n", 2)
    assert graphs.get("EvoScientist", dirs) is first
    _write_mcp_yaml("servers: {fs: {}}\n", 3)
    assert graphs.get("EvoScientist", dirs) is not first


def test_graph_cache_rebuilds_when_mcp_yaml_is_removed(builds, tmp_path):
    """Removing the file drops the MCP tools too."""
    path = _write_mcp_yaml("servers: {}\n", 1)
    graphs = GraphCache()
    dirs = SessionDirs(Workspace(tmp_path))
    first = graphs.get("EvoScientist", dirs)
    path.unlink()
    assert graphs.get("EvoScientist", dirs) is not first


def test_graph_cache_edit_during_a_build_is_not_cached(builds, monkeypatch, tmp_path):
    """A build that reads the file before an edit must not be cached under
    the edit's stamp: it is returned to its access but the next access
    builds again."""
    _write_mcp_yaml("servers: {}\n", 1)

    def build_during_edit(graph_id, dirs):
        _write_mcp_yaml("servers: {fs: {}}\n", 2)
        builds.append((graph_id, dirs))
        return object()

    monkeypatch.setattr(factories, "_build", build_during_edit)
    graphs = GraphCache()
    dirs = SessionDirs(Workspace(tmp_path))
    graphs.get("EvoScientist", dirs)
    builds.clear()
    graphs.get("EvoScientist", dirs)
    assert builds == [("EvoScientist", dirs)]


def test_graph_cache_never_caches_a_graph_built_on_unparsable_mcp_yaml(
    builds, tmp_path
):
    """A graph built while ``mcp.yaml`` cannot be parsed binds the empty
    MCP config: it is served to its access but never cached, so restoring
    the same bytes later cannot leave a tool-less graph behind."""
    _write_mcp_yaml("servers: {}\n", 1)
    graphs = GraphCache()
    dirs = SessionDirs(Workspace(tmp_path))
    first = graphs.get("EvoScientist", dirs)
    other = SessionDirs(Workspace(tmp_path / "other"))
    _write_mcp_yaml("servers: [unclosed\n", 2)
    tool_less = graphs.get("EvoScientist", other)
    assert tool_less is not first
    # The graphs the server already had are still served while the file
    # does not parse.
    assert graphs.get("EvoScientist", dirs) is first
    _write_mcp_yaml("servers: {}\n", 3)
    # The restored file digests to the original stamp; the tool-less build
    # must still not have been cached under ``other``'s key.
    builds.clear()
    assert graphs.get("EvoScientist", other) is not tool_less
    assert builds == [("EvoScientist", other)]


def test_graph_cache_invalidation_during_a_build_is_not_cached(
    builds, monkeypatch, tmp_path
):
    """A concurrent access can observe an ``mcp.yaml`` edit first and
    invalidate the cache itself: the overlapping build's closing file check
    then sees no change, so only the generation says its result is stale."""
    _write_mcp_yaml("servers: {}\n", 1)
    graphs = GraphCache()
    dirs = SessionDirs(Workspace(tmp_path))
    graphs.get("EvoScientist", dirs)

    def build_during_invalidation(graph_id, build_dirs):
        _write_mcp_yaml("servers: {fs: {}}\n", 2)
        graphs._drop_graphs_if_mcp_yaml_changed()
        builds.append((graph_id, build_dirs))
        return object()

    monkeypatch.setattr(factories, "_build", build_during_invalidation)
    other = SessionDirs(Workspace(tmp_path / "other"))
    stale = graphs.get("EvoScientist", other)
    builds.clear()
    assert graphs.get("EvoScientist", other) is not stale
    assert builds == [("EvoScientist", other)]
