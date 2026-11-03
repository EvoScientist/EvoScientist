"""Graph factories: each served graph is built per project and run folder.

langgraph-api calls a registered factory on every access to a graph: runs,
state and history reads, state updates and schema reads. The factory picks
the folders the access is for and returns the graph built for them from a
cache, built once per (graph, project, run folder).

Graphs in ``FOLDER_GRAPH_IDS`` work in a thread's run folder, so their key
includes it; every other graph works in the project root and is built once
per project.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import Any

from langgraph_sdk.runtime import ServerRuntime
from starlette.exceptions import HTTPException

from ..paths import SessionDirs, Workspace
from ..sessions import (
    FOLDER_GRAPH_IDS,
    RESOLVED_FOLDERS,
    ThreadBindingError,
    thread_binding,
)
from . import registry
from .folder_check import (
    ProjectNotServedError,
    RunFolderMismatchError,
    resolve_folders,
)

logger = logging.getLogger(__name__)


def _main(dirs: SessionDirs) -> Any:
    from ..EvoScientist import build_main_graph

    return build_main_graph(dirs.workspace, work_dir=dirs.work_dir)


def _async_subagent(name: str) -> Callable[[SessionDirs], Any]:
    def build(dirs: SessionDirs) -> Any:
        from ..subagents._factory import build_async_subagent_graph

        if name in FOLDER_GRAPH_IDS:
            return build_async_subagent_graph(
                name, workspace=dirs.workspace, work_dir=dirs.work_dir
            )
        # Scheduled tasks belong to the workspace, not to one run.
        return build_async_subagent_graph(name, workspace=dirs.workspace)

    return build


def _expert_container(dirs: SessionDirs) -> Any:
    # One graph for every installed expert skill, parameterised per invocation
    # by the ``skill_name`` payload; the loader middleware resolves the skill
    # body at model-call time.
    from ..subagents.expert_container_async import build_expert_container_async_graph

    return build_expert_container_async_graph(
        workspace=dirs.workspace, work_dir=dirs.work_dir
    )


def _memory_worker(source: str) -> Callable[[SessionDirs], Any]:
    def build(dirs: SessionDirs) -> Any:
        from ..memory.agents import build_memory_worker_graph
        from ..memory.types import MemorySourceType

        return build_memory_worker_graph(
            MemorySourceType[source], workspace=dirs.workspace
        )

    return build


def _observation_linker(dirs: SessionDirs) -> Any:
    from ..memory.agents import build_observation_linker_graph

    return build_observation_linker_graph(workspace=dirs.workspace)


def _autoskills(dirs: SessionDirs) -> Any:
    from ..memory.agents import build_autoskills_graph

    return build_autoskills_graph(workspace=dirs.workspace)


def _proactive(dirs: SessionDirs) -> Any:
    # Proactive pushes (#263): one tick of the workspace's proactive cron.
    from ..proactive.graph import build_proactive_graph

    return build_proactive_graph(workspace=dirs.workspace)


# How each graph in ``langgraph.json`` is built. Blocking: builds load MCP and
# touch the disk.
BUILDERS: dict[str, Callable[[SessionDirs], Any]] = {
    "EvoScientist": _main,
    "writing-agent": _async_subagent("writing-agent"),
    "data-analysis-agent": _async_subagent("data-analysis-agent"),
    "scheduler": _async_subagent("scheduler"),
    "expert-container-async": _expert_container,
    "evomemory-subagent-worker": _memory_worker("SUBAGENT"),
    "evomemory-turn-worker": _memory_worker("TURN"),
    "evomemory-observation-linker": _observation_linker,
    "evomemory-autoskills": _autoskills,
    "proactive": _proactive,
}


def _build(graph_id: str, dirs: SessionDirs) -> Any:
    """Build *graph_id* for *dirs*, its checkpoint reads filtered."""
    from langgraph.graph.state import CompiledStateGraph

    from .main_graph import _EvoFilteredGraph

    try:
        graph = BUILDERS[graph_id](dirs)
    except Exception as exc:
        raise RuntimeError(
            f"Graph {graph_id!r} failed to build for {dirs.workspace.root}: {exc}"
        ) from exc
    # Strip private state fields from checkpoint reads (see ``main_graph``).
    if not isinstance(graph, CompiledStateGraph):
        raise TypeError(f"Graph {graph_id!r} is not a compiled graph: {graph!r}")
    graph.__class__ = _EvoFilteredGraph
    return graph


class GraphCache:
    """Built graphs by (graph, project, run folder).

    Builds share process-wide state (the active config, chat model and MCP
    caches, ``os.environ``), so they run one at a time; a graph already built
    is returned without waiting for another's build. A project's graphs stay
    built; graphs built for a run folder are dropped, least recently used
    first, past ``RUN_FOLDER_GRAPHS``. An unregistered project's graphs stay
    too: resolving an access refuses that project before its graph is asked
    for.
    """

    RUN_FOLDER_GRAPHS = 32

    def __init__(self) -> None:
        self._build_lock = threading.Lock()
        self._lock = threading.Lock()
        self._graphs: OrderedDict[tuple[str, str, str | None], Any] = OrderedDict()

    def _cached(self, key: tuple[str, str, str | None]) -> Any:
        with self._lock:
            graph = self._graphs.get(key)
            if graph is not None and key[2] is not None:
                self._graphs.move_to_end(key)
            return graph

    def get(self, graph_id: str, dirs: SessionDirs) -> Any:
        """The graph for *dirs*, built on a miss. Blocking."""
        if graph_id not in FOLDER_GRAPH_IDS:
            dirs = SessionDirs(dirs.workspace)
        run_dir = dirs.run_dir.as_posix() if dirs.run_dir is not None else None
        key = (graph_id, dirs.workspace.key, run_dir)
        if (graph := self._cached(key)) is not None:
            return graph
        with self._build_lock:
            if (graph := self._cached(key)) is not None:
                return graph
            graph = _build(graph_id, dirs)
            with self._lock:
                self._graphs[key] = graph
                run_folder_keys = [k for k in self._graphs if k[2] is not None]
                for stale in run_folder_keys[: -self.RUN_FOLDER_GRAPHS]:
                    del self._graphs[stale]
            return graph


GRAPHS = GraphCache()


def _graph_for(
    graph_id: str, config: dict[str, Any], access: str
) -> tuple[Any, SessionDirs | None]:
    """The graph for an access, and the folders its writes are stamped with."""
    configurable = config.get("configurable") or {}
    thread_id = configurable.get("thread_id")
    writes = access in ("threads.create_run", "threads.update")
    bound = thread_binding(thread_id) if thread_id and writes else None
    dirs, stamp = resolve_folders(
        graph_id, access, configurable, registry.REGISTRY, bound
    )
    return GRAPHS.get(graph_id, dirs), stamp


async def _drop_system_cron(cron_id: str, workspace: Workspace) -> None:
    """Delete *cron_id* if it is one of *workspace*'s system crons; a failure is
    logged, so the refusal still reaches the run's client."""
    from .server_crons import drop_system_cron

    try:
        await drop_system_cron(cron_id, workspace)
    except Exception:
        logger.exception("Could not delete cron %s of %s.", cron_id, workspace.root)


def served_graph(graph_id: str) -> Callable[..., Awaitable[Any]]:
    """The factory registered for *graph_id* in ``langgraph.json``.

    Picks the folders of each access from the registry, its thread's binding
    and its request (``folder_check.resolve_folders``), refusing an access
    this server must not serve, and hands them to the checkpointer to stamp.
    A system cron's run refused because its project's folder is missing
    deletes the cron (``server_crons.drop_system_cron``).
    """

    async def factory(config: dict[str, Any], runtime: ServerRuntime) -> Any:
        try:
            # Reading the binding, resolving folders and building touch the
            # disk, which blockbuster refuses on the event loop.
            graph, stamp = await asyncio.to_thread(
                _graph_for, graph_id, config, runtime.access_context
            )
        except (RunFolderMismatchError, ThreadBindingError) as exc:
            run = runtime.access_context == "threads.create_run"
            cron_id = (config.get("configurable") or {}).get("cron_id")
            missing = isinstance(exc, ProjectNotServedError) and exc.missing_folder
            if run and cron_id and missing:
                await _drop_system_cron(cron_id, exc.workspace)
            if run:
                # A run's error reaches its client by name.
                raise
            status = 409 if isinstance(exc, RunFolderMismatchError) else 503
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        # Set here, in the access's own task, so the checkpoint writes of the
        # run or update that follow see it; a value set in the worker thread
        # would stay there.
        RESOLVED_FOLDERS.set(stamp)
        return graph

    return factory
