"""Shared types for graph/thread gateway implementations."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.types import Command

    from ..middleware.events import SessionEvents

GraphEvent: TypeAlias = dict[str, Any]
# String alias keeps this module langgraph-free at import time (~950 modules).
GraphRunInput: TypeAlias = "str | Command"
GraphStateValues: TypeAlias = dict[str, Any]
DEFAULT_GRAPH_ID = "EvoScientist"


def resolve_per_run_config(
    thread_id: str,
    configurable_extra: Mapping[str, Any] | None,
    *,
    per_run_overrides: Mapping[str, Any] | None = None,
    recursion_limit: int | None = None,
) -> dict[str, Any]:
    """Assemble the per-run LangGraph config for a gateway stream call.

    Pure assembly - this module reads no config. The server gateway extracts
    the per-run overrides (``configurable.model`` /
    ``configurable.model_provider`` from the live session config,
    ``recursion_limit`` as a first-class ``RunnableConfig`` key) and passes
    them in; a per-call ``recursion_limit`` overrides the server's
    construction-time ``.with_config`` binding, so a keepalive server picks
    up the client's live limit per run instead of at restart. The local
    backend passes neither: its agent is rebuilt on model switches and
    already binds ``recursion_limit`` at construction from the same live
    config, so per-run injection there is redundant.

    Merges, in precedence order (lowest to highest):

    1. ``per_run_overrides`` (server backend session defaults),
    2. caller-supplied ``configurable_extra`` (e.g. ``active_teams``) - an
       explicit per-run injection is more specific than the session default,
    3. ``thread_id`` - structural key, always set last.
    """
    configurable: dict[str, Any] = dict(per_run_overrides or {})
    if configurable_extra:
        configurable.update(configurable_extra)
    configurable["thread_id"] = thread_id

    run_config: dict[str, Any] = {"configurable": configurable}
    if recursion_limit is not None:
        run_config["recursion_limit"] = recursion_limit
    return run_config


@dataclass(frozen=True, slots=True)
class GraphTarget:
    """Identifies the graph/workspace a thread operation targets.

    ``local_graph`` is the in-process execution handle required only by the
    local backend. Server backends select execution via ``graph_id``.
    """

    graph_id: str = DEFAULT_GRAPH_ID
    workspace_dir: str | None = None
    local_graph: CompiledStateGraph | None = None


@dataclass(frozen=True, slots=True)
class RunRequest:
    """A graph turn request, independent of the UI that initiated it."""

    message: GraphRunInput
    thread_id: str
    metadata: dict[str, Any] | None = None
    media: list[str] | None = None
    target: GraphTarget | None = None
    configurable_extra: dict[str, Any] | None = None
    """Extra keys to merge into the LangGraph ``configurable`` dict alongside
    ``thread_id`` — e.g. ``{"active_teams": [...]}`` from the TUI
    ``/expert`` command. WebUI callers achieve the same effect via
    ``langgraph_sdk``'s ``config.configurable`` on their own; this field is
    the local-gateway equivalent so CLI / TUI / headless serve can bias
    the run identically."""


@dataclass(frozen=True, slots=True)
class ThreadResolution:
    """Result of resolving an exact or prefix thread id."""

    thread_id: str | None
    matches: tuple[str, ...] = ()

    @property
    def found(self) -> bool:
        return self.thread_id is not None

    @property
    def ambiguous(self) -> bool:
        return self.thread_id is None and bool(self.matches)


class ThreadStore(Protocol):
    """Thread persistence operations used by graph gateways."""

    def generate_thread_id(self) -> str:
        """Generate a new thread id."""

    async def list_threads(
        self,
        *,
        limit: int = 20,
        include_message_count: bool = False,
        include_preview: bool = False,
    ) -> list[dict[str, Any]]:
        """Return persisted threads."""

    async def resolve_thread_id_prefix(
        self,
        thread_id_or_prefix: str,
    ) -> tuple[str | None, list[str]]:
        """Resolve an exact or prefix thread id."""

    async def get_thread_metadata(self, thread_id: str) -> dict[str, Any] | None:
        """Return persisted metadata for a thread, if available."""

    async def get_thread_messages(self, thread_id: str) -> list[Any]:
        """Return persisted messages for a thread."""

    async def thread_exists(self, thread_id: str) -> bool:
        """Return whether a thread exists."""

    async def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread and its persisted state."""


class GraphGateway(Protocol):
    """One authority for graph runs and thread lifecycle operations."""

    events: SessionEvents | None

    async def create_thread(
        self,
        target: GraphTarget | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create or reserve a new thread id."""

    async def list_threads(
        self,
        *,
        limit: int = 20,
        include_message_count: bool = False,
        include_preview: bool = False,
        target: GraphTarget | None = None,
    ) -> list[dict[str, Any]]:
        """Return user-facing threads for the active backend."""

    async def resolve_thread(
        self,
        thread_id_or_prefix: str,
        target: GraphTarget | None = None,
    ) -> ThreadResolution:
        """Resolve a thread id or prefix."""

    async def get_thread_metadata(
        self,
        thread_id: str,
        target: GraphTarget | None = None,
    ) -> dict[str, Any] | None:
        """Return persisted metadata for a thread, if available."""

    async def get_thread_messages(
        self,
        thread_id: str,
        target: GraphTarget | None = None,
    ) -> list[Any]:
        """Return persisted messages for a thread."""

    async def thread_exists(
        self,
        thread_id: str,
        target: GraphTarget | None = None,
    ) -> bool:
        """Return whether a thread exists in the active backend."""

    async def delete_thread(
        self,
        thread_id: str,
        target: GraphTarget | None = None,
    ) -> bool:
        """Delete a thread and its persisted state."""

    async def clone_thread(
        self,
        source_thread_id: str,
        *,
        metadata: dict[str, Any] | None = None,
        target: GraphTarget | None = None,
    ) -> str:
        """Clone a thread and return the cloned thread id."""

    def stream_events(self, request: RunRequest) -> AsyncIterator[GraphEvent]:
        """Stream normalized graph events for the request target."""

    async def get_state_values(
        self,
        target: GraphTarget,
        thread_id: str,
    ) -> GraphStateValues:
        """Return the graph state values for a thread."""

    async def update_state_values(
        self,
        target: GraphTarget,
        thread_id: str,
        values: GraphStateValues,
    ) -> None:
        """Update graph state values for a thread."""

    async def get_run_status(
        self,
        target: GraphTarget,
        thread_id: str,
        run_id: str,
    ) -> str:
        """Return the task run's status from the server it runs on.

        Async sub-agent tasks run on the langgraph dev server under both
        backends, so this reads the run's status there. The client-side
        async-task read path uses it to detect completion without the
        in-process notifier. Propagates read errors (server unavailable, run
        not found) like the other state reads; the reader treats a failed read
        as "not yet terminal".
        """

    async def get_process_status(
        self,
        target: GraphTarget,
        thread_id: str,
        process_id: str,
    ) -> str:
        """Return a background process's live status from the graph process.

        Background processes launched via ``run_in_background`` run in the graph
        process (in-process on the local backend, the langgraph dev server on the
        server backend), so their status lives in that process's registry. The
        client-side ``bg_processes`` read path polls this to detect exit without
        the in-process notifier, mirroring :meth:`get_run_status`. Returns one of
        ``running`` / ``success`` / ``error`` / ``interrupted`` / ``unknown``;
        a failed read is treated as "not yet terminal" by the reader.
        """
