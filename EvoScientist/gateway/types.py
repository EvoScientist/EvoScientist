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
    include_per_run_overrides: bool = False,
    config: Any = None,
) -> dict[str, Any]:
    """Assemble the per-run LangGraph config for a gateway stream call.

    Single assembly point for both gateway backends. Always merges, in
    precedence order (lowest to highest):

    1. nothing (see ``include_per_run_overrides`` below),
    2. caller-supplied ``configurable_extra`` (e.g. ``active_teams``),
    3. ``thread_id`` — structural key, always set last.

    With ``include_per_run_overrides`` (server backend), additionally:

    - ``configurable.model`` / ``configurable.model_provider`` from the LIVE
      session config, picked up server-side by ``ConfigurableModelMiddleware``
      (which re-resolves and caches the model per ``(model, provider)``) —
      this is the per-run model channel that lets ``/model`` apply without
      an agent rebuild.
    - ``recursion_limit`` as a first-class top-level ``RunnableConfig`` key;
      a per-call value overrides the server's construction-time
      ``.with_config`` binding, so a keepalive server picks up the client's
      live limit per run instead of at restart.

    The local backend passes ``include_per_run_overrides=False``: its agent
    is rebuilt on model switches and already binds ``recursion_limit`` at
    construction from the same live config, so per-run injection there is
    redundant — and would couple every local stream call to a config read.

    ``config`` defaults to the live config object (``_ensure_config`` — the
    cached, in-place-mutated session config), NOT a fresh disk read: a disk
    read would clobber mid-session ``/model`` edits that have not been
    ``--save``d. The deferred import keeps this module import-light and
    cycle-free.
    """
    resolved_overrides: dict[str, Any] = {}
    recursion_limit: int | None = None
    if include_per_run_overrides:
        if config is None:
            from ..EvoScientist import _ensure_config

            config = _ensure_config()
        model = getattr(config, "model", None)
        provider = getattr(config, "provider", None)
        if model:
            resolved_overrides["model"] = model
        if provider:
            resolved_overrides["model_provider"] = provider
        limit = getattr(config, "recursion_limit", None)
        if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
            recursion_limit = limit

    configurable: dict[str, Any] = dict(resolved_overrides)
    if configurable_extra:
        # Caller-supplied extras win over the resolved session defaults —
        # an explicit per-run injection is more specific than the config.
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
