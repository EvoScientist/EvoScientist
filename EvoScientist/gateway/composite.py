"""Composite gateway: read from one backend, execute on another.

Stage 4 of the gateway migration (issue #432) keeps thread listing, prefix
resolution, and history reads on the local SQLite readers while routing graph
execution to the langgraph server. Reads and execution are already separate
methods on the :class:`~EvoScientist.gateway.types.GraphGateway` protocol, but a
single gateway instance serves both, so the split lives here: a wrapper holding
one ``read`` gateway and one ``execute`` gateway and dispatching each protocol
method to the right one.

Routing table (read = local, execute = langgraph server):

- Catalog: ``list_threads`` unions both stores (local authoritative, deduped by
  ``thread_id``); ``resolve_thread`` tries local first, then the server for a
  UUID input the local store didn't know; ``get_thread_metadata`` prefers the
  server for a UUID (falling back to local), local for a legacy id.
- History/state: ``get_thread_messages`` follows the execution store for UUIDs
  (server-executed turns write server checkpoints, so the local DB would be
  stale), falling back to local when the server has no history yet (a UUID
  thread created before the cutover). ``get_state_values`` routes UUIDs to the
  server with NO local fallback (local ``get_state_values`` needs a local graph
  the server-execute caller does not supply). Legacy ids read local for both.
- Existence: ``thread_exists`` is true if either store has it (the server is
  consulted only for a UUID id).
- Execution: ``create_thread`` goes to the server (it mints the id, so there is
  no id to guard); ``stream_events`` / ``update_state_values`` / ``clone_thread``
  go to the server guarded on a UUID id; ``get_run_status`` / ``get_process_status``
  go to the server, where runs and background processes live.
- ``delete_thread`` fans out to both stores; success if either deleted, so a
  thread that only exists in one store is still removed without error.

**Legacy (non-UUID) thread ids** are local-only: langgraph-api rejects non-UUID
ids, so a legacy thread can never be registered or executed server-side. Reads
keep working through the local route; execution raises
:class:`LegacyThreadServerExecutionError` with clear guidance rather than failing
silently or splitting execution back across two live executors.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from .server import _is_uuid
from .types import (
    GraphEvent,
    GraphStateValues,
    GraphTarget,
    RunRequest,
    ThreadResolution,
)

if TYPE_CHECKING:
    from ..middleware.events import SessionEvents
    from .types import GraphGateway

logger = logging.getLogger(__name__)


class LegacyThreadServerExecutionError(RuntimeError):
    """Raised when server execution is requested for a legacy (non-UUID) thread.

    langgraph-api only accepts UUID-shaped thread ids, so a pre-migration thread
    (short-hex id) can never run server-side. Its history stays readable through
    the local route; only execution is refused.
    """

    def __init__(self, thread_id: str) -> None:
        super().__init__(
            f"Thread {thread_id!r} predates server execution (legacy non-UUID id) "
            f"and cannot run on the langgraph server. Start a new session with "
            f"/new to continue on the server backend; the old thread stays readable."
        )
        self.thread_id = thread_id


class CompositeGraphGateway:
    """Route reads to ``read`` and execution to ``execute`` per the module docstring."""

    def __init__(self, *, read: GraphGateway, execute: GraphGateway) -> None:
        self._read = read
        self._execute = execute

    # ``events`` is the streaming event sink; it belongs to the execution side.
    @property
    def events(self) -> SessionEvents | None:
        return self._execute.events

    @events.setter
    def events(self, value: SessionEvents | None) -> None:
        self._execute.events = value

    # --- Catalog / read -------------------------------------------------------

    async def list_threads(
        self,
        *,
        limit: int = 20,
        include_message_count: bool = False,
        include_preview: bool = False,
        target: GraphTarget | None = None,
    ) -> list[dict[str, Any]]:
        local = await self._read.list_threads(
            limit=limit,
            include_message_count=include_message_count,
            include_preview=include_preview,
            target=target,
        )
        server = await self._execute.list_threads(
            limit=limit,
            include_message_count=include_message_count,
            include_preview=include_preview,
            target=target,
        )
        seen: set[Any] = set()
        merged: list[dict[str, Any]] = []
        for entry in (*local, *server):  # local first → local wins on dedup
            tid = entry.get("thread_id")
            if tid in seen:
                continue
            seen.add(tid)
            merged.append(entry)
        if limit and limit > 0:
            merged = merged[:limit]
        return merged

    async def resolve_thread(
        self,
        thread_id_or_prefix: str,
        target: GraphTarget | None = None,
    ) -> ThreadResolution:
        local = await self._read.resolve_thread(thread_id_or_prefix, target)
        if local.found or local.matches:
            return local
        if _is_uuid(thread_id_or_prefix):
            return await self._execute.resolve_thread(thread_id_or_prefix, target)
        return local

    async def get_thread_metadata(
        self,
        thread_id: str,
        target: GraphTarget | None = None,
    ) -> dict[str, Any] | None:
        if _is_uuid(thread_id):
            meta = await self._execute.get_thread_metadata(thread_id, target)
            if meta is not None:
                return meta
        return await self._read.get_thread_metadata(thread_id, target)

    async def get_thread_messages(
        self,
        thread_id: str,
        target: GraphTarget | None = None,
    ) -> list[Any]:
        if not _is_uuid(thread_id):
            return await self._read.get_thread_messages(thread_id, target)
        messages = await self._execute.get_thread_messages(thread_id, target)
        if messages:
            return messages
        # A UUID thread with no server history yet (created before the cutover):
        # fall back to the local store rather than report it empty.
        return await self._read.get_thread_messages(thread_id, target)

    async def get_state_values(
        self,
        target: GraphTarget,
        thread_id: str,
    ) -> GraphStateValues:
        # No local fallback here: local get_state_values needs target.local_graph,
        # which a server-execute caller does not supply.
        if _is_uuid(thread_id):
            return await self._execute.get_state_values(target, thread_id)
        return await self._read.get_state_values(target, thread_id)

    async def thread_exists(
        self,
        thread_id: str,
        target: GraphTarget | None = None,
    ) -> bool:
        if await self._read.thread_exists(thread_id, target):
            return True
        if _is_uuid(thread_id):
            return await self._execute.thread_exists(thread_id, target)
        return False

    async def delete_thread(
        self,
        thread_id: str,
        target: GraphTarget | None = None,
    ) -> bool:
        read_ok = await self._read.delete_thread(thread_id, target)
        exec_ok = False
        if _is_uuid(thread_id):
            try:
                exec_ok = await self._execute.delete_thread(thread_id, target)
            except Exception:  # server hiccup must not block the local delete
                logger.debug(
                    "composite: server-side delete failed for thread %s",
                    thread_id,
                    exc_info=True,
                )
        return read_ok or exec_ok

    # --- Execution ------------------------------------------------------------

    async def create_thread(
        self,
        target: GraphTarget | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return await self._execute.create_thread(target, metadata=metadata)

    async def clone_thread(
        self,
        source_thread_id: str,
        *,
        metadata: dict[str, Any] | None = None,
        target: GraphTarget | None = None,
    ) -> str:
        if not _is_uuid(source_thread_id):
            raise LegacyThreadServerExecutionError(source_thread_id)
        return await self._execute.clone_thread(
            source_thread_id, metadata=metadata, target=target
        )

    def stream_events(self, request: RunRequest) -> AsyncIterator[GraphEvent]:
        if not _is_uuid(request.thread_id):
            raise LegacyThreadServerExecutionError(request.thread_id)
        return self._execute.stream_events(request)

    async def update_state_values(
        self,
        target: GraphTarget,
        thread_id: str,
        values: GraphStateValues,
    ) -> None:
        if not _is_uuid(thread_id):
            raise LegacyThreadServerExecutionError(thread_id)
        await self._execute.update_state_values(target, thread_id, values)

    async def get_run_status(
        self,
        target: GraphTarget,
        thread_id: str,
        run_id: str,
    ) -> str:
        return await self._execute.get_run_status(target, thread_id, run_id)

    async def get_process_status(
        self,
        target: GraphTarget,
        thread_id: str,
        process_id: str,
    ) -> str:
        return await self._execute.get_process_status(target, thread_id, process_id)
