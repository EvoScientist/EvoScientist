"""Local in-process gateway backend preserving current behavior."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .. import sessions as session_store
from .types import (
    GraphEvent,
    GraphStateValues,
    GraphTarget,
    RunRequest,
    ThreadResolution,
    ThreadStore,
)

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


@dataclass(frozen=True, slots=True)
class LocalThreadStore:
    """Thread store backed by the current ``sessions.py`` module."""

    def generate_thread_id(self) -> str:
        return session_store.generate_thread_id()

    async def list_threads(
        self,
        *,
        limit: int = 20,
        include_message_count: bool = False,
        include_preview: bool = False,
    ) -> list[dict[str, Any]]:
        return await session_store.list_threads(
            limit=limit,
            include_message_count=include_message_count,
            include_preview=include_preview,
        )

    async def resolve_thread_id_prefix(
        self,
        thread_id_or_prefix: str,
    ) -> tuple[str | None, list[str]]:
        return await session_store.resolve_thread_id_prefix(thread_id_or_prefix)

    async def get_thread_metadata(self, thread_id: str) -> dict[str, Any] | None:
        return await session_store.get_thread_metadata(thread_id)

    async def get_thread_messages(self, thread_id: str) -> list[Any]:
        return await session_store.get_thread_messages(thread_id)

    async def thread_exists(self, thread_id: str) -> bool:
        return await session_store.thread_exists(thread_id)

    async def delete_thread(self, thread_id: str) -> bool:
        return await session_store.delete_thread(thread_id)


@dataclass(slots=True)
class LocalGraphGateway:
    """Gateway backed by the current in-process graph and session helpers."""

    thread_store: ThreadStore = field(default_factory=LocalThreadStore)

    async def create_thread(
        self,
        target: GraphTarget | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return self.thread_store.generate_thread_id()

    async def list_threads(
        self,
        *,
        limit: int = 20,
        include_message_count: bool = False,
        include_preview: bool = False,
        target: GraphTarget | None = None,
    ) -> list[dict[str, Any]]:
        return await self.thread_store.list_threads(
            limit=limit,
            include_message_count=include_message_count,
            include_preview=include_preview,
        )

    async def resolve_thread(
        self,
        thread_id_or_prefix: str,
        target: GraphTarget | None = None,
    ) -> ThreadResolution:
        resolved, matches = await self.thread_store.resolve_thread_id_prefix(
            thread_id_or_prefix
        )
        return ThreadResolution(resolved, tuple(matches))

    async def get_thread_metadata(
        self,
        thread_id: str,
        target: GraphTarget | None = None,
    ) -> dict[str, Any] | None:
        return await self.thread_store.get_thread_metadata(thread_id)

    async def get_thread_messages(
        self,
        thread_id: str,
        target: GraphTarget | None = None,
    ) -> list[Any]:
        return await self.thread_store.get_thread_messages(thread_id)

    async def thread_exists(
        self,
        thread_id: str,
        target: GraphTarget | None = None,
    ) -> bool:
        return await self.thread_store.thread_exists(thread_id)

    async def delete_thread(
        self,
        thread_id: str,
        target: GraphTarget | None = None,
    ) -> bool:
        return await self.thread_store.delete_thread(thread_id)

    async def clone_thread(
        self,
        source_thread_id: str,
        *,
        metadata: dict[str, Any] | None = None,
        target: GraphTarget | None = None,
    ) -> str:
        raise NotImplementedError("LocalGraphGateway does not support thread cloning.")

    def stream_events(self, request: RunRequest) -> AsyncIterator[GraphEvent]:
        target = request.target
        local_graph = self._require_local_graph(target)
        if target is None:
            raise RuntimeError("LocalGraphGateway requires GraphTarget.local_graph")
        return self._stream_events(local_graph, target, request)

    async def _stream_events(
        self,
        local_graph: CompiledStateGraph,
        target: GraphTarget,
        request: RunRequest,
    ) -> AsyncIterator[GraphEvent]:
        from ..stream.events import stream_agent_events

        inner = stream_agent_events(
            local_graph,
            request.message,
            request.thread_id,
            metadata=request.metadata,
            media=request.media,
        )
        try:
            async for event in inner:
                yield event
        finally:
            await inner.aclose()

    async def get_state_values(
        self,
        target: GraphTarget,
        thread_id: str,
    ) -> GraphStateValues:
        local_graph = self._require_local_graph(target)
        snapshot = await local_graph.aget_state(
            {"configurable": {"thread_id": thread_id}}
        )
        values: GraphStateValues = snapshot.values
        return values

    async def update_state_values(
        self,
        target: GraphTarget,
        thread_id: str,
        values: GraphStateValues,
    ) -> None:
        local_graph = self._require_local_graph(target)
        as_node = "model" if "_summarization_event" in values else None
        await local_graph.aupdate_state(
            {"configurable": {"thread_id": thread_id}},
            values,
            as_node=as_node,
        )

    def _require_local_graph(self, target: GraphTarget | None) -> CompiledStateGraph:
        if target is None or target.local_graph is None:
            raise RuntimeError("LocalGraphGateway requires GraphTarget.local_graph")
        return target.local_graph
