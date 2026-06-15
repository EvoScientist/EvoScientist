"""Shared types for graph/thread gateway implementations."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias

GraphEvent: TypeAlias = dict[str, Any]


@dataclass(frozen=True, slots=True)
class RunRequest:
    """A graph turn request, independent of the UI that initiated it."""

    message: Any
    thread_id: str
    metadata: dict[str, Any] | None = None
    media: list[str] | None = None


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

    async def create_thread(self) -> str:
        """Create or reserve a new thread id."""

    async def list_threads(
        self,
        *,
        limit: int = 20,
        include_message_count: bool = False,
        include_preview: bool = False,
    ) -> list[dict[str, Any]]:
        """Return user-facing threads for the active backend."""

    async def resolve_thread(self, thread_id_or_prefix: str) -> ThreadResolution:
        """Resolve a thread id or prefix."""

    async def get_thread_metadata(self, thread_id: str) -> dict[str, Any] | None:
        """Return persisted metadata for a thread, if available."""

    async def get_thread_messages(self, thread_id: str) -> list[Any]:
        """Return persisted messages for a thread."""

    async def thread_exists(self, thread_id: str) -> bool:
        """Return whether a thread exists in the active backend."""

    async def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread and its persisted state."""

    def stream_events(self, request: RunRequest) -> AsyncIterator[GraphEvent]:
        """Stream normalized EvoScientist graph events for a run."""
