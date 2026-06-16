"""Shared test doubles for gateway/runtime boundaries."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass
from typing import Any

from EvoScientist.channels.base import Channel
from EvoScientist.channels.bus.events import InboundMessage, OutboundMessage
from EvoScientist.gateway import (
    GraphEvent,
    GraphGateway,
    RunRequest,
    ThreadResolution,
    ThreadStore,
)


@dataclass
class FakeChannelConfig:
    """Minimal config surface consumed by channel base tests."""

    text_chunk_limit: int = 4096
    allowed_senders: list | None = None
    allowed_channels: list | None = None
    proxy: str | None = None
    require_mention: str = "group"
    dm_policy: str = "allowlist"


class StubChannel(Channel):
    """Minimal concrete channel for unit tests of channel base behavior."""

    name = "stub"

    def __init__(self, config: Any | None = None) -> None:
        super().__init__(config or FakeChannelConfig())
        self._sent_chunks: list[tuple] = []
        self._typing_started: list[str] = []
        self._typing_stopped: list[str] = []
        self._started = False

    async def start(self) -> None:
        self._started = True
        self._running = True

    async def _send_chunk(
        self,
        chat_id: str,
        formatted_text: str,
        raw_text: str,
        reply_to: str | None,
        metadata: dict,
    ) -> None:
        self._sent_chunks.append(
            (chat_id, formatted_text, raw_text, reply_to, metadata)
        )

    async def _send_typing_action(self, chat_id: str) -> None:
        self._typing_started.append(chat_id)


class QueueFakeChannel(Channel):
    """Concrete channel with queue receive and captured outbound messages."""

    name = "fake"

    def __init__(self, config: Any | None = None) -> None:
        super().__init__(config or FakeChannelConfig())
        self._started = False
        self._stopped = False
        self._sent: list[OutboundMessage] = []

    async def start(self) -> None:
        self._started = True

    async def stop(self) -> None:
        self._stopped = True

    async def receive(self) -> AsyncIterator[InboundMessage]:
        while True:
            try:
                msg = await asyncio.wait_for(self._queue.get(), timeout=0.5)
                yield msg
            except TimeoutError:
                return

    async def send(self, message: OutboundMessage) -> bool:
        self._sent.append(message)
        return True

    async def _send_chunk(
        self,
        chat_id: str,
        formatted_text: str,
        raw_text: str,
        reply_to: str | None,
        metadata: dict,
    ) -> None:
        pass


class FakeThreadStore(ThreadStore):
    """Configurable ``ThreadStore`` test double with call recording."""

    def __init__(
        self,
        *,
        generated_thread_id: str = "unused",
        threads: list[dict[str, Any]] | None = None,
        resolved_thread_id: str | None = None,
        matches: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        messages: list[Any] | None = None,
        exists: bool = False,
        deleted: bool = False,
        errors: dict[str, BaseException] | None = None,
    ) -> None:
        self.generated_thread_id = generated_thread_id
        self.threads = threads or []
        self.resolved_thread_id = resolved_thread_id
        self.matches = matches or []
        self.metadata = metadata
        self.messages = messages or []
        self.exists = exists
        self.deleted = deleted
        self.errors = errors or {}
        self.calls: list[tuple[str, Any]] = []

    def _maybe_raise(self, method: str) -> None:
        error = self.errors.get(method)
        if error is not None:
            raise error

    def generate_thread_id(self) -> str:
        self.calls.append(("generate_thread_id", None))
        self._maybe_raise("generate_thread_id")
        return self.generated_thread_id

    async def list_threads(
        self,
        *,
        limit: int = 20,
        include_message_count: bool = False,
        include_preview: bool = False,
    ) -> list[dict[str, Any]]:
        self.calls.append(
            (
                "list_threads",
                {
                    "limit": limit,
                    "include_message_count": include_message_count,
                    "include_preview": include_preview,
                },
            )
        )
        self._maybe_raise("list_threads")
        return self.threads

    async def resolve_thread_id_prefix(
        self,
        thread_id_or_prefix: str,
    ) -> tuple[str | None, list[str]]:
        self.calls.append(("resolve_thread_id_prefix", thread_id_or_prefix))
        self._maybe_raise("resolve_thread_id_prefix")
        return self.resolved_thread_id, self.matches

    async def get_thread_metadata(self, thread_id: str) -> dict[str, Any] | None:
        self.calls.append(("get_thread_metadata", thread_id))
        self._maybe_raise("get_thread_metadata")
        return self.metadata

    async def get_thread_messages(self, thread_id: str) -> list[Any]:
        self.calls.append(("get_thread_messages", thread_id))
        self._maybe_raise("get_thread_messages")
        return self.messages

    async def thread_exists(self, thread_id: str) -> bool:
        self.calls.append(("thread_exists", thread_id))
        self._maybe_raise("thread_exists")
        return self.exists

    async def delete_thread(self, thread_id: str) -> bool:
        self.calls.append(("delete_thread", thread_id))
        self._maybe_raise("delete_thread")
        return self.deleted


FakeStreamFactory = Callable[[RunRequest], AsyncIterator[GraphEvent]]


class FakeGraphGateway(GraphGateway):
    """Configurable graph gateway test double with request recording."""

    def __init__(
        self,
        events: Iterable[GraphEvent] | None = None,
        *,
        stream: FakeStreamFactory | None = None,
        thread_store: ThreadStore | None = None,
    ) -> None:
        self.events = list(events or [])
        self.stream = stream
        self.thread_store = thread_store or FakeThreadStore()
        self.requests: list[RunRequest] = []

    async def create_thread(self) -> str:
        return self.thread_store.generate_thread_id()

    async def list_threads(
        self,
        *,
        limit: int = 20,
        include_message_count: bool = False,
        include_preview: bool = False,
    ) -> list[dict[str, Any]]:
        return await self.thread_store.list_threads(
            limit=limit,
            include_message_count=include_message_count,
            include_preview=include_preview,
        )

    async def resolve_thread(self, thread_id_or_prefix: str) -> ThreadResolution:
        resolved, matches = await self.thread_store.resolve_thread_id_prefix(
            thread_id_or_prefix
        )
        return ThreadResolution(resolved, tuple(matches))

    async def get_thread_metadata(self, thread_id: str) -> dict[str, Any] | None:
        return await self.thread_store.get_thread_metadata(thread_id)

    async def get_thread_messages(self, thread_id: str) -> list[Any]:
        return await self.thread_store.get_thread_messages(thread_id)

    async def thread_exists(self, thread_id: str) -> bool:
        return await self.thread_store.thread_exists(thread_id)

    async def delete_thread(self, thread_id: str) -> bool:
        return await self.thread_store.delete_thread(thread_id)

    def stream_events(self, request: RunRequest) -> AsyncIterator[GraphEvent]:
        self.requests.append(request)
        if self.stream is not None:
            return self.stream(request)

        async def _events() -> AsyncIterator[GraphEvent]:
            for event in self.events:
                yield event

        return _events()
