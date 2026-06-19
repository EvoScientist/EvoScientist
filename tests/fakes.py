"""Shared test doubles for gateway/runtime boundaries."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass
from typing import Any

import httpx
from langgraph_sdk.client import LangGraphClient

from EvoScientist.channels.base import Channel
from EvoScientist.channels.bus.events import InboundMessage, OutboundMessage
from EvoScientist.commands.base import CommandUI
from EvoScientist.gateway import (
    GraphEvent,
    GraphGateway,
    GraphStateValues,
    GraphTarget,
    RunRequest,
    ThreadResolution,
    ThreadStore,
)

_DEFAULT_COPY_RESPONSE = object()


class FakeCommandUI(CommandUI):
    """Command UI test double with recorded calls and inert pickers."""

    def __init__(self, *, supports_interactive: bool = True) -> None:
        self._supports_interactive = supports_interactive
        self.system_messages: list[str] = []
        self.renderables: list[object] = []
        self.started = 0
        self.stopped = 0
        self.updated_tokens: list[int] = []
        self.chat_cleared = False
        self.quit_requested = False
        self.force_quit_requested = False
        self.started_sessions = 0
        self.resumed_sessions: list[tuple[str, str | None]] = []
        self.flushes = 0

    @property
    def supports_interactive(self) -> bool:
        return self._supports_interactive

    def append_system(self, text: str, style: str = "dim") -> None:
        self.system_messages.append(text)

    def mount_renderable(self, renderable: object) -> None:
        self.renderables.append(renderable)

    async def wait_for_thread_pick(
        self,
        threads: list[dict],
        current_thread: str,
        title: str,
    ) -> str | None:
        return None

    async def wait_for_skill_browse(
        self,
        index: list[dict],
        installed_names: set[str],
        pre_filter_tag: str,
    ) -> list[str] | None:
        return None

    async def wait_for_mcp_browse(
        self,
        servers: list,
        installed_names: set[str],
        pre_filter_tag: str,
    ) -> list | None:
        return None

    async def wait_for_model_pick(
        self,
        entries: list[tuple[str, str, str]],
        current_model: str | None,
        current_provider: str | None,
    ) -> tuple[str, str] | None:
        return None

    def clear_chat(self) -> None:
        self.chat_cleared = True

    def request_quit(self) -> None:
        self.quit_requested = True

    def force_quit(self) -> None:
        self.force_quit_requested = True

    async def start_new_session(self) -> None:
        self.started_sessions += 1

    async def handle_session_resume(
        self,
        thread_id: str,
        workspace_dir: str | None = None,
    ) -> None:
        self.resumed_sessions.append((thread_id, workspace_dir))

    async def flush(self) -> None:
        self.flushes += 1

    async def start_compacting_indicator(self) -> None:
        self.started += 1

    async def stop_compacting_indicator(self) -> None:
        self.stopped += 1

    def update_status_after_compact(self, tokens_after: int) -> None:
        self.updated_tokens.append(tokens_after)


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
        state_values: GraphStateValues | None = None,
        state_error: BaseException | None = None,
        generated_thread_ids: Iterable[str] | None = None,
        thread_store: ThreadStore | None = None,
    ) -> None:
        self.events = list(events or [])
        self.stream = stream
        self.state_values = state_values or {}
        self.state_error = state_error
        self.generated_thread_ids = list(generated_thread_ids or [])
        self.thread_store = thread_store or FakeThreadStore()
        self.requests: list[RunRequest] = []
        self.clone_calls: list[
            tuple[str, dict[str, Any] | None, GraphTarget | None]
        ] = []
        self.updated_states: list[tuple[GraphTarget, str, GraphStateValues]] = []

    async def create_thread(
        self,
        target: GraphTarget | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if self.generated_thread_ids:
            return self.generated_thread_ids.pop(0)
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
        self.clone_calls.append((source_thread_id, metadata, target))
        if self.generated_thread_ids:
            return self.generated_thread_ids.pop(0)
        return f"{source_thread_id}-clone"

    def stream_events(self, request: RunRequest) -> AsyncIterator[GraphEvent]:
        self.requests.append(request)
        if self.stream is not None:
            return self.stream(request)

        async def _events() -> AsyncIterator[GraphEvent]:
            for event in self.events:
                yield event

        return _events()

    async def get_state_values(
        self,
        target: GraphTarget,
        thread_id: str,
    ) -> GraphStateValues:
        if self.state_error is not None:
            raise self.state_error
        return self.state_values

    async def update_state_values(
        self,
        target: GraphTarget,
        thread_id: str,
        values: GraphStateValues,
    ) -> None:
        self.updated_states.append((target, thread_id, values))


class FakeLangGraphRunModule:
    """Fake thread-stream run controller for server gateway tests."""

    def __init__(self) -> None:
        self.starts: list[dict[str, Any]] = []
        self.responses: list[dict[str, Any]] = []

    async def start(
        self,
        *,
        input: object = None,
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.starts.append(
            {
                "input": input,
                "config": config,
                "metadata": metadata,
            }
        )
        return {"run_id": "run-1"}

    async def respond(
        self,
        response: object,
        *,
        interrupt_id: str | None = None,
    ) -> dict[str, Any]:
        self.responses.append(
            {
                "response": response,
                "interrupt_id": interrupt_id,
            }
        )
        return {"run_id": "run-1"}


class FakeLangGraphThreadStream:
    """Finite fake of the LangGraph SDK thread stream."""

    def __init__(
        self,
        thread_id: str,
        events: Iterable[dict[str, Any]] | None = None,
        *,
        interrupts: list[dict[str, Any]] | None = None,
        interrupted: bool = False,
    ) -> None:
        self.thread_id = thread_id
        self.events = list(events or [])
        self.interrupts = interrupts or []
        self.interrupted = interrupted
        self.run = FakeLangGraphRunModule()
        self.subscribed_channels: list[list[str]] = []
        self.entered = False
        self.exited = False

    async def __aenter__(self) -> FakeLangGraphThreadStream:
        self.entered = True
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.exited = True

    async def _iter_events(self) -> AsyncIterator[dict[str, Any]]:
        for event in self.events:
            yield event

    def subscribe(self, channels: list[str]) -> AsyncIterator[dict[str, Any]]:
        self.subscribed_channels.append(channels)
        return self._iter_events()


class FakeLangGraphThreadsClient:
    """Fake LangGraph ``client.threads`` surface."""

    def __init__(
        self,
        *,
        threads: list[dict[str, Any]] | None = None,
        states: dict[str, dict[str, Any]] | None = None,
        streams: dict[str, FakeLangGraphThreadStream] | None = None,
        copy_response: object = _DEFAULT_COPY_RESPONSE,
    ) -> None:
        self.threads = threads or []
        self.states = states or {}
        self.streams = streams or {}
        self.copy_response = copy_response
        self.created: list[dict[str, Any]] = []
        self.copied: list[str] = []
        self.metadata_updates: list[tuple[str, dict[str, Any]]] = []
        self.deleted: list[str] = []
        self.gets: list[str] = []
        self.searches: list[dict[str, Any]] = []
        self.stream_calls: list[tuple[str, str]] = []
        self.state_updates: list[tuple[str, GraphStateValues, str | None]] = []

    async def create(
        self,
        *,
        metadata: dict[str, Any] | None = None,
        thread_id: str | None = None,
        if_exists: str | None = None,
        graph_id: str | None = None,
    ) -> dict[str, Any]:
        if thread_id is not None and if_exists == "do_nothing":
            for thread in self.threads:
                if thread.get("thread_id") == thread_id:
                    return thread
        created = {
            "thread_id": thread_id or "server-thread",
            "metadata": {
                **(metadata or {}),
                **({"graph_id": graph_id} if graph_id else {}),
            },
        }
        self.created.append(created)
        self.threads.append(created)
        return created

    async def search(
        self,
        *,
        metadata: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        sort_by: str | None = None,
        sort_order: str | None = None,
    ) -> list[dict[str, Any]]:
        self.searches.append(
            {
                "metadata": metadata,
                "limit": limit,
                "offset": offset,
                "sort_by": sort_by,
                "sort_order": sort_order,
            }
        )
        rows = self.threads
        if metadata:
            rows = [
                thread
                for thread in rows
                if all(
                    (thread.get("metadata") or {}).get(key) == value
                    for key, value in metadata.items()
                )
            ]
        return rows[offset : offset + limit]

    async def get(self, thread_id: str) -> dict[str, Any]:
        from langgraph_sdk.errors import NotFoundError

        self.gets.append(thread_id)
        for thread in self.threads:
            if thread.get("thread_id") == thread_id:
                return thread
        raise NotFoundError("not found", response=_not_found_response(), body=None)

    async def copy(self, thread_id: str) -> object:
        source = await self.get(thread_id)
        self.copied.append(thread_id)
        if self.copy_response is not _DEFAULT_COPY_RESPONSE:
            return self.copy_response
        copied = {
            "thread_id": f"{thread_id}-copy",
            "metadata": dict(source.get("metadata") or {}),
        }
        self.threads.append(copied)
        return copied

    async def update(
        self,
        thread_id: str,
        *,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        thread = await self.get(thread_id)
        existing_metadata = thread.get("metadata")
        merged = {
            **(existing_metadata if isinstance(existing_metadata, dict) else {}),
            **metadata,
        }
        thread["metadata"] = merged
        self.metadata_updates.append((thread_id, metadata))
        return thread

    async def get_state(self, thread_id: str) -> dict[str, Any]:
        from langgraph_sdk.errors import NotFoundError

        if thread_id in self.states:
            return self.states[thread_id]
        raise NotFoundError("not found", response=_not_found_response(), body=None)

    async def update_state(
        self,
        thread_id: str,
        values: GraphStateValues,
        *,
        as_node: str | None = None,
    ) -> dict[str, Any]:
        self.state_updates.append((thread_id, values, as_node))
        return {"checkpoint": {"thread_id": thread_id}}

    async def delete(self, thread_id: str) -> None:
        await self.get(thread_id)
        self.deleted.append(thread_id)
        self.threads = [
            thread for thread in self.threads if thread.get("thread_id") != thread_id
        ]

    def stream(
        self,
        thread_id: str | None = None,
        *,
        assistant_id: str,
    ) -> FakeLangGraphThreadStream:
        assert thread_id is not None
        self.stream_calls.append((thread_id, assistant_id))
        return self.streams[thread_id]


class FakeLangGraphClient(LangGraphClient):
    """Fake LangGraph SDK async client."""

    def __init__(self, threads: FakeLangGraphThreadsClient) -> None:
        self.threads = threads


def _not_found_response() -> httpx.Response:
    request = httpx.Request("GET", "https://test.local/not-found")
    return httpx.Response(404, request=request)
