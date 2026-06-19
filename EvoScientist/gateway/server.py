"""LangGraph server-backed gateway implementation."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import BaseMessage, convert_to_messages, messages_from_dict
from langgraph.types import Command
from langgraph_sdk import get_client
from langgraph_sdk._async.stream import AsyncThreadStream
from langgraph_sdk.client import LangGraphClient
from langgraph_sdk.errors import NotFoundError
from langgraph_sdk.schema import Thread, ThreadState

from ..sessions import _apply_summarization_event
from ..stream.emitter import StreamEventEmitter
from ..stream.events import (
    _SubagentRegistry,
    _V3EventProcessor,
    build_agent_stream_input,
)
from ..stream.summarization import _find_summarization_event_payload
from ..stream.v3_payloads import _as_raw_map, _event_namespace
from .types import (
    DEFAULT_GRAPH_ID,
    GraphEvent,
    GraphStateValues,
    GraphTarget,
    RunRequest,
    ThreadResolution,
    ThreadStore,
)

_THREAD_SEARCH_LIMIT = 1000
_RUN_SUBSCRIBE_CHANNELS = [
    "messages",
    "tools",
    "updates",
    "values",
    "tasks",
    "lifecycle",
    "input",
]


LangGraphClientFactory = Callable[
    [str, Mapping[str, str] | None],
    LangGraphClient,
]


def _default_client_factory(
    base_url: str,
    headers: Mapping[str, str] | None,
) -> LangGraphClient:
    return get_client(url=base_url, headers=headers)


def _thread_metadata(thread: Thread) -> dict[str, Any]:
    metadata = thread.get("metadata")
    return dict(metadata) if isinstance(metadata, dict) else {}


def _build_thread_metadata(
    *,
    graph_id: str,
    workspace_dir: str | None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged = dict(metadata or {})
    merged["graph_id"] = graph_id
    if graph_id == DEFAULT_GRAPH_ID:
        merged["agent_name"] = DEFAULT_GRAPH_ID
    else:
        merged.pop("agent_name", None)
    if workspace_dir is not None:
        merged["workspace_dir"] = workspace_dir
    merged.setdefault("updated_at", datetime.now(UTC).isoformat())
    return merged


def _thread_preview(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if getattr(message, "type", None) != "human":
            continue
        content = message.content
        if isinstance(content, str):
            return content.strip().replace("\n", " ")[:120]
        if isinstance(content, list):
            text_parts = [
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            if text := " ".join(part for part in text_parts if part).strip():
                return text.replace("\n", " ")[:120]
    return ""


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
    except ValueError:
        return False
    return True


def _input_requested_event_from_interrupt(
    interrupt: Mapping[str, object],
) -> dict[str, Any]:
    return {
        "type": "event",
        "method": "input.requested",
        "params": {
            "namespace": interrupt.get("namespace") or [],
            "data": {
                "interrupt_id": interrupt.get("interrupt_id")
                or interrupt.get("id")
                or "default",
                "value": interrupt.get("value"),
            },
        },
    }


def _state_interrupts(state: ThreadState) -> list[Mapping[str, object]]:
    interrupts = state.get("interrupts")
    if not isinstance(interrupts, list):
        return []
    return [interrupt for interrupt in interrupts if isinstance(interrupt, Mapping)]


def _is_interrupt_event(event: Mapping[str, object]) -> bool:
    return event.get("type") in {"interrupt", "ask_user"}


def _messages_from_state(state: ThreadState) -> list[BaseMessage]:
    values = state.get("values")
    if not isinstance(values, dict):
        return []
    raw_messages = values.get("messages")
    if not isinstance(raw_messages, list):
        return []
    event = values.get("_summarization_event")
    summarization_event = dict(event) if isinstance(event, Mapping) else None
    effective_messages = _apply_summarization_event(
        raw_messages,
        summarization_event,
    )
    try:
        return list(convert_to_messages(effective_messages))
    except ValueError:
        return messages_from_dict(
            [message for message in effective_messages if isinstance(message, dict)]
        )


@dataclass(frozen=True, slots=True)
class LangGraphServerThreadStore(ThreadStore):
    """Thread store backed by the LangGraph server Threads API."""

    base_url: str
    graph_id: str = DEFAULT_GRAPH_ID
    headers: Mapping[str, str] | None = None
    client_factory: LangGraphClientFactory = _default_client_factory
    _client: LangGraphClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_client",
            self.client_factory(self.base_url, self.headers),
        )

    @property
    def client(self) -> LangGraphClient:
        return self._client

    def generate_thread_id(self) -> str:
        return str(uuid.uuid4())

    def _target_graph_id(self, graph_id: str | None = None) -> str:
        return graph_id or self.graph_id

    async def create_thread(
        self,
        graph_id: str | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
        workspace_dir: str | None = None,
    ) -> str:
        target_graph_id = self._target_graph_id(graph_id)
        thread = await self.client.threads.create(
            graph_id=target_graph_id,
            metadata=_build_thread_metadata(
                graph_id=target_graph_id,
                workspace_dir=workspace_dir,
                metadata=metadata,
            ),
        )
        return thread["thread_id"]

    async def ensure_thread_exists(
        self,
        thread_id: str,
        graph_id: str | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
        workspace_dir: str | None = None,
    ) -> None:
        target_graph_id = self._target_graph_id(graph_id)
        await self.client.threads.create(
            thread_id=thread_id,
            graph_id=target_graph_id,
            metadata=_build_thread_metadata(
                graph_id=target_graph_id,
                workspace_dir=workspace_dir,
                metadata=metadata,
            ),
            if_exists="do_nothing",
        )

    async def list_threads(
        self,
        *,
        limit: int = 20,
        include_message_count: bool = False,
        include_preview: bool = False,
        graph_id: str | None = None,
    ) -> list[dict[str, Any]]:
        target_graph_id = self._target_graph_id(graph_id)

        threads = await self._search_threads(
            target_graph_id=target_graph_id,
            limit=limit,
        )
        rows: list[dict[str, Any]] = []
        for thread in threads:
            thread_id = thread["thread_id"]
            metadata = _thread_metadata(thread)
            row: dict[str, Any] = {
                "thread_id": thread_id,
                "created_at": thread.get("created_at"),
                "updated_at": thread.get("updated_at"),
                "workspace_dir": metadata.get("workspace_dir"),
                "model": metadata.get("model"),
                "metadata": metadata,
            }
            if include_message_count or include_preview:
                messages = await self.get_thread_messages(thread_id)
                if include_message_count:
                    row["message_count"] = len(messages)
                if include_preview:
                    row["preview"] = _thread_preview(messages)
            rows.append(row)
        return rows

    async def resolve_thread_id_prefix(
        self,
        thread_id_or_prefix: str,
        graph_id: str | None = None,
    ) -> tuple[str | None, list[str]]:
        target_graph_id = self._target_graph_id(graph_id)
        if _is_uuid(thread_id_or_prefix):
            try:
                thread = await self.client.threads.get(thread_id_or_prefix)
                if _thread_metadata(thread).get("graph_id") == target_graph_id:
                    return thread["thread_id"], []
            except NotFoundError:
                pass

        threads = await self._search_threads(target_graph_id=target_graph_id)
        matches = sorted(
            thread["thread_id"]
            for thread in threads
            if thread["thread_id"].startswith(thread_id_or_prefix)
        )
        if len(matches) == 1:
            return matches[0], []
        return None, matches

    async def _search_threads(
        self,
        *,
        target_graph_id: str,
        limit: int | None = None,
    ) -> list[Thread]:
        if limit is not None and limit > 0:
            return await self._search_thread_page(
                target_graph_id=target_graph_id,
                limit=limit,
            )
        return await self._search_all_threads(target_graph_id=target_graph_id)

    async def _search_thread_page(
        self,
        *,
        target_graph_id: str,
        limit: int,
        offset: int = 0,
    ) -> list[Thread]:
        return await self.client.threads.search(
            metadata={"graph_id": target_graph_id},
            limit=limit,
            offset=offset,
            sort_by="updated_at",
            sort_order="desc",
        )

    async def _search_all_threads(self, *, target_graph_id: str) -> list[Thread]:
        threads: list[Thread] = []
        offset = 0
        while True:
            page = await self._search_thread_page(
                target_graph_id=target_graph_id,
                limit=_THREAD_SEARCH_LIMIT,
                offset=offset,
            )
            threads.extend(page)
            if len(page) < _THREAD_SEARCH_LIMIT:
                break
            offset += _THREAD_SEARCH_LIMIT
        return threads

    async def get_thread_metadata(self, thread_id: str) -> dict[str, Any] | None:
        try:
            thread = await self.client.threads.get(thread_id)
        except NotFoundError:
            return None
        return _thread_metadata(thread)

    async def get_thread_messages(self, thread_id: str) -> list[BaseMessage]:
        try:
            state = await self.client.threads.get_state(thread_id)
        except NotFoundError:
            return []
        return _messages_from_state(state)

    async def thread_exists(self, thread_id: str) -> bool:
        try:
            await self.client.threads.get(thread_id)
        except NotFoundError:
            return False
        return True

    async def delete_thread(self, thread_id: str) -> bool:
        try:
            await self.client.threads.delete(thread_id)
        except NotFoundError:
            return False
        return True

    async def clone_thread(
        self,
        source_thread_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        copy_response: object = await self.client.threads.copy(source_thread_id)
        if not isinstance(copy_response, Mapping):
            raise RuntimeError(
                "LangGraph thread copy did not return a cloned thread id"
            )
        cloned_thread_id = copy_response.get("thread_id")
        if not isinstance(cloned_thread_id, str) or not cloned_thread_id:
            raise RuntimeError(
                "LangGraph thread copy did not return a cloned thread id"
            )
        if metadata:
            await self.client.threads.update(
                cloned_thread_id,
                metadata=metadata,
            )
        return cloned_thread_id


@dataclass(slots=True)
class _ServerSubagentTracker:
    """Infer subagent start/end events from LangGraph server namespaces."""

    emitter: StreamEventEmitter
    registry: _SubagentRegistry
    _active: dict[tuple[str, ...], tuple[str, str | None]] = field(default_factory=dict)

    def process(self, event: Mapping[str, Any]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        namespace = tuple(_event_namespace(event))
        if namespace:
            events.extend(self._ensure_registered(namespace[:1], tool_call_id=None))

        method = event.get("method")
        params = _as_raw_map(event.get("params"))
        data = _as_raw_map(params.get("data")) if params is not None else None
        if data is None:
            return events

        if method == "lifecycle":
            phase = data.get("event")
            if phase == "started" and namespace:
                events.extend(self._ensure_registered(namespace, tool_call_id=None))
            elif phase in ("completed", "failed") and namespace:
                events.extend(self._end(namespace))
        elif method == "tasks":
            if "result" in data:
                events.extend(self._end_triggered_child(namespace, data.get("id")))
            elif namespace:
                events.extend(self._ensure_registered(namespace, tool_call_id=None))
        return events

    def finish(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for path in sorted(
            self._active.keys(), key=lambda item: len(item), reverse=True
        ):
            events.extend(self._end(path))
        self.registry.close()
        return events

    def _ensure_registered(
        self,
        path: tuple[str, ...],
        *,
        tool_call_id: str | None,
    ) -> list[dict[str, Any]]:
        if not path or path in self._active:
            return []
        name, parsed_tool_call_id = self._parse_namespace_segment(path[-1])
        trigger_call_id = tool_call_id or parsed_tool_call_id
        instance_id = ":".join(path)
        self._active[path] = (name, trigger_call_id)
        self.registry.register(path, name)
        return [
            self.emitter.subagent_start(
                name,
                "",
                instance_id=instance_id,
                tool_call_id=trigger_call_id or "",
            ).data
        ]

    def _end(self, path: tuple[str, ...]) -> list[dict[str, Any]]:
        active = self._active.pop(path, None)
        if active is None:
            return []
        name, _tool_call_id = active
        return [self.emitter.subagent_end(name, instance_id=":".join(path)).data]

    def _end_triggered_child(
        self,
        namespace: tuple[str, ...],
        result_id: object,
    ) -> list[dict[str, Any]]:
        if not result_id:
            return []
        events: list[dict[str, Any]] = []
        for path, (_name, tool_call_id) in list(self._active.items()):
            if path[:-1] == namespace and tool_call_id == result_id:
                events.extend(self._end(path))
        return events

    @staticmethod
    def _parse_namespace_segment(segment: str) -> tuple[str, str | None]:
        name, sep, task_id = segment.partition(":")
        return name, task_id if sep else None


@dataclass(slots=True)
class LangGraphServerGateway:
    """Gateway backed by a running LangGraph server."""

    thread_store: LangGraphServerThreadStore
    graph_id: str = DEFAULT_GRAPH_ID
    interrupt_wait_seconds: float = 5.0

    def _target_graph_id(self, target: GraphTarget | None = None) -> str:
        return target.graph_id if target is not None else self.graph_id

    async def create_thread(
        self,
        target: GraphTarget | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return await self.thread_store.create_thread(
            graph_id=self._target_graph_id(target),
            metadata=metadata,
            workspace_dir=target.workspace_dir if target is not None else None,
        )

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
            graph_id=self._target_graph_id(target),
        )

    async def resolve_thread(
        self,
        thread_id_or_prefix: str,
        target: GraphTarget | None = None,
    ) -> ThreadResolution:
        resolved, matches = await self.thread_store.resolve_thread_id_prefix(
            thread_id_or_prefix,
            graph_id=self._target_graph_id(target),
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
    ) -> list[BaseMessage]:
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
        return await self.thread_store.clone_thread(
            source_thread_id,
            metadata=metadata,
        )

    async def _start_or_resume(
        self,
        stream: AsyncThreadStream,
        request: RunRequest,
    ) -> None:
        config: dict[str, Any] = {"configurable": {"thread_id": request.thread_id}}
        await self.thread_store.ensure_thread_exists(
            request.thread_id,
            graph_id=self._target_graph_id(request.target),
            metadata=request.metadata,
            workspace_dir=(
                request.target.workspace_dir if request.target is not None else None
            ),
        )
        request_workspace = (
            request.target.workspace_dir if request.target is not None else None
        )
        if request.metadata or request_workspace is not None:
            await self.thread_store.client.threads.update(
                request.thread_id,
                metadata=_build_thread_metadata(
                    graph_id=self._target_graph_id(request.target),
                    workspace_dir=request_workspace,
                    metadata=request.metadata,
                ),
            )
        if isinstance(request.message, Command):
            if request.message.resume is not None:
                await self._respond_to_interrupt(stream, request.message.resume)
                return
            raise RuntimeError(
                "LangGraph server gateway only supports Command(resume=...) messages."
            )

        run_input = await build_agent_stream_input(
            request.message,
            media=request.media,
        )
        await stream.run.start(
            input=run_input,
            config=config,
            metadata=request.metadata,
        )

    async def _respond_to_interrupt(
        self,
        stream: AsyncThreadStream,
        response: object,
    ) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.interrupt_wait_seconds
        while not stream.interrupts and loop.time() < deadline:
            await asyncio.sleep(0.05)
        interrupt_id = None
        if len(stream.interrupts) == 1:
            interrupt_id = str(stream.interrupts[0].get("interrupt_id") or "")
        await stream.run.respond(response, interrupt_id=interrupt_id or None)

    def stream_events(self, request: RunRequest) -> AsyncIterator[GraphEvent]:
        return self._stream_events(request)

    async def get_state_values(
        self,
        target: GraphTarget,
        thread_id: str,
    ) -> GraphStateValues:
        return await self._get_state_values(thread_id)

    async def update_state_values(
        self,
        target: GraphTarget,
        thread_id: str,
        values: GraphStateValues,
    ) -> None:
        as_node = "model" if "_summarization_event" in values else None
        await self.thread_store.client.threads.update_state(
            thread_id,
            values,
            as_node=as_node,
        )

    async def _get_state_values(self, thread_id: str) -> GraphStateValues:
        state = await self.thread_store.client.threads.get_state(thread_id)
        values = state.get("values")
        if not isinstance(values, dict):
            return {}
        return {str(key): value for key, value in values.items()}

    async def _pending_interrupt_events(
        self,
        stream: AsyncThreadStream,
        thread_id: str,
        processor: _V3EventProcessor,
    ) -> list[GraphEvent]:
        events: list[GraphEvent] = []
        for interrupt in stream.interrupts:
            events.extend(
                await processor.process(
                    _input_requested_event_from_interrupt(interrupt)
                )
            )

        if events or not stream.interrupted:
            return events

        try:
            state = await self.thread_store.client.threads.get_state(thread_id)
        except NotFoundError:
            return events

        for interrupt in _state_interrupts(state):
            events.extend(
                await processor.process(
                    _input_requested_event_from_interrupt(interrupt)
                )
            )
        return events

    async def _stream_events(self, request: RunRequest) -> AsyncIterator[GraphEvent]:
        emitter = StreamEventEmitter()
        state_values: GraphStateValues = {}
        existing_summarization_event: Mapping[str, object] | None = None
        process_value_messages = True
        try:
            state_values = await self._get_state_values(request.thread_id)
            existing_summarization_event = _find_summarization_event_payload(
                state_values
            )
        except NotFoundError:
            pass
        except Exception:
            process_value_messages = False

        subagents = _SubagentRegistry()
        processor = _V3EventProcessor(
            emitter,
            subagents,
            existing_summarization_event,
            state_values.get("messages"),
            process_value_messages=process_value_messages,
        )
        tracker = _ServerSubagentTracker(emitter, subagents)
        stream = self.thread_store.client.threads.stream(
            request.thread_id,
            assistant_id=self._target_graph_id(request.target),
        )

        try:
            async with stream:
                await self._start_or_resume(stream, request)
                emitted_interrupt = False
                async for event in stream.subscribe(_RUN_SUBSCRIBE_CHANNELS):
                    raw_event = _as_raw_map(event)
                    if raw_event is None:
                        continue
                    event_map: dict[str, Any] = dict(raw_event)
                    for subagent_event in tracker.process(event_map):
                        yield subagent_event
                    for normalized in await processor.process(event_map):
                        emitted_interrupt = emitted_interrupt or _is_interrupt_event(
                            normalized
                        )
                        yield normalized
                if not emitted_interrupt:
                    for event in await self._pending_interrupt_events(
                        stream,
                        request.thread_id,
                        processor,
                    ):
                        yield event
        except Exception as exc:
            yield emitter.error(str(exc)).data
            raise
        finally:
            for event in tracker.finish():
                yield event
        yield emitter.done(processor.full_response).data
