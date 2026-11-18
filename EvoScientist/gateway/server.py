"""LangGraph server-backed gateway implementation."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..middleware.events import SessionEvents

from langchain_core.messages import BaseMessage, convert_to_messages, messages_from_dict
from langgraph.types import Command
from langgraph_sdk._async.stream import AsyncThreadStream
from langgraph_sdk.client import LangGraphClient
from langgraph_sdk.errors import NotFoundError
from langgraph_sdk.schema import Thread, ThreadState

from ..langgraph_dev.assistants import launch_assistant_id
from ..langgraph_dev.manager import RUNTIME
from ..message_meta import is_pinned_skill, is_rubric_feedback
from ..middleware.events import MIDDLEWARE_EVENT_TAG, MiddlewareEvent
from ..sessions import _apply_summarization_event
from ..stream.emitter import StreamEventEmitter
from ..stream.events import (
    _SubagentRegistry,
    _V3EventProcessor,
    build_agent_stream_input,
)
from ..stream.summarization import _find_summarization_event_payload
from ..stream.v3_payloads import _as_raw_map, _event_namespace, _text_from_content
from .background_runs import _acancel_thread_runs
from .types import (
    DEFAULT_GRAPH_ID,
    GraphEvent,
    GraphStateValues,
    GraphTarget,
    RunRequest,
    ThreadResolution,
    ThreadStore,
    resolve_per_run_config,
)

logger = logging.getLogger(__name__)

_THREAD_SEARCH_LIMIT = 1000
_RUN_REQUEST_ID_METADATA_KEY = "_evoscientist_gateway_request_id"
_RUN_LOOKUP_PAGE_SIZE = 100
_ACTIVE_RUN_STATUSES = ("pending", "running")
_REPLAY_QUIET_SECONDS = 2.0
_REPLAY_FIRST_EVENT_SECONDS = 5.0
_REPLAY_MAX_SECONDS = 30.0
# langgraph-api's text for an exception it does not pass on to clients.
_SERVER_HIDDEN_ERROR = "An internal error occurred"


def _run_has_request_id(run: Mapping[str, Any], request_id: str) -> bool:
    metadata = run.get("metadata")
    return (
        isinstance(metadata, Mapping)
        and metadata.get(_RUN_REQUEST_ID_METADATA_KEY) == request_id
    )


def _thread_metadata(thread: Thread) -> dict[str, Any]:
    metadata = thread.get("metadata")
    return dict(metadata) if isinstance(metadata, dict) else {}


def _build_thread_metadata(
    *,
    graph_id: str,
    workspace_dir: str | None,
    run_dir: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged = dict(metadata or {})
    merged["graph_id"] = graph_id
    if graph_id == DEFAULT_GRAPH_ID:
        merged["agent_name"] = DEFAULT_GRAPH_ID
    else:
        merged["agent_name"] = None
    if workspace_dir is not None:
        merged["workspace_dir"] = workspace_dir
    if run_dir is not None:
        merged["run_dir"] = run_dir
    merged.setdefault("updated_at", datetime.now(UTC).isoformat())
    return merged


def _thread_preview(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if (
            getattr(message, "type", None) != "human"
            or is_pinned_skill(message)
            or is_rubric_feedback(message)
        ):
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
                "payload": interrupt.get("value"),
            },
        },
    }


def _state_interrupts(state: ThreadState) -> list[Mapping[str, object]]:
    interrupts = state.get("interrupts")
    if not isinstance(interrupts, list):
        return []
    return [interrupt for interrupt in interrupts if isinstance(interrupt, Mapping)]


def _as_tuple(value: object) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _thread_state_as_snapshot(state: Mapping[str, Any]) -> SimpleNamespace:
    """Normalize SDK ``threads.get_state`` to the recovery snapshot surface.

    ``_recover_interrupted_graph_state`` inspects ``next``, ``tasks`` (with
    ``name`` / ``interrupts``), ``interrupts``, and ``values`` via attributes,
    matching ``langgraph.types.StateSnapshot``. The SDK returns a dict.
    """
    tasks: list[Any] = []
    for task in state.get("tasks") or ():
        if isinstance(task, Mapping):
            tasks.append(
                SimpleNamespace(
                    name=task.get("name") or "",
                    interrupts=_as_tuple(task.get("interrupts")),
                )
            )
        else:
            tasks.append(task)
    values = state.get("values")
    if not isinstance(values, dict):
        values = {}
    return SimpleNamespace(
        next=_as_tuple(state.get("next")),
        tasks=tuple(tasks),
        interrupts=_as_tuple(state.get("interrupts")),
        values=values,
    )


class _PauseStateReadError(RuntimeError):
    """``threads.get_state`` failed after a pause; leave the run paused."""


class _ReplayCapError(RuntimeError):
    """The thread SSE never went quiet within the replay cap."""


def _resume_target(
    thread_id: str, response: object, pending: list[str]
) -> tuple[str, object]:
    """The pending interrupt *response* answers, and the answer itself.

    A single-key payload keyed by a pending id answers that interrupt
    (``build_hitl_resume`` keys approvals that way). A single-key payload whose
    value is a mapping, keyed by any other id, answers that id too, and the
    server checks it: thread state lists a sub-graph's interrupt only once the
    sub-graph's other tasks finish, while its prompt arrives at once, and the
    server refuses a stale or hand-built id. Unwrapping it onto the pending
    interrupt would pass the wrapper as the answer. Any other payload answers
    the only pending interrupt: ask_user's ``{"status": "cancelled"}`` (a
    string value) and ``{"answers": [...], "status": "answered"}`` (two keys)
    are not keyed.
    """
    if isinstance(response, Mapping) and len(response) == 1:
        (key,) = response
        if str(key) in pending or isinstance(response[key], Mapping):
            return str(key), response[key]
    if not pending:
        raise RuntimeError(
            f"Thread {thread_id} has no pending interrupt; nothing to resume"
        )
    if len(pending) > 1:
        raise RuntimeError(
            f"Thread {thread_id} has {len(pending)} pending interrupts; "
            "resume requires an id-keyed payload matching one of them"
        )
    return pending[0], response


def _is_interrupt_event(event: Mapping[str, object]) -> bool:
    return event.get("type") in {"interrupt", "ask_user"}


def _synth_event_run_id(event: Mapping[str, Any]) -> str | None:
    """Return the run id in a synthetic ``synth:{run_id}:...`` event id, if any.

    Plain stream-entry ids carry no run id; they belong to the run whose
    synthetic lifecycle was seen last.
    """
    event_id = event.get("event_id")
    if not isinstance(event_id, str) or not event_id.startswith("synth:"):
        return None
    _, _, rest = event_id.partition(":")
    run_id, sep, _ = rest.partition(":")
    if not sep or not run_id:
        return None
    return run_id


def _root_lifecycle_data(event: Mapping[str, Any]) -> Mapping[str, object] | None:
    """The data of a root-namespace lifecycle envelope, else None."""
    if event.get("method") != "lifecycle" or _event_namespace(event):
        return None
    params = _as_raw_map(event.get("params"))
    return _as_raw_map(params.get("data")) if params is not None else None


def _root_lifecycle_phase(event: Mapping[str, Any]) -> object:
    """The phase of a root-namespace lifecycle envelope, else None."""
    data = _root_lifecycle_data(event)
    return data.get("event") if data is not None else None


def _run_failure_message(event: Mapping[str, Any]) -> str:
    """The turn's error for a root ``lifecycle: failed`` envelope."""
    data = _root_lifecycle_data(event)
    error = data.get("error") if data is not None else None
    if not error or error == _SERVER_HIDDEN_ERROR:
        return f"Run failed on the server; the cause is in {RUNTIME.log_file}"
    return f"Run failed: {error}"


class _ThreadEvents:
    """``stream.events`` read ahead into a queue, so the history replay can be skipped.

    A fresh thread SSE replays every earlier run first, and the server sends a
    run's envelopes only to connections that finished the replay before the
    run published them, so a turn starts its run after the replay.
    """

    def __init__(
        self,
        events: AsyncIterator[Any],
        *,
        quiet_seconds: float | None = None,
        first_event_seconds: float | None = None,
        max_seconds: float | None = None,
        clock: Callable[[], float] | None = None,
        wait_for: Callable[..., Any] = asyncio.wait_for,
    ) -> None:
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        # None means "read the module constants when skip_replay runs" so
        # tests can lower ``_REPLAY_*`` without reconstructing instances.
        self._quiet_seconds = quiet_seconds
        self._first_event_seconds = first_event_seconds
        self._max_seconds = max_seconds
        self._clock = clock or asyncio.get_running_loop().time
        self._wait_for = wait_for
        self._reader = asyncio.create_task(self._read(events))

    async def __aenter__(self) -> _ThreadEvents:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self._reader.cancel()
        try:
            await self._reader
        except asyncio.CancelledError:
            pass

    async def _read(self, events: AsyncIterator[Any]) -> None:
        try:
            async for event in events:
                self._queue.put_nowait(event)
        except Exception as exc:
            self._queue.put_nowait(exc)
        else:
            self._queue.put_nowait(None)

    async def skip_replay(self) -> int:
        """Drop envelopes until none arrives for the quiet window; return the count.

        The quiet window only starts after the first envelope. A fresh SSE can
        take longer than the quiet window to deliver the first replayed event,
        and treating that setup delay as the end of the replay would start the
        run too early and lose its live envelopes.

        Timeouts are read from the module ``_REPLAY_*`` constants when the
        constructor left them unset, so tests can lower them for the suite.
        """
        quiet_seconds = (
            _REPLAY_QUIET_SECONDS
            if self._quiet_seconds is None
            else self._quiet_seconds
        )
        first_event_seconds = (
            _REPLAY_FIRST_EVENT_SECONDS
            if self._first_event_seconds is None
            else self._first_event_seconds
        )
        max_seconds = (
            _REPLAY_MAX_SECONDS if self._max_seconds is None else self._max_seconds
        )
        deadline = self._clock() + max_seconds
        skipped = 0
        while (left := deadline - self._clock()) > 0:
            # Wait longer for the first event than for later quiet gaps: a
            # fresh SSE can take a moment to deliver the first replayed
            # envelope. After that, a short quiet stretch means the replay
            # is done.
            timeout = (
                min(first_event_seconds, left)
                if skipped == 0
                else min(quiet_seconds, left)
            )
            try:
                event = await self._wait_for(self._queue.get(), timeout)
            except TimeoutError:
                # A wait shorter than the quiet window means the overall cap
                # trimmed the timeout, so replay is still unfinished.
                if skipped > 0 and timeout < quiet_seconds:
                    raise _ReplayCapError(
                        f"Thread history replay did not finish within {max_seconds:g}s"
                    ) from None
                return skipped
            if event is None or isinstance(event, Exception):
                self._queue.put_nowait(event)
                return skipped
            skipped += 1
        # Hitting the cap means replay is still producing envelopes. Starting
        # the run now would miss its live events on this connection.
        raise _ReplayCapError(
            f"Thread history replay did not finish within {max_seconds:g}s"
        )

    def __aiter__(self) -> _ThreadEvents:
        return self

    async def __anext__(self) -> Any:
        event = await self._queue.get()
        if event is None:
            raise StopAsyncIteration
        if isinstance(event, Exception):
            raise event
        return event


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

    client: LangGraphClient
    graph_id: str = DEFAULT_GRAPH_ID

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
        run_dir: str | None = None,
    ) -> str:
        target_graph_id = self._target_graph_id(graph_id)
        thread = await self.client.threads.create(
            graph_id=target_graph_id,
            metadata=_build_thread_metadata(
                graph_id=target_graph_id,
                workspace_dir=workspace_dir,
                run_dir=run_dir,
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
        run_dir: str | None = None,
    ) -> None:
        target_graph_id = self._target_graph_id(graph_id)
        await self.client.threads.create(
            thread_id=thread_id,
            graph_id=target_graph_id,
            metadata=_build_thread_metadata(
                graph_id=target_graph_id,
                workspace_dir=workspace_dir,
                run_dir=run_dir,
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
                "run_dir": metadata.get("run_dir"),
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
        # Interrupt live runs first: the server's cascade delete clears
        # queued runs from the registry but does not stop a run that is
        # already executing (issue #358).
        try:
            await _acancel_thread_runs(
                self.client, thread_id, name="thread delete", strict=True
            )
        except NotFoundError:
            # The listed runs have ended, or the thread is already gone.
            pass
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


def _lifecycle_graph_name(data: Mapping[str, Any] | None) -> str | None:
    """The sub-agent name a lifecycle start event carries, if any.

    LangGraph's lifecycle transformer copies the nested run's
    ``lc_agent_name`` (its own or one inherited from the parent) into
    ``graph_name`` on its start event, and the namespace segment when there
    is none. The server's stream suppresses the raw ``tasks`` events that
    carry the same name in their metadata, so this is where the name
    actually arrives.
    """
    if data is None:
        return None
    name = data.get("graph_name")
    if not isinstance(name, str):
        return None
    stripped = name.strip()
    return stripped or None


def _lifecycle_is_subagent(data: Mapping[str, Any] | None) -> bool:
    """Whether the lifecycle event names a real sub-agent.

    LangGraph's lifecycle transformer sets ``cause`` to a ``toolCall`` only
    when the nested run was started by a tool call -- a run started by
    anything else (an agent run inside a middleware hook, for instance) gets
    no cause at all. The cause is therefore what tells the two apart, and it
    decides on its own.

    The absence of a cause means no tool call started a named run; a
    tool-started run without a name keeps its namespace segment either way.
    """
    if data is None:
        return False
    cause = _as_raw_map(data.get("cause"))
    if cause is None:
        return False
    return cause.get("type") == "toolCall"


def _lifecycle_cause_tool_call_id(data: Mapping[str, Any] | None) -> str | None:
    """The LLM tool call that started a nested run, if the event names it.

    The transformer's cause carries the tool call the model made, so the
    sub-agent can be announced under its own ``task`` line -- the way the
    local gateway does -- rather than under the namespace segment.
    """
    if data is None:
        return None
    cause = _as_raw_map(data.get("cause"))
    if cause is None:
        return None
    call_id = cause.get("tool_call_id")
    if not isinstance(call_id, str):
        return None
    stripped = call_id.strip()
    return stripped or None


# Async sub-agents run on their own threads, so the server stream never
# carries their lifecycle events. The tracker names them from what the
# parent stream does carry -- the launch tool call and the async_tasks
# state -- under a synthetic path, through the same _ensure_registered /
# _end rule as the sync children. Only the name source differs: the sync
# path reads it from the lifecycle event's graph_name, the async path from
# the start_async_task arguments and the async_tasks record.
_ASYNC_PATH_PREFIX = ("async",)

# Terminal async-task statuses: the child is done and gets its end event.
# Mirrors cli.async_notifier.TERMINAL_STATUSES; keep the two in step.
# "waiting" is deliberately absent: a task waiting for input on its own
# thread is still alive.
_ASYNC_TERMINAL_STATUSES = frozenset(
    {"cancelled", "success", "error", "timeout", "interrupted"}
)

# The launch line _build_task_envelope writes into the tool result
# (middleware/expert_async_subagent.py). A tool-finished carrying it is a
# launch that succeeded; any other output is a failed launch.
_ASYNC_LAUNCH_PREFIX = "Launched async subagent. task_id: "


def _tool_input_subagent_type(input_args: object) -> str | None:
    """The subagent_type a start_async_task call was launched with, if any."""
    input_map = _as_raw_map(input_args)
    if input_map is None:
        return None
    name = input_map.get("subagent_type")
    if not isinstance(name, str):
        return None
    stripped = name.strip()
    return stripped or None


def _message_text(message: object) -> str:
    """The text of a serialized message, whatever shape it arrived in."""
    if isinstance(message, str):
        return message
    message_map = _as_raw_map(message)
    if message_map is None:
        return ""
    text = _text_from_content(message_map.get("content"))
    if text:
        return text
    # LangGraph serializes messages as {"type": ..., "data": {...}}.
    nested = _as_raw_map(message_map.get("data"))
    if nested is not None:
        return _text_from_content(nested.get("content"))
    return ""


def _launched_task(output: object) -> tuple[str | None, str | None, str | None]:
    """The (task_id, agent_name, run_id) a task Command result names, if any.

    The tool result is a serialized Command whose ``update`` holds the
    task record (see ``middleware/expert_async_subagent``: a launch
    writes it with ``_build_task_envelope``; a check, an update and a
    list write it with their own builders). A failed call returns a
    plain string instead, and a serialized ToolMessage carries the
    launch text in its content. Either way the task id comes from the
    result itself, never from a guess.
    """
    if isinstance(output, str):
        task_id = _launched_task_id(output)
        return (task_id, None, None) if task_id is not None else (None, None, None)
    output_map = _as_raw_map(output)
    if output_map is None:
        return (None, None, None)
    update = _as_raw_map(output_map.get("update"))
    if update is None:
        task_id = _launched_task_id(_message_text(output))
        return (task_id, None, None) if task_id is not None else (None, None, None)
    task_id: str | None = None
    agent_name: str | None = None
    run_id: str | None = None
    tasks = _as_raw_map(update.get("async_tasks"))
    if isinstance(tasks, Mapping):
        for candidate_id, record in tasks.items():
            record_map = _as_raw_map(record)
            if record_map is None:
                continue
            task_id = candidate_id if isinstance(candidate_id, str) else None
            agent = record_map.get("agent_name")
            agent_name = agent if isinstance(agent, str) and agent.strip() else None
            run = record_map.get("run_id")
            run_id = run if isinstance(run, str) and run else None
            break  # one launch names one task
    if task_id is None:
        # Fall back to the launch text in the update's messages.
        messages = update.get("messages")
        if isinstance(messages, list):
            for message in messages:
                task_id = _launched_task_id(_message_text(message))
                if task_id is not None:
                    break
    return (task_id, agent_name, run_id)


def _launched_task_id(text: str) -> str | None:
    """The task id a successful launch result names, if it names one."""
    if not text.startswith(_ASYNC_LAUNCH_PREFIX):
        return None
    task_id = text[len(_ASYNC_LAUNCH_PREFIX) :].strip()
    return task_id or None


def _notification_event_from_kwargs(
    kwargs: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """The task event a notification message carries in its metadata, if any.

    deepagents labels the message twice: langchain's ``lc_source`` /
    ``lc_data`` (source "async_subagent") and its own
    ``deepagents_notification`` key. Either one names the finished task.
    """
    if kwargs.get("lc_source") == "async_subagent":
        data = _as_raw_map(kwargs.get("lc_data"))
        if data is not None:
            return data
    event = _as_raw_map(kwargs.get("deepagents_notification"))
    if event is not None:
        return event
    return None


@dataclass(slots=True)
class _ServerSubagentTracker:
    """Infer subagent start/end events from LangGraph server namespaces."""

    emitter: StreamEventEmitter
    registry: _SubagentRegistry
    _active: dict[tuple[str, ...], str] = field(default_factory=dict)
    # Paths whose child already ended. A late event for one of these must not
    # announce a second start with no matching end.
    _ended: set[tuple[str, ...]] = field(default_factory=set)
    # start_async_task calls seen but not yet resolved: tool_call_id ->
    # subagent_type. A launch is only announced once it succeeds, so a
    # failed launch leaves nothing behind.
    _pending_async: dict[str, str] = field(default_factory=dict)
    # update_async_task calls seen but not yet resolved: tool_call_ids.
    # An update can resume a finished task onto a new run, so its result
    # may name a run the record is allowed to re-arm on (see
    # _async_pending_runs). A check or a list merely observes the task
    # and is never tracked here.
    _pending_async_updates: set[str] = field(default_factory=set)
    # Run ids an update tool-finished named for each async child path.
    # A record is state, not a launch event: it announces or re-arms a
    # child only when its run id is one an update (or a launch) actually
    # produced in this turn. Without that, a check of an earlier-turn
    # task would announce a child that was never launched here. The
    # gate is the only signal available for this, and it is not a
    # proof: it cannot order two pending runs (both are in the set, so
    # the later one can arrive first and move the child backwards),
    # and it cannot tell an external client's genuine new run -- one
    # no update in this turn produced -- from a stale record.
    _async_pending_runs: dict[tuple[str, ...], set[str]] = field(default_factory=dict)
    # Current run id per async child path. A completion naming an older run
    # than the one the child is on is stale and must not end it.
    _async_run_ids: dict[tuple[str, ...], str | None] = field(default_factory=dict)
    # Tool call id that launched each async task (task_id -> tool_call_id).
    # The tool-finished Command names the task it launched, so the launch
    # call id is correlated by task id -- never guessed by agent name,
    # which would swap call ids when two same-name children launch at
    # once. The id is placement only: it is carried on the start event
    # so the CLI places the child under its own task line, never a
    # guard on an end (no completion signal carries one).
    _async_launch_calls: dict[str, str] = field(default_factory=dict)
    # Run id each ended async child was on when it ended (path -> run id).
    # A terminal signal for an older run must not close a newer run, and a
    # late non-terminal record for the ended run must not re-announce it.
    # The tombstone is kept across a revival: it is the only way a
    # run-id-less signal can tell a revived child from the run it ended on.
    _ended_async_runs: dict[tuple[str, ...], str | None] = field(default_factory=dict)
    # Run ids ever observed for each async child path. A record naming a
    # run that was already seen while the child is on a different run is
    # stale in the cases the tests cover: the current run should not
    # move backwards, and a revival should name a run never seen
    # before. That is the intent, not a guarantee the tracker can
    # keep: runs are NOT delivered in order -- parallel tool calls
    # gather their outputs in call order, so a record for an old run
    # can arrive after a newer run ended -- and when two runs are both
    # pending, the pending-run gate cannot order them, so the later
    # one can arrive first and move the child backwards. The seen set
    # alone cannot tell a stale never-seen run from a genuine new one;
    # the pending-run gate in _process_async_task_record is the only
    # signal for that, and it is not sufficient (see the note on
    # _async_pending_runs).
    _async_seen_runs: dict[tuple[str, ...], set[str]] = field(default_factory=dict)

    def process(self, event: Mapping[str, Any]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        namespace = tuple(_event_namespace(event))
        method = event.get("method")
        params = _as_raw_map(event.get("params"))
        data = _as_raw_map(params.get("data")) if params is not None else None

        if method == "lifecycle":
            phase = data.get("event") if data is not None else None
            if phase == "started" and namespace:
                # A nested run announces its own name in ``graph_name``: the
                # server's lifecycle transformer copies the sub-agent's
                # ``lc_agent_name`` there when the nested run is a sub-agent.
                # Without it the name falls back to the namespace segment,
                # which is the tool name -- so a sync sub-agent showed up as
                # "tools" (#791).
                events.extend(
                    self._ensure_registered(
                        namespace,
                        tool_call_id=_lifecycle_cause_tool_call_id(data),
                        name=self._lifecycle_name(data),
                    )
                )
            elif phase in ("completed", "failed") and namespace:
                events.extend(self._end(namespace))
        elif method == "tools":
            events.extend(self._process_async_tool_event(data))
            self._register_head_for_routing(namespace)
        elif method == "updates":
            events.extend(self._process_async_update_event(data))
            self._register_head_for_routing(namespace)
        elif namespace:
            self._register_head_for_routing(namespace)
        return events

    def finish(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for path in sorted(
            self._active.keys(), key=lambda item: len(item), reverse=True
        ):
            if self._is_async_path(path):
                # An async child outlives the parent turn: the run that
                # launched it is over, but the child keeps working on its
                # own thread. Its end is emitted when its completion
                # reaches the tracker in a later turn, not here -- ending
                # it now would report a still-running child as done.
                continue
            events.extend(self._end(path))
        self._active.clear()
        self._ended.clear()
        self._pending_async.clear()
        self._pending_async_updates.clear()
        self._async_pending_runs.clear()
        self._async_run_ids.clear()
        self._async_launch_calls.clear()
        self._ended_async_runs.clear()
        self._async_seen_runs.clear()
        self.registry.close()
        return events

    def _lifecycle_name(self, data: Mapping[str, Any] | None) -> str | None:
        """The name to announce for a nested run the lifecycle event names.

        The transformer copies any ``lc_agent_name`` the nested run carries
        into ``graph_name``, including one inherited from the parent (an
        unnamed agent run inside a middleware hook), so the name counts only
        when ``cause`` says a tool call started the run.
        """
        if not _lifecycle_is_subagent(data):
            return None
        return _lifecycle_graph_name(data)

    def _register_head_for_routing(
        self, namespace: tuple[str, ...]
    ) -> None:
        """Register a namespace head for routing only, never announcing.

        Mirrors the merged #852 fallback: a child's input values come
        before its lifecycle start, and the server mirrors tool results
        under ``tools:<call id>`` -- registering the head lets the event
        processor route, while only a lifecycle start announces a child.
        Announcing here would name a child after the namespace segment
        (a ``tools`` sub-agent that does not exist). The ``tasks`` branch
        the old base had is gone with it: the lifecycle transformer
        suppresses tasks events, so they never reach the server log.
        """
        if not namespace:
            return
        head = namespace[:1]
        if self.registry.resolve(head) is None:
            self.registry.register(head, self._parse_namespace_segment(head[0])[0])

    def _ensure_registered(
        self,
        path: tuple[str, ...],
        *,
        tool_call_id: str | None,
        name: str | None = None,
    ) -> list[dict[str, Any]]:
        if not path:
            return []
        if path in self._ended:
            # The child already ended. A late start for it must not announce
            # a second start, and must not overwrite the registry's name.
            return []
        active = self._active.get(path)
        if active is not None:
            # Already announced: the start is not repeated and the name is not
            # rewritten, so a late event cannot rename a child the user
            # already sees.
            return []
        fallback_name, parsed_tool_call_id = self._parse_namespace_segment(path[-1])
        resolved = name or fallback_name
        instance_id = ":".join(path)
        self._active[path] = resolved
        self.registry.register(path, resolved)
        return [
            self.emitter.subagent_start(
                resolved,
                "",
                instance_id=instance_id,
                tool_call_id=tool_call_id or parsed_tool_call_id or "",
            ).data
        ]

    def _end(self, path: tuple[str, ...]) -> list[dict[str, Any]]:
        name = self._active.get(path)
        if name is None:
            return []
        self._active.pop(path, None)
        self._ended.add(path)
        end = self.emitter.subagent_end(name, instance_id=":".join(path))
        return [end.data]

    @staticmethod
    def _async_path(task_id: str) -> tuple[str, str]:
        """The tracker path for an async child, keyed by its task id."""
        return (*_ASYNC_PATH_PREFIX, task_id)

    @staticmethod
    def _is_async_path(path: tuple[str, ...]) -> bool:
        return path[:1] == _ASYNC_PATH_PREFIX

    def _register_async(
        self,
        task_id: str,
        name: str,
        trigger_call_id: str | None,
        *,
        run_id: str | None,
    ) -> list[dict[str, Any]]:
        """Announce an async child through the same rule as a sync one.

        The name comes from the launch instead of a lifecycle graph_name,
        but the registration and the fixed name are the same. The launch
        call id is carried on the start event as ``tool_call_id`` so the
        CLI places the child under its own task line, the way the sync
        path carries the cause's tool call id -- it is placement, not a
        guard. There is no trigger matching on an async end: no async
        completion signal carries a tool call id to compare against (a
        terminal record and a notification carry a task id and a run id
        only), so a match could never fire in production. The maintainer
        removed the same matching from the sync path for the same reason
        (the lifecycle terminal payload carries no trigger call id). The
        protection for an async completion is the run-id checks, not a
        trigger comparison.

        A non-terminal record for an ended path is a follow-up run
        that revived the task ONLY when it names a run never seen before
        for this path; a late record for a run already seen, or one with
        no run id at all, cannot prove a revival and announces nothing.
        The tombstone is deliberately kept across the revival: it is the
        only way a run-id-less signal can tell the revived child from the
        run it ended on. The record handler additionally requires a
        pending run -- a run an update tool-finished actually produced
        in this turn -- before it calls this; a launch announces
        directly from its own tool-finished result.

        Unlike a sync namespace, an async task id IS reused across runs
        of the same task -- update_async_task resumes a finished task
        onto a new run under the same task id -- so the path must be able
        to come back for the next run. That is the one place the async
        path keeps a re-arm the sync path dropped: the sync namespace
        carries a fresh id per nested run and is never reused, while an
        async task id is the task's own identity across its runs.
        """
        path = self._async_path(task_id)
        if path in self._ended:
            seen = self._async_seen_runs.get(path, set())
            if run_id is None or run_id in seen:
                return []
            self._ended.discard(path)
        if run_id is not None:
            self._async_run_ids[path] = run_id
            self._async_seen_runs.setdefault(path, set()).add(run_id)
        if trigger_call_id is not None:
            self._async_launch_calls[task_id] = trigger_call_id
        return self._ensure_registered(
            path,
            tool_call_id=trigger_call_id,
            name=name,
        )

    def _end_async(self, path: tuple[str, ...]) -> list[dict[str, Any]]:
        """End an async child.

        There is no trigger matching here: no async completion signal
        carries a tool call id to compare against, so a match could never
        fire in production -- the same reason the maintainer removed the
        matching from the sync path. The protection for an async
        completion is the run-id tombstone below, kept by the caller.
        """
        events = self._end(path)
        if events:
            # Remember the run the child ended on: a late signal for that
            # run must not re-announce it, and a terminal signal for it
            # must not close a child revived onto a newer run. Only the
            # pending run for the run that just ended dies with the
            # child: each run is produced by exactly one tool call, so
            # the run that ended is the only pending run the ending
            # signal can be about. A pending run for a different run
            # was recorded by a different, parallel call -- an update
            # that resumed the task -- and it survives: its record may
            # still arrive and legitimately revive the child.
            ended_run = self._async_run_ids.get(path)
            self._ended_async_runs[path] = ended_run
            self._async_run_ids.pop(path, None)
            pending = self._async_pending_runs.get(path)
            if pending is not None:
                if ended_run is not None:
                    pending.discard(ended_run)
                if not pending:
                    self._async_pending_runs.pop(path, None)
        return events

    def _end_async_by_task(
        self,
        task_id: str,
        name: str | None,
        run_id: str | None,
    ) -> list[dict[str, Any]]:
        """End an async child by task id.

        When the launch was in an earlier turn, this turn's tracker never
        saw it: the start was already announced there, so the child is
        registered silently and only the end is emitted. A completion
        naming an older run than the one the child is on is stale and ends
        nothing -- that applies to terminal records as well as to
        notifications, because a record is only the state as last streamed,
        not proof of what the child is doing now. A signal with no run id
        ends a child that has not been revived, but not one that ended and
        was revived onto a newer run: the signal cannot say which run it
        is for, and the likelier reading is the ended run's late
        completion.
        """
        path = self._async_path(task_id)
        if path in self._ended:
            ended_run = self._ended_async_runs.get(path)
            if run_id is None or run_id == ended_run:
                return []
            if run_id in self._async_seen_runs.get(path, set()):
                # A terminal record for a run already seen is stale: the
                # child already ended on a newer run, and this record
                # must not re-arm it only to end it a second time and
                # rewrite the tombstone to the old run.
                return []
            pending = self._async_pending_runs.get(path)
            if not pending or run_id not in pending:
                # A terminal record for a run no tool call in this turn
                # produced is a stale cached record, not a follow-up run
                # that finished without its start streaming: a follow-up
                # run is produced by an update, and the update records
                # its run as pending. Without that, a cached list entry
                # for an old run would re-arm the child only to end it
                # a second time and rewrite the tombstone to the old run.
                return []
            # A terminal signal for a follow-up run an update produced
            # in this turn, which finished without its start streaming.
            # Re-arm, then end it below. The tombstone is kept: the end
            # below overwrites it with the run it ends on.
            pending.discard(run_id)
            self._ended.discard(path)
        if path in self._active:
            stored = self._async_run_ids.get(path)
            if run_id is not None and stored is not None and run_id != stored:
                return []
            if run_id is None and stored is not None and path in self._ended_async_runs:
                # A signal with no run id cannot say which run it is for.
                # The path ended before and was revived onto a newer run
                # (that is what a tombstone plus a stored run means), so
                # the likelier reading is the ended run's late completion,
                # not the revived child's. The tombstone is checked by
                # membership, not value: a child that ended with no known
                # run leaves a None tombstone, and None must not read as
                # "no tombstone".
                return []
            return self._end_async(path)
        if name is None:
            return []
        self._active[path] = name
        self.registry.register(path, name)
        if run_id is not None:
            self._async_run_ids[path] = run_id
            self._async_seen_runs.setdefault(path, set()).add(run_id)
        return self._end_async(path)

    def _process_async_tool_event(
        self, data: Mapping[str, Any] | None
    ) -> list[dict[str, Any]]:
        """Track start_async_task calls: remember the launch, name it on success.

        The start is announced from the launch result, never from the call
        itself: a failed launch must announce nothing. A launch result is
        a serialized Command whose update holds the task record; the
        record names the task, its agent and its run, and the call id
        that produced it is recorded as the launch call id and carried
        on the start event, so the CLI places the child under its own
        task line. It is placement, not a guard: no async completion
        signal carries a tool call id, so there is no trigger matching
        on an async end.

        A record is state, not a launch event, so it never announces a
        child on its own: the launch result is the start signal, and an
        update that resumes a finished task is the only other start
        signal (its result names the new run, which the record may then
        re-arm on). A check or a list merely observes the task, so its
        result sets no pending run and its record announces nothing.

        Only ``tool-started`` carries a tool_name (langgraph's tool
        events): ``tool-finished`` and ``tool-error`` carry only the
        event, the tool call id and the result, so the call id alone
        routes them.
        """
        if data is None:
            return []
        tool_call_id = data.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            return []
        phase = data.get("event")
        if phase == "tool-started":
            if data.get("tool_name") == "update_async_task":
                self._pending_async_updates.add(tool_call_id)
                return []
            if data.get("tool_name") != "start_async_task":
                return []
            subagent_type = _tool_input_subagent_type(data.get("input"))
            if subagent_type is not None:
                self._pending_async[tool_call_id] = subagent_type
            return []
        if phase == "tool-finished":
            # Only a start_async_task call we saw start can be a launch:
            # check_async_task, update_async_task and list_async_tasks
            # return Commands whose updates also carry async_tasks records,
            # and mistaking one of those for a launch would announce a
            # child that was never launched and overwrite the real
            # launch's trigger.
            subagent_type = self._pending_async.pop(tool_call_id, None)
            if subagent_type is not None:
                task_id, agent_name, run_id = _launched_task(data.get("output"))
                name = agent_name or subagent_type
                if task_id is None or name is None:
                    return []
                self._async_launch_calls[task_id] = tool_call_id
                events = self._register_async(
                    task_id, name, tool_call_id, run_id=run_id
                )
                path = self._async_path(task_id)
                if path in self._active:
                    # The child was already active -- a re-launch of the
                    # same task, or a run an update resumed -- and this
                    # launch is the one that owns it now, so the name is
                    # corrected to the real launch's name. (The launch
                    # call id is already recorded in _async_launch_calls
                    # above; there is no trigger stored on the active
                    # path to correct, since _active holds names only.)
                    self._active[path] = name
                return events
            if tool_call_id in self._pending_async_updates:
                # An update can resume a finished task onto a new run.
                # Its result names that run, so the record that follows
                # may re-arm on it. A check or a list never reaches here.
                self._pending_async_updates.discard(tool_call_id)
                task_id, _agent_name, run_id = _launched_task(data.get("output"))
                if task_id is not None and run_id is not None:
                    self._async_pending_runs.setdefault(
                        self._async_path(task_id), set()
                    ).add(run_id)
                return []
            return []
        if phase == "tool-error":
            self._pending_async.pop(tool_call_id, None)
            self._pending_async_updates.discard(tool_call_id)
            return []
        return []

    def _process_async_update_event(
        self, data: Mapping[str, Any] | None
    ) -> list[dict[str, Any]]:
        """Name async children from state, end them on their completion."""
        if data is None:
            return []
        events: list[dict[str, Any]] = []
        self._process_async_state_update(data, events)
        return events

    def _process_async_state_update(
        self, data: object, events: list[dict[str, Any]]
    ) -> None:
        """Read async_tasks/messages from one state update.

        The server normalizer shapes an updates payload three ways: a
        single-node update arrives top-level (``{"async_tasks": ...}``);
        parallel tool calls arrive as ``{"value": [...]}``, one state dict
        per call; and a multi-node update arrives node-keyed
        (``{"agent": {...}, "tools": {...}}``). All three are read the
        same way.
        """
        if not isinstance(data, Mapping):
            return
        tasks = data.get("async_tasks")
        if isinstance(tasks, Mapping):
            for task_id, record in tasks.items():
                events.extend(self._process_async_task_record(task_id, record))
        messages = data.get("messages")
        if isinstance(messages, list):
            for message in messages:
                events.extend(self._process_async_notification_message(message))
        value = data.get("value")
        if isinstance(value, list):
            for item in value:
                self._process_async_state_update(item, events)
        for key, node_update in data.items():
            if key in ("async_tasks", "messages", "value"):
                continue
            self._process_async_state_update(node_update, events)

    def _process_async_task_record(
        self, task_id: object, record: object
    ) -> list[dict[str, Any]]:
        """One async_tasks record: announce a live task, end a finished one."""
        if not isinstance(task_id, str) or not task_id:
            return []
        record_map = _as_raw_map(record)
        if record_map is None:
            return []
        name = record_map.get("agent_name")
        if not isinstance(name, str) or not name.strip():
            return []
        status = record_map.get("status")
        run_id = record_map.get("run_id")
        run_id = run_id if isinstance(run_id, str) and run_id else None
        path = self._async_path(task_id)
        if isinstance(status, str) and status in _ASYNC_TERMINAL_STATUSES:
            # A terminal record is only the state as last streamed: it can
            # name an older run than the one the child is on, so its run
            # id is checked like any other completion signal.
            return self._end_async_by_task(task_id, name, run_id)
        if path in self._active:
            # Still running. The current run never moves backwards: a
            # record may move it only to a run a tool call in this turn
            # actually produced -- the run it is already on, a run an
            # update recorded as pending, or (when the launch never told
            # us its run) the first run we learn. Any other run is a
            # stale record from an earlier turn or a cached list, and it
            # changes nothing. Run ids carry no ordering, so a
            # never-seen old run looks exactly like a genuine new one;
            # only the pending-run gate tells them apart.
            if run_id is not None:
                seen = self._async_seen_runs.setdefault(path, set())
                stored = self._async_run_ids.get(path)
                pending = self._async_pending_runs.get(path)
                if (
                    stored is not None
                    and run_id != stored
                    and not (pending and run_id in pending)
                ):
                    return []
                self._async_run_ids[path] = run_id
                seen.add(run_id)
                if pending is not None:
                    pending.discard(run_id)
            return []
        # A record is state, not a launch event. It announces a child
        # only when a run-causing tool call actually produced that run
        # in this turn: a launch (which announces itself, so the record
        # is redundant there) or an update that resumed a finished task
        # onto a new run. A check or a list merely observes the task,
        # so its record must not announce a child the agent never
        # launched here -- that is how a check of an earlier-turn task
        # used to emit a false start. And a record for a run no tool
        # call produced cannot be a genuine new run: parallel tool
        # calls gather their outputs in call order, so a stale record
        # for an old run can arrive after a newer run ended, and
        # re-arming on it would revive a child that is done.
        pending = self._async_pending_runs.get(path)
        if run_id is None or not pending or run_id not in pending:
            return []
        pending.discard(run_id)
        trigger = self._async_launch_calls.get(task_id)
        return self._register_async(task_id, name, trigger, run_id=run_id)

    def _process_async_notification_message(
        self, message: object
    ) -> list[dict[str, Any]]:
        """End async children a deepagents completion notification names.

        The notification arrives as a user message on the parent thread,
        labelled in its metadata (langchain's ``lc_source`` / ``lc_data``
        and deepagents' ``deepagents_notification`` key). It is read from
        the full message the updates event carries, never from streamed
        deltas. The CLI's synthetic "[Async tasks update]" message is
        NOT read here: it is run input, not an updates payload, and the
        completion it reports reaches the tracker through the terminal
        async_tasks record the task check writes after the message wakes
        the agent.
        """
        msg = _as_raw_map(message)
        if msg is None:
            return []
        # LangGraph serializes messages as {"type": ..., "data": {...}}.
        nested = _as_raw_map(msg.get("data"))
        if nested is not None:
            msg = nested
        kwargs = _as_raw_map(msg.get("additional_kwargs"))
        if kwargs is None:
            return []
        event = _notification_event_from_kwargs(kwargs)
        if event is None:
            return []
        return self._end_from_notification_event(event)

    def _end_from_notification_event(
        self, event: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        task_id = event.get("task_id")
        status = event.get("status")
        if not isinstance(task_id, str) or not task_id:
            return []
        if not isinstance(status, str) or status not in _ASYNC_TERMINAL_STATUSES:
            return []
        name = event.get("agent_name")
        name = name if isinstance(name, str) and name.strip() else None
        run_id = event.get("run_id")
        run_id = run_id if isinstance(run_id, str) and run_id else None
        return self._end_async_by_task(task_id, name, run_id)

    @staticmethod
    def _parse_namespace_segment(segment: str) -> tuple[str, str | None]:
        name, sep, task_id = segment.partition(":")
        return name, task_id if sep else None


@dataclass(slots=True)
class LangGraphServerGateway:
    """Gateway backed by a running LangGraph server."""

    thread_store: LangGraphServerThreadStore
    graph_id: str = DEFAULT_GRAPH_ID
    events: SessionEvents | None = None
    """Frontend event sink — delivery point for middleware custom events.

    Consumers read ``gateway.events`` via the ``GraphGateway`` protocol
    (``tui_interactive.py``, ``commands/implementation/model.py``). On the
    server backend this sink additionally receives middleware events
    (tool-selection lifecycle, fallback narration) mirrored by the server
    process onto the run's ``custom`` stream channel
    (``StreamBroadcastSink``): each tagged payload consumed from the stream
    is dispatched onto this sink during ``stream_events`` iteration.
    ``None`` (headless single-shot runs) drops those events.
    """

    def _target_graph_id(self, target: GraphTarget | None = None) -> str:
        return target.graph_id if target is not None else self.graph_id

    def _target_assistant_id(self, target: GraphTarget | None = None) -> str:
        """The assistant a turn on *target* runs on: its graph's assistant for
        the target's workspace, which names the project; the graph itself for
        a target without a workspace."""
        return launch_assistant_id(
            self._target_graph_id(target),
            target.workspace_dir if target is not None else None,
        )

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
            run_dir=target.run_dir if target is not None else None,
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
        metadata = await self.thread_store.get_thread_metadata(thread_id)
        if metadata is None or metadata.get("graph_id") != self._target_graph_id(
            target
        ):
            return False
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

    def _resolve_run_config(
        self,
        thread_id: str,
        configurable_extra: Mapping[str, Any] | None,
        *,
        target: GraphTarget | None = None,
        rubric_run: bool = False,
    ) -> dict[str, Any]:
        """Assemble this run's config, reading the live session config here.

        The target's folders go in last and replace any a caller put in
        ``configurable_extra``. The server reads the run's project from its
        assistant (or its thread's binding) and the run folder from
        ``run_dir``, and refuses a run folder that is not one of that
        project's.

        ``_ensure_config`` returns the cached, in-place-mutated session
        config — NOT a fresh disk read — so mid-session ``/model`` edits that
        have not been ``--save``d still reach the server per run. The
        ``configurable.model`` / ``model_provider`` overrides are picked up
        server-side by ``ConfigurableModelMiddleware``; ``recursion_limit``
        overrides the server's construction-time ``.with_config`` binding.
        """
        from ..backends import (
            HITL_APPROVE_ALL_KEY,
            hitl_suppressed_for_run,
            model_retry_for_run,
        )
        from ..EvoScientist import _ensure_config

        cfg = _ensure_config()
        overrides: dict[str, Any] = {}
        model = getattr(cfg, "model", None)
        provider = getattr(cfg, "provider", None)
        if model:
            overrides["model"] = model
        if provider:
            overrides["model_provider"] = provider
        limit = getattr(cfg, "recursion_limit", None)
        recursion_limit = (
            limit
            if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0
            else None
        )
        folders = (
            {"workspace_dir": target.workspace_dir, "run_dir": target.run_dir}
            if target is not None
            else {}
        )
        # The session grant never reaches the server. The server turns a new
        # message on a thread parked at an approval into a resume and keeps
        # this run's config, so a keyed run would execute a batch nobody
        # approved.
        extra = {
            key: value
            for key, value in (configurable_extra or {}).items()
            if key not in folders and key != HITL_APPROVE_ALL_KEY
        }
        config = resolve_per_run_config(
            thread_id,
            extra,
            per_run_overrides=overrides,
            recursion_limit=recursion_limit,
            hitl_suppressed=hitl_suppressed_for_run(cfg),
            rubric_run=rubric_run,
            model_retry=model_retry_for_run(cfg),
        )
        for key, value in folders.items():
            if value:
                config["configurable"][key] = value
        return config

    async def _ensure_thread(self, request: RunRequest) -> None:
        await self.thread_store.ensure_thread_exists(
            request.thread_id,
            graph_id=self._target_graph_id(request.target),
            metadata=request.metadata,
            workspace_dir=(
                request.target.workspace_dir if request.target is not None else None
            ),
            run_dir=request.target.run_dir if request.target is not None else None,
        )

    async def _start_or_resume(
        self,
        stream: AsyncThreadStream,
        request: RunRequest,
        *,
        thread_ready: bool = False,
        on_run_attempt: Callable[[], None],
        request_id: str,
        run_metadata: Mapping[str, object] | None = None,
    ) -> str | None:
        config = self._resolve_run_config(
            request.thread_id,
            request.configurable_extra,
            target=request.target,
            rubric_run=request.rubric is not None,
        )
        # ``_stream_events`` already registered the thread before the pre-run
        # state read. Skip the second ``threads.create(if_exists="do_nothing")``
        # when that succeeded; retry only if it raised (the warn-and-continue
        # branch swallows the failure and leaves the thread unregistered).
        if not thread_ready:
            await self._ensure_thread(request)
        # Refresh metadata on every run: ensure_thread_exists is a no-op on
        # existing threads (if_exists="do_nothing"), so without this update
        # fields like updated_at and model would go stale after the first run.
        # Best-effort: the graph is selected by the stream's assistant_id and
        # the workspace by server config, so a failed refresh only leaves
        # listing metadata stale until the next successful turn.
        metadata = _build_thread_metadata(
            graph_id=self._target_graph_id(request.target),
            workspace_dir=(
                request.target.workspace_dir if request.target is not None else None
            ),
            run_dir=request.target.run_dir if request.target is not None else None,
            metadata=request.metadata,
        )
        try:
            await self.thread_store.client.threads.update(
                request.thread_id, metadata=metadata
            )
        except Exception:
            logger.warning(
                "Thread metadata refresh failed for %s; continuing the run",
                request.thread_id,
                exc_info=True,
            )
        if isinstance(request.message, Command):
            if request.message.resume is not None:
                return await self._resume_interrupt(
                    request.thread_id,
                    request.message.resume,
                    config=config,
                    metadata=run_metadata,
                    request_id=request_id,
                    on_run_attempt=on_run_attempt,
                )
            raise RuntimeError(
                "LangGraph server gateway only supports Command(resume=...) messages."
            )

        run_input = await build_agent_stream_input(
            request.message,
            media=request.media,
            rubric=request.rubric,
            goal_request=request.goal_request,
        )
        on_run_attempt()
        result = await stream.run.start(
            input=run_input,
            config=config,
            metadata=dict(run_metadata) if run_metadata is not None else None,
        )
        run_id = result.get("run_id") if isinstance(result, Mapping) else None
        return str(run_id) if run_id else None

    async def _resume_interrupt(
        self,
        thread_id: str,
        response: object,
        *,
        config: dict[str, Any],
        metadata: Mapping[str, object] | None,
        request_id: str,
        on_run_attempt: Callable[[], None],
    ) -> str | None:
        """Resume the thread's pending interrupt with this run's config.

        The answer goes out as an ``input.respond`` command, which always
        resumes, also while another task of the paused run is still working.
        ``run.start`` would not: it takes its input as a resume only once the
        thread record lists the interrupt, which happens when the run stops,
        and otherwise starts the graph over. ``run.respond`` sends no config
        or metadata, so the command is posted directly. The pending ids are
        read from thread state, since ``stream.interrupts`` cannot tell them
        apart from answered ones, which a new stream replays too. The
        command's result has no run id, so the run is found by its request id.
        """
        client = self.thread_store.client
        # The prompt can reach the client before the state lists its
        # interrupt. A root one is listed once the turn has waited out the
        # replay's quiet window (``skip_replay``); a sub-graph's only once the
        # sub-graph's other tasks finish, so a keyed answer is sent unlisted.
        state = await client.threads.get_state(thread_id)
        pending = [
            str(interrupt_id)
            for interrupt in _state_interrupts(state)
            if (interrupt_id := interrupt.get("id"))
        ]
        interrupt_id, answer = _resume_target(thread_id, response, pending)
        # The server checks the id; for a subgraph's interrupt it accepts an
        # empty namespace, the only one thread state could give here.
        try:
            reply = await client.http.post(
                f"/threads/{thread_id}/commands",
                json={
                    "id": 1,
                    "method": "input.respond",
                    "params": {
                        "interrupt_id": interrupt_id,
                        "namespace": [],
                        "response": answer,
                        "config": config,
                        "metadata": (dict(metadata) if metadata is not None else None),
                    },
                },
            )
        except BaseException:
            # A lost reply can still have started the run.
            on_run_attempt()
            raise
        # A refused command comes back with a success status and starts no
        # run, so it is no run attempt: an unlisted id is the server's to check.
        if isinstance(reply, Mapping) and reply.get("type") == "error":
            raise RuntimeError(
                f"Thread {thread_id} refused the answer to interrupt "
                f"{interrupt_id}: {reply.get('error')}: {reply.get('message')}"
            )
        on_run_attempt()
        return await self._find_run_for_request(thread_id, request_id)

    async def _thread_is_busy(self, thread_id: str) -> bool:
        """Whether a run is in flight on the thread; ``False`` when unknown."""
        try:
            thread = await self.thread_store.client.threads.get(thread_id)
        except Exception:
            logger.warning(
                "Could not read the status of thread %s", thread_id, exc_info=True
            )
            return False
        return thread.get("status") == "busy"

    async def _find_run_for_request(
        self, thread_id: str, request_id: str
    ) -> str | None:
        """Find this request's run among all of the thread's runs.

        Not only active ones: a short resume can finish before the lookup.
        """
        offset = 0
        while True:
            # Whole rows carry every run's input (attached images included).
            page = await self.thread_store.client.runs.list(
                thread_id,
                limit=_RUN_LOOKUP_PAGE_SIZE,
                offset=offset,
                select=["run_id", "metadata"],
            )
            for run in page:
                if _run_has_request_id(run, request_id):
                    run_id = run.get("run_id")
                    return str(run_id) if run_id else None
            if len(page) < _RUN_LOOKUP_PAGE_SIZE:
                return None
            offset += _RUN_LOOKUP_PAGE_SIZE

    async def _list_active_runs(self, thread_id: str) -> list[Mapping[str, Any]] | None:
        """Return active runs on the thread, or ``None`` when listing fails."""
        runs: list[Mapping[str, Any]] = []
        try:
            for status in _ACTIVE_RUN_STATUSES:
                offset = 0
                while True:
                    page = await self.thread_store.client.runs.list(
                        thread_id,
                        limit=_RUN_LOOKUP_PAGE_SIZE,
                        offset=offset,
                        status=status,
                    )
                    runs.extend(page)
                    if len(page) < _RUN_LOOKUP_PAGE_SIZE:
                        break
                    offset += _RUN_LOOKUP_PAGE_SIZE
        except Exception:
            logger.warning(
                "Could not list active runs on thread %s",
                thread_id,
                exc_info=True,
            )
            return None
        return runs

    async def _find_active_run_for_request(
        self,
        thread_id: str,
        request_id: str,
    ) -> str | None:
        """Recover a run accepted before ``run.start`` lost its response."""
        for run in await self._list_active_runs(thread_id) or []:
            if _run_has_request_id(run, request_id):
                run_id = run.get("run_id")
                return str(run_id) if run_id else None
        return None

    async def _cancel_incomplete_run(
        self,
        thread_id: str,
        run_id: str | None,
        request_id: str | None,
    ) -> str | None:
        """Cancel an identified run, recovering a lost response conservatively."""
        resolved_run_id = run_id
        if resolved_run_id is None and request_id is not None:
            resolved_run_id = await self._find_active_run_for_request(
                thread_id, request_id
            )
        if resolved_run_id is None:
            return None
        try:
            await self.thread_store.client.runs.cancel_many(
                thread_id=thread_id,
                run_ids=[resolved_run_id],
            )
        except Exception:
            logger.warning(
                "Failed to cancel incomplete run %s on thread %s",
                resolved_run_id,
                thread_id,
                exc_info=True,
            )
            return None
        return resolved_run_id

    async def _has_other_active_runs(self, thread_id: str, run_id: str) -> bool:
        """Conservatively protect another request before repairing thread state."""
        runs = await self._list_active_runs(thread_id)
        return runs is None or any(
            str(run.get("run_id") or "") != run_id for run in runs
        )

    async def _repair_stuck_thread_state(self, thread_id: str) -> None:
        """Clear a non-empty ``next`` left by a failed run, preserving HITL pauses.

        When an exception occurs mid-run the LangGraph checkpoint can be left
        with a non-empty ``next`` tuple — the graph is stuck waiting to resume
        at a specific node. On the next invocation the server replays the
        broken step instead of starting a fresh turn. This mirrors the local
        path's ``_clear_interrupted_graph_state`` (``stream/events.py``): it
        fetches thread state, clears ``next`` via ``update_state(values=None,
        as_node="__end__")`` — but only when the state is *not* a genuine
        human-in-the-loop interrupt (those also leave ``next`` non-empty and
        must be preserved). Best-effort: failures are logged at DEBUG.
        """
        try:
            state = await self.thread_store.client.threads.get_state(thread_id)
        except NotFoundError:
            return
        except Exception:
            logger.debug(
                "Could not read thread state for repair on thread %s",
                thread_id,
                exc_info=True,
            )
            return

        next_nodes = state.get("next")
        if not next_nodes:
            return

        if _state_interrupts(state):
            logger.debug(
                "Leaving interrupted thread state intact for thread %s "
                "(pending human-in-the-loop interrupt)",
                thread_id,
            )
            return

        try:
            await self.thread_store.client.threads.update_state(
                thread_id,
                values=None,
                as_node="__end__",
            )
        except Exception:
            logger.debug(
                "Could not clear interrupted thread state for thread %s",
                thread_id,
                exc_info=True,
            )
            return
        logger.debug(
            "Cleared interrupted thread state for thread %s (was stuck at: %s)",
            thread_id,
            next_nodes,
        )

    def stream_events(self, request: RunRequest) -> AsyncIterator[GraphEvent]:
        return self._stream_events(request)

    async def get_state_snapshot(
        self,
        target: GraphTarget,
        thread_id: str,
    ) -> SimpleNamespace:
        state = await self.thread_store.client.threads.get_state(thread_id)
        return _thread_state_as_snapshot(state)

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
        values: GraphStateValues | None,
        *,
        as_node: str | None = None,
    ) -> None:
        if (
            as_node is None
            and isinstance(values, dict)
            and "_summarization_event" in values
        ):
            as_node = "model"
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

    async def get_run_status(
        self,
        target: GraphTarget,
        thread_id: str,
        run_id: str,
    ) -> str:
        run = await self.thread_store.client.runs.get(thread_id, run_id)
        return run["status"]

    async def get_process_status(
        self,
        target: GraphTarget,
        thread_id: str,
        process_id: str,
    ) -> str:
        # Background processes run in the langgraph dev server process; read their
        # status from the custom route on the server's http sub-app (there is no
        # SDK resource for OS processes, unlike runs).
        data = await self.thread_store.client.http.get(
            "/api/bg_process_status", params={"process_id": process_id}
        )
        return data["status"]

    async def _paused_interrupt_events(
        self,
        thread_id: str,
        processor: _V3EventProcessor,
        *,
        paused: bool,
        prompted: bool,
        run_id: str,
    ) -> list[GraphEvent]:
        """The prompts of a turn that saw its own pause, read from thread state.

        The ``input.requested`` envelope already gave a prompt when *prompted*;
        the state adds the rest: the turn stops at the root ``interrupted`` that
        follows the first interrupting task's envelopes, before another task's,
        and a pause seen only as a root ``lifecycle: interrupted`` has none. The
        processor drops a prompt it already emitted. ``stream.interrupts`` is no source: a new stream replays
        interrupts that were already answered, the SDK leaves their value empty
        (the server sends it as ``data.payload``), and ``stream.interrupted``
        comes from a separate SSE that can lag this one.
        """
        if not paused:
            return []
        events: list[GraphEvent] = []
        try:
            state = await self.thread_store.client.threads.get_state(thread_id)
        except NotFoundError:
            return events
        except Exception as exc:
            logger.warning(
                "Pause state read failed for thread %s; leaving the pause",
                thread_id,
                exc_info=True,
            )
            if prompted:
                # The turn already shows its prompt.
                return events
            # The run already paused. Propagate as _PauseStateReadError so the
            # outer handler can emit error without cancelling the pause.
            raise _PauseStateReadError(
                f"Could not read the pause state for thread {thread_id}"
            ) from exc
        # Once the envelope gave the prompt, a checkpoint an earlier run wrote
        # adds nothing: it can still list the interrupt just answered.
        written_by = (state.get("metadata") or {}).get("run_id")
        if prompted and written_by is not None and written_by != run_id:
            return events
        for interrupt in _state_interrupts(state):
            events.extend(
                await processor.process(
                    _input_requested_event_from_interrupt(interrupt)
                )
            )
        return events

    def _deliver_custom_middleware_events(
        self, raw_event: Mapping[str, Any]
    ) -> list[GraphEvent]:
        """Dispatch tagged middleware payloads from the ``custom`` channel.

        Wire shape (v3 protocol): ``{"method": "custom", "params": {"data":
        {MIDDLEWARE_EVENT_TAG: {"kind": ..., ...}}}}``, written server-side by
        ``StreamBroadcastSink``. Payloads without the tag — any other
        custom-channel traffic — are ignored, as are malformed ones: this is
        display narration, so a bad payload must degrade to silence rather
        than fail the run. Headless runs (``events is None``) drop everything.

        Fallback notices render directly through the sink's display callback,
        but tool-selection state is only *recorded* by the sink — its read
        side (``consume_tool_selection``, the dedup + render decision) is
        polled by the local stream suppressor, which does not exist on this
        path. So after dispatching writes, this method polls the read side
        and returns ``tool_selection`` events (same shape the local path
        yields) for the caller to emit — frontends render both identically.
        The poll runs only on custom events: pending is set only by the
        dispatched ``tool_selection`` write above, and consume-once in the
        sink guarantees it is drained by the poll on this very event.
        """
        if self.events is None:
            return []
        if raw_event.get("method") != "custom":
            return []
        params = _as_raw_map(raw_event.get("params"))
        payload = _as_raw_map(params.get("data")) if params is not None else None
        if payload is not None:
            payload = _as_raw_map(payload.get(MIDDLEWARE_EVENT_TAG))
        if payload is not None:
            # Kind dispatch lives on the event dataclasses (middleware.events):
            # unknown kinds are None (silence), malformed payloads raise and
            # degrade to the DEBUG log below rather than fail the run.
            try:
                event = MiddlewareEvent.from_wire(payload)
                if event is not None:
                    event.dispatch(self.events)
            except Exception:
                logger.debug(
                    "malformed middleware custom event %r", dict(payload), exc_info=True
                )
        return self._poll_tool_selection()

    def _poll_tool_selection(self) -> list[GraphEvent]:
        """Consume any pending selection and shape it as a stream event.

        Consume-once/dedup semantics live in the sink; polling returns at
        most one event per pending selection. Outside the stream loop, the
        ``GraphGateway`` protocol's sink is a ``SessionEvents`` (write+read
        sides); guard with getattr for sinks missing the read side.
        """
        consume = getattr(self.events, "consume_tool_selection", None)
        if consume is None:
            return []
        try:
            had_pending, render = consume()
        except Exception:
            logger.debug("tool-selection poll failed", exc_info=True)
            return []
        if had_pending and render is not None:
            return [StreamEventEmitter.tool_selection(render).data]
        return []

    async def _stream_events(self, request: RunRequest) -> AsyncIterator[GraphEvent]:
        emitter = StreamEventEmitter()
        state_values: GraphStateValues = {}
        existing_summarization_event: Mapping[str, object] | None = None
        process_value_messages = True
        thread_ready = False
        try:
            # Register the thread before the pre-run state read: a thread
            # whose checkpoints exist (shared checkpointer) but which is
            # missing from the server registry would fail the read with
            # NotFoundError, leaving the suppression baselines empty and
            # replaying prior turns on the next values snapshot (#490).
            await self._ensure_thread(request)
            thread_ready = True
            state_values = await self._get_state_values(request.thread_id)
            existing_summarization_event = _find_summarization_event_payload(
                state_values
            )
        except Exception:
            # A 404 after registration is unexpected (a missing registry
            # entry is handled by ``_ensure_thread``; a registered thread
            # returns an empty snapshot, not NotFoundError). Treat it like
            # any other pre-run read failure: disable value-message
            # processing instead of replaying with an empty baseline.
            process_value_messages = False
            logger.warning(
                "Pre-run state fetch failed for thread %s; "
                "value-message processing disabled for this run",
                request.thread_id,
                exc_info=True,
            )

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
            assistant_id=self._target_assistant_id(request.target),
        )

        request_id = uuid.uuid4().hex
        run_metadata: dict[str, object] = {
            **(request.metadata or {}),
            _RUN_REQUEST_ID_METADATA_KEY: request_id,
        }
        run_attempted = False
        run_id: str | None = None
        run_completed = False
        run_failed = False
        emitted_interrupt = False
        run_cleaned_up = False

        def mark_run_attempted() -> None:
            nonlocal run_attempted
            run_attempted = True

        try:
            # Every turn reads ``stream.events`` and stops at its own run's end
            # or pause. A fresh SSE replays the thread's history, so
            # ``subscribe`` would end at an earlier run's root ``completed``
            # (#558); and on a resume the replayed pause has already set the
            # SDK's ``interrupted`` flag, so ``subscribe`` would never end at
            # the resumed run's next pause (#738).
            paused = False
            # Past the replay cap the queue holds a busy run's live envelopes.
            behind_busy_run = False
            async with stream, _ThreadEvents(stream.events) as events:
                # A thread without state has no earlier runs to replay.
                # Skip also when the state read failed: an empty baseline
                # would leave us draining history and risk losing live
                # envelopes for the run we are about to start.
                if state_values or not process_value_messages:
                    try:
                        await events.skip_replay()
                    except _ReplayCapError:
                        # Another task of the paused run can keep the SSE
                        # from going quiet. The answer queues behind that
                        # run, whose live envelopes mean this connection is
                        # normally past the replay when the answer's run
                        # starts.
                        if not (
                            isinstance(request.message, Command)
                            and await self._thread_is_busy(request.thread_id)
                        ):
                            raise
                        behind_busy_run = True
                run_id = await self._start_or_resume(
                    stream,
                    request,
                    thread_ready=thread_ready,
                    on_run_attempt=mark_run_attempted,
                    request_id=request_id,
                    run_metadata=run_metadata,
                )
                if run_id is None:
                    # ``events`` stays open after the run; without its id
                    # no stop condition below can fire.
                    raise RuntimeError(
                        f"Could not identify the run for thread {request.thread_id}"
                    )
                # Envelopes without a run id (model tokens) belong to the run
                # of the last tagged one: behind a busy run, wait for this
                # run's own.
                accepting_run = not behind_busy_run
                # See #709
                async for event in events:
                    raw_event = _as_raw_map(event)
                    if raw_event is None:
                        continue
                    event_run_id = _synth_event_run_id(raw_event)
                    if event_run_id is not None:
                        accepting_run = event_run_id == run_id
                    if not accepting_run:
                        continue
                    for selection_event in self._deliver_custom_middleware_events(
                        raw_event
                    ):
                        yield selection_event
                    event_map: dict[str, Any] = dict(raw_event)
                    for subagent_event in tracker.process(event_map):
                        yield subagent_event
                    for normalized in await processor.process(event_map):
                        emitted_interrupt = emitted_interrupt or _is_interrupt_event(
                            normalized
                        )
                        yield normalized
                    # Stop at this run's root completed/failed/interrupted.
                    # The server sends the root interrupted right after the
                    # pause's input.requested, also while another task of
                    # the run still works.
                    phase = _root_lifecycle_phase(event_map)
                    if phase == "failed":
                        # End the turn through the error path, like a local failure.
                        run_failed = True
                        raise RuntimeError(_run_failure_message(event_map))
                    if phase == "completed":
                        break
                    if phase == "interrupted":
                        paused = True
                        break
                else:
                    raise RuntimeError(
                        "Thread event stream ended before the run finished"
                    )
                run_completed = True
                for event in await self._paused_interrupt_events(
                    request.thread_id,
                    processor,
                    paused=paused,
                    prompted=emitted_interrupt,
                    run_id=run_id,
                ):
                    yield event
            for event in tracker.finish():
                yield event
        except _PauseStateReadError as exc:
            # Pause already accepted; emit the gateway error shape and leave
            # the run so a later turn can still show the prompt.
            yield emitter.error(str(exc)).data
            for event in tracker.finish():
                yield event
            raise
        except Exception as exc:
            if emitted_interrupt:
                # The turn already shows its prompt; cancel or repair could
                # stop a task still working or clear the pause it answers.
                yield emitter.error(str(exc)).data
                for event in tracker.finish():
                    yield event
                raise
            # Cancel this request's run, then repair, both before the first yield:
            # a consumer that stops iterating after the error event (aclose /
            # abandon) triggers GeneratorExit at the yield, so nothing after it
            # runs and the thread keeps its non-empty ``next`` — the failed step
            # would replay on the next request. A run the server reported as
            # failed has already ended, so only the repair is left.
            cancelled_run_id = None
            if not run_failed:
                cancelled_run_id = await self._cancel_incomplete_run(
                    request.thread_id,
                    run_id,
                    request_id if run_attempted else None,
                )
            run_cleaned_up = True
            can_repair = (
                cancelled_run_id is not None
                and not await self._has_other_active_runs(
                    request.thread_id, cancelled_run_id
                )
            )
            if cancelled_run_id is None and run_attempted:
                # A synchronous start rejection creates no run to
                # cancel. Preserve the existing repair in that safe case, but
                # never clear state while any unidentified active run remains.
                can_repair = await self._list_active_runs(request.thread_id) == []
            if can_repair:
                await self._repair_stuck_thread_state(request.thread_id)
            yield emitter.error(str(exc)).data
            for event in tracker.finish():
                yield event
            raise
        finally:
            if (
                run_attempted
                and not run_completed
                and not emitted_interrupt
                and not run_cleaned_up
            ):
                await self._cancel_incomplete_run(
                    request.thread_id,
                    run_id,
                    request_id,
                )
        yield emitter.done(processor.full_response).data
