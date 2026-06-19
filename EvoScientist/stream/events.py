"""Stream event generator and v3 protocol helpers.

Async generator that streams UI events from a DeepAgents/LangGraph v3 run.
"""

import asyncio
import base64
import inspect
import mimetypes
import os
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from langgraph.graph import END
from langgraph.types import Command, Interrupt

from ..memory.worker_activity import clear_memory_worker_saved_counts
from .emitter import StreamEventEmitter
from .summarization import (
    _extract_summary_message_text,
    _find_summarization_event_payload,
    _summarization_event_signature,
)
from .tool_results import _extract_command_tool_content, _extract_tool_content
from .tool_selection import _ToolSelectionSuppressor
from .utils import DisplayLimits, is_success
from .v3_payloads import (
    RawMap,
    _as_raw_map,
    _event_data,
    _event_namespace,
    _reasoning_from_content,
    _split_message_event_data,
    _strip_legacy_thinking_tags,
    _text_from_content,
    _usage_counts,
)

UserMessageContent: TypeAlias = str | list[dict[str, object]]
GraphRunInput: TypeAlias = str | Command
LangGraphStreamInput: TypeAlias = dict[str, list[dict[str, object]]] | Command
_ValueMessageKey: TypeAlias = tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _AssistantValueMessage:
    key: _ValueMessageKey
    content: object


def _is_interrupt_error_message(message: object) -> bool:
    if not isinstance(message, str):
        return False
    stripped = message.strip()
    return stripped.startswith("(Interrupt(value=")


def _snapshot_has_pending_interrupt(snapshot: Any) -> bool:
    """True if the graph is parked at a genuine human-in-the-loop interrupt.

    A non-empty ``next`` alone can't distinguish "crashed mid-node" from
    "legitimately waiting at ``interrupt()`` for a ``Command(resume=...)``".
    The presence of pending interrupts — surfaced both on the snapshot and on
    its tasks — is what tells them apart.
    """
    if getattr(snapshot, "interrupts", None):
        return True
    for task in getattr(snapshot, "tasks", None) or ():
        if getattr(task, "interrupts", None):
            return True
    return False


async def _clear_interrupted_graph_state(
    agent: Any,
    config: dict[str, Any],
) -> None:
    """Force the graph back to a clean (non-interrupted) state after an error.

    When an exception occurs mid-run the LangGraph checkpoint can be left with a
    non-empty ``next`` tuple — the graph is stuck waiting to resume at a specific
    node. On the next invocation with a fresh user message LangGraph tries to
    **resume** that interrupted step rather than starting a new turn: it ignores
    the new human message and replays the broken step, which typically produces
    no output and leaves the messages channel unchanged. From the user's side the
    conversation looks like it lost all history because the agent stops responding.

    The fix: ``aupdate_state(config, None, as_node=END)`` clears all pending tasks
    and writes a checkpoint whose ``next`` is the empty tuple, without touching
    any channel values (message history is preserved).

    Critically, this only runs when the stuck state is *not* a legitimate
    human-in-the-loop interrupt. The agent pauses via ``interrupt()`` /
    ``Command(resume=...)`` for ask-user flows, which also leaves ``next``
    non-empty; clearing those would silently discard a pending question the user
    still needs to answer. ``_snapshot_has_pending_interrupt`` distinguishes the
    two.

    Best-effort: any failure is logged at DEBUG and swallowed so it never shadows
    the original exception that triggered recovery.
    """
    import logging

    _log = logging.getLogger(__name__)
    try:
        snapshot = await agent.aget_state(config)
        # Only act when the graph is genuinely stuck (non-empty next tuple)...
        if not snapshot or not getattr(snapshot, "next", None):
            return
        # ...and not parked at a real human-in-the-loop interrupt.
        if _snapshot_has_pending_interrupt(snapshot):
            _log.debug(
                "Leaving interrupted graph state intact for thread %s: "
                "pending human-in-the-loop interrupt (next=%s)",
                config.get("configurable", {}).get("thread_id", "?"),
                snapshot.next,
            )
            return

        stuck_at = snapshot.next
        await agent.aupdate_state(config, None, as_node=END)
        _log.debug(
            "Cleared interrupted graph state for thread %s (was stuck at: %s)",
            config.get("configurable", {}).get("thread_id", "?"),
            stuck_at,
        )
    except Exception as exc:  # pragma: no cover — best-effort recovery
        _log.debug(
            "Could not clear interrupted graph state: %s",
            exc,
            exc_info=True,
        )


@dataclass(frozen=True)
class _SubagentInfo:
    path: tuple[str, ...]
    name: str
    description: str = ""

    @property
    def instance_id(self) -> str:
        return ":".join(self.path)


class _SubagentRegistry:
    """Track v3 DeepAgents subagent handles by namespace path."""

    def __init__(self) -> None:
        self._by_path: dict[tuple[str, ...], _SubagentInfo] = {}
        self._changed = asyncio.Event()
        self._closed = False

    def register(self, path: tuple[str, ...], name: str, description: str = "") -> None:
        if not path:
            return
        self._by_path[path] = _SubagentInfo(path, name, description)
        self._changed.set()

    def close(self) -> None:
        self._closed = True
        self._changed.set()

    def resolve(self, namespace: tuple[str, ...]) -> _SubagentInfo | None:
        for depth in range(len(namespace), 0, -1):
            info = self._by_path.get(namespace[:depth])
            if info is not None:
                return info
        return None

    async def wait_resolve(self, namespace: tuple[str, ...]) -> _SubagentInfo | None:
        if not namespace:
            return None
        while not self._closed:
            if info := self.resolve(namespace):
                return info
            self._changed.clear()
            if info := self.resolve(namespace):
                return info
            if self._closed:
                break
            await self._changed.wait()
        return self.resolve(namespace)


class _V3EventProcessor:
    """Translate v3 protocol channel events into EvoScientist UI events."""

    def __init__(
        self,
        emitter: StreamEventEmitter,
        subagents: _SubagentRegistry,
        existing_summarization_event: Mapping[str, object] | None,
        existing_messages: object = None,
        process_value_messages: bool = False,
    ) -> None:
        self.emitter = emitter
        self.subagents = subagents
        self._suppressed_summarization_signature = _summarization_event_signature(
            existing_summarization_event
        )
        self._seen_value_message_keys = self._message_keys(existing_messages)
        self._process_value_message_snapshots = process_value_messages
        self.full_response = ""
        self._summarization_in_progress = False
        self._tool_inputs: dict[
            tuple[tuple[str, ...], str], tuple[str, dict[str, Any]]
        ] = {}
        self._emitted_tool_calls: set[tuple[tuple[str, ...], str]] = set()
        self._emitted_interrupts: set[str] = set()
        self._selector = _ToolSelectionSuppressor(emitter)

    @staticmethod
    def _tool_scope(
        namespace: tuple[str, ...], subagent: _SubagentInfo | None
    ) -> tuple[str, ...]:
        return subagent.path if subagent is not None else namespace

    async def process(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        method = event.get("method")
        namespace = _event_namespace(event)
        subagent = None
        if namespace and method in ("messages", "tools"):
            subagent = await self.subagents.wait_resolve(namespace)
            if subagent is None:
                return []

        if method == "messages":
            events = self._process_message_event(
                _event_data(event), subagent, namespace
            )
            if not namespace and any(item.get("type") == "text" for item in events):
                self._process_value_message_snapshots = False
            return events
        if method == "tools":
            return self._process_tool_event(namespace, _event_data(event), subagent)
        if method == "updates":
            return self._process_update_event(_event_data(event))
        if method == "values":
            events: list[dict[str, Any]] = []
            params = event.get("params") or {}
            interrupts = params.get("interrupts") or ()
            if interrupts:
                events.extend(self._process_update_event({"__interrupt__": interrupts}))
            if self._process_value_message_snapshots and not namespace:
                events.extend(self._process_value_messages(_event_data(event)))
            return events
        if method == "input.requested":
            return self._process_input_requested(event.get("params"))
        return []

    @classmethod
    def _message_keys(cls, messages: object) -> set[_ValueMessageKey]:
        if not isinstance(messages, list):
            return set()
        keys: set[_ValueMessageKey] = set()
        for message in messages:
            if parsed := cls._assistant_value_message(message):
                keys.add(parsed.key)
        return keys

    @staticmethod
    def _assistant_value_message(message: object) -> _AssistantValueMessage | None:
        message_map = _as_raw_map(message)
        if message_map is not None:
            raw_id = message_map.get("id")
            raw_role = message_map.get("type") or message_map.get("role")
            content = message_map.get("content")
        elif isinstance(message, BaseMessage):
            raw_id = message.id
            raw_role = message.type
            content = message.content
        else:
            return None

        if raw_role not in ("ai", "assistant"):
            return None
        if raw_id:
            key = ("id", str(raw_id))
        else:
            key = ("body", str(raw_role), repr(content))
        return _AssistantValueMessage(key=key, content=content)

    def _process_value_messages(self, data: object) -> list[dict[str, Any]]:
        data_map = _as_raw_map(data)
        if data_map is None:
            return []
        messages = data_map.get("messages")
        if not isinstance(messages, list):
            return []

        events: list[dict[str, Any]] = []
        for message in messages:
            parsed = self._assistant_value_message(message)
            if parsed is None:
                continue
            if parsed.key in self._seen_value_message_keys:
                continue
            text = _text_from_content(parsed.content)
            if text:
                self._seen_value_message_keys.add(parsed.key)
                events.extend(self._emit_text(text, subagent=None))
        return events

    def _process_message_event(
        self,
        data: object,
        subagent: _SubagentInfo | None,
        namespace: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        payload, metadata = _split_message_event_data(data)
        payload_map = _as_raw_map(payload)
        if metadata.get("lc_source") == "summarization":
            if payload_map is not None:
                text = self._text_from_message_payload(payload_map)
            elif isinstance(payload, BaseMessage):
                text = self._text_from_message_payload(payload)
            else:
                text = ""
            return self._emit_summarization_text(text)

        if payload_map is not None and "event" in payload_map:
            return self._process_protocol_message_payload(
                payload_map, subagent, namespace
            )

        if isinstance(payload, AIMessage | AIMessageChunk):
            return self._process_whole_message(payload, subagent, namespace)
        return []

    def _process_protocol_message_payload(
        self,
        payload: RawMap,
        subagent: _SubagentInfo | None,
        namespace: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        event_type = payload.get("event")
        if event_type == "content-block-delta":
            delta = _as_raw_map(payload.get("delta"))
            if delta is None:
                return []
            delta_type = delta.get("type")
            if delta_type == "text-delta":
                text = delta.get("text")
                return self._emit_text(text if isinstance(text, str) else "", subagent)
            if delta_type == "reasoning-delta":
                reasoning = delta.get("reasoning")
                return self._emit_thinking(
                    reasoning if isinstance(reasoning, str) else "", subagent
                )
            if delta_type == "block-delta":
                fields = _as_raw_map(delta.get("fields"))
                if fields is not None:
                    return self._process_message_tool_block(fields, subagent, namespace)
            return []

        if event_type == "content-block-finish":
            content = _as_raw_map(payload.get("content"))
            if content is not None:
                return self._process_message_tool_block(content, subagent, namespace)
            return []

        if event_type == "message-finish":
            events: list[dict[str, Any]] = []
            pending = self._selector.flush_pending_text()
            if pending and not pending.isspace():
                if subagent is not None:
                    events.append(
                        self.emitter.subagent_text(
                            subagent.name,
                            pending,
                            instance_id=subagent.instance_id,
                        ).data
                    )
                else:
                    self.full_response += pending
                    events.append(self.emitter.text(pending).data)
            if subagent is None:
                usage = _as_raw_map(payload.get("usage"))
                inp, out = _usage_counts(usage) if usage is not None else (0, 0)
                if inp or out:
                    events.append(self.emitter.usage_stats(inp, out).data)
            return events
        return []

    def _process_message_tool_block(
        self,
        block: RawMap,
        subagent: _SubagentInfo | None,
        namespace: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        block_type = block.get("type")
        if block_type not in (
            "tool_call",
            "tool_call_chunk",
            "server_tool_call",
            "server_tool_call_chunk",
        ):
            return []
        name = block.get("name") or ""
        if self._selector.observe_tool_block(str(name)):
            return []
        events = self._selector.flush_selection()
        tool_call = self._tool_call_from_message_block(block)
        if tool_call is None:
            return events
        tool_name, args, tool_call_id = tool_call
        events.extend(
            self._emit_tool_call_once(
                namespace=namespace,
                subagent=subagent,
                name=tool_name,
                args=args,
                tool_call_id=tool_call_id,
            )
        )
        return events

    @staticmethod
    def _tool_call_from_message_block(
        block: RawMap,
    ) -> tuple[str, dict[str, Any], str] | None:
        tool_call_id = str(block.get("id") or block.get("tool_call_id") or "")
        name = str(block.get("name") or block.get("tool_name") or "")
        if not tool_call_id or not name:
            return None
        raw_args = block.get("args") if "args" in block else block.get("input")
        args = _as_raw_map(raw_args)
        if args is None:
            return None
        return name, dict(args), tool_call_id

    def _emit_tool_call_once(
        self,
        *,
        namespace: tuple[str, ...],
        subagent: _SubagentInfo | None,
        name: str,
        args: dict[str, Any],
        tool_call_id: str,
    ) -> list[dict[str, Any]]:
        key = (self._tool_scope(namespace, subagent), tool_call_id)
        if key in self._emitted_tool_calls:
            return []
        self._emitted_tool_calls.add(key)
        if subagent is not None:
            return [
                self.emitter.subagent_tool_call(
                    subagent.name,
                    name,
                    args,
                    tool_call_id,
                    instance_id=subagent.instance_id,
                ).data
            ]
        return [self.emitter.tool_call(name, args, tool_call_id).data]

    def _process_whole_message(
        self,
        msg: AIMessage | AIMessageChunk,
        subagent: _SubagentInfo | None,
        namespace: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        additional = msg.additional_kwargs
        reasoning = additional.get("reasoning_content")
        emitted_reasoning = False
        if isinstance(reasoning, str):
            events.extend(self._emit_thinking(reasoning, subagent))
            emitted_reasoning = bool(reasoning)

        content = msg.content
        if not emitted_reasoning:
            events.extend(
                self._emit_thinking(_reasoning_from_content(content), subagent)
            )
        events.extend(self._emit_text(_text_from_content(content), subagent))

        for raw_tool_call_block in msg.tool_calls:
            tool_call_block = _as_raw_map(raw_tool_call_block)
            if tool_call_block is None:
                continue
            tool_call = self._tool_call_from_message_block(tool_call_block)
            if tool_call is None:
                continue
            tool_name, args, tool_call_id = tool_call
            events.extend(
                self._emit_tool_call_once(
                    namespace=namespace,
                    subagent=subagent,
                    name=tool_name,
                    args=args,
                    tool_call_id=tool_call_id,
                )
            )

        if subagent is None:
            inp, out = _usage_counts(msg.usage_metadata)
            if inp or out:
                events.append(self.emitter.usage_stats(inp, out).data)
        return events

    def _process_tool_event(
        self,
        namespace: tuple[str, ...],
        data: object,
        subagent: _SubagentInfo | None,
    ) -> list[dict[str, Any]]:
        data_map = _as_raw_map(data)
        if data_map is None:
            return []
        event_type = data_map.get("event")
        tool_call_id = str(data_map.get("tool_call_id") or "")
        if event_type == "tool-started":
            events = self._selector.flush_selection()
            if not tool_call_id:
                return events
            name = str(data_map.get("tool_name") or "")
            if not name:
                return events
            input_args = _as_raw_map(data_map.get("input"))
            args = dict(input_args) if input_args is not None else {}
            self._tool_inputs[(self._tool_scope(namespace, subagent), tool_call_id)] = (
                name,
                args,
            )
            events.extend(
                self._emit_tool_call_once(
                    namespace=namespace,
                    subagent=subagent,
                    name=name,
                    args=args,
                    tool_call_id=tool_call_id,
                )
            )
            return events

        if event_type in ("tool-finished", "tool-error"):
            events = self._selector.flush_selection()
            if not tool_call_id:
                return events
            name, _args = self._tool_inputs.pop(
                (self._tool_scope(namespace, subagent), tool_call_id),
                (str(data_map.get("tool_name") or "unknown"), {}),
            )
            message = data_map.get("message")
            # LangGraph v3 reports interrupt() as a tool-error before the
            # structured __interrupt__ update; the real result arrives on resume.
            if event_type == "tool-error" and _is_interrupt_error_message(message):
                return events
            if event_type == "tool-error":
                content = str(message or "")
                success = False
            else:
                output = data_map.get("output")
                if isinstance(output, ToolMessage):
                    name = output.name or name
                    raw_content, _ = _extract_tool_content(output)
                elif isinstance(output, Command):
                    command_content = _extract_command_tool_content(
                        output, tool_call_id
                    )
                    raw_content = (
                        command_content if command_content is not None else str(output)
                    )
                else:
                    raw_content = "" if output is None else str(output)
                content = raw_content[: DisplayLimits.TOOL_RESULT_MAX]
                if len(raw_content) > DisplayLimits.TOOL_RESULT_MAX:
                    content += "\n... (truncated)"
                success = is_success(content)

            if subagent is not None:
                events.append(
                    self.emitter.subagent_tool_result(
                        subagent.name,
                        name,
                        content,
                        success,
                        tool_call_id,
                        instance_id=subagent.instance_id,
                    ).data
                )
                return events
            events.append(
                self.emitter.tool_result(
                    name, content, success, tool_call_id=tool_call_id
                ).data
            )
            return events

        return []

    def _process_update_event(self, data: object) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        data_map = _as_raw_map(data)
        if data_map is not None and "__interrupt__" in data_map:
            events.extend(self._process_interrupts(data_map["__interrupt__"]))

        summarization_event = _find_summarization_event_payload(data)
        if summarization_event and not self._summarization_in_progress:
            signature = _summarization_event_signature(summarization_event)
            if (
                signature is not None
                and signature == self._suppressed_summarization_signature
            ):
                return events
            summary_message = summarization_event.get("summary_message")
            summary_text = _extract_summary_message_text(
                summary_message if isinstance(summary_message, BaseMessage) else None
            )
            events.extend(self._emit_summarization_text(summary_text))
        return events

    def _process_interrupts(self, interrupts: object) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if not isinstance(interrupts, list | tuple):
            return events

        for interrupt_obj in interrupts:
            if not isinstance(interrupt_obj, Interrupt):
                continue

            interrupt_value = interrupt_obj.value
            interrupt_id = interrupt_obj.id or "default"
            events.extend(self._process_interrupt_value(interrupt_id, interrupt_value))
        return events

    def _process_input_requested(self, params: object) -> list[dict[str, Any]]:
        params_map = _as_raw_map(params)
        if params_map is None:
            return []
        data = _as_raw_map(params_map.get("data"))
        if data is None:
            return []
        interrupt_id = str(data.get("interrupt_id") or "default")
        return self._process_interrupt_value(interrupt_id, data.get("value"))

    def _process_interrupt_value(
        self,
        interrupt_id: str,
        interrupt_value: object,
    ) -> list[dict[str, Any]]:
        interrupt_map = _as_raw_map(interrupt_value)
        if interrupt_map is None:
            return []

        iv_type = interrupt_map.get("type")
        if iv_type == "ask_user":
            raw_questions = interrupt_map.get("questions")
            questions = raw_questions if isinstance(raw_questions, list) else []
            tc_id = str(interrupt_map.get("tool_call_id", ""))
            return self._dedupe_interrupt_event(
                self.emitter.ask_user_interrupt(
                    interrupt_id,
                    questions,
                    tc_id,
                ).data
            )

        raw_action_reqs = interrupt_map.get("action_requests")
        action_reqs = raw_action_reqs if isinstance(raw_action_reqs, list) else []
        raw_review_cfgs = interrupt_map.get("review_configs")
        review_cfgs = raw_review_cfgs if isinstance(raw_review_cfgs, list) else None
        if action_reqs:
            return self._dedupe_interrupt_event(
                self.emitter.interrupt(
                    interrupt_id,
                    action_reqs,
                    review_cfgs,
                ).data
            )
        return []

    def _dedupe_interrupt_event(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        signature = repr(event)
        if signature in self._emitted_interrupts:
            return []
        self._emitted_interrupts.add(signature)
        return [event]

    def _emit_text(
        self, text: str, subagent: _SubagentInfo | None
    ) -> list[dict[str, Any]]:
        if not text:
            return []
        cleaned = _strip_legacy_thinking_tags(text)
        if not cleaned or cleaned.isspace():
            return []
        suppressed, events, emit_text = self._selector.process_text(cleaned)
        if suppressed:
            return events
        if not emit_text or emit_text.isspace():
            return events
        if subagent is not None:
            events.append(
                self.emitter.subagent_text(
                    subagent.name, emit_text, instance_id=subagent.instance_id
                ).data
            )
            return events
        self.full_response += emit_text
        events.append(self.emitter.text(emit_text).data)
        return events

    def _emit_thinking(
        self, text: str, subagent: _SubagentInfo | None
    ) -> list[dict[str, Any]]:
        if not text or subagent is not None:
            return []
        return [self.emitter.thinking(text).data]

    def _emit_summarization_text(self, text: str) -> list[dict[str, Any]]:
        if not text:
            return []
        events: list[dict[str, Any]] = []
        if not self._summarization_in_progress:
            events.append(self.emitter.summarization_start().data)
        self._summarization_in_progress = True
        events.append(self.emitter.summarization(text).data)
        return events

    def _text_from_message_payload(self, payload: RawMap | BaseMessage) -> str:
        if not isinstance(payload, BaseMessage):
            if payload.get("event") == "content-block-delta":
                delta = _as_raw_map(payload.get("delta"))
                if delta is not None and delta.get("type") == "text-delta":
                    text = delta.get("text")
                    return text if isinstance(text, str) else ""
            return ""
        return _text_from_content(payload.content)


async def build_agent_stream_input(
    message: GraphRunInput,
    *,
    media: list[str] | None = None,
) -> LangGraphStreamInput:
    """Build the LangGraph run input shared by local and server gateways."""
    if not isinstance(message, str):
        return message

    user_content: UserMessageContent = message
    if media:
        image_exts = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"})
        max_inline_size = 5 * 1024 * 1024
        content_blocks: list[dict[str, object]] = []
        if message:
            content_blocks.append({"type": "text", "text": message})

        def _read_file_b64(path: str) -> str:
            with open(path, "rb") as fh:
                return base64.b64encode(fh.read()).decode("ascii")

        file_refs: list[str] = []
        for path in media:
            ext = os.path.splitext(path)[1].lower()
            is_image = ext in image_exts and await asyncio.to_thread(
                os.path.isfile, path
            )
            if is_image:
                fsize = await asyncio.to_thread(os.path.getsize, path)
                if fsize <= max_inline_size:
                    mime = mimetypes.guess_type(path)[0] or "image/png"
                    b64 = await asyncio.to_thread(_read_file_b64, path)
                    content_blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{b64}",
                            },
                        }
                    )
                else:
                    file_refs.append(path)
            else:
                file_refs.append(path)
        if file_refs:
            ref_text = "\n".join(
                f"[attached file: {os.path.basename(p)}] path: {p}" for p in file_refs
            )
            content_blocks.append({"type": "text", "text": ref_text})
        if content_blocks:
            user_content = content_blocks
    return {"messages": [{"role": "user", "content": user_content}]}


async def stream_agent_events(
    agent: Any,
    message: GraphRunInput,
    thread_id: str,
    metadata: dict[str, Any] | None = None,
    media: list[str] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Stream events from a DeepAgents/LangGraph v3 run.

    The ingestion side uses ``astream_events(..., version="v3")``:
    raw protocol events preserve arrival order for messages/tools/updates,
    while DeepAgents' native ``stream.subagents`` projection provides the
    user-facing subagent identity that raw graph namespaces intentionally hide.

    Args:
        agent: Compiled state graph from create_deep_agent()
        message: User message
        thread_id: Thread ID for conversation persistence
        metadata: Optional metadata dict merged into the LangGraph config
            (e.g. agent_name, updated_at for checkpoint persistence).
        media: Optional list of local file paths for attachments.

    Yields:
        Event dicts: thinking, text, tool_call, tool_result,
                     subagent_start, subagent_tool_call, subagent_tool_result, subagent_end,
                     done, error
    """
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    if metadata:
        config["metadata"] = metadata
    emitter = StreamEventEmitter()
    existing_summarization_event: Mapping[str, object] | None = None
    try:
        snapshot = await agent.aget_state(config)
        existing_summarization_event = _find_summarization_event_payload(
            getattr(snapshot, "values", None)
        )
    except Exception:
        pass

    clear_memory_worker_saved_counts()
    astream_input = await build_agent_stream_input(message, media=media)

    stream: Any | None = None
    producers: list[asyncio.Task[Any]] = []
    _run_raised: bool = False
    try:
        from langgraph.stream.transformers import UpdatesTransformer

        try:
            stream_result = agent.astream_events(
                astream_input,
                config=config,
                version="v3",
                transformers=[UpdatesTransformer],
            )
        except AttributeError as exc:
            raise RuntimeError(
                "This agent does not expose astream_events(); EvoScientist requires "
                "DeepAgents/LangGraph stream v3."
            ) from exc

        subagents = _SubagentRegistry()
        processor = _V3EventProcessor(
            emitter,
            subagents,
            existing_summarization_event,
        )
        queue: asyncio.Queue[Any] = asyncio.Queue()
        producer_done = object()

        stream = (
            await stream_result if inspect.isawaitable(stream_result) else stream_result
        )

        async def _put_events(events: list[dict[str, Any]]) -> None:
            for event in events:
                await queue.put(event)

        async def _consume_protocol_events() -> None:
            async for event in stream:
                await _put_events(await processor.process(event))

        async def _await_subagent_done(
            subagent: Any, name: str, instance_id: str
        ) -> None:
            try:
                result = subagent.output()
                if inspect.isawaitable(result):
                    await result
            finally:
                await queue.put(
                    emitter.subagent_end(name, instance_id=instance_id).data
                )

        subagent_iter: AsyncIterator[Any] = aiter(stream.subagents)

        async def _consume_subagents() -> None:
            completion_tasks: list[asyncio.Task[Any]] = []
            try:
                async for subagent in subagent_iter:
                    path = tuple(str(part) for part in subagent.path)
                    name = subagent.name
                    if not name:
                        continue
                    name = str(name)
                    description = ""
                    cause = subagent.cause
                    trigger_call_id = ""
                    if cause and cause["type"] == "toolCall":
                        trigger_call_id = cause["tool_call_id"]
                    instance_id = ":".join(path)
                    await queue.put(
                        emitter.subagent_start(
                            name,
                            description,
                            instance_id=instance_id,
                            tool_call_id=trigger_call_id,
                        ).data
                    )
                    subagents.register(path, name, description)
                    completion_tasks.append(
                        asyncio.create_task(
                            _await_subagent_done(subagent, name, instance_id)
                        )
                    )
                if completion_tasks:
                    await asyncio.gather(*completion_tasks)
            finally:
                subagents.close()

        async def _run_producer(coro: Any) -> None:
            try:
                await coro
            except BaseException as exc:
                await queue.put(exc)
            finally:
                await queue.put(producer_done)

        producers = [
            asyncio.create_task(_run_producer(_consume_subagents())),
            asyncio.create_task(_run_producer(_consume_protocol_events())),
        ]
        pending_producers = len(producers)

        while pending_producers:
            item = await queue.get()
            if item is producer_done:
                pending_producers -= 1
                continue
            if isinstance(item, BaseException):
                for task in producers:
                    task.cancel()
                await asyncio.gather(*producers, return_exceptions=True)
                raise item
            yield item
    except Exception as e:
        _run_raised = True
        yield emitter.error(str(e)).data
        raise
    finally:
        if stream is not None:
            try:
                result = stream.abort()
                if inspect.isawaitable(result):
                    await result
            except Exception:
                pass
        for task in producers:
            if not task.done():
                task.cancel()
        if producers:
            await asyncio.gather(*producers, return_exceptions=True)
        # When the run ended with an exception the LangGraph checkpoint may be
        # left interrupted (``next`` non-empty). Clear it — unless it's a real
        # human-in-the-loop pause — so the next user message starts a fresh turn
        # instead of replaying the broken step (which would look like lost history).
        if _run_raised:
            await _clear_interrupted_graph_state(agent, config)

    yield emitter.done(processor.full_response).data
