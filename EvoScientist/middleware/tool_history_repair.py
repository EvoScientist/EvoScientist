"""Repair interrupted tool-call history before provider requests.

Strict providers (OpenAI, etc.) reject a message thread in which an assistant
tool call has no matching tool result. That happens whenever a run is
interrupted (cancelled, crashed, timed out) after the model emitted tool calls
but before those tools produced results. This middleware rewrites the outgoing
request so every dangling tool call is closed with a synthetic error result and
every orphan ``ToolMessage`` (a result whose originating call is gone) is
dropped. It also removes tool calls without names, which strict
OpenAI-compatible providers reject during history replay.

It covers cases that deepagents' ``PatchToolCallsMiddleware`` does not:

1. Orphan ``ToolMessage`` dropping -- a tool result whose originating tool call
   is no longer present in history is removed, rather than left to trip strict
   providers.
2. Mid-run coverage -- repair runs at the model boundary on every request
   (including malformed / ``invalid_tool_calls``), not only at agent start, so
   interruptions that happen partway through a run are healed too.
3. Blank ``tool_call_id`` normalization -- some streaming providers (notably
   Kimi and Zhipu on streamed tool calls) occasionally emit tool calls whose
   id is an empty string, whitespace, or ``None``. Strict providers reject the
   next turn with ``invalid tool_call_id`` (HTTP 400, code 3). Each blank id
   is replaced with a fresh unique id, preserving the AIMessage↔ToolMessage
   pairing by positional FIFO matching.

Because the middleware only rewrites the request and cannot mutate thread
state, the repaired synthetic results are recomputed on every model call. To
avoid re-logging the same repair forever, warnings are deduplicated per unique
tool-call id via a ``warned`` set owned by the middleware instance.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage

logger = logging.getLogger(__name__)
_INTERRUPTED_RESULT = "Tool execution was interrupted before completion."
_REPAIR_ID_PREFIX = "_repair_"


def _is_blank_tool_call_id(value: object) -> bool:
    """True for ids strict providers reject (None / empty / whitespace-only).

    Non-string values (ints, lists, etc.) are left for the main loop's existing
    strict-type filtering to handle -- only None and blank/whitespace strings
    count as "blank" here.
    """
    return value is None or (isinstance(value, str) and not value.strip())


def _normalize_blank_tool_call_ids(
    messages: Sequence[AnyMessage],
) -> list[AnyMessage]:
    """Replace blank tool_call_ids with deterministic, positionally-stable ids.

    Each blank id is rewritten as ``_repair_{msg_idx}_{kind}{call_idx}`` (kind
    is ``v`` for valid calls, ``i`` for invalid). The id is therefore stable
    across model calls -- the middleware re-runs on every request but the
    message thread is unchanged, so the same blank call gets the same repair
    id every time. That lets the main loop's existing ``warned``-set dedup
    suppress the synthesized-result warning on the second and later calls
    without any pre-add hack, and it keeps the wire payload stable.

    Pairs each blank-id call on an ``AIMessage`` with the next blank-id
    ``ToolMessage`` *in the same exchange* (FIFO) so existing exchanges stay
    paired. The FIFO queue is closed at every non-``ToolMessage`` boundary
    (new ``AIMessage``, ``HumanMessage``, ``SystemMessage``, ...) so a later
    exchange's blank ``ToolMessage`` can never steal an id from an earlier,
    already-interrupted exchange.

    Only valid ``tool_calls`` push their fresh ids into ``pending_slots``.
    ``invalid_tool_calls`` get fresh ids too (langchain-openai's serializer
    puts ``tool_calls + invalid_tool_calls`` on the wire when either parsed
    list is non-empty, so a blank id on an invalid call reaches the provider
    just as readily), but those ids are NOT pushed to ``pending_slots``:
    invalid calls are never executed by LangGraph, so no real ToolMessage can
    exist for them, and pushing the slot would let an orphan blank result
    from some other call mis-pair with the invalid call.

    Orphan blank ``ToolMessage``\\ s (no preceding blank valid call in their
    exchange to claim) keep their blank id -- the main repair loop drops them
    as orphans.

    The raw ``additional_kwargs["tool_calls"]`` form is dropped on any touched
    message so langchain-openai's serializer rebuilds the wire payload from
    the now-valid parsed form instead of consulting the raw form (which still
    carries the blank id).
    """
    pending_slots: list[str] = []
    normalized: list[AnyMessage] = []
    saw_blank = False

    for msg_idx, message in enumerate(messages):
        if isinstance(message, AIMessage):
            pending_slots.clear()
            new_tool_calls: list[dict[str, object]] = []
            any_changed = False
            for call_idx, call in enumerate(message.tool_calls):
                if _is_blank_tool_call_id(call.get("id")):
                    new_id = f"{_REPAIR_ID_PREFIX}{msg_idx}_v{call_idx}"
                    pending_slots.append(new_id)
                    new_tool_calls.append({**call, "id": new_id})
                    any_changed = True
                    saw_blank = True
                else:
                    new_tool_calls.append(dict(call))

            new_invalid: list[dict[str, object]] = []
            invalid_changed = False
            for call_idx, call in enumerate(
                getattr(message, "invalid_tool_calls", None) or []
            ):
                if _is_blank_tool_call_id(call.get("id")):
                    new_id = f"{_REPAIR_ID_PREFIX}{msg_idx}_i{call_idx}"
                    new_invalid.append({**call, "id": new_id})
                    invalid_changed = True
                    saw_blank = True
                else:
                    new_invalid.append(dict(call))

            if any_changed or invalid_changed:
                update: dict[str, object] = {"tool_calls": new_tool_calls}
                if new_invalid:
                    update["invalid_tool_calls"] = new_invalid
                additional_kwargs = message.additional_kwargs
                if isinstance(additional_kwargs, dict) and isinstance(
                    additional_kwargs.get("tool_calls"), list
                ):
                    new_additional = dict(additional_kwargs)
                    new_additional.pop("tool_calls", None)
                    update["additional_kwargs"] = new_additional
                message = message.model_copy(update=update)
        elif isinstance(message, ToolMessage):
            if _is_blank_tool_call_id(message.tool_call_id) and pending_slots:
                new_id = pending_slots.pop(0)
                message = message.model_copy(update={"tool_call_id": new_id})
                saw_blank = True
        else:
            pending_slots.clear()
        normalized.append(message)

    if saw_blank:
        logger.debug("Normalized blank tool_call_id(s) in message history")
    return normalized


def repair_tool_history(
    messages: Sequence[AnyMessage],
    warned: set[str] | None = None,
) -> list[AnyMessage]:
    """Return provider-valid history, preserving every complete tool exchange.

    When ``warned`` is provided, repair warnings are emitted only for tool-call
    ids not already present in it; newly-warned ids are added. This keeps the
    warning to once per unique interrupted/malformed call even though the
    middleware re-runs on every model call.
    """
    messages = _normalize_blank_tool_call_ids(messages)

    repaired: list[AnyMessage] = []
    pending: dict[str, str | None] = {}
    synthesized: list[str] = []
    dropped: list[str] = []

    def close_pending() -> None:
        for tool_call_id, tool_name in pending.items():
            repaired.append(
                ToolMessage(
                    content=_INTERRUPTED_RESULT,
                    tool_call_id=tool_call_id,
                    name=tool_name,
                    status="error",
                )
            )
            synthesized.append(tool_call_id)
        pending.clear()

    for message in messages:
        if isinstance(message, ToolMessage):
            tool_call_id = message.tool_call_id
            if tool_call_id in pending:
                repaired.append(message)
                pending.pop(tool_call_id)
            else:
                dropped.append(tool_call_id)
            continue

        if pending:
            close_pending()
        if isinstance(message, AIMessage):
            tool_calls = [call for call in message.tool_calls if call.get("name")]
            invalid_calls = [
                call
                for call in (getattr(message, "invalid_tool_calls", []) or [])
                if call.get("name")
            ]
            additional_kwargs = message.additional_kwargs
            raw_calls = additional_kwargs.get("tool_calls")
            valid_raw_calls = []
            if "tool_calls" in additional_kwargs:
                if isinstance(raw_calls, list):
                    for call in raw_calls:
                        function = (
                            call.get("function") if isinstance(call, dict) else None
                        )
                        name = (
                            function.get("name") if isinstance(function, dict) else None
                        )
                        tool_call_id = (
                            call.get("id") if isinstance(call, dict) else None
                        )
                        # Entries without a usable str name AND id can never be
                        # closed by a result, so keeping them would leave
                        # provider-invalid history in the payload.
                        if (
                            isinstance(name, str)
                            and name
                            and isinstance(tool_call_id, str)
                            and tool_call_id
                        ):
                            valid_raw_calls.append(call)
                additional_kwargs = dict(additional_kwargs)
                if valid_raw_calls:
                    additional_kwargs["tool_calls"] = valid_raw_calls
                else:
                    # Also covers non-list junk (dict/str/int), which would
                    # otherwise crash langchain's serializer downstream.
                    additional_kwargs.pop("tool_calls", None)

            message = message.model_copy(
                update={
                    "tool_calls": tool_calls,
                    "invalid_tool_calls": invalid_calls,
                    "additional_kwargs": additional_kwargs,
                }
            )
            all_calls = tool_calls + invalid_calls
            for call in all_calls:
                if tool_call_id := call.get("id"):
                    pending[tool_call_id] = call.get("name")
            for call in valid_raw_calls:
                pending[call["id"]] = call["function"]["name"]
        repaired.append(message)

    if pending:
        close_pending()

    if warned is not None:
        synthesized = [tid for tid in synthesized if tid not in warned]
        dropped = [tid for tid in dropped if tid not in warned]
        warned.update(synthesized)
        warned.update(dropped)

    if synthesized or dropped:
        logger.warning(
            "Repaired interrupted tool history: synthesized=%s dropped=%s",
            synthesized,
            dropped,
        )
    return repaired


class ToolHistoryRepairMiddleware(AgentMiddleware):
    """Repair dangling calls and orphan results at the model boundary."""

    name = "tool_history_repair"

    def __init__(self) -> None:
        super().__init__()
        self._warned: set[str] = set()

    def modify_request(self, request: ModelRequest) -> ModelRequest:
        messages = repair_tool_history(request.messages, warned=self._warned)
        return request.override(messages=messages)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(self.modify_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(self.modify_request(request))
