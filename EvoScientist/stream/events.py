"""Stream event generator and v3 protocol helpers.

Async generator that streams UI events from a DeepAgents/LangGraph v3 run.
"""

import asyncio
import base64
import inspect
import mimetypes
import os
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from langgraph.graph import END
from langgraph.types import Command, Interrupt

from ..memory.worker_activity import clear_completed_memory_activity_counts
from ..middleware.tool_history_repair import (
    _INTERRUPTED_RESULT as _INTERRUPTED_TOOL_RESULT,
)
from .emitter import StreamEventEmitter
from .summarization import (
    _extract_summary_message_text,
    _find_summarization_event_payload,
    _summarization_event_signature,
)
from .tool_results import _extract_command_tool_content, _extract_tool_content
from .tool_selection import _ToolSelectionSuppressor
from .utils import DisplayLimits, is_success

if TYPE_CHECKING:
    from ..backends import ParkedCloseReason
    from ..middleware.events import ToolSelectionView
from .v3_payloads import (
    GOAL_DRAFT_LINEAGE,
    RUBRIC_GRADER_LINEAGE,
    RawMap,
    _as_raw_map,
    _event_data,
    _event_namespace,
    _reasoning_from_content,
    _split_message_event_data,
    _strip_legacy_thinking_tags,
    _text_from_content,
    _under,
    _usage_counts,
)

UserMessageContent: TypeAlias = str | list[dict[str, object]]
GraphRunInput: TypeAlias = str | Command
LangGraphStreamInput: TypeAlias = dict[str, object] | Command
_ValueMessageKey: TypeAlias = tuple[str, ...]
_INTERRUPT_CHECKPOINT_WAIT_SECONDS = 5.0
_INTERRUPT_CHECKPOINT_POLL_SECONDS = 0.05


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


def _snapshot_interrupt_ids(snapshot: Any) -> set[str]:
    """Return interrupt IDs currently persisted at the root or task level."""
    if snapshot is None:
        return set()
    interrupts = list(getattr(snapshot, "interrupts", None) or ())
    for task in getattr(snapshot, "tasks", None) or ():
        interrupts.extend(getattr(task, "interrupts", None) or ())
    return {
        str(interrupt_id)
        for interrupt in interrupts
        if (interrupt_id := getattr(interrupt, "id", None)) is not None
    }


async def _wait_for_checkpointed_interrupt(
    agent: Any, config: dict[str, Any], interrupt_id: str
) -> None:
    """Wait for an emitted interrupt to become visible in its checkpoint."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _INTERRUPT_CHECKPOINT_WAIT_SECONDS
    while True:
        try:
            snapshot = await agent.aget_state(config)
        except Exception:
            return
        if str(interrupt_id) in _snapshot_interrupt_ids(snapshot):
            return
        if not _snapshot_needs_recovery(snapshot):
            return
        remaining = deadline - loop.time()
        if remaining <= 0:
            return
        await asyncio.sleep(min(_INTERRUPT_CHECKPOINT_POLL_SECONDS, remaining))


def _snapshot_needs_recovery(snapshot: Any) -> bool:
    """True if the checkpoint has unfinished work or uncommitted task writes.

    LangGraph computes ``next`` as the tasks that have not yet produced
    writes. A SIGKILL between ``put_writes`` and ``put_checkpoint`` can
    leave every tool call finished (so ``next`` is empty) while
    ``snapshot.tasks`` still holds those tools with their pending results.
    A completed turn has both ``next`` and ``tasks`` empty.
    """
    if snapshot is None:
        return False
    return bool(getattr(snapshot, "next", None) or getattr(snapshot, "tasks", None))


class _GatewayCheckpointOps:
    """Compiled-graph state ops backed by a GraphGateway.

    ``_recover_interrupted_graph_state`` talks to ``aget_state`` /
    ``aupdate_state``. HITL close already has a gateway, and after #470
    ``GraphTarget.local_graph`` may be None; this adapter is the seam so
    recovery does not reach into the in-process graph.
    """

    def __init__(self, gateway: Any, target: Any, thread_id: str) -> None:
        self._gateway = gateway
        self._target = target
        self._thread_id = thread_id

    async def aget_state(self, _config: dict[str, Any]) -> Any:
        return await self._gateway.get_state_snapshot(self._target, self._thread_id)

    async def aupdate_state(
        self,
        _config: dict[str, Any],
        values: dict[str, Any] | None,
        as_node: str | None = None,
    ) -> None:
        await self._gateway.update_state_values(
            self._target,
            self._thread_id,
            values,
            as_node=as_node,
        )


# Where the root graph parks while a gated call waits for approval: nothing of
# that batch has run yet. Any other position means tools were running.
_HITL_PARK_NODE = "HumanInTheLoopMiddleware.after_model"


def _one_line(value: Any, limit: int = 200) -> str:
    """*value* as one bounded log line (a server refusal embeds the run payload)."""
    return " ".join(str(value).split())[:limit]


def _every_call_answered(snapshot: Any) -> bool:
    """Whether no call of the last AI message lacks a result (unreadable: no)."""
    from ..backends import _unanswered_calls

    values = getattr(snapshot, "values", None) or {}
    return _unanswered_calls(list(values.get("messages") or [])) == []


def _interrupted_dangling_patch(messages: list) -> list[ToolMessage]:
    """Synthetic error results for tool calls still unanswered in *messages*."""
    answered_ids = {
        m.tool_call_id
        for m in messages
        if getattr(m, "type", None) == "tool" and getattr(m, "tool_call_id", None)
    }
    return [
        ToolMessage(
            content=_INTERRUPTED_TOOL_RESULT,
            tool_call_id=call["id"],
            name=call.get("name") or "unknown",
            status="error",
        )
        for m in messages
        if isinstance(m, AIMessage)
        for call in (
            *m.tool_calls,
            *(getattr(m, "invalid_tool_calls", None) or ()),
        )
        if call.get("id") and call["id"] not in answered_ids
    ]


async def _recover_interrupted_graph_state(
    agent: Any,
    config: dict[str, Any],
    snapshot: Any | None = None,
    *,
    close_interrupts: bool = False,
    close_reason: "ParkedCloseReason" = "round_limit",
) -> bool:
    """Close out a run that ended mid-step so the next turn starts fresh.

    A cancelled (Ctrl+C / ``/stop``), crashed, or hard-killed run can leave the
    LangGraph checkpoint in two kinds of trouble:

    1. Pending tasks: ``next`` stays parked on the interrupted node. New
       input makes LangGraph discard those tasks, along with the results of
       any tool call in the batch that had already finished. ``next`` is
       computed as the tasks without writes, so a SIGKILL between
       ``put_writes`` and ``put_checkpoint`` can leave ``next`` empty while
       ``snapshot.tasks`` still holds the finished tools; recovery has to
       look at both.
    2. Dangling tool calls: the assistant message that dispatched the tools was
       committed, but its results never were. deepagents'
       ``PatchToolCallsMiddleware`` then closes each dangling call with "was
       cancelled - another message came in before it could be completed" and
       rewrites the whole messages channel — which re-broadcasts every
       historical tool call to the UI and actively invites the model to
       re-issue the entire interrupted batch, re-applying real side effects.

    Recovery commits the turn as finished via ``aupdate_state(config, None,
    as_node=END)`` — which also lands the pending writes of tool calls that
    did finish, so they keep their real results — and synthesizes an honest
    ``ToolMessage`` for every still-dangling call (telling the model the run
    was interrupted and side effects may already have partially applied),
    leaving ``next`` empty and a self-consistent history the next turn can
    build on normally. Channel values (message history) are otherwise
    preserved.

    The dangling-call patch is computed from the state re-read after the
    first END, the same view the verify step sees, so the patch and the
    final checkpoint agree on which calls are still unanswered.

    Critically, the default path only runs when the stuck state is *not* a
    legitimate human-in-the-loop interrupt. The agent pauses via
    ``interrupt()`` / ``Command(resume=...)`` for ask-user flows, which also
    leaves ``next`` non-empty; clearing those would silently discard a pending
    question the user still needs to answer.
    ``_snapshot_has_pending_interrupt`` distinguishes the two.

    Pass ``close_interrupts=True`` to close a genuine HITL/ask_user pause
    anyway (the turn ended there: declined, unanswered, or out of rounds).
    Synthetic results are then worded for ``close_reason`` and for whether
    the graph had stopped before any tool of the batch ran. The checkpoint
    is verified empty of pending tasks and interrupts, and every call of the
    last AI message must have a result: a close that stopped half-way is
    reported, not repaired.

    ``agent`` is any object with ``aget_state`` / ``aupdate_state`` — the
    compiled graph on the crash-recovery path, or ``_GatewayCheckpointOps``
    when HITL close drives recovery through ``GraphGateway``.

    Returns:
        ``True`` when the thread is safe to run: it was already clean, it is
        parked at a genuine human-in-the-loop interrupt (left untouched,
        unless ``close_interrupts``), or recovery completed and the
        checkpoint verified clean. ``False`` when the checkpoint may still
        be stuck — recovery failed or could not be verified. The next run is
        still safe to start: LangGraph discards the stale tasks and
        deepagents closes the dangling calls, with the "was cancelled"
        wording this recovery exists to avoid.

    Best-effort: any failure is logged and swallowed so it never shadows the
    original exception or cancellation that triggered recovery — the return
    value is how callers learn the outcome. HITL budget close raises when
    this returns ``False``.
    """
    import logging

    _log = logging.getLogger(__name__)
    try:
        if snapshot is None:
            snapshot = await agent.aget_state(config)
        if not snapshot:
            return True
        parked = _snapshot_has_pending_interrupt(snapshot)
        if parked and not close_interrupts:
            _log.debug(
                "Leaving interrupted graph state intact for thread %s: "
                "pending human-in-the-loop interrupt (next=%s)",
                config.get("configurable", {}).get("thread_id", "?"),
                snapshot.next,
            )
            return True
        # Only act when the graph is genuinely stuck: non-empty next, or
        # tasks whose writes never made it into a checkpoint. A HITL close
        # also proceeds when the pause is parked even if ``next`` is empty.
        if not _snapshot_needs_recovery(snapshot) and not (close_interrupts and parked):
            if close_interrupts and not _every_call_answered(snapshot):
                _log.warning(
                    "Thread %s has nothing pending but a tool call without a "
                    "result; an earlier close did not finish",
                    config.get("configurable", {}).get("thread_id", "?"),
                )
                return False
            return True

        stuck_at = snapshot.next or tuple(
            t.name for t in (getattr(snapshot, "tasks", None) or ())
        )
        # Clear first: it commits the pending writes of calls that did finish.
        # Patching first would reuse a finished task's id and lose the patch.
        await agent.aupdate_state(config, None, as_node=END)
        after = await agent.aget_state(config)
        values = getattr(after, "values", None) or {}
        messages = values.get("messages") or []
        if close_interrupts:
            from ..backends import abandoned_tool_messages

            patch = abandoned_tool_messages(
                list(messages),
                reason=close_reason,
                parked_before_tools=bool(stuck_at)
                and all(node == _HITL_PARK_NODE for node in stuck_at),
            )
        else:
            patch = _interrupted_dangling_patch(list(messages))
        if patch:
            # Attribute the synthetic results to the stuck execution node so
            # they land in history exactly where the real results would have.
            write_node = (
                "tools"
                if "tools" in stuck_at
                else (stuck_at[0] if stuck_at else "tools")
            )
            await agent.aupdate_state(config, {"messages": patch}, as_node=write_node)
        # Unconditional: with pending writes and an empty patch, the first
        # END commits those writes and leaves next == ('model',). Nesting the
        # trailing clear under ``if patch`` would then warn-fail a thread
        # that is actually fine.
        await agent.aupdate_state(config, None, as_node=END)
        # Verify the checkpoint actually cleared, so a failed repair is
        # observable (logged + reported via the return value) rather than
        # silently reverting to the dangling-calls state.
        verify = await agent.aget_state(config)
        still_stuck = _snapshot_needs_recovery(verify) or (
            close_interrupts and _snapshot_has_pending_interrupt(verify)
        )
        if still_stuck:
            stuck = getattr(verify, "next", None) or getattr(verify, "tasks", None)
            _log.warning(
                "Interrupted graph state for thread %s is still stuck at %s "
                "after recovery",
                config.get("configurable", {}).get("thread_id", "?"),
                _one_line(stuck) if close_interrupts else stuck,
            )
            return False
        if close_interrupts and not _every_call_answered(verify):
            _log.warning(
                "Closed the parked checkpoint of thread %s but a tool call "
                "still has no result",
                config.get("configurable", {}).get("thread_id", "?"),
            )
            return False
        _log.debug(
            "Recovered interrupted graph state for thread %s (was stuck at: "
            "%s; closed %d dangling tool call(s))",
            config.get("configurable", {}).get("thread_id", "?"),
            stuck_at,
            len(patch),
        )
        return True
    except asyncio.CancelledError:
        if close_interrupts:
            _log.warning(
                "Closing the parked checkpoint of thread %s was cancelled "
                "before it finished",
                config.get("configurable", {}).get("thread_id", "?"),
            )
        raise
    except Exception as exc:
        thread_id = config.get("configurable", {}).get("thread_id", "?")
        if close_interrupts:
            _log.warning(
                "Could not close the parked checkpoint of thread %s: %s",
                thread_id,
                _one_line(exc),
            )
            _log.debug("Parked checkpoint close failed", exc_info=True)
        else:
            _log.warning(
                "Could not recover interrupted graph state for thread %s: %s",
                thread_id,
                exc,
                exc_info=True,
            )
        return False


@dataclass(frozen=True)
class _SubagentInfo:
    path: tuple[str, ...]
    name: str
    description: str = ""

    @property
    def instance_id(self) -> str:
        return ":".join(self.path)


@dataclass
class _CallRecord:
    """One provider call of a namespace, between its ``model_call`` start and end events."""

    call: str
    state: str = "open"  # open | clean | error | failed
    text_chars: int = 0
    thinking_chars: int = 0
    tool_call_ids: list[str] = field(default_factory=list)
    full_response_len: int = 0
    selector_snapshot: tuple[bool, str, bool] | None = None

    def discard_payload(self) -> dict[str, Any]:
        return {
            "text_chars": self.text_chars,
            "thinking_chars": self.thinking_chars,
            "tool_call_ids": list(self.tool_call_ids),
        }

    @property
    def discardable(self) -> bool:
        return self.state != "clean"

    @property
    def ended_badly(self) -> bool:
        return self.state in ("error", "failed")

    @property
    def empty(self) -> bool:
        return not (self.text_chars or self.thinking_chars or self.tool_call_ids)


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


def _count(value: Any) -> int:
    """A non-negative int from a custom stream payload field, else 0."""
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(value, 0)


class _V3EventProcessor:
    """Translate v3 protocol channel events into EvoScientist UI events."""

    def __init__(
        self,
        emitter: StreamEventEmitter,
        subagents: _SubagentRegistry,
        existing_summarization_event: Mapping[str, object] | None,
        existing_messages: object = None,
        process_value_messages: bool = False,
        events: "ToolSelectionView | None" = None,
    ) -> None:
        from ..middleware.events import NO_OP_SINK

        self.emitter = emitter
        self.subagents = subagents
        self._suppressed_summarization_signature = _summarization_event_signature(
            existing_summarization_event
        )
        self._seen_value_message_keys = self._message_keys(existing_messages)
        # Snapshot recovery delivers messages the transport did not stream. It
        # is suspended once root text streams and resumed when that text is
        # discarded, since the answer that replaces it may not stream.
        self._value_recovery = process_value_messages
        self._process_value_message_snapshots = process_value_messages
        self.full_response = ""
        # Where the pass a graded run's last verdict applies to starts in
        # ``full_response``; ``None`` until a verdict arrives.
        self._last_pass_start: int | None = None
        self._summarization_in_progress = False
        self._tool_inputs: dict[
            tuple[tuple[str, ...], str], tuple[str, dict[str, Any]]
        ] = {}
        self._emitted_tool_calls: set[tuple[tuple[str, ...], str]] = set()
        self._emitted_interrupts: set[str] = set()
        self._selector = _ToolSelectionSuppressor(emitter, events or NO_OP_SINK)
        self._records: dict[tuple[str, ...], _CallRecord] = {}
        self._namespace: tuple[str, ...] = ()

    @staticmethod
    def _tool_scope(
        namespace: tuple[str, ...], subagent: _SubagentInfo | None
    ) -> tuple[str, ...]:
        return subagent.path if subagent is not None else namespace

    # -- provider calls: what a retry or a fallback may discard -------------
    # The boundary is the ``model_call`` custom event pair, never message
    # metadata: the server transport strips run_id and node before SSE.
    # Output is attributed by timing, so a stream that bypasses the boundary
    # yet overlaps an open call in the same namespace is counted with it.

    def _current_record(self) -> _CallRecord | None:
        record = self._records.get(self._namespace)
        return record if record is not None and record.state == "open" else None

    def _count_text(self, chars: int) -> None:
        record = self._current_record()
        if record is not None:
            record.text_chars += chars

    def _count_thinking(self, chars: int) -> None:
        record = self._current_record()
        if record is not None:
            record.thinking_chars += chars

    def _discard(
        self, record: _CallRecord, subagent: _SubagentInfo | None
    ) -> dict[str, Any] | None:
        """Roll back to the call's start; the payload the clients apply, or None if nothing was emitted."""
        if not self._namespace:
            # The root answer loses everything it gained since the call began;
            # report that amount so clients trim exactly what is cut here.
            record.text_chars = max(
                0, len(self.full_response) - record.full_response_len
            )
            self.full_response = self.full_response[: record.full_response_len]
            if record.selector_snapshot is not None:
                self._selector.restore(record.selector_snapshot)
            self._process_value_message_snapshots = self._value_recovery
        scope = self._tool_scope(self._namespace, subagent)
        for tool_call_id in record.tool_call_ids:
            self._emitted_tool_calls.discard((scope, tool_call_id))
            self._tool_inputs.pop((scope, tool_call_id), None)
        self._records.pop(self._namespace, None)
        return None if record.empty else record.discard_payload()

    def _process_model_call(
        self, payload: RawMap, subagent: _SubagentInfo | None
    ) -> list[dict[str, Any]]:
        """Open or close this namespace's call; a new call discards a predecessor that ended badly.

        A predecessor whose end mark never arrived is left alone: dropping it
        would cost a good answer if only the mark was lost.
        """
        call = str(payload.get("call") or "")
        phase = payload.get("phase")
        record = self._records.get(self._namespace)
        if phase == "end":
            if record is not None and record.state == "open" and record.call == call:
                status = payload.get("status")
                record.state = status if status in ("clean", "error") else "failed"
            return []
        if phase != "start":
            return []
        events: list[dict[str, Any]] = []
        if record is not None and record.ended_badly:
            discarded = self._discard(record, subagent)
            if discarded is not None:
                events.append(
                    self.emitter.model_output_discarded(
                        instance_id=subagent.instance_id if subagent else "",
                        discarded=discarded,
                    ).data
                )
        self._records[self._namespace] = _CallRecord(
            call=call,
            full_response_len=len(self.full_response),
            selector_snapshot=self._selector.snapshot()
            if not self._namespace
            else None,
        )
        return events

    def _process_model_retry(
        self, payload: RawMap, subagent: _SubagentInfo | None
    ) -> list[dict[str, Any]]:
        record = self._records.get(self._namespace)
        discarded = None
        if record is not None and record.discardable:
            discarded = self._discard(record, subagent)
        name = ""
        if subagent is not None:
            # The event names its own agent; a namespace registered by the
            # server tracker is only named after its graph node.
            agent = payload.get("agent")
            name = agent if isinstance(agent, str) and agent else subagent.name
        return [
            self.emitter.model_retry(
                attempt=_count(payload.get("attempt")) or 1,
                wait_seconds=float(payload.get("wait_seconds") or 0.0),
                reason=str(payload.get("reason") or ""),
                budget_left_seconds=float(payload.get("budget_left_seconds") or 0.0),
                cause=str(payload.get("cause") or "exception"),
                instance_id=subagent.instance_id if subagent else "",
                subagent=name,
                discarded=discarded
                or {"text_chars": 0, "thinking_chars": 0, "tool_call_ids": []},
            ).data
        ]

    async def process(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        method = event.get("method")
        namespace = _event_namespace(event)
        self._namespace = namespace
        subagent = None
        # Waiting for the drafter's namespace to resolve would hold every
        # later event back until the stream ends; its own output is not shown.
        drafting = _under(namespace, GOAL_DRAFT_LINEAGE)
        if namespace and method in ("messages", "tools"):
            # The grader's own output is not shown either.
            if drafting or _under(namespace, RUBRIC_GRADER_LINEAGE):
                return []
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
            if not namespace:
                if self._process_value_message_snapshots:
                    events.extend(self._process_value_messages(_event_data(event)))
                elif self._value_recovery:
                    # Streamed messages are already delivered: remember them
                    # so a later resume does not deliver them again.
                    self._mark_value_messages_seen(_event_data(event))
            return events
        if method == "input.requested":
            return self._process_input_requested(event.get("params"))
        if method == "custom":
            payload = _as_raw_map(_event_data(event))
            # A goal event is about the thread, whichever run wrote it (the
            # drafting line comes from a task below a hook): no sub-agent to
            # wait for.
            if payload is not None and payload.get("type") == "goal_state":
                return self._process_goal_event(payload)
            if drafting:
                return []
            subagent = (
                await self.subagents.wait_resolve(namespace) if namespace else None
            )
            return self._process_custom_event(_event_data(event), subagent)
        return []

    def _process_rubric_event(self, kind: str, payload: Any) -> list[dict[str, Any]]:
        if kind == "rubric_evaluation_deferred":
            return [
                self.emitter.rubric_evaluation_deferred(
                    tasks=_count(payload.get("tasks")),
                    processes=_count(payload.get("processes")),
                ).data
            ]
        raw_iteration = payload.get("iteration")
        iteration = raw_iteration if isinstance(raw_iteration, int) else 0
        if kind == "rubric_evaluation_start":
            return [self.emitter.rubric_evaluation_start(iteration=iteration).data]
        result = payload.get("result")
        explanation = payload.get("explanation")
        criteria = payload.get("criteria")
        if self._last_pass_start is None:
            self._last_pass_start = 0
        if result == "needs_revision":
            if self.full_response.strip() and not self.full_response.endswith("\n\n"):
                # Keep the draft; the revision starts on its own paragraph.
                self.full_response += "\n\n"
            self._last_pass_start = len(self.full_response)
        return [
            self.emitter.rubric_evaluation(
                iteration=iteration,
                result=result if isinstance(result, str) else "",
                explanation=explanation if isinstance(explanation, str) else "",
                criteria=[dict(c) for c in criteria if isinstance(c, Mapping)]
                if isinstance(criteria, list)
                else [],
                unverified=payload.get("unverified") is True,
            ).data
        ]

    @property
    def last_pass_response(self) -> str | None:
        """The text after the last revision request; ``None`` for an ungraded run."""
        if self._last_pass_start is None:
            return None
        return self.full_response[self._last_pass_start :]

    def _process_goal_event(self, payload: Any) -> list[dict[str, Any]]:
        change = payload.get("change")
        if not isinstance(change, str) or not change:
            return []

        def _text(key: str) -> str:
            value = payload.get(key)
            return value if isinstance(value, str) else ""

        return [
            self.emitter.goal_state(
                change=change,
                status=_text("status"),
                note=_text("note"),
                objective=_text("objective"),
                criteria=_text("criteria"),
                message=_text("message"),
                unreviewed=payload.get("unreviewed") is True,
            ).data
        ]

    def _process_custom_event(
        self, data: object, subagent: _SubagentInfo | None = None
    ) -> list[dict[str, Any]]:
        """Translate a ``custom`` stream payload into UI events.

        Currently handles the ``subagent`` lifecycle emitted by
        ``langchain_quickjs`` for in-eval ``task()`` fan-out: ``start`` /
        ``complete`` / ``error`` become ``panel_dispatch_start`` /
        ``panel_dispatch_complete`` / ``panel_dispatch_error`` UI events.
        Also handles the acceptance grader's ``rubric_evaluation_start`` /
        ``rubric_evaluation_end`` / ``rubric_evaluation_deferred``. Unrecognised event types are ignored so
        future producers can extend the custom channel without breaking
        existing consumers.
        """
        payload = _as_raw_map(data)
        if payload is None:
            return []
        kind = payload.get("type")
        if kind == "model_call":
            return self._process_model_call(payload, subagent)
        if kind == "model_retry":
            return self._process_model_retry(payload, subagent)
        if kind in (
            "rubric_evaluation_start",
            "rubric_evaluation_end",
            "rubric_evaluation_deferred",
        ):
            return self._process_rubric_event(kind, payload)
        if kind != "subagent":
            return []
        phase = payload.get("phase")
        eval_id = payload.get("eval_id")
        dispatch_id = payload.get("id")
        if not isinstance(dispatch_id, str):
            return []
        eval_id_str = eval_id if isinstance(eval_id, str) else ""
        if phase == "start":
            subagent_type = payload.get("subagent_type")
            label = payload.get("label")
            description = payload.get("description")
            return [
                self.emitter.panel_dispatch_start(
                    eval_id=eval_id_str,
                    dispatch_id=dispatch_id,
                    subagent_type=subagent_type
                    if isinstance(subagent_type, str)
                    else "",
                    label=label if isinstance(label, str) else "",
                    description=description if isinstance(description, str) else "",
                ).data
            ]
        raw_duration = payload.get("duration_ms")
        duration_ms = raw_duration if isinstance(raw_duration, int) else 0
        if phase == "complete":
            return [
                self.emitter.panel_dispatch_complete(
                    eval_id=eval_id_str,
                    dispatch_id=dispatch_id,
                    duration_ms=duration_ms,
                ).data
            ]
        if phase == "error":
            error = payload.get("error")
            return [
                self.emitter.panel_dispatch_error(
                    eval_id=eval_id_str,
                    dispatch_id=dispatch_id,
                    duration_ms=duration_ms,
                    error=error if isinstance(error, str) else "",
                ).data
            ]
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

    def _mark_value_messages_seen(self, data: object) -> None:
        data_map = _as_raw_map(data)
        messages = data_map.get("messages") if data_map is not None else None
        self._seen_value_message_keys |= self._message_keys(messages)

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
                self._count_text(len(pending))
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
        record = self._current_record()
        if record is not None:
            record.tool_call_ids.append(tool_call_id)
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
        self._count_text(len(emit_text))
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
        suppressed, events = self._selector.process_thinking(text)
        if suppressed:
            return events
        self._count_thinking(len(text))
        events.append(self.emitter.thinking(text).data)
        return events

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
    rubric: str | None = None,
    goal_request: dict[str, Any] | None = None,
) -> LangGraphStreamInput:
    """Build the LangGraph run input shared by local and server gateways.

    Every new input writes ``goal_request``: a ``/goal`` request travels
    without a message, and a message clears a request whose own run never
    finished, so it cannot take over this turn.
    """
    if not isinstance(message, str):
        return message
    if goal_request is not None:
        return {"messages": [], "goal_request": dict(goal_request)}

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
    run_input: dict[str, object] = {
        "messages": [{"role": "user", "content": user_content}],
        "goal_request": None,
    }
    if rubric is not None:
        run_input["rubric"] = rubric
    return run_input


async def stream_agent_events(
    agent: Any,
    message: GraphRunInput,
    thread_id: str,
    metadata: dict[str, Any] | None = None,
    media: list[str] | None = None,
    events: "ToolSelectionView | None" = None,
    configurable_extra: dict[str, Any] | None = None,
    rubric: str | None = None,
    goal_request: dict[str, Any] | None = None,
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
        configurable_extra: Optional extra keys merged into ``configurable``
            alongside ``thread_id`` — e.g. ``{"active_teams": [...]}`` from
            TUI ``/expert`` bindings, mirroring what WebUI writes via
            langgraph-sdk. Ignored if ``None`` or empty.

    Yields:
        Event dicts: thinking, text, tool_call, tool_result,
                     subagent_start, subagent_tool_call, subagent_tool_result, subagent_end,
                     done, error
    """
    if events is None:
        from .sink import SessionEventSink

        events = SessionEventSink()

    # Single assembly point for the run config (see gateway.types). No
    # per-run model/limit overrides on the local path: the local agent is
    # rebuilt on model switches and binds recursion_limit at construction
    # from the same live config. The HITL suppression flag is the exception -
    # a per-run signal both backends need (the graph is always armed).
    from ..backends import (
        HITL_APPROVE_ALL_KEY,
        hitl_suppressed_for_run,
        model_retry_for_run,
        resume_declines,
    )
    from ..gateway.types import resolve_per_run_config

    config = resolve_per_run_config(
        thread_id,
        configurable_extra,
        hitl_suppressed=hitl_suppressed_for_run(),
        rubric_run=rubric is not None,
        model_retry=model_retry_for_run(),
    )
    # The callers already keep the grant off a declining answer. Every local
    # run passes here, so one that slips still cannot run a declined call.
    if resume_declines(message) and config["configurable"].pop(
        HITL_APPROVE_ALL_KEY, None
    ):
        import logging

        logging.getLogger(__name__).warning(
            "dropped %s from a resume that does not keep the grant",
            HITL_APPROVE_ALL_KEY,
        )
    if metadata:
        config["metadata"] = metadata
    emitter = StreamEventEmitter()
    existing_summarization_event: Mapping[str, object] | None = None
    snapshot: Any = None
    try:
        snapshot = await agent.aget_state(config)
        existing_summarization_event = _find_summarization_event_payload(
            getattr(snapshot, "values", None)
        )
    except Exception:
        pass

    # A previous run cancelled (Ctrl+C), crashed, or hard-killed mid-step leaves
    # the checkpoint parked with pending tasks and dangling tool calls. Close
    # it out before this run's input is applied, or deepagents patches the
    # calls as "was cancelled", re-broadcasts the history to the UI and invites
    # the model to re-issue the batch. A failed repair is logged, not fatal.
    # Also covers the all-writes-uncommitted case (next empty, tasks present).
    if _snapshot_needs_recovery(snapshot):
        await _recover_interrupted_graph_state(agent, config, snapshot=snapshot)

    clear_completed_memory_activity_counts()
    astream_input = await build_agent_stream_input(
        message, media=media, rubric=rubric, goal_request=goal_request
    )

    stream: Any | None = None
    producers: list[asyncio.Task[Any]] = []
    _run_raised: bool = False
    event_sink_token = None
    try:
        from langgraph.stream.transformers import CustomTransformer, UpdatesTransformer

        from ..middleware.events import (
            MiddlewareEventSink,
            bind_run_event_sink,
            reset_run_event_sink,
        )

        if isinstance(events, MiddlewareEventSink):
            event_sink_token = bind_run_event_sink(events)

        try:
            stream_result = agent.astream_events(
                astream_input,
                config=config,
                version="v3",
                transformers=[UpdatesTransformer, CustomTransformer],
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
            events=events,
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
                # A nested run that stopped at a pause goes on with the turn.
                paused = getattr(subagent, "status", None) == "interrupted"
                await queue.put(
                    emitter.subagent_end(
                        name, instance_id=instance_id, paused=paused
                    ).data
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
                    if _under(path, RUBRIC_GRADER_LINEAGE):
                        # Registered so its custom events (call marks, retry
                        # notice) resolve at once and carry its name.
                        subagents.register(path, name, description)
                        continue
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
            if item.get("type") == "interrupt":
                # The client may close this generator as soon as it sees the
                # pause. Let LangGraph persist the interrupt before aborting
                # the underlying stream in our ``finally`` block.
                await _wait_for_checkpointed_interrupt(
                    agent, config, item["interrupt_id"]
                )
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
        if event_sink_token is not None:
            reset_run_event_sink(event_sink_token)
        # When the run ended with an exception the LangGraph checkpoint may be
        # left interrupted (``next`` non-empty). Recover — unless it's a real
        # human-in-the-loop pause. Cancelled and hard-killed runs are recovered
        # at the start of the next run instead.
        if _run_raised:
            await _recover_interrupted_graph_state(agent, config)

    yield emitter.done(
        processor.full_response, last_pass_response=processor.last_pass_response
    ).data
