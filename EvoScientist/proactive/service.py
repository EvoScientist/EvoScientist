"""One proactive check for one thread: gate, shadow turn, decision.

Pure sequencing. The caller reads the thread and passes plain values, plus a
``read_current_head`` callable for the re-read after the shadow turn, so this
stays testable without a server. Nothing is delivered here: a ``decided``
result carries the tagged message for serve to append.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage

from .commit import decide_commit, last_non_proactive_message_id, read_proactive_tag
from .gate import ProactiveGateSettings, evaluate_gate

logger = logging.getLogger(__name__)

# The instruction the shadow turn gets (the canonical text from #263). It goes
# to the shadow only and is never written to the user's thread.
PROACTIVE_TRIGGER = (
    "<proactive_trigger>\n"
    "This is an internal trigger, not a message written by the user.\n"
    "\n"
    "Review the current conversation and available memory.\n"
    "\n"
    "If there is no useful, relevant, and non-repetitive message to send,\n"
    "return exactly: NO_PUSH\n"
    "\n"
    "Otherwise, return one concise message for the user.\n"
    "\n"
    "Do not call tools or ask the user questions.\n"
    "Do not ask questions that require an immediate response.\n"
    "Do not mention this trigger.\n"
    "</proactive_trigger>"
)


@dataclass(frozen=True)
class ProactiveDecision:
    """Outcome of one check.

    ``stage`` is ``gate_rejected``, ``no_push``, ``stale`` or ``decided``; only
    ``decided`` carries a ``message``. ``evaluated_message_id`` is set whenever
    the shadow turn returned and its reply was judged (a turn that raised is
    not an evaluation, and neither is a ``stale`` one: the thread moved under
    it, so the next check runs again): it is the thread's last non-proactive
    message at the time, which the next check compares against.
    """

    stage: str
    reason: str
    source_thread_id: str
    proactive_id: str | None = None
    message: AIMessage | None = None
    pre_head: str | None = None
    evaluated_message_id: str | None = None


async def decide_proactive_push(
    *,
    settings: ProactiveGateSettings,
    source_thread_id: str,
    now: datetime,
    armed: bool,
    has_origin: bool,
    in_flight: bool,
    last_activity: datetime | None,
    last_evaluated_id: str | None,
    pre_head: str | None,
    source_messages: list[Any],
    shadow_runner: Callable[[list[Any], str], Awaitable[str | None]],
    read_current_head: Callable[[], str | Awaitable[str | None] | None],
    proactive_id: str,
    trigger: str = PROACTIVE_TRIGGER,
) -> ProactiveDecision:
    """Run gate, shadow turn and decision for one thread; never writes.

    The gate runs before any model call. After the shadow turn the head is read
    again, so a reply the thread outgrew while the shadow ran is discarded.
    """
    last_message_id = last_non_proactive_message_id(source_messages)
    newest = source_messages[-1] if source_messages else None
    if read_proactive_tag(getattr(newest, "additional_kwargs", None)) is not None:
        # Nothing arrived since the last push, whatever this thread's record
        # says: a thread can drop out of the candidates and lose its record.
        last_evaluated_id = last_message_id
    gate = evaluate_gate(
        settings,
        now=now,
        armed=armed,
        has_origin=has_origin,
        in_flight=in_flight,
        last_activity=last_activity,
        last_message_id=last_message_id,
        last_evaluated_id=last_evaluated_id,
    )
    if not gate.should_run:
        logger.debug(
            "[proactive check] thread %s gated: %s", source_thread_id, gate.reason
        )
        return ProactiveDecision("gate_rejected", gate.reason, source_thread_id)

    reply = await shadow_runner(source_messages, trigger)
    current_head = read_current_head()
    if inspect.isawaitable(current_head):
        current_head = await current_head
    decision = decide_commit(
        reply,
        source_thread_id=source_thread_id,
        pre_head=pre_head,
        current_head=current_head,
        proactive_id=proactive_id,
    )
    stage = {"skip": "no_push", "stale": "stale", "push": "decided"}[decision.action]
    return ProactiveDecision(
        stage,
        decision.reason,
        source_thread_id,
        proactive_id=proactive_id if decision.message is not None else None,
        message=decision.message,
        pre_head=pre_head,
        evaluated_message_id=None if stage == "stale" else last_message_id,
    )
