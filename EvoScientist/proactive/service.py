"""Proactive decision orchestrator (issue #263, Shape B).

Composes one proactive decision for a source thread: gate -> tool-stripped
shadow turn -> commit decision. Pure sequencing — thread reads (messages, head)
are done by the caller and passed in as plain values plus a ``read_current_head``
callable for the post-shadow stale re-read, so this stays fully unit-testable
without a database. There is no delivery step here: under Shape B, serve applies
and delivers the decided push in-process (see the serve integration), so this
returns a :class:`ProactiveDecision` carrying the tagged message for serve to
append.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime

from langchain_core.messages import AIMessage

from .commit import decide_commit
from .gate import ProactiveGateSettings, evaluate_gate

logger = logging.getLogger(__name__)

# The internal shadow instruction (canonical text from issue #263). It is fed to
# the shadow turn only and is never written to the source thread.
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
    """Outcome of one proactive check, before serve applies it.

    ``stage`` is one of ``gate_rejected`` / ``no_push`` / ``stale`` / ``decided``.
    For ``decided`` the ``message`` is the tagged ``AIMessage`` serve appends to
    the source thread (as the ``model`` node) and ``proactive_id`` identifies the
    push; every other stage carries no message.
    """

    stage: str
    reason: str
    reply: str | None = None
    message: AIMessage | None = None
    proactive_id: str | None = None
    source_thread_id: str | None = None
    # For a ``decided`` push: the source head the decision is valid against, so
    # serve can re-check at apply time that no user turn has landed since.
    head: str | None = None


async def _resolve_head(
    read_current_head: Callable[[], str | Awaitable[str | None] | None],
) -> str | None:
    """Call ``read_current_head`` and await it if it returned an awaitable."""
    head = read_current_head()
    if inspect.isawaitable(head):
        head = await head
    return head


async def decide_proactive_push(
    *,
    settings: ProactiveGateSettings,
    source_thread_id: str,
    now: datetime,
    last_activity: datetime | None,
    pre_head: str | None,
    has_origin: bool,
    in_flight: bool,
    source_messages: list,
    shadow_runner: Callable[[list, str], Awaitable[str | None]],
    read_current_head: Callable[[], str | Awaitable[str | None] | None],
    trigger: str,
    workspace_dir: str | None,
    model: str | None,
    proactive_id: str,
) -> ProactiveDecision:
    """Run gate -> shadow -> commit-decision for one source thread (no delivery).

    Returns a :class:`ProactiveDecision`; only ``stage == "decided"`` carries a
    ``message`` for serve to append. The gate short-circuits before any model
    call; the shadow reply is judged by :func:`decide_commit`, whose post-shadow
    head re-read (``read_current_head``) discards a reply the source thread
    outgrew while the shadow was generating.
    """
    gate = evaluate_gate(
        settings,
        now=now,
        last_activity=last_activity,
        has_origin=has_origin,
        in_flight=in_flight,
    )
    if not gate.should_run:
        return ProactiveDecision(
            "gate_rejected", gate.reason, source_thread_id=source_thread_id
        )

    reply = await shadow_runner(source_messages, trigger)
    current_head = await _resolve_head(read_current_head)

    decision = decide_commit(
        reply,
        source_thread_id=source_thread_id,
        workspace_dir=workspace_dir,
        model=model,
        pre_head=pre_head,
        current_head=current_head,
        proactive_id=proactive_id,
    )
    if decision.action == "skip":
        return ProactiveDecision(
            "no_push", decision.reason, reply=reply, source_thread_id=source_thread_id
        )
    if decision.action == "stale":
        return ProactiveDecision(
            "stale", decision.reason, reply=reply, source_thread_id=source_thread_id
        )

    logger.info(
        "proactive check decided a push for source %s (proactive_id=%s)",
        source_thread_id,
        proactive_id,
    )
    return ProactiveDecision(
        "decided",
        "ok",
        reply=reply,
        message=decision.message,
        proactive_id=proactive_id,
        source_thread_id=source_thread_id,
        head=current_head,
    )
