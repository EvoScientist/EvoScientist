"""Proactive check orchestrator (issue #263).

Ties the pieces together for one proactive check: gate → shadow turn → commit
decision → deliver. Thread reads are done by the caller and passed in as plain
values plus a ``read_current_head`` callable for the post-shadow stale re-read,
so the check stays pure sequencing and fully unit-testable without a database.
Delivery is abstracted behind the ``Deliverer`` protocol; in production the
``GatewayDeliverer`` commits the push into the source thread.

``run_proactive_scan`` runs one check per candidate thread and
``run_proactive_tick`` is the full cron tick (enumerate candidates, then scan).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from .commit import commit_with_conflict_retry, decide_commit, resolve_head
from .eligibility import list_channel_thread_candidates
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

# In-memory log of recent check outcomes (observability; bounded).
_LEDGER: deque[ProactiveCheckResult] = deque(maxlen=200)


async def _conflict_backoff(attempt: int) -> None:
    """Wait before re-trying a commit that hit a 409: 1s, 2s, 4s, capped at 8s.

    A user run in flight on the source thread needs real time to finish; without
    a delay the retries collapse into consecutive HTTP round trips.
    """
    await asyncio.sleep(min(2**attempt, 8))


class _DeliveryFailed(Exception):
    """A deliverer returned False (hard delivery failure, not a conflict).

    Raised inside the ``commit_fn`` adapter so it propagates past
    ``commit_with_conflict_retry`` (which only retries ``ConflictError``) and is
    mapped to a ``delivery_failed`` outcome instead of being retried.
    """


class Deliverer(Protocol):
    """Sends a decided proactive reply to a fixed, explicit destination."""

    async def deliver(
        self,
        source_thread_id: str,
        reply: str,
        proactive_id: str,
        *,
        source_messages: list | None = None,
    ) -> bool: ...


@dataclass(frozen=True)
class ProactiveCheckResult:
    """Outcome of one proactive check.

    ``stage`` is one of ``gate_rejected`` / ``no_push`` / ``stale`` /
    ``delivered`` / ``delivery_failed`` / ``conflict_exhausted`` (a user run
    stayed in flight across every commit retry → push dropped) / ``scan_error``
    (a per-thread read/check raised during a scan — isolated, tick continued).
    """

    stage: str
    reason: str
    reply: str | None = None
    delivered: bool = False
    source_thread_id: str | None = None


def get_ledger() -> list[ProactiveCheckResult]:
    """Return a snapshot of the in-memory decision ledger (newest last)."""
    return list(_LEDGER)


async def run_proactive_check(
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
    deliverer: Deliverer,
    trigger: str,
    workspace_dir: str | None,
    model: str | None,
    proactive_id: str,
) -> ProactiveCheckResult:
    """Run gate → shadow → commit-decision → deliver for one source thread."""

    def finish(stage, reason, *, reply=None, delivered=False):
        return _record(
            ProactiveCheckResult(
                stage,
                reason,
                reply=reply,
                delivered=delivered,
                source_thread_id=source_thread_id,
            )
        )

    gate = evaluate_gate(
        settings,
        now=now,
        last_activity=last_activity,
        has_origin=has_origin,
        in_flight=in_flight,
    )
    if not gate.should_run:
        return finish("gate_rejected", gate.reason)

    reply = await shadow_runner(source_messages, trigger)

    decision = decide_commit(
        reply,
        source_thread_id=source_thread_id,
        workspace_dir=workspace_dir,
        model=model,
        pre_head=pre_head,
        current_head=await resolve_head(read_current_head),
        proactive_id=proactive_id,
    )
    if decision.action == "skip":
        return finish("no_push", decision.reason, reply=reply)
    if decision.action == "stale":
        return finish("stale", decision.reason, reply=reply)

    # action == "would_commit": commit-and-deliver with conflict handling. The
    # commit is delegated to the deliverer; wrapping it as a commit_fn lets
    # commit_with_conflict_retry own the during-commit stale re-check and the
    # ConflictError defer-and-retry (a gateway-backed deliverer raises it when a
    # user run is in flight on the source thread).
    async def _commit(_message, _metadata, _as_node):
        # The deliverer rebuilds the tagged AIMessage from reply/proactive_id, so
        # the decided message/metadata/as_node are not needed here. A False
        # return is a hard delivery failure, not a conflict, so surface it past
        # the retry loop.
        if not await deliverer.deliver(
            source_thread_id, reply, proactive_id, source_messages=source_messages
        ):
            raise _DeliveryFailed

    try:
        outcome = await commit_with_conflict_retry(
            decision,
            source_thread_id=source_thread_id,
            commit_fn=_commit,
            read_current_head=read_current_head,
            pre_head=pre_head,
            backoff=_conflict_backoff,
        )
    except _DeliveryFailed:
        return finish("delivery_failed", "deliver_returned_false", reply=reply)
    except Exception:
        logger.warning("proactive delivery raised", exc_info=True)
        return finish("delivery_failed", "exception", reply=reply)

    if outcome.status == "stale":
        return finish("stale", outcome.reason, reply=reply)
    if outcome.status == "conflict_exhausted":
        return finish("conflict_exhausted", outcome.reason, reply=reply)
    logger.info(
        "proactive check delivered for source %s (proactive_id=%s)",
        source_thread_id,
        proactive_id,
    )
    return finish("delivered", "ok", reply=reply, delivered=True)


@dataclass(frozen=True)
class SourceRead:
    """Per-thread inputs a scan gathers before running one check.

    Populated server-side (SDK reads: thread messages, head token, in-flight run,
    origin presence) in production; injected in tests. ``read_current_head`` is the
    post-shadow stale re-read the check calls after generation.
    """

    messages: list
    pre_head: str | None
    has_origin: bool
    in_flight: bool
    read_current_head: Callable[[], str | Awaitable[str | None] | None]


async def run_proactive_scan(
    candidates: list[tuple[str, datetime | None]],
    *,
    now: datetime,
    settings: ProactiveGateSettings,
    read_source: Callable[[str], Awaitable[SourceRead]],
    shadow_runner: Callable[[list, str], Awaitable[str | None]],
    deliverer: Deliverer,
    trigger: str,
    workspace_dir: str | None,
    model: str | None,
    new_proactive_id: Callable[[], str] | None = None,
) -> list[ProactiveCheckResult]:
    """Run a proactive check on each candidate thread — one cron tick.

    ``candidates`` is ``(thread_id, last_activity)`` from
    ``eligibility.list_channel_thread_candidates``. For each, gathers per-thread
    inputs via ``read_source`` and runs :func:`run_proactive_check` with the shared
    shadow/deliverer/settings, minting a fresh ``proactive_id`` per thread. The gate
    (idle/quiet-hours/origin/in-flight) stays authoritative inside each check — the
    scan only sequences candidates and threads their ``last_activity`` through.

    Per-candidate failures are isolated: a raised ``read_source`` or check is logged
    and recorded as a ``scan_error`` result, and the tick continues — one bad thread
    never aborts the whole scan.
    """
    gen_id = new_proactive_id or (lambda: uuid.uuid4().hex)
    results: list[ProactiveCheckResult] = []
    for thread_id, last_activity in candidates:
        try:
            src = await read_source(thread_id)
            result = await run_proactive_check(
                settings=settings,
                source_thread_id=thread_id,
                now=now,
                last_activity=last_activity,
                pre_head=src.pre_head,
                has_origin=src.has_origin,
                in_flight=src.in_flight,
                source_messages=src.messages,
                shadow_runner=shadow_runner,
                read_current_head=src.read_current_head,
                deliverer=deliverer,
                trigger=trigger,
                workspace_dir=workspace_dir,
                model=model,
                proactive_id=gen_id(),
            )
        except Exception:
            logger.warning(
                "proactive scan: check raised for thread %s", thread_id, exc_info=True
            )
            result = _record(
                ProactiveCheckResult(
                    "scan_error", "exception", source_thread_id=thread_id
                )
            )
        results.append(result)
    return results


def make_read_source(client: Any) -> Callable[[str], Awaitable[SourceRead]]:
    """Build a server-backed ``read_source`` for :func:`run_proactive_scan`.

    Reads thread state via ``client.threads.get_state`` (shapes verified against a
    live langgraph-dev): ``messages`` from ``values.messages``, the head token from
    ``checkpoint_id`` (falling back to ``checkpoint.checkpoint_id``). The read is
    awaited (the tick runs on langgraph-dev's event loop, which forbids blocking
    calls — an async client is required; a sync/awaitable result is handled either
    way). Candidates are already pre-filtered to idle channel threads by
    :func:`~EvoScientist.proactive.eligibility.list_channel_thread_candidates`
    (``status="idle"`` + the channel marker), so ``has_origin=True`` and
    ``in_flight=False`` by construction.

    ``read_current_head`` re-fetches the thread's head (``get_state`` →
    ``checkpoint_id``) each time it is called, so the pre-commit stale check sees a
    user turn or another tick's push that landed while the shadow was running and
    discards the now-stale reply instead of appending after it.
    """

    async def _fetch_state(thread_id: str) -> Any:
        result = client.threads.get_state(thread_id)
        return await result if isinstance(result, Awaitable) else result

    async def _read(thread_id: str) -> SourceRead:
        state = await _fetch_state(thread_id)

        async def _current_head() -> str | None:
            return _state_checkpoint_id(await _fetch_state(thread_id))

        return SourceRead(
            messages=_state_messages(state),
            pre_head=_state_checkpoint_id(state),
            has_origin=True,
            in_flight=False,
            read_current_head=_current_head,
        )

    return _read


def _state_messages(state: Any) -> list:
    values = state.get("values") if isinstance(state, dict) else None
    messages = values.get("messages") if isinstance(values, dict) else None
    return messages if isinstance(messages, list) else []


def _state_checkpoint_id(state: Any) -> str | None:
    if not isinstance(state, dict):
        return None
    checkpoint_id = state.get("checkpoint_id")
    if checkpoint_id:
        return str(checkpoint_id)
    checkpoint = state.get("checkpoint")
    if isinstance(checkpoint, dict) and checkpoint.get("checkpoint_id"):
        return str(checkpoint["checkpoint_id"])
    return None


async def run_proactive_tick(
    client: Any,
    *,
    now: datetime,
    settings: ProactiveGateSettings,
    read_source: Callable[[str], Awaitable[SourceRead]],
    shadow_runner: Callable[[list, str], Awaitable[str | None]],
    deliverer: Deliverer,
    trigger: str,
    workspace_dir: str | None,
    model: str | None,
    marker: dict[str, Any] | None = None,
    limit: int = 50,
    new_proactive_id: Callable[[], str] | None = None,
) -> list[ProactiveCheckResult]:
    """One proactive cron tick: enumerate server-registered channel-thread
    candidates, then run a check on each.

    Composes the two halves — ``eligibility.list_channel_thread_candidates`` (the
    server-scoped, always-server-born candidate set) and :func:`run_proactive_scan`
    — leaving the server-backed ``read_source`` / ``shadow_runner`` / ``deliverer``
    as injected seams. This is the body the ``proactive`` graph node calls.
    """
    candidates = await list_channel_thread_candidates(
        client, marker=marker, limit=limit
    )
    logger.info("proactive tick: %d candidate thread(s)", len(candidates))
    return await run_proactive_scan(
        candidates,
        now=now,
        settings=settings,
        read_source=read_source,
        shadow_runner=shadow_runner,
        deliverer=deliverer,
        trigger=trigger,
        workspace_dir=workspace_dir,
        model=model,
        new_proactive_id=new_proactive_id,
    )


def _record(result: ProactiveCheckResult) -> ProactiveCheckResult:
    _LEDGER.append(result)
    logger.debug(
        "proactive check result: stage=%s reason=%s", result.stage, result.reason
    )
    return result
