"""Serve-driven proactive tick + apply (issue #263, Shape B).

Serve runs the proactive check itself: it enumerates the threads that have a
remembered channel origin, reads each thread's state from the langgraph server,
runs the tool-stripped shadow, and decides (:func:`decide_proactive_push`). The
decided pushes are handed back to serve, which applies them in its single
serialized consumer — appending the tagged ``AIMessage`` as the ``model`` node
then clearing the pending ``next`` — so no out-of-band writer races a user run
(the Shape-B property that removes the 409/conflict-retry machinery).

Reads and the apply use the langgraph SDK's ``client.threads`` surface directly
(``update_state`` accepts ``as_node`` natively), so no gateway widening is
needed.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from langgraph_sdk.errors import NotFoundError

from ..gateway.server import _messages_from_state, _state_interrupts
from .gate import ProactiveGateSettings
from .service import ProactiveDecision, decide_proactive_push

logger = logging.getLogger(__name__)


def _host_tz_name() -> str:
    """The host's IANA timezone name, or 'UTC' when it cannot be resolved."""
    try:
        import tzlocal

        return str(tzlocal.get_localzone_name() or "UTC")
    except Exception:
        return "UTC"


def gate_settings_from_config(cfg: Any) -> ProactiveGateSettings:
    """Build gate settings from the effective config (empty tz -> host zone)."""
    return ProactiveGateSettings(
        enabled=bool(getattr(cfg, "proactive_enabled", False)),
        idle_minutes=int(getattr(cfg, "proactive_idle_minutes", 120)),
        quiet_hours=(getattr(cfg, "proactive_quiet_hours", "") or None),
        timezone=(getattr(cfg, "proactive_timezone", "") or _host_tz_name()),
    )


def _parse_iso_utc(value: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp to a UTC-aware datetime, or None."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return (
        parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
    )


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


@dataclass(frozen=True)
class _SourceRead:
    messages: list
    pre_head: str | None
    last_activity: datetime | None
    in_flight: bool


async def _read_source(client: Any, thread_id: str) -> _SourceRead:
    """Read a source thread's messages, head, last activity, and in-flight state."""
    state = await client.threads.get_state(thread_id)
    thread = await client.threads.get(thread_id)
    status = thread.get("status") if isinstance(thread, dict) else None
    last_activity = _parse_iso_utc(
        thread.get("updated_at") if isinstance(thread, dict) else None
    )
    # A ``busy`` thread is executing; an ``interrupted`` status or a pending
    # interrupt means it is parked awaiting a human decision. Treat all of them
    # as in-flight so the gate never lets the apply overwrite that state (the
    # append would drop a pending HITL interrupt — the exact call the server-side
    # repair path refuses to make). ``_messages_from_state`` is the canonical
    # reader: it applies summarization/compaction so the shadow reasons over the
    # same history the real agent would, not the raw pre-compaction wire dicts.
    parked = status in ("busy", "interrupted") or bool(_state_interrupts(state))
    return _SourceRead(
        messages=_messages_from_state(state),
        pre_head=_state_checkpoint_id(state),
        last_activity=last_activity,
        in_flight=parked,
    )


async def run_serve_proactive_tick(
    client: Any,
    *,
    candidate_ids: list[str],
    now: datetime,
    settings: ProactiveGateSettings,
    shadow_runner: Callable[[list, str], Awaitable[str | None]],
    trigger: str,
    workspace_dir: str | None,
    model: str | None,
    gen_id: Callable[[], str],
    origin_present: Callable[[str], bool],
) -> list[ProactiveDecision]:
    """Decide a proactive push for each candidate thread (no apply).

    For each candidate: read the source thread, then run
    :func:`decide_proactive_push` (gate -> shadow -> decision) with a fresh
    ``proactive_id``. ``origin_present`` (injected so this module stays free of
    ``cli.channel``) reports whether the thread still has a channel origin at tick
    time — a thread whose origin was forgotten between listing and now gates out
    as ``no_origin``. A candidate that no longer exists (e.g. ``/delete`` left the
    origin behind) is skipped quietly. Other per-thread failures are isolated and
    logged so one bad thread never aborts the tick.
    """
    decisions: list[ProactiveDecision] = []
    for thread_id in candidate_ids:
        try:
            src = await _read_source(client, thread_id)

            async def _current_head(_tid: str = thread_id) -> str | None:
                return _state_checkpoint_id(await client.threads.get_state(_tid))

            decision = await decide_proactive_push(
                settings=settings,
                source_thread_id=thread_id,
                now=now,
                last_activity=src.last_activity,
                pre_head=src.pre_head,
                has_origin=origin_present(thread_id),
                in_flight=src.in_flight,
                source_messages=src.messages,
                shadow_runner=shadow_runner,
                read_current_head=_current_head,
                trigger=trigger,
                workspace_dir=workspace_dir,
                model=model,
                proactive_id=gen_id(),
            )
        except NotFoundError:
            # The thread was deleted (its channel origin lingers); skip quietly
            # rather than logging a traceback for it every tick.
            logger.debug(
                "proactive tick: thread %s no longer exists; skipping", thread_id
            )
            continue
        except Exception:
            logger.warning(
                "proactive tick: check raised for thread %s", thread_id, exc_info=True
            )
            continue
        decisions.append(decision)
    return decisions


async def apply_proactive_decision(
    client: Any,
    decision: ProactiveDecision,
    *,
    publish: Callable[[str, str], bool],
    origin_present: Callable[[str], bool],
) -> bool:
    """Apply a decided proactive push in-process, then deliver it.

    Must be called from serve's serialized consumer so no user run races the
    write. Three guards run against the state fetched here, in order, and any one
    drops the push without writing:

    - ``origin_present`` is false — the channel origin was forgotten during the
      shadow (a ``/new``), so there is nothing to deliver to; writing would leave
      a phantom assistant message in real thread history that the user never saw.
    - the head moved past ``decision.head`` — a user turn landed, so the reply is
      stale and must not be appended after a fresh turn.
    - the thread is parked on a human-in-the-loop interrupt — the append would
      drop that pending interrupt (the server-side repair path refuses the same
      write for exactly this reason).

    Otherwise it appends the tagged ``AIMessage`` as the ``model`` node. No
    separate ``as_node=END`` clear follows: a plain assistant message has no
    ``tool_calls``, so ``model`` routes straight to ``END`` and ``next`` is
    already empty — and that clear is the call that would clobber a parked
    interrupt. Returns whether delivery was scheduled; a write that could not be
    delivered is logged at WARNING so the phantom message is findable.
    """
    if decision.stage != "decided" or decision.message is None:
        return False
    thread_id = decision.source_thread_id
    if not thread_id:
        return False

    if not origin_present(thread_id):
        logger.info(
            "[proactive apply] skip: thread %s lost its channel origin before "
            "apply — dropping push (nothing written)",
            thread_id,
        )
        return False

    state = await client.threads.get_state(thread_id)
    if _state_checkpoint_id(state) != decision.head:
        logger.info(
            "[proactive apply] skip: thread %s changed before apply "
            "(head moved from %s) — dropping stale push",
            thread_id,
            decision.head,
        )
        return False
    if _state_interrupts(state):
        logger.info(
            "[proactive apply] skip: thread %s parked on a human-in-the-loop "
            "interrupt — dropping push (would clobber the pending interrupt)",
            thread_id,
        )
        return False

    await client.threads.update_state(
        thread_id, {"messages": [decision.message]}, as_node="model"
    )
    delivered = publish(thread_id, decision.reply or "")
    if delivered:
        logger.info(
            "[proactive apply] appended and delivered push to thread %s "
            "(proactive_id=%s)",
            thread_id,
            decision.proactive_id,
        )
    else:
        logger.warning(
            "[proactive apply] wrote push to thread %s (proactive_id=%s) but "
            "delivery was NOT scheduled (origin/bus missing) — the message is in "
            "thread history undelivered",
            thread_id,
            decision.proactive_id,
        )
    return delivered
