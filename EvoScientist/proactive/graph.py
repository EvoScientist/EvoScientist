"""The ``proactive`` graph: one cron tick over the threads serve can deliver to.

The cron's input lists the candidate threads. A tick reads each one from this
server, checks it with :func:`decide_proactive_push`, and records the outcome
in the cron thread's state:

- ``decisions``: the pushes this tick decided, for serve to apply. Each tick
  replaces the previous tick's list, so the record stays small; serve polls
  far more often than the cron fires and skips a decision it already applied.
- ``evaluated``: per candidate thread, the last non-proactive message the
  previous shadow turn saw, so a thread with nothing new is skipped. Threads
  that are no longer candidates drop out.
- ``failures``: per candidate thread whose shadow turn raised, the message it
  failed on, how often, and how many ticks to skip before the next try (1, 2,
  4, ... ticks apart, at most :data:`MAX_BACKOFF_TICKS`). A context overflow
  or a revoked key fails every time, and each try is a model call the size of
  the conversation. A new message, or a turn that returns, clears the record.
- ``last_tick``: thread, stage and reason for every candidate (never text), to
  see why a tick did or did not push.

Building the graph is cheap and happens at import. The config, the client and
the shadow agents are built on the first tick, off the event loop: a graph
that fails at import takes every graph the server loads down with it.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ..paths import SessionDirs, Workspace
from .commit import last_non_proactive_message_id
from .gate import ProactiveGateSettings
from .service import ProactiveDecision, decide_proactive_push

logger = logging.getLogger(__name__)

# Most ticks a failing thread waits between two tries: six hours at the
# cron's ten-minute interval.
MAX_BACKOFF_TICKS = 35


class ProactiveTickState(TypedDict, total=False):
    candidates: list[dict[str, Any]]
    decisions: list[dict[str, Any]]
    evaluated: dict[str, str]
    failures: dict[str, dict[str, Any]]
    last_tick: list[dict[str, Any]]


@dataclass
class _Deps:
    """What a tick needs, rebuilt when the config it came from changes."""

    key: tuple[Any, ...]
    cfg: Any
    settings: ProactiveGateSettings
    url: str
    shadows: dict[tuple[str, str | None], Any] = field(default_factory=dict)


def _host_timezone() -> str:
    try:
        from tzlocal import get_localzone_name

        return get_localzone_name() or "UTC"
    except Exception:
        return "UTC"


def _gate_settings(cfg: Any) -> ProactiveGateSettings:
    return ProactiveGateSettings(
        idle_minutes=int(cfg.proactive_idle_minutes),
        quiet_hours=cfg.proactive_quiet_hours,
        timezone=cfg.proactive_timezone or _host_timezone(),
    )


def _load_deps(previous: _Deps | None) -> _Deps:
    """Read the effective config; keep *previous* while nothing it uses changed.

    Blocking (reads config files): runs in a worker thread. Re-read every tick,
    so a change of model or of a proactive setting applies without a server
    restart. Any other change (MCP servers, skills, middleware options) reaches
    the cached shadow agents only with one of those, or on restart.
    """
    from ..config.settings import get_effective_config
    from ..langgraph_dev.sdk import langgraph_dev_url

    cfg = get_effective_config()
    settings = _gate_settings(cfg)
    url = langgraph_dev_url(cfg)
    key = (cfg.provider, cfg.model, settings, url)
    if previous is not None and previous.key == key:
        return previous
    logger.info(
        "[proactive] tick settings: model=%s/%s idle_minutes=%d quiet_hours=%r "
        "timezone=%s",
        cfg.provider,
        cfg.model,
        settings.idle_minutes,
        settings.quiet_hours,
        settings.timezone,
    )
    return _Deps(key=key, cfg=cfg, settings=settings, url=url)


def _build_shadow(deps: _Deps, dirs: SessionDirs) -> Any:
    """The shadow agent for *dirs*, built once per folder pair and config."""
    cache_key = (dirs.workspace.key, dirs.run_dir.as_posix() if dirs.run_dir else None)
    graph = deps.shadows.get(cache_key)
    if graph is None:
        from ..EvoScientist import _build_chat_model
        from .shadow import build_shadow_graph

        graph = build_shadow_graph(deps.cfg, _build_chat_model(deps.cfg), dirs)
        deps.shadows[cache_key] = graph
    return graph


def _parse_time(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed


class _ShadowTurnFailed(Exception):
    """The shadow turn raised; carries the message it was run on."""

    def __init__(self, message_id: str | None) -> None:
        super().__init__(message_id)
        self.message_id = message_id


def _failure_after(
    record: dict[str, Any] | None, message_id: str | None
) -> dict[str, Any]:
    """The failure record after one more failed try on *message_id*."""
    count = 1
    if record is not None and record.get("message_id") == message_id:
        count = int(record.get("count") or 0) + 1
    skip = min(2 ** (count - 1) - 1, MAX_BACKOFF_TICKS)
    return {"message_id": message_id, "count": count, "skip": skip}


def _checkpoint_id(state: Any) -> str | None:
    checkpoint = state.get("checkpoint") if isinstance(state, dict) else None
    if isinstance(checkpoint, dict) and checkpoint.get("checkpoint_id"):
        return str(checkpoint["checkpoint_id"])
    return None


def _decision_record(decision: ProactiveDecision, now: datetime) -> dict[str, Any]:
    return {
        "proactive_id": decision.proactive_id,
        "thread_id": decision.source_thread_id,
        "head": decision.pre_head,
        "text": decision.message.content if decision.message is not None else None,
        "decided_at": now.isoformat(),
    }


async def _check_thread(
    client: Any,
    deps: _Deps,
    workspace: Workspace,
    thread_id: str,
    *,
    now: datetime,
    last_evaluated_id: str | None,
    failure: dict[str, Any] | None = None,
) -> ProactiveDecision:
    """Read one candidate thread from this server and decide for it.

    A thread whose last shadow turn failed on the same message is skipped as
    ``backoff`` while its *failure* record has ticks left to skip. A shadow
    turn that raises raises :class:`_ShadowTurnFailed`.
    """
    from ..gateway.server import _messages_from_state, _state_interrupts

    thread = await client.threads.get(thread_id)
    metadata = thread.get("metadata") or {}
    dirs = await asyncio.to_thread(
        SessionDirs.from_stored, metadata.get("workspace_dir"), metadata.get("run_dir")
    )
    if dirs is None:
        # No stored folders, or folders that cannot be resolved.
        return ProactiveDecision("gate_rejected", "no_folders", thread_id)
    if dirs.workspace.key != workspace.key:
        # A thread of another workspace never gets this workspace's graph.
        return ProactiveDecision("gate_rejected", "other_workspace", thread_id)

    state = await client.threads.get_state(thread_id)
    status = thread.get("status")
    in_flight = status in ("busy", "interrupted") or bool(_state_interrupts(state))

    source_messages = _messages_from_state(state)
    last_message_id = last_non_proactive_message_id(source_messages)
    if (
        failure is not None
        and failure.get("message_id") == last_message_id
        and int(failure.get("skip") or 0) > 0
    ):
        return ProactiveDecision("gate_rejected", "backoff", thread_id)

    async def shadow_runner(messages: list[Any], trigger: str) -> str | None:
        from .shadow import run_shadow_turn

        # Built here, after the gate passed: a thread the gate skips costs no
        # build, and a failing build cannot hide the gate's reason.
        try:
            shadow = await asyncio.to_thread(_build_shadow, deps, dirs)
            return await run_shadow_turn(shadow, messages, trigger)
        except Exception as exc:
            raise _ShadowTurnFailed(last_message_id) from exc

    async def read_current_head() -> str | None:
        return _checkpoint_id(await client.threads.get_state(thread_id))

    return await decide_proactive_push(
        settings=deps.settings,
        source_thread_id=thread_id,
        now=now,
        armed=metadata.get("proactive") == "on",
        # Serve lists only threads it can deliver to; it checks the origin
        # again when it applies a push.
        has_origin=True,
        in_flight=in_flight,
        last_activity=_parse_time(
            thread.get("state_updated_at") or thread.get("updated_at")
        ),
        last_evaluated_id=last_evaluated_id,
        pre_head=_checkpoint_id(state),
        source_messages=source_messages,
        shadow_runner=shadow_runner,
        read_current_head=read_current_head,
        proactive_id=uuid.uuid4().hex,
    )


def build_proactive_graph(*, workspace: Workspace) -> Any:
    """Compile the ``proactive`` graph for the workspace this server serves."""
    deps_holder: dict[str, Any] = {"deps": None, "client": None}

    async def tick(state: ProactiveTickState) -> ProactiveTickState:
        from langgraph_sdk.errors import NotFoundError

        from ..langgraph_dev.sdk import get_langgraph_async_client

        deps = await asyncio.to_thread(_load_deps, deps_holder["deps"])
        if deps is not deps_holder["deps"]:
            deps_holder["deps"] = deps
            deps_holder["client"] = get_langgraph_async_client(url=deps.url)
        client = deps_holder["client"]

        now = datetime.now(UTC)
        evaluated_before = state.get("evaluated") or {}
        failures_before = state.get("failures") or {}
        candidates = [
            str(c["thread_id"])
            for c in state.get("candidates") or []
            if isinstance(c, dict) and c.get("thread_id")
        ]
        decisions: list[dict[str, Any]] = []
        evaluated: dict[str, str] = {}
        failures: dict[str, dict[str, Any]] = {}
        last_tick: list[dict[str, Any]] = []
        for thread_id in candidates:
            failure = failures_before.get(thread_id)
            try:
                decision = await _check_thread(
                    client,
                    deps,
                    workspace,
                    thread_id,
                    now=now,
                    last_evaluated_id=evaluated_before.get(thread_id),
                    failure=failure,
                )
            except NotFoundError:
                last_tick.append(
                    {"thread_id": thread_id, "stage": "gone", "reason": ""}
                )
                continue
            except _ShadowTurnFailed as exc:
                failures[thread_id] = _failure_after(failure, exc.message_id)
                logger.warning(
                    "[proactive] shadow turn failed for thread %s (%d in a row); "
                    "next try in %d tick(s)",
                    thread_id,
                    failures[thread_id]["count"],
                    failures[thread_id]["skip"] + 1,
                    exc_info=exc.__cause__,
                )
                last_tick.append(
                    {"thread_id": thread_id, "stage": "error", "reason": ""}
                )
                if thread_id in evaluated_before:
                    evaluated[thread_id] = evaluated_before[thread_id]
                continue
            except Exception:
                logger.warning(
                    "[proactive] check failed for thread %s", thread_id, exc_info=True
                )
                last_tick.append(
                    {"thread_id": thread_id, "stage": "error", "reason": ""}
                )
                if thread_id in evaluated_before:
                    evaluated[thread_id] = evaluated_before[thread_id]
                if failure is not None:
                    failures[thread_id] = failure
                continue
            last_tick.append(
                {
                    "thread_id": thread_id,
                    "stage": decision.stage,
                    "reason": decision.reason,
                }
            )
            if decision.evaluated_message_id is not None:
                evaluated[thread_id] = decision.evaluated_message_id
            elif thread_id in evaluated_before:
                evaluated[thread_id] = evaluated_before[thread_id]
            if decision.reason == "backoff" and failure is not None:
                failures[thread_id] = {**failure, "skip": failure["skip"] - 1}
            elif decision.evaluated_message_id is None and failure is not None:
                # Gated or stale: the failed message is still the one to retry.
                failures[thread_id] = failure
            if decision.stage == "decided":
                decisions.append(_decision_record(decision, now))
        logger.info("[proactive] tick: %s", last_tick)
        return {
            "decisions": decisions,
            "evaluated": evaluated,
            "failures": failures,
            "last_tick": last_tick,
        }

    builder = StateGraph(ProactiveTickState)
    builder.add_node("tick", tick)
    builder.add_edge(START, "tick")
    builder.add_edge("tick", END)
    return builder.compile()
