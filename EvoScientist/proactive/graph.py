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
from .gate import ProactiveGateSettings
from .service import ProactiveDecision, decide_proactive_push

logger = logging.getLogger(__name__)


class ProactiveTickState(TypedDict, total=False):
    candidates: list[dict[str, Any]]
    decisions: list[dict[str, Any]]
    evaluated: dict[str, str]
    last_tick: list[dict[str, Any]]


@dataclass
class _Deps:
    """What a tick needs, rebuilt when the config it came from changes."""

    key: tuple[Any, ...]
    cfg: Any
    settings: ProactiveGateSettings
    url: str
    shadows: dict[tuple[str, str | None], Any] = field(default_factory=dict)


def _gate_settings(cfg: Any) -> ProactiveGateSettings:
    from ..langgraph_dev.sdk import default_scheduler_timezone

    return ProactiveGateSettings(
        idle_minutes=int(cfg.proactive_idle_minutes),
        quiet_hours=cfg.proactive_quiet_hours,
        timezone=cfg.proactive_timezone or default_scheduler_timezone(cfg) or "UTC",
    )


def _load_deps(previous: _Deps | None) -> _Deps:
    """Read the effective config; keep *previous* while nothing it uses changed.

    Blocking (reads config files): runs in a worker thread. Re-read every tick,
    so a model or setting change applies without a server restart.
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
) -> ProactiveDecision:
    """Read one candidate thread from this server and decide for it."""
    from ..gateway.server import _messages_from_state, _state_interrupts

    thread = await client.threads.get(thread_id)
    metadata = thread.get("metadata") or {}
    dirs = await asyncio.to_thread(
        SessionDirs.from_stored, metadata.get("workspace_dir"), metadata.get("run_dir")
    )
    if dirs is None or dirs.workspace.key != workspace.key:
        # A thread of another workspace never gets this workspace's graph.
        return ProactiveDecision("gate_rejected", "other_workspace", thread_id)

    state = await client.threads.get_state(thread_id)
    status = thread.get("status")
    in_flight = status in ("busy", "interrupted") or bool(_state_interrupts(state))
    shadow = await asyncio.to_thread(_build_shadow, deps, dirs)

    async def shadow_runner(messages: list[Any], trigger: str) -> str | None:
        from .shadow import run_shadow_turn

        return await run_shadow_turn(shadow, messages, trigger)

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
        source_messages=_messages_from_state(state),
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
        candidates = [
            str(c["thread_id"])
            for c in state.get("candidates") or []
            if isinstance(c, dict) and c.get("thread_id")
        ]
        decisions: list[dict[str, Any]] = []
        evaluated: dict[str, str] = {}
        last_tick: list[dict[str, Any]] = []
        for thread_id in candidates:
            try:
                decision = await _check_thread(
                    client,
                    deps,
                    workspace,
                    thread_id,
                    now=now,
                    last_evaluated_id=evaluated_before.get(thread_id),
                )
            except NotFoundError:
                last_tick.append(
                    {"thread_id": thread_id, "stage": "gone", "reason": ""}
                )
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
            if decision.stage == "decided":
                decisions.append(_decision_record(decision, now))
        logger.info("[proactive] tick: %s", last_tick)
        return {"decisions": decisions, "evaluated": evaluated, "last_tick": last_tick}

    builder = StateGraph(ProactiveTickState)
    builder.add_node("tick", tick)
    builder.add_edge(START, "tick")
    builder.add_edge("tick", END)
    return builder.compile()
