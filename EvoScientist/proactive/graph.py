"""The ``proactive`` graph (registered in ``langgraph.json``, fired by a cron).

One cron tick: enumerate server-registered idle channel threads and run a
proactive check on each (gate → tool-stripped shadow → commit into the SOURCE
thread via the gateway; the serve-side ``delivery_watcher`` then pushes it).

``run_proactive_graph_tick`` is the node body with every dependency injected, so
it is unit-testable without a server or model. The langgraph ``StateGraph`` node
that langgraph-dev runs constructs the real deps lazily on the first tick — the
gateway's async SDK client, the server gateway (``create_runtime_gateways(
backend="langgraph_server")``), and a tool-stripped shadow runner (``shadow.py``'s
``build_shadow_graph``/``run_shadow_turn``) — and calls it.

Importing this module is cheap (no MCP/model load), so it is safe for
langgraph-dev to import at startup.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .gate import ProactiveGateSettings
from .gateway_delivery import GatewayDeliverer
from .service import ProactiveCheckResult, make_read_source, run_proactive_tick

logger = logging.getLogger(__name__)


async def run_proactive_graph_tick(
    *,
    client: Any,
    gateway: Any,
    target: Any,
    shadow_runner: Callable[[list, str], Awaitable[str | None]],
    settings: ProactiveGateSettings,
    trigger: str,
    workspace_dir: str | None,
    model: str | None,
    now: datetime | None = None,
    marker: dict[str, Any] | None = None,
    limit: int = 50,
) -> list[ProactiveCheckResult]:
    """Run one proactive tick with the server adapters wired in.

    Composes ``make_read_source`` (server ``get_state`` reads), ``GatewayDeliverer``
    (commit into the source thread via ``gateway``), and ``run_proactive_tick``
    (enumerate idle marked channel threads → per-thread gate/shadow/commit).
    ``client`` is the sync langgraph SDK client (enumeration + state reads);
    ``gateway``/``target`` are the server gateway used for the commit.
    """
    return await run_proactive_tick(
        client,
        now=now or datetime.now(UTC),
        settings=settings,
        read_source=make_read_source(client),
        shadow_runner=shadow_runner,
        deliverer=GatewayDeliverer(gateway, target),
        trigger=trigger,
        workspace_dir=workspace_dir,
        model=model,
        marker=marker,
        limit=limit,
    )


# ---------------------------------------------------------------------------
# The langgraph ``proactive`` graph (registered in langgraph.json).
#
# Importing this module + building the graph object is cheap (a one-node
# StateGraph compile, no MCP/model), so langgraph-dev can import it at startup.
# The real dependencies (SDK client, server gateway, in-process tool-stripped
# shadow) are built lazily on the FIRST tick and memoized — never at import.
# ---------------------------------------------------------------------------


class _ProactiveTickState(TypedDict, total=False):
    """Cron-run state. ``messages`` is the (ignored) cron input; ``result`` holds
    one summary dict per checked thread (see :func:`summarize_results`)."""

    messages: list
    result: list


_REPLY_PREVIEW_CHARS = 600


def summarize_results(results: list[ProactiveCheckResult]) -> list[dict[str, Any]]:
    """Per-thread tick summary stored on the cron run's state.

    ``stage``/``reason`` say what happened (``no_push`` with reason ``no_reply``
    is a failed shadow, with reason ``no_push`` the model's own decision);
    ``reply`` is a preview of the shadow's text so a NO_PUSH verdict or a
    delivered push can be inspected from the run without server logs.
    """
    summary: list[dict[str, Any]] = []
    for r in results:
        reply = (r.reply or "").strip()
        summary.append(
            {
                "thread_id": r.source_thread_id,
                "stage": r.stage,
                "reason": r.reason,
                "reply": reply[:_REPLY_PREVIEW_CHARS] if reply else None,
            }
        )
    return summary


@dataclass(frozen=True)
class _ProactiveDeps:
    client: Any
    gateway: Any
    target: Any
    shadow_runner: Callable[[list, str], Awaitable[str | None]]
    settings: ProactiveGateSettings
    trigger: str
    workspace_dir: str | None
    model: str | None


_DEPS: _ProactiveDeps | None = None
# Config fingerprint the memoized deps were built from. The deps are rebuilt when
# it changes, so a ``/model`` switch or a ``proactive_*`` edit in config.yaml
# applies on the next tick instead of after a langgraph-dev restart.
_DEPS_KEY: tuple[Any, ...] | None = None


def _system_timezone() -> str:
    """Best-effort IANA name of the machine's local timezone (falls back to UTC).

    Uses ``tzlocal``, which honours the ``TZ`` env var the langgraph-dev subprocess
    inherits — so the gate's quiet hours are evaluated in the user's actual local
    time rather than a hardcoded UTC. Falls back to UTC if resolution fails.
    """
    try:
        import tzlocal

        return tzlocal.get_localzone_name()
    except Exception:
        logger.warning(
            "[proactive graph] could not resolve system timezone; using UTC",
            exc_info=True,
        )
        return "UTC"


def _resolve_timezone(configured: str | None) -> str:
    """Resolve the gate timezone: the configured IANA name, else the system tz.

    Validates the result against ``zoneinfo`` (the gate builds a ``ZoneInfo`` from
    it), falling back to UTC on an unknown name so a bad value can never crash a
    tick.
    """
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    name = configured.strip() if configured and configured.strip() else ""
    name = name or _system_timezone()
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        logger.warning(
            "[proactive graph] proactive_timezone=%r is not a valid IANA timezone; "
            "using UTC",
            name,
        )
        return "UTC"
    return name


def gate_settings_from_config(cfg: Any) -> ProactiveGateSettings:
    """Build the gate settings from the ``proactive_*`` config fields.

    ``proactive_quiet_hours`` empty/blank disables the quiet-hours window;
    ``proactive_timezone`` empty resolves to the host's local zone.
    """
    quiet = str(cfg.proactive_quiet_hours or "").strip() or None
    return ProactiveGateSettings(
        enabled=bool(cfg.proactive_enabled),
        idle_minutes=int(cfg.proactive_idle_minutes),
        quiet_hours=quiet,
        timezone=_resolve_timezone(cfg.proactive_timezone),
    )


def _build_deps() -> _ProactiveDeps:
    """Construct (and memoize) the live tick dependencies.

    Heavy — runs on the first tick inside langgraph-dev, not at import. The
    effective config is re-read on every tick (cheap; this runs off-loop) and the
    deps are rebuilt only when the model, provider, workspace, or gate settings
    changed, so config edits take effect without a server restart.
    """
    global _DEPS, _DEPS_KEY

    from .. import paths
    from ..config.settings import get_effective_config
    from ..EvoScientist import _build_chat_model
    from ..gateway import create_runtime_gateways
    from ..gateway.types import GraphTarget
    from ..langgraph_dev.sdk import (
        configured_langgraph_dev_url,
        langgraph_dev_headers,
    )
    from .service import PROACTIVE_TRIGGER
    from .shadow import build_shadow_graph, run_shadow_turn

    cfg = get_effective_config()
    workspace_dir = cfg.default_workdir or str(paths.WORKSPACE_ROOT)
    settings = gate_settings_from_config(cfg)
    key = (cfg.model, cfg.provider, workspace_dir, settings)
    if _DEPS is not None and _DEPS_KEY == key:
        return _DEPS

    url = configured_langgraph_dev_url()
    logger.info(
        "[proactive graph] %s tick deps (model=%s provider=%s workspace=%s "
        "enabled=%s idle_minutes=%d quiet_hours=%s timezone=%s)",
        "rebuilding" if _DEPS is not None else "building",
        cfg.model,
        cfg.provider,
        workspace_dir,
        settings.enabled,
        settings.idle_minutes,
        settings.quiet_hours,
        settings.timezone,
    )

    gateways = create_runtime_gateways(
        backend="langgraph_server", base_url=url, headers=langgraph_dev_headers()
    )
    # Reuse the gateway's ASYNC SDK client for enumeration + state reads: every I/O
    # in the tick runs on langgraph-dev's event loop, which forbids blocking calls
    # (blockbuster), so sync-client reads are out.
    client = gateways.thread_store.client
    shadow_graph = build_shadow_graph(cfg, _build_chat_model(cfg), workspace_dir)

    async def _shadow(messages: list, trigger: str) -> str | None:
        return await run_shadow_turn(shadow_graph, messages, trigger)

    _DEPS = _ProactiveDeps(
        client=client,
        gateway=gateways.graph_gateway,
        target=GraphTarget(),  # server update_state_values ignores target
        shadow_runner=_shadow,
        settings=settings,
        trigger=PROACTIVE_TRIGGER,
        workspace_dir=workspace_dir,
        model=cfg.model,
    )
    _DEPS_KEY = key
    return _DEPS


async def _proactive_tick_node(state: _ProactiveTickState) -> _ProactiveTickState:
    # Building deps loads the in-process shadow agent (create_cli_agent does sync
    # filesystem I/O), which blockbuster forbids on the event loop — build it in a
    # thread. Memoized, so this cost is paid once.
    deps = await asyncio.to_thread(_build_deps)
    results = await run_proactive_graph_tick(
        client=deps.client,
        gateway=deps.gateway,
        target=deps.target,
        shadow_runner=deps.shadow_runner,
        settings=deps.settings,
        trigger=deps.trigger,
        workspace_dir=deps.workspace_dir,
        model=deps.model,
    )
    summary = summarize_results(results)
    logger.info("[proactive graph] tick complete: %s", summary)
    return {"result": summary}


def _build_graph():
    graph = StateGraph(_ProactiveTickState)
    graph.add_node("tick", _proactive_tick_node)
    graph.add_edge(START, "tick")
    graph.add_edge("tick", END)
    return graph.compile()


# Module-level compiled graph langgraph.json points at (cheap; deps stay lazy).
proactive_graph = _build_graph()
