"""Tests for ``EvoScientist._maybe_swap_async_subagents``.

Covers the fallback / swap / strip-internal-flag paths that decide whether
sub-agents are routed in-process (sync ``task`` tool) or to the langgraph
dev subprocess (``AsyncSubAgent`` over HTTP).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from EvoScientist.EvoScientist import _maybe_swap_async_subagents


def _sub(name: str, *, async_flag: bool, description: str = "desc") -> dict:
    """Build a sub-agent dict shaped like ``utils.load_subagents`` output."""
    return {
        "name": name,
        "description": description,
        "system_prompt": "x",
        "tools": [],
        "_async": async_flag,
    }


# =============================================================================
# Async disabled in config → return unchanged
# =============================================================================


def test_returns_unchanged_when_async_disabled_and_strips_flag():
    """No swap when config.enable_async_subagents is False (the default).

    Even in this disabled-path, the internal ``_async`` flag must be stripped
    before sub-agents reach deepagents (which may schema-validate the dicts).
    """
    cfg = SimpleNamespace(enable_async_subagents=False)
    subs = [
        _sub("planner-agent", async_flag=False),
        _sub("writing-agent", async_flag=True),
    ]
    with patch("EvoScientist.EvoScientist._ensure_config", return_value=cfg):
        out = _maybe_swap_async_subagents(subs)
    assert out is subs
    for s in out:
        assert "_async" not in s, f"_async leaked into {s['name']}"


# =============================================================================
# Async enabled but langgraph dev unreachable → strip flag, return as sync
# =============================================================================


class TestFallbackPath:
    def _setup(self):
        return SimpleNamespace(
            enable_async_subagents=True,
            langgraph_dev_port=6174,
        )

    def test_returns_subs_unchanged(self):
        cfg = self._setup()
        subs = [
            _sub("planner-agent", async_flag=False),
            _sub("writing-agent", async_flag=True),
        ]
        with (
            patch("EvoScientist.EvoScientist._ensure_config", return_value=cfg),
            patch(
                "EvoScientist.langgraph_dev.manager.is_async_subagents_available",
                return_value=False,
            ),
        ):
            out = _maybe_swap_async_subagents(subs)
        assert out is subs

    def test_strips_async_flag_from_all_subs(self):
        """Even fallback path must strip _async before deepagents handoff."""
        cfg = self._setup()
        subs = [
            _sub("planner-agent", async_flag=False),
            _sub("writing-agent", async_flag=True),
        ]
        with (
            patch("EvoScientist.EvoScientist._ensure_config", return_value=cfg),
            patch(
                "EvoScientist.langgraph_dev.manager.is_async_subagents_available",
                return_value=False,
            ),
        ):
            out = _maybe_swap_async_subagents(subs)
        for s in out:
            assert "_async" not in s, f"_async leaked into {s['name']}"


# =============================================================================
# Async enabled + reachable + nothing flagged async → return all as sync (stripped)
# =============================================================================


def test_no_async_flagged_subs_strips_and_returns():
    cfg = SimpleNamespace(enable_async_subagents=True, langgraph_dev_port=6174)
    subs = [
        _sub("planner-agent", async_flag=False),
        _sub("research-agent", async_flag=False),
    ]
    with (
        patch("EvoScientist.EvoScientist._ensure_config", return_value=cfg),
        patch(
            "EvoScientist.langgraph_dev.manager.is_async_subagents_available",
            return_value=True,
        ),
    ):
        out = _maybe_swap_async_subagents(subs)
    assert out is subs  # nothing to swap, returned as-is
    for s in out:
        assert "_async" not in s


# =============================================================================
# Async enabled + reachable + has async-flagged subs → swap to AsyncSubAgent
# =============================================================================


def test_swaps_async_flagged_subs():
    cfg = SimpleNamespace(enable_async_subagents=True, langgraph_dev_port=6174)
    subs = [
        _sub("planner-agent", async_flag=False, description="plan"),
        _sub("writing-agent", async_flag=True, description="write report"),
        _sub("data-analysis-agent", async_flag=True, description="analyze"),
    ]
    with (
        patch("EvoScientist.EvoScientist._ensure_config", return_value=cfg),
        patch(
            "EvoScientist.langgraph_dev.manager.is_async_subagents_available",
            return_value=True,
        ),
    ):
        out = _maybe_swap_async_subagents(subs)

    assert len(out) == 3

    # Sync sub kept as a plain dict, _async stripped.
    by_name = {s["name"]: s for s in out}
    planner = by_name["planner-agent"]
    assert isinstance(planner, dict)
    assert "_async" not in planner

    # Async subs are AsyncSubAgent specs (TypedDict) pointing at the right URL.
    writing = by_name["writing-agent"]
    assert writing["graph_id"] == "writing-agent"
    assert writing["url"] == "http://127.0.0.1:6174"
    assert writing["description"] == "write report"

    data = by_name["data-analysis-agent"]
    assert data["graph_id"] == "data-analysis-agent"
    assert data["url"] == "http://127.0.0.1:6174"


def test_swap_uses_configured_port():
    """AsyncSubAgent.url should reflect cfg.langgraph_dev_port, not hardcoded."""
    cfg = SimpleNamespace(enable_async_subagents=True, langgraph_dev_port=9999)
    subs = [_sub("writing-agent", async_flag=True)]
    with (
        patch("EvoScientist.EvoScientist._ensure_config", return_value=cfg),
        patch(
            "EvoScientist.langgraph_dev.manager.is_async_subagents_available",
            return_value=True,
        ),
    ):
        out = _maybe_swap_async_subagents(subs)
    assert out[0]["url"] == "http://127.0.0.1:9999"


def test_swap_uses_configured_host():
    """A backend pinned to one interface can't be self-dispatched over
    loopback, so the URL must track cfg.langgraph_dev_host too."""
    cfg = SimpleNamespace(
        enable_async_subagents=True,
        langgraph_dev_port=6174,
        langgraph_dev_host="192.168.1.5",
    )
    subs = [_sub("writing-agent", async_flag=True)]
    with (
        patch("EvoScientist.EvoScientist._ensure_config", return_value=cfg),
        patch(
            "EvoScientist.langgraph_dev.manager.is_async_subagents_available",
            return_value=True,
        ),
    ):
        out = _maybe_swap_async_subagents(subs)
    assert out[0]["url"] == "http://192.168.1.5:6174"


def test_swap_maps_wildcard_host_to_loopback():
    """A 0.0.0.0 bind includes loopback, and connecting *to* 0.0.0.0 is
    rejected outright on Windows — the client URL must collapse to
    127.0.0.1 rather than echo the bind address back."""
    cfg = SimpleNamespace(
        enable_async_subagents=True,
        langgraph_dev_port=6174,
        langgraph_dev_host="0.0.0.0",
    )
    subs = [_sub("writing-agent", async_flag=True)]
    with (
        patch("EvoScientist.EvoScientist._ensure_config", return_value=cfg),
        patch(
            "EvoScientist.langgraph_dev.manager.is_async_subagents_available",
            return_value=True,
        ),
    ):
        out = _maybe_swap_async_subagents(subs)
    assert out[0]["url"] == "http://127.0.0.1:6174"


# =============================================================================
# AsyncWatcherMiddleware appended on swap
# =============================================================================


def test_maybe_swap_appends_watcher_middleware_when_enabled():
    """When async subagents are swapped, AsyncWatcherMiddleware must be appended."""
    from EvoScientist.middleware.async_watcher import AsyncWatcherMiddleware

    cfg = SimpleNamespace(enable_async_subagents=True, langgraph_dev_port=6174)

    middleware: list = []
    with (
        patch("EvoScientist.EvoScientist._ensure_config", return_value=cfg),
        patch(
            "EvoScientist.langgraph_dev.manager.is_async_subagents_available",
            return_value=True,
        ),
    ):
        subs = [_sub("writing-agent", async_flag=True)]
        _maybe_swap_async_subagents(subs, middleware)

    assert len(middleware) == 1
    assert isinstance(middleware[0], AsyncWatcherMiddleware)


def test_maybe_swap_skips_middleware_when_no_async_flagged():
    """Without any _async-flagged subagents, no middleware is appended."""
    cfg = SimpleNamespace(enable_async_subagents=True, langgraph_dev_port=6174)

    middleware: list = []
    with (
        patch("EvoScientist.EvoScientist._ensure_config", return_value=cfg),
        patch(
            "EvoScientist.langgraph_dev.manager.is_async_subagents_available",
            return_value=True,
        ),
    ):
        subs = [_sub("planner-agent", async_flag=False)]
        _maybe_swap_async_subagents(subs, middleware)

    assert middleware == []


def test_maybe_swap_skips_middleware_when_langgraph_unreachable():
    """Fallback path must not append middleware (no watchers will fire)."""
    cfg = SimpleNamespace(enable_async_subagents=True, langgraph_dev_port=6174)

    middleware: list = []
    with (
        patch("EvoScientist.EvoScientist._ensure_config", return_value=cfg),
        patch(
            "EvoScientist.langgraph_dev.manager.is_async_subagents_available",
            return_value=False,
        ),
    ):
        subs = [_sub("writing-agent", async_flag=True)]
        _maybe_swap_async_subagents(subs, middleware)

    assert middleware == []


def test_maybe_swap_no_middleware_arg_does_not_crash():
    """Backward-compat: middleware parameter is optional."""
    cfg = SimpleNamespace(enable_async_subagents=True, langgraph_dev_port=6174)

    with (
        patch("EvoScientist.EvoScientist._ensure_config", return_value=cfg),
        patch(
            "EvoScientist.langgraph_dev.manager.is_async_subagents_available",
            return_value=True,
        ),
    ):
        subs = [_sub("writing-agent", async_flag=True)]
        out = _maybe_swap_async_subagents(subs)

    assert len(out) == 1
    assert out[0]["graph_id"] == "writing-agent"
