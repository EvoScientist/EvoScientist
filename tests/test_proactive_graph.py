"""Tests for EvoScientist.proactive.graph (graph node body + config resolution).

Fully DI'd — a fake client (search + get_state, real captured shapes), a fake
gateway, a fake shadow. No server or model. Proves the whole server-path tick
composes: enumerate idle marked thread → read state → gate → shadow → commit into
the source thread via the gateway.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

from langgraph.graph import END

from EvoScientist.proactive.gate import ProactiveGateSettings
from EvoScientist.proactive.graph import run_proactive_graph_tick

_NOON = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)


class _FakeThreads:
    def __init__(self, search_result, state):
        self._search = search_result
        self._state = state
        self.search_kwargs = None

    def search(self, **kwargs):
        self.search_kwargs = kwargs

        async def _co():
            return self._search

        return _co()

    def get_state(self, thread_id):
        return self._state


class _FakeClient:
    def __init__(self, search_result, state):
        self.threads = _FakeThreads(search_result, state)


class _FakeGateway:
    def __init__(self):
        self.calls = []

    async def update_state_values(
        self, target, thread_id, values, *, as_node=None, metadata=None
    ):
        self.calls.append(
            {"thread_id": thread_id, "values": values, "as_node": as_node}
        )


def _settings():
    return ProactiveGateSettings(
        enabled=True, idle_minutes=120, quiet_hours=None, timezone="UTC"
    )


def _run(shadow_reply, *, thread_updated="2026-08-20T09:00:00+00:00", head="h1"):
    thread = {"thread_id": "t1", "updated_at": thread_updated}
    state = {
        "checkpoint_id": head,
        "values": {"messages": [{"content": "prior", "type": "human"}]},
    }
    client = _FakeClient([thread], state)
    gw = _FakeGateway()

    async def _shadow(messages, trigger):
        return shadow_reply

    results = asyncio.run(
        run_proactive_graph_tick(
            client=client,
            gateway=gw,
            target="tg",
            shadow_runner=_shadow,
            settings=_settings(),
            trigger="<proactive_trigger>",
            workspace_dir="/ws",
            model="m",
            now=_NOON,
        )
    )
    return results, client, gw


def test_graph_tick_delivers_into_source_thread():
    results, client, gw = _run("Want to continue?")
    assert [r.stage for r in results] == ["delivered"]
    # enumeration used the idle-status + marker query
    assert client.threads.search_kwargs["status"] == "idle"
    # committed into the SOURCE thread (t1): append tagged msg as model + clear
    append = [c for c in gw.calls if c["values"] is not None]
    clear = [c for c in gw.calls if c["values"] is None]
    assert len(append) == 1
    assert len(clear) == 1
    assert append[0]["thread_id"] == "t1"
    msg = append[0]["values"]["messages"][0]
    assert msg.additional_kwargs["evoscientist"]["is_proactive_push"] is True
    assert append[0]["as_node"] == "model"
    assert clear[0]["as_node"] == END


def test_graph_tick_no_push_does_not_commit():
    results, _, gw = _run("NO_PUSH")
    assert [r.stage for r in results] == ["no_push"]
    assert gw.calls == []


def test_graph_tick_gate_rejects_recent_thread():
    # thread active 10 min ago → not idle (idle_minutes=120) → gate_rejected
    results, _, gw = _run("Hi", thread_updated="2026-08-20T11:50:00+00:00")
    assert [r.stage for r in results] == ["gate_rejected"]
    assert gw.calls == []


def test_graph_tick_no_candidates_is_empty():
    client = _FakeClient([], {"checkpoint_id": "h1", "values": {"messages": []}})
    gw = _FakeGateway()

    async def _shadow(messages, trigger):
        return "x"

    results = asyncio.run(
        run_proactive_graph_tick(
            client=client,
            gateway=gw,
            target="tg",
            shadow_runner=_shadow,
            settings=_settings(),
            trigger="<t>",
            workspace_dir="/ws",
            model="m",
            now=_NOON,
        )
    )
    assert results == []
    assert gw.calls == []


# ---- gate settings from config ----


def _cfg(**overrides):
    from EvoScientist.config.settings import EvoScientistConfig

    return EvoScientistConfig(**overrides)


def test_gate_settings_defaults_are_off_and_shipped_values(monkeypatch):
    from EvoScientist.proactive import graph

    monkeypatch.setattr(graph, "_system_timezone", lambda: "Europe/Warsaw")
    settings = graph.gate_settings_from_config(_cfg())
    assert settings.enabled is False
    assert settings.idle_minutes == 120
    assert settings.quiet_hours == "22:00-08:00"
    assert settings.timezone == "Europe/Warsaw"  # empty config => host zone


def test_gate_settings_blank_quiet_hours_disables_window():
    from EvoScientist.proactive import graph

    settings = graph.gate_settings_from_config(
        _cfg(
            proactive_enabled=True, proactive_quiet_hours="  ", proactive_timezone="UTC"
        )
    )
    assert settings.enabled is True
    assert settings.quiet_hours is None
    assert settings.timezone == "UTC"


def test_gate_settings_invalid_timezone_falls_back_to_utc(caplog):
    from EvoScientist.proactive import graph

    with caplog.at_level("WARNING", logger="EvoScientist.proactive.graph"):
        settings = graph.gate_settings_from_config(
            _cfg(proactive_timezone="Mars/Olympus")
        )
    assert settings.timezone == "UTC"
    assert any("Mars/Olympus" in r.message for r in caplog.records)


def test_proactive_env_mappings_cover_every_field():
    from EvoScientist.config.settings import _ENV_MAPPINGS

    assert _ENV_MAPPINGS["proactive_enabled"] == "EVOSCIENTIST_PROACTIVE_ENABLED"
    assert (
        _ENV_MAPPINGS["proactive_idle_minutes"] == "EVOSCIENTIST_PROACTIVE_IDLE_MINUTES"
    )
    assert (
        _ENV_MAPPINGS["proactive_quiet_hours"] == "EVOSCIENTIST_PROACTIVE_QUIET_HOURS"
    )
    assert _ENV_MAPPINGS["proactive_timezone"] == "EVOSCIENTIST_PROACTIVE_TIMEZONE"


# ---- tick summary stored on the run ----


def test_summarize_results_carries_stage_reason_and_reply_preview():
    from EvoScientist.proactive.graph import summarize_results
    from EvoScientist.proactive.service import ProactiveCheckResult

    long_reply = "x" * 900
    out = summarize_results(
        [
            ProactiveCheckResult(
                "no_push", "no_push", reply="NO_PUSH", source_thread_id="a"
            ),
            ProactiveCheckResult(
                "no_push", "no_reply", reply=None, source_thread_id="b"
            ),
            ProactiveCheckResult(
                "delivered", "ok", reply=long_reply, source_thread_id="c"
            ),
        ]
    )
    assert out[0] == {
        "thread_id": "a",
        "stage": "no_push",
        "reason": "no_push",
        "reply": "NO_PUSH",
    }
    assert out[1] == {
        "thread_id": "b",
        "stage": "no_push",
        "reason": "no_reply",
        "reply": None,
    }
    assert out[2]["stage"] == "delivered"
    assert len(out[2]["reply"]) == 600


# ---- deps memoization keyed on config ----


def _patch_deps_builders(monkeypatch, cfg_holder):
    """Stub every heavy builder _build_deps touches; count shadow builds."""
    import EvoScientist.EvoScientist as evo
    import EvoScientist.gateway as gateway_pkg
    import EvoScientist.langgraph_dev.sdk as sdk
    from EvoScientist.config import settings as settings_mod
    from EvoScientist.proactive import graph, shadow

    builds = []
    monkeypatch.setattr(settings_mod, "get_effective_config", lambda: cfg_holder["cfg"])
    monkeypatch.setattr(evo, "_build_chat_model", lambda cfg: object())
    monkeypatch.setattr(
        shadow,
        "build_shadow_graph",
        lambda cfg, m, ws: builds.append(cfg.model) or object(),
    )
    monkeypatch.setattr(
        gateway_pkg,
        "create_runtime_gateways",
        lambda **kw: SimpleNamespace(
            thread_store=SimpleNamespace(client=object()), graph_gateway=object()
        ),
    )
    monkeypatch.setattr(sdk, "configured_langgraph_dev_url", lambda: "http://x")
    monkeypatch.setattr(sdk, "langgraph_dev_headers", lambda: {})
    monkeypatch.setattr(graph, "_DEPS", None)
    monkeypatch.setattr(graph, "_DEPS_KEY", None)
    return builds


def test_build_deps_memoizes_until_config_changes(monkeypatch):
    from EvoScientist.proactive import graph

    holder = {"cfg": _cfg(model="model-a", proactive_timezone="UTC")}
    builds = _patch_deps_builders(monkeypatch, holder)

    first = graph._build_deps()
    again = graph._build_deps()
    assert again is first
    assert builds == ["model-a"]
    assert first.model == "model-a"

    holder["cfg"] = _cfg(model="model-b", proactive_timezone="UTC")
    rebuilt = graph._build_deps()
    assert rebuilt is not first
    assert rebuilt.model == "model-b"
    assert builds == ["model-a", "model-b"]

    # gate settings change also rebuilds (idle threshold edited in config.yaml)
    holder["cfg"] = _cfg(
        model="model-b", proactive_timezone="UTC", proactive_idle_minutes=5
    )
    assert graph._build_deps().settings.idle_minutes == 5
    assert len(builds) == 3
