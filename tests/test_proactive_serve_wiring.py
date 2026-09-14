"""Tests for serve's proactive wiring helpers (gating, client, origin listing)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import EvoScientist.cli.channel as channel_mod
from EvoScientist.cli.commands import _serve_proactive_enabled, _serve_server_client


@pytest.fixture(autouse=True)
def _clear_origins():
    with channel_mod._thread_channel_origins_lock:
        channel_mod._thread_channel_origins.clear()
    yield
    with channel_mod._thread_channel_origins_lock:
        channel_mod._thread_channel_origins.clear()


def _msg(chat_id):
    return SimpleNamespace(
        channel_type="qq", chat_id=chat_id, sender="u", metadata=None
    )


# ---- list_channel_origin_thread_ids -----------------------------------------


def test_list_origin_ids_empty():
    assert channel_mod.list_channel_origin_thread_ids() == []


def test_list_origin_ids_populated():
    channel_mod.remember_channel_origin("t1", _msg("c1"))
    channel_mod.remember_channel_origin("t2", _msg("c2"))
    assert set(channel_mod.list_channel_origin_thread_ids()) == {"t1", "t2"}


# ---- _serve_proactive_enabled -----------------------------------------------


def _cfg(**over):
    base = {"proactive_enabled": True, "gateway_backend": "langgraph_server"}
    base.update(over)
    return SimpleNamespace(**base)


def test_proactive_enabled_requires_flag_and_server_backend():
    assert _serve_proactive_enabled(_cfg()) is True
    assert _serve_proactive_enabled(_cfg(proactive_enabled=False)) is False
    assert _serve_proactive_enabled(_cfg(gateway_backend="local")) is False
    assert _serve_proactive_enabled(None) is False


# ---- _serve_server_client ---------------------------------------------------


def _runtime_state(*, client):
    execute = SimpleNamespace(thread_store=SimpleNamespace(client=client))
    gateway = SimpleNamespace(execute_gateway=execute)
    return SimpleNamespace(runtime_gateways=SimpleNamespace(graph_gateway=gateway))


def test_server_client_present_on_composite():
    sentinel = object()
    assert _serve_server_client(_runtime_state(client=sentinel)) is sentinel


def test_server_client_none_on_local_backend():
    # A local gateway has no execute_gateway attribute.
    rs = SimpleNamespace(
        runtime_gateways=SimpleNamespace(graph_gateway=SimpleNamespace())
    )
    assert _serve_server_client(rs) is None


# ---- _serve_drain_proactive (the serve-drives-the-apply seam) ----------------


class _SyncRuntime:
    def run_sync(self, fn):
        import asyncio

        return asyncio.run(fn())


def _server_runtime_state(client):
    execute = SimpleNamespace(thread_store=SimpleNamespace(client=client))
    return SimpleNamespace(
        runtime_gateways=SimpleNamespace(
            graph_gateway=SimpleNamespace(execute_gateway=execute)
        ),
        async_runtime=_SyncRuntime(),
    )


def _decision(pid):
    from langchain_core.messages import AIMessage

    from EvoScientist.proactive.service import ProactiveDecision

    return ProactiveDecision(
        "decided",
        "ok",
        reply="hi",
        message=AIMessage(content="hi", id=f"proactive-{pid}"),
        proactive_id=pid,
        source_thread_id="t1",
        head="h0",
    )


def test_drain_applies_each_queued_decision_once(monkeypatch):
    import queue

    from EvoScientist.cli.commands import _serve_drain_proactive

    applied = []

    async def _fake_apply(client, decision, *, publish, origin_present):
        applied.append(decision.proactive_id)
        return True

    monkeypatch.setattr(
        "EvoScientist.proactive.serve_runner.apply_proactive_decision", _fake_apply
    )

    q = queue.Queue()
    q.put(_decision("p1"))
    q.put(_decision("p2"))
    _serve_drain_proactive(_server_runtime_state(object()), q)

    assert applied == ["p1", "p2"]
    assert q.empty()  # each decision dequeued exactly once


def test_drain_is_noop_on_local_backend():
    import queue

    from EvoScientist.cli.commands import _serve_drain_proactive

    rs = SimpleNamespace(
        runtime_gateways=SimpleNamespace(graph_gateway=SimpleNamespace())
    )
    # No execute client -> returns immediately, must not raise.
    _serve_drain_proactive(rs, queue.Queue())
