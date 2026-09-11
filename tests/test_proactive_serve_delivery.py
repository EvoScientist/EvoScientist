"""Tests for the serve-side proactive wiring.

Covers the seams added to wire ``proactive.delivery_watcher`` into serve: the
channel-origins accessor, the server-backend gate, the turn metadata marker,
the delivery poll composition, and the cron auto-registration. The watcher's own
logic is tested in
test_proactive_delivery_watcher.py; here we test the serve glue with fakes (no
server, no model, no bus).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from EvoScientist.cli import channel as channel_mod
from EvoScientist.cli import commands


@pytest.fixture(autouse=True)
def _clear_origins():
    def _reset():
        with channel_mod._thread_channel_origins_lock:
            channel_mod._thread_channel_origins.clear()

    _reset()
    yield
    _reset()


class _FakeAsyncRuntime:
    """run_sync that drives the async factory to completion synchronously."""

    def run_sync(self, factory, **_kw):
        return asyncio.run(factory())


class _FakeThreads:
    def __init__(self, state_by_thread=None):
        self._state = state_by_thread or {}
        self.updated: list[tuple[str, dict | None]] = []

    async def get_state(self, thread_id):
        return self._state.get(thread_id, {"values": {"messages": []}})

    async def update(self, thread_id, metadata=None):
        self.updated.append((thread_id, metadata))


class _FakeClient:
    def __init__(self, state_by_thread=None):
        self.threads = _FakeThreads(state_by_thread)


def _runtime_state(*, client, thread_id="t1", config=None):
    thread_store = (
        SimpleNamespace(client=client) if client is not None else SimpleNamespace()
    )
    return SimpleNamespace(
        config=config if config is not None else _proactive_cfg(),
        runtime_gateways=SimpleNamespace(thread_store=thread_store),
        thread_id=thread_id,
        async_runtime=_FakeAsyncRuntime(),
    )


def _proactive_cfg(**overrides):
    from EvoScientist.config.settings import EvoScientistConfig

    base = {"gateway_backend": "langgraph_server", "proactive_enabled": True}
    base.update(overrides)
    return EvoScientistConfig(**base)


def _register(thread_id):
    msg = SimpleNamespace(
        channel_type="telegram", chat_id="c1", sender="u1", metadata=None
    )
    channel_mod.remember_channel_origin(thread_id, msg)


def _tagged_state(proactive_id="pid1", content="ping"):
    return {
        "values": {
            "messages": [
                {"type": "human", "content": "hi"},
                {
                    "type": "ai",
                    "content": content,
                    "additional_kwargs": {
                        "evoscientist": {
                            "is_proactive_push": True,
                            "proactive_id": proactive_id,
                        }
                    },
                },
            ]
        }
    }


# ---- origins accessor ---------------------------------------------------------


def test_list_channel_origin_thread_ids_empty():
    assert channel_mod.list_channel_origin_thread_ids() == []


def test_list_channel_origin_thread_ids_populated():
    _register("t1")
    _register("t2")
    assert set(channel_mod.list_channel_origin_thread_ids()) == {"t1", "t2"}


# ---- server-backend gate ------------------------------------------------------


def test_server_client_none_on_local_backend():
    assert commands._serve_server_client(_runtime_state(client=None)) is None


def test_server_client_present_on_server_backend():
    client = _FakeClient()
    assert commands._serve_server_client(_runtime_state(client=client)) is client


# ---- turn metadata marker -----------------------------------------------------


def test_turn_metadata_carries_marker_when_proactive_on_server():
    meta = commands._serve_turn_metadata(
        _runtime_state(client=_FakeClient()), "/ws", "m"
    )
    assert meta["has_channel_origin"] is True
    assert meta["workspace_dir"] == "/ws"
    assert meta["model"] == "m"


def test_turn_metadata_plain_when_proactive_disabled_or_local():
    for cfg in (
        _proactive_cfg(proactive_enabled=False),
        _proactive_cfg(gateway_backend="local"),
    ):
        meta = commands._serve_turn_metadata(
            _runtime_state(client=_FakeClient(), config=cfg), "/ws", "m"
        )
        assert "has_channel_origin" not in meta
        assert meta["workspace_dir"] == "/ws"


# ---- delivery poll ------------------------------------------------------------


def test_deliver_publishes_then_idempotent(monkeypatch):
    published: list[tuple[str, str]] = []
    monkeypatch.setattr(
        commands,
        "publish_to_channel_origin",
        lambda tid, content: (published.append((tid, content)), True)[1],
    )
    client = _FakeClient({"t1": _tagged_state("pid1", "ping")})
    _register("t1")
    seen: set[str] = set()
    rs = _runtime_state(client=client)

    commands._serve_deliver_proactive_pushes(runtime_state=rs, seen=seen)
    assert published == [("t1", "ping")]
    assert "pid1" in seen

    # Second poll: id already seen → no re-publish.
    commands._serve_deliver_proactive_pushes(runtime_state=rs, seen=seen)
    assert published == [("t1", "ping")]


def test_deliver_noop_on_local(monkeypatch):
    published: list = []
    monkeypatch.setattr(
        commands,
        "publish_to_channel_origin",
        lambda tid, content: (published.append((tid, content)), True)[1],
    )
    _register("t1")
    commands._serve_deliver_proactive_pushes(
        runtime_state=_runtime_state(client=None), seen=set()
    )
    assert published == []


def test_deliver_noop_on_empty_origins(monkeypatch):
    published: list = []
    monkeypatch.setattr(
        commands,
        "publish_to_channel_origin",
        lambda tid, content: (published.append((tid, content)), True)[1],
    )
    client = _FakeClient({"t1": _tagged_state()})
    # No _register → no channel origins → nothing to poll.
    commands._serve_deliver_proactive_pushes(
        runtime_state=_runtime_state(client=client), seen=set()
    )
    assert published == []


def test_deliver_failed_publish_not_seen_then_retries(monkeypatch):
    calls: list[tuple[str, str]] = []

    def _pub(tid, content):
        calls.append((tid, content))
        return len(calls) > 1  # fail first, succeed second

    monkeypatch.setattr(commands, "publish_to_channel_origin", _pub)
    client = _FakeClient({"t1": _tagged_state("pid1", "ping")})
    _register("t1")
    seen: set[str] = set()
    rs = _runtime_state(client=client)

    commands._serve_deliver_proactive_pushes(runtime_state=rs, seen=seen)
    assert "pid1" not in seen  # first publish returned False

    commands._serve_deliver_proactive_pushes(runtime_state=rs, seen=seen)
    assert "pid1" in seen  # retried and succeeded
    assert len(calls) == 2


# ---- config flag --------------------------------------------------------------


def test_gateway_backend_default_is_local():
    from EvoScientist.config.settings import EvoScientistConfig

    assert EvoScientistConfig().gateway_backend == "local"


def test_gateway_backend_env_override(monkeypatch):
    from EvoScientist.config.settings import get_effective_config

    monkeypatch.setenv("EVOSCIENTIST_GATEWAY_BACKEND", "langgraph_server")
    assert get_effective_config().gateway_backend == "langgraph_server"


# ---- proactive gate + cron auto-registration ----------------------------------


def test_serve_proactive_enabled_requires_both_flag_and_server_backend():
    from EvoScientist.cli.commands import _serve_proactive_enabled

    assert _serve_proactive_enabled(_proactive_cfg()) is True
    assert _serve_proactive_enabled(_proactive_cfg(proactive_enabled=False)) is False
    assert _serve_proactive_enabled(_proactive_cfg(gateway_backend="local")) is False
    assert _serve_proactive_enabled(None) is False


def _patch_cron_client(monkeypatch, existing):
    from EvoScientist.proactive import cron

    fake = MagicMock()
    fake.crons.search.return_value = existing
    fake.crons.create_for_thread.return_value = {
        "cron_id": "p-new",
        "schedule": "*/10 * * * *",
    }
    fake.threads.search.return_value = []
    fake.threads.create.return_value = {"thread_id": "pt-1"}
    monkeypatch.setattr(cron, "_client", lambda: fake)
    monkeypatch.setattr(cron, "_default_timezone", lambda: "UTC")
    return fake


def test_ensure_proactive_cron_creates_when_absent(monkeypatch):
    from EvoScientist.cli.commands import _serve_ensure_proactive_cron

    fake = _patch_cron_client(monkeypatch, existing=[])
    _serve_ensure_proactive_cron(_proactive_cfg())
    fake.crons.create_for_thread.assert_called_once()
    assert fake.crons.create_for_thread.call_args.args == ("pt-1", "proactive")


def test_ensure_proactive_cron_reuses_existing(monkeypatch):
    from EvoScientist.cli.commands import _serve_ensure_proactive_cron

    fake = _patch_cron_client(monkeypatch, existing=[{"cron_id": "p-1"}])
    _serve_ensure_proactive_cron(_proactive_cfg())
    fake.crons.create_for_thread.assert_not_called()


def test_ensure_proactive_cron_noop_when_disabled_or_local(monkeypatch):
    from EvoScientist.cli.commands import _serve_ensure_proactive_cron

    fake = _patch_cron_client(monkeypatch, existing=[])
    _serve_ensure_proactive_cron(_proactive_cfg(proactive_enabled=False))
    _serve_ensure_proactive_cron(_proactive_cfg(gateway_backend="local"))
    fake.crons.search.assert_not_called()
    fake.crons.create_for_thread.assert_not_called()


def test_ensure_proactive_cron_failure_is_logged_not_raised(monkeypatch, caplog):
    from EvoScientist.cli.commands import _serve_ensure_proactive_cron

    fake = _patch_cron_client(monkeypatch, existing=[])
    fake.crons.search.side_effect = RuntimeError("server down")
    with caplog.at_level("WARNING", logger="EvoScientist.cli.commands"):
        _serve_ensure_proactive_cron(_proactive_cfg())
    assert any("registration failed" in r.message for r in caplog.records)
