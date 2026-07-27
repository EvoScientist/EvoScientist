"""Smoke test for the /api/models route mounted via langgraph.json's
``http`` field. We test the FastAPI app directly — no need to spin up
langgraph dev.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from EvoScientist.config import EvoScientistConfig
from EvoScientist.langgraph_dev.http import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_notifier_queues():
    """Clear the notifier's module-level queues before and after every test.

    The notifier state is module-global (``_notifications_by_thread`` +
    ``_unrouted_queue``) and pollutes across tests otherwise — a routed
    notification enqueued but never drained by a test that only inspected
    response metadata will surface in the next test's ``drain_notifications``
    call. Autouse gives us a fresh queue per test without every test having
    to remember the setup.
    """
    from EvoScientist.cli import async_notifier

    async_notifier.drain_notifications()  # clear all buckets
    yield
    async_notifier.drain_notifications()


@pytest.fixture(autouse=True)
def _zero_sse_poll_interval(monkeypatch):
    """Zero the SSE loop's inter-iteration sleep so tests don't pay real wall
    time between yields. Tests that also need the heartbeat or max-lifetime
    interval tuned monkeypatch those constants separately.
    """
    monkeypatch.setattr("EvoScientist.langgraph_dev.http._SSE_POLL_INTERVAL_S", 0)


def test_get_models_returns_entries_and_default():
    mock_cfg = EvoScientistConfig(
        model="claude-sonnet-4-6", provider="custom-anthropic"
    )
    with patch(
        "EvoScientist.langgraph_dev.http.get_effective_config", return_value=mock_cfg
    ):
        resp = client.get("/api/models")
    assert resp.status_code == 200
    body = resp.json()
    assert "entries" in body
    assert "default" in body
    assert body["default"] == {
        "name": "claude-sonnet-4-6",
        "provider": "custom-anthropic",
    }
    assert isinstance(body["entries"], list)
    assert len(body["entries"]) > 0
    # Every entry has the three required keys
    for entry in body["entries"]:
        assert set(entry.keys()) == {"name", "model_id", "provider"}
        assert isinstance(entry["name"], str)
        assert entry["name"]
        assert isinstance(entry["model_id"], str)
        assert entry["model_id"]
        assert isinstance(entry["provider"], str)
        assert entry["provider"]


def test_entries_preserve_registry_order():
    """The picker uses position-in-list to rank providers per short name —
    the JSON must preserve the order returned by ``list_models_by_provider``.

    Stubs ``get_effective_config`` to keep the assertion focused on
    registry order rather than implicitly depending on the ambient
    deploy config.
    """
    from EvoScientist.llm.models import list_models_by_provider

    expected = [
        {"name": n, "model_id": m, "provider": p}
        for n, m, p in list_models_by_provider()
    ]
    mock_cfg = EvoScientistConfig()
    with patch(
        "EvoScientist.langgraph_dev.http.get_effective_config", return_value=mock_cfg
    ):
        resp = client.get("/api/models")
    assert resp.json()["entries"] == expected


def test_default_passes_through_arbitrary_config_pair():
    """If config.yaml names a (name, provider) pair that isn't in the
    registry (typo, retired model), still report it as default — the
    picker labels it as the active selection regardless.
    """
    mock_cfg = EvoScientistConfig(model="some-retired-name", provider="some-provider")
    with patch(
        "EvoScientist.langgraph_dev.http.get_effective_config", return_value=mock_cfg
    ):
        resp = client.get("/api/models")
    assert resp.json()["default"] == {
        "name": "some-retired-name",
        "provider": "some-provider",
    }


def test_ollama_models_appended_when_base_url_configured():
    """Mirrors the TUI ``/model`` picker: when ``ollama_base_url`` is set,
    locally-pulled Ollama models are appended after the static registry
    as ``provider: "ollama"`` entries.
    """
    mock_cfg = EvoScientistConfig(
        model="claude-sonnet-4-6",
        provider="custom-anthropic",
        ollama_base_url="http://localhost:11434",
    )

    async def fake_discover(_base_url, *, timeout):
        return ["llama3:8b", "mistral:7b"]

    with (
        patch(
            "EvoScientist.langgraph_dev.http.get_effective_config",
            return_value=mock_cfg,
        ),
        patch(
            "EvoScientist.llm.ollama_discovery.discover_ollama_models",
            new=fake_discover,
        ),
    ):
        body = client.get("/api/models").json()

    # Assert the response is the static registry followed by the discovered
    # Ollama suffix — robust to future static Ollama entries in the registry.
    from EvoScientist.llm.models import list_models_by_provider

    static_entries = [
        {"name": n, "model_id": m, "provider": p}
        for n, m, p in list_models_by_provider()
    ]
    discovered_entries = [
        {"name": "llama3:8b", "model_id": "llama3:8b", "provider": "ollama"},
        {"name": "mistral:7b", "model_id": "mistral:7b", "provider": "ollama"},
    ]
    assert body["entries"][: len(static_entries)] == static_entries
    assert body["entries"][len(static_entries) :] == discovered_entries
    # TUI's "Custom Ollama model…" sentinel is a widget-specific affordance —
    # it must not appear on the HTTP surface.
    assert not any(e["model_id"] == "__custom_ollama__" for e in body["entries"])


def test_ollama_discovery_skipped_when_base_url_absent():
    """No Ollama discovery should happen when ``ollama_base_url`` is unset —
    matches the ``/model`` picker's gating. The probe function should never
    be called in that case.
    """
    mock_cfg = EvoScientistConfig(
        model="claude-sonnet-4-6", provider="custom-anthropic"
    )
    calls: list[str | None] = []

    async def spy_discover(base_url, *, timeout):
        calls.append(base_url)
        return []

    with (
        patch(
            "EvoScientist.langgraph_dev.http.get_effective_config",
            return_value=mock_cfg,
        ),
        patch(
            "EvoScientist.llm.ollama_discovery.discover_ollama_models",
            new=spy_discover,
        ),
    ):
        body = client.get("/api/models").json()

    assert calls == []
    # Response is exactly the static registry — no Ollama additions whatsoever.
    from EvoScientist.llm.models import list_models_by_provider

    assert body["entries"] == [
        {"name": n, "model_id": m, "provider": p}
        for n, m, p in list_models_by_provider()
    ]


# ============================================================================
# /api/async-notifications/stream/{thread_id}
# ============================================================================

# SSE tests drive the underlying async generator directly with a fake
# Request. Going through Starlette's ``TestClient.stream()`` deadlocks
# against an infinite server-side loop: the client's sync httpx transport
# has no clean way to signal disconnect between yields, so ``iter_lines()``
# blocks waiting for content after the first line while the generator's
# ``asyncio.sleep`` keeps ticking. Unit-testing the generator instead
# gives deterministic control over the loop's exit condition (via the
# fake request's ``is_disconnected``) and covers the same contract:
# per-thread filtering, wire schema, unrouted exclusion.


class _FakeRequest:
    """Minimal Request stand-in — reports connected until asked otherwise.

    ``stream_async_notifications`` only touches ``path_params`` and
    ``is_disconnected()`` on the Request, so a full Starlette Request is
    overkill. ``disconnect_after`` bounds the loop after N connectedness
    checks so each test drives the generator to a deterministic exit.
    """

    def __init__(self, thread_id: str, disconnect_after: int = 1) -> None:
        self.path_params = {"thread_id": thread_id}
        self._checks_remaining = disconnect_after

    async def is_disconnected(self) -> bool:
        if self._checks_remaining <= 0:
            return True
        self._checks_remaining -= 1
        return False


def _make_notification(task_id: str, thread_id: str | None = None):
    from EvoScientist.cli.async_notifier import AsyncTaskNotification

    return AsyncTaskNotification(
        task_id=task_id,
        agent_name="literature-review",
        status="success",
        received_at="2026-07-22T16:55:12Z",
        prompt="Produce a literature review on continual learning strategies",
        kind="agent",
        origin_cli_thread_id=thread_id,
    )


async def _collect_events(request: _FakeRequest) -> list[str]:
    """Drive the SSE handler's response body generator to completion and
    collect every yielded chunk."""
    from EvoScientist.langgraph_dev.http import stream_async_notifications

    response = await stream_async_notifications(request)  # type: ignore[arg-type]
    # StreamingResponse stores the async iterator on ``body_iterator``.
    chunks: list[str] = []
    async for chunk in response.body_iterator:
        chunks.append(chunk if isinstance(chunk, str) else chunk.decode("utf-8"))
    return chunks


def _parse_data_events(chunks: list[str]) -> list[dict]:
    """Extract the JSON payloads of all ``data:`` events from raw chunks."""
    import json as _json

    payloads: list[dict] = []
    for chunk in chunks:
        for line in chunk.splitlines():
            if line.startswith("data:"):
                payloads.append(_json.loads(line[len("data:") :].strip()))
    return payloads


async def test_sse_response_declares_event_stream_media_type():
    """Response Content-Type must be text/event-stream so browsers activate
    the EventSource parser path."""
    from EvoScientist.cli import async_notifier
    from EvoScientist.langgraph_dev.http import stream_async_notifications

    async_notifier._enqueue(_make_notification("t-ct", "thread-ct"))
    response = await stream_async_notifications(
        _FakeRequest("thread-ct")  # type: ignore[arg-type]
    )
    assert response.media_type == "text/event-stream"


async def test_sse_emits_enqueued_notification_for_matching_thread():
    """A notification enqueued with origin_cli_thread_id=X must surface on
    the stream for thread X, projected through the wire schema."""
    from EvoScientist.cli import async_notifier

    async_notifier._enqueue(_make_notification("t-match", "thread-match"))
    chunks = await _collect_events(_FakeRequest("thread-match"))
    payloads = _parse_data_events(chunks)

    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["task_id"] == "t-match"
    assert payload["agent_name"] == "literature-review"
    assert payload["status"] == "success"
    assert payload["kind"] == "agent"
    # Wire schema deliberately excludes origin_cli_thread_id (redundant —
    # the client already knows the thread it opened the stream against).
    assert "origin_cli_thread_id" not in payload


async def test_sse_does_not_emit_unrouted_notifications():
    """Unrouted notifications (origin_cli_thread_id=None) must NOT reach
    the per-thread stream. This protects multi-tab clients: a notification
    with no origin would otherwise be broadcast to every open SSE stream
    and duplicate the injected turn across unrelated conversations."""
    from EvoScientist.cli import async_notifier

    async_notifier._enqueue(_make_notification("t-unrouted", None))
    async_notifier._enqueue(_make_notification("t-routed", "thread-iso"))

    chunks = await _collect_events(_FakeRequest("thread-iso"))
    payloads = _parse_data_events(chunks)

    assert [p["task_id"] for p in payloads] == ["t-routed"]
    # Unrouted survivor stays in the unrouted queue.
    unrouted = async_notifier.drain_notifications()
    assert [n.task_id for n in unrouted] == ["t-unrouted"]


async def test_sse_does_not_emit_notifications_for_other_threads():
    """A notification for thread-A must not leak to thread-B's stream."""
    from EvoScientist.cli import async_notifier

    async_notifier._enqueue(_make_notification("t-A", "thread-A"))
    async_notifier._enqueue(_make_notification("t-B", "thread-B"))

    chunks = await _collect_events(_FakeRequest("thread-B"))
    payloads = _parse_data_events(chunks)

    assert [p["task_id"] for p in payloads] == ["t-B"]
    # thread-A's notification survives — nobody drained it.
    remaining = async_notifier.drain_notifications("thread-A")
    assert [n.task_id for n in remaining] == ["t-A"]


async def test_sse_returns_cleanly_when_client_is_disconnected_before_first_check():
    """If the client has already dropped by the time the loop opens, the
    generator returns without touching the queue or yielding anything."""
    from EvoScientist.cli import async_notifier

    async_notifier._enqueue(_make_notification("t-x", "thread-x"))
    chunks = await _collect_events(_FakeRequest("thread-x", disconnect_after=0))
    assert chunks == []
    # Notification stayed on the queue — the disconnected loop never drained.
    remaining = async_notifier.drain_notifications("thread-x")
    assert [n.task_id for n in remaining] == ["t-x"]


async def test_sse_preserves_queue_tail_on_mid_batch_disconnect(monkeypatch):
    """A disconnect after the first yielded event must leave later events
    on the queue instead of draining them into an already-broken socket.
    This is the invariant behind the one-at-a-time dequeue in the handler:
    ``is_disconnected`` is re-checked between every pop.
    """
    from EvoScientist.cli import async_notifier

    async_notifier._enqueue(_make_notification("t-1", "thread-mid"))
    async_notifier._enqueue(_make_notification("t-2", "thread-mid"))
    async_notifier._enqueue(_make_notification("t-3", "thread-mid"))

    # disconnect_after=2 → loop yields t-1, yields t-2, then observes
    # disconnect on the next iteration and returns.
    chunks = await _collect_events(_FakeRequest("thread-mid", disconnect_after=2))
    payloads = _parse_data_events(chunks)
    assert [p["task_id"] for p in payloads] == ["t-1", "t-2"]

    # t-3 survives on the queue for the next reconnect.
    remaining = async_notifier.drain_notifications("thread-mid")
    assert [n.task_id for n in remaining] == ["t-3"]


async def test_sse_emits_heartbeat_when_queue_is_idle(monkeypatch):
    """With no notifications enqueued, an idle iteration past the heartbeat
    interval emits a ``: keepalive`` comment line. Comment lines are the
    SSE-spec keep-alive that browsers silently ignore.
    """
    monkeypatch.setattr("EvoScientist.langgraph_dev.http._SSE_HEARTBEAT_INTERVAL_S", 0)

    chunks = await _collect_events(_FakeRequest("thread-idle", disconnect_after=1))
    # No data: events (queue was empty), one keepalive comment.
    assert _parse_data_events(chunks) == []
    keepalives = [c for c in chunks if c.startswith(": keepalive")]
    assert len(keepalives) == 1


async def test_sse_returns_when_max_lifetime_elapsed(monkeypatch):
    """On lifetime expiry the generator must return so the EventSource
    client can transparently reconnect (SSE spec-defined behavior).
    ``disconnect_after`` is set generously so a return can only come from
    the lifetime check, not from the disconnect path.
    """
    from EvoScientist.cli import async_notifier

    monkeypatch.setattr("EvoScientist.langgraph_dev.http._SSE_MAX_LIFETIME_S", 0)
    async_notifier._enqueue(_make_notification("t-late", "thread-late"))

    chunks = await _collect_events(_FakeRequest("thread-late", disconnect_after=1000))
    assert chunks == []
    # Notification stayed on the queue — the loop returned before draining.
    remaining = async_notifier.drain_notifications("thread-late")
    assert [n.task_id for n in remaining] == ["t-late"]
