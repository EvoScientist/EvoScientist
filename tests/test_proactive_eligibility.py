"""Tests for EvoScientist.proactive.eligibility (server-scoped candidates).

Server-only in production (enumerates the server registry via the SDK), so
validated with a fake threads client — no running server. The key guarantees:
candidates come only from server-registry threads carrying the channel-origin
marker (the "always server-born" invariant), and last-activity is parsed for the
gate rather than the idle test being re-implemented here.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from EvoScientist.proactive.eligibility import (
    CHANNEL_ORIGIN_MARKER,
    list_channel_thread_candidates,
)


class _FakeThreads:
    def __init__(self, result, *, is_async=True):
        self._result = result
        self._is_async = is_async
        self.search_kwargs = None
        self.update_calls = []

    def search(self, **kwargs):
        self.search_kwargs = kwargs
        return self._maybe_async(self._result)

    def update(self, thread_id, **kwargs):
        self.update_calls.append((thread_id, kwargs))
        return self._maybe_async(None)

    def _maybe_async(self, value):
        if self._is_async:

            async def _co():
                return value

            return _co()
        return value


class _FakeClient:
    def __init__(self, result, *, is_async=True):
        self.threads = _FakeThreads(result, is_async=is_async)


def _run(result, *, is_async=True, **over):
    client = _FakeClient(result, is_async=is_async)
    out = asyncio.run(list_channel_thread_candidates(client, **over))
    return out, client


def test_returns_ids_with_parsed_last_activity():
    threads = [
        {"thread_id": "t1", "updated_at": "2026-09-04T10:00:00+00:00"},
        {"thread_id": "t2", "updated_at": "2026-09-04T09:00:00+00:00"},
    ]
    out, client = _run(threads)
    assert out == [
        ("t1", datetime(2026, 9, 4, 10, 0, tzinfo=UTC)),
        ("t2", datetime(2026, 9, 4, 9, 0, tzinfo=UTC)),
    ]
    # enforces the invariant: the query is scoped by the channel-origin marker
    assert client.threads.search_kwargs["metadata"] == CHANNEL_ORIGIN_MARKER
    assert client.threads.search_kwargs["limit"] == 50
    # in-flight condition enforced server-side: only idle threads are candidates
    assert client.threads.search_kwargs["status"] == "idle"


def test_custom_marker_and_limit_passed_through():
    out, client = _run([], marker={"run_kind": "channel"}, limit=5)
    assert out == []
    assert client.threads.search_kwargs["metadata"] == {"run_kind": "channel"}
    assert client.threads.search_kwargs["limit"] == 5


def test_status_filter_can_be_disabled():
    _, client = _run([], status=None)
    assert client.threads.search_kwargs["status"] is None


def test_missing_updated_at_yields_none_activity():
    out, _ = _run([{"thread_id": "t1"}, {"thread_id": "t2", "updated_at": ""}])
    assert out == [("t1", None), ("t2", None)]


def test_threads_without_id_are_skipped():
    out, _ = _run([{"updated_at": "2026-09-04T10:00:00+00:00"}, {"thread_id": "keep"}])
    assert out == [("keep", None)]


def test_empty_registry_is_empty_list():
    # Fail-closed: no marker-stamped threads (e.g. serve hasn't stamped yet).
    out, _ = _run([])
    assert out == []


def test_none_result_is_empty_list():
    out, _ = _run(None)
    assert out == []


def test_sync_search_result_supported():
    # A sync client (non-awaitable search) is handled too.
    threads = [{"thread_id": "t1", "updated_at": "2026-09-04T10:00:00+00:00"}]
    out, _ = _run(threads, is_async=False)
    assert out == [("t1", datetime(2026, 9, 4, 10, 0, tzinfo=UTC))]


def test_datetime_updated_at_passthrough():
    dt = datetime(2026, 9, 4, 8, 0, tzinfo=UTC)
    out, _ = _run([{"thread_id": "t1", "updated_at": dt}])
    assert out == [("t1", dt)]
