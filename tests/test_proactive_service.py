"""Tests for EvoScientist.proactive.service (the check orchestrator)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from langgraph_sdk.errors import ConflictError

from EvoScientist.proactive.commit import NO_PUSH_SENTINEL
from EvoScientist.proactive.gate import ProactiveGateSettings
from EvoScientist.proactive.service import (
    SourceRead,
    make_read_source,
    run_proactive_check,
    run_proactive_scan,
    run_proactive_tick,
)

_NOON = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)


def _settings(**over):
    base = {
        "enabled": True,
        "idle_minutes": 120,
        "quiet_hours": None,
        "timezone": "UTC",
    }
    base.update(over)
    return ProactiveGateSettings(**base)


class _FakeDeliverer:
    def __init__(self, result=True):
        self.calls = []
        self._result = result

    async def deliver(
        self, source_thread_id, reply, proactive_id, *, source_messages=None
    ):
        self.calls.append((source_thread_id, reply, proactive_id))
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def _run(**over):
    shadow_calls = []
    reply = over.pop("_reply", "Hey, want to pick this up?")

    async def _shadow(messages, trigger):
        shadow_calls.append((messages, trigger))
        return reply

    deliverer = over.pop("_deliverer", None) or _FakeDeliverer()
    head_seq = over.pop("_current_head", "h1")

    kwargs = {
        "settings": over.pop("settings", _settings()),
        "source_thread_id": "src",
        "now": _NOON,
        "last_activity": _NOON - timedelta(minutes=180),
        "pre_head": "h1",
        "has_origin": True,
        "in_flight": False,
        "source_messages": [{"role": "user", "content": "prior"}],
        "shadow_runner": _shadow,
        "read_current_head": lambda: head_seq,
        "deliverer": deliverer,
        "trigger": "<proactive_trigger>",
        "workspace_dir": "/ws",
        "model": "m",
        "proactive_id": "pid1",
    }
    kwargs.update(over)
    result = asyncio.run(run_proactive_check(**kwargs))
    return result, deliverer, shadow_calls


def test_gate_rejection_short_circuits_before_shadow():
    result, deliverer, shadow_calls = _run(settings=_settings(enabled=False))
    assert result.stage == "gate_rejected"
    assert result.reason == "disabled"
    assert shadow_calls == []  # no model call
    assert deliverer.calls == []


def test_full_path_delivers():
    result, deliverer, shadow_calls = _run()
    assert result.stage == "delivered"
    assert result.delivered is True
    assert result.source_thread_id == "src"
    assert len(shadow_calls) == 1
    assert deliverer.calls == [("src", "Hey, want to pick this up?", "pid1")]


def test_no_push_reply_does_not_deliver():
    result, deliverer, _ = _run(_reply=NO_PUSH_SENTINEL)
    assert result.stage == "no_push"
    assert result.reason == "no_push"
    assert deliverer.calls == []


def test_empty_reply_does_not_deliver():
    result, deliverer, _ = _run(_reply="")
    assert result.stage == "no_push"
    assert deliverer.calls == []


def test_none_reply_does_not_deliver():
    result, deliverer, _ = _run(_reply=None)
    assert result.stage == "no_push"
    assert deliverer.calls == []


def test_stale_source_does_not_deliver():
    # current head differs from pre_head → a user turn landed during the shadow.
    result, deliverer, _ = _run(_current_head="h2")
    assert result.stage == "stale"
    assert deliverer.calls == []


def test_delivery_failure_recorded():
    result, _deliverer, _ = _run(_deliverer=_FakeDeliverer(result=False))
    assert result.stage == "delivery_failed"
    assert result.reason == "deliver_returned_false"


def test_delivery_exception_recorded():
    result, _, _ = _run(_deliverer=_FakeDeliverer(result=RuntimeError("boom")))
    assert result.stage == "delivery_failed"
    assert result.reason == "exception"


def test_ledger_records_results():
    from EvoScientist.proactive.service import get_ledger

    _run()
    ledger = get_ledger()
    assert ledger  # at least our result is present
    assert ledger[-1].stage == "delivered"


# ---- ConflictError defer-and-retry wiring ----


def _conflict() -> ConflictError:
    resp = httpx.Response(409, request=httpx.Request("POST", "http://test"))
    return ConflictError("run in flight", response=resp, body=None)


class _ConflictDeliverer:
    """Raises ConflictError on the first ``fail`` deliver calls, then returns True.

    Contrived — real deliverers never raise ConflictError (only the gateway commit
    does) — but it drives the service's conflict wiring: a ConflictError from the
    commit_fn must reach ``commit_with_conflict_retry``'s retry loop, not the outer
    delivery_failed handler.
    """

    def __init__(self, fail: int) -> None:
        self.fail = fail
        self.calls = 0

    async def deliver(
        self, source_thread_id, reply, proactive_id, *, source_messages=None
    ):
        self.calls += 1
        if self.calls <= self.fail:
            raise _conflict()
        return True


@pytest.fixture
def no_conflict_backoff(monkeypatch):
    """Keep the conflict tests fast; the real backoff sleeps 1s/2s/4s."""
    import EvoScientist.proactive.service as service_mod

    async def _instant(attempt):
        _instant.calls.append(attempt)

    _instant.calls = []
    monkeypatch.setattr(service_mod, "_conflict_backoff", _instant)
    return _instant


def test_conflict_then_succeeds_delivers(no_conflict_backoff):
    d = _ConflictDeliverer(fail=1)
    result, _, _ = _run(_deliverer=d)
    assert no_conflict_backoff.calls == [0]  # one deferred retry
    assert result.stage == "delivered"
    assert result.delivered is True
    assert d.calls == 2


def test_conflict_exhausted_recorded(no_conflict_backoff):
    d = _ConflictDeliverer(fail=99)
    result, _, _ = _run(_deliverer=d)
    assert no_conflict_backoff.calls == [0, 1, 2, 3]
    assert result.stage == "conflict_exhausted"
    assert result.reason == "retries_exhausted"
    assert d.calls == 4  # max_retries + 1


# ---- run_proactive_scan (one cron tick over candidate threads) ----


class _FakeSource:
    def __init__(
        self,
        *,
        pre_head="h1",
        has_origin=True,
        in_flight=False,
        head="h1",
        raise_for=(),
    ):
        self.pre_head = pre_head
        self.has_origin = has_origin
        self.in_flight = in_flight
        self.head = head
        self.raise_for = set(raise_for)
        self.reads = []

    async def __call__(self, thread_id):
        self.reads.append(thread_id)
        if thread_id in self.raise_for:
            raise RuntimeError("boom")
        return SourceRead(
            messages=[{"role": "user", "content": f"prior for {thread_id}"}],
            pre_head=self.pre_head,
            has_origin=self.has_origin,
            in_flight=self.in_flight,
            read_current_head=lambda: self.head,
        )


def _scan(candidates, *, source=None, deliverer=None, settings=None, reply="Hey!"):
    src = source or _FakeSource()
    dlv = deliverer or _FakeDeliverer()

    async def _shadow(messages, trigger):
        return reply

    results = asyncio.run(
        run_proactive_scan(
            candidates,
            now=_NOON,
            settings=settings or _settings(),
            read_source=src,
            shadow_runner=_shadow,
            deliverer=dlv,
            trigger="<proactive_trigger>",
            workspace_dir="/ws",
            model="m",
        )
    )
    return results, src, dlv


_IDLE = _NOON - timedelta(minutes=180)  # older than idle_minutes=120


def test_scan_empty_candidates_reads_nothing():
    results, src, dlv = _scan([])
    assert results == []
    assert src.reads == []
    assert dlv.calls == []


def test_scan_delivers_per_candidate_with_unique_ids():
    results, src, dlv = _scan([("t1", _IDLE), ("t2", _IDLE)])
    assert [r.stage for r in results] == ["delivered", "delivered"]
    assert src.reads == ["t1", "t2"]
    assert [c[0] for c in dlv.calls] == ["t1", "t2"]
    pids = [c[2] for c in dlv.calls]
    assert len(set(pids)) == 2  # fresh proactive_id per thread


def test_scan_threads_last_activity_into_the_gate():
    # last_activity = now → not idle (idle_minutes=120) → gate_rejected, no deliver
    results, _, dlv = _scan([("t1", _NOON)])
    assert results[0].stage == "gate_rejected"
    assert dlv.calls == []


def test_scan_isolates_a_per_candidate_failure():
    src = _FakeSource(raise_for=("t1",))
    results, _, dlv = _scan([("t1", _IDLE), ("t2", _IDLE)], source=src)
    assert results[0].stage == "scan_error"
    assert results[1].stage == "delivered"  # t2 still processed after t1 raised
    assert [c[0] for c in dlv.calls] == ["t2"]


# ---- run_proactive_tick (enumerate then check — one tick body) ----


class _FakeThreadsAPI:
    def __init__(self, result):
        self._result = result

    def search(self, **kwargs):
        async def _co():
            return self._result

        return _co()


class _FakeTickClient:
    def __init__(self, threads_result):
        self.threads = _FakeThreadsAPI(threads_result)


def _tick(threads_result, *, source=None, deliverer=None):
    src = source or _FakeSource()
    dlv = deliverer or _FakeDeliverer()

    async def _shadow(messages, trigger):
        return "Hey!"

    results = asyncio.run(
        run_proactive_tick(
            _FakeTickClient(threads_result),
            now=_NOON,
            settings=_settings(),
            read_source=src,
            shadow_runner=_shadow,
            deliverer=dlv,
            trigger="<proactive_trigger>",
            workspace_dir="/ws",
            model="m",
        )
    )
    return results, src, dlv


def test_tick_enumerates_then_checks_each_candidate():
    threads = [
        {"thread_id": "t1", "updated_at": "2026-08-20T09:00:00+00:00"},  # idle
        {"thread_id": "t2", "updated_at": "2026-08-20T09:30:00+00:00"},  # idle
    ]
    results, src, dlv = _tick(threads)
    assert [r.stage for r in results] == ["delivered", "delivered"]
    assert src.reads == ["t1", "t2"]
    assert [c[0] for c in dlv.calls] == ["t1", "t2"]


def test_tick_no_candidates_does_nothing():
    results, src, dlv = _tick([])
    assert results == []
    assert src.reads == []
    assert dlv.calls == []


# ---- make_read_source (server-backed read, real get_state shape) ----


class _FakeStateThreads:
    def __init__(self, state):
        self._state = state
        self.calls = []

    def get_state(self, thread_id):
        self.calls.append(thread_id)
        return self._state() if callable(self._state) else self._state


class _FakeStateClient:
    def __init__(self, state):
        self.threads = _FakeStateThreads(state)


def test_make_read_source_builds_from_get_state():
    # shape captured from a live langgraph-dev probe
    state = {
        "checkpoint_id": "ck-1",
        "values": {"messages": [{"content": "hi", "type": "human"}]},
    }
    read_source = make_read_source(_FakeStateClient(state))
    src = asyncio.run(read_source("t1"))
    assert src.pre_head == "ck-1"
    assert src.messages == [{"content": "hi", "type": "human"}]
    assert src.has_origin is True
    assert src.in_flight is False
    assert asyncio.run(src.read_current_head()) == "ck-1"


def test_read_current_head_refetches_the_head():
    """The pre-commit stale check must see a head that moved while the shadow ran
    (a user turn or a concurrent tick's push), so it re-fetches instead of
    returning the captured value."""
    heads = iter(["ck-1", "ck-1", "ck-2"])
    client = _FakeStateClient(
        lambda: {"checkpoint_id": next(heads), "values": {"messages": []}}
    )
    src = asyncio.run(make_read_source(client)("t1"))
    assert src.pre_head == "ck-1"
    assert asyncio.run(src.read_current_head()) == "ck-1"
    assert asyncio.run(src.read_current_head()) == "ck-2"
    assert len(client.threads.calls) == 3


def test_check_discards_push_when_head_moved_during_shadow():
    """Server path end-to-end: the head moves between the read and the commit →
    stale, nothing delivered (covers a user turn or an overlapping tick)."""
    from EvoScientist.proactive.gate import ProactiveGateSettings

    heads = iter(["ck-1", "ck-2", "ck-2", "ck-2"])
    client = _FakeStateClient(
        lambda: {"checkpoint_id": next(heads), "values": {"messages": []}}
    )
    src = asyncio.run(make_read_source(client)("t1"))

    class _Deliverer:
        calls = 0

        async def deliver(self, *a, **k):
            self.calls += 1
            return True

    deliverer = _Deliverer()

    async def _shadow(messages, trigger):
        return "push text"

    result = asyncio.run(
        run_proactive_check(
            settings=ProactiveGateSettings(
                enabled=True, idle_minutes=0, quiet_hours=None, timezone="UTC"
            ),
            source_thread_id="t1",
            now=datetime.now(UTC),
            last_activity=datetime.now(UTC) - timedelta(hours=1),
            pre_head=src.pre_head,
            has_origin=True,
            in_flight=False,
            source_messages=src.messages,
            shadow_runner=_shadow,
            read_current_head=src.read_current_head,
            deliverer=deliverer,
            trigger="t",
            workspace_dir=None,
            model=None,
            proactive_id="p1",
        )
    )
    assert result.stage == "stale"
    assert deliverer.calls == 0


def test_read_source_checkpoint_id_falls_back_to_nested():
    state = {"checkpoint": {"checkpoint_id": "ck-9"}, "values": {"messages": []}}
    src = asyncio.run(make_read_source(_FakeStateClient(state))("t1"))
    assert src.pre_head == "ck-9"


def test_read_source_missing_values_yields_empty_messages():
    src = asyncio.run(make_read_source(_FakeStateClient({"checkpoint_id": "c"}))("t1"))
    assert src.messages == []
    assert src.pre_head == "c"


def test_conflict_backoff_grows_and_caps(monkeypatch):
    import EvoScientist.proactive.service as service_mod

    slept = []

    async def _fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(service_mod.asyncio, "sleep", _fake_sleep)
    for attempt in range(5):
        asyncio.run(service_mod._conflict_backoff(attempt))
    assert slept == [1, 2, 4, 8, 8]
