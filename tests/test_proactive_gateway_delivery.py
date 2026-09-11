"""Tests for EvoScientist.proactive.gateway_delivery (server-path delivery).

Fake gateway (widened update_state_values) — no server. Validates the commit
shape (append tagged msg as model + clear) and that a ConflictError from the
append propagates so run_proactive_check's retry loop handles it.
"""

from __future__ import annotations

import asyncio

import httpx
from langgraph.graph import END
from langgraph_sdk.errors import ConflictError

from EvoScientist.proactive.gateway_delivery import GatewayDeliverer


def _conflict():
    resp = httpx.Response(409, request=httpx.Request("POST", "http://test"))
    return ConflictError("run in flight", response=resp, body=None)


class _FakeGateway:
    def __init__(self, *, fail_appends=0):
        self._fail_appends = fail_appends
        self._appends = 0
        self.calls = []

    async def update_state_values(
        self, target, thread_id, values, *, as_node=None, metadata=None
    ):
        self.calls.append(
            {"thread_id": thread_id, "values": values, "as_node": as_node}
        )
        if values is not None:  # append
            self._appends += 1
            if self._appends <= self._fail_appends:
                raise _conflict()


def test_deliver_appends_tagged_message_then_clears():
    gw = _FakeGateway()
    deliverer = GatewayDeliverer(gw, "target")
    ok = asyncio.run(deliverer.deliver("src", "Checking in!", "pid-1"))
    assert ok is True
    assert len(gw.calls) == 2
    append, clear = gw.calls
    assert append["thread_id"] == "src"
    msg = append["values"]["messages"][0]
    assert msg.content == "Checking in!"
    assert msg.additional_kwargs == {
        "evoscientist": {"is_proactive_push": True, "proactive_id": "pid-1"}
    }
    assert msg.id == "proactive-pid-1"
    assert append["as_node"] == "model"
    assert clear["values"] is None
    assert clear["as_node"] == END


def test_deliver_propagates_append_conflict():
    # The outer commit_with_conflict_retry (in run_proactive_check) must be the one
    # that catches this — GatewayDeliverer itself does not swallow append conflicts.
    gw = _FakeGateway(fail_appends=1)
    deliverer = GatewayDeliverer(gw, "target")
    try:
        asyncio.run(deliverer.deliver("src", "hi", "pid-1"))
        raise AssertionError("expected ConflictError to propagate")
    except ConflictError:
        pass


def test_deliver_end_to_end_via_run_proactive_check_retry():
    # GatewayDeliverer wired into the real check: append conflicts once then lands.
    from datetime import UTC, datetime, timedelta

    from EvoScientist.proactive.gate import ProactiveGateSettings
    from EvoScientist.proactive.service import run_proactive_check

    now = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)
    gw = _FakeGateway(fail_appends=1)

    async def _shadow(messages, trigger):
        return "Want to continue?"

    result = asyncio.run(
        run_proactive_check(
            settings=ProactiveGateSettings(
                enabled=True, idle_minutes=120, quiet_hours=None, timezone="UTC"
            ),
            source_thread_id="src",
            now=now,
            last_activity=now - timedelta(minutes=180),
            pre_head="h1",
            has_origin=True,
            in_flight=False,
            source_messages=[{"role": "user", "content": "prior"}],
            shadow_runner=_shadow,
            read_current_head=lambda: "h1",
            deliverer=GatewayDeliverer(gw, "target"),
            trigger="<proactive_trigger>",
            workspace_dir="/ws",
            model="m",
            proactive_id="pid-1",
        )
    )
    assert result.stage == "delivered"
    appends = [c for c in gw.calls if c["values"] is not None]
    assert len(appends) == 2  # first conflicted, retry landed
