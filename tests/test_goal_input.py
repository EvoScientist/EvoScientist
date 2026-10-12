"""A ``/goal`` request reaches the graph through both gateways' run input."""

from __future__ import annotations

from EvoScientist.gateway import GraphTarget, RunRequest
from EvoScientist.gateway.local import LocalGraphGateway

REQUEST = {"kind": "pause", "id": "r1", "target": "goal:g1", "review": True}


async def test_the_local_gateway_forwards_the_request(monkeypatch):
    seen: dict = {}

    async def _recording(agent, message, thread_id, **kwargs):
        seen.update(kwargs, message=message)
        yield {"type": "done", "response": ""}

    monkeypatch.setattr("EvoScientist.stream.events.stream_agent_events", _recording)
    gateway = LocalGraphGateway()
    request = RunRequest(
        message="",
        thread_id="t1",
        target=GraphTarget(local_graph=object()),
        goal_request=REQUEST,
    )
    events = [event async for event in gateway.stream_events(request)]
    assert [event["type"] for event in events] == ["done"]
    assert seen["goal_request"] == REQUEST
    assert seen["message"] == ""


async def test_a_plain_message_carries_no_request():
    request = RunRequest(message="hi", thread_id="t1")
    assert request.goal_request is None
