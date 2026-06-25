"""Tests for the stream-json output sink (EvoScientist/stream/json_sink.py).

The sink serializes the agent's native event stream as line-delimited JSON
(JSONL) to a writable stream, one JSON object per line. It is the headless
output path selected by ``--output-format stream-json``.
"""

import io
import json

import pytest

from EvoScientist.stream.json_sink import stream_json, write_events_as_json


async def _agen(items):
    """Wrap a list as an async generator (a real event source, not a mock)."""
    for item in items:
        yield item


def test_writes_each_event_as_one_jsonl_line(run_async):
    """Each event dict is serialized to exactly one JSON line, in order."""
    events = [
        {"type": "thinking", "content": "hmm", "id": 0},
        {"type": "text", "content": "hello"},
        {
            "type": "tool_call",
            "name": "write_file",
            "args": {"path": "a.md"},
            "id": "t1",
        },
        {"type": "done", "content": "hello", "response": "hello"},
    ]
    out = io.StringIO()

    run_async(write_events_as_json(_agen(events), out))

    lines = out.getvalue().splitlines()
    assert len(lines) == len(events)
    parsed = [json.loads(line) for line in lines]
    assert [e["type"] for e in parsed] == ["thinking", "text", "tool_call", "done"]
    assert parsed[2]["args"] == {"path": "a.md"}


def test_returns_final_response_from_done_event(run_async):
    """The sink returns the response text carried by the terminal `done` event."""
    events = [
        {"type": "text", "content": "partial"},
        {"type": "done", "content": "the answer", "response": "the answer"},
    ]
    out = io.StringIO()

    result = run_async(write_events_as_json(_agen(events), out))

    assert result == "the answer"


def test_non_serializable_arg_does_not_crash_the_stream(run_async):
    """A non-JSON-serializable value degrades to its str form instead of raising."""

    class Weird:
        """A value json.dumps cannot serialize, used to exercise the str fallback."""

        def __str__(self):
            """Return a sentinel so the fallback is observable in the output."""
            return "WEIRD"

    events = [
        {"type": "tool_call", "name": "x", "args": {"obj": Weird()}, "id": "t1"},
        {"type": "done", "content": "", "response": ""},
    ]
    out = io.StringIO()

    run_async(write_events_as_json(_agen(events), out))

    lines = out.getvalue().splitlines()
    # Both lines must be valid JSON; the non-serializable value falls back to str.
    first = json.loads(lines[0])
    assert first["args"]["obj"] == "WEIRD"


def test_stream_json_sources_events_from_gateway(run_async):
    """stream_json pulls events from gateway.stream_events(request) and serializes
    them — it does not reach past the gateway abstraction."""
    seen: dict[str, object] = {}
    events = [
        {"type": "text", "content": "hi"},
        {"type": "done", "content": "hi", "response": "hi"},
    ]

    class _FakeGateway:
        """A gateway stub whose stream_events yields a fixed event sequence."""

        def stream_events(self, request):
            """Record the request and return the canned event stream."""
            seen["request"] = request
            return _agen(events)

    out = io.StringIO()
    result = run_async(stream_json(_FakeGateway(), object(), out=out))

    assert result == "hi"
    assert "request" in seen  # the request was forwarded to the gateway
    types = [json.loads(line)["type"] for line in out.getvalue().splitlines()]
    assert types == ["text", "done"]


def test_stream_json_propagates_gateway_errors(run_async):
    """An error from the gateway stream propagates out of stream_json so the CLI
    dispatch can turn it into a clean exit."""

    async def _boom():
        """Yield one event, then fail like a mid-run graph error."""
        yield {"type": "text", "content": "partial"}
        raise RuntimeError("boom")

    class _FakeGateway:
        """A gateway stub whose stream raises partway through."""

        def stream_events(self, request):
            """Return a stream that fails after the first event."""
            return _boom()

    out = io.StringIO()
    with pytest.raises(RuntimeError, match="boom"):
        run_async(stream_json(_FakeGateway(), object(), out=out))
