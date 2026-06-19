"""Shared DeepAgents v3 protocol fakes for stream tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

from EvoScientist.stream.events import stream_agent_events
from tests.conftest import run_async


async def async_iter(items: Iterable[Any]) -> AsyncIterator[Any]:
    for item in items:
        yield item


def collect_events(
    agent,
    message: str = "hi",
    thread_id: str = "t1",
):
    """Collect stream_agent_events output for synchronous tests."""

    async def _run():
        events = []
        async for ev in stream_agent_events(
            agent,
            message,
            thread_id,
        ):
            events.append(ev)
        return events

    return run_async(_run())


def protocol_event(
    method: str,
    data,
    namespace: Iterable[Any] = (),
    **params,
) -> dict[str, Any]:
    """Build a minimal DeepAgents v3 protocol event."""
    return {
        "type": "event",
        "method": method,
        "params": {
            "namespace": list(namespace),
            "timestamp": 0,
            "data": data,
            **params,
        },
    }


def message_delta(
    text: str,
    metadata: dict[str, Any] | None = None,
    namespace: Iterable[Any] = (),
) -> dict[str, Any]:
    return protocol_event(
        "messages",
        (
            {
                "event": "content-block-delta",
                "index": 0,
                "delta": {"type": "text-delta", "text": text},
            },
            metadata or {},
        ),
        namespace,
    )


def message_finish(
    usage: dict[str, int] | None = None,
    metadata: dict[str, Any] | None = None,
    namespace: Iterable[Any] = (),
) -> dict[str, Any]:
    payload: dict[str, Any] = {"event": "message-finish"}
    if usage is not None:
        payload["usage"] = usage
    return protocol_event("messages", (payload, metadata or {}), namespace)


def message_tool_call_block(
    name: str,
    args: dict[str, Any] | None = None,
    *,
    tool_call_id: str = "tc1",
    namespace: Iterable[Any] = (),
) -> dict[str, Any]:
    return protocol_event(
        "messages",
        (
            {
                "event": "content-block-finish",
                "content": {
                    "type": "tool_call",
                    "id": tool_call_id,
                    "name": name,
                    "args": args or {},
                },
            },
            {},
        ),
        namespace,
    )


def tool_started(
    name: str,
    args: dict[str, Any] | None = None,
    *,
    tool_call_id: str = "tc1",
    namespace: Iterable[Any] = (),
) -> dict[str, Any]:
    return protocol_event(
        "tools",
        {
            "event": "tool-started",
            "tool_name": name,
            "input": args or {},
            "tool_call_id": tool_call_id,
        },
        namespace,
    )


def tool_finished(
    output,
    *,
    tool_call_id: str = "tc1",
    namespace: Iterable[Any] = (),
) -> dict[str, Any]:
    return protocol_event(
        "tools",
        {
            "event": "tool-finished",
            "output": output,
            "tool_call_id": tool_call_id,
        },
        namespace,
    )


@dataclass
class FakeStateSnapshot:
    values: dict[str, Any] = field(default_factory=dict)


class FakeV3Run:
    def __init__(self, events: Iterable[Any], subagents: Iterable[Any] | None = None):
        self._events = list(events)
        self.subagents = async_iter(list(subagents or []))
        self.aborted = False

    def __aiter__(self):
        return async_iter(self._events)

    async def abort(self) -> None:
        self.aborted = True


class FakeV3Agent:
    def __init__(
        self,
        events: Iterable[Any],
        *,
        state_values: dict[str, Any] | None = None,
        subagents: Iterable[Any] | None = None,
    ):
        self._run = FakeV3Run(events, subagents=subagents)
        self.astream_events = MagicMock(return_value=self._run)
        self._state_values = state_values if state_values is not None else {}

    async def aget_state(self, _config):
        return FakeStateSnapshot(values=self._state_values)


class ErroringV3Agent:
    def __init__(self, exc: Exception):
        self.exc = exc

    def astream_events(self, *_args, **_kwargs):
        raise self.exc

    async def aget_state(self, _config):
        return FakeStateSnapshot()


class HangingV3Run:
    def __init__(self, events: Iterable[Any]):
        self._events = list(events)
        self.subagents = async_iter([])
        self.aborted = False

    async def abort(self) -> None:
        self.aborted = True

    def __aiter__(self):
        return self._iter_events()

    async def _iter_events(self):
        for event in self._events:
            yield event
        await asyncio.Event().wait()


class HangingV3Agent:
    def __init__(self, events: Iterable[Any]):
        self._run = HangingV3Run(events)
        self.astream_events = MagicMock(return_value=self._run)

    @property
    def aborted(self) -> bool:
        return self._run.aborted

    async def aget_state(self, _config):
        return FakeStateSnapshot()


class LazySubagentChannel:
    """Projection fake that only queues handles after subscription."""

    def __init__(self, subagents: Iterable[Any]):
        self._subagents = list(subagents)
        self.subscribed = False

    def __aiter__(self):
        self.subscribed = True
        return async_iter(self._subagents)

    def drop_if_unsubscribed(self) -> None:
        if not self.subscribed:
            self._subagents = []


class SubscriptionSensitiveV3Run:
    def __init__(self, events: Iterable[Any], subagents: Iterable[Any]):
        self._events = list(events)
        self.subagents = LazySubagentChannel(subagents)

    def __aiter__(self):
        self.subagents.drop_if_unsubscribed()
        return async_iter(self._events)


class SubscriptionSensitiveV3Agent:
    def __init__(self, events: Iterable[Any], subagents: Iterable[Any]):
        self._run = SubscriptionSensitiveV3Run(events, subagents)
        self.astream_events = MagicMock(return_value=self._run)

    async def aget_state(self, _config):
        return FakeStateSnapshot()


@dataclass
class FakeSubagent:
    path: Iterable[Any]
    name: str = "research-agent"
    tool_call_id: str = ""

    def __post_init__(self):
        self.path = tuple(self.path)
        if not self.tool_call_id:
            self.tool_call_id = "call_" + "_".join(str(p) for p in self.path)

    @property
    def cause(self) -> dict[str, str]:
        return {"type": "toolCall", "tool_call_id": self.tool_call_id}

    async def output(self):
        return {"messages": []}
