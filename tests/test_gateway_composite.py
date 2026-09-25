"""Routing tests for CompositeGraphGateway (read local, execute server)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from EvoScientist.gateway import (
    LocalGraphGateway,
    LocalThreadStore,
    RunRequest,
    create_runtime_gateways,
    create_runtime_gateways_for_config,
)
from EvoScientist.gateway.composite import (
    CompositeGraphGateway,
    LegacyThreadServerExecutionError,
)
from EvoScientist.gateway.types import ThreadResolution

UUID = "11111111-1111-1111-1111-111111111111"
LEGACY = "abcd1234"  # 8-hex, pre-migration CLI id (not a UUID)


async def _empty_async_iter():
    if False:  # pragma: no cover - never yields
        yield


async def _drain(agen, sink: list) -> None:
    """Consume an async event stream into ``sink`` (keeps pytest.raises simple)."""
    async for event in agen:
        sink.append(event)


class RecordingGateway:
    """Minimal GraphGateway that records which methods were called."""

    def __init__(
        self,
        name: str,
        *,
        threads: list[dict[str, Any]] | None = None,
        resolution: ThreadResolution | None = None,
        metadata: dict[str, Any] | None = None,
        messages: list[Any] | None = None,
        exists: bool = False,
        delete_result: bool = True,
        state: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.calls: list[tuple] = []
        self.events = None
        self.delete_raises = False
        self._threads = threads or []
        self._resolution = resolution or ThreadResolution(None, ())
        self._metadata = metadata
        self._messages = messages or []
        self._exists = exists
        self._delete_result = delete_result
        self._state = state or {}

    def called(self, method: str) -> bool:
        return any(c[0] == method for c in self.calls)

    async def list_threads(
        self,
        *,
        limit=20,
        include_message_count=False,
        include_preview=False,
        target=None,
    ):
        self.calls.append(("list_threads",))
        return list(self._threads)

    async def resolve_thread(self, tid, target=None):
        self.calls.append(("resolve_thread", tid))
        return self._resolution

    async def get_thread_metadata(self, tid, target=None):
        self.calls.append(("get_thread_metadata", tid))
        return self._metadata

    async def get_thread_messages(self, tid, target=None):
        self.calls.append(("get_thread_messages", tid))
        return list(self._messages)

    async def get_state_values(self, target, tid):
        self.calls.append(("get_state_values", tid))
        return dict(self._state)

    async def thread_exists(self, tid, target=None):
        self.calls.append(("thread_exists", tid))
        return self._exists

    async def delete_thread(self, tid, target=None):
        self.calls.append(("delete_thread", tid))
        if self.delete_raises:
            raise RuntimeError("boom")
        return self._delete_result

    async def create_thread(self, target=None, *, metadata=None):
        self.calls.append(("create_thread",))
        return f"{self.name}-created"

    async def clone_thread(self, src, *, metadata=None, target=None):
        self.calls.append(("clone_thread", src))
        return f"{self.name}-clone"

    def stream_events(self, request):
        self.calls.append(("stream_events", request.thread_id))
        return _empty_async_iter()

    async def update_state_values(self, target, tid, values):
        self.calls.append(("update_state_values", tid))

    async def get_run_status(self, target, tid, run_id):
        self.calls.append(("get_run_status", tid))
        return "success"

    async def get_process_status(self, target, tid, pid):
        self.calls.append(("get_process_status", tid))
        return "running"


def _composite(
    read: RecordingGateway, execute: RecordingGateway
) -> CompositeGraphGateway:
    return CompositeGraphGateway(read=read, execute=execute)


# --- Catalog / read routing (local reader only) -------------------------------
# Both backends write the same sessions.db, so the local reader already sees
# server-executed threads; the composite never consults the server for a read.


async def test_list_threads_reads_local_only():
    read = RecordingGateway("read", threads=[{"thread_id": "A"}, {"thread_id": "B"}])
    execute = RecordingGateway("exec", threads=[{"thread_id": "C"}])
    merged = await _composite(read, execute).list_threads(limit=0)
    assert [e["thread_id"] for e in merged] == ["A", "B"]
    assert read.called("list_threads")
    assert not execute.called("list_threads")


async def test_resolve_uuid_reads_local_only():
    read = RecordingGateway("read", resolution=ThreadResolution(UUID, ()))
    execute = RecordingGateway("exec")
    res = await _composite(read, execute).resolve_thread(UUID)
    assert res.thread_id == UUID
    assert not execute.called("resolve_thread")


async def test_resolve_uuid_miss_does_not_hit_server():
    # A local miss is authoritative (shared db); the server is not a second
    # place to look, even for a UUID it happens to know.
    read = RecordingGateway("read", resolution=ThreadResolution(None, ()))
    execute = RecordingGateway("exec", resolution=ThreadResolution(UUID, ()))
    res = await _composite(read, execute).resolve_thread(UUID)
    assert res.thread_id is None
    assert not execute.called("resolve_thread")


async def test_resolve_legacy_miss_does_not_hit_server():
    read = RecordingGateway("read", resolution=ThreadResolution(None, ()))
    execute = RecordingGateway("exec")
    res = await _composite(read, execute).resolve_thread(LEGACY)
    assert res.thread_id is None
    assert not execute.called("resolve_thread")


async def test_metadata_reads_local_only():
    read = RecordingGateway("read", metadata={"src": "local"})
    execute = RecordingGateway("exec", metadata={"src": "server"})
    comp = _composite(read, execute)
    assert (await comp.get_thread_metadata(UUID)) == {"src": "local"}
    assert (await comp.get_thread_metadata(LEGACY)) == {"src": "local"}
    assert not execute.called("get_thread_metadata")


async def test_messages_read_local_only():
    read = RecordingGateway("read", messages=["local"])
    execute = RecordingGateway("exec", messages=["server"])
    comp = _composite(read, execute)
    assert (await comp.get_thread_messages(UUID)) == ["local"]
    assert (await comp.get_thread_messages(LEGACY)) == ["local"]
    assert not execute.called("get_thread_messages")


async def test_state_values_uuid_server_legacy_local():
    # get_state_values is the one read kept on the server: it reflects the run's
    # live next / interrupts, which only the executor holds.
    read = RecordingGateway("read", state={"src": "local"})
    execute = RecordingGateway("exec", state={"src": "server"})
    comp = _composite(read, execute)
    assert (await comp.get_state_values(None, UUID)) == {"src": "server"}
    assert (await comp.get_state_values(None, LEGACY)) == {"src": "local"}


async def test_thread_exists_reads_local_only():
    # local hit -> True, server never consulted
    comp = _composite(RecordingGateway("r", exists=True), RecordingGateway("e"))
    assert await comp.thread_exists(UUID) is True
    # local miss -> False for both UUID and legacy, server never consulted
    read = RecordingGateway("r", exists=False)
    execute = RecordingGateway("e", exists=True)
    comp2 = _composite(read, execute)
    assert await comp2.thread_exists(UUID) is False
    assert await comp2.thread_exists(LEGACY) is False
    assert not execute.called("thread_exists")


# --- Execution routing --------------------------------------------------------


async def test_stream_events_uuid_routes_to_server():
    read = RecordingGateway("r")
    execute = RecordingGateway("e")
    # Async generator: delegation happens on iteration, not at call time.
    async for _ in _composite(read, execute).stream_events(
        RunRequest(message="hi", thread_id=UUID)
    ):
        pass
    assert execute.called("stream_events")


async def test_stream_events_legacy_yields_error_then_raises():
    read = RecordingGateway("r")
    execute = RecordingGateway("e")
    events: list = []
    stream = _composite(read, execute).stream_events(
        RunRequest(message="hi", thread_id=LEGACY)
    )
    with pytest.raises(LegacyThreadServerExecutionError):
        await _drain(stream, events)
    # An error event reaches the consumer before the raise, so stream-json emits
    # it instead of an empty stream; the server is never touched.
    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert LEGACY in events[0]["message"]
    assert not execute.called("stream_events")


async def test_stream_json_sink_emits_error_line_for_legacy_thread():
    """The stream-json sink writes an error line, not an empty stream.

    Regression for the raise-at-call-time bug: with a plain method the guard
    escaped before ``write_events_as_json`` iterated, leaving stdout empty.
    """
    import io
    import json as _json

    from EvoScientist.stream.json_sink import write_events_as_json

    comp = _composite(RecordingGateway("r"), RecordingGateway("e"))
    out = io.StringIO()
    with pytest.raises(LegacyThreadServerExecutionError):
        await write_events_as_json(
            comp.stream_events(RunRequest(message="hi", thread_id=LEGACY)), out
        )
    lines = [line for line in out.getvalue().splitlines() if line]
    assert len(lines) == 1
    assert _json.loads(lines[0])["type"] == "error"


async def test_update_state_values_legacy_refused():
    read = RecordingGateway("r")
    execute = RecordingGateway("e")
    comp = _composite(read, execute)
    with pytest.raises(LegacyThreadServerExecutionError):
        await comp.update_state_values(None, LEGACY, {})
    assert not execute.called("update_state_values")
    await comp.update_state_values(None, UUID, {})
    assert execute.called("update_state_values")


async def test_clone_thread_uuid_server_legacy_refused():
    read = RecordingGateway("r")
    execute = RecordingGateway("e")
    comp = _composite(read, execute)
    assert (await comp.clone_thread(UUID)) == "e-clone"
    with pytest.raises(LegacyThreadServerExecutionError):
        await comp.clone_thread(LEGACY)


async def test_create_thread_routes_to_server():
    read = RecordingGateway("r")
    execute = RecordingGateway("e")
    assert (await _composite(read, execute).create_thread()) == "e-created"
    assert not read.called("create_thread")


async def test_run_and_process_status_route_to_server():
    read = RecordingGateway("r")
    execute = RecordingGateway("e")
    comp = _composite(read, execute)
    assert (await comp.get_run_status(None, UUID, "run1")) == "success"
    assert (await comp.get_process_status(None, UUID, "proc1")) == "running"
    assert not read.called("get_run_status")


# --- delete fan-out -----------------------------------------------------------


async def test_delete_fans_out_success_if_either():
    read = RecordingGateway("r", delete_result=False)
    execute = RecordingGateway("e", delete_result=True)
    assert await _composite(read, execute).delete_thread(UUID) is True
    assert read.called("delete_thread")
    assert execute.called("delete_thread")


async def test_delete_both_miss_returns_false():
    read = RecordingGateway("r", delete_result=False)
    execute = RecordingGateway("e", delete_result=False)
    assert await _composite(read, execute).delete_thread(UUID) is False


async def test_delete_legacy_local_only():
    read = RecordingGateway("r", delete_result=True)
    execute = RecordingGateway("e", delete_result=True)
    assert await _composite(read, execute).delete_thread(LEGACY) is True
    assert not execute.called("delete_thread")


async def test_delete_server_error_is_non_fatal():
    read = RecordingGateway("r", delete_result=True)
    execute = RecordingGateway("e", delete_result=True)
    execute.delete_raises = True
    assert await _composite(read, execute).delete_thread(UUID) is True


def test_execute_gateway_exposes_execution_side():
    read = RecordingGateway("r")
    execute = RecordingGateway("e")
    assert _composite(read, execute).execute_gateway is execute


async def test_events_property_reflects_execution_side():
    read = RecordingGateway("r")
    execute = RecordingGateway("e")
    execute.events = "sink"
    comp = _composite(read, execute)
    assert comp.events == "sink"
    comp.events = "new"
    assert execute.events == "new"


# --- runtime.py wiring --------------------------------------------------------


def test_create_runtime_gateways_split_builds_composite():
    gws = create_runtime_gateways(
        backend="langgraph_server",
        read_backend="local",
        langgraph_client=object(),
    )
    assert isinstance(gws.graph_gateway, CompositeGraphGateway)
    assert isinstance(gws.thread_store, LocalThreadStore)


def test_create_runtime_gateways_passthrough_unchanged():
    gws = create_runtime_gateways(backend="local", read_backend="local")
    assert isinstance(gws.graph_gateway, LocalGraphGateway)


def test_create_runtime_gateways_rejects_unsupported_split():
    with pytest.raises(ValueError, match="Unsupported read/execute backend split"):
        create_runtime_gateways(backend="local", read_backend="langgraph_server")


def test_gateways_for_config_local_backend():
    gws = create_runtime_gateways_for_config(SimpleNamespace(gateway_backend="local"))
    assert isinstance(gws.graph_gateway, LocalGraphGateway)


def test_gateways_for_config_defaults_to_local_when_field_absent():
    gws = create_runtime_gateways_for_config(SimpleNamespace())
    assert isinstance(gws.graph_gateway, LocalGraphGateway)


def _stub_dev_server_up(monkeypatch, *, available: bool = True) -> None:
    """Stub the dev-server URL and its availability flag for factory tests.

    The factory only builds the composite when ``is_async_subagents_available()``
    is True (the dev server actually came up); otherwise it falls back to local.
    """
    import EvoScientist.langgraph_dev.manager as manager
    import EvoScientist.langgraph_dev.sdk as sdk

    monkeypatch.setattr(
        sdk, "configured_langgraph_dev_url", lambda: "http://localhost:2024"
    )
    monkeypatch.setattr(manager, "_ASYNC_SUBAGENTS_AVAILABLE", available)


def test_gateways_for_config_server_backend_builds_composite(monkeypatch):
    _stub_dev_server_up(monkeypatch)
    gws = create_runtime_gateways_for_config(
        SimpleNamespace(gateway_backend="langgraph_server")
    )
    assert isinstance(gws.graph_gateway, CompositeGraphGateway)
    assert isinstance(gws.thread_store, LocalThreadStore)


def test_gateways_for_config_server_backend_falls_back_when_dev_unavailable(
    monkeypatch,
):
    """Server backend + no live dev server -> in-process gateway, not a dead URL.

    ``ensure_langgraph_dev`` soft-fails (leaves the availability flag False) when
    the port is held or the subprocess dies; the factory must degrade to local
    rather than build the composite against an unreachable server.
    """
    _stub_dev_server_up(monkeypatch, available=False)
    gws = create_runtime_gateways_for_config(
        SimpleNamespace(gateway_backend="langgraph_server")
    )
    assert isinstance(gws.graph_gateway, LocalGraphGateway)


def test_gateways_for_config_explicit_backend_overrides_global(monkeypatch):
    """A surface's resolved ``backend`` wins over the config's global flag.

    The global is ``local`` but the surface resolved ``langgraph_server`` (e.g.
    a per-surface override), so the factory must build the composite from the
    explicit backend, not re-read the global and pick local.
    """
    _stub_dev_server_up(monkeypatch)
    gws = create_runtime_gateways_for_config(
        SimpleNamespace(gateway_backend="local"),
        backend="langgraph_server",
    )
    assert isinstance(gws.graph_gateway, CompositeGraphGateway)


def test_gateways_for_config_explicit_local_backend_over_server_global():
    """Explicit ``local`` builds the in-process gateway even if global is server."""
    gws = create_runtime_gateways_for_config(
        SimpleNamespace(gateway_backend="langgraph_server"),
        backend="local",
    )
    assert isinstance(gws.graph_gateway, LocalGraphGateway)


def test_gateways_for_config_backend_none_falls_back_to_global():
    """``backend=None`` keeps the pre-per-surface behavior (read the global)."""
    gws = create_runtime_gateways_for_config(
        SimpleNamespace(gateway_backend="local"), backend=None
    )
    assert isinstance(gws.graph_gateway, LocalGraphGateway)
