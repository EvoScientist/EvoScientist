from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from langgraph_sdk import get_client
from langgraph_sdk.client import LangGraphClient

from .composite import CompositeGraphGateway
from .local import LocalGraphGateway, LocalThreadStore
from .server import (
    DEFAULT_GRAPH_ID,
    LangGraphServerGateway,
    LangGraphServerThreadStore,
)
from .types import GraphGateway, ThreadStore

if TYPE_CHECKING:
    from ..middleware.events import SessionEvents

RuntimeGatewayBackend = Literal["local", "langgraph_server"]


@dataclass(frozen=True, slots=True)
class RuntimeGateways:
    """Gateway handles for one CLI/TUI/serve runtime."""

    thread_store: ThreadStore
    graph_gateway: GraphGateway


def create_runtime_gateways(
    *,
    backend: RuntimeGatewayBackend = "local",
    read_backend: RuntimeGatewayBackend | None = None,
    base_url: str | None = None,
    graph_id: str = DEFAULT_GRAPH_ID,
    headers: dict[str, str] | None = None,
    langgraph_client: LangGraphClient | None = None,
    events: SessionEvents | None = None,
) -> RuntimeGateways:
    """Create gateway handles for CLI/TUI/serve execution.

    ``events`` is the frontend event sink; it is attached to the local gateway
    so the streaming path shares the same sink instance the frontend injects
    into the agent's middleware. On the server backend it is the delivery
    point for middleware events mirrored over the run's ``custom`` stream
    channel.

    ``read_backend`` splits reads from execution (Stage 4 of #432): when it is
    ``None`` or equal to ``backend`` the gateway is a single uniform backend (the
    original behavior, byte-identical for existing callers). The only supported
    split is ``read_backend="local"`` with ``backend="langgraph_server"``, which
    reads threads from the local SQLite store and executes on the langgraph
    server via :class:`CompositeGraphGateway`; ``thread_store`` stays the local
    store so catalog consumers enumerate locally.
    """
    if read_backend is not None and read_backend != backend:
        if not (read_backend == "local" and backend == "langgraph_server"):
            raise ValueError(
                "Unsupported read/execute backend split: "
                f"read={read_backend!r}, execute={backend!r}"
            )
        if base_url is None and langgraph_client is None:
            raise ValueError("base_url is required for langgraph_server gateways")
        local_thread_store = LocalThreadStore()
        server_thread_store = LangGraphServerThreadStore(
            client=langgraph_client
            if langgraph_client is not None
            else get_client(url=base_url, headers=headers),
            graph_id=graph_id,
        )
        return RuntimeGateways(
            thread_store=local_thread_store,
            graph_gateway=CompositeGraphGateway(
                read=LocalGraphGateway(thread_store=local_thread_store, events=None),
                execute=LangGraphServerGateway(
                    server_thread_store,
                    graph_id=graph_id,
                    events=events,
                ),
            ),
        )

    if backend == "langgraph_server":
        if base_url is None and langgraph_client is None:
            raise ValueError("base_url is required for langgraph_server gateways")
        server_thread_store = LangGraphServerThreadStore(
            client=langgraph_client
            if langgraph_client is not None
            else get_client(url=base_url, headers=headers),
            graph_id=graph_id,
        )

        return RuntimeGateways(
            thread_store=server_thread_store,
            graph_gateway=LangGraphServerGateway(
                server_thread_store,
                graph_id=graph_id,
                events=events,
            ),
        )

    if backend != "local":
        raise ValueError(f"Unsupported runtime gateway backend: {backend}")

    local_thread_store = LocalThreadStore()

    return RuntimeGateways(
        thread_store=local_thread_store,
        graph_gateway=LocalGraphGateway(thread_store=local_thread_store, events=events),
    )
