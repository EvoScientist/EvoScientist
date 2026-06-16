from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

from .local import LocalGraphGateway, LocalThreadStore
from .server import (
    DEFAULT_GRAPH_ID,
    LangGraphClientFactory,
    LangGraphServerGateway,
    LangGraphServerThreadStore,
)
from .types import GraphGateway, ThreadStore

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

RuntimeGatewayBackend = Literal["local", "langgraph_server"]
GraphGatewayFactory: TypeAlias = Callable[["CompiledStateGraph"], GraphGateway]


@dataclass(frozen=True, slots=True)
class RuntimeGateways:
    """Gateway handles for one CLI/TUI/serve runtime."""

    thread_store: ThreadStore
    graph_gateway_factory: GraphGatewayFactory

    def graph_gateway(self, agent: CompiledStateGraph) -> GraphGateway:
        return self.graph_gateway_factory(agent)


def create_runtime_gateways(
    *,
    backend: RuntimeGatewayBackend = "local",
    base_url: str | None = None,
    graph_id: str = DEFAULT_GRAPH_ID,
    headers: dict[str, str] | None = None,
    client_factory: LangGraphClientFactory | None = None,
) -> RuntimeGateways:
    """Create gateway handles for CLI/TUI/serve execution."""
    if backend == "langgraph_server":
        if base_url is None:
            raise ValueError("base_url is required for langgraph_server gateways")
        if client_factory is not None:
            server_thread_store = LangGraphServerThreadStore(
                base_url=base_url,
                graph_id=graph_id,
                headers=headers,
                client_factory=client_factory,
            )
        else:
            server_thread_store = LangGraphServerThreadStore(
                base_url=base_url,
                graph_id=graph_id,
                headers=headers,
            )

        def _server_gateway_factory(_agent: CompiledStateGraph) -> GraphGateway:
            return LangGraphServerGateway(server_thread_store, graph_id=graph_id)

        return RuntimeGateways(
            thread_store=server_thread_store,
            graph_gateway_factory=_server_gateway_factory,
        )

    if backend != "local":
        raise ValueError(f"Unsupported runtime gateway backend: {backend}")

    local_thread_store = LocalThreadStore()

    def _local_gateway_factory(agent: CompiledStateGraph) -> GraphGateway:
        return LocalGraphGateway(
            agent,
            thread_store=local_thread_store,
        )

    return RuntimeGateways(
        thread_store=local_thread_store,
        graph_gateway_factory=_local_gateway_factory,
    )
