from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from .local import LocalGraphGateway, LocalThreadStore
from .types import GraphGateway, ThreadStore

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


class GraphGatewayFactory(Protocol):
    """Callable that binds a compiled graph to a graph gateway."""

    def __call__(self, agent: CompiledStateGraph) -> GraphGateway: ...


@dataclass(frozen=True, slots=True)
class RuntimeGateways:
    """Gateway handles for one CLI/TUI/serve runtime."""

    thread_store: ThreadStore
    graph_gateway_factory: GraphGatewayFactory

    def graph_gateway(self, agent: CompiledStateGraph) -> GraphGateway:
        return self.graph_gateway_factory(agent)


def create_runtime_gateways() -> RuntimeGateways:
    """Create the default gateway set for local CLI/TUI/serve execution."""
    thread_store = LocalThreadStore()

    def _graph_gateway_factory(agent: CompiledStateGraph) -> GraphGateway:
        return LocalGraphGateway(
            agent,
            thread_store=thread_store,
        )

    return RuntimeGateways(
        thread_store=thread_store,
        graph_gateway_factory=_graph_gateway_factory,
    )
