"""Graph/thread gateway abstractions.

The gateway package is the migration seam between UI surfaces and graph
execution. CLI, TUI, channels, and future frontends should depend on this
package for thread/run operations instead of reaching directly into
``sessions.py``, ``stream.events``, or the LangGraph SDK.
"""

from .local import LocalGraphGateway, LocalThreadStore
from .runtime import (
    GraphGatewayFactory,
    RuntimeGatewayBackend,
    RuntimeGateways,
    create_runtime_gateways,
)
from .server import (
    DEFAULT_GRAPH_ID,
    LangGraphServerGateway,
    LangGraphServerThreadStore,
)
from .types import (
    GraphEvent,
    GraphGateway,
    GraphRunInput,
    RunRequest,
    ThreadResolution,
    ThreadStore,
)

__all__ = [
    "DEFAULT_GRAPH_ID",
    "GraphEvent",
    "GraphGateway",
    "GraphGatewayFactory",
    "GraphRunInput",
    "LangGraphServerGateway",
    "LangGraphServerThreadStore",
    "LocalGraphGateway",
    "LocalThreadStore",
    "RunRequest",
    "RuntimeGatewayBackend",
    "RuntimeGateways",
    "ThreadResolution",
    "ThreadStore",
    "create_runtime_gateways",
]
