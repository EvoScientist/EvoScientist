"""Graph/thread gateway abstractions.

The gateway package is the migration seam between UI surfaces and graph
execution. CLI, TUI, channels, and future frontends should depend on this
package for thread/run operations instead of reaching directly into
``sessions.py``, ``stream.events``, or the LangGraph SDK.
"""

from .local import LocalGraphGateway, LocalThreadStore
from .runtime import GraphGatewayFactory, RuntimeGateways, create_runtime_gateways
from .types import (
    GraphEvent,
    GraphGateway,
    RunRequest,
    ThreadResolution,
    ThreadStore,
)

__all__ = [
    "GraphEvent",
    "GraphGateway",
    "GraphGatewayFactory",
    "LocalGraphGateway",
    "LocalThreadStore",
    "RunRequest",
    "RuntimeGateways",
    "ThreadResolution",
    "ThreadStore",
    "create_runtime_gateways",
]
