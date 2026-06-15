"""Graph/thread gateway abstractions.

The gateway package is the migration seam between UI surfaces and graph
execution. CLI, TUI, channels, and future frontends should depend on this
package for thread/run operations instead of reaching directly into
``sessions.py``, ``stream.events``, or the LangGraph SDK.
"""

from .local import LocalGraphGateway, LocalThreadStore
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
    "LocalGraphGateway",
    "LocalThreadStore",
    "RunRequest",
    "ThreadResolution",
    "ThreadStore",
]
