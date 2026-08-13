"""Graph/thread gateway abstractions.

The gateway package is the migration seam between UI surfaces and graph
execution. CLI, TUI, channels, and future frontends should depend on this
package for thread/run operations instead of reaching directly into
``sessions.py``, ``stream.events``, or the LangGraph SDK.

Backend implementations are attached lazily via :mod:`lazy_loader` (SPEC-1 /
PEP 562): importing the shared :mod:`.types` protocols must not cascade into
``sessions``/langgraph/langgraph_sdk, which every CLI invocation would pay.
"""

from typing import TYPE_CHECKING

import lazy_loader as _lazy

from .types import (
    DEFAULT_GRAPH_ID,
    GraphEvent,
    GraphGateway,
    GraphRunInput,
    GraphStateValues,
    GraphTarget,
    RunRequest,
    ThreadResolution,
    ThreadStore,
)

if TYPE_CHECKING:
    # Static counterparts of the lazy attach below — type checkers don't
    # infer names served through __getattr__.
    from . import background_runs
    from .local import LocalGraphGateway, LocalThreadStore
    from .runtime import (
        RuntimeGatewayBackend,
        RuntimeGateways,
        create_runtime_gateways,
    )
    from .server import (
        LangGraphServerGateway,
        LangGraphServerThreadStore,
    )

__getattr__, _attach_dir, _ = _lazy.attach(
    __name__,
    submodules=["background_runs"],
    submod_attrs={
        "local": ["LocalGraphGateway", "LocalThreadStore"],
        "runtime": [
            "RuntimeGatewayBackend",
            "RuntimeGateways",
            "create_runtime_gateways",
        ],
        "server": [
            "LangGraphServerGateway",
            "LangGraphServerThreadStore",
        ],
    },
)


def __dir__() -> list[str]:
    # attach() only knows the lazy names; include the eager type exports too.
    return sorted(set(_attach_dir()) | set(__all__))


__all__ = [
    "DEFAULT_GRAPH_ID",
    "GraphEvent",
    "GraphGateway",
    "GraphRunInput",
    "GraphStateValues",
    "GraphTarget",
    "LangGraphServerGateway",
    "LangGraphServerThreadStore",
    "LocalGraphGateway",
    "LocalThreadStore",
    "RunRequest",
    "RuntimeGatewayBackend",
    "RuntimeGateways",
    "ThreadResolution",
    "ThreadStore",
    "background_runs",
    "create_runtime_gateways",
]
