"""
Stream module - streaming event processing for CLI display.

Provides:
- StreamEventEmitter: Standardized event creation
- ToolCallTracker: Incremental JSON parsing for tool parameters
- ToolResultFormatter: Content-aware result formatting with Rich
- Utility functions and constants
- SubAgentState / StreamState: Stream state tracking
- stream_agent_events: Async event generator
- Display functions: Rich rendering for streaming and final output

Re-exports are attached lazily via :mod:`lazy_loader` (SPEC-1 / PEP 562) so
that ``import EvoScientist.stream`` (or ``from .stream.state import ...``)
does not drag in ``stream.display``/``stream.events`` — that load cascades
into ``langchain_core.messages`` and is deferred until an actually-used
symbol forces it.
"""

import lazy_loader as _lazy

__getattr__, __dir__, __all__ = _lazy.attach(
    __name__,
    submodules=[
        "console",
        "diff_format",
        "display",
        "emitter",
        "events",
        "formatter",
        "state",
        "tracker",
        "utils",
    ],
    submod_attrs={
        "console": ["console"],
        "diff_format": ["build_edit_diff", "format_diff_rich"],
        "display": [
            "_astream_to_console",
            "create_streaming_display",
            "display_final_results",
            "format_tool_result_compact",
            "formatter",
        ],
        "emitter": ["StreamEvent", "StreamEventEmitter"],
        "events": ["stream_agent_events"],
        "formatter": ["ContentType", "FormattedResult", "ToolResultFormatter"],
        "state": [
            "StreamState",
            "SubAgentState",
            "_build_todo_stats",
            "_parse_todo_items",
        ],
        "tracker": ["ToolCallInfo", "ToolCallTracker"],
        "utils": [
            "FAILURE_PREFIX",
            "SUCCESS_PREFIX",
            "DisplayLimits",
            "ToolStatus",
            "count_lines",
            "format_tool_compact",
            "format_tree_output",
            "get_status_symbol",
            "has_args",
            "is_success",
            "truncate",
            "truncate_with_line_hint",
        ],
    },
)
