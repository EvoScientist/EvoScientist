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
"""

from .emitter import StreamEventEmitter, StreamEvent
from .tracker import ToolCallTracker, ToolCallInfo
from .formatter import ToolResultFormatter, ContentType, FormattedResult
from .utils import (
    SUCCESS_PREFIX,
    FAILURE_PREFIX,
    ToolStatus,
    DisplayLimits,
    has_args,
    is_success,
    truncate,
    format_tool_compact,
    format_tree_output,
    count_lines,
    truncate_with_line_hint,
    get_status_symbol,
)
from .state import SubAgentState, StreamState, _parse_todo_items, _build_todo_stats
from .events import stream_agent_events
from .display import (
    console,
    formatter,
    format_tool_result_compact,
    create_streaming_display,
    display_final_results,
)

__all__ = [
    # Emitter
    "StreamEventEmitter",
    "StreamEvent",
    # Tracker
    "ToolCallTracker",
    "ToolCallInfo",
    # Formatter
    "ToolResultFormatter",
    "ContentType",
    "FormattedResult",
    # Utils
    "SUCCESS_PREFIX",
    "FAILURE_PREFIX",
    "ToolStatus",
    "DisplayLimits",
    "has_args",
    "is_success",
    "truncate",
    "format_tool_compact",
    "format_tree_output",
    "count_lines",
    "truncate_with_line_hint",
    "get_status_symbol",
    # State
    "SubAgentState",
    "StreamState",
    "_parse_todo_items",
    "_build_todo_stats",
    # Events
    "stream_agent_events",
    # Display
    "console",
    "formatter",
    "format_tool_result_compact",
    "create_streaming_display",
    "display_final_results",
]
