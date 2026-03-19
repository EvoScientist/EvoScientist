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

from .diff_format import build_edit_diff, format_diff_rich
from .display import (
    _astream_to_console,
    console,
    create_streaming_display,
    display_final_results,
    format_tool_result_compact,
    formatter,
)
from .emitter import StreamEvent, StreamEventEmitter
from .events import stream_agent_events
from .formatter import ContentType, FormattedResult, ToolResultFormatter
from .state import StreamState, SubAgentState, _build_todo_stats, _parse_todo_items
from .tracker import ToolCallInfo, ToolCallTracker
from .utils import (
    FAILURE_PREFIX,
    SUCCESS_PREFIX,
    DisplayLimits,
    ToolStatus,
    count_lines,
    format_tool_compact,
    format_tree_output,
    get_status_symbol,
    has_args,
    is_success,
    truncate,
    truncate_with_line_hint,
)

__all__ = [
    "FAILURE_PREFIX",
    # Utils
    "SUCCESS_PREFIX",
    "ContentType",
    "DisplayLimits",
    "FormattedResult",
    "StreamEvent",
    # Emitter
    "StreamEventEmitter",
    "StreamState",
    # State
    "SubAgentState",
    "ToolCallInfo",
    # Tracker
    "ToolCallTracker",
    # Formatter
    "ToolResultFormatter",
    "ToolStatus",
    "_astream_to_console",
    "_build_todo_stats",
    "_parse_todo_items",
    # Diff formatting
    "build_edit_diff",
    # Display
    "console",
    "count_lines",
    "create_streaming_display",
    "display_final_results",
    "format_diff_rich",
    "format_tool_compact",
    "format_tool_result_compact",
    "format_tree_output",
    "formatter",
    "get_status_symbol",
    "has_args",
    "is_success",
    # Events
    "stream_agent_events",
    "truncate",
    "truncate_with_line_hint",
]
