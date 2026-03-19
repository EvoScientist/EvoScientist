"""
Stream utility functions and constants.

Provides tool status indicators, display limits, and formatting helpers
adapted for deepagents tool names.
"""

import sys
from enum import StrEnum
from pathlib import PurePath

# === Status marker constants ===
SUCCESS_PREFIX = "[OK]"
FAILURE_PREFIX = "[FAILED]"


# === Tool status indicators ===
class ToolStatus(StrEnum):
    """Tool execution status indicators."""

    RUNNING = "\u25cf"  # Running - yellow
    SUCCESS = "\u25cf"  # Success - green
    ERROR = "\u25cf"  # Failed - red
    PENDING = "\u25cb"  # Pending - gray


def get_status_symbol(status: ToolStatus) -> str:
    """Get status symbol with ASCII fallback for terminals without Unicode."""
    try:
        supports_unicode = sys.stdout.encoding and "utf" in sys.stdout.encoding.lower()
    except Exception:
        supports_unicode = False

    if supports_unicode:
        return status.value

    fallback = {
        ToolStatus.RUNNING: "*",
        ToolStatus.SUCCESS: "+",
        ToolStatus.ERROR: "x",
        ToolStatus.PENDING: "-",
    }
    return fallback.get(status, "?")


# === Display limit constants ===
class DisplayLimits:
    """Display length limits."""

    THINKING_STREAM = 1000
    THINKING_FINAL = 2000
    ARGS_INLINE = 100
    ARGS_FORMATTED = 300
    TOOL_RESULT_STREAM = 500
    TOOL_RESULT_FINAL = 800
    TOOL_RESULT_MAX = 2000


def has_args(args) -> bool:
    """Check if args has content (handles empty dict falsy issue)."""
    return args is not None and args != {}


def is_success(content: str) -> bool:
    """Determine if tool output indicates successful execution.

    Only checks the first few lines of output for error patterns to avoid
    false positives from code/docs that contain "Error:" as literal text.
    """
    content = content.strip()
    if content.startswith(SUCCESS_PREFIX):
        return True
    if content.startswith(FAILURE_PREFIX):
        return False
    # Only check the first 3 lines — real tool errors appear at the top,
    # not buried deep inside file content or command output.
    head = "\n".join(content.splitlines()[:3])
    error_patterns = [
        "Traceback (most recent call last)",
        "Exception:",
        "Error:",
        "Error invoking tool",
        "Failed ",
    ]
    return not any(pattern in head for pattern in error_patterns)


def truncate(content: str, max_length: int, suffix: str = "\n... (truncated)") -> str:
    """Truncate content to specified length."""
    if len(content) > max_length:
        return content[:max_length] + suffix
    return content


# === Compact formatting for deepagents tools ===


def _shorten_path(path: str, max_len: int = 40) -> str:
    """Shorten a file path for display."""
    if len(path) <= max_len:
        return path
    path_obj = PurePath(path)
    parts = path_obj.parts
    if len(parts) > 2:
        return ".../" + "/".join(parts[-2:])
    return path


def format_tool_compact(name: str, args: dict | None) -> str:
    """Format as compact tool call string: ToolName(key_arg).

    Adapted for deepagents tool names: execute, read_file, write_file,
    edit_file, grep, glob, ls, write_todos, read_todos, task,
    tavily_search, think_tool.
    """
    if not args:
        return f"{name}()"

    name_lower = name.lower()

    # Shell execution
    if name_lower == "execute":
        cmd = args.get("command", "")
        if len(cmd) > 50:
            cmd = cmd[:47] + "\u2026"
        return f"execute({cmd})"

    # File operations (with special case for memory files)
    if name_lower == "read_file":
        path = args.get("path", "")
        if path.endswith("/MEMORY.md") or path == "/MEMORY.md":
            return "Reading memory"
        return f"read_file({_shorten_path(path)})"

    if name_lower == "write_file":
        path = args.get("path", "")
        if path.endswith("/MEMORY.md") or path == "/MEMORY.md":
            return "Updating memory"
        return f"write_file({_shorten_path(path)})"

    if name_lower == "edit_file":
        path = args.get("path", "")
        if path.endswith("/MEMORY.md") or path == "/MEMORY.md":
            return "Updating memory"
        return f"edit_file({_shorten_path(path)})"

    # Search operations
    if name_lower == "glob":
        pattern = args.get("pattern", "")
        if len(pattern) > 40:
            pattern = pattern[:37] + "\u2026"
        return f"glob({pattern})"

    if name_lower == "grep":
        pattern = args.get("pattern", "")
        path = args.get("path", ".")
        if len(pattern) > 30:
            pattern = pattern[:27] + "\u2026"
        return f"grep({pattern}, {path})"

    # Directory listing
    if name_lower == "ls":
        path = args.get("path", ".")
        return f"ls({path})"

    # Todo management
    if name_lower == "write_todos":
        todos = args.get("todos", [])
        if isinstance(todos, list):
            return f"write_todos({len(todos)} items)"
        return "write_todos(...)"

    if name_lower == "read_todos":
        return "read_todos()"

    # Sub-agent delegation — display as "Cooking with {agent}" instead of "task()"
    if name_lower == "task":
        sa_type = args.get("subagent_type", "").strip()
        task_desc = args.get("description", args.get("task", "")).strip()
        task_desc = task_desc.split("\n")[0].strip() if task_desc else ""
        if sa_type:
            if task_desc:
                if len(task_desc) > 50:
                    task_desc = task_desc[:47] + "\u2026"
                return f"Cooking with {sa_type} — {task_desc}"
            return f"Cooking with {sa_type}"
        # Fallback if no subagent_type
        if task_desc:
            if len(task_desc) > 50:
                task_desc = task_desc[:47] + "\u2026"
            return f"Cooking with sub-agent — {task_desc}"
        return "Cooking with sub-agent"

    # Web search
    if name_lower in ("tavily_search", "internet_search"):
        query = args.get("query", "")
        if len(query) > 40:
            query = query[:37] + "\u2026"
        return f"{name}({query})"

    # Think/reflection
    if name_lower == "think_tool":
        reflection = args.get("reflection", "")
        if len(reflection) > 40:
            reflection = reflection[:37] + "\u2026"
        return f"think_tool({reflection})"

    # Default: show first few params
    params = []
    for k, v in list(args.items())[:2]:
        v_str = str(v)
        if len(v_str) > 20:
            v_str = v_str[:17] + "\u2026"
        params.append(f"{k}={v_str}")

    params_str = ", ".join(params)
    if len(params_str) > 50:
        params_str = params_str[:47] + "\u2026"

    return f"{name}({params_str})"


def format_tree_output(lines: list[str], max_lines: int = 5, indent: str = "  ") -> str:
    """Format output as tree structure.

    Example:
        └ On branch main
          Your branch is up to date
          ... +16 lines
    """
    if not lines:
        return ""

    result = []
    display_lines = lines[:max_lines]

    for i, line in enumerate(display_lines):
        prefix = "\u2514" if i == 0 else " "
        result.append(f"{indent}{prefix} {line}")

    remaining = len(lines) - max_lines
    if remaining > 0:
        result.append(f"{indent}  ... +{remaining} lines")

    return "\n".join(result)


def count_lines(content: str) -> int:
    """Count number of lines in content."""
    if not content:
        return 0
    return len(content.strip().split("\n"))


def truncate_with_line_hint(content: str, max_lines: int = 5) -> tuple[str, int]:
    """Truncate by line count, returning remaining line count."""
    lines = content.strip().split("\n")
    total = len(lines)

    if total <= max_lines:
        return content.strip(), 0

    truncated = "\n".join(lines[:max_lines])
    remaining = total - max_lines
    return truncated, remaining
