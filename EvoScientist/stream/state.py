"""Stream state tracking for CLI display.

Contains SubAgentState, StreamState, and todo-item parsing helpers.
Rich is only imported lazily inside ``StreamState.get_response_markdown``
so the bulk of the module stays import-cheap and free of UI dependencies.
"""

import ast
import json
from enum import StrEnum


class ResearchPhase(StrEnum):
    """Research phase constants used by the TUI status bar."""

    IDLE = "idle"
    THINKING = "thinking"
    RESEARCHING = "researching"
    WRITING = "writing"


class SubAgentState:
    """Tracks a single sub-agent's activity."""

    def __init__(
        self,
        name: str,
        description: str = "",
        instance_id: str = "",
        parent_tool_call_id: str = "",
    ):
        self.name = name
        self.description = description
        self.instance_id = instance_id
        self.parent_tool_call_id = parent_tool_call_id
        self.tool_calls: list[dict] = []
        self.tool_results: list[dict] = []
        self._result_map: dict[str, dict] = {}  # tool_call_id -> result
        self.is_active = True

    def add_tool_call(self, name: str, args: dict, tool_id: str):
        if not name or not tool_id:
            return
        tc_data = {"id": tool_id, "name": name, "args": args}
        for i, tc in enumerate(self.tool_calls):
            if tc["id"] == tool_id:
                self.tool_calls[i]["name"] = name
                if args:
                    self.tool_calls[i]["args"] = args
                return
        self.tool_calls.append(tc_data)

    def add_tool_result(
        self,
        name: str,
        content: str,
        success: bool,
        tool_call_id: str,
    ):
        result = {
            "name": name,
            "content": content,
            "success": success,
            "tool_call_id": tool_call_id,
        }
        self.tool_results.append(result)
        for tc in self.tool_calls:
            if tc["id"] == tool_call_id:
                self._result_map[tool_call_id] = result
                return

    def get_result_for(self, tc: dict) -> dict | None:
        """Get matched result for a tool call."""
        return self._result_map.get(tc["id"])


class StreamState:
    """Accumulates stream state for display updates."""

    def __init__(self):
        self.thinking_text = ""
        self.summarization_text = ""
        self.is_summarizing = False
        self.response_text = ""
        self.tool_calls = []
        self.tool_results = []
        self.is_thinking = False
        self.is_responding = False
        self.is_processing = False
        # Sub-agent tracking
        self.subagents: list[SubAgentState] = []
        self._subagent_map: dict[str, SubAgentState] = {}
        # Todo list tracking
        self.todo_items: list[dict] = []
        # Latest text segment (reset on each tool_call)
        self.latest_text = ""
        self.narrated_response_end = 0
        self.narration_segments = []
        # Token usage tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        # Tool selection tracking (LLMToolSelectorMiddleware)
        self.selected_tools: list[str] = []
        # HITL interrupt tracking
        self.pending_interrupt: dict | None = None
        # ask_user interrupt tracking
        self.pending_ask_user: dict | None = None
        # Cached Markdown object for Rich CLI display (avoids O(n²) re-parsing)
        self._cached_md_text: str = ""
        self._cached_md: object | None = None

    def get_response_markdown(self):
        """Return cached Markdown object, only re-parsing when text changes."""
        from rich.markdown import Markdown

        from .display import _fix_markdown_heading_spacing

        text = (self.response_text or "").strip()
        if text != self._cached_md_text:
            self._cached_md_text = text
            self._cached_md = (
                Markdown(_fix_markdown_heading_spacing(text)) if text else None
            )
        return self._cached_md

    def _get_or_create_subagent(
        self,
        name: str,
        description: str,
        instance_id: str,
        parent_tool_call_id: str = "",
    ) -> SubAgentState:
        key = instance_id
        if key not in self._subagent_map:
            sa = SubAgentState(name, description, instance_id, parent_tool_call_id)
            self.subagents.append(sa)
            self._subagent_map[key] = sa
        else:
            existing = self._subagent_map[key]
            if name and existing.name != name:
                existing.name = name
            if description and not existing.description:
                existing.description = description
            if instance_id and not existing.instance_id:
                existing.instance_id = instance_id
            if parent_tool_call_id and not existing.parent_tool_call_id:
                existing.parent_tool_call_id = parent_tool_call_id
        return self._subagent_map[key]

    def handle_event(self, event: dict) -> str:
        """Process a single stream event, update internal state, return event type."""
        event_type: str = event.get("type", "")

        if event_type == "thinking":
            self.is_thinking = True
            self.is_responding = False
            self.is_processing = False
            self.thinking_text += event.get("content", "")

        elif event_type == "text":
            self.is_summarizing = False
            self.is_thinking = False
            self.is_responding = True
            self.is_processing = False
            text_content = event.get("content", "")
            self.response_text += text_content
            self.latest_text += text_content

        elif event_type == "tool_call":
            self.is_thinking = False
            self.is_responding = False
            self.is_processing = False

            tool_id = event["id"]
            tool_name = event.get("name", "unknown")
            tool_args = event.get("args", {})
            tc_data = {
                "id": tool_id,
                "name": tool_name,
                "args": tool_args,
            }

            updated = False
            for i, tc in enumerate(self.tool_calls):
                if tc["id"] == tool_id:
                    self.tool_calls[i] = tc_data
                    updated = True
                    break
            if not updated:
                if self.latest_text.strip():
                    self.narration_segments.append(
                        (len(self.tool_calls), self.latest_text)
                    )
                    self.narrated_response_end = len(self.response_text)
                self.tool_calls.append(tc_data)

            self.latest_text = ""  # Reset -- next text segment is a new message

            # Capture todo items from write_todos args (most reliable source)
            if tool_name == "write_todos":
                todos = tool_args.get("todos", [])
                if isinstance(todos, list) and todos:
                    self.todo_items = todos

        elif event_type == "tool_result":
            result_name = event.get("name", "unknown")
            self.is_processing = True
            result_content = event.get("content", "")
            self.tool_results.append(
                {
                    "name": result_name,
                    "content": result_content,
                    "id": event["id"],
                    "success": event.get("success", True),
                }
            )
            # Update todo list from write_todos / read_todos tool results.
            if result_name in ("write_todos", "read_todos"):
                parsed = _parse_todo_items(result_content)
                if parsed:
                    self.todo_items = parsed

        elif event_type == "subagent_start":
            name = event["name"]
            desc = event.get("description", "")
            instance_id = event["instance_id"]
            sa = self._get_or_create_subagent(
                name, desc, instance_id, event["tool_call_id"]
            )
            sa.is_active = True

        elif event_type == "subagent_tool_call":
            instance_id = event["instance_id"]
            sa = self._subagent_map.get(instance_id)
            if sa is None:
                return event_type
            sa.add_tool_call(
                event.get("name", "unknown"),
                event.get("args", {}),
                event["id"],
            )

        elif event_type == "subagent_tool_result":
            instance_id = event["instance_id"]
            sa = self._subagent_map.get(instance_id)
            if sa is None:
                return event_type
            sa.add_tool_result(
                event.get("name", "unknown"),
                event.get("content", ""),
                event.get("success", True),
                event["id"],
            )

        elif event_type == "subagent_end":
            instance_id = event["instance_id"]
            key = instance_id
            if key in self._subagent_map:
                self._subagent_map[key].is_active = False

        elif event_type == "interrupt":
            self.pending_interrupt = event

        elif event_type == "ask_user":
            self.pending_ask_user = event

        elif event_type == "tool_selection":
            self.selected_tools = event.get("tools", [])

        elif event_type == "summarization_start":
            self.is_summarizing = True

        elif event_type == "summarization":
            self.is_summarizing = True
            self.summarization_text += event.get("content", "")

        elif event_type == "usage_stats":
            try:
                input_tokens = max(0, int(event.get("input_tokens") or 0))
            except (TypeError, ValueError):
                input_tokens = 0
            try:
                output_tokens = max(0, int(event.get("output_tokens") or 0))
            except (TypeError, ValueError):
                output_tokens = 0
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            if input_tokens > 0:
                self.last_input_tokens = input_tokens
            if output_tokens > 0:
                self.last_output_tokens = output_tokens

        elif event_type == "done":
            self.is_summarizing = False
            self.is_processing = False
            if not self.response_text:
                self.response_text = event.get("response", "")

        elif event_type == "error":
            self.is_summarizing = False
            self.is_processing = False
            self.is_thinking = False
            self.is_responding = False
            error_msg = event.get("message", "Unknown error")
            self.response_text += f"\n\n[Error] {error_msg}"

        return event_type

    def visible_tool_counts(self) -> tuple[int, int]:
        """Return (completed, total) counts for tool calls."""
        n_total = len(self.tool_calls)
        return min(len(self.tool_results), n_total), n_total

    def has_pending_work(self) -> bool:
        """Return True if tools or sub-agents are still running."""
        n_done, n_visible = self.visible_tool_counts()
        has_pending = n_visible > n_done
        any_active_sa = any(sa.is_active for sa in self.subagents)
        return has_pending or any_active_sa or self.is_processing

    def compute_phase(self) -> ResearchPhase:
        """Derive the current research phase from internal state.

        Returns:
            A ``ResearchPhase`` enum member.
        """
        if self.is_thinking:
            return ResearchPhase.THINKING
        if self.has_pending_work():
            return ResearchPhase.RESEARCHING
        if self.is_responding:
            return ResearchPhase.WRITING
        if self.visible_tool_counts()[1] > 0 or self.subagents:
            # Tools/sub-agents finished but model hasn't started responding
            # yet — it may call more tools, so don't claim WRITING.
            return ResearchPhase.RESEARCHING
        return ResearchPhase.IDLE

    def get_display_args(self) -> dict:
        """Get kwargs for create_streaming_display()."""
        return {
            "thinking_text": self.thinking_text,
            "summarization_text": self.summarization_text,
            "is_summarizing": self.is_summarizing,
            "response_text": self.response_text,
            "latest_text": self.latest_text,
            "narrated_response_end": self.narrated_response_end,
            "narration_segments": self.narration_segments,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "is_thinking": self.is_thinking,
            "is_responding": self.is_responding,
            "is_processing": self.is_processing,
            "subagents": self.subagents,
            "todo_items": self.todo_items,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "selected_tools": self.selected_tools,
        }


def _parse_todo_items(content: str) -> list[dict] | None:
    """Parse todo items from write_todos output.

    Attempts to extract a list of dicts with 'status' and 'content' keys
    from the tool result string. Returns None if parsing fails.

    Handles formats like:
      - Raw JSON/Python list: [{"content": "...", "status": "..."}]
      - Prefixed: "Updated todo list to [{'content': '...', ...}]"
    """
    content = content.strip()

    def _try_parse(text: str) -> list[dict] | None:
        """Try JSON then Python literal parsing."""
        text = text.strip()
        try:
            data = json.loads(text)
            if isinstance(data, list) and data and isinstance(data[0], dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass
        try:
            data = ast.literal_eval(text)
            if isinstance(data, list) and data and isinstance(data[0], dict):
                return data
        except (ValueError, SyntaxError):
            pass
        return None

    # Try the full content directly
    result = _try_parse(content)
    if result:
        return result

    # Extract embedded [...] from content (e.g. "Updated todo list to [{...}]")
    bracket_start = content.find("[")
    if bracket_start != -1:
        bracket_end = content.rfind("]")
        if bracket_end > bracket_start:
            embedded = content[bracket_start : bracket_end + 1]
            result = _try_parse(embedded)
            if result:
                return result

    # Try line-by-line scan
    for line in content.split("\n"):
        line = line.strip()
        if "[" in line:
            start = line.find("[")
            end = line.rfind("]")
            if end > start:
                result = _try_parse(line[start : end + 1])
                if result:
                    return result

    return None


def _build_todo_stats(items: list[dict]) -> str:
    """Build stats string like '2 active | 1 pending | 3 done'."""
    counts: dict[str, int] = {}
    for item in items:
        status = str(item.get("status", "todo")).lower()
        # Normalize status names
        if status in ("done", "completed", "complete"):
            status = "done"
        elif status in ("active", "in_progress", "in-progress", "working"):
            status = "active"
        else:
            status = "pending"
        counts[status] = counts.get(status, 0) + 1

    parts = []
    for key in ("active", "pending", "done"):
        if counts.get(key, 0) > 0:
            parts.append(f"{counts[key]} {key}")
    return " | ".join(parts) if parts else f"{len(items)} items"
