"""Stream state tracking for CLI display.

Contains SubAgentState, StreamState, and todo-item parsing helpers.
No Rich dependencies — stdlib only.
"""

import ast
import json


class SubAgentState:
    """Tracks a single sub-agent's activity."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.tool_calls: list[dict] = []
        self.tool_results: list[dict] = []
        self._result_map: dict[str, dict] = {}  # tool_call_id -> result
        self.is_active = True

    def add_tool_call(self, name: str, args: dict, tool_id: str = ""):
        # Skip empty-name calls without an id (incomplete streaming chunks)
        if not name and not tool_id:
            return
        tc_data = {"id": tool_id, "name": name, "args": args}
        if tool_id:
            for i, tc in enumerate(self.tool_calls):
                if tc.get("id") == tool_id:
                    # Merge: keep the non-empty name/args
                    if name:
                        self.tool_calls[i]["name"] = name
                    if args:
                        self.tool_calls[i]["args"] = args
                    return
        # Skip if name is empty and we can't deduplicate by id
        if not name:
            return
        self.tool_calls.append(tc_data)

    def add_tool_result(self, name: str, content: str, success: bool = True):
        result = {"name": name, "content": content, "success": success}
        self.tool_results.append(result)
        # Try to match result to the first unmatched tool call with same name
        for tc in self.tool_calls:
            tc_id = tc.get("id", "")
            tc_name = tc.get("name", "")
            if tc_id and tc_id not in self._result_map and tc_name == name:
                self._result_map[tc_id] = result
                return
        # Fallback: match first unmatched tool call
        for tc in self.tool_calls:
            tc_id = tc.get("id", "")
            if tc_id and tc_id not in self._result_map:
                self._result_map[tc_id] = result
                return

    def get_result_for(self, tc: dict) -> dict | None:
        """Get matched result for a tool call."""
        tc_id = tc.get("id", "")
        if tc_id:
            return self._result_map.get(tc_id)
        # Fallback: index-based matching
        try:
            idx = self.tool_calls.index(tc)
            if idx < len(self.tool_results):
                return self.tool_results[idx]
        except ValueError:
            pass
        return None


class StreamState:
    """Accumulates stream state for display updates."""

    def __init__(self):
        self.thinking_text = ""
        self.response_text = ""
        self.tool_calls = []
        self.tool_results = []
        self.is_thinking = False
        self.is_responding = False
        self.is_processing = False
        # Sub-agent tracking
        self.subagents: list[SubAgentState] = []
        self._subagent_map: dict[str, SubAgentState] = {}  # name -> state
        # Todo list tracking
        self.todo_items: list[dict] = []
        # Latest text segment (reset on each tool_call)
        self.latest_text = ""

    def _get_or_create_subagent(self, name: str, description: str = "") -> SubAgentState:
        if name not in self._subagent_map:
            # Case 1: real name arrives, "sub-agent" entry exists -> rename it
            if name != "sub-agent" and "sub-agent" in self._subagent_map:
                old_sa = self._subagent_map.pop("sub-agent")
                old_sa.name = name
                if description:
                    old_sa.description = description
                self._subagent_map[name] = old_sa
                return old_sa
            # Case 2: "sub-agent" arrives but a pre-registered real-name entry
            #         exists with no tool calls -> merge into it
            if name == "sub-agent":
                active_named = [
                    sa for sa in self.subagents
                    if sa.is_active and sa.name != "sub-agent"
                ]
                if len(active_named) == 1 and not active_named[0].tool_calls:
                    self._subagent_map[name] = active_named[0]
                    return active_named[0]
            sa = SubAgentState(name, description)
            self.subagents.append(sa)
            self._subagent_map[name] = sa
        else:
            existing = self._subagent_map[name]
            if description and not existing.description:
                existing.description = description
            # If this entry was created as "sub-agent" placeholder and the
            # actual name is different, update.
            if name != "sub-agent" and existing.name == "sub-agent":
                existing.name = name
        return self._subagent_map[name]

    def _resolve_subagent_name(self, name: str) -> str:
        """Resolve "sub-agent" to the single active named sub-agent when possible."""
        if name != "sub-agent":
            return name
        active_named = [
            sa.name for sa in self.subagents
            if sa.is_active and sa.name != "sub-agent"
        ]
        if len(active_named) == 1:
            return active_named[0]
        return name

    def handle_event(self, event: dict) -> str:
        """Process a single stream event, update internal state, return event type."""
        event_type: str = event.get("type", "")

        if event_type == "thinking":
            self.is_thinking = True
            self.is_responding = False
            self.is_processing = False
            self.thinking_text += event.get("content", "")

        elif event_type == "text":
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
            self.latest_text = ""  # Reset -- next text segment is a new message

            tool_id = event.get("id", "")
            tool_name = event.get("name", "unknown")
            tool_args = event.get("args", {})
            tc_data = {
                "id": tool_id,
                "name": tool_name,
                "args": tool_args,
            }

            if tool_id:
                updated = False
                for i, tc in enumerate(self.tool_calls):
                    if tc.get("id") == tool_id:
                        self.tool_calls[i] = tc_data
                        updated = True
                        break
                if not updated:
                    self.tool_calls.append(tc_data)
            else:
                self.tool_calls.append(tc_data)

            # Capture todo items from write_todos args (most reliable source)
            if tool_name == "write_todos":
                todos = tool_args.get("todos", [])
                if isinstance(todos, list) and todos:
                    self.todo_items = todos

        elif event_type == "tool_result":
            self.is_processing = True
            result_name = event.get("name", "unknown")
            result_content = event.get("content", "")
            self.tool_results.append({
                "name": result_name,
                "content": result_content,
            })
            # Update todo list from write_todos / read_todos results (fallback)
            if result_name in ("write_todos", "read_todos"):
                parsed = _parse_todo_items(result_content)
                if parsed:
                    self.todo_items = parsed

        elif event_type == "subagent_start":
            name = event.get("name", "sub-agent")
            desc = event.get("description", "")
            sa = self._get_or_create_subagent(name, desc)
            sa.is_active = True

        elif event_type == "subagent_tool_call":
            sa_name = self._resolve_subagent_name(event.get("subagent", "sub-agent"))
            sa = self._get_or_create_subagent(sa_name)
            sa.add_tool_call(
                event.get("name", "unknown"),
                event.get("args", {}),
                event.get("id", ""),
            )

        elif event_type == "subagent_tool_result":
            sa_name = self._resolve_subagent_name(event.get("subagent", "sub-agent"))
            sa = self._get_or_create_subagent(sa_name)
            sa.add_tool_result(
                event.get("name", "unknown"),
                event.get("content", ""),
                event.get("success", True),
            )

        elif event_type == "subagent_end":
            name = self._resolve_subagent_name(event.get("name", "sub-agent"))
            if name in self._subagent_map:
                self._subagent_map[name].is_active = False
            elif name == "sub-agent":
                # Couldn't resolve -- deactivate the oldest active sub-agent
                for sa in self.subagents:
                    if sa.is_active:
                        sa.is_active = False
                        break

        elif event_type == "done":
            self.is_processing = False
            if not self.response_text:
                self.response_text = event.get("response", "")

        elif event_type == "error":
            self.is_processing = False
            self.is_thinking = False
            self.is_responding = False
            error_msg = event.get("message", "Unknown error")
            self.response_text += f"\n\n[Error] {error_msg}"

        return event_type

    def get_display_args(self) -> dict:
        """Get kwargs for create_streaming_display()."""
        return {
            "thinking_text": self.thinking_text,
            "response_text": self.response_text,
            "latest_text": self.latest_text,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "is_thinking": self.is_thinking,
            "is_responding": self.is_responding,
            "is_processing": self.is_processing,
            "subagents": self.subagents,
            "todo_items": self.todo_items,
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
            embedded = content[bracket_start:bracket_end + 1]
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
                result = _try_parse(line[start:end + 1])
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
