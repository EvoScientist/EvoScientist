"""
StreamEventEmitter - standardized event format.

All events contain a type and associated data dict.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class StreamEvent:
    """Unified stream event."""

    type: str
    data: dict[str, Any]


class StreamEventEmitter:
    """Stream event emitter - creates standardized event dicts."""

    @staticmethod
    def thinking(content: str, thinking_id: int = 0) -> StreamEvent:
        """Thinking content event."""
        return StreamEvent(
            "thinking", {"type": "thinking", "content": content, "id": thinking_id}
        )

    @staticmethod
    def text(content: str) -> StreamEvent:
        """Text content event."""
        return StreamEvent("text", {"type": "text", "content": content})

    @staticmethod
    def tool_call(name: str, args: dict[str, Any], tool_id: str = "") -> StreamEvent:
        """Tool call event."""
        return StreamEvent(
            "tool_call",
            {"type": "tool_call", "name": name, "args": args, "id": tool_id},
        )

    @staticmethod
    def tool_result(name: str, content: str, success: bool = True) -> StreamEvent:
        """Tool result event."""
        return StreamEvent(
            "tool_result",
            {
                "type": "tool_result",
                "name": name,
                "content": content,
                "success": success,
            },
        )

    @staticmethod
    def subagent_start(name: str, description: str) -> StreamEvent:
        """Sub-agent delegation started."""
        return StreamEvent(
            "subagent_start",
            {
                "type": "subagent_start",
                "name": name,
                "description": description,
            },
        )

    @staticmethod
    def subagent_tool_call(
        subagent: str, name: str, args: dict[str, Any], tool_id: str = ""
    ) -> StreamEvent:
        """Tool call from inside a sub-agent."""
        return StreamEvent(
            "subagent_tool_call",
            {
                "type": "subagent_tool_call",
                "subagent": subagent,
                "name": name,
                "args": args,
                "id": tool_id,
            },
        )

    @staticmethod
    def subagent_tool_result(
        subagent: str, name: str, content: str, success: bool = True
    ) -> StreamEvent:
        """Tool result from inside a sub-agent."""
        return StreamEvent(
            "subagent_tool_result",
            {
                "type": "subagent_tool_result",
                "subagent": subagent,
                "name": name,
                "content": content,
                "success": success,
            },
        )

    @staticmethod
    def subagent_end(name: str) -> StreamEvent:
        """Sub-agent delegation completed."""
        return StreamEvent("subagent_end", {"type": "subagent_end", "name": name})

    @staticmethod
    def done(response: str = "") -> StreamEvent:
        """Done event."""
        return StreamEvent(
            "done", {"type": "done", "content": response, "response": response}
        )

    @staticmethod
    def usage_stats(input_tokens: int, output_tokens: int) -> StreamEvent:
        """Token usage statistics event."""
        return StreamEvent(
            "usage_stats",
            {
                "type": "usage_stats",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        )

    @staticmethod
    def interrupt(
        interrupt_id: str,
        action_requests: list,
        review_configs: list | None = None,
    ) -> StreamEvent:
        """Human-in-the-loop interrupt event."""
        return StreamEvent(
            "interrupt",
            {
                "type": "interrupt",
                "interrupt_id": interrupt_id,
                "action_requests": action_requests,
                "review_configs": review_configs or [],
            },
        )

    @staticmethod
    def ask_user_interrupt(
        interrupt_id: str,
        questions: list,
        tool_call_id: str = "",
    ) -> StreamEvent:
        """Agent-initiated ask_user interrupt event."""
        return StreamEvent(
            "ask_user",
            {
                "type": "ask_user",
                "interrupt_id": interrupt_id,
                "questions": questions,
                "tool_call_id": tool_call_id,
            },
        )

    @staticmethod
    def summarization(content: str) -> StreamEvent:
        """Context summarization event."""
        return StreamEvent(
            "summarization", {"type": "summarization", "content": content}
        )

    @staticmethod
    def error(message: str) -> StreamEvent:
        """Error event."""
        return StreamEvent("error", {"type": "error", "message": message})
