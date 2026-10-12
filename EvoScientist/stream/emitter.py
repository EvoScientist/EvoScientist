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
    def tool_call(name: str, args: dict[str, Any], tool_id: str) -> StreamEvent:
        """Tool call event."""
        return StreamEvent(
            "tool_call",
            {"type": "tool_call", "name": name, "args": args, "id": tool_id},
        )

    @staticmethod
    def tool_result(
        name: str,
        content: str,
        success: bool,
        tool_call_id: str,
    ) -> StreamEvent:
        """Tool result event."""
        return StreamEvent(
            "tool_result",
            {
                "type": "tool_result",
                "name": name,
                "content": content,
                "success": success,
                "id": tool_call_id,
            },
        )

    @staticmethod
    def subagent_start(
        name: str, description: str, instance_id: str, tool_call_id: str
    ) -> StreamEvent:
        """Sub-agent delegation started."""
        return StreamEvent(
            "subagent_start",
            {
                "type": "subagent_start",
                "name": name,
                "description": description,
                "instance_id": instance_id,
                "tool_call_id": tool_call_id,
            },
        )

    @staticmethod
    def subagent_tool_call(
        subagent: str,
        name: str,
        args: dict[str, Any],
        tool_id: str,
        instance_id: str,
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
                "instance_id": instance_id,
            },
        )

    @staticmethod
    def subagent_tool_result(
        subagent: str,
        name: str,
        content: str,
        success: bool,
        tool_call_id: str,
        instance_id: str,
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
                "id": tool_call_id,
                "instance_id": instance_id,
            },
        )

    @staticmethod
    def subagent_text(subagent: str, content: str, instance_id: str) -> StreamEvent:
        """Text content from a sub-agent."""
        return StreamEvent(
            "subagent_text",
            {
                "type": "subagent_text",
                "subagent": subagent,
                "content": content,
                "instance_id": instance_id,
            },
        )

    @staticmethod
    def subagent_end(
        name: str, instance_id: str, *, paused: bool = False
    ) -> StreamEvent:
        """Sub-agent delegation completed, or stopped at a pause (``paused``)."""
        data = {"type": "subagent_end", "name": name, "instance_id": instance_id}
        if paused:
            data["paused"] = True
        return StreamEvent("subagent_end", data)

    @staticmethod
    def panel_dispatch_start(
        *,
        eval_id: str,
        dispatch_id: str,
        subagent_type: str,
        label: str,
        description: str,
    ) -> StreamEvent:
        """A ``task()`` dispatch inside a ``code_interpreter`` eval started.

        Sourced from ``langchain_quickjs``'s custom-stream subagent events.
        Distinct family from ``subagent_start`` because the semantics differ:
        short-lived expert dispatch inside a QuickJS eval, grouped by
        ``eval_id`` (parent ``code_interpreter`` tool_call_id), not a full
        nested subagent turn.
        """
        return StreamEvent(
            "panel_dispatch_start",
            {
                "type": "panel_dispatch_start",
                "eval_id": eval_id,
                "id": dispatch_id,
                "subagent_type": subagent_type,
                "label": label,
                "description": description,
            },
        )

    @staticmethod
    def panel_dispatch_complete(
        *,
        eval_id: str,
        dispatch_id: str,
        duration_ms: int,
    ) -> StreamEvent:
        """A ``task()`` dispatch inside a ``code_interpreter`` eval finished OK."""
        return StreamEvent(
            "panel_dispatch_complete",
            {
                "type": "panel_dispatch_complete",
                "eval_id": eval_id,
                "id": dispatch_id,
                "duration_ms": duration_ms,
            },
        )

    @staticmethod
    def panel_dispatch_error(
        *,
        eval_id: str,
        dispatch_id: str,
        duration_ms: int,
        error: str,
    ) -> StreamEvent:
        """A ``task()`` dispatch inside a ``code_interpreter`` eval raised."""
        return StreamEvent(
            "panel_dispatch_error",
            {
                "type": "panel_dispatch_error",
                "eval_id": eval_id,
                "id": dispatch_id,
                "duration_ms": duration_ms,
                "error": error,
            },
        )

    @staticmethod
    def rubric_evaluation_start(*, iteration: int) -> StreamEvent:
        """The acceptance grader started a pass over the finished work."""
        return StreamEvent(
            "rubric_evaluation_start",
            {"type": "rubric_evaluation_start", "iteration": iteration},
        )

    @staticmethod
    def rubric_evaluation(
        *,
        iteration: int,
        result: str,
        explanation: str,
        criteria: list[dict],
        unverified: bool,
    ) -> StreamEvent:
        """The verdict of one grading pass."""
        return StreamEvent(
            "rubric_evaluation",
            {
                "type": "rubric_evaluation",
                "iteration": iteration,
                "result": result,
                "explanation": explanation,
                "criteria": criteria,
                "unverified": unverified,
            },
        )

    @staticmethod
    def goal_state(
        *,
        change: str,
        status: str = "",
        note: str = "",
        objective: str = "",
        criteria: str = "",
        message: str = "",
    ) -> StreamEvent:
        """Something happened to the thread's goal (drafted, accepted, paused, ...)."""
        return StreamEvent(
            "goal_state",
            {
                "type": "goal_state",
                "change": change,
                "status": status,
                "note": note,
                "objective": objective,
                "criteria": criteria,
                "message": message,
            },
        )

    @staticmethod
    def rubric_evaluation_deferred(*, tasks: int, processes: int) -> StreamEvent:
        """Grading waits for this thread's background sub-agents or processes."""
        return StreamEvent(
            "rubric_evaluation_deferred",
            {
                "type": "rubric_evaluation_deferred",
                "tasks": tasks,
                "processes": processes,
            },
        )

    @staticmethod
    def model_retry(
        *,
        attempt: int,
        wait_seconds: float,
        reason: str,
        budget_left_seconds: float,
        cause: str,
        instance_id: str,
        subagent: str,
        discarded: dict[str, Any],
    ) -> StreamEvent:
        return StreamEvent(
            "model_retry",
            {
                "type": "model_retry",
                "attempt": attempt,
                "wait_seconds": wait_seconds,
                "reason": reason,
                "budget_left_seconds": budget_left_seconds,
                "cause": cause,
                "instance_id": instance_id,
                "subagent": subagent,
                "discarded": discarded,
            },
        )

    @staticmethod
    def model_output_discarded(
        *, instance_id: str, discarded: dict[str, Any]
    ) -> StreamEvent:
        return StreamEvent(
            "model_output_discarded",
            {
                "type": "model_output_discarded",
                "instance_id": instance_id,
                "discarded": discarded,
            },
        )

    @staticmethod
    def done(
        response: str = "", *, last_pass_response: str | None = None
    ) -> StreamEvent:
        """Done event; ``last_pass_response`` only for a run that was graded."""
        data = {"type": "done", "content": response, "response": response}
        if last_pass_response is not None:
            data["last_pass_response"] = last_pass_response
        return StreamEvent("done", data)

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
    def tool_selection(tools: list[str]) -> StreamEvent:
        """Tool selection event from LLMToolSelectorMiddleware."""
        return StreamEvent("tool_selection", {"type": "tool_selection", "tools": tools})

    @staticmethod
    def summarization(content: str) -> StreamEvent:
        """Context summarization event."""
        return StreamEvent(
            "summarization", {"type": "summarization", "content": content}
        )

    @staticmethod
    def summarization_start() -> StreamEvent:
        """Context summarization started."""
        return StreamEvent(
            "summarization_start",
            {"type": "summarization_start"},
        )

    @staticmethod
    def error(message: str) -> StreamEvent:
        """Error event."""
        return StreamEvent("error", {"type": "error", "message": message})
