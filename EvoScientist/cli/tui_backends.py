"""TUI backend abstractions for streaming output."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from ..stream.display import _run_streaming


class StreamingTUIBackend(Protocol):
    """Protocol for TUI backends that can render agent streaming output."""

    name: str

    def run_streaming(
        self,
        *,
        agent: Any,
        message: str,
        thread_id: str,
        show_thinking: bool,
        interactive: bool,
        on_thinking: Callable[[str], None] | None = None,
        on_todo: Callable[[list[dict]], None] | None = None,
        on_file_write: Callable[[str], None] | None = None,
        on_stream_event: Callable[[str, Any], Any] | None = None,
        status_footer_builder: Callable[[], Any] | None = None,
        metadata: dict | None = None,
        hitl_prompt_fn: Callable[[list], list[dict] | None] | None = None,
        ask_user_prompt_fn: Callable[[dict], dict] | None = None,
        cancel_scope: str | None = None,
    ) -> str:
        """Run streaming and return final response text."""


@dataclass(slots=True)
class RichStreamingBackend:
    """Default Rich backend wrapper around the existing streaming renderer."""

    name: str = "cli"

    def run_streaming(
        self,
        *,
        agent: Any,
        message: str,
        thread_id: str,
        show_thinking: bool,
        interactive: bool,
        on_thinking: Callable[[str], None] | None = None,
        on_todo: Callable[[list[dict]], None] | None = None,
        on_file_write: Callable[[str], None] | None = None,
        on_stream_event: Callable[[str, Any], Any] | None = None,
        status_footer_builder: Callable[[], Any] | None = None,
        metadata: dict | None = None,
        hitl_prompt_fn: Callable[[list], list[dict] | None] | None = None,
        ask_user_prompt_fn: Callable[[dict], dict] | None = None,
        cancel_scope: str | None = None,
    ) -> str:
        return _run_streaming(
            agent=agent,
            message=message,
            thread_id=thread_id,
            show_thinking=show_thinking,
            interactive=interactive,
            on_thinking=on_thinking,
            on_todo=on_todo,
            on_file_write=on_file_write,
            on_stream_event=on_stream_event,
            status_footer_builder=status_footer_builder,
            metadata=metadata,
            hitl_prompt_fn=hitl_prompt_fn,
            ask_user_prompt_fn=ask_user_prompt_fn,
            cancel_scope=cancel_scope,
        )
