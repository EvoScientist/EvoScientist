"""Runtime context middleware for EvoScientist."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)

RUNTIME_CONTEXT_TEMPLATE = """<runtime_context>
Current date: {date}
Local timezone: {timezone}

Use this context to resolve relative time references like today, yesterday, and
next week.
</runtime_context>"""


def _format_timezone(now: datetime) -> str:
    """Return a compact local timezone label for prompt injection."""
    offset = now.utcoffset()
    if offset is None:
        return "local"

    total_seconds = int(offset.total_seconds())
    sign = "+" if total_seconds >= 0 else "-"
    total_seconds = abs(total_seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes = remainder // 60
    offset_text = f"UTC{sign}{hours:02d}:{minutes:02d}"

    name = now.tzname()
    if name and name != offset_text:
        return f"{name} ({offset_text})"
    return offset_text


class RuntimeContextMiddleware(AgentMiddleware):
    """Inject per-turn runtime context into model calls."""

    def __init__(self, *, now_fn: Callable[[], datetime] | None = None) -> None:
        self._now_fn = now_fn or (lambda: datetime.now().astimezone())

    def _runtime_context(self) -> str:
        now = self._now_fn()
        return RUNTIME_CONTEXT_TEMPLATE.format(
            date=now.strftime("%Y-%m-%d"),
            timezone=_format_timezone(now),
        )

    def modify_request(self, request: ModelRequest) -> ModelRequest:
        """Append runtime context to the system prompt."""
        from deepagents.middleware._utils import append_to_system_message

        new_system = append_to_system_message(
            request.system_message,
            self._runtime_context(),
        )
        return request.override(system_message=new_system)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject runtime context before the sync model handler."""
        return handler(self.modify_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Inject runtime context before the async model handler."""
        return await handler(self.modify_request(request))


def create_runtime_context_middleware(
    *, now_fn: Callable[[], datetime] | None = None
) -> RuntimeContextMiddleware:
    """Build runtime-context middleware."""
    return RuntimeContextMiddleware(now_fn=now_fn)
