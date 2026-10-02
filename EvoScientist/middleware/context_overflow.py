"""Middleware that maps provider-specific context limit errors to a standard ContextOverflowError.

When the LLM returns a 400 error indicating the context length has been
exceeded, this middleware detects it and raises a standard
``langchain_core.exceptions.ContextOverflowError``.  This allows the
underlying framework (e.g. deepagents' SummarizationMiddleware) to catch
the error, trigger summarization, and retry the request automatically.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)

logger = logging.getLogger(__name__)


class ContextOverflowMapperMiddleware(AgentMiddleware):
    """Map provider-specific context limit errors to ContextOverflowError."""

    name = "context_overflow_mapper"

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        try:
            return handler(request)
        except Exception as exc:
            if self._is_context_limit_error(exc):
                from langchain_core.exceptions import ContextOverflowError

                logger.warning(
                    "Context limit exceeded. Mapping to ContextOverflowError..."
                )

                raise ContextOverflowError(str(exc)) from exc
            raise

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        try:
            return await handler(request)
        except Exception as exc:
            if self._is_context_limit_error(exc):
                from langchain_core.exceptions import ContextOverflowError

                logger.warning(
                    "Context limit exceeded. Mapping to ContextOverflowError..."
                )

                raise ContextOverflowError(str(exc)) from exc
            raise

    def _is_context_limit_error(self, exc: Exception) -> bool:
        """Detect if an exception is a context length/limit error.

        It triggers when there's an error 400 raised and one of specified patterns exists in the error message.
        """
        err_msg = str(exc).lower()

        patterns = [
            "context_length_exceeded",
            "context length exceeded",
            "too many tokens",
            "maximum context length",
            "output too large",
            "context_window_exceeded",
            "string_too_long",
            "max_tokens_exceeded",
        ]
        is_400 = "400" in err_msg or "bad request" in err_msg

        return any(p in err_msg for p in patterns) and is_400
