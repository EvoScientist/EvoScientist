"""Middleware package for EvoScientist.

Re-exports middleware classes and factory functions so that existing
``from EvoScientist.middleware import X`` imports continue to work.
"""

from .ask_user import (
    AskUserMiddleware,
    AskUserRequest,
    AskUserWidgetResult,
    Choice,
    Question,
)
from .code_interpreter import create_code_interpreter_middleware
from .configurable_model import ConfigurableModelMiddleware
from .context_editing import (
    compute_context_editing_trigger,
    create_context_editing_middleware,
)
from .context_overflow import ContextOverflowMapperMiddleware
from .memory import (
    EvoMemoryMiddleware,
    create_memory_middleware,
)
from .model_fallback import ModelFallbackMiddleware, load_fallback_chain
from .runtime_context import RuntimeContextMiddleware, create_runtime_context_middleware
from .tool_error_handler import ToolErrorHandlerMiddleware
from .tool_selector import create_tool_selector_middleware
from .utils import disable_thinking

__all__ = [
    "AskUserMiddleware",
    "AskUserRequest",
    "AskUserWidgetResult",
    "Choice",
    "ConfigurableModelMiddleware",
    "ContextOverflowMapperMiddleware",
    "EvoMemoryMiddleware",
    "ModelFallbackMiddleware",
    "Question",
    "RuntimeContextMiddleware",
    "ToolErrorHandlerMiddleware",
    "compute_context_editing_trigger",
    "create_code_interpreter_middleware",
    "create_context_editing_middleware",
    "create_memory_middleware",
    "create_runtime_context_middleware",
    "create_tool_selector_middleware",
    "disable_thinking",
    "load_fallback_chain",
]
