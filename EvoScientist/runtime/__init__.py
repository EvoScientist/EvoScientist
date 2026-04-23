"""Shared runtime primitives for interactive surfaces (TUI/WebUI/CLI)."""

from .command_runtime import build_command_catalog, execute_command_line
from .interaction_bridge import (
    RuntimeBridgeCallbacks,
    build_runtime_bridge,
    make_threadsafe_sender,
    mark_run_finished,
    mark_run_started,
)
from .thread_registry import ThreadRuntimeRegistry, get_thread_runtime_registry

__all__ = [
    "RuntimeBridgeCallbacks",
    "ThreadRuntimeRegistry",
    "build_command_catalog",
    "build_runtime_bridge",
    "execute_command_line",
    "get_thread_runtime_registry",
    "make_threadsafe_sender",
    "mark_run_finished",
    "mark_run_started",
]
