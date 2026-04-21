"""EvoScientist CLI package.

Most re-exports are served lazily through ``__getattr__`` so that a bare
``import EvoScientist.cli`` only costs what ``main()`` actually needs.  That
keeps ``evosci --help`` fast — the heavy chat-model/TUI/langgraph imports
only pay their cost when someone actually touches those names.
"""

from __future__ import annotations

from . import commands  # noqa: F401 — registers @app.command decorators
from ._app import app

__all__ = [
    "DEFAULT_UI_BACKEND",
    "SUPPORTED_UI_BACKENDS",
    "StreamState",
    "SubAgentState",
    "WELCOME_SLOGANS",
    "_build_todo_stats",
    "_channels_is_running",
    "_channels_stop",
    "_deduplicate_run_name",
    "_parse_todo_items",
    "app",
    "get_backend",
    "main",
    "normalize_ui_backend",
    "resolve_ui_backend",
    "run_streaming",
]

# Map attribute name -> (relative-module, attribute-in-module).
# Paths starting with ".." reach out of this package.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "StreamState": ("..stream.state", "StreamState"),
    "SubAgentState": ("..stream.state", "SubAgentState"),
    "_build_todo_stats": ("..stream.state", "_build_todo_stats"),
    "_parse_todo_items": ("..stream.state", "_parse_todo_items"),
    "WELCOME_SLOGANS": ("._constants", "WELCOME_SLOGANS"),
    "_deduplicate_run_name": (".agent", "_deduplicate_run_name"),
    "_channels_is_running": (".channel", "_channels_is_running"),
    "_channels_stop": (".channel", "_channels_stop"),
    "DEFAULT_UI_BACKEND": (".tui_runtime", "DEFAULT_UI_BACKEND"),
    "SUPPORTED_UI_BACKENDS": (".tui_runtime", "SUPPORTED_UI_BACKENDS"),
    "get_backend": (".tui_runtime", "get_backend"),
    "normalize_ui_backend": (".tui_runtime", "normalize_ui_backend"),
    "resolve_ui_backend": (".tui_runtime", "resolve_ui_backend"),
    "run_streaming": (".tui_runtime", "run_streaming"),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module_path, attr = target
    module = import_module(module_path, package=__name__)
    value = getattr(module, attr)
    globals()[name] = value
    return value


def main():
    """CLI entry point."""
    import os
    import warnings

    warnings.filterwarnings("ignore", message=".*not known to support tools.*")
    warnings.filterwarnings(
        "ignore", message=".*type is unknown and inference may fail.*"
    )
    from ..config import load_config
    from .commands import _configure_logging

    # Priority: env var > config file > default (WARNING)
    config = load_config()
    _log_level = os.environ.get("EVOSCIENTIST_LOG_LEVEL", "") or config.log_level
    _configure_logging()
    app()
