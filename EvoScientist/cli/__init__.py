"""EvoScientist CLI package.

Most re-exports are served lazily through ``__getattr__`` so that a bare
``import EvoScientist.cli`` only costs what ``main()`` actually needs.  That
keeps ``evosci --help`` fast — the heavy chat-model/TUI/langgraph imports
only pay their cost when someone actually touches those names.
"""

from __future__ import annotations

from .. import deploy as _deploy_pkg  # noqa: F401 — registers `deploy` @app.command
from . import commands  # noqa: F401 — registers @app.command decorators
from ._app import app

__all__ = [
    "DEFAULT_UI_BACKEND",
    "SUPPORTED_UI_BACKENDS",
    "WELCOME_SLOGANS",
    "StreamState",
    "SubAgentState",
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


def _write_redirected_output_as_utf8() -> None:
    """Switch redirected stdout and stderr to UTF-8 (#660).

    On Windows, Python writes to a pipe, a file or the NUL device in the ANSI
    code page, so the first character outside it (a check mark, a
    box-drawing character, a letter in a path) stops the command with
    ``UnicodeEncodeError``. On Windows both streams are switched even when
    ``isatty()`` is true, since it is true for NUL as well and a console
    gets UTF-8 already (unless ``PYTHONLEGACYWINDOWSSTDIO`` is set);
    elsewhere a terminal keeps its locale encoding. A ``PYTHONIOENCODING``
    the user set is kept; Python ignores an empty one, so empty counts as
    unset. Each stream keeps its error handler, which
    ``reconfigure(encoding=...)`` would otherwise reset to ``strict``.
    """
    import os
    import sys

    if os.environ.get("PYTHONIOENCODING"):
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            if sys.platform == "win32" or not stream.isatty():
                reconfigure(encoding="utf-8", errors=stream.errors)
        except (OSError, ValueError):
            pass


def main():
    """CLI entry point."""
    import os
    import warnings

    # Keep MCP stdio subprocess spawning async on Windows (see #283). Must run
    # before any event loop is created, hence at the very top of the entrypoint.
    from .._winloop import ensure_proactor_event_loop_policy

    ensure_proactor_event_loop_policy()
    # Before anything is printed.
    _write_redirected_output_as_utf8()

    warnings.filterwarnings("ignore", message=".*not known to support tools.*")
    warnings.filterwarnings(
        "ignore", message=".*type is unknown and inference may fail.*"
    )
    # v3 streaming is a deliberate choice (#268), so its beta notice is noise.
    # Matched by message, not category, to keep other beta warnings visible.
    warnings.filterwarnings(
        "ignore", message=".*v3 streaming protocol on Pregel is experimental.*"
    )
    from ..config import load_config
    from .commands import _configure_logging

    # Priority: env var > config file > default (WARNING)
    config = load_config()
    _log_level = os.environ.get("EVOSCIENTIST_LOG_LEVEL", "") or config.log_level
    # Before any child process starts, so they all find the private Node and,
    # on Windows, the private Git.
    from ..setup import git as setup_git
    from ..setup import node as setup_node

    setup_node.activate_runtime()
    setup_git.activate_runtime()
    _configure_logging()
    app()
