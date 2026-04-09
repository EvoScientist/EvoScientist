"""EvoScientist CLI package."""

# Backward-compat re-exports (tests import these from EvoScientist.cli)
from ..stream.state import (  # noqa: F401
    StreamState,
    SubAgentState,
    _build_todo_stats,
    _parse_todo_items,
)
from . import commands  # noqa: F401 — registers @app.command decorators
from ._app import app
from ._constants import WELCOME_SLOGANS  # noqa: F401
from .agent import _deduplicate_run_name  # noqa: F401
from .channel import _channels_is_running, _channels_stop  # noqa: F401

# UI runtime re-exports (merged from former tui/ package)
from .tui_runtime import (  # noqa: F401
    DEFAULT_UI_BACKEND,
    SUPPORTED_UI_BACKENDS,
    get_backend,
    normalize_ui_backend,
    resolve_ui_backend,
    run_streaming,
)


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
