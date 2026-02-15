"""EvoScientist CLI package."""

# Backward-compat re-exports (tests import these from EvoScientist.cli)
from ..stream.state import (  # noqa: F401
    SubAgentState,
    StreamState,
    _parse_todo_items,
    _build_todo_stats,
)
from .channel import _channels_is_running, _channels_stop  # noqa: F401
from .agent import _deduplicate_run_name  # noqa: F401

from ._app import app  # noqa: F401
from . import commands  # noqa: F401 — registers @app.command decorators


def main():
    """CLI entry point."""
    import warnings

    warnings.filterwarnings("ignore", message=".*not known to support tools.*")
    warnings.filterwarnings("ignore", message=".*type is unknown and inference may fail.*")
    from .commands import _configure_logging

    _configure_logging()
    app()
