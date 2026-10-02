"""EvoScientist deploy mode.

Standalone LangGraph server launcher. ``EvoSci deploy`` starts a
``langgraph dev`` subprocess hosting the fully-equipped main agent
(MCP + async sub-agents enabled), exposed at
``http://localhost:{port}`` for external LangChain-compatible UIs
and SDK clients.

This module is intentionally separate from ``EvoScientist.cli``:
deploy mode does NOT load an in-process CLI agent, session DB, channel
runtime, or TUI — those are all TUI / serve concerns. Deploy only
manages the lifecycle of the langgraph dev subprocess (and ccproxy if
OAuth is configured).
"""

from __future__ import annotations

# Importing the server submodule registers the ``@app.command()`` decorator
# on the shared Typer ``app`` from ``EvoScientist.cli._app``.
from . import server

__all__ = ["server"]
