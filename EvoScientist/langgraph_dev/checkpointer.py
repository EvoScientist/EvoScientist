"""The checkpointer langgraph-api opens at server start (``checkpointer.path``).

langgraph-api enters it before it collects the graphs and starts its cron
scheduler and run workers, so the server records here what it starts with,
before any graph can be built.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from ..sessions import PruningCheckpointer, create_checkpointer_for_langgraph_api
from .server_home import check_store_loaded
from .settings_api import capture_restart_baseline


@asynccontextmanager
async def server_checkpointer() -> AsyncIterator[PruningCheckpointer]:
    """The SQLite checkpointer, opened after the settings API's restart baseline
    and the check that the runtime loaded its store."""
    # Off the event loop: blockbuster refuses its file reads there. The
    # runtime has loaded its store by now (start_pool comes first).
    await asyncio.to_thread(capture_restart_baseline)
    await asyncio.to_thread(check_store_loaded)
    async with create_checkpointer_for_langgraph_api() as saver:
        yield saver
