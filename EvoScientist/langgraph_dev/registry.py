"""The projects this server serves: the ``workspaces`` table in ``sessions.db``.

A local client registers a project (``POST /api/workspaces``); once its setup
is done it is recorded as registered, and served while its folder exists.
Unregistering keeps the row, and everything the project owns, so the server
still restores the project's threads.
"""

from __future__ import annotations

import sqlite3
import threading

from ..paths import Workspace
from ..sessions import get_db_path
from .assistants import workspace_assistants

_TABLE = """
    CREATE TABLE IF NOT EXISTS workspaces (
        workspace_dir TEXT PRIMARY KEY,
        registered INTEGER NOT NULL
    )
"""


class Registry:
    """The registry as this server process sees it, by ``Workspace.key``.

    Loaded from ``sessions.db`` on first use (as the server starts) and
    written through; only the server writes it. Blocking: call it off the
    event loop.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        # Every project the registry knows -> whether it is registered.
        self._projects: dict[str, bool] | None = None
        # assistant ID -> (graph, project key), for every known project.
        self._assistants: dict[str, tuple[str, str]] = {}

    @staticmethod
    def _connect() -> sqlite3.Connection:
        return sqlite3.connect(get_db_path(), timeout=30.0)

    def _map_assistants(self, key: str) -> None:
        for assistant_id, graph_id in workspace_assistants(key).items():
            self._assistants[assistant_id] = (graph_id, key)

    def _loaded(self) -> dict[str, bool]:
        if self._projects is None:
            conn = self._connect()
            try:
                with conn:
                    conn.execute(_TABLE)
                rows = conn.execute(
                    "SELECT workspace_dir, registered FROM workspaces"
                ).fetchall()
            finally:
                conn.close()
            self._projects = {key: bool(registered) for key, registered in rows}
            self._assistants = {}
            for key in self._projects:
                self._map_assistants(key)
        return self._projects

    def known(self) -> set[str]:
        """Every project the registry knows, registered or not."""
        with self._lock:
            return set(self._loaded())

    def registered(self) -> list[str]:
        """The registered projects."""
        with self._lock:
            return [key for key, registered in self._loaded().items() if registered]

    def is_registered(self, key: str) -> bool:
        with self._lock:
            return self._loaded().get(key, False)

    def set_registered(self, key: str, registered: bool) -> None:
        """Record whether *key* is registered, creating its row."""
        # Writes are serialised on their own lock and written outside the
        # lookup lock, so the factories' lookups never wait on SQLite.
        with self._write_lock:
            with self._lock:
                known = self._loaded().get(key)
            if known != registered:
                conn = self._connect()
                try:
                    with conn:
                        conn.execute(
                            "INSERT INTO workspaces (workspace_dir, registered) "
                            "VALUES (?, ?) ON CONFLICT(workspace_dir) "
                            "DO UPDATE SET registered = excluded.registered",
                            (key, int(registered)),
                        )
                finally:
                    conn.close()
            with self._lock:
                if known is None:
                    self._map_assistants(key)
                self._loaded()[key] = registered

    def project_of_assistant(
        self, graph_id: str, assistant_id: object
    ) -> Workspace | None:
        """The project *assistant_id* is *graph_id*'s workspace assistant for.

        Any project the registry knows, registered or not, so a run on an
        unregistered project's assistant is refused for that project.
        """
        if not isinstance(assistant_id, str):
            return None
        with self._lock:
            self._loaded()
            found = self._assistants.get(assistant_id)
        if found is None or found[0] != graph_id:
            return None
        return Workspace(found[1])


REGISTRY = Registry()
