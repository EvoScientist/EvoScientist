"""Session persistence using LangGraph's SQLite checkpoint storage.

Provides thread CRUD operations, prefix-matched resume, and an async
context manager for the shared ``AsyncSqliteSaver`` checkpointer.

Adapted from upstream ``deepagents_cli/sessions.py``.

Per-step pruning:
    LangGraph's checkpointer writes a full state snapshot per super-step,
    causing unbounded growth (multi-GB sessions.db). EvoScientist never
    reads historical checkpoints — resume always reads the latest, HITL
    interrupts attach pending writes to the just-written row. So
    ``get_checkpointer()`` yields a ``PruningCheckpointer`` that prunes
    older rows for the same ``(thread_id, checkpoint_ns)`` after every
    ``aput()``. The first-run migration sweep cleans up legacy bloat.
"""

import asyncio
import atexit
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Monkey-patch aiosqlite for langgraph-checkpoint >= 2.1.0 compatibility
# ---------------------------------------------------------------------------
if not hasattr(aiosqlite.Connection, "is_alive"):

    def _is_alive(self: aiosqlite.Connection) -> bool:
        return self._connection is not None

    aiosqlite.Connection.is_alive = _is_alive  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGENT_NAME = "EvoScientist"


# ---------------------------------------------------------------------------
# Paths & ID generation
# ---------------------------------------------------------------------------


def _to_short_path(path: str) -> str:
    """Try to convert a Windows path to its 8.3 short form.

    On Windows, sqlite3 may fail to open databases at paths containing
    non-ASCII characters (e.g., Chinese usernames).  Short paths are
    ASCII-safe when available, but conversion is best-effort: it fails
    when 8.3 name generation is disabled, on non-NTFS volumes, or for
    nonexistent targets.  Returns the original path on non-Windows or
    on failure.
    """
    import sys

    if sys.platform != "win32":
        return path
    import ctypes

    buf = ctypes.create_unicode_buffer(32767)
    if ctypes.windll.kernel32.GetShortPathNameW(path, buf, len(buf)):
        return buf.value
    return path


def get_db_path() -> Path:
    """Return the sessions database path, creating parents.

    Uses ``paths.DATA_DIR`` (~/.evoscientist/ by default), then applies
    a best-effort Windows 8.3 short-path conversion on the *directory*
    (which exists after ``mkdir``) so sqlite3 can handle non-ASCII paths.
    """
    from .paths import DATA_DIR

    db_dir = DATA_DIR
    db_dir.mkdir(parents=True, exist_ok=True)
    return Path(_to_short_path(str(db_dir))) / "sessions.db"


def generate_thread_id() -> str:
    """Generate an 8-char hex thread ID."""
    return uuid.uuid4().hex[:8]


# ---------------------------------------------------------------------------
# Checkpoint pruning
# ---------------------------------------------------------------------------

# Default kept when the caller cannot resolve config (tests, unit-init paths).
# Production callers use ``EvoScientistConfig.checkpoint_keep_per_thread``.
_DEFAULT_KEEP_PER_NS = 10


class PruningCheckpointer(AsyncSqliteSaver):
    """``AsyncSqliteSaver`` that prunes stale checkpoints after every ``aput()``.

    After a successful ``aput()``, deletes rows in ``checkpoints`` and
    ``writes`` whose ``(thread_id, checkpoint_ns)`` matches the just-written
    row but whose ``checkpoint_id`` is not among the ``keep_per_ns`` most
    recent ids. The just-written row is always kept (it is the head of the
    descending order and ``keep_per_ns >= 1`` is enforced).

    Inherits from ``AsyncSqliteSaver`` (rather than wrapping it) so
    LangGraph's ``compile()`` ``isinstance(x, BaseCheckpointSaver)`` check
    succeeds. All other behavior — ``aget_tuple``, ``alist``,
    ``aput_writes``, ``adelete_thread``, ``setup``, the async context
    manager protocol, the connection lock — is inherited unchanged.

    HITL safety: pregel records pending writes (e.g. ``interrupt``) against
    the checkpoint id returned by the most recent ``aput()``, which is
    exactly the row we keep. Older rows can never receive new writes after
    a newer ``aput()`` lands, so deleting them is provably safe.

    Setting ``keep_per_ns <= 0`` disables pruning (escape hatch for debug).
    """

    def __init__(
        self,
        conn: aiosqlite.Connection,
        *,
        keep_per_ns: int = _DEFAULT_KEEP_PER_NS,
        serde: Any = None,
    ) -> None:
        super().__init__(conn, serde=serde)
        self._keep_per_ns = max(0, int(keep_per_ns))
        # Outer lock guarantees ``super().aput()`` and ``_prune_after_put()``
        # are atomic *as a pair*. Without this, a concurrent ``aput()`` on a
        # different ``(thread_id, checkpoint_ns)`` could land between the
        # two phases and squeeze the earlier caller's just-written row out
        # of the top-N retention window (only matters when ``keep_per_ns``
        # is small or N parallel writers race; harmless otherwise but the
        # invariant "the just-written row is always kept" must hold).
        self._aput_lock = asyncio.Lock()

    @classmethod
    @asynccontextmanager
    async def from_conn_string_with_keep(
        cls, conn_string: str, keep_per_ns: int = _DEFAULT_KEEP_PER_NS
    ) -> AsyncIterator["PruningCheckpointer"]:
        """Build a ``PruningCheckpointer`` from a SQLite connection string.

        Mirrors ``AsyncSqliteSaver.from_conn_string`` but threads
        ``keep_per_ns`` into ``__init__``. The native ``from_conn_string``
        classmethod cannot accept extra kwargs, so callers that need
        retention control should use this method instead.
        """
        async with aiosqlite.connect(conn_string) as conn:
            yield cls(conn, keep_per_ns=keep_per_ns)

    async def aput(
        self,
        config: Any,
        checkpoint: Any,
        metadata: Any,
        new_versions: Any,
    ) -> Any:
        """Delegate to ``super().aput``, then prune older rows atomically.

        Wraps both the inner write and the prune in ``self._aput_lock`` so
        a concurrent ``aput()`` cannot squeeze this caller's just-written
        row out of the top-N retention window. The inner ``self.lock``
        (held by ``super().aput`` and by ``_prune_after_put``) is a
        separate, finer-grained lock that protects the SQLite connection;
        the outer lock here is about the put+prune pair invariant.

        Pruning is best-effort: any exception is logged at WARNING and
        swallowed so a transient SQLite error never fails the agent step.
        """
        async with self._aput_lock:
            result = await super().aput(config, checkpoint, metadata, new_versions)
            if self._keep_per_ns <= 0:
                return result
            try:
                thread_id = config["configurable"]["thread_id"]
                checkpoint_ns = config["configurable"].get("checkpoint_ns", "") or ""
                await self._prune_after_put(str(thread_id), str(checkpoint_ns))
            except Exception as exc:  # pragma: no cover - defensive
                _logger.warning("checkpoint pruning failed: %s", exc, exc_info=True)
            return result

    async def _prune_after_put(self, thread_id: str, checkpoint_ns: str) -> None:
        """Run the two DELETE queries (writes first, then checkpoints).

        Restricted to rows whose ``metadata.agent_name == AGENT_NAME``.
        ``json_extract(metadata, '$.agent_name') = ?`` evaluates to NULL
        (and so fails the predicate) for any row whose metadata does not
        carry an ``agent_name`` key — by design, those rows belong to
        third-party LangGraph users and must never be pruned by us.

        Runs through the saver's connection and lock for atomicity with
        concurrent ``aput()`` calls on the same thread.
        """
        keep = self._keep_per_ns
        agent = AGENT_NAME
        # writes first — we look up which checkpoint ids will be deleted, then
        # delete the writes pointing at them. Doing checkpoints first would
        # leave orphan writes whose `checkpoint_id` we can no longer resolve.
        del_writes = (
            "DELETE FROM writes "
            "WHERE thread_id = ? AND checkpoint_ns = ? "
            "  AND checkpoint_id IN ("
            "    SELECT checkpoint_id FROM checkpoints "
            "    WHERE thread_id = ? AND checkpoint_ns = ? "
            "      AND json_extract(metadata, '$.agent_name') = ? "
            "      AND checkpoint_id NOT IN ("
            "        SELECT checkpoint_id FROM checkpoints "
            "        WHERE thread_id = ? AND checkpoint_ns = ? "
            "          AND json_extract(metadata, '$.agent_name') = ? "
            "        ORDER BY checkpoint_id DESC LIMIT ?"
            "      )"
            "  )"
        )
        del_checkpoints = (
            "DELETE FROM checkpoints "
            "WHERE thread_id = ? AND checkpoint_ns = ? "
            "  AND json_extract(metadata, '$.agent_name') = ? "
            "  AND checkpoint_id NOT IN ("
            "    SELECT checkpoint_id FROM checkpoints "
            "    WHERE thread_id = ? AND checkpoint_ns = ? "
            "      AND json_extract(metadata, '$.agent_name') = ? "
            "    ORDER BY checkpoint_id DESC LIMIT ?"
            "  )"
        )
        async with self.lock:
            # Skip the writes DELETE on a legacy DB that only has the
            # checkpoints table — without this guard, a missing ``writes``
            # would raise sqlite3.OperationalError, the outer try/except
            # in ``aput`` would log+swallow it, and pruning would silently
            # stop working on the very databases this feature is meant to
            # clean up. ``AsyncSqliteSaver.setup()`` normally creates both
            # tables, but inherited DBs from older builds may lag.
            if await _table_exists(self.conn, "writes"):
                await self.conn.execute(
                    del_writes,
                    (
                        thread_id,
                        checkpoint_ns,
                        thread_id,
                        checkpoint_ns,
                        agent,
                        thread_id,
                        checkpoint_ns,
                        agent,
                        keep,
                    ),
                )
            await self.conn.execute(
                del_checkpoints,
                (
                    thread_id,
                    checkpoint_ns,
                    agent,
                    thread_id,
                    checkpoint_ns,
                    agent,
                    keep,
                ),
            )
            await self.conn.commit()


# ---------------------------------------------------------------------------
# Checkpointer context manager
# ---------------------------------------------------------------------------


def _resolve_keep_per_ns() -> int:
    """Resolve the retention count from EvoScientistConfig, with safe fallback."""
    try:
        from .config import get_effective_config

        return max(0, int(get_effective_config().checkpoint_keep_per_thread))
    except Exception:  # pragma: no cover - defensive (config import errors)
        return _DEFAULT_KEEP_PER_NS


@asynccontextmanager
async def get_checkpointer() -> AsyncIterator[PruningCheckpointer]:
    """Yield a pruning-enabled checkpointer connected to the sessions DB.

    Wraps ``AsyncSqliteSaver`` with ``PruningCheckpointer`` so every
    super-step trims the per-(thread, ns) history to ``keep_per_ns``. The
    retention count is read from ``EvoScientistConfig`` at context entry;
    setting it to 0 disables pruning entirely.

    Also opportunistically kicks the legacy-bloat migration sweep as a
    background task on first entry of an oversized DB. The sweep is a
    no-op once ``PRAGMA user_version`` has been bumped, so subsequent
    invocations cost nothing.
    """
    keep = _resolve_keep_per_ns()
    async with PruningCheckpointer.from_conn_string_with_keep(
        str(get_db_path()), keep_per_ns=keep
    ) as saver:
        # Fire-and-forget; runs concurrent with the agent on the same loop.
        maybe_kick_migration_sweep(keep)
        yield saver


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _table_exists(conn: aiosqlite.Connection, table: str) -> bool:
    query = "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?"
    async with conn.execute(query, (table,)) as cur:
        return await cur.fetchone() is not None


async def _load_checkpoint_messages(
    conn: aiosqlite.Connection,
    thread_id: str,
    serde: JsonPlusSerializer,
) -> list:
    """Load messages from the most recent checkpoint for *thread_id*.

    Returns a list of LangChain message objects, or an empty list on failure.
    """
    channel_values = await _load_checkpoint_channel_values(conn, thread_id, serde)
    messages = channel_values.get("messages", [])
    if not isinstance(messages, list):
        return []
    event = channel_values.get("_summarization_event")
    return _apply_summarization_event(
        messages, event if isinstance(event, dict) else None
    )


async def _load_checkpoint_channel_values(
    conn: aiosqlite.Connection,
    thread_id: str,
    serde: JsonPlusSerializer,
) -> dict:
    """Load channel_values from the most recent checkpoint for *thread_id*."""
    query = """
        SELECT type, checkpoint
        FROM checkpoints
        WHERE thread_id = ?
          AND json_extract(metadata, '$.agent_name') = ?
        ORDER BY checkpoint_id DESC
        LIMIT 1
    """
    async with conn.execute(query, (thread_id, AGENT_NAME)) as cur:
        row = await cur.fetchone()
        if not row or not row[0] or not row[1]:
            return {}
        try:
            data = serde.loads_typed((row[0], row[1]))
            channel_values = data.get("channel_values", {})
            return channel_values if isinstance(channel_values, dict) else {}
        except (ValueError, TypeError, KeyError):
            return {}


def _apply_summarization_event(messages: list, event: dict | None) -> list:
    """Return the effective message list after applying a summarization event."""
    if not event:
        return list(messages)

    try:
        summary_message = event["summary_message"]
        cutoff_index = int(event["cutoff_index"])
    except (KeyError, TypeError, ValueError):
        return list(messages)

    if summary_message is None:
        return list(messages)

    if cutoff_index < 0 or cutoff_index > len(messages):
        return list(messages)

    return [summary_message, *messages[cutoff_index:]]


async def _count_messages(
    conn: aiosqlite.Connection,
    thread_id: str,
    serde: JsonPlusSerializer,
) -> int:
    """Count messages in the most recent checkpoint for *thread_id*."""
    msgs = await _load_checkpoint_messages(conn, thread_id, serde)
    return len(msgs)


def _extract_preview(messages: list, max_len: int = 50) -> str:
    """Extract the first human message as a preview string."""
    for msg in messages:
        if getattr(msg, "type", None) != "human":
            continue
        content = getattr(msg, "content", "") or ""
        if isinstance(content, list):
            parts = [
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            content = " ".join(parts)
        content = content.strip()
        if content:
            return content[:max_len] + "..." if len(content) > max_len else content
    return ""


def _format_relative_time(iso_ts: str | None) -> str:
    """Convert ISO timestamp to a human-readable relative string."""
    if not iso_ts:
        return ""
    try:
        dt = datetime.fromisoformat(iso_ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        now = datetime.now(UTC)
        delta = now - dt
        seconds = int(delta.total_seconds())
        if seconds < 60:
            return "just now"
        minutes = seconds // 60
        if minutes < 60:
            return f"{minutes} min ago"
        hours = minutes // 60
        if hours < 24:
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        days = hours // 24
        if days < 30:
            return f"{days} day{'s' if days != 1 else ''} ago"
        months = days // 30
        return f"{months} month{'s' if months != 1 else ''} ago"
    except (ValueError, TypeError):
        return ""


# ---------------------------------------------------------------------------
# Thread CRUD
# ---------------------------------------------------------------------------


async def list_threads(
    limit: int = 20,
    include_message_count: bool = False,
    include_preview: bool = False,
) -> list[dict]:
    """List EvoScientist threads, most-recent first.

    Returns list of dicts with keys: ``thread_id``, ``updated_at``,
    ``workspace_dir``, ``model``, and optionally ``message_count``
    and ``preview``.
    """
    db_path = str(get_db_path())
    async with aiosqlite.connect(db_path, timeout=30.0) as conn:
        if not await _table_exists(conn, "checkpoints"):
            return []

        query = """
            SELECT thread_id,
                   MAX(json_extract(metadata, '$.updated_at')) as updated_at,
                   json_extract(metadata, '$.workspace_dir') as workspace_dir,
                   json_extract(metadata, '$.model') as model
            FROM checkpoints
            WHERE json_extract(metadata, '$.agent_name') = ?
            GROUP BY thread_id
            ORDER BY updated_at DESC
        """
        params: tuple = (AGENT_NAME,)
        if limit > 0:
            query += "    LIMIT ?\n"
            params = (AGENT_NAME, limit)
        async with conn.execute(query, params) as cur:
            rows = await cur.fetchall()

        threads = [
            {
                "thread_id": r[0],
                "updated_at": r[1],
                "workspace_dir": r[2],
                "model": r[3],
            }
            for r in rows
        ]

        if (include_message_count or include_preview) and threads:
            serde = JsonPlusSerializer()
            for t in threads:
                msgs = await _load_checkpoint_messages(conn, t["thread_id"], serde)
                if include_message_count:
                    t["message_count"] = len(msgs)
                if include_preview:
                    t["preview"] = _extract_preview(msgs)

        return threads


async def get_most_recent() -> str | None:
    """Return the most recent EvoScientist thread ID, or ``None``."""
    db_path = str(get_db_path())
    async with aiosqlite.connect(db_path, timeout=30.0) as conn:
        if not await _table_exists(conn, "checkpoints"):
            return None
        query = """
            SELECT thread_id FROM checkpoints
            WHERE json_extract(metadata, '$.agent_name') = ?
            ORDER BY checkpoint_id DESC
            LIMIT 1
        """
        async with conn.execute(query, (AGENT_NAME,)) as cur:
            row = await cur.fetchone()
            return row[0] if row else None


async def thread_exists(thread_id: str) -> bool:
    """Return ``True`` if *thread_id* has at least one EvoScientist checkpoint."""
    db_path = str(get_db_path())
    async with aiosqlite.connect(db_path, timeout=30.0) as conn:
        if not await _table_exists(conn, "checkpoints"):
            return False
        query = """
            SELECT 1 FROM checkpoints
            WHERE thread_id = ? AND json_extract(metadata, '$.agent_name') = ?
            LIMIT 1
        """
        async with conn.execute(query, (thread_id, AGENT_NAME)) as cur:
            return (await cur.fetchone()) is not None


async def find_similar_threads(thread_id: str, limit: int = 5) -> list[str]:
    """Find EvoScientist thread IDs that start with *thread_id* (prefix match)."""
    db_path = str(get_db_path())
    async with aiosqlite.connect(db_path, timeout=30.0) as conn:
        if not await _table_exists(conn, "checkpoints"):
            return []
        # Escape SQL LIKE wildcards so user-supplied prefixes are matched
        # literally (e.g. `--resume %` must not match every thread).
        escaped = (
            thread_id.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        )
        query = r"""
            SELECT DISTINCT thread_id
            FROM checkpoints
            WHERE thread_id LIKE ? ESCAPE '\'
              AND json_extract(metadata, '$.agent_name') = ?
            ORDER BY thread_id
            LIMIT ?
        """
        async with conn.execute(query, (escaped + "%", AGENT_NAME, limit)) as cur:
            rows = await cur.fetchall()
            return [r[0] for r in rows]


async def resolve_thread_id_prefix(tid: str) -> tuple[str | None, list[str]]:
    """Resolve a (possibly partial) thread ID.

    Returns ``(resolved_id, matches)``:
    - ``(full_id, [])`` when *tid* is an exact hit or a unique prefix.
    - ``(None, [a, b, ...])`` when the prefix is ambiguous (multiple matches).
    - ``(None, [])`` when no thread matches.
    """
    if await thread_exists(tid):
        return tid, []
    similar = await find_similar_threads(tid)
    if len(similar) == 1:
        return similar[0], []
    return None, similar


async def delete_thread(thread_id: str) -> bool:
    """Delete all EvoScientist checkpoints (and writes) for *thread_id*."""
    db_path = str(get_db_path())
    async with aiosqlite.connect(db_path, timeout=30.0) as conn:
        if not await _table_exists(conn, "checkpoints"):
            return False
        # Delete writes FIRST — the subquery needs checkpoints to still exist
        if await _table_exists(conn, "writes"):
            await conn.execute(
                """DELETE FROM writes
                   WHERE thread_id = ?
                     AND checkpoint_id IN (
                         SELECT checkpoint_id FROM checkpoints
                         WHERE thread_id = ?
                           AND json_extract(metadata, '$.agent_name') = ?
                     )""",
                (thread_id, thread_id, AGENT_NAME),
            )
        cur = await conn.execute(
            "DELETE FROM checkpoints WHERE thread_id = ? AND json_extract(metadata, '$.agent_name') = ?",
            (thread_id, AGENT_NAME),
        )
        deleted = cur.rowcount > 0
        await conn.commit()
        return deleted


async def get_thread_metadata(thread_id: str) -> dict | None:
    """Return metadata dict for *thread_id*, or ``None`` if not found.

    Keys: ``workspace_dir``, ``model``, ``updated_at``.
    """
    db_path = str(get_db_path())
    async with aiosqlite.connect(db_path, timeout=30.0) as conn:
        if not await _table_exists(conn, "checkpoints"):
            return None
        query = """
            SELECT json_extract(metadata, '$.workspace_dir') as workspace_dir,
                   json_extract(metadata, '$.model') as model,
                   json_extract(metadata, '$.updated_at') as updated_at
            FROM checkpoints
            WHERE thread_id = ?
              AND json_extract(metadata, '$.agent_name') = ?
            ORDER BY checkpoint_id DESC
            LIMIT 1
        """
        async with conn.execute(query, (thread_id, AGENT_NAME)) as cur:
            row = await cur.fetchone()
            if not row:
                return None
            return {
                "workspace_dir": row[0],
                "model": row[1],
                "updated_at": row[2],
            }


async def get_thread_messages(thread_id: str) -> list:
    """Return the list of LangChain message objects for *thread_id*.

    Only returns messages for EvoScientist threads.
    Returns an empty list if the thread has no checkpoints.
    """
    db_path = str(get_db_path())
    async with aiosqlite.connect(db_path, timeout=30.0) as conn:
        if not await _table_exists(conn, "checkpoints"):
            return []
        # Verify this thread belongs to EvoScientist before loading messages
        check = """
            SELECT 1 FROM checkpoints
            WHERE thread_id = ? AND json_extract(metadata, '$.agent_name') = ?
            LIMIT 1
        """
        async with conn.execute(check, (thread_id, AGENT_NAME)) as cur:
            if not await cur.fetchone():
                return []
        serde = JsonPlusSerializer()
        channel_values = await _load_checkpoint_channel_values(conn, thread_id, serde)
        messages = channel_values.get("messages", [])
        event = channel_values.get("_summarization_event")
        return _apply_summarization_event(messages, event)


# ---------------------------------------------------------------------------
# Migration sweep & VACUUM (one-time legacy cleanup)
# ---------------------------------------------------------------------------

# PRAGMA user_version is a 32-bit int slot in the SQLite file header. We
# bump this to 1 once the legacy-bloat sweep has run successfully so it
# never runs again. Future structural migrations can use 2, 3, ...
_MIGRATION_VERSION = 1

# Threshold below which the sweep is skipped (DB is already small enough
# that legacy bloat is not the user's problem). 100 MB is chosen so a
# normally-pruned DB after a few months of use never triggers the sweep,
# while the 2.6 GB pathology is comfortably above the line.
_MIGRATION_THRESHOLD_BYTES = 100 * 1024 * 1024

# Inter-pair sleep so the sweep yields to the agent loop and never spikes
# CPU on a large DB. Tunable for tests via monkeypatch.
_SWEEP_YIELD_SECONDS = 0.0


async def _get_user_version(conn: aiosqlite.Connection) -> int:
    async with conn.execute("PRAGMA user_version") as cur:
        row = await cur.fetchone()
        return int(row[0]) if row else 0


async def _set_user_version(conn: aiosqlite.Connection, version: int) -> None:
    # PRAGMAs cannot be parameter-bound; the integer is interpolated safely
    # because we control the value (constant int).
    await conn.execute(f"PRAGMA user_version = {int(version)}")
    await conn.commit()


async def _needs_migration() -> bool:
    """Return True if the legacy-bloat sweep should run now.

    True iff the DB exists, is larger than ``_MIGRATION_THRESHOLD_BYTES``,
    and ``PRAGMA user_version`` is below ``_MIGRATION_VERSION``.
    """
    db_path = get_db_path()
    if not db_path.exists():
        return False
    try:
        size = db_path.stat().st_size
    except OSError:
        return False
    if size < _MIGRATION_THRESHOLD_BYTES:
        return False
    try:
        async with aiosqlite.connect(str(db_path), timeout=30.0) as conn:
            if not await _table_exists(conn, "checkpoints"):
                return False
            return await _get_user_version(conn) < _MIGRATION_VERSION
    except aiosqlite.Error:
        return False


async def _run_migration_sweep(keep: int) -> int:
    """Prune all ``(thread_id, checkpoint_ns)`` pairs to ``keep`` rows each.

    Iterates pairs in deterministic order, applies the same DELETE pattern
    the per-step pruner uses, and yields to the event loop between pairs
    so the agent stays responsive. On success bumps ``PRAGMA user_version``
    so the sweep never reruns.

    Returns the number of pairs pruned.
    """
    if keep <= 0:
        return 0
    db_path = str(get_db_path())
    pairs_pruned = 0
    async with aiosqlite.connect(db_path, timeout=60.0) as conn:
        if not await _table_exists(conn, "checkpoints"):
            return 0
        if await _get_user_version(conn) >= _MIGRATION_VERSION:
            return 0
        # ``writes`` is optional on legacy DBs: skip the writes DELETE
        # if the table is absent rather than aborting the whole sweep.
        has_writes = await _table_exists(conn, "writes")

        async with conn.execute(
            "SELECT DISTINCT thread_id, checkpoint_ns FROM checkpoints "
            "WHERE json_extract(metadata, '$.agent_name') = ?",
            (AGENT_NAME,),
        ) as cur:
            pairs = await cur.fetchall()

        del_writes = (
            "DELETE FROM writes "
            "WHERE thread_id = ? AND checkpoint_ns = ? "
            "  AND checkpoint_id IN ("
            "    SELECT checkpoint_id FROM checkpoints "
            "    WHERE thread_id = ? AND checkpoint_ns = ? "
            "      AND json_extract(metadata, '$.agent_name') = ? "
            "      AND checkpoint_id NOT IN ("
            "        SELECT checkpoint_id FROM checkpoints "
            "        WHERE thread_id = ? AND checkpoint_ns = ? "
            "          AND json_extract(metadata, '$.agent_name') = ? "
            "        ORDER BY checkpoint_id DESC LIMIT ?"
            "      )"
            "  )"
        )
        del_checkpoints = (
            "DELETE FROM checkpoints "
            "WHERE thread_id = ? AND checkpoint_ns = ? "
            "  AND json_extract(metadata, '$.agent_name') = ? "
            "  AND checkpoint_id NOT IN ("
            "    SELECT checkpoint_id FROM checkpoints "
            "    WHERE thread_id = ? AND checkpoint_ns = ? "
            "      AND json_extract(metadata, '$.agent_name') = ? "
            "    ORDER BY checkpoint_id DESC LIMIT ?"
            "  )"
        )
        for thread_id, checkpoint_ns in pairs:
            ns = checkpoint_ns or ""
            if has_writes:
                await conn.execute(
                    del_writes,
                    (
                        thread_id,
                        ns,
                        thread_id,
                        ns,
                        AGENT_NAME,
                        thread_id,
                        ns,
                        AGENT_NAME,
                        keep,
                    ),
                )
            await conn.execute(
                del_checkpoints,
                (
                    thread_id,
                    ns,
                    AGENT_NAME,
                    thread_id,
                    ns,
                    AGENT_NAME,
                    keep,
                ),
            )
            await conn.commit()
            pairs_pruned += 1
            if _SWEEP_YIELD_SECONDS >= 0:
                await asyncio.sleep(_SWEEP_YIELD_SECONDS)

        await _set_user_version(conn, _MIGRATION_VERSION)

    # Schedule VACUUM at process exit (must run after the long-lived saver
    # connection closes to acquire the exclusive lock VACUUM requires).
    # Pass ``db_path`` explicitly so test-time monkeypatches of
    # ``get_db_path`` don't leak into atexit and hit the real DB.
    _schedule_vacuum_atexit(db_path)
    return pairs_pruned


_vacuum_scheduled = False


def _schedule_vacuum_atexit(db_path: str) -> None:
    """Register the atexit VACUUM hook exactly once per process.

    Captures ``db_path`` at registration time so the hook always operates
    on the path that was current when the sweep ran. This matters in
    tests, where ``get_db_path`` is monkey-patched to a temp file but the
    patch has unwound by the time atexit fires — re-resolving at exit
    would point at the user's real ``sessions.db`` and trigger an
    unwanted VACUUM on production data.
    """
    global _vacuum_scheduled
    if _vacuum_scheduled:
        return
    _vacuum_scheduled = True
    atexit.register(_atexit_vacuum, db_path)


def _atexit_vacuum(db_path: str) -> None:
    """Run ``VACUUM`` synchronously at process exit on the captured path.

    Uses stdlib ``sqlite3`` (atexit can't await aiosqlite). Best-effort:
    swallow any error since this runs during shutdown when stderr may be
    closed.
    """
    import os
    import sqlite3

    if not os.path.exists(db_path):
        return
    try:
        with sqlite3.connect(db_path, timeout=60.0) as conn:
            # VACUUM cannot run inside a transaction; sqlite3 starts one
            # implicitly on the first execute, so isolation_level=None ensures
            # we are in autocommit mode for the VACUUM statement.
            conn.isolation_level = None
            conn.execute("VACUUM")
    except sqlite3.Error as exc:
        # Best-effort during shutdown: stderr may already be closed, but
        # try to log so a persistent VACUUM failure is at least diagnosable
        # from the next session. Swallow any logging error in turn.
        try:
            _logger.warning("VACUUM at exit failed: %s", exc)
        except Exception:
            pass


def maybe_kick_migration_sweep(keep: int) -> asyncio.Task | None:
    """Schedule the legacy-bloat sweep as a background task if needed.

    Returns the spawned ``asyncio.Task`` (caller does NOT await — sweep
    runs concurrent with the agent), or ``None`` if migration is not
    needed. Safe to call multiple times: the task itself rechecks the
    user_version marker so a duplicate fire is a no-op.
    """
    if keep <= 0:
        return None
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        return None

    async def _runner() -> None:
        try:
            if not await _needs_migration():
                return
            pairs = await _run_migration_sweep(keep)
            if pairs:
                _logger.info(
                    "checkpoint migration sweep pruned %d (thread, ns) pairs; "
                    "VACUUM will run at exit",
                    pairs,
                )
        except Exception as exc:  # pragma: no cover - defensive
            _logger.warning("migration sweep failed: %s", exc, exc_info=True)

    return loop.create_task(_runner())


# ---------------------------------------------------------------------------
# DB stats (read-only diagnostic for `EvoSci sessions stats`)
# ---------------------------------------------------------------------------


async def db_stats(top_n: int = 5) -> dict[str, Any]:
    """Return read-only diagnostics about the sessions DB.

    Keys: ``db_path`` (str), ``size_bytes`` (int, file size or 0 if absent),
    ``thread_count`` (int, EvoScientist threads), ``checkpoint_count``
    (int, EvoScientist rows), ``write_count`` (int, EvoScientist writes
    only — scoped via JOIN to ``checkpoints.metadata.agent_name`` so
    co-located non-EvoSci agents are excluded), ``top_threads`` (list of
    dicts with ``thread_id`` and ``count``, sorted desc).
    """
    db_path = get_db_path()
    size = db_path.stat().st_size if db_path.exists() else 0
    out: dict[str, Any] = {
        "db_path": str(db_path),
        "size_bytes": size,
        "thread_count": 0,
        "checkpoint_count": 0,
        "write_count": 0,
        "top_threads": [],
    }
    if not db_path.exists():
        return out
    try:
        async with aiosqlite.connect(str(db_path), timeout=30.0) as conn:
            if not await _table_exists(conn, "checkpoints"):
                return out
            async with conn.execute(
                "SELECT COUNT(DISTINCT thread_id), COUNT(*) FROM checkpoints "
                "WHERE json_extract(metadata, '$.agent_name') = ?",
                (AGENT_NAME,),
            ) as cur:
                row = await cur.fetchone()
                if row:
                    out["thread_count"] = int(row[0] or 0)
                    out["checkpoint_count"] = int(row[1] or 0)

            if await _table_exists(conn, "writes"):
                # Scope writes to EvoScientist rows by joining against
                # checkpoints — the ``writes`` table itself has no
                # ``agent_name`` column, so a bare ``COUNT(*)`` would
                # over-report when other LangGraph apps share this DB.
                async with conn.execute(
                    "SELECT COUNT(*) FROM writes w "
                    "JOIN checkpoints c "
                    "  ON c.thread_id = w.thread_id "
                    " AND c.checkpoint_ns = w.checkpoint_ns "
                    " AND c.checkpoint_id = w.checkpoint_id "
                    "WHERE json_extract(c.metadata, '$.agent_name') = ?",
                    (AGENT_NAME,),
                ) as cur:
                    row = await cur.fetchone()
                    if row:
                        out["write_count"] = int(row[0] or 0)

            async with conn.execute(
                "SELECT thread_id, COUNT(*) AS n FROM checkpoints "
                "WHERE json_extract(metadata, '$.agent_name') = ? "
                "GROUP BY thread_id ORDER BY n DESC LIMIT ?",
                (AGENT_NAME, int(top_n)),
            ) as cur:
                rows = await cur.fetchall()
                out["top_threads"] = [
                    {"thread_id": r[0], "count": int(r[1])} for r in rows
                ]
    except aiosqlite.Error:
        # Read-only — corrupt/locked DB → return zeroed stats rather than crash.
        return out
    return out
