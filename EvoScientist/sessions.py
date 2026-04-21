"""Session persistence using LangGraph's SQLite checkpoint storage.

Provides thread CRUD operations, prefix-matched resume, and an async
context manager for the shared ``AsyncSqliteSaver`` checkpointer.

Adapted from upstream ``deepagents_cli/sessions.py``.
"""

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

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
# Checkpointer context manager
# ---------------------------------------------------------------------------


@asynccontextmanager
async def get_checkpointer() -> AsyncIterator[AsyncSqliteSaver]:
    """Yield an ``AsyncSqliteSaver`` connected to the global sessions DB."""
    async with AsyncSqliteSaver.from_conn_string(str(get_db_path())) as cp:
        yield cp


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
