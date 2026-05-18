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
import math
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Overwrite

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
# Kept in sync with the dataclass default so config-failure fallbacks
# don't silently regress to the pre-DeltaChannel aggressive value (which
# could prune away ``_DeltaSnapshot`` seeds and break message replay).
_DEFAULT_KEEP_PER_NS = 1000


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

    # Safety cap on the snapshot walk. Upstream default
    # ``snapshot_frequency`` is 1000; ``DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT``
    # is 5000. 10000 is a generous ceiling that catches pathological data
    # (cycles, malformed parents) without raising a hard limit on normal
    # operation.
    _MAX_SNAPSHOT_WALK_STEPS = 10000

    async def _prune_after_put(self, thread_id: str, checkpoint_ns: str) -> None:
        """Prune old checkpoints with DeltaChannel awareness.

        Naively keeping the N most-recent rows can sever the
        ``_DeltaSnapshot`` chain that ``messages`` reconstruction
        depends on — the surviving "latest" checkpoint is rarely a
        snapshot point itself, so delta channels silently reconstruct
        as empty (upstream ``BaseCheckpointSaver.prune`` spells out the
        same failure mode).

        After selecting the N most-recent anchor ids, walk back from
        the OLDEST anchor's parent via ``parent_checkpoint_id`` until
        hitting an ancestor whose ``channel_values["messages"]`` is a
        seed (``_DeltaSnapshot`` blob or plain list — both detected via
        ``_unwrap_messages_seed``). All visited ancestors are preserved
        alongside the anchor set. Anchors form a contiguous head, so a
        single walk from the oldest one covers all of them.

        Restricted to rows whose ``metadata.agent_name == AGENT_NAME``.
        ``json_extract(metadata, '$.agent_name') = ?`` evaluates to NULL
        (and so fails the predicate) for any row whose metadata lacks an
        ``agent_name`` key — by design, those rows belong to third-party
        LangGraph users and must never be pruned by us.

        Walk + DELETEs held under ``self.lock`` for atomicity with
        concurrent ``aput()`` on the same thread.
        """
        keep = self._keep_per_ns
        agent = AGENT_NAME
        async with self.lock:
            # Skip on a legacy DB lacking the ``writes`` table — without
            # this guard, the writes DELETE would raise OperationalError
            # and the caller's try/except would silently swallow the entire
            # prune. ``AsyncSqliteSaver.setup()`` creates both tables but
            # inherited DBs from older builds may lag.
            if not await _table_exists(self.conn, "writes"):
                return

            anchor_ids = await self._fetch_recent_checkpoint_ids(
                thread_id, checkpoint_ns, agent, keep
            )
            if len(anchor_ids) < keep:
                return  # nothing to prune yet

            extra_preserve = await self._walk_to_snapshot_ancestor(
                thread_id, checkpoint_ns, anchor_ids[-1]
            )
            kept = set(anchor_ids) | extra_preserve

            await self._delete_outside(thread_id, checkpoint_ns, agent, kept)
            await self.conn.commit()

    async def _fetch_recent_checkpoint_ids(
        self,
        thread_id: str,
        checkpoint_ns: str,
        agent: str,
        limit: int,
    ) -> list[str]:
        """Return the ``limit`` most-recent checkpoint ids (newest first)."""
        query = (
            "SELECT checkpoint_id FROM checkpoints "
            "WHERE thread_id = ? AND checkpoint_ns = ? "
            "  AND json_extract(metadata, '$.agent_name') = ? "
            "ORDER BY checkpoint_id DESC LIMIT ?"
        )
        async with self.conn.execute(
            query, (thread_id, checkpoint_ns, agent, limit)
        ) as cur:
            rows = await cur.fetchall()
        return [r[0] for r in rows]

    async def _walk_to_snapshot_ancestor(
        self,
        thread_id: str,
        checkpoint_ns: str,
        oldest_anchor_id: str,
    ) -> set[str]:
        """Walk parent chain until hitting a ``messages`` seed.

        Returns the set of ancestor ids to preserve (inclusive of the
        snapshot ancestor). On chain-break or deserialization failure,
        returns what was visited so far — the safe side is over-preserve.
        """
        extra: set[str] = set()
        cursor = await self._fetch_parent_checkpoint_id(
            thread_id, checkpoint_ns, oldest_anchor_id
        )
        steps = 0
        while cursor is not None and steps < self._MAX_SNAPSHOT_WALK_STEPS:
            steps += 1
            blob = await self._fetch_checkpoint_blob(thread_id, checkpoint_ns, cursor)
            if blob is None:
                break  # chain broken (legacy DB); preserve what we have
            extra.add(cursor)
            try:
                ck = self.serde.loads_typed(blob)
            except Exception as exc:
                _logger.warning(
                    "Failed to deserialize checkpoint %s while walking to "
                    "snapshot for thread %s: %s",
                    cursor,
                    thread_id,
                    exc,
                )
                break  # safe-side: preserve everything visited so far
            cv = ck.get("channel_values") or {}
            if _unwrap_messages_seed(cv.get("messages")) is not None:
                break  # found seed; this ancestor anchors reconstruction
            cursor = await self._fetch_parent_checkpoint_id(
                thread_id, checkpoint_ns, cursor
            )
        return extra

    async def _fetch_parent_checkpoint_id(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> str | None:
        query = (
            "SELECT parent_checkpoint_id FROM checkpoints "
            "WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?"
        )
        async with self.conn.execute(
            query, (thread_id, checkpoint_ns, checkpoint_id)
        ) as cur:
            row = await cur.fetchone()
        return row[0] if row and row[0] else None

    async def _fetch_checkpoint_blob(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> tuple[str, bytes] | None:
        query = (
            "SELECT type, checkpoint FROM checkpoints "
            "WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?"
        )
        async with self.conn.execute(
            query, (thread_id, checkpoint_ns, checkpoint_id)
        ) as cur:
            row = await cur.fetchone()
        if not row or not row[0] or not row[1]:
            return None
        return (row[0], row[1])

    async def _delete_outside(
        self,
        thread_id: str,
        checkpoint_ns: str,
        agent: str,
        kept_ids: set[str],
    ) -> None:
        """DELETE rows whose ``checkpoint_id`` is NOT in ``kept_ids``.

        Writes deleted first to preserve referential ordering — if we
        dropped checkpoints first, surviving writes' ``checkpoint_id``
        would become orphans.

        Empty ``kept_ids`` is a no-op rather than "delete everything" —
        a defensive check; the caller always passes anchor_ids which is
        non-empty by construction (already checked ``len >= keep`` in
        the caller).
        """
        if not kept_ids:
            return
        kept_list = list(kept_ids)
        placeholders = ",".join("?" * len(kept_list))

        del_writes = (
            "DELETE FROM writes "
            "WHERE thread_id = ? AND checkpoint_ns = ? "
            "  AND checkpoint_id IN ("
            "    SELECT checkpoint_id FROM checkpoints "
            "    WHERE thread_id = ? AND checkpoint_ns = ? "
            "      AND json_extract(metadata, '$.agent_name') = ? "
            f"     AND checkpoint_id NOT IN ({placeholders})"
            "  )"
        )
        await self.conn.execute(
            del_writes,
            (
                thread_id,
                checkpoint_ns,
                thread_id,
                checkpoint_ns,
                agent,
                *kept_list,
            ),
        )

        del_checkpoints = (
            "DELETE FROM checkpoints "
            "WHERE thread_id = ? AND checkpoint_ns = ? "
            "  AND json_extract(metadata, '$.agent_name') = ? "
            f" AND checkpoint_id NOT IN ({placeholders})"
        )
        await self.conn.execute(
            del_checkpoints,
            (thread_id, checkpoint_ns, agent, *kept_list),
        )


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

    Also runs the legacy-bloat migration sweep synchronously *before*
    yielding the saver when needed. Sequencing sweep ahead of any agent
    ``aput()`` eliminates the SQLite file-lock contention that produced
    "database is locked" when channel inbound raced the sweep mid-DELETE.
    The sweep is gated by ``PRAGMA user_version`` so it runs at most
    once across all future launches; subsequent invocations cost nothing.
    On failure ``user_version`` is NOT bumped, so the next launch retries.
    """
    keep = _resolve_keep_per_ns()
    async with PruningCheckpointer.from_conn_string_with_keep(
        str(get_db_path()), keep_per_ns=keep
    ) as saver:
        # The whole gate is wrapped in a broad try/except: any unexpected
        # failure (incl. preview / status print) must degrade to "yield
        # the saver, log a warning" — never to "hang startup".
        if keep > 0:
            try:
                if await _needs_migration():
                    import time

                    from rich.console import Console

                    # stderr-bound so non-interactive callers redirecting
                    # stdout don't capture migration progress noise.
                    console = Console(stderr=True)
                    size_str, pair_count, size_bytes = await _migration_preview()
                    eta_str = _format_duration(_estimate_sweep_seconds(size_bytes))
                    console.print(
                        f"[dim]·[/dim] Compacting sessions DB "
                        f"([cyan]{size_str}[/cyan], "
                        f"[yellow]{pair_count} thread-namespace pairs[/yellow]"
                        f"[dim], est. ~{eta_str}[/dim])"
                    )

                    t0 = time.time()

                    with console.status(
                        "[dim]Compacting...[/dim]", spinner="dots"
                    ) as status:

                        async def _on_progress(done: int, total: int) -> None:
                            elapsed = time.time() - t0
                            pct = (done * 100 // total) if total else 0
                            eta = (elapsed / done) * (total - done) if done > 0 else 0
                            status.update(
                                f"[dim]Compacting[/dim] "
                                f"[yellow]{pct}%[/yellow] "
                                f"[dim]({done}/{total} pairs · "
                                f"{_format_duration(elapsed)} elapsed · "
                                f"~{_format_duration(eta)} remaining)[/dim]"
                            )

                        await _run_migration_sweep(keep, progress_cb=_on_progress)

                    console.print(
                        f"[dim]·[/dim] [green]✓[/green] Compaction done in "
                        f"[green]{_format_duration(time.time() - t0)}[/green]"
                    )
            except Exception as exc:
                _logger.warning("migration sweep failed: %s", exc, exc_info=True)
        yield saver


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _table_exists(conn: aiosqlite.Connection, table: str) -> bool:
    query = "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?"
    async with conn.execute(query, (table,)) as cur:
        return await cur.fetchone() is not None


async def _load_checkpoint_messages(
    saver: AsyncSqliteSaver,
    thread_id: str,
) -> list:
    """Load messages from the most recent checkpoint for *thread_id*.

    Delegates the ``messages`` DeltaChannel walk to upstream
    ``BaseCheckpointSaver.aget_delta_channel_history`` — it finds the
    nearest ancestor whose ``channel_values["messages"]`` carries a seed
    (``_DeltaSnapshot`` blob or plain list) and returns that plus every
    on-path pending write oldest→newest. The deepagents
    ``_messages_delta_reducer`` is then applied in a single batched call,
    preserving dedup-by-id, ``RemoveMessage`` tombstones, and
    ``REMOVE_ALL_MESSAGES`` reset semantics.

    ``Overwrite`` (``langgraph.types.Overwrite``) is not a message-like
    and the reducer doesn't recognize it — split the batch at each
    occurrence, replacing accumulated state with the wrapped value before
    resuming reducer application.

    ``_summarization_event`` doesn't ride on the ``messages`` channel,
    so it's fetched separately from the latest checkpoint's
    ``channel_values``.

    Returns a list of LangChain message objects, or an empty list on failure.
    """
    try:
        from deepagents._messages_reducer import _messages_delta_reducer
    except ImportError:
        _messages_delta_reducer = None  # type: ignore[assignment]

    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    target = await saver.aget_tuple(config)
    if target is None:
        return []

    # ``aget_delta_channel_history`` walks from ``target.parent_config`` —
    # it deliberately excludes the target itself (its caller is the
    # runtime preparing to apply a NEW delta on top). For /resume we want
    # the state AT the latest checkpoint, so check the target's own
    # ``channel_values`` first, falling back to the ancestor walk only
    # when no seed is materialized locally.
    target_cv = target.checkpoint.get("channel_values") or {}
    target_seed = _unwrap_messages_seed(target_cv.get("messages"))
    if target_seed is not None:
        accumulated: list = target_seed
        writes: list = [w for w in (target.pending_writes or []) if w[1] == "messages"]
    else:
        history = await saver.aget_delta_channel_history(
            config=config, channels=["messages"]
        )
        entry = history.get("messages", {})
        accumulated = _unwrap_messages_seed(entry.get("seed")) or []
        writes = list(entry.get("writes", []))
        # ``target.pending_writes`` are the deltas recorded at this step;
        # apply them on top of whatever the ancestor walk reconstructed.
        writes.extend(w for w in (target.pending_writes or []) if w[1] == "messages")

    # Batched reducer: collect contiguous message-like writes and flush
    # in one call. Overwrite splits the batch because it resets state
    # rather than appending.
    batch: list = []

    def _flush() -> None:
        nonlocal accumulated
        if not batch:
            return
        if _messages_delta_reducer is None:
            # ``deepagents._messages_delta_reducer`` is unavailable
            # (upstream rename / packaging change). A naive concat
            # would silently break dedup-by-id and RemoveMessage /
            # REMOVE_ALL_MESSAGES semantics. Skip the batch with a
            # warning instead — the caller gets a partial but
            # internally consistent view.
            _logger.warning(
                "deepagents._messages_delta_reducer unavailable; "
                "skipping %d messages deltas for thread %s",
                len(batch),
                thread_id,
            )
            batch.clear()
            return
        try:
            accumulated = _messages_delta_reducer(accumulated, batch)
        except Exception as exc:
            _logger.warning(
                "Failed to apply %d messages deltas for thread %s: %s",
                len(batch),
                thread_id,
                exc,
            )
        batch.clear()

    for _task_id, _channel, delta in writes:
        # ``Overwrite`` wraps a value with "replace this channel"
        # semantics — flush any pending writes first, then reset state
        # to the wrapped value.
        if isinstance(delta, Overwrite):
            _flush()
            inner = getattr(delta, "value", None)
            accumulated = (
                list(inner)
                if isinstance(inner, list)
                else ([inner] if inner is not None else [])
            )
        else:
            batch.append(delta)
    _flush()

    # ``_summarization_event`` rides on its own channel; pick it off the
    # target's ``channel_values`` we already deserialized above.
    event = target_cv.get("_summarization_event")
    summarization_event = event if isinstance(event, dict) else None

    if not accumulated:
        await _log_orphan_warning_if_pruned(saver.conn, thread_id)

    if not isinstance(accumulated, list):
        return []
    return _apply_summarization_event(accumulated, summarization_event)


async def _log_orphan_warning_if_pruned(
    conn: aiosqlite.Connection, thread_id: str
) -> None:
    """Emit a WARNING if *thread_id* has orphan ``parent_checkpoint_id`` refs.

    Empty reconstructed history + a broken parent chain is the signature
    of a pre-fix DB where the old ``keep_per_ns=10`` default pruned early
    writes before the snapshot frequency materialized a ``_DeltaSnapshot``
    seed. The DB has no path to recover those messages — log so /resume
    showing a stub history isn't silent.
    """
    query = """
        SELECT 1 FROM checkpoints c1
        WHERE c1.thread_id = ? AND c1.checkpoint_ns = ''
          AND c1.parent_checkpoint_id IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM checkpoints c2
              WHERE c2.thread_id = c1.thread_id
                AND c2.checkpoint_ns = ''
                AND c2.checkpoint_id = c1.parent_checkpoint_id
          )
        LIMIT 1
    """
    async with conn.execute(query, (thread_id,)) as cur:
        if await cur.fetchone():
            _logger.warning(
                "Thread %s has orphan checkpoints (pre-fix pruning) "
                "and no surviving DeltaChannel snapshot; reconstructed "
                "history is empty. Early messages cannot be recovered.",
                thread_id,
            )


def _unwrap_messages_seed(value: object) -> list | None:
    """Coerce a ``channel_values["messages"]`` snapshot seed into a plain list.

    LangGraph 1.2 stores snapshot blobs as ``_DeltaSnapshot(value=[...])``
    (a ``NamedTuple``, NOT a list subclass), so a bare ``isinstance(v, list)``
    check silently ignores the seed and reconstruction starts from whatever
    writes survived pruning. Pre-migration / non-DeltaChannel checkpoints
    still store a plain list. Returns ``None`` when no usable seed is
    present (caller leaves accumulated state untouched).
    """
    if value is None:
        return None
    if isinstance(value, list):
        return list(value)
    inner = getattr(value, "value", None)
    if isinstance(inner, list):
        return list(inner)
    return None


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
            # Share one saver across all threads so ``setup()`` runs once.
            serde = JsonPlusSerializer()
            saver = AsyncSqliteSaver(conn, serde=serde)
            for t in threads:
                msgs = await _load_checkpoint_messages(saver, t["thread_id"])
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

    Reconstructs the full message history by walking the checkpoint chain
    and applying pending writes — required under deepagents 0.6
    ``DeltaChannel`` where messages live in the ``writes`` table rather
    than the latest checkpoint's ``channel_values``.
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
        saver = AsyncSqliteSaver(conn, serde=serde)
        return await _load_checkpoint_messages(saver, thread_id)


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


def _format_duration(seconds: float) -> str:
    """Render a number of seconds as ``45s`` / ``2m 18s`` / ``1h 5m``.

    Guards against ``inf`` / ``nan`` so a stray non-finite value (e.g. an
    ETA computed before any pair completes) cannot raise ``OverflowError``
    via ``int(float('inf'))``.
    """
    if not math.isfinite(seconds):
        return "?"
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{m}m {s}s" if s else f"{m}m"
    h, rem = divmod(seconds, 3600)
    m = rem // 60
    return f"{h}h {m}m" if m else f"{h}h"


def _estimate_sweep_seconds(size_bytes: int) -> int:
    """Rough ETA for a full migration sweep on the given DB size.

    Empirical baseline: 14.53 GB ≈ 138s on a typical SSD, i.e. ~10 s/GB.
    Used only for a user-facing pre-sweep hint — real time is reported
    after completion so the estimate doesn't need to be precise.
    """
    return max(2, round(size_bytes / (1024 * 1024 * 1024) * 9.5))


async def _migration_preview() -> tuple[str, int, int]:
    """Return ``(size_str, pair_count, size_bytes)`` for the pre-sweep status line.

    ``size_str`` is a human-readable file size (``"2.62 GB"`` / ``"364.3 MB"``).
    ``pair_count`` is the number of distinct ``(thread_id, checkpoint_ns)``
    pairs the sweep will iterate. ``size_bytes`` is the raw byte count
    (used by ``_estimate_sweep_seconds`` for the ETA). Best-effort:
    returns zeros on any error.
    """
    db_path = get_db_path()
    try:
        size_bytes = db_path.stat().st_size
    except OSError:
        size_bytes = 0
    if size_bytes < 1024 * 1024:
        size_str = f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        size_str = f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

    pair_count = 0
    try:
        async with aiosqlite.connect(str(db_path), timeout=30.0) as conn:
            if await _table_exists(conn, "checkpoints"):
                async with conn.execute(
                    "SELECT COUNT(*) FROM ("
                    "  SELECT DISTINCT thread_id, checkpoint_ns FROM checkpoints "
                    "  WHERE json_extract(metadata, '$.agent_name') = ?"
                    ")",
                    (AGENT_NAME,),
                ) as cur:
                    row = await cur.fetchone()
                    pair_count = int(row[0]) if row else 0
    except aiosqlite.Error:
        pass
    return size_str, pair_count, size_bytes


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


async def _run_migration_sweep(
    keep: int,
    progress_cb: Callable[[int, int], Awaitable[None]] | None = None,
) -> int:
    """Prune all ``(thread_id, checkpoint_ns)`` pairs to ``keep`` rows each.

    Iterates pairs in deterministic order, applies the same DELETE pattern
    the per-step pruner uses, and yields to the event loop between pairs
    so the agent stays responsive. On success bumps ``PRAGMA user_version``
    so the sweep never reruns.

    ``progress_cb``: optional ``async (done: int, total: int) -> None``
    fired after each pair is committed; used by callers to drive a live
    progress indicator.

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
            if progress_cb is not None:
                try:
                    await progress_cb(pairs_pruned, len(pairs))
                except Exception:  # pragma: no cover - never let UI break sweep
                    pass
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
