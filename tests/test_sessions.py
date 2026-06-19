"""Tests for EvoScientist.sessions — thread CRUD, ID generation, helpers."""

import asyncio
import json
import os
import tempfile
import unittest
import uuid
from datetime import UTC
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from EvoScientist.sessions import (
    AGENT_NAME,
    _format_relative_time,
    _reduce_messages_delta,
    delete_thread,
    find_similar_threads,
    generate_thread_id,
    get_db_path,
    get_most_recent,
    get_thread_messages,
    get_thread_metadata,
    list_threads,
    resolve_thread_id_prefix,
    thread_exists,
)
from tests.conftest import run_async as _run


def _mock_path(db_path: str):
    """Build a Path-like object for patching ``EvoScientist.sessions.get_db_path``.

    Implements the subset of ``pathlib.Path`` that ``sessions.py`` actually
    touches: ``__str__``, ``__fspath__``, ``exists``, ``stat``.
    """
    return type(
        "MockPath",
        (),
        {
            "__str__": lambda s: db_path,
            "__fspath__": lambda s: db_path,
            "exists": lambda s: os.path.exists(db_path),
            "stat": lambda s: os.stat(db_path),
        },
    )()


class TestGenerateThreadId(unittest.TestCase):
    def test_uuid_format(self):
        # Full UUID so langgraph-api can address CLI threads (WebUI interop).
        tid = generate_thread_id()
        assert tid == str(uuid.UUID(tid))

    def test_uniqueness(self):
        ids = {generate_thread_id() for _ in range(100)}
        assert len(ids) == 100


class TestGetDbPath(unittest.TestCase):
    def test_uses_data_dir(self):
        path = get_db_path()
        assert str(path).endswith("sessions.db")
        # On Windows ``get_db_path`` may return the 8.3 short-path
        # form (e.g. ``.../EVOSCI~1/``), hiding the literal
        # ``.evoscientist`` segment. ``resolve()`` walks back through
        # the short-name mapping when possible, restoring the long
        # form for substring matching.
        try:
            long_form = str(path.resolve())
        except OSError:
            long_form = str(path)
        assert ".evoscientist" in long_form or "evoscientist" in long_form.lower()


class TestFormatRelativeTime(unittest.TestCase):
    def test_none(self):
        assert _format_relative_time(None) == ""

    def test_invalid(self):
        assert _format_relative_time("not-a-date") == ""

    def test_recent(self):
        from datetime import datetime

        now = datetime.now(UTC).isoformat()
        result = _format_relative_time(now)
        assert "just now" in result

    def test_minutes(self):
        from datetime import datetime, timedelta

        ts = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
        result = _format_relative_time(ts)
        assert "min ago" in result

    def test_hours(self):
        from datetime import datetime, timedelta

        ts = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        result = _format_relative_time(ts)
        assert "hour" in result

    def test_days(self):
        from datetime import datetime, timedelta

        ts = (datetime.now(UTC) - timedelta(days=3)).isoformat()
        result = _format_relative_time(ts)
        assert "day" in result

    def test_months(self):
        from datetime import datetime, timedelta

        ts = (datetime.now(UTC) - timedelta(days=65)).isoformat()
        result = _format_relative_time(ts)
        assert "month" in result


class TestThreadFunctions(unittest.TestCase):
    """Tests using a real temporary SQLite database."""

    @classmethod
    def setUpClass(cls):
        """Create a temp DB and populate with test data."""
        cls._tmpdir = tempfile.mkdtemp()
        cls._db_path = os.path.join(cls._tmpdir, "test_sessions.db")

        async def _setup():
            import aiosqlite

            async with aiosqlite.connect(cls._db_path) as conn:
                # Create tables matching LangGraph checkpoint schema
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        thread_id TEXT NOT NULL,
                        checkpoint_ns TEXT NOT NULL DEFAULT '',
                        checkpoint_id TEXT NOT NULL,
                        parent_checkpoint_id TEXT,
                        type TEXT,
                        checkpoint BLOB,
                        metadata TEXT NOT NULL DEFAULT '{}',
                        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                    )
                """)
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS writes (
                        thread_id TEXT NOT NULL,
                        checkpoint_ns TEXT NOT NULL DEFAULT '',
                        checkpoint_id TEXT NOT NULL,
                        task_id TEXT NOT NULL,
                        idx INTEGER NOT NULL,
                        channel TEXT NOT NULL,
                        type TEXT,
                        value BLOB,
                        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
                    )
                """)

                # Insert test checkpoints. ``type`` + ``checkpoint`` are
                # populated with a serialized empty-state blob so upstream
                # ``aget_tuple`` (used by message reconstruction) can
                # deserialize them — production checkpoints always have
                # these set; bare-metadata rows are a test fiction.
                serde = JsonPlusSerializer()
                empty_ck_type, empty_ck_blob = serde.dumps_typed({"channel_values": {}})
                for i, tid in enumerate(["abc12345", "abc12399", "def00001"]):
                    meta = json.dumps(
                        {
                            "agent_name": AGENT_NAME,
                            "updated_at": f"2025-01-{15 + i}T10:00:00+00:00",
                            "workspace_dir": f"/tmp/ws_{tid}",
                            "model": "claude-sonnet-4-6",
                        }
                    )
                    await conn.execute(
                        "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, type, checkpoint, metadata) VALUES (?, '', ?, ?, ?, ?)",
                        (tid, f"cp_{i}", empty_ck_type, empty_ck_blob, meta),
                    )

                # Insert a non-EvoScientist checkpoint (should be filtered)
                other_meta = json.dumps(
                    {
                        "agent_name": "OtherAgent",
                        "updated_at": "2025-01-20T10:00:00+00:00",
                    }
                )
                await conn.execute(
                    "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, type, checkpoint, metadata) VALUES (?, '', ?, ?, ?, ?)",
                    ("zzz99999", "cp_other", empty_ck_type, empty_ck_blob, other_meta),
                )
                await conn.commit()

        _run(_setup())

        # Patch get_db_path to point to our temp DB
        cls._patcher = patch(
            "EvoScientist.sessions.get_db_path",
            return_value=type(
                "P",
                (),
                {
                    "__str__": lambda s: cls._db_path,
                    "__fspath__": lambda s: cls._db_path,
                },
            )(),
        )
        cls._patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls._patcher.stop()
        try:
            os.unlink(cls._db_path)
            os.rmdir(cls._tmpdir)
        except OSError:
            pass

    def test_list_threads(self):
        threads = _run(list_threads(limit=10))
        # Should only contain EvoScientist threads
        assert len(threads) == 3
        # Most recent first
        assert threads[0]["thread_id"] == "def00001"

    def test_list_threads_with_message_count(self):
        threads = _run(list_threads(limit=10, include_message_count=True))
        assert "message_count" in threads[0]

    def test_thread_exists_true(self):
        assert _run(thread_exists("abc12345"))

    def test_thread_exists_false(self):
        assert not _run(thread_exists("nonexist"))

    def test_find_similar(self):
        similar = _run(find_similar_threads("abc1"))
        assert len(similar) == 2
        assert "abc12345" in similar
        assert "abc12399" in similar

    def test_find_similar_no_match(self):
        similar = _run(find_similar_threads("xyz"))
        assert len(similar) == 0

    def test_resolve_prefix_exact_match(self):
        resolved, matches = _run(resolve_thread_id_prefix("abc12345"))
        assert resolved == "abc12345"
        assert matches == []

    def test_resolve_prefix_unique_prefix(self):
        resolved, matches = _run(resolve_thread_id_prefix("def00"))
        assert resolved == "def00001"
        assert matches == []

    def test_resolve_prefix_ambiguous(self):
        resolved, matches = _run(resolve_thread_id_prefix("abc1"))
        assert resolved is None
        assert set(matches) == {"abc12345", "abc12399"}

    def test_resolve_prefix_not_found(self):
        resolved, matches = _run(resolve_thread_id_prefix("zzz"))
        assert resolved is None
        assert matches == []

    def test_find_similar_escapes_sql_wildcards(self):
        # '%' / '_' must be treated as literal characters, not SQL LIKE
        # wildcards, so a prefix that doesn't occur verbatim returns nothing
        # (prior buggy behavior: '%' matched every thread).
        assert _run(find_similar_threads("%")) == []
        assert _run(find_similar_threads("_")) == []

    def test_get_most_recent(self):
        recent = _run(get_most_recent())
        assert recent is not None
        assert recent == "def00001"

    def test_get_thread_metadata(self):
        meta = _run(get_thread_metadata("abc12345"))
        assert meta is not None
        assert meta["workspace_dir"] == "/tmp/ws_abc12345"
        assert meta["model"] == "claude-sonnet-4-6"

    def test_get_thread_metadata_missing(self):
        meta = _run(get_thread_metadata("nonexist"))
        assert meta is None

    def test_delete_thread(self):
        # Insert a thread to delete
        async def _insert():
            import aiosqlite

            async with aiosqlite.connect(self._db_path) as conn:
                meta = json.dumps(
                    {
                        "agent_name": AGENT_NAME,
                        "updated_at": "2025-01-01T00:00:00+00:00",
                    }
                )
                await conn.execute(
                    "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, metadata) VALUES (?, '', ?, ?)",
                    ("todelete", "cp_del", meta),
                )
                await conn.commit()

        _run(_insert())

        assert _run(thread_exists("todelete"))
        assert _run(delete_thread("todelete"))
        assert not _run(thread_exists("todelete"))

    def test_delete_nonexistent(self):
        assert not _run(delete_thread("nope1234"))

    def test_get_thread_messages_applies_summarization_event(self):
        async def _insert():
            import aiosqlite

            serde = JsonPlusSerializer()
            messages = [
                HumanMessage(content="first"),
                AIMessage(content="second"),
                HumanMessage(content="third"),
            ]
            summary_message = AIMessage(content="summary")
            checkpoint = {
                "channel_values": {
                    "messages": messages,
                    "_summarization_event": {
                        "cutoff_index": 2,
                        "summary_message": summary_message,
                        "file_path": None,
                    },
                }
            }
            meta = json.dumps(
                {
                    "agent_name": AGENT_NAME,
                    "updated_at": "2025-01-25T10:00:00+00:00",
                }
            )

            async with aiosqlite.connect(self._db_path) as conn:
                await conn.execute(
                    """
                    INSERT INTO checkpoints (
                        thread_id, checkpoint_ns, checkpoint_id, type, checkpoint, metadata
                    ) VALUES (?, '', ?, ?, ?, ?)
                    """,
                    (
                        "sum12345",
                        "cp_sum",
                        *serde.dumps_typed(checkpoint),
                        meta,
                    ),
                )
                await conn.commit()

        async def _cleanup():
            import aiosqlite

            async with aiosqlite.connect(self._db_path) as conn:
                await conn.execute(
                    "DELETE FROM checkpoints WHERE thread_id = ?",
                    ("sum12345",),
                )
                await conn.commit()

        _run(_insert())
        try:
            messages = _run(get_thread_messages("sum12345"))
            assert len(messages) == 2
            assert isinstance(messages[0], AIMessage)
            assert messages[0].content == "summary"
            assert isinstance(messages[1], HumanMessage)
            assert messages[1].content == "third"
        finally:
            _run(_cleanup())

    def test_get_thread_messages_reconstructs_multi_delta_chain(self):
        """3-checkpoint chain with ``_DeltaSnapshot`` seed + pending writes.

        Exercises the upstream ``aget_delta_channel_history`` walk: the
        latest checkpoint has no materialized seed, so the walk must climb
        back through an intermediate delta-only ancestor to a snapshot
        further back, then accumulate writes oldest→newest on top.
        """
        from langgraph.checkpoint.serde.types import _DeltaSnapshot

        async def _insert():
            import aiosqlite

            serde = JsonPlusSerializer()

            seed_messages = [
                HumanMessage(content="m1", id="m1"),
                AIMessage(content="m2", id="m2"),
            ]
            cp1_type, cp1_blob = serde.dumps_typed(
                {"channel_values": {"messages": _DeltaSnapshot(value=seed_messages)}}
            )
            cp_empty_type, cp_empty_blob = serde.dumps_typed({"channel_values": {}})

            w2_type, w2_blob = serde.dumps_typed([HumanMessage(content="m3", id="m3")])
            w3_type, w3_blob = serde.dumps_typed([AIMessage(content="m4", id="m4")])

            meta = json.dumps(
                {
                    "agent_name": AGENT_NAME,
                    "updated_at": "2025-01-26T10:00:00+00:00",
                }
            )

            async with aiosqlite.connect(self._db_path) as conn:
                for cid, parent, ck_type, ck_blob in [
                    ("cp_chain_1", None, cp1_type, cp1_blob),
                    ("cp_chain_2", "cp_chain_1", cp_empty_type, cp_empty_blob),
                    ("cp_chain_3", "cp_chain_2", cp_empty_type, cp_empty_blob),
                ]:
                    await conn.execute(
                        "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata) "
                        "VALUES (?, '', ?, ?, ?, ?, ?)",
                        ("chain12345", cid, parent, ck_type, ck_blob, meta),
                    )
                for cid, wtype, wblob in [
                    ("cp_chain_2", w2_type, w2_blob),
                    ("cp_chain_3", w3_type, w3_blob),
                ]:
                    await conn.execute(
                        "INSERT INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value) "
                        "VALUES (?, '', ?, ?, ?, ?, ?, ?)",
                        ("chain12345", cid, "task0", 0, "messages", wtype, wblob),
                    )
                await conn.commit()

        async def _cleanup():
            import aiosqlite

            async with aiosqlite.connect(self._db_path) as conn:
                await conn.execute(
                    "DELETE FROM checkpoints WHERE thread_id = ?",
                    ("chain12345",),
                )
                await conn.execute(
                    "DELETE FROM writes WHERE thread_id = ?",
                    ("chain12345",),
                )
                await conn.commit()

        async def _assert_walk_branch_active():
            """Confirm cp_chain_3 has no materialized ``messages`` seed.

            Without this guard, a future refactor that ends up writing a
            ``_DeltaSnapshot`` at every checkpoint would silently move
            this test onto the hybrid's "target seed" branch — the
            reconstruction would still match, but the ancestor walk
            being tested here would never run. The assertion locks in
            which branch the test exercises.
            """
            import aiosqlite

            async with aiosqlite.connect(self._db_path) as conn:
                async with conn.execute(
                    "SELECT type, checkpoint FROM checkpoints "
                    "WHERE thread_id = ? AND checkpoint_id = ?",
                    ("chain12345", "cp_chain_3"),
                ) as cur:
                    row = await cur.fetchone()
            assert row is not None
            ck = JsonPlusSerializer().loads_typed((row[0], row[1]))
            assert "messages" not in (ck.get("channel_values") or {}), (
                "cp_chain_3 must NOT carry a messages seed — this test "
                "exercises the ancestor-walk branch, not the target-seed "
                "shortcut."
            )

        _run(_insert())
        try:
            _run(_assert_walk_branch_active())
            messages = _run(get_thread_messages("chain12345"))
            assert [m.content for m in messages] == ["m1", "m2", "m3", "m4"]
            assert isinstance(messages[0], HumanMessage)
            assert isinstance(messages[1], AIMessage)
            assert isinstance(messages[2], HumanMessage)
            assert isinstance(messages[3], AIMessage)
        finally:
            _run(_cleanup())

    def test_get_thread_messages_handles_overwrite_bare_message(self):
        """``Overwrite(value=<bare BaseMessage>)`` wraps to a single-element list.

        The ``Overwrite`` reset branch in ``_load_checkpoint_messages``
        has three sub-cases: list value (most common), ``None`` (clears
        state), and a bare ``BaseMessage`` (rare but valid). The
        last case has no other test coverage — this guards against a
        refactor that silently drops the ``[inner]`` wrapping fallback.
        """
        from langgraph.checkpoint.serde.types import _DeltaSnapshot
        from langgraph.types import Overwrite

        async def _insert():
            import aiosqlite

            serde = JsonPlusSerializer()
            seed_messages = [
                HumanMessage(content="m1", id="m1"),
                AIMessage(content="m2", id="m2"),
            ]
            ck_type, ck_blob = serde.dumps_typed(
                {"channel_values": {"messages": _DeltaSnapshot(value=seed_messages)}}
            )
            # Bare message (NOT wrapped in a list) — the rare third case
            # the Overwrite branch handles.
            ow = Overwrite(value=HumanMessage(content="replaced", id="repl"))
            w_type, w_blob = serde.dumps_typed(ow)
            meta = json.dumps({"agent_name": AGENT_NAME})
            async with aiosqlite.connect(self._db_path) as conn:
                await conn.execute(
                    "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, "
                    "parent_checkpoint_id, type, checkpoint, metadata) "
                    "VALUES (?, '', ?, NULL, ?, ?, ?)",
                    ("ow_bare01", "cp_bare", ck_type, ck_blob, meta),
                )
                await conn.execute(
                    "INSERT INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value) "
                    "VALUES (?, '', ?, 'task0', 0, 'messages', ?, ?)",
                    ("ow_bare01", "cp_bare", w_type, w_blob),
                )
                await conn.commit()

        async def _cleanup():
            import aiosqlite

            async with aiosqlite.connect(self._db_path) as conn:
                await conn.execute(
                    "DELETE FROM checkpoints WHERE thread_id = ?", ("ow_bare01",)
                )
                await conn.execute(
                    "DELETE FROM writes WHERE thread_id = ?", ("ow_bare01",)
                )
                await conn.commit()

        _run(_insert())
        try:
            messages = _run(get_thread_messages("ow_bare01"))
            # Overwrite replaced the seed completely; bare message wrapped
            # in a 1-element list.
            assert len(messages) == 1
            assert isinstance(messages[0], HumanMessage)
            assert messages[0].content == "replaced"
            assert messages[0].id == "repl"
        finally:
            _run(_cleanup())

    def test_get_thread_messages_ignores_colliding_other_agent(self):
        """Multi-agent DB with thread_id collision: must surface only ours.

        Without the agent_name filter on the head-checkpoint lookup,
        ``saver.aget_tuple()`` returns the latest by ``checkpoint_id``
        alone — so if a third-party agent's checkpoint for the same
        ``thread_id`` happens to have a higher id (lexicographically),
        we'd leak its transcript into ``/resume``. Pinning the head to
        the latest EvoScientist-agent row prevents that.
        """
        from langgraph.checkpoint.serde.types import _DeltaSnapshot

        async def _insert():
            import aiosqlite

            serde = JsonPlusSerializer()
            evo_messages = [
                HumanMessage(content="ours_1", id="o1"),
                AIMessage(content="ours_2", id="o2"),
            ]
            other_messages = [
                HumanMessage(content="theirs_1", id="t1"),
                AIMessage(content="theirs_2", id="t2"),
                HumanMessage(content="theirs_3", id="t3"),
            ]
            evo_type, evo_blob = serde.dumps_typed(
                {"channel_values": {"messages": _DeltaSnapshot(value=evo_messages)}}
            )
            other_type, other_blob = serde.dumps_typed(
                {"channel_values": {"messages": _DeltaSnapshot(value=other_messages)}}
            )
            evo_meta = json.dumps({"agent_name": AGENT_NAME})
            other_meta = json.dumps({"agent_name": "ThirdPartyAgent"})

            async with aiosqlite.connect(self._db_path) as conn:
                # EvoScientist's checkpoint id is LEXICOGRAPHICALLY
                # SMALLER than the third-party agent's, so a naive
                # "latest by checkpoint_id" lookup would pick the wrong
                # one.
                await conn.execute(
                    "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, "
                    "parent_checkpoint_id, type, checkpoint, metadata) "
                    "VALUES (?, '', 'aaa_evo', NULL, ?, ?, ?)",
                    ("collide01", evo_type, evo_blob, evo_meta),
                )
                await conn.execute(
                    "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, "
                    "parent_checkpoint_id, type, checkpoint, metadata) "
                    "VALUES (?, '', 'zzz_other', NULL, ?, ?, ?)",
                    ("collide01", other_type, other_blob, other_meta),
                )
                await conn.commit()

        async def _cleanup():
            import aiosqlite

            async with aiosqlite.connect(self._db_path) as conn:
                await conn.execute(
                    "DELETE FROM checkpoints WHERE thread_id = ?", ("collide01",)
                )
                await conn.commit()

        _run(_insert())
        try:
            messages = _run(get_thread_messages("collide01"))
            assert [m.content for m in messages] == ["ours_1", "ours_2"]
            # Defense-in-depth: explicitly forbid leakage of the other
            # agent's content.
            for msg in messages:
                assert not msg.content.startswith("theirs_")
        finally:
            _run(_cleanup())

    # -- Agent isolation: OtherAgent data should never be visible --

    def test_thread_exists_ignores_other_agent(self):
        assert not _run(thread_exists("zzz99999"))

    def test_find_similar_ignores_other_agent(self):
        similar = _run(find_similar_threads("zzz"))
        assert len(similar) == 0

    def test_get_metadata_ignores_other_agent(self):
        meta = _run(get_thread_metadata("zzz99999"))
        assert meta is None

    def test_delete_ignores_other_agent(self):
        # Should not delete OtherAgent's data
        assert not _run(delete_thread("zzz99999"))

    def test_delete_thread_preserves_other_agent_writes(self):
        """Deleting a shared thread_id must only remove writes linked to
        EvoScientist checkpoints, leaving OtherAgent's writes intact."""

        shared_tid = "shared01"

        async def _insert():
            import aiosqlite

            async with aiosqlite.connect(self._db_path) as conn:
                # EvoScientist checkpoint + write
                evo_meta = json.dumps(
                    {
                        "agent_name": AGENT_NAME,
                        "updated_at": "2025-02-01T00:00:00+00:00",
                    }
                )
                await conn.execute(
                    "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, metadata) VALUES (?, '', ?, ?)",
                    (shared_tid, "cp_evo_shared", evo_meta),
                )
                await conn.execute(
                    "INSERT INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value) "
                    "VALUES (?, '', ?, 't1', 0, 'ch', 'str', X'AA')",
                    (shared_tid, "cp_evo_shared"),
                )

                # OtherAgent checkpoint + write on the SAME thread_id
                other_meta = json.dumps(
                    {
                        "agent_name": "OtherAgent",
                        "updated_at": "2025-02-01T00:00:00+00:00",
                    }
                )
                await conn.execute(
                    "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, metadata) VALUES (?, '', ?, ?)",
                    (shared_tid, "cp_other_shared", other_meta),
                )
                await conn.execute(
                    "INSERT INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value) "
                    "VALUES (?, '', ?, 't2', 0, 'ch', 'str', X'BB')",
                    (shared_tid, "cp_other_shared"),
                )
                await conn.commit()

        _run(_insert())

        # Delete — should only affect EvoScientist's data
        _run(delete_thread(shared_tid))

        # Verify OtherAgent's writes survive
        async def _check():
            import aiosqlite

            async with aiosqlite.connect(self._db_path) as conn:
                async with conn.execute(
                    "SELECT checkpoint_id FROM writes WHERE thread_id = ?",
                    (shared_tid,),
                ) as cur:
                    rows = await cur.fetchall()
                return [r[0] for r in rows]

        remaining = _run(_check())
        assert "cp_other_shared" in remaining
        assert "cp_evo_shared" not in remaining


class TestPruningCheckpointer(unittest.TestCase):
    """Integration tests for ``PruningCheckpointer`` against a real
    ``AsyncSqliteSaver`` backed by a temp SQLite file.
    """

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._db_path = os.path.join(self._tmpdir, "prune.db")

    def tearDown(self):
        try:
            os.unlink(self._db_path)
        except OSError:
            pass
        try:
            os.rmdir(self._tmpdir)
        except OSError:
            pass

    def _run_with_wrapper(self, keep: int, body):
        """Open ``PruningCheckpointer`` against the temp DB on a single
        loop, invoke ``body(saver)`` (an async callable), then close
        cleanly.

        Required because ``aiosqlite.Connection`` is bound to the event
        loop it was opened on; reusing it across separate ``run_async``
        calls raises ``ValueError("no active connection")``.
        """
        from EvoScientist.sessions import PruningCheckpointer

        async def _go():
            async with PruningCheckpointer.from_conn_string_with_keep(
                self._db_path, keep_per_ns=keep
            ) as saver:
                await saver.setup()
                return await body(saver)

        return _run(_go())

    @staticmethod
    def _config(thread_id: str, ns: str = "") -> dict:
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": ns,
                "checkpoint_id": None,
            }
        }

    @staticmethod
    def _checkpoint(cid: str, step: int = 0) -> dict:
        # Minimal Checkpoint dict accepted by JsonPlusSerializer.dumps_typed.
        return {
            "v": 1,
            "ts": "2026-01-01T00:00:00+00:00",
            "id": cid,
            "channel_values": {},
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }

    @staticmethod
    def _metadata() -> dict:
        return {"agent_name": AGENT_NAME, "step": 0, "writes": {}, "parents": {}}

    def _row_count(self, thread_id: str, ns: str = "") -> int:
        async def _count():
            import aiosqlite

            async with aiosqlite.connect(self._db_path) as conn:
                async with conn.execute(
                    "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ?",
                    (thread_id, ns),
                ) as cur:
                    row = await cur.fetchone()
                    return int(row[0]) if row else 0

        return _run(_count())

    def test_aput_prunes_after_insert(self):
        tid = "tprune01"

        async def _body(wrapper):
            for i in range(7):
                await wrapper.aput(
                    self._config(tid),
                    self._checkpoint(f"cp_{i:04d}", step=i),
                    self._metadata(),
                    {},
                )

        self._run_with_wrapper(keep=3, body=_body)
        assert self._row_count(tid) == 3

    def test_aput_keeps_latest_for_resume(self):
        """After pruning, ``aget_tuple`` must return the just-written checkpoint."""
        tid = "tresume1"

        async def _body(wrapper):
            last_cfg = None
            for i in range(5):
                last_cfg = await wrapper.aput(
                    self._config(tid),
                    self._checkpoint(f"cpr_{i:04d}", step=i),
                    self._metadata(),
                    {},
                )
            tuple_ = await wrapper.aget_tuple(
                {"configurable": {"thread_id": tid, "checkpoint_ns": ""}}
            )
            return last_cfg, tuple_

        last_cfg, tuple_ = self._run_with_wrapper(keep=2, body=_body)
        assert last_cfg["configurable"]["checkpoint_id"] == "cpr_0004"
        assert tuple_ is not None
        assert tuple_.checkpoint["id"] == "cpr_0004"

    def test_aput_writes_against_kept_checkpoint(self):
        """HITL safety: ``aput_writes`` after prune still attaches successfully."""
        tid = "twrites1"

        async def _body(wrapper):
            last = None
            for i in range(4):
                last = await wrapper.aput(
                    self._config(tid),
                    self._checkpoint(f"cpw_{i:04d}", step=i),
                    self._metadata(),
                    {},
                )
            # Attach a write to the just-written checkpoint id (mimics how
            # pregel stores ``interrupt`` pending writes).
            await wrapper.aput_writes(last, [("__interrupt__", "v")], "task1")
            return last

        last_cfg = self._run_with_wrapper(keep=2, body=_body)

        async def _check():
            import aiosqlite

            async with aiosqlite.connect(self._db_path) as conn:
                async with conn.execute(
                    "SELECT COUNT(*) FROM writes WHERE thread_id = ? AND checkpoint_id = ?",
                    (tid, last_cfg["configurable"]["checkpoint_id"]),
                ) as cur:
                    row = await cur.fetchone()
                    return int(row[0]) if row else 0

        assert _run(_check()) == 1

    def test_aput_partitions_by_ns(self):
        """Two checkpoint namespaces are pruned independently."""
        tid = "tns01"

        async def _body(wrapper):
            for i in range(4):
                await wrapper.aput(
                    self._config(tid, ns=""),
                    self._checkpoint(f"main_{i:04d}", step=i),
                    self._metadata(),
                    {},
                )
                await wrapper.aput(
                    self._config(tid, ns="sub:1"),
                    self._checkpoint(f"sub_{i:04d}", step=i),
                    self._metadata(),
                    {},
                )

        self._run_with_wrapper(keep=2, body=_body)
        assert self._row_count(tid, ns="") == 2
        assert self._row_count(tid, ns="sub:1") == 2

    def test_inherits_base_checkpoint_saver(self):
        """LangGraph's ``compile()`` requires ``isinstance(saver, BaseCheckpointSaver)``.

        Inheriting from ``AsyncSqliteSaver`` (which inherits from
        ``BaseCheckpointSaver``) is what unblocks agent compilation.
        """
        from langgraph.checkpoint.base import BaseCheckpointSaver
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        from EvoScientist.sessions import PruningCheckpointer

        async def _body(saver):
            assert isinstance(saver, BaseCheckpointSaver)
            assert isinstance(saver, AsyncSqliteSaver)
            assert isinstance(saver, PruningCheckpointer)
            # Critical inherited attributes/methods.
            assert saver.serde is not None
            assert saver.lock is not None
            assert saver.conn is not None
            assert callable(saver.aget_tuple)
            assert callable(saver.aput_writes)

        self._run_with_wrapper(keep=2, body=_body)

    def test_prune_failure_does_not_break_aput(self):
        """If pruning raises, ``aput`` still returns successfully."""
        tid = "tfail01"

        async def _body(wrapper):
            async def _boom(*args, **kwargs):
                raise RuntimeError("simulated prune failure")

            wrapper._prune_after_put = _boom
            return await wrapper.aput(
                self._config(tid),
                self._checkpoint("cpf_0001", step=0),
                self._metadata(),
                {},
            )

        result = self._run_with_wrapper(keep=2, body=_body)
        assert result["configurable"]["checkpoint_id"] == "cpf_0001"

    def test_prune_keep_zero_disables(self):
        """``keep_per_ns=0`` is a no-op — all rows survive."""
        tid = "tzero01"

        async def _body(wrapper):
            for i in range(4):
                await wrapper.aput(
                    self._config(tid),
                    self._checkpoint(f"cz_{i:04d}", step=i),
                    self._metadata(),
                    {},
                )

        self._run_with_wrapper(keep=0, body=_body)
        assert self._row_count(tid) == 4

    def test_prune_preserves_other_agent(self):
        """A row with a different ``agent_name`` is never deleted."""
        tid = "tother1"

        async def _body(saver):
            # Seed the OtherAgent row through the same connection so it
            # shares the loop with the saver.
            other_meta = json.dumps({"agent_name": "OtherAgent", "step": 0})
            async with saver.lock:
                await saver.conn.execute(
                    "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, metadata) "
                    "VALUES (?, '', ?, ?)",
                    (tid, "cp_other_keep", other_meta),
                )
                await saver.conn.commit()

            for i in range(5):
                await saver.aput(
                    self._config(tid),
                    self._checkpoint(f"co_{i:04d}", step=i),
                    self._metadata(),
                    {},
                )

        self._run_with_wrapper(keep=2, body=_body)

        # OtherAgent's row + 2 EvoScientist rows = 3 total
        assert self._row_count(tid) == 3

        async def _check_other():
            import aiosqlite

            async with aiosqlite.connect(self._db_path) as conn:
                async with conn.execute(
                    "SELECT 1 FROM checkpoints WHERE thread_id = ? AND checkpoint_id = ?",
                    (tid, "cp_other_keep"),
                ) as cur:
                    return (await cur.fetchone()) is not None

        assert _run(_check_other())

    def test_keep_one_boundary(self):
        """``keep_per_ns=1`` keeps only the latest row, deletes the rest."""
        tid = "tk1_001"

        async def _body(saver):
            for i in range(2):
                await saver.aput(
                    self._config(tid),
                    self._checkpoint(f"k1_{i:04d}", step=i),
                    self._metadata(),
                    {},
                )

        self._run_with_wrapper(keep=1, body=_body)
        assert self._row_count(tid) == 1

        async def _which():
            import aiosqlite

            async with aiosqlite.connect(self._db_path) as conn:
                async with conn.execute(
                    "SELECT checkpoint_id FROM checkpoints WHERE thread_id = ?",
                    (tid,),
                ) as cur:
                    row = await cur.fetchone()
                    return row[0] if row else None

        # The newest write (highest checkpoint_id) is the one kept.
        assert _run(_which()) == "k1_0001"

    def test_concurrent_same_thread_aput_invariant(self):
        """Concurrent ``aput()`` calls cannot squeeze either caller's
        just-written row out of the top-N retention window.

        Directly validates put+prune serialization: gates ``_prune_after_put``
        on the first call so we can launch the second ``aput()`` while
        the first is paused mid-prune. The second call must be blocked
        by ``_aput_lock`` — without that outer lock, the ``self.lock``
        held by ``super().aput`` would not span the prune phase, and
        the two callers would interleave with the buggy result.
        """
        tid = "tcc_001"

        async def _body(saver):
            entered_prune = asyncio.Event()
            release_prune = asyncio.Event()
            orig_prune = saver._prune_after_put

            async def _gated_prune(thread_id: str, checkpoint_ns: str):
                if not entered_prune.is_set():
                    entered_prune.set()
                    await release_prune.wait()
                await orig_prune(thread_id, checkpoint_ns)

            saver._prune_after_put = _gated_prune

            cfg_a = self._config(tid)
            cfg_b = self._config(tid)
            cp_a = self._checkpoint("cc_a", step=0)
            cp_b = self._checkpoint("cc_b", step=1)
            t1 = asyncio.create_task(saver.aput(cfg_a, cp_a, self._metadata(), {}))
            await entered_prune.wait()
            t2 = asyncio.create_task(saver.aput(cfg_b, cp_b, self._metadata(), {}))
            await asyncio.sleep(0)
            assert not t2.done()  # verifies second call is blocked by outer lock
            release_prune.set()
            results = await asyncio.gather(t1, t2)
            return results

        results = self._run_with_wrapper(keep=1, body=_body)
        # Whichever caller landed last is the one survivor; importantly,
        # the row count is exactly 1 (no torn state where both rows
        # disappeared or both survived).
        assert self._row_count(tid) == 1

        async def _winner():
            import aiosqlite

            async with aiosqlite.connect(self._db_path) as conn:
                async with conn.execute(
                    "SELECT checkpoint_id FROM checkpoints WHERE thread_id = ?",
                    (tid,),
                ) as cur:
                    row = await cur.fetchone()
                    return row[0] if row else None

        survivor = _run(_winner())
        # The survivor must be one of the two we wrote, not some torn ID.
        assert survivor in {"cc_a", "cc_b"}
        # And both aput results must report a valid checkpoint_id (neither
        # call raised mid-prune).
        for r in results:
            assert r["configurable"]["checkpoint_id"] in {"cc_a", "cc_b"}

    def test_uuid_ordering_keeps_latest(self):
        """Uses langgraph's actual UUIDv6-shaped checkpoint IDs to confirm
        ``ORDER BY checkpoint_id DESC`` keeps the chronologically latest.

        ``checkpoint_id`` is set by pregel from ``uuid6.uuid6()``, which
        is monotonic-by-time. Lexicographic sort of the canonical hex form
        therefore matches creation order — but the prune SQL relies on
        this, so we exercise it explicitly.
        """
        # langgraph ships its own ``uuid6`` (60-bit timestamp + counter,
        # canonical hex form is monotonic by time). Pregel uses this to
        # mint checkpoint ids — the prune SQL relies on
        # ``ORDER BY checkpoint_id DESC`` matching chronological order.
        from langgraph.checkpoint.base.id import uuid6 as _uuid6

        tid = "tuu_001"

        async def _body(saver):
            ids: list[str] = []
            for i in range(5):
                cid = str(_uuid6(clock_seq=i))
                ids.append(cid)
                cp = {
                    "v": 1,
                    "ts": "2026-01-01T00:00:00+00:00",
                    "id": cid,
                    "channel_values": {},
                    "channel_versions": {},
                    "versions_seen": {},
                    "pending_sends": [],
                }
                await saver.aput(self._config(tid), cp, self._metadata(), {})
            return ids

        ids = self._run_with_wrapper(keep=2, body=_body)
        assert self._row_count(tid) == 2

        async def _check():
            import aiosqlite

            async with aiosqlite.connect(self._db_path) as conn:
                async with conn.execute(
                    "SELECT checkpoint_id FROM checkpoints WHERE thread_id = ? ORDER BY checkpoint_id DESC",
                    (tid,),
                ) as cur:
                    return [r[0] for r in await cur.fetchall()]

        survivors = _run(_check())
        # The two latest UUIDv6 ids — by chronological generation —
        # must be the survivors. Lexicographic DESC ordering must match.
        assert survivors == [ids[4], ids[3]]


class TestPruningCheckpointerDeltaChannel(unittest.TestCase):
    """Tests for DeltaChannel-aware pruning.

    The naive ``keep_latest`` pruner can sever the ``_DeltaSnapshot``
    chain that ``messages`` reconstruction depends on. These tests
    exercise the walk-to-snapshot-ancestor extension that preserves the
    chain head between each kept anchor and the nearest snapshot.

    Checkpoints are inserted directly via SQL with explicit
    ``parent_checkpoint_id`` to give precise control over the chain
    structure (the chained-``aput`` pattern in
    ``TestPruningCheckpointer`` doesn't let us pick which checkpoint
    materializes a snapshot).
    """

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._db_path = os.path.join(self._tmpdir, "delta_prune.db")

    def tearDown(self):
        try:
            os.unlink(self._db_path)
            os.rmdir(self._tmpdir)
        except OSError:
            pass

    @staticmethod
    async def _insert_chain(conn, thread_id, specs):
        """Insert a chain of checkpoints in oldest→newest order.

        ``specs`` is a list of ``(cid, parent_cid, msgs_seed)`` tuples:
        - ``cid``: checkpoint_id
        - ``parent_cid``: parent_checkpoint_id (``None`` for chain root)
        - ``msgs_seed``: value stored at ``channel_values["messages"]``
          — pass ``None`` for delta-only (no seed), a ``list`` for
          plain seed, or a ``_DeltaSnapshot`` for wrapped seed.
        """
        serde = JsonPlusSerializer()
        meta = json.dumps({"agent_name": AGENT_NAME})
        for cid, parent_cid, msgs in specs:
            cv = {"messages": msgs} if msgs is not None else {}
            ck_type, ck_blob = serde.dumps_typed({"channel_values": cv})
            await conn.execute(
                "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, "
                "parent_checkpoint_id, type, checkpoint, metadata) "
                "VALUES (?, '', ?, ?, ?, ?, ?)",
                (thread_id, cid, parent_cid, ck_type, ck_blob, meta),
            )
        await conn.commit()

    @staticmethod
    async def _surviving_ids(conn, thread_id):
        async with conn.execute(
            "SELECT checkpoint_id FROM checkpoints WHERE thread_id = ? "
            "ORDER BY checkpoint_id ASC",
            (thread_id,),
        ) as cur:
            return [r[0] for r in await cur.fetchall()]

    def test_preserves_snapshot_ancestor(self):
        """Snapshot lives outside the anchor window → walk reaches and stops."""
        from langgraph.checkpoint.serde.types import _DeltaSnapshot

        from EvoScientist.sessions import PruningCheckpointer

        tid = "tdsa_001"

        async def _go():
            import aiosqlite

            async with aiosqlite.connect(self._db_path) as conn:
                saver = PruningCheckpointer(conn, keep_per_ns=5)
                await saver.setup()
                # 10 checkpoints; snapshot at cp_003 (outside the anchor
                # window of cp_006..cp_010). Walk from cp_005 → cp_004 →
                # cp_003 (seed found, stop). Survivors: cp_003..cp_010
                # (8). Pruned: cp_001, cp_002.
                specs = [
                    ("cp_001", None, None),
                    ("cp_002", "cp_001", None),
                    ("cp_003", "cp_002", _DeltaSnapshot(value=[])),
                    *[(f"cp_{i:03d}", f"cp_{i - 1:03d}", None) for i in range(4, 11)],
                ]
                await self._insert_chain(conn, tid, specs)
                await saver._prune_after_put(tid, "")
                await conn.commit()
                return await self._surviving_ids(conn, tid)

        survivors = _run(_go())
        assert survivors == [f"cp_{i:03d}" for i in range(3, 11)]

    def test_preserves_full_chain_when_no_snapshot(self):
        """No snapshot anywhere → walk reaches root, preserves everything."""
        from EvoScientist.sessions import PruningCheckpointer

        tid = "tdsa_002"

        async def _go():
            import aiosqlite

            async with aiosqlite.connect(self._db_path) as conn:
                saver = PruningCheckpointer(conn, keep_per_ns=5)
                await saver.setup()
                # 10 delta-only checkpoints, no seed anywhere. Walk
                # exhausts to root (cp_001's parent is None → break).
                # All 10 must survive — the alternative is silent
                # truncation, which is the bug Fix #B prevents.
                specs = [("cp_001", None, None)] + [
                    (f"cp_{i:03d}", f"cp_{i - 1:03d}", None) for i in range(2, 11)
                ]
                await self._insert_chain(conn, tid, specs)
                await saver._prune_after_put(tid, "")
                await conn.commit()
                return await self._surviving_ids(conn, tid)

        survivors = _run(_go())
        assert survivors == [f"cp_{i:03d}" for i in range(1, 11)]

    def test_plain_list_seed_also_terminates_walk(self):
        """Pre-DeltaChannel format (plain list in channel_values) also counts as seed."""
        from EvoScientist.sessions import PruningCheckpointer

        tid = "tdsa_003"

        async def _go():
            import aiosqlite

            async with aiosqlite.connect(self._db_path) as conn:
                saver = PruningCheckpointer(conn, keep_per_ns=3)
                await saver.setup()
                # 6 checkpoints; cp_002 has plain-list seed (legacy
                # format). Walk from cp_003 → cp_002 (seed) → stop.
                # Survivors: cp_002..cp_006 (5). Pruned: cp_001.
                specs = [
                    ("cp_001", None, None),
                    ("cp_002", "cp_001", []),  # plain list seed
                    ("cp_003", "cp_002", None),
                    ("cp_004", "cp_003", None),
                    ("cp_005", "cp_004", None),
                    ("cp_006", "cp_005", None),
                ]
                await self._insert_chain(conn, tid, specs)
                await saver._prune_after_put(tid, "")
                await conn.commit()
                return await self._surviving_ids(conn, tid)

        survivors = _run(_go())
        assert survivors == ["cp_002", "cp_003", "cp_004", "cp_005", "cp_006"]

    def test_chain_break_stops_walk_cleanly(self):
        """Missing ancestor row breaks the chain; walk stops without raising."""
        from EvoScientist.sessions import PruningCheckpointer

        tid = "tdsa_004"

        async def _go():
            import aiosqlite

            async with aiosqlite.connect(self._db_path) as conn:
                saver = PruningCheckpointer(conn, keep_per_ns=2)
                await saver.setup()
                # Insert cp_001..cp_005, then DELETE cp_002 to break
                # the chain. Walk from cp_003 → tries cp_002 →
                # _fetch_checkpoint_blob returns None → break with
                # nothing added (because we add the cursor's id only
                # AFTER fetching its blob succeeds).
                specs = [
                    ("cp_001", None, None),
                    ("cp_002", "cp_001", None),
                    ("cp_003", "cp_002", None),
                    ("cp_004", "cp_003", None),
                    ("cp_005", "cp_004", None),
                ]
                await self._insert_chain(conn, tid, specs)
                await conn.execute(
                    "DELETE FROM checkpoints WHERE thread_id = ? AND checkpoint_id = ?",
                    (tid, "cp_002"),
                )
                await conn.commit()
                await saver._prune_after_put(tid, "")
                await conn.commit()
                return await self._surviving_ids(conn, tid)

        survivors = _run(_go())
        # anchors = cp_004, cp_005. Walk visits cp_003 (preserved),
        # then cp_002 → None → break. cp_001 pruned. cp_002 already
        # absent. Survivors: cp_003, cp_004, cp_005.
        assert survivors == ["cp_003", "cp_004", "cp_005"]

    def test_deserialization_failure_safe_side_over_preserves(self):
        """Corrupt blob mid-walk: pruner preserves what it visited so far."""
        from EvoScientist.sessions import PruningCheckpointer

        tid = "tdsa_005"

        async def _go():
            import aiosqlite

            async with aiosqlite.connect(self._db_path) as conn:
                saver = PruningCheckpointer(conn, keep_per_ns=2)
                await saver.setup()
                # cp_001..cp_005, all delta-only. Then overwrite cp_003
                # with a corrupt blob. Walk from cp_003: fetch blob
                # succeeds (returns garbage bytes), add cp_003 to
                # extra_preserve, deserialize FAILS → break with cp_003
                # already preserved.
                specs = [
                    ("cp_001", None, None),
                    ("cp_002", "cp_001", None),
                    ("cp_003", "cp_002", None),
                    ("cp_004", "cp_003", None),
                    ("cp_005", "cp_004", None),
                ]
                await self._insert_chain(conn, tid, specs)
                await conn.execute(
                    "UPDATE checkpoints SET type = ?, checkpoint = ? "
                    "WHERE thread_id = ? AND checkpoint_id = ?",
                    ("garbage_type", b"not a real blob", tid, "cp_003"),
                )
                await conn.commit()
                await saver._prune_after_put(tid, "")
                await conn.commit()
                return await self._surviving_ids(conn, tid)

        survivors = _run(_go())
        # anchors = cp_004, cp_005. Walk visits cp_003 (added to
        # extra_preserve before deserialize fails). cp_001, cp_002
        # pruned. Survivors: cp_003, cp_004, cp_005.
        assert survivors == ["cp_003", "cp_004", "cp_005"]

    def test_anchor_count_below_keep_is_noop(self):
        """When checkpoint count < keep_per_ns, prune returns early without DELETE."""
        from EvoScientist.sessions import PruningCheckpointer

        tid = "tdsa_006"

        async def _go():
            import aiosqlite

            async with aiosqlite.connect(self._db_path) as conn:
                saver = PruningCheckpointer(conn, keep_per_ns=5)
                await saver.setup()
                # Only 3 checkpoints; keep=5. anchor_ids has 3 items,
                # 3 < 5, prune returns early — all survive untouched.
                specs = [
                    ("cp_001", None, None),
                    ("cp_002", "cp_001", None),
                    ("cp_003", "cp_002", None),
                ]
                await self._insert_chain(conn, tid, specs)
                await saver._prune_after_put(tid, "")
                await conn.commit()
                return await self._surviving_ids(conn, tid)

        survivors = _run(_go())
        assert survivors == ["cp_001", "cp_002", "cp_003"]


class TestMigrationSweep(unittest.TestCase):
    """Tests for the legacy-bloat migration sweep."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._db_path = os.path.join(self._tmpdir, "sweep.db")
        # Patch get_db_path so all sessions.py helpers point at our temp DB.
        self._patcher = patch(
            "EvoScientist.sessions.get_db_path",
            return_value=_mock_path(self._db_path),
        )
        self._patcher.start()
        # Mock atexit.register so sweep-spawned hooks don't leak past the fixture.
        self._atexit_patcher = patch("EvoScientist.sessions.atexit.register")
        self._atexit_patcher.start()
        import EvoScientist.sessions as _sessions_mod

        self._prev_vacuum_scheduled = _sessions_mod._vacuum_scheduled
        _sessions_mod._vacuum_scheduled = False

    def tearDown(self):
        self._patcher.stop()
        self._atexit_patcher.stop()
        import EvoScientist.sessions as _sessions_mod

        _sessions_mod._vacuum_scheduled = self._prev_vacuum_scheduled
        try:
            os.unlink(self._db_path)
        except OSError:
            pass
        try:
            os.rmdir(self._tmpdir)
        except OSError:
            pass

    def _seed(self, threads_x_ns_x_count: list[tuple[str, str, int]]):
        async def _go():
            import aiosqlite

            async with aiosqlite.connect(self._db_path) as conn:
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        thread_id TEXT NOT NULL,
                        checkpoint_ns TEXT NOT NULL DEFAULT '',
                        checkpoint_id TEXT NOT NULL,
                        parent_checkpoint_id TEXT,
                        type TEXT,
                        checkpoint BLOB,
                        metadata TEXT NOT NULL DEFAULT '{}',
                        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                    )
                    """
                )
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS writes (
                        thread_id TEXT NOT NULL,
                        checkpoint_ns TEXT NOT NULL DEFAULT '',
                        checkpoint_id TEXT NOT NULL,
                        task_id TEXT NOT NULL,
                        idx INTEGER NOT NULL,
                        channel TEXT NOT NULL,
                        type TEXT,
                        value BLOB,
                        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
                    )
                    """
                )
                meta = json.dumps({"agent_name": AGENT_NAME, "step": 0})
                for tid, ns, n in threads_x_ns_x_count:
                    for i in range(n):
                        await conn.execute(
                            "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, metadata) "
                            "VALUES (?, ?, ?, ?)",
                            (tid, ns, f"{tid}_{ns}_{i:04d}", meta),
                        )
                await conn.commit()

        _run(_go())

    def _user_version(self) -> int:
        async def _go():
            import aiosqlite

            async with aiosqlite.connect(self._db_path) as conn:
                async with conn.execute("PRAGMA user_version") as cur:
                    row = await cur.fetchone()
                    return int(row[0]) if row else 0

        return _run(_go())

    def _row_count(self, thread_id: str, ns: str) -> int:
        async def _go():
            import aiosqlite

            async with aiosqlite.connect(self._db_path) as conn:
                async with conn.execute(
                    "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ?",
                    (thread_id, ns),
                ) as cur:
                    row = await cur.fetchone()
                    return int(row[0]) if row else 0

        return _run(_go())

    def test_sweep_partitions_threads_and_ns(self):
        from EvoScientist.sessions import _run_migration_sweep

        self._seed(
            [
                ("t1", "", 8),
                ("t1", "sub:1", 6),
                ("t2", "", 4),
            ]
        )

        pairs = _run(_run_migration_sweep(keep=3))
        assert pairs == 3

        assert self._row_count("t1", "") == 3
        assert self._row_count("t1", "sub:1") == 3
        assert self._row_count("t2", "") == 3

    def test_sweep_sets_user_version(self):
        from EvoScientist.sessions import _MIGRATION_VERSION, _run_migration_sweep

        self._seed([("ta", "", 5)])
        assert self._user_version() == 0
        _run(_run_migration_sweep(keep=2))
        assert self._user_version() == _MIGRATION_VERSION

    def test_sweep_skipped_when_marker_set(self):
        from EvoScientist.sessions import (
            _MIGRATION_VERSION,
            _run_migration_sweep,
            _set_user_version,
        )

        self._seed([("tb", "", 5)])

        async def _bump():
            import aiosqlite

            async with aiosqlite.connect(self._db_path) as conn:
                await _set_user_version(conn, _MIGRATION_VERSION)

        _run(_bump())
        # Already at marker → sweep is a no-op even though many rows exist.
        pairs = _run(_run_migration_sweep(keep=2))
        assert pairs == 0
        assert self._row_count("tb", "") == 5

    def test_needs_migration_below_threshold(self):
        from EvoScientist.sessions import _needs_migration

        # Empty DB (file doesn't exist yet) → False
        assert not _run(_needs_migration())
        # Tiny DB → False
        self._seed([("tc", "", 1)])
        assert not _run(_needs_migration())

    def test_needs_migration_above_threshold(self):
        """Use monkeypatch on the threshold constant so tests stay fast."""
        from EvoScientist import sessions as sessions_module

        self._seed([("td", "", 3)])
        with patch.object(sessions_module, "_MIGRATION_THRESHOLD_BYTES", 1):
            # Tiny DB exceeds the 1-byte threshold → marker check kicks in.
            assert _run(sessions_module._needs_migration())

    def test_keep_zero_short_circuits_sweep(self):
        from EvoScientist.sessions import _run_migration_sweep

        self._seed([("te", "", 4)])
        pairs = _run(_run_migration_sweep(keep=0))
        assert pairs == 0
        assert self._row_count("te", "") == 4

    def test_sweep_handles_missing_writes_table(self):
        """Legacy DB with only ``checkpoints`` (no ``writes``) must still prune.

        Regression test: the sweep used to unconditionally
        ``DELETE FROM writes`` and would abort on the first iteration
        with ``no such table: writes``, leaving the bloat in place.
        """
        from EvoScientist.sessions import _run_migration_sweep

        # Seed creates both tables; drop ``writes`` to simulate legacy.
        self._seed([("tw", "", 5)])

        async def _drop_writes():
            import aiosqlite

            async with aiosqlite.connect(self._db_path) as conn:
                await conn.execute("DROP TABLE writes")
                await conn.commit()

        _run(_drop_writes())

        pairs = _run(_run_migration_sweep(keep=2))
        assert pairs == 1
        assert self._row_count("tw", "") == 2

    def test_get_checkpointer_blocks_on_sweep_then_idempotent(self):
        """End-to-end: ``get_checkpointer()`` must run the sweep BEFORE
        yielding the saver so a concurrent ``aput()`` can't race the
        DELETEs. After the first call sets ``user_version=1``, subsequent
        calls must skip the sweep entirely.
        """
        from EvoScientist import sessions as sessions_module
        from EvoScientist.sessions import (
            _MIGRATION_VERSION,
            get_checkpointer,
        )

        self._seed([("ge", "", 6)])

        # Force the sweep to be needed regardless of file size.
        with patch.object(sessions_module, "_MIGRATION_THRESHOLD_BYTES", 1):
            # First entry: sweep should run, prune to keep=10 (default), and
            # set user_version. With only 6 rows in one (thread, ns) pair,
            # the prune is a no-op but user_version is still bumped.
            async def _first():
                async with get_checkpointer() as saver:
                    return saver is not None

            assert _run(_first())
            assert self._user_version() == _MIGRATION_VERSION

            # Second entry: sweep must be skipped — patch _run_migration_sweep
            # to raise so any accidental re-invocation fails the test loudly.
            async def _exploding_sweep(*_args, **_kwargs):
                raise AssertionError("sweep must not re-run after marker is set")

            with patch.object(
                sessions_module, "_run_migration_sweep", _exploding_sweep
            ):

                async def _second():
                    async with get_checkpointer() as saver:
                        return saver is not None

                assert _run(_second())

    def test_sweep_preserves_snapshot_ancestor(self):
        """Migration sweep must apply the same DeltaChannel walk as steady-state.

        Without this, legacy users upgrading to PR #231 would hit a
        one-shot silent truncation: the bloat sweep would naive-prune
        the ``_DeltaSnapshot`` seed out of long threads, then the
        ``user_version`` marker locks the sweep so it never re-runs —
        leaving permanently empty ``/resume`` history.

        Mirrors ``TestPruningCheckpointerDeltaChannel.test_preserves_
        snapshot_ancestor`` but drives via ``_run_migration_sweep``.
        """
        from langgraph.checkpoint.serde.types import _DeltaSnapshot

        from EvoScientist.sessions import _run_migration_sweep

        tid = "tsweep_delta"

        async def _seed():
            import aiosqlite

            serde = JsonPlusSerializer()
            snapshot_type, snapshot_blob = serde.dumps_typed(
                {"channel_values": {"messages": _DeltaSnapshot(value=[])}}
            )
            empty_type, empty_blob = serde.dumps_typed({"channel_values": {}})
            meta = json.dumps({"agent_name": AGENT_NAME})

            async with aiosqlite.connect(self._db_path) as conn:
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        thread_id TEXT NOT NULL,
                        checkpoint_ns TEXT NOT NULL DEFAULT '',
                        checkpoint_id TEXT NOT NULL,
                        parent_checkpoint_id TEXT,
                        type TEXT,
                        checkpoint BLOB,
                        metadata TEXT NOT NULL DEFAULT '{}',
                        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                    )
                    """
                )
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS writes (
                        thread_id TEXT NOT NULL,
                        checkpoint_ns TEXT NOT NULL DEFAULT '',
                        checkpoint_id TEXT NOT NULL,
                        task_id TEXT NOT NULL,
                        idx INTEGER NOT NULL,
                        channel TEXT NOT NULL,
                        type TEXT,
                        value BLOB,
                        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
                    )
                    """
                )
                # cp_001..cp_002 delta-only, cp_003 carries the snapshot,
                # cp_004..cp_010 delta-only. Anchor window with keep=5 is
                # cp_006..cp_010; walk from cp_005 backward hits cp_003
                # (seed) → stop. Survivors: cp_003..cp_010.
                for i in range(1, 11):
                    cid = f"cp_{i:03d}"
                    parent = f"cp_{i - 1:03d}" if i > 1 else None
                    if i == 3:
                        ct, cb = snapshot_type, snapshot_blob
                    else:
                        ct, cb = empty_type, empty_blob
                    await conn.execute(
                        "INSERT INTO checkpoints (thread_id, checkpoint_ns, "
                        "checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata) "
                        "VALUES (?, '', ?, ?, ?, ?, ?)",
                        (tid, cid, parent, ct, cb, meta),
                    )
                await conn.commit()

        _run(_seed())
        pairs = _run(_run_migration_sweep(keep=5))
        assert pairs == 1

        async def _survivors():
            import aiosqlite

            async with aiosqlite.connect(self._db_path) as conn:
                async with conn.execute(
                    "SELECT checkpoint_id FROM checkpoints WHERE thread_id = ? "
                    "ORDER BY checkpoint_id ASC",
                    (tid,),
                ) as cur:
                    return [r[0] for r in await cur.fetchall()]

        survivors = _run(_survivors())
        # cp_001, cp_002 pruned. cp_003 (snapshot) + walk-through (cp_004,
        # cp_005) + anchors (cp_006..cp_010) survive.
        assert survivors == [f"cp_{i:03d}" for i in range(3, 11)]
        # Explicit absence of the pruned ids — guards against a future
        # refactor that accidentally returns an empty survivors list.
        assert "cp_001" not in survivors
        assert "cp_002" not in survivors


class TestDbStats(unittest.TestCase):
    """Tests for the read-only ``db_stats`` diagnostic helper."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._db_path = os.path.join(self._tmpdir, "stats.db")
        self._patcher = patch(
            "EvoScientist.sessions.get_db_path",
            return_value=_mock_path(self._db_path),
        )
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        try:
            os.unlink(self._db_path)
        except OSError:
            pass
        try:
            os.rmdir(self._tmpdir)
        except OSError:
            pass

    def _seed(self):
        async def _go():
            import aiosqlite

            async with aiosqlite.connect(self._db_path) as conn:
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        thread_id TEXT NOT NULL,
                        checkpoint_ns TEXT NOT NULL DEFAULT '',
                        checkpoint_id TEXT NOT NULL,
                        parent_checkpoint_id TEXT,
                        type TEXT,
                        checkpoint BLOB,
                        metadata TEXT NOT NULL DEFAULT '{}',
                        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                    )
                    """
                )
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS writes (
                        thread_id TEXT NOT NULL,
                        checkpoint_ns TEXT NOT NULL DEFAULT '',
                        checkpoint_id TEXT NOT NULL,
                        task_id TEXT NOT NULL,
                        idx INTEGER NOT NULL,
                        channel TEXT NOT NULL,
                        type TEXT,
                        value BLOB,
                        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
                    )
                    """
                )
                evo = json.dumps({"agent_name": AGENT_NAME, "step": 0})
                other = json.dumps({"agent_name": "OtherAgent", "step": 0})
                # 2 EvoScientist threads, 5 + 3 = 8 checkpoints
                for i in range(5):
                    await conn.execute(
                        "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, metadata) "
                        "VALUES (?, '', ?, ?)",
                        ("evo01", f"ce01_{i}", evo),
                    )
                for i in range(3):
                    await conn.execute(
                        "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, metadata) "
                        "VALUES (?, '', ?, ?)",
                        ("evo02", f"ce02_{i}", evo),
                    )
                # 1 OtherAgent thread (excluded from EvoSci counts)
                await conn.execute(
                    "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, metadata) "
                    "VALUES (?, '', ?, ?)",
                    ("oth01", "co01_0", other),
                )
                # 4 writes linked to an EvoScientist checkpoint
                # (counted by db_stats via the JOIN to checkpoints).
                for i in range(4):
                    await conn.execute(
                        "INSERT INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value) "
                        "VALUES ('evo01', '', 'ce01_0', 't1', ?, 'ch', 'str', X'AA')",
                        (i,),
                    )
                # 2 writes linked to OtherAgent's checkpoint — must NOT
                # be counted in ``write_count`` (db_stats joins to
                # checkpoints and filters by agent_name).
                for i in range(2):
                    await conn.execute(
                        "INSERT INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value) "
                        "VALUES ('oth01', '', 'co01_0', 't2', ?, 'ch', 'str', X'BB')",
                        (i,),
                    )
                await conn.commit()

        _run(_go())

    def test_stats_returns_evo_only_counts(self):
        """All counts (incl. ``write_count``) must scope to EvoScientist rows.

        Regression for the previous bare ``COUNT(*) FROM writes`` which
        over-reported when other LangGraph apps share the DB. The seed
        fixture inserts 4 EvoSci writes and 2 OtherAgent writes; only the
        4 should count.
        """
        from EvoScientist.sessions import db_stats

        self._seed()
        stats = _run(db_stats())
        assert stats["thread_count"] == 2
        assert stats["checkpoint_count"] == 8  # OtherAgent's 1 row excluded
        assert stats["write_count"] == 4  # 2 OtherAgent writes excluded
        assert stats["size_bytes"] > 0
        assert stats["db_path"].endswith("stats.db")

    def test_stats_top_threads_ordered_desc(self):
        from EvoScientist.sessions import db_stats

        self._seed()
        stats = _run(db_stats(top_n=5))
        ids = [row["thread_id"] for row in stats["top_threads"]]
        counts = [row["count"] for row in stats["top_threads"]]
        # Sorted desc by count: evo01 (5) before evo02 (3); OtherAgent excluded
        assert ids == ["evo01", "evo02"]
        assert counts == [5, 3]

    def test_stats_missing_db(self):
        """No DB on disk → returns zeroed stats, never raises."""
        from EvoScientist.sessions import db_stats

        # Don't seed — file doesn't exist.
        stats = _run(db_stats())
        assert stats["thread_count"] == 0
        assert stats["checkpoint_count"] == 0
        assert stats["write_count"] == 0
        assert stats["size_bytes"] == 0


class TestReduceMessagesDeltaNoneState(unittest.TestCase):
    """Direct unit tests for the inline ``_reduce_messages_delta`` reducer.

    Regression for the None-state crash: ``DeltaChannel.replay_writes``
    can hand the reducer ``state=None`` for threads whose earliest
    checkpoint never seeded ``messages: []``. Before the fix, the slow
    path passed ``None`` straight into ``convert_to_messages`` and raised;
    now ``state or []`` is substituted.
    """

    def test_none_state_simple_append(self):
        result = _reduce_messages_delta(None, [[HumanMessage(content="hi", id="1")]])
        assert len(result) == 1
        assert result[0].content == "hi"
        assert result[0].id == "1"

    def test_none_state_empty_writes(self):
        # No state and nothing to append → empty list, no crash.
        assert _reduce_messages_delta(None, []) == []

    def test_empty_state_still_appends(self):
        # Regression guard: an explicit empty-list state must behave the
        # same as None — append the single write.
        result = _reduce_messages_delta([], [[AIMessage(content="yo", id="2")]])
        assert len(result) == 1
        assert result[0].content == "yo"
        assert result[0].id == "2"


def _signature(messages):
    """Comparable shape for reducer-output equality assertions."""
    return [(type(m).__name__, m.id, m.content) for m in messages]


def _import_upstream_reducer():
    """Import deepagents' private delta reducer, or fail loudly.

    A ``pytest.fail`` (not ``skip``) is deliberate: this test is the
    tripwire that fires when the upstream private symbol is renamed or
    relocated. A silent skip would let semantic drift between EvoSci's
    inline copy (``sessions.py``) and upstream go unnoticed.
    """
    try:
        from deepagents._messages_reducer import (
            _messages_delta_reducer as upstream,
        )
    except ImportError as exc:  # pragma: no cover - tripwire path
        pytest.fail(
            "deepagents._messages_reducer._messages_delta_reducer could not "
            f"be imported ({exc}). The upstream private reducer that EvoSci's "
            "inline copy in sessions.py (_reduce_messages_delta) mirrors has "
            "moved or been renamed. Re-locate the upstream symbol and "
            "re-evaluate the inline copy for semantic drift before adjusting "
            "this test."
        )
    return upstream


class TestReduceMessagesDeltaUpstreamParity:
    """Behavioral parity vs deepagents' private ``_messages_delta_reducer``.

    Drift detector: if upstream changes the reducer's semantics (dedup,
    tombstone, reset, coercion) the EvoSci inline copy must be updated to
    match. These cases pass equivalent batched writes to both functions
    and assert identical output. (EvoSci's signature is ``writes: list[Any]``
    and upstream's is ``list[list[AnyMessage]]``, but both flatten lists
    vs single items the same way, so batched-list writes are equivalent.)
    """

    @pytest.fixture(scope="class")
    def upstream(self):
        return _import_upstream_reducer()

    def _assert_parity(self, upstream, state, writes):
        evo_out = _reduce_messages_delta(state, writes)
        up_out = upstream(state, writes)
        assert _signature(evo_out) == _signature(up_out)
        return evo_out

    def test_parity_none_state_append(self, upstream):
        out = self._assert_parity(
            upstream, None, [[HumanMessage(content="hi", id="a1")]]
        )
        assert _signature(out) == [("HumanMessage", "a1", "hi")]

    def test_parity_dedup_by_id(self, upstream):
        state = [HumanMessage(content="orig", id="1")]
        writes = [[HumanMessage(content="updated", id="1")]]
        out = self._assert_parity(upstream, state, writes)
        # In-place update, no duplicate appended.
        assert _signature(out) == [("HumanMessage", "1", "updated")]

    def test_parity_remove_message_tombstone(self, upstream):
        state = [
            HumanMessage(content="keep", id="1"),
            AIMessage(content="drop", id="2"),
        ]
        writes = [[RemoveMessage(id="2")]]
        out = self._assert_parity(upstream, state, writes)
        assert _signature(out) == [("HumanMessage", "1", "keep")]

    def test_parity_remove_all_then_append(self, upstream):
        state = [
            HumanMessage(content="old1", id="1"),
            AIMessage(content="old2", id="2"),
        ]
        writes = [
            [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                HumanMessage(content="fresh", id="3"),
            ]
        ]
        out = self._assert_parity(upstream, state, writes)
        # Sentinel wipes prior state + earlier writes; only "fresh" remains.
        assert _signature(out) == [("HumanMessage", "3", "fresh")]

    def test_parity_dict_shorthand_coercion(self, upstream):
        # Raw dict shorthand must coerce to a typed BaseMessage identically
        # in both reducers.
        writes = [[{"role": "user", "content": "x", "id": "d1"}]]
        out = self._assert_parity(upstream, None, writes)
        assert _signature(out) == [("HumanMessage", "d1", "x")]


class TestCreateCheckpointerForLanggraphApi(unittest.TestCase):
    """Tests for ``create_checkpointer_for_langgraph_api`` — the WebUI/deploy
    SQLite checkpointer factory that replaces the default ``InMemorySaver``."""

    def test_yields_pruning_checkpointer(self):
        """Factory yields a ``PruningCheckpointer`` instance."""
        from EvoScientist.sessions import (
            PruningCheckpointer,
            create_checkpointer_for_langgraph_api,
        )

        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "sessions.db")
            with patch(
                "EvoScientist.sessions.get_db_path",
                return_value=_mock_path(db),
            ):

                async def _run_inner():
                    async with create_checkpointer_for_langgraph_api() as cp:
                        assert isinstance(cp, PruningCheckpointer)

                _run(_run_inner())

    def test_checkpointer_is_set_up(self):
        """Factory calls ``setup()`` so tables exist before yielding."""
        import aiosqlite

        from EvoScientist.sessions import create_checkpointer_for_langgraph_api

        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "sessions.db")
            with patch(
                "EvoScientist.sessions.get_db_path",
                return_value=_mock_path(db),
            ):

                async def _run_inner():
                    async with create_checkpointer_for_langgraph_api():
                        async with aiosqlite.connect(db) as conn:
                            async with conn.execute(
                                "SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints'"
                            ) as cur:
                                row = await cur.fetchone()
                            assert row is not None, (
                                "checkpoints table must exist after setup()"
                            )

                _run(_run_inner())

    def test_checkpointer_persists_across_contexts(self):
        """Data written in one context manager is readable in a new one.

        This is the core regression test: verifies that session data
        survives process restarts (simulated as two separate ``async with``
        blocks sharing the same DB file).
        """
        from EvoScientist.sessions import create_checkpointer_for_langgraph_api

        thread_id = "testthread1"

        async def _run_inner():
            with tempfile.TemporaryDirectory() as td:
                db = os.path.join(td, "sessions.db")

                def _patch():
                    return patch(
                        "EvoScientist.sessions.get_db_path",
                        return_value=_mock_path(db),
                    )

                # --- First "process": write a checkpoint ---
                with _patch():
                    async with create_checkpointer_for_langgraph_api() as cp:
                        config = {
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_ns": "",
                            }
                        }
                        checkpoint = {
                            "v": 1,
                            "id": "ckpt-001",
                            "ts": "2024-01-01T00:00:00+00:00",
                            "channel_values": {"messages": []},
                            "channel_versions": {},
                            "versions_seen": {},
                            "pending_sends": [],
                        }
                        metadata = {
                            "source": "input",
                            "step": 0,
                            "writes": {},
                            "parents": {},
                            "agent_name": "EvoScientist",
                        }
                        await cp.aput(config, checkpoint, metadata, {})

                # --- Second "process": read back the checkpoint ---
                with _patch():
                    async with create_checkpointer_for_langgraph_api() as cp2:
                        result = await cp2.aget_tuple(
                            {
                                "configurable": {
                                    "thread_id": thread_id,
                                    "checkpoint_ns": "",
                                }
                            }
                        )
                        assert result is not None, (
                            "Checkpoint written in first context must be readable in second context "
                            "(simulates data survival across process restarts)"
                        )
                        assert result.config["configurable"]["thread_id"] == thread_id

        _run(_run_inner())

    def test_capability_surface_matches_langgraph_api_probe(self):
        """Document the REAL capability surface langgraph-api will detect.

        The adapter's ``_is_overridden`` probe compares each method against
        ``BaseCheckpointSaver``: inherited raising stubs do NOT count as
        capability. ``callable()`` checks would pass on the stubs and give
        false confidence (the original PR test did exactly that).
        """
        from langgraph.checkpoint.base import BaseCheckpointSaver

        from EvoScientist.sessions import _ApiPruningCheckpointer

        def overridden(name: str) -> bool:
            base = getattr(BaseCheckpointSaver, name, None)
            sub = getattr(_ApiPruningCheckpointer, name, None)
            return base is not None and sub is not None and sub is not base

        # Real implementations the adapter will detect and use.
        for method in ("adelete_thread", "aget_tuple", "aput", "aput_writes"):
            assert overridden(method), f"'{method}' must be a real implementation"
        # Known degradations: still BaseCheckpointSaver raising stubs.
        # rollback cleanup raises at runtime; thread copy uses the adapter's
        # slow generic fallback. If these start passing, the docstring in
        # create_checkpointer_for_langgraph_api should be updated.
        for method in ("aprune", "adelete_for_runs", "acopy_thread"):
            assert not overridden(method), (
                f"'{method}' is now overridden — update the capability "
                "docstring in create_checkpointer_for_langgraph_api"
            )

    def test_aput_stamps_workspace_metadata_for_graph_rows(self):
        """Graph rows get workspace metadata; only main rows get agent_name."""
        import json

        import aiosqlite

        from EvoScientist.sessions import (
            AGENT_NAME,
            create_checkpointer_for_langgraph_api,
        )

        def _checkpoint(cid: str) -> dict:
            return {
                "v": 1,
                "id": cid,
                "ts": "2024-01-01T00:00:00+00:00",
                "channel_values": {"messages": []},
                "channel_versions": {},
                "versions_seen": {},
                "pending_sends": [],
            }

        async def _run_inner(db: str):
            async with create_checkpointer_for_langgraph_api() as cp:
                # Simulates a WebUI main-graph run (langgraph-api metadata).
                await cp.aput(
                    {
                        "configurable": {
                            "thread_id": "11111111-1111-1111-1111-111111111111",
                            "checkpoint_ns": "",
                        }
                    },
                    _checkpoint("ckpt-main"),
                    {"source": "loop", "step": 1, "graph_id": AGENT_NAME},
                    {},
                )
                # Simulates a memory-worker run: must stay unstamped.
                await cp.aput(
                    {
                        "configurable": {
                            "thread_id": "22222222-2222-2222-2222-222222222222",
                            "checkpoint_ns": "",
                        }
                    },
                    _checkpoint("ckpt-worker"),
                    {"source": "loop", "step": 1, "graph_id": "evomemory-turn-worker"},
                    {},
                )
            async with aiosqlite.connect(db) as conn:
                rows = {}
                async with conn.execute(
                    "SELECT thread_id, metadata FROM checkpoints"
                ) as cur:
                    async for tid, meta in cur:
                        rows[tid] = json.loads(meta)
            main = rows["11111111-1111-1111-1111-111111111111"]
            worker = rows["22222222-2222-2222-2222-222222222222"]
            assert main.get("agent_name") == AGENT_NAME
            assert main.get("workspace_dir") == "/tmp/test-workspace"
            assert main.get("updated_at"), "updated_at drives /threads ordering"
            assert "agent_name" not in worker
            assert worker.get("workspace_dir") == "/tmp/test-workspace"
            assert worker.get("updated_at")

        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "sessions.db")
            with (
                patch(
                    "EvoScientist.sessions.get_db_path",
                    return_value=_mock_path(db),
                ),
                patch.dict(
                    os.environ,
                    {"EVOSCIENTIST_WORKSPACE_DIR": "/tmp/test-workspace"},
                ),
                patch(
                    "EvoScientist.sessions._api_workspace_dir",
                    return_value="/tmp/test-workspace",
                ),
            ):
                _run(_run_inner(db))


class TestRestoreWebuiThreadsToGlobalStore(unittest.TestCase):
    """Tests for ``_restore_webui_threads_to_global_store``.

    Verifies that UUID-format threads written to SQLite by ``langgraph dev``
    runs are re-populated into ``GlobalStore["threads"]`` on server restart,
    so the WebUI sidebar is not empty after a package upgrade or clean restart.
    """

    _WS = "/tmp/restore-test-workspace"

    def _make_db_with_threads(
        self,
        db_path: str,
        thread_ids: list[str],
        assistant_id: str | None = "aaaa-bbbb",
        graph_id: str | None = "EvoScientist",
        workspace_dir: str | None = _WS,
        model: str | None = "test-model",
        agent_name: str | None = "EvoScientist",
        ckpt_prefix: str = "ckpt",
    ) -> None:
        """Insert minimal checkpoint rows for the given thread_ids into a fresh DB."""
        import json
        import sqlite3

        con = sqlite3.connect(db_path)
        con.execute(
            "CREATE TABLE IF NOT EXISTS checkpoints "
            "(thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT PRIMARY KEY, "
            " parent_checkpoint_id TEXT, type TEXT, checkpoint BLOB, metadata TEXT)"
        )
        for tid in thread_ids:
            meta_dict: dict = {"updated_at": "2025-01-01T00:00:00+00:00"}
            if agent_name is not None:
                meta_dict["agent_name"] = agent_name
            if assistant_id is not None:
                meta_dict["assistant_id"] = assistant_id
            if graph_id is not None:
                meta_dict["graph_id"] = graph_id
            if workspace_dir is not None:
                meta_dict["workspace_dir"] = workspace_dir
            if model is not None:
                meta_dict["model"] = model
            meta = json.dumps(meta_dict)
            con.execute(
                "INSERT INTO checkpoints VALUES (?,?,?,?,?,?,?)",
                (tid, "", f"{ckpt_prefix}-{tid[:8]}", None, "empty", b"", meta),
            )
        con.commit()
        con.close()

    def _patch_workspace(self):
        from unittest.mock import patch

        return patch("EvoScientist.sessions._api_workspace_dir", return_value=self._WS)

    def test_restores_uuid_threads_into_global_store(self):
        """UUID-format thread IDs from SQLite are injected into GlobalStore."""
        import sys
        from unittest.mock import MagicMock, patch

        from EvoScientist.sessions import _restore_webui_threads_to_global_store

        uuid_id = "12345678-1234-1234-1234-123456789abc"
        asst_uuid_id = "a2b49500-c49b-5560-b664-d42ee8b66d3c"
        short_id = "abcd1234"  # CLI-style, must be excluded

        mock_store: dict = {"threads": []}
        mock_global_store = MagicMock()
        mock_global_store.get.side_effect = mock_store.get
        mock_global_store.__getitem__ = lambda self, k: mock_store[k]
        mock_global_store.__setitem__ = lambda self, k, v: mock_store.__setitem__(k, v)

        fake_module = MagicMock()
        fake_module.GLOBAL_STORE = mock_global_store

        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "sessions.db")
            self._make_db_with_threads(
                db, [uuid_id, short_id], assistant_id=asst_uuid_id
            )
            with (
                patch(
                    "EvoScientist.sessions.get_db_path",
                    return_value=_mock_path(db),
                ),
                patch.dict(
                    sys.modules, {"langgraph_runtime_inmem.database": fake_module}
                ),
                self._patch_workspace(),
            ):
                _run(_restore_webui_threads_to_global_store())

        # Only the UUID thread should have been added; the short-hex CLI thread
        # should not appear because it doesn't match the UUID LIKE pattern.
        added = mock_store["threads"]
        assert len(added) == 1, f"Expected 1 restored thread, got {len(added)}: {added}"
        import uuid as _uuid_mod

        # thread_id MUST be stored as a uuid.UUID object, not a plain string.
        # langgraph_runtime_inmem._get_with_filters compares with == against
        # _ensure_uuid(thread_id), which returns a UUID object.  A string never
        # equals a UUID object, causing every Threads.get() call to 404.
        assert isinstance(added[0]["thread_id"], _uuid_mod.UUID), (
            f"thread_id must be uuid.UUID, got {type(added[0]['thread_id'])}"
        )
        assert added[0]["thread_id"] == _uuid_mod.UUID(uuid_id)
        assert added[0]["status"] == "idle"
        # metadata.assistant_id must stay a STRING: the runtime stores
        # str(assistant_id) (ops.py Threads.create) and search filters compare
        # with raw == against JSON strings — a uuid.UUID here would silently
        # exclude restored threads from assistant_id-filtered searches.
        assert added[0]["metadata"].get("assistant_id") == asst_uuid_id
        assert isinstance(added[0]["metadata"].get("assistant_id"), str)
        assert added[0]["metadata"].get("graph_id") == "EvoScientist"
        assert added[0]["metadata"].get("workspace_dir") == self._WS
        assert added[0]["metadata"].get("model") == "test-model"
        # created_at / updated_at must be datetime objects, not ISO strings.
        # Threads.search() sorts by these fields using sorted(); mixing
        # datetime and str raises TypeError: '<' not supported.
        from datetime import datetime as _dt

        assert isinstance(added[0]["created_at"], _dt), (
            f"created_at must be datetime, got {type(added[0]['created_at'])}"
        )
        assert isinstance(added[0]["updated_at"], _dt)

    def test_fixes_existing_string_thread_ids_in_place(self):
        """Threads already in GlobalStore with string thread_id get fixed in-place.

        When .pckl loads successfully, threads are already in the store but
        thread_id is a plain string (as pickled).  The restore must:
          1. Convert thread_id to uuid.UUID so Threads.get() comparison works.
          2. Backfill missing metadata.assistant_id / graph_id from SQLite.
          3. Not create duplicate entries.
        """
        import sys
        import uuid as _uuid_mod
        from unittest.mock import MagicMock, patch

        from EvoScientist.sessions import _restore_webui_threads_to_global_store

        uuid_id = "aaaabbbb-aaaa-bbbb-cccc-ddddeeeeffff"
        asst_uuid_id = "a2b49500-c49b-5560-b664-d42ee8b66d3c"

        # Simulate .pckl-restored thread: thread_id is a string, metadata empty.
        existing_stub: dict = {"thread_id": uuid_id, "status": "idle", "metadata": {}}
        mock_store: dict = {"threads": [existing_stub]}
        mock_global_store = MagicMock()
        mock_global_store.get.side_effect = mock_store.get
        mock_global_store.__getitem__ = lambda self, k: mock_store[k]
        mock_global_store.__setitem__ = lambda self, k, v: mock_store.__setitem__(k, v)

        fake_module = MagicMock()
        fake_module.GLOBAL_STORE = mock_global_store

        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "sessions.db")
            self._make_db_with_threads(db, [uuid_id], assistant_id=asst_uuid_id)
            with (
                patch(
                    "EvoScientist.sessions.get_db_path",
                    return_value=_mock_path(db),
                ),
                patch.dict(
                    sys.modules, {"langgraph_runtime_inmem.database": fake_module}
                ),
                self._patch_workspace(),
            ):
                _run(_restore_webui_threads_to_global_store())

        # No duplicate: still exactly one entry.
        assert len(mock_store["threads"]) == 1, (
            f"Expected 1 thread, got {len(mock_store['threads'])}"
        )
        t = mock_store["threads"][0]
        # thread_id must now be a UUID object, not a string.
        assert isinstance(t["thread_id"], _uuid_mod.UUID), (
            f"thread_id must be uuid.UUID after fix, got {type(t['thread_id'])}"
        )
        assert t["thread_id"] == _uuid_mod.UUID(uuid_id)
        # metadata must be backfilled — assistant_id as str (runtime convention).
        assert t["metadata"].get("assistant_id") == asst_uuid_id
        assert isinstance(t["metadata"].get("assistant_id"), str)
        assert t["metadata"].get("graph_id") == "EvoScientist"
        assert t["metadata"].get("workspace_dir") == self._WS
        assert t["metadata"].get("model") == "test-model"

    def test_restore_includes_current_workspace_graph_threads_only(self):
        """Restore includes current-workspace graph threads only.

        Threads from other workspaces and pre-stamping rows without
        workspace_dir must NOT be resurrected — sessions.db is machine-global
        and an unscoped restore would expose them on the unauthenticated API
        (worst case --tunnel). Current-workspace async-subagent graph threads
        are restored; memory-worker graph threads remain disposable until
        worker cloning lands.
        """
        import sys
        import uuid as _uuid_mod
        from unittest.mock import MagicMock, patch

        from EvoScientist.sessions import _restore_webui_threads_to_global_store

        mine = "11111111-1111-1111-1111-111111111111"
        other_ws = "22222222-2222-2222-2222-222222222222"
        worker = "33333333-3333-3333-3333-333333333333"
        subagent = "44444444-4444-4444-4444-444444444444"
        legacy_no_ws = "55555555-5555-5555-5555-555555555555"

        mock_store: dict = {"threads": []}
        mock_global_store = MagicMock()
        mock_global_store.get.side_effect = mock_store.get
        mock_global_store.__getitem__ = lambda self, k: mock_store[k]
        mock_global_store.__setitem__ = lambda self, k, v: mock_store.__setitem__(k, v)

        fake_module = MagicMock()
        fake_module.GLOBAL_STORE = mock_global_store

        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "sessions.db")
            self._make_db_with_threads(db, [mine])
            self._make_db_with_threads(db, [other_ws], workspace_dir="/elsewhere")
            self._make_db_with_threads(db, [worker], graph_id="evomemory-turn-worker")
            self._make_db_with_threads(db, [subagent], graph_id="writing-agent")
            self._make_db_with_threads(db, [legacy_no_ws], workspace_dir=None)
            with (
                patch(
                    "EvoScientist.sessions.get_db_path",
                    return_value=_mock_path(db),
                ),
                patch.dict(
                    sys.modules, {"langgraph_runtime_inmem.database": fake_module}
                ),
                self._patch_workspace(),
            ):
                _run(_restore_webui_threads_to_global_store())

        added = mock_store["threads"]
        restored = {entry["thread_id"]: entry for entry in added}
        assert set(restored) == {
            _uuid_mod.UUID(mine),
            _uuid_mod.UUID(subagent),
        }
        assert restored[_uuid_mod.UUID(mine)]["metadata"].get("graph_id") == (
            "EvoScientist"
        )
        assert restored[_uuid_mod.UUID(mine)]["metadata"].get("workspace_dir") == (
            self._WS
        )
        assert restored[_uuid_mod.UUID(mine)]["metadata"].get("model") == ("test-model")
        assert restored[_uuid_mod.UUID(subagent)]["metadata"].get("graph_id") == (
            "writing-agent"
        )
        assert restored[_uuid_mod.UUID(subagent)]["metadata"].get("workspace_dir") == (
            self._WS
        )

    def test_purge_removes_only_evomemory_rows(self):
        """Startup purge drops evomemory-* residue, leaves everything else."""
        import sqlite3
        from unittest.mock import patch

        from EvoScientist.sessions import _purge_internal_worker_threads

        keep_main = "11111111-1111-1111-1111-111111111111"
        keep_cli = "abcd1234"
        drop_worker = "33333333-3333-3333-3333-333333333333"
        keep_subagent = "44444444-4444-4444-4444-444444444444"

        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "sessions.db")
            self._make_db_with_threads(db, [keep_main, keep_cli])
            self._make_db_with_threads(
                db, [drop_worker], graph_id="evomemory-turn-worker"
            )
            self._make_db_with_threads(db, [keep_subagent], graph_id="writing-agent")
            with patch(
                "EvoScientist.sessions.get_db_path",
                return_value=_mock_path(db),
            ):
                _run(_purge_internal_worker_threads())
                # Idempotent: second run is a no-op, not an error.
                _run(_purge_internal_worker_threads())

            con = sqlite3.connect(db)
            remaining = {
                r[0] for r in con.execute("SELECT DISTINCT thread_id FROM checkpoints")
            }
            con.close()

        assert remaining == {keep_main, keep_cli, keep_subagent}

    def test_cli_session_filters_exclude_non_main_graph_rows(self):
        from unittest.mock import patch

        from EvoScientist.sessions import (
            list_threads,
            resolve_thread_id_prefix,
            thread_exists,
        )

        main_thread = "11111111-1111-1111-1111-111111111111"
        worker_thread = "33333333-3333-3333-3333-333333333333"

        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "sessions.db")
            self._make_db_with_threads(db, [main_thread])
            self._make_db_with_threads(
                db, [worker_thread], graph_id="evomemory-turn-worker"
            )
            with patch(
                "EvoScientist.sessions.get_db_path",
                return_value=_mock_path(db),
            ):
                assert [row["thread_id"] for row in _run(list_threads())] == [
                    main_thread
                ]
                assert _run(thread_exists(main_thread))
                assert not _run(thread_exists(worker_thread))
                assert _run(resolve_thread_id_prefix(worker_thread[:8])) == (None, [])

    def test_restores_cli_rows_and_excludes_worker_residue(self):
        """CLI rows (agent_name, no graph_id) are restored with graph_id
        backfilled; crashed-worker residue (agent_name AND graph_id=
        evomemory-*) stays excluded — graph_id wins over agent_name."""
        import sys
        import uuid as _uuid_mod
        from unittest.mock import MagicMock, patch

        from EvoScientist.sessions import _restore_webui_threads_to_global_store

        cli_thread = "11111111-1111-1111-1111-111111111111"
        worker_residue = "22222222-2222-2222-2222-222222222222"

        mock_store: dict = {"threads": []}
        mock_global_store = MagicMock()
        mock_global_store.get.side_effect = mock_store.get
        mock_global_store.__getitem__ = lambda self, k: mock_store[k]
        mock_global_store.__setitem__ = lambda self, k, v: mock_store.__setitem__(k, v)

        fake_module = MagicMock()
        fake_module.GLOBAL_STORE = mock_global_store

        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "sessions.db")
            # CLI row: build_metadata stamps agent_name/workspace_dir but
            # never graph_id or assistant_id.
            self._make_db_with_threads(
                db, [cli_thread], assistant_id=None, graph_id=None
            )
            # Crashed memory-worker residue: stamps BOTH.
            self._make_db_with_threads(
                db, [worker_residue], graph_id="evomemory-turn-worker"
            )
            with (
                patch(
                    "EvoScientist.sessions.get_db_path",
                    return_value=_mock_path(db),
                ),
                patch.dict(
                    sys.modules, {"langgraph_runtime_inmem.database": fake_module}
                ),
                self._patch_workspace(),
            ):
                _run(_restore_webui_threads_to_global_store())

        added = mock_store["threads"]
        assert len(added) == 1
        assert added[0]["thread_id"] == _uuid_mod.UUID(cli_thread)
        # graph_id backfilled so Threads.State.get works on the CLI stub.
        assert added[0]["metadata"].get("graph_id") == "EvoScientist"
        assert added[0]["metadata"].get("workspace_dir") == self._WS
        assert added[0]["metadata"].get("model") == "test-model"

    def test_mixed_cli_webui_rows_keep_assistant_and_graph_id(self):
        """Interop thread (CLI rows + WebUI rows under one UUID): bare
        columns under GROUP BY let SQLite pick an arbitrary row's NULL —
        all metadata fields must be MAX-aggregated (Codex F2)."""
        import sys
        import uuid as _uuid_mod
        from unittest.mock import MagicMock, patch

        from EvoScientist.sessions import _restore_webui_threads_to_global_store

        tid = "11111111-1111-1111-1111-111111111111"
        asst = "a2b49500-c49b-5560-b664-d42ee8b66d3c"

        mock_store: dict = {"threads": []}
        mock_global_store = MagicMock()
        mock_global_store.get.side_effect = mock_store.get
        mock_global_store.__getitem__ = lambda self, k: mock_store[k]
        mock_global_store.__setitem__ = lambda self, k, v: mock_store.__setitem__(k, v)

        fake_module = MagicMock()
        fake_module.GLOBAL_STORE = mock_global_store

        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "sessions.db")
            # Older CLI row: no assistant_id / graph_id. "a-..." checkpoint
            # id sorts BEFORE the WebUI row's so a bare-column GROUP BY
            # would tend to surface this row's NULLs.
            self._make_db_with_threads(
                db, [tid], assistant_id=None, graph_id=None, ckpt_prefix="a"
            )
            # Newer WebUI row on the SAME thread: carries both.
            self._make_db_with_threads(db, [tid], assistant_id=asst, ckpt_prefix="b")
            with (
                patch(
                    "EvoScientist.sessions.get_db_path",
                    return_value=_mock_path(db),
                ),
                patch.dict(
                    sys.modules, {"langgraph_runtime_inmem.database": fake_module}
                ),
                self._patch_workspace(),
            ):
                _run(_restore_webui_threads_to_global_store())

        added = mock_store["threads"]
        assert len(added) == 1, f"expected 1 thread, got {added}"
        assert added[0]["thread_id"] == _uuid_mod.UUID(tid)
        assert added[0]["metadata"].get("assistant_id") == asst
        assert added[0]["metadata"].get("graph_id") == "EvoScientist"
        assert added[0]["metadata"].get("workspace_dir") == self._WS
        assert added[0]["metadata"].get("model") == "test-model"

    def test_restored_stub_gets_title_from_first_human_message(self):
        """Stubs carry metadata.title derived from the thread's first human
        message, so the WebUI sidebar doesn't show "Untitled Thread"."""
        import sys
        from unittest.mock import MagicMock, patch

        from langchain_core.messages import HumanMessage

        from EvoScientist.sessions import (
            AGENT_NAME,
            _restore_webui_threads_to_global_store,
            create_checkpointer_for_langgraph_api,
        )

        thread_id = "33333333-3333-3333-3333-333333333333"

        mock_store: dict = {"threads": []}
        mock_global_store = MagicMock()
        mock_global_store.get.side_effect = mock_store.get
        mock_global_store.__getitem__ = lambda self, k: mock_store[k]
        mock_global_store.__setitem__ = lambda self, k, v: mock_store.__setitem__(k, v)

        fake_module = MagicMock()
        fake_module.GLOBAL_STORE = mock_global_store

        async def _write_then_restore():
            async with create_checkpointer_for_langgraph_api() as cp:
                await cp.aput(
                    {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}},
                    {
                        "v": 1,
                        "id": "ckpt-title",
                        "ts": "2024-01-01T00:00:00+00:00",
                        "channel_values": {
                            "messages": [HumanMessage(content="hello title test")]
                        },
                        "channel_versions": {},
                        "versions_seen": {},
                        "pending_sends": [],
                    },
                    {"source": "loop", "step": 1, "graph_id": AGENT_NAME},
                    {},
                )
            await _restore_webui_threads_to_global_store()

        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "sessions.db")
            with (
                patch(
                    "EvoScientist.sessions.get_db_path",
                    return_value=_mock_path(db),
                ),
                patch.dict(
                    sys.modules, {"langgraph_runtime_inmem.database": fake_module}
                ),
                patch(
                    "EvoScientist.sessions._api_workspace_dir",
                    return_value=self._WS,
                ),
            ):
                _run(_write_then_restore())

        added = mock_store["threads"]
        assert len(added) == 1, f"expected 1 restored thread, got {added}"
        assert added[0]["metadata"].get("title") == "hello title test"

    def test_removes_preloaded_uuid_entries_outside_restore_scope(self):
        """Stale and out-of-scope .pckl UUID entries are dropped.

        Stale UUID entries point at deleted/lost state and render as empty
        sessions (the #277 symptom). Out-of-scope UUID entries point at another
        workspace's state and must not remain in this server's unauthenticated
        thread registry. CLI-style non-UUID entries are never touched.
        """
        import sys
        import uuid as _uuid_mod
        from unittest.mock import MagicMock, patch

        from EvoScientist.sessions import _restore_webui_threads_to_global_store

        in_scope = "11111111-1111-1111-1111-111111111111"
        out_of_scope = "22222222-2222-2222-2222-222222222222"
        ghost = "99999999-9999-9999-9999-999999999999"

        ghost_entry: dict = {"thread_id": ghost, "status": "idle", "metadata": {}}
        out_of_scope_entry: dict = {
            "thread_id": out_of_scope,
            "status": "idle",
            "metadata": {},
        }
        cli_entry: dict = {"thread_id": "notauuid", "status": "idle"}
        mock_store: dict = {"threads": [ghost_entry, out_of_scope_entry, cli_entry]}
        mock_global_store = MagicMock()
        mock_global_store.get.side_effect = mock_store.get
        mock_global_store.__getitem__ = lambda self, k: mock_store[k]
        mock_global_store.__setitem__ = lambda self, k, v: mock_store.__setitem__(k, v)

        fake_module = MagicMock()
        fake_module.GLOBAL_STORE = mock_global_store

        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "sessions.db")
            self._make_db_with_threads(db, [in_scope])
            self._make_db_with_threads(db, [out_of_scope], workspace_dir="/elsewhere")
            with (
                patch(
                    "EvoScientist.sessions.get_db_path",
                    return_value=_mock_path(db),
                ),
                patch.dict(
                    sys.modules, {"langgraph_runtime_inmem.database": fake_module}
                ),
                self._patch_workspace(),
            ):
                _run(_restore_webui_threads_to_global_store())

        ids = [t["thread_id"] for t in mock_store["threads"]]
        assert _uuid_mod.UUID(ghost) not in ids, f"ghost must be removed, got {ids}"
        assert ghost not in ids, f"ghost must be removed (str form), got {ids}"
        assert "notauuid" in ids, "CLI-style entries must never be touched"
        assert _uuid_mod.UUID(out_of_scope) not in ids
        assert out_of_scope not in ids
        # In-scope thread restored as usual.
        assert _uuid_mod.UUID(in_scope) in ids

    def test_no_op_when_langgraph_runtime_inmem_absent(self):
        """ImportError for langgraph_runtime_inmem is silently swallowed."""
        import sys
        from unittest.mock import patch

        from EvoScientist.sessions import _restore_webui_threads_to_global_store

        with patch.dict(sys.modules, {"langgraph_runtime_inmem.database": None}):
            # Must not raise.
            _run(_restore_webui_threads_to_global_store())

    def test_no_op_when_db_has_no_checkpoints_table(self):
        """Missing checkpoints table is handled gracefully."""
        import sys
        from unittest.mock import MagicMock, patch

        from EvoScientist.sessions import _restore_webui_threads_to_global_store

        mock_store: dict = {"threads": []}
        mock_global_store = MagicMock()
        mock_global_store.get.side_effect = mock_store.get
        mock_global_store.__getitem__ = lambda self, k: mock_store[k]

        fake_module = MagicMock()
        fake_module.GLOBAL_STORE = mock_global_store

        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "empty.db")
            # Create a valid but empty SQLite DB (no checkpoints table).
            import sqlite3

            sqlite3.connect(db).close()
            with (
                patch(
                    "EvoScientist.sessions.get_db_path",
                    return_value=_mock_path(db),
                ),
                patch.dict(
                    sys.modules, {"langgraph_runtime_inmem.database": fake_module}
                ),
            ):
                _run(_restore_webui_threads_to_global_store())

        # threads list untouched.
        assert mock_store["threads"] == []

    def test_create_checkpointer_calls_restore(self):
        """create_checkpointer_for_langgraph_api calls _restore_webui_threads_to_global_store."""
        from unittest.mock import patch

        from EvoScientist.sessions import create_checkpointer_for_langgraph_api

        restore_called = []

        async def fake_restore():
            restore_called.append(True)

        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "sessions.db")
            with (
                patch(
                    "EvoScientist.sessions.get_db_path",
                    return_value=_mock_path(db),
                ),
                patch(
                    "EvoScientist.sessions._restore_webui_threads_to_global_store",
                    side_effect=fake_restore,
                ),
            ):

                async def _run_inner():
                    async with create_checkpointer_for_langgraph_api():
                        pass

                _run(_run_inner())

        assert restore_called, "_restore_webui_threads_to_global_store must be called"


if __name__ == "__main__":
    unittest.main()
