"""Tests for EvoScientist.sessions — thread CRUD, ID generation, helpers."""

import json
import os
import tempfile
import unittest
from datetime import UTC
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from EvoScientist.sessions import (
    AGENT_NAME,
    _format_relative_time,
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


class TestGenerateThreadId(unittest.TestCase):
    def test_length(self):
        tid = generate_thread_id()
        assert len(tid) == 8

    def test_hex(self):
        tid = generate_thread_id()
        int(tid, 16)  # Should not raise

    def test_uniqueness(self):
        ids = {generate_thread_id() for _ in range(100)}
        assert len(ids) == 100


class TestGetDbPath(unittest.TestCase):
    def test_uses_data_dir(self):
        path = get_db_path()
        assert str(path).endswith("sessions.db")
        assert ".evoscientist" in str(path)


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
                        blob BLOB,
                        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
                    )
                """)

                # Insert test checkpoints
                for i, tid in enumerate(["abc12345", "abc12399", "def00001"]):
                    meta = json.dumps(
                        {
                            "agent_name": AGENT_NAME,
                            "updated_at": f"2025-01-{15 + i}T10:00:00+00:00",
                            "workspace_dir": f"/tmp/ws_{tid}",
                            "model": "claude-sonnet-4-5",
                        }
                    )
                    await conn.execute(
                        "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, metadata) VALUES (?, '', ?, ?)",
                        (tid, f"cp_{i}", meta),
                    )

                # Insert a non-EvoScientist checkpoint (should be filtered)
                other_meta = json.dumps(
                    {
                        "agent_name": "OtherAgent",
                        "updated_at": "2025-01-20T10:00:00+00:00",
                    }
                )
                await conn.execute(
                    "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, metadata) VALUES (?, '', ?, ?)",
                    ("zzz99999", "cp_other", other_meta),
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

    def test_get_most_recent(self):
        recent = _run(get_most_recent())
        assert recent is not None
        assert recent == "def00001"

    def test_get_thread_metadata(self):
        meta = _run(get_thread_metadata("abc12345"))
        assert meta is not None
        assert meta["workspace_dir"] == "/tmp/ws_abc12345"
        assert meta["model"] == "claude-sonnet-4-5"

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
                    "INSERT INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, blob) "
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
                    "INSERT INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, blob) "
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


if __name__ == "__main__":
    unittest.main()
