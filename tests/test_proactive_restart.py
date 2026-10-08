"""Restart survival of proactive pushes: origin in run metadata, restore at startup."""

from __future__ import annotations

import json
import logging
import sqlite3
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langgraph_sdk.errors import NotFoundError

from EvoScientist.cli.channel import (
    forget_channel_origin,
    get_channel_origin,
    restore_channel_origin,
)
from EvoScientist.cli.commands import _serve_turn_metadata
from EvoScientist.commands.base import ChannelRuntime
from EvoScientist.paths import SessionDirs
from EvoScientist.proactive.serve import (
    ORIGIN_METADATA_KEYS,
    ProactiveServe,
    origin_from_row,
    origin_run_metadata,
)
from EvoScientist.sessions import AGENT_NAME, latest_origin_checkpoints

RESERVED_REQUEST_METADATA_KEYS = {
    "thread_id",
    "assistant_id",
    "run_id",
    "cron_id",
    "counters_since_delta_snapshot",
}


def _origin(metadata=None):
    return SimpleNamespace(
        channel_type="slack", chat_id="C1", sender="u1", metadata=metadata
    )


# ---- the stored origin ---------------------------------------------------------


def test_origin_metadata_keeps_only_the_reply_handles():
    meta = origin_run_metadata(
        _origin(
            {
                "thread_ts": "1700.1",
                "subject": "Re: results",
                "source_name": "Alice",
                "sender_id": "+48123",
                "event_id": "e1",
            }
        )
    )
    assert meta["origin_channel_type"] == "slack"
    assert meta["origin_chat_id"] == "C1"
    assert meta["origin_sender"] == "u1"
    assert json.loads(meta["origin_metadata"]) == {
        "thread_ts": "1700.1",
        "event_id": "e1",
    }
    assert all(isinstance(v, str) for v in meta.values())
    assert not set(meta) & RESERVED_REQUEST_METADATA_KEYS


def test_origin_without_reply_handles_stores_no_metadata_key():
    assert "origin_metadata" not in origin_run_metadata(_origin({"subject": "x"}))
    assert "origin_metadata" not in origin_run_metadata(_origin(None))


def test_an_origin_value_that_does_not_encode_is_dropped_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="EvoScientist.proactive.serve"):
        meta = origin_run_metadata(_origin({"thread_ts": object()}))
    assert "origin_metadata" not in meta
    assert meta["origin_chat_id"] == "C1"
    assert "does not encode" in caplog.text


def test_origin_round_trips_through_a_stored_row():
    row = origin_run_metadata(_origin({"thread_ts": "1700.1", "subject": "x"}))
    assert origin_from_row(row) == {
        "channel_type": "slack",
        "chat_id": "C1",
        "sender": "u1",
        "metadata": {"thread_ts": "1700.1"},
    }


def test_a_row_without_an_origin_gives_none():
    assert origin_from_row({"origin_channel_type": "slack"}) is None


def test_allowlist_is_what_the_spec_names():
    assert set(ORIGIN_METADATA_KEYS) == {
        "thread_ts",
        "thread_id",
        "chat_type",
        "msg_type",
        "event_id",
        "context_token",
    }


# ---- serve's run metadata ------------------------------------------------------


def _controller_runtime(armed: bool):
    controller = MagicMock()
    controller.is_armed.return_value = armed
    return ChannelRuntime(thread_id="t1", proactive=controller)


@pytest.fixture
def remembered_origin():
    restore_channel_origin(
        "t1",
        channel_type="slack",
        chat_id="C1",
        sender="u1",
        metadata={"thread_ts": "9"},
    )
    yield
    forget_channel_origin("t1")


def test_an_armed_turn_carries_the_origin(workspace, remembered_origin):
    meta = _serve_turn_metadata(
        SessionDirs(workspace), "m", "t1", _controller_runtime(armed=True)
    )
    assert meta["origin_chat_id"] == "C1"
    assert json.loads(meta["origin_metadata"]) == {"thread_ts": "9"}
    assert meta["workspace_dir"] == workspace.key


@pytest.mark.parametrize("runtime", ["unarmed", "no_controller", "none"])
def test_other_turns_carry_only_the_usual_metadata(
    workspace, remembered_origin, runtime
):
    channel_runtime = {
        "unarmed": _controller_runtime(armed=False),
        "no_controller": ChannelRuntime(thread_id="t1"),
        "none": None,
    }[runtime]
    meta = _serve_turn_metadata(SessionDirs(workspace), "m", "t1", channel_runtime)
    assert not any(key.startswith("origin_") for key in meta)


# ---- the startup query ---------------------------------------------------------


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "sessions.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE checkpoints (thread_id TEXT, checkpoint_ns TEXT, "
            "checkpoint_id TEXT, metadata TEXT)"
        )
    with patch("EvoScientist.sessions.get_db_path", return_value=path):
        yield path


def _row(db, thread_id, checkpoint_id, **metadata):
    meta = {"agent_name": AGENT_NAME, **metadata}
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO checkpoints VALUES (?, '', ?, ?)",
            (thread_id, checkpoint_id, json.dumps(meta)),
        )


async def test_latest_origin_checkpoints_takes_the_newest_row_with_an_origin(db):
    origin = {"origin_channel_type": "telegram", "origin_chat_id": "42"}
    _row(db, "t1", "c1", updated_at="2026-10-01T10:00:00", proactive="on", **origin)
    _row(
        db,
        "t1",
        "c2",
        updated_at="2026-10-01T11:00:00",
        proactive="on",
        **{**origin, "origin_chat_id": "43"},
    )
    # A later checkpoint without origin keys (/compact, a HITL resume).
    _row(db, "t1", "c3", updated_at="2026-10-01T12:00:00", proactive="on")
    _row(db, "t2", "c4", updated_at="2026-10-02T09:00:00", proactive="off", **origin)
    _row(db, "t3", "c5", updated_at="2026-10-03T09:00:00")  # no origin at all

    rows = await latest_origin_checkpoints()

    assert [r["thread_id"] for r in rows] == ["t2", "t1"]
    t1 = rows[1]
    assert t1["origin_chat_id"] == "43"
    assert t1["updated_at"] == "2026-10-01T11:00:00"
    assert t1["proactive"] == "on"


async def test_latest_origin_checkpoints_without_a_table_is_empty(tmp_path):
    with patch("EvoScientist.sessions.get_db_path", return_value=tmp_path / "x.db"):
        assert await latest_origin_checkpoints() == []


# ---- restore and the fresh thread ------------------------------------------------


class _Client:
    def __init__(self, threads):
        self.threads = SimpleNamespace(get=self._get)
        self._threads = threads

    def _get(self, thread_id):
        if thread_id not in self._threads:
            raise NotFoundError("gone", response=MagicMock(), body=None)
        return self._threads[thread_id]


def _restore_row(workspace, thread_id, **overrides):
    row = {
        "thread_id": thread_id,
        "workspace_dir": workspace.key,
        "run_dir": None,
        "proactive": "on",
        "origin_channel_type": "telegram",
        "origin_chat_id": "42",
        "origin_sender": "u1",
        "origin_metadata": None,
    }
    row.update(overrides)
    return row


def test_restore_rearms_the_newest_armed_thread_of_this_workspace(workspace, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    client = _Client(
        {"t1": {"metadata": {}}, "t2": {"metadata": {}}, "t3": {"metadata": {}}}
    )
    controller = ProactiveServe(workspace=workspace, client=client)
    restored = []
    rows = [
        _restore_row(workspace, "t0", proactive="off"),
        _restore_row(workspace, "t1", workspace_dir=other.as_posix()),
        _restore_row(workspace, "gone"),
        _restore_row(workspace, "t2"),
        _restore_row(workspace, "t3"),
    ]

    result = controller.restore(
        rows, restore_origin=lambda tid, origin: restored.append((tid, origin))
    )

    assert result == "t2"
    assert controller.armed == {"t2"}
    assert controller.restored_thread_id == "t2"
    assert restored == [
        (
            "t2",
            {
                "channel_type": "telegram",
                "chat_id": "42",
                "sender": "u1",
                "metadata": None,
            },
        )
    ]


def test_restore_skips_a_thread_disarmed_after_its_last_turn(workspace):
    client = _Client({"t1": {"metadata": {"proactive": "off"}}})
    controller = ProactiveServe(workspace=workspace, client=client)
    assert (
        controller.restore([_restore_row(workspace, "t1")], restore_origin=MagicMock())
        is None
    )
    assert controller.armed == set()


def test_restore_does_nothing_when_unavailable(workspace):
    controller = ProactiveServe(workspace=workspace, unavailable_reason="local backend")
    origin = MagicMock()
    assert (
        controller.restore([_restore_row(workspace, "t1")], restore_origin=origin)
        is None
    )
    origin.assert_not_called()


def test_a_turn_in_the_fresh_thread_ends_the_restored_thread(workspace):
    forgotten = []
    controller = ProactiveServe(
        workspace=workspace, client=MagicMock(), forget_origin=forgotten.append
    )
    controller.armed.add("t-old")
    controller.restored_thread_id = "t-old"
    controller.mirror = MagicMock()

    controller.on_turn("t-fresh")

    assert controller.restored_thread_id is None
    assert controller.armed == set()
    assert forgotten == ["t-old"]
    controller.mirror.assert_called_once_with()


def test_a_turn_in_the_restored_thread_keeps_it(workspace):
    controller = ProactiveServe(workspace=workspace, client=MagicMock())
    controller.armed.add("t-old")
    controller.restored_thread_id = "t-old"
    controller.on_turn("t-old")
    assert controller.restored_thread_id == "t-old"


def test_restore_channel_origin_feeds_the_delivery_registry():
    restore_channel_origin(
        "t9", channel_type="qq", chat_id="g1", sender="", metadata={"event_id": "e"}
    )
    try:
        origin = get_channel_origin("t9")
        assert (origin.channel_type, origin.chat_id, origin.metadata) == (
            "qq",
            "g1",
            {"event_id": "e"},
        )
    finally:
        forget_channel_origin("t9")
