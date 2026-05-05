"""Tests for EvoScientist.audit — session audit log."""

from __future__ import annotations

import importlib
import json
import re
import time
from pathlib import Path

import pytest

import EvoScientist.paths as paths
from EvoScientist.audit import AuditLogger


@pytest.fixture()
def tmp_sessions_dir(tmp_path):
    return tmp_path / "sessions"


@pytest.fixture()
def logger(tmp_sessions_dir):
    return AuditLogger(thread_id="test-thread-001", sessions_dir=tmp_sessions_dir)


class TestAuditLoggerCreation:
    def test_log_file_not_created_before_first_event(self, logger, tmp_sessions_dir):
        log_path = tmp_sessions_dir / "test-thread-001" / "audit.jsonl"
        assert not log_path.exists()

    def test_log_file_created_on_first_event(self, logger, tmp_sessions_dir):
        logger.log_session_start(model="claude-sonnet-4-6", provider="anthropic")
        log_path = tmp_sessions_dir / "test-thread-001" / "audit.jsonl"
        assert log_path.exists()

    def test_log_path_property(self, logger, tmp_sessions_dir):
        expected = tmp_sessions_dir / "test-thread-001" / "audit.jsonl"
        assert logger.log_path == expected

    def test_sessions_dir_can_be_overridden_by_env(self, monkeypatch, tmp_path):
        custom_sessions = tmp_path / "custom-sessions"
        monkeypatch.setenv("EVOSCIENTIST_SESSIONS_DIR", str(custom_sessions))

        reloaded_paths = importlib.reload(paths)
        try:
            assert reloaded_paths.SESSIONS_DIR == custom_sessions
        finally:
            monkeypatch.delenv("EVOSCIENTIST_SESSIONS_DIR", raising=False)
            importlib.reload(paths)


class TestSessionMetadata:
    def test_session_start_writes_metadata(self, logger, tmp_sessions_dir):
        logger.log_session_start(model="claude-sonnet-4-6", provider="anthropic")
        lines = _read_lines(tmp_sessions_dir / "test-thread-001" / "audit.jsonl")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["kind"] == "session_start"
        assert entry["thread_id"] == "test-thread-001"
        assert entry["model"] == "claude-sonnet-4-6"
        assert entry["provider"] == "anthropic"
        assert re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z",
            entry["ts"],
        )

    def test_session_end_appends_end_entry(self, logger, tmp_sessions_dir):
        logger.log_session_start(model="claude-sonnet-4-6", provider="anthropic")
        logger.log_session_end()
        lines = _read_lines(tmp_sessions_dir / "test-thread-001" / "audit.jsonl")
        assert len(lines) == 2
        end_entry = json.loads(lines[-1])
        assert end_entry["kind"] == "session_end"
        assert end_entry["ts"].endswith("Z")

    def test_session_start_is_written_once(self, logger, tmp_sessions_dir):
        logger.log_session_start(model="claude-sonnet-4-6", provider="anthropic")
        logger.log_session_start(model="claude-sonnet-4-6", provider="anthropic")
        lines = _read_lines(tmp_sessions_dir / "test-thread-001" / "audit.jsonl")
        assert [json.loads(line)["kind"] for line in lines] == ["session_start"]

    def test_session_end_is_written_once(self, logger, tmp_sessions_dir):
        logger.log_session_start(model="claude-sonnet-4-6", provider="anthropic")
        logger.log_session_end()
        logger.log_session_end()
        lines = _read_lines(tmp_sessions_dir / "test-thread-001" / "audit.jsonl")
        assert [json.loads(line)["kind"] for line in lines] == [
            "session_start",
            "session_end",
        ]


class TestToolCallLogging:
    def test_tool_call_logged(self, logger, tmp_sessions_dir):
        logger.log_event(
            "tool_call",
            {"name": "read_file", "args": {"path": "/tmp/x"}, "id": "tc-1"},
        )
        lines = _read_lines(tmp_sessions_dir / "test-thread-001" / "audit.jsonl")
        entry = json.loads(lines[0])
        assert entry["kind"] == "tool_call"
        assert entry["tool"] == "read_file"
        assert entry["args"] == {"path": "/tmp/x"}
        assert entry["tool_id"] == "tc-1"
        assert "ts" in entry

    def test_tool_call_with_empty_args_then_full_args_keeps_empty_args_once(
        self, logger, tmp_sessions_dir
    ):
        logger.log_event("tool_call", {"name": "read_file", "args": {}, "id": "tc-1"})
        logger.log_event(
            "tool_call",
            {"name": "read_file", "args": {"path": "/tmp/x"}, "id": "tc-1"},
        )
        lines = _read_lines(tmp_sessions_dir / "test-thread-001" / "audit.jsonl")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["kind"] == "tool_call"
        assert entry["args"] == {}
        assert entry["tool_id"] == "tc-1"

    def test_tool_call_with_empty_args_is_flushed_on_session_end(
        self, logger, tmp_sessions_dir
    ):
        logger.log_session_start(model="claude-sonnet-4-6", provider="anthropic")
        logger.log_event("tool_call", {"name": "list_dir", "args": {}, "id": "tc-1"})
        logger.log_session_end()
        entries = _read_entries(tmp_sessions_dir / "test-thread-001" / "audit.jsonl")
        assert [entry["kind"] for entry in entries] == [
            "session_start",
            "tool_call",
            "session_end",
        ]
        assert entries[1]["args"] == {}

    def test_tool_call_with_empty_args_is_flushed_before_tool_result(
        self, logger, tmp_sessions_dir
    ):
        logger.log_event("tool_call", {"name": "list_dir", "args": {}, "id": "tc-1"})
        logger.log_event(
            "tool_result",
            {"name": "list_dir", "content": "ok", "success": True, "id": "tc-1"},
        )
        entries = _read_entries(tmp_sessions_dir / "test-thread-001" / "audit.jsonl")
        assert [entry["kind"] for entry in entries] == ["tool_call", "tool_result"]
        assert entries[0]["args"] == {}

    def test_tool_call_with_missing_args_waits_for_full_args(
        self, logger, tmp_sessions_dir
    ):
        logger.log_event("tool_call", {"name": "read_file", "id": "tc-1"})
        logger.log_event(
            "tool_call",
            {"name": "read_file", "args": {"path": "/tmp/x"}, "id": "tc-1"},
        )
        entries = _read_entries(tmp_sessions_dir / "test-thread-001" / "audit.jsonl")
        assert len(entries) == 1
        assert entries[0]["args"] == {"path": "/tmp/x"}

    def test_tool_call_args_truncated_at_2000(self, logger, tmp_sessions_dir):
        logger.log_event(
            "tool_call",
            {
                "name": "write_file",
                "args": {"path": "/tmp/x", "content": "x" * 3000},
                "id": "tc-1",
            },
        )
        lines = _read_lines(tmp_sessions_dir / "test-thread-001" / "audit.jsonl")
        entry = json.loads(lines[0])
        assert isinstance(entry["args"], dict)
        assert entry["args"]["truncated"] is True
        assert entry["args"]["original_type"] == "dict"
        assert len(entry["args"]["serialized"]) == 2000

    def test_tool_result_logged(self, logger, tmp_sessions_dir):
        logger.log_event("tool_result", {"name": "read_file", "content": "hello", "success": True})
        lines = _read_lines(tmp_sessions_dir / "test-thread-001" / "audit.jsonl")
        entry = json.loads(lines[0])
        assert entry["kind"] == "tool_result"
        assert entry["tool"] == "read_file"
        assert entry["success"] is True
        assert entry["content"] == "hello"

    def test_tool_result_content_truncated_at_2000(self, logger, tmp_sessions_dir):
        long_content = "x" * 3000
        logger.log_event("tool_result", {"name": "grep", "content": long_content, "success": True})
        lines = _read_lines(tmp_sessions_dir / "test-thread-001" / "audit.jsonl")
        entry = json.loads(lines[0])
        assert len(entry["content"]) == 2000

    def test_tool_result_short_content_not_truncated(self, logger, tmp_sessions_dir):
        logger.log_event("tool_result", {"name": "grep", "content": "short", "success": True})
        lines = _read_lines(tmp_sessions_dir / "test-thread-001" / "audit.jsonl")
        entry = json.loads(lines[0])
        assert entry["content"] == "short"


class TestSubagentLogging:
    def test_subagent_start_logged(self, logger, tmp_sessions_dir):
        event = {
            "type": "subagent_start",
            "name": "research-agent",
            "description": "search papers",
        }
        logger.log_event("subagent_start", event)
        lines = _read_lines(tmp_sessions_dir / "test-thread-001" / "audit.jsonl")
        entry = json.loads(lines[0])
        assert entry["kind"] == "subagent_start"
        assert entry["name"] == "research-agent"
        assert entry["description"] == "search papers"
        assert "agent_id" not in entry

    def test_subagent_end_logged(self, logger, tmp_sessions_dir):
        event = {"type": "subagent_end", "name": "research-agent"}
        logger.log_event("subagent_end", event)
        lines = _read_lines(tmp_sessions_dir / "test-thread-001" / "audit.jsonl")
        entry = json.loads(lines[0])
        assert entry["kind"] == "subagent_end"
        assert entry["name"] == "research-agent"
        assert "agent_id" not in entry

    def test_subagent_tool_call_logged(self, logger, tmp_sessions_dir):
        event = {
            "type": "subagent_tool_call",
            "subagent": "research-agent",
            "name": "read_file",
            "args": {"path": "/tmp/x"},
            "id": "tc-1",
        }
        logger.log_event("subagent_tool_call", event)
        lines = _read_lines(tmp_sessions_dir / "test-thread-001" / "audit.jsonl")
        entry = json.loads(lines[0])
        assert entry["kind"] == "subagent_tool_call"
        assert entry["subagent"] == "research-agent"
        assert entry["tool"] == "read_file"
        assert entry["args"] == {"path": "/tmp/x"}
        assert entry["tool_id"] == "tc-1"

    def test_subagent_tool_result_logged(self, logger, tmp_sessions_dir):
        event = {
            "type": "subagent_tool_result",
            "subagent": "research-agent",
            "name": "read_file",
            "content": "hello",
            "success": True,
            "id": "tc-1",
        }
        logger.log_event("subagent_tool_result", event)
        lines = _read_lines(tmp_sessions_dir / "test-thread-001" / "audit.jsonl")
        entry = json.loads(lines[0])
        assert entry["kind"] == "subagent_tool_result"
        assert entry["subagent"] == "research-agent"
        assert entry["tool"] == "read_file"
        assert entry["content"] == "hello"
        assert entry["success"] is True
        assert entry["tool_id"] == "tc-1"

    def test_unknown_event_type_ignored(self, logger, tmp_sessions_dir):
        logger.log_event("thinking", {"content": "hmm"})
        log_path = tmp_sessions_dir / "test-thread-001" / "audit.jsonl"
        assert not log_path.exists()


class TestErrorHandling:
    def test_write_failure_does_not_raise(self, tmp_path):
        # Point sessions_dir at a file (not a dir) to force an OS error
        blocker = tmp_path / "sessions"
        blocker.write_text("not a dir")
        audit = AuditLogger(thread_id="t1", sessions_dir=blocker)
        # Must not raise
        audit.log_event("tool_call", {"name": "read_file", "args": {}, "id": "x"})

    def test_session_start_failure_does_not_raise(self, tmp_path):
        blocker = tmp_path / "sessions"
        blocker.write_text("not a dir")
        audit = AuditLogger(thread_id="t1", sessions_dir=blocker)
        audit.log_session_start(model="claude-sonnet-4-6", provider="anthropic")

    def test_session_end_failure_does_not_raise(self, tmp_path):
        blocker = tmp_path / "sessions"
        blocker.write_text("not a dir")
        audit = AuditLogger(thread_id="t1", sessions_dir=blocker)
        audit.log_session_end()


class TestMultipleEvents:
    def test_multiple_events_appended_in_order(self, logger, tmp_sessions_dir):
        logger.log_event("tool_call", {"name": "read_file", "args": {}, "id": "tc-1"})
        logger.log_event("tool_result", {"name": "read_file", "content": "data", "success": True})
        logger.log_event(
            "tool_call",
            {"name": "write_file", "args": {"path": "/tmp/y"}, "id": "tc-2"},
        )
        lines = _read_lines(tmp_sessions_dir / "test-thread-001" / "audit.jsonl")
        assert len(lines) == 3
        assert json.loads(lines[0])["kind"] == "tool_call"
        assert json.loads(lines[1])["kind"] == "tool_result"
        assert json.loads(lines[2])["kind"] == "tool_call"

    def test_each_line_is_valid_json(self, logger, tmp_sessions_dir):
        logger.log_session_start(model="m", provider="p")
        logger.log_event("tool_call", {"name": "t", "args": {}, "id": "i"})
        logger.log_event("tool_result", {"name": "t", "content": "c", "success": True})
        logger.log_session_end()
        lines = _read_lines(tmp_sessions_dir / "test-thread-001" / "audit.jsonl")
        for line in lines:
            json.loads(line)  # must not raise

    def test_tool_tracking_state_pruned_after_matching_result(self, logger):
        logger.log_event("tool_call", {"name": "read_file", "id": "tc-1"})
        logger.log_event(
            "tool_call",
            {"name": "read_file", "args": {"path": "/tmp/x"}, "id": "tc-1"},
        )
        logger.log_event(
            "tool_result",
            {"name": "read_file", "content": "ok", "success": True, "id": "tc-1"},
        )

        assert logger._pending_tool_calls == {}
        assert logger._logged_tool_calls == set()


def _read_lines(path: Path) -> list[str]:
    return [ln for ln in path.read_text().splitlines() if ln.strip()]


def _read_entries(path: Path) -> list[dict]:
    return [json.loads(line) for line in _read_lines(path)]
