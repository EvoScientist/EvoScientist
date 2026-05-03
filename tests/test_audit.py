"""Tests for EvoScientist.audit — session audit log."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

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
        logger.log_session_start(model="claude-sonnet-4-6")
        log_path = tmp_sessions_dir / "test-thread-001" / "audit.jsonl"
        assert log_path.exists()

    def test_log_path_property(self, logger, tmp_sessions_dir):
        expected = tmp_sessions_dir / "test-thread-001" / "audit.jsonl"
        assert logger.log_path == expected


class TestSessionMetadata:
    def test_session_start_writes_metadata(self, logger, tmp_sessions_dir):
        logger.log_session_start(model="claude-sonnet-4-6")
        lines = _read_lines(tmp_sessions_dir / "test-thread-001" / "audit.jsonl")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["kind"] == "session_start"
        assert entry["thread_id"] == "test-thread-001"
        assert entry["model"] == "claude-sonnet-4-6"
        assert "ts" in entry

    def test_session_end_appends_end_entry(self, logger, tmp_sessions_dir):
        logger.log_session_start(model="claude-sonnet-4-6")
        logger.log_session_end()
        lines = _read_lines(tmp_sessions_dir / "test-thread-001" / "audit.jsonl")
        assert len(lines) == 2
        end_entry = json.loads(lines[-1])
        assert end_entry["kind"] == "session_end"
        assert "ts" in end_entry


class TestToolCallLogging:
    def test_tool_call_logged(self, logger, tmp_sessions_dir):
        logger.log_event("tool_call", {"name": "read_file", "args": {"path": "/tmp/x"}, "id": "tc-1"})
        lines = _read_lines(tmp_sessions_dir / "test-thread-001" / "audit.jsonl")
        entry = json.loads(lines[0])
        assert entry["kind"] == "tool_call"
        assert entry["tool"] == "read_file"
        assert entry["args"] == {"path": "/tmp/x"}
        assert entry["tool_id"] == "tc-1"
        assert "ts" in entry

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
        logger.log_event("subagent_start", {"name": "research-agent", "agent_id": "sa-1"})
        lines = _read_lines(tmp_sessions_dir / "test-thread-001" / "audit.jsonl")
        entry = json.loads(lines[0])
        assert entry["kind"] == "subagent_start"
        assert entry["name"] == "research-agent"
        assert entry["agent_id"] == "sa-1"

    def test_subagent_end_logged(self, logger, tmp_sessions_dir):
        logger.log_event("subagent_end", {"name": "research-agent", "agent_id": "sa-1"})
        lines = _read_lines(tmp_sessions_dir / "test-thread-001" / "audit.jsonl")
        entry = json.loads(lines[0])
        assert entry["kind"] == "subagent_end"

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
        audit.log_session_start(model="claude-sonnet-4-6")

    def test_session_end_failure_does_not_raise(self, tmp_path):
        blocker = tmp_path / "sessions"
        blocker.write_text("not a dir")
        audit = AuditLogger(thread_id="t1", sessions_dir=blocker)
        audit.log_session_end()


class TestMultipleEvents:
    def test_multiple_events_appended_in_order(self, logger, tmp_sessions_dir):
        logger.log_event("tool_call", {"name": "read_file", "args": {}, "id": "tc-1"})
        logger.log_event("tool_result", {"name": "read_file", "content": "data", "success": True})
        logger.log_event("tool_call", {"name": "write_file", "args": {}, "id": "tc-2"})
        lines = _read_lines(tmp_sessions_dir / "test-thread-001" / "audit.jsonl")
        assert len(lines) == 3
        assert json.loads(lines[0])["kind"] == "tool_call"
        assert json.loads(lines[1])["kind"] == "tool_result"
        assert json.loads(lines[2])["kind"] == "tool_call"

    def test_each_line_is_valid_json(self, logger, tmp_sessions_dir):
        logger.log_session_start(model="m")
        logger.log_event("tool_call", {"name": "t", "args": {}, "id": "i"})
        logger.log_event("tool_result", {"name": "t", "content": "c", "success": True})
        logger.log_session_end()
        lines = _read_lines(tmp_sessions_dir / "test-thread-001" / "audit.jsonl")
        for line in lines:
            json.loads(line)  # must not raise


def _read_lines(path: Path) -> list[str]:
    return [ln for ln in path.read_text().splitlines() if ln.strip()]
