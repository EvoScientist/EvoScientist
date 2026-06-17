"""Tests for synthesis-tracking additions to EvoScientist.memory.worker_activity.

This PR adds:
- synthesis_running field to MemoryWorkerStatusSnapshot
- knowledge_created / knowledge_updated / knowledge_archived fields
- mark_synthesis_started / mark_synthesis_finished functions
- knowledge delta tracking in _memory_output_delta and related helpers
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from EvoScientist.memory.worker_activity import (
    MemoryWorkerStatusSnapshot,
    mark_synthesis_finished,
    mark_synthesis_started,
    memory_worker_observed_outputs,
    memory_worker_status,
    reset_memory_worker_status_for_tests,
    snapshot_memory_outputs,
)


@pytest.fixture(autouse=True)
def _reset_worker_state():
    """Isolate worker-activity module globals around every test."""
    reset_memory_worker_status_for_tests()
    yield
    reset_memory_worker_status_for_tests()


# ---------------------------------------------------------------------------
# MemoryWorkerStatusSnapshot
# ---------------------------------------------------------------------------


class TestMemoryWorkerStatusSnapshot:
    def test_default_values(self):
        snap = MemoryWorkerStatusSnapshot()
        assert snap.is_running is False
        assert snap.synthesis_running is False
        assert snap.profile_updates == 0
        assert snap.observations_recorded == 0
        assert snap.knowledge_created == 0
        assert snap.knowledge_updated == 0
        assert snap.knowledge_archived == 0

    def test_custom_values(self):
        snap = MemoryWorkerStatusSnapshot(
            is_running=True,
            synthesis_running=True,
            profile_updates=2,
            observations_recorded=3,
            knowledge_created=1,
            knowledge_updated=4,
            knowledge_archived=0,
        )
        assert snap.is_running is True
        assert snap.synthesis_running is True
        assert snap.knowledge_created == 1
        assert snap.knowledge_updated == 4


# ---------------------------------------------------------------------------
# mark_synthesis_started / mark_synthesis_finished
# ---------------------------------------------------------------------------


class TestMarkSynthesisStartedFinished:
    def test_synthesis_running_after_started(self, tmp_path):
        mark_synthesis_started(
            project_id="P-test",
            context_digest="digest-1",
            memory_dir=tmp_path,
        )
        snap = memory_worker_status()
        assert snap.synthesis_running is True
        assert snap.is_running is False  # Memory workers not running

    def test_synthesis_not_running_after_finished(self, tmp_path):
        mark_synthesis_started(
            project_id="P-test",
            context_digest="digest-1",
            memory_dir=tmp_path,
        )
        mark_synthesis_finished(project_id="P-test", context_digest="digest-1")
        snap = memory_worker_status()
        assert snap.synthesis_running is False

    def test_multiple_synthesis_contexts_tracked(self, tmp_path):
        mark_synthesis_started(
            project_id="P-test",
            context_digest="digest-1",
            memory_dir=tmp_path,
        )
        mark_synthesis_started(
            project_id="P-test",
            context_digest="digest-2",
            memory_dir=tmp_path,
        )
        snap = memory_worker_status()
        assert snap.synthesis_running is True

        mark_synthesis_finished(project_id="P-test", context_digest="digest-1")
        snap = memory_worker_status()
        assert snap.synthesis_running is True  # Still one active

        mark_synthesis_finished(project_id="P-test", context_digest="digest-2")
        snap = memory_worker_status()
        assert snap.synthesis_running is False

    def test_finish_nonexistent_context_is_noop(self):
        # Should not raise
        mark_synthesis_finished(
            project_id="P-unknown", context_digest="no-such-digest"
        )
        snap = memory_worker_status()
        assert snap.synthesis_running is False

    def test_synthesis_started_with_before_outputs(self, tmp_path):
        before = snapshot_memory_outputs(tmp_path)
        mark_synthesis_started(
            project_id="P-test",
            context_digest="digest-with-before",
            memory_dir=tmp_path,
            before_outputs=before,
        )
        snap = memory_worker_status()
        assert snap.synthesis_running is True


# ---------------------------------------------------------------------------
# Knowledge count tracking via synthesis finish
# ---------------------------------------------------------------------------


class TestKnowledgeCountTracking:
    def _write_knowledge_file(
        self,
        memory_dir: Path,
        *,
        knowledge_id: str = "K-abc1234567890123",
        status: str = "active",
    ) -> Path:
        """Write a minimal knowledge markdown file for delta tracking."""
        from EvoScientist.memory.observations import record_observation_file
        from EvoScientist.memory.knowledge import record_knowledge_file
        from EvoScientist.memory.types import (
            MemoryScope,
            MemorySourceType,
            MemoryType,
        )

        # Create a supporting observation first
        obs = record_observation_file(
            memory_dir=memory_dir,
            project_id="P-test",
            memory_type=MemoryType.SEMANTIC,
            summary="Test obs for knowledge",
            observation="Test finding.",
            why_it_matters="Future reference.",
            scope=MemoryScope.GLOBAL,
            source_type=MemorySourceType.SUBAGENT,
            source_session_id="t-1",
            source_agent="test",
        )
        k = record_knowledge_file(
            memory_dir=memory_dir,
            project_id="P-test",
            memory_type=MemoryType.SEMANTIC,
            summary="Test knowledge",
            knowledge="Synthesized finding.",
            supporting_observation_ids=[obs["observation_id"]],
            scope=MemoryScope.GLOBAL,
            knowledge_id=knowledge_id,
            status=status,  # type: ignore[arg-type]
        )
        return memory_dir / k["path"].removeprefix("/memories/")

    def test_knowledge_created_counted_after_synthesis_finish(self, tmp_path):
        memories = tmp_path / "memories"
        before = snapshot_memory_outputs(memories)

        mark_synthesis_started(
            project_id="P-test",
            context_digest="digest-1",
            memory_dir=memories,
            before_outputs=before,
        )

        # Write a new knowledge file after synthesis started
        self._write_knowledge_file(memories)

        mark_synthesis_finished(project_id="P-test", context_digest="digest-1")

        snap = memory_worker_status()
        assert snap.knowledge_created == 1
        assert snap.knowledge_updated == 0
        assert snap.knowledge_archived == 0

    def test_knowledge_archived_counted_correctly(self, tmp_path):
        memories = tmp_path / "memories"
        # Create knowledge BEFORE tracking starts
        self._write_knowledge_file(memories, status="active")
        before = snapshot_memory_outputs(memories)

        mark_synthesis_started(
            project_id="P-test",
            context_digest="digest-archive",
            memory_dir=memories,
            before_outputs=before,
        )

        # Archive the knowledge (rewrite with archived status)
        from EvoScientist.memory.knowledge import archive_knowledge_file

        archive_knowledge_file(
            memory_dir=memories,
            project_id="P-test",
            knowledge_id="K-abc1234567890123",
            reason="Obsolete.",
        )

        mark_synthesis_finished(project_id="P-test", context_digest="digest-archive")

        snap = memory_worker_status()
        assert snap.knowledge_archived == 1


# ---------------------------------------------------------------------------
# snapshot_memory_outputs knowledge tracking
# ---------------------------------------------------------------------------


class TestSnapshotMemoryOutputsKnowledge:
    def test_empty_dir_returns_empty_knowledge(self, tmp_path):
        snap = snapshot_memory_outputs(tmp_path)
        assert snap.knowledge_files == {}

    def test_knowledge_file_appears_in_snapshot(self, tmp_path):
        memories = tmp_path / "memories"
        # Write a minimal knowledge file
        knowledge_path = (
            memories / "knowledge" / "global" / "K-abc1234567890123.md"
        )
        knowledge_path.parent.mkdir(parents=True, exist_ok=True)
        knowledge_path.write_text(
            "---\nid: K-abc1234567890123\nstatus: active\n---\nBody.\n",
            encoding="utf-8",
        )

        snap = snapshot_memory_outputs(memories)
        assert len(snap.knowledge_files) == 1
        key = list(snap.knowledge_files.keys())[0]
        assert "K-abc1234567890123" in key
        assert snap.knowledge_files[key].status == "active"

    def test_knowledge_archived_status_captured(self, tmp_path):
        memories = tmp_path / "memories"
        knowledge_path = (
            memories / "knowledge" / "global" / "K-archived1234567.md"
        )
        knowledge_path.parent.mkdir(parents=True, exist_ok=True)
        knowledge_path.write_text(
            "---\nid: K-archived1234567\nstatus: archived\n---\nBody.\n",
            encoding="utf-8",
        )

        snap = snapshot_memory_outputs(memories)
        key = list(snap.knowledge_files.keys())[0]
        assert snap.knowledge_files[key].status == "archived"


# ---------------------------------------------------------------------------
# memory_worker_observed_outputs with synthesis workers
# ---------------------------------------------------------------------------


class TestMemoryWorkerObservedOutputs:
    def test_synthesis_running_reflected_in_observed(self, tmp_path):
        mark_synthesis_started(
            project_id="P-test",
            context_digest="digest-obs",
            memory_dir=tmp_path,
        )
        snap = memory_worker_observed_outputs()
        assert snap.synthesis_running is True

    def test_synthesis_not_running_when_none_active(self):
        snap = memory_worker_observed_outputs()
        assert snap.synthesis_running is False


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_synthesis_starts_are_safe(self, tmp_path):
        errors = []

        def _start(i: int) -> None:
            try:
                mark_synthesis_started(
                    project_id="P-test",
                    context_digest=f"digest-{i}",
                    memory_dir=tmp_path,
                )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_start, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        snap = memory_worker_status()
        assert snap.synthesis_running is True