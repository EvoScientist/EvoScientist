"""Tests for EvoScientist.memory.knowledge (all new in this PR)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from EvoScientist.memory.knowledge import (
    KNOWLEDGE_DIR,
    KNOWLEDGE_ID_RE,
    SYNTHESIS_AGENT_NAME,
    SearchMemoryArgs,
    archive_knowledge_file,
    create_read_memory_tool,
    create_search_memory_tool,
    knowledge_search_documents,
    read_knowledge_file,
    read_memory_file,
    record_knowledge_file,
    search_knowledge_files,
    search_memory_files,
)
from EvoScientist.memory.observations import record_observation_file
from EvoScientist.memory.types import (
    KnowledgeStatus,
    MemoryLevelFilter,
    MemoryScope,
    MemorySearchMode,
    MemorySourceType,
    MemoryType,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _record_obs(
    memory_dir,
    *,
    project_id: str = "P-test",
    summary: str = "Test observation summary.",
    observation: str = "The test finding.",
    why_it_matters: str = "Future agents need this.",
    scope: MemoryScope = MemoryScope.GLOBAL,
    memory_type: MemoryType = MemoryType.SEMANTIC,
):
    """Helper to record an observation used as supporting evidence."""
    return record_observation_file(
        memory_dir=memory_dir,
        project_id=project_id,
        memory_type=memory_type,
        summary=summary,
        observation=observation,
        why_it_matters=why_it_matters,
        scope=scope,
        source_type=MemorySourceType.SUBAGENT,
        source_session_id="thread-1",
        source_agent="test-agent",
    )


def _record_knowledge(
    memory_dir,
    *,
    obs_id: str,
    project_id: str = "P-test",
    summary: str = "Synthesized knowledge.",
    knowledge: str = "The synthesized finding.",
    scope: MemoryScope = MemoryScope.GLOBAL,
    memory_type: MemoryType = MemoryType.SEMANTIC,
    status: KnowledgeStatus = KnowledgeStatus.ACTIVE,
):
    """Helper to record a knowledge file with one supporting observation."""
    return record_knowledge_file(
        memory_dir=memory_dir,
        project_id=project_id,
        memory_type=memory_type,
        summary=summary,
        knowledge=knowledge,
        supporting_observation_ids=[obs_id],
        scope=scope,
        status=status,
    )


def _parse_document(path: Path):
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    frontmatter, body = text.removeprefix("---\n").split("\n---\n", 1)
    metadata = yaml.safe_load(frontmatter)
    return metadata, body


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_knowledge_dir(self):
        assert KNOWLEDGE_DIR == "/knowledge"

    def test_synthesis_agent_name(self):
        assert SYNTHESIS_AGENT_NAME == "evomemory-synthesizer"

    def test_knowledge_id_pattern(self):
        assert KNOWLEDGE_ID_RE.fullmatch("K-1234567890abcdef")
        assert not KNOWLEDGE_ID_RE.fullmatch("O-1234567890abcdef")
        assert not KNOWLEDGE_ID_RE.fullmatch("K-1234567890ABCDEF")  # uppercase invalid
        assert not KNOWLEDGE_ID_RE.fullmatch("K-short")


# ---------------------------------------------------------------------------
# record_knowledge_file
# ---------------------------------------------------------------------------


class TestRecordKnowledgeFile:
    def test_creates_global_knowledge_file(self, tmp_path):
        memories = tmp_path / "memories"
        obs = _record_obs(memories)
        result = _record_knowledge(memories, obs_id=obs["observation_id"])

        assert result["created"] is True
        assert result["knowledge_id"].startswith("K-")
        assert KNOWLEDGE_ID_RE.fullmatch(result["knowledge_id"])
        assert result["scope"] == MemoryScope.GLOBAL
        assert result["memory_type"] == MemoryType.SEMANTIC
        assert result["status"] == KnowledgeStatus.ACTIVE
        assert "project_id" not in result

    def test_creates_project_scoped_knowledge_file(self, tmp_path):
        memories = tmp_path / "memories"
        obs = _record_obs(memories, scope=MemoryScope.PROJECT)
        result = _record_knowledge(
            memories,
            obs_id=obs["observation_id"],
            scope=MemoryScope.PROJECT,
        )

        assert result["created"] is True
        assert result["scope"] == MemoryScope.PROJECT
        assert result["project_id"] == "P-test"
        assert "projects/P-test" in result["path"]

    def test_update_existing_returns_created_false(self, tmp_path):
        memories = tmp_path / "memories"
        obs = _record_obs(memories)
        first = _record_knowledge(memories, obs_id=obs["observation_id"])
        second = _record_knowledge(
            memories,
            obs_id=obs["observation_id"],
        )

        # Same content → same id; file exists → created=False
        assert second["knowledge_id"] == first["knowledge_id"]
        assert second["created"] is False

    def test_raises_on_empty_summary(self, tmp_path):
        memories = tmp_path / "memories"
        obs = _record_obs(memories)
        with pytest.raises(ValueError, match="summary"):
            record_knowledge_file(
                memory_dir=memories,
                project_id="P-test",
                memory_type=MemoryType.SEMANTIC,
                summary="",
                knowledge="valid knowledge",
                supporting_observation_ids=[obs["observation_id"]],
                scope=MemoryScope.GLOBAL,
            )

    def test_raises_on_empty_knowledge(self, tmp_path):
        memories = tmp_path / "memories"
        obs = _record_obs(memories)
        with pytest.raises(ValueError, match="knowledge"):
            record_knowledge_file(
                memory_dir=memories,
                project_id="P-test",
                memory_type=MemoryType.SEMANTIC,
                summary="Valid summary",
                knowledge="",
                supporting_observation_ids=[obs["observation_id"]],
                scope=MemoryScope.GLOBAL,
            )

    def test_raises_on_no_supporting_observations(self, tmp_path):
        memories = tmp_path / "memories"
        with pytest.raises(ValueError):
            record_knowledge_file(
                memory_dir=memories,
                project_id="P-test",
                memory_type=MemoryType.SEMANTIC,
                summary="Valid summary",
                knowledge="Valid knowledge",
                supporting_observation_ids=[],
                scope=MemoryScope.GLOBAL,
            )

    def test_raises_on_missing_supporting_observation(self, tmp_path):
        memories = tmp_path / "memories"
        with pytest.raises(ValueError, match="do not exist"):
            record_knowledge_file(
                memory_dir=memories,
                project_id="P-test",
                memory_type=MemoryType.SEMANTIC,
                summary="Valid summary",
                knowledge="Valid knowledge",
                supporting_observation_ids=["O-nonexistent1234567"],
                scope=MemoryScope.GLOBAL,
            )

    def test_archived_skips_observation_validation(self, tmp_path):
        memories = tmp_path / "memories"
        # No observations exist — should work because status=ARCHIVED
        result = record_knowledge_file(
            memory_dir=memories,
            project_id="P-test",
            memory_type=MemoryType.SEMANTIC,
            summary="Archived knowledge",
            knowledge="This was once valid.",
            supporting_observation_ids=["O-nonexistent1234567"],
            scope=MemoryScope.GLOBAL,
            status=KnowledgeStatus.ARCHIVED,
        )
        assert result["status"] == KnowledgeStatus.ARCHIVED

    def test_explicit_knowledge_id_used(self, tmp_path):
        memories = tmp_path / "memories"
        obs = _record_obs(memories)
        result = record_knowledge_file(
            memory_dir=memories,
            project_id="P-test",
            memory_type=MemoryType.SEMANTIC,
            summary="Explicit ID knowledge",
            knowledge="Content here.",
            supporting_observation_ids=[obs["observation_id"]],
            scope=MemoryScope.GLOBAL,
            knowledge_id="K-cafebabe12345678",
        )
        assert result["knowledge_id"] == "K-cafebabe12345678"

    def test_invalid_explicit_knowledge_id_raises(self, tmp_path):
        memories = tmp_path / "memories"
        obs = _record_obs(memories)
        with pytest.raises(ValueError, match="knowledge_id"):
            record_knowledge_file(
                memory_dir=memories,
                project_id="P-test",
                memory_type=MemoryType.SEMANTIC,
                summary="Bad ID knowledge",
                knowledge="Content here.",
                supporting_observation_ids=[obs["observation_id"]],
                scope=MemoryScope.GLOBAL,
                knowledge_id="INVALID-ID",
            )

    def test_file_content_has_correct_frontmatter(self, tmp_path):
        memories = tmp_path / "memories"
        obs = _record_obs(memories)
        result = record_knowledge_file(
            memory_dir=memories,
            project_id="P-test",
            memory_type=MemoryType.PROCEDURAL,
            summary="Frontmatter test knowledge",
            knowledge="Run this command: pytest tests/",
            supporting_observation_ids=[obs["observation_id"]],
            scope=MemoryScope.GLOBAL,
            when_to_use="Use before running full suite.",
        )
        # Find the file and check its content
        path = memories / result["path"].removeprefix("/memories/")
        metadata, body = _parse_document(path)

        assert metadata["id"] == result["knowledge_id"]
        assert metadata["summary"] == "Frontmatter test knowledge"
        assert metadata["memory_type"] == "procedural"
        assert metadata["scope"] == "global"
        assert metadata["status"] == "active"
        assert obs["observation_id"] in metadata["supporting_observation_ids"]
        assert "## Knowledge" in body
        assert "pytest tests/" in body
        assert "## When To Use" in body
        assert "Use before running full suite." in body

    def test_dedupes_supporting_observations(self, tmp_path):
        memories = tmp_path / "memories"
        obs = _record_obs(memories)
        result = record_knowledge_file(
            memory_dir=memories,
            project_id="P-test",
            memory_type=MemoryType.SEMANTIC,
            summary="Deduped observations",
            knowledge="Finding with deduped support.",
            supporting_observation_ids=[
                obs["observation_id"],
                obs["observation_id"],
            ],
            scope=MemoryScope.GLOBAL,
        )
        path = memories / result["path"].removeprefix("/memories/")
        metadata, _ = _parse_document(path)
        # Should only appear once
        assert metadata["supporting_observation_ids"].count(obs["observation_id"]) == 1

    def test_created_at_preserved_on_update(self, tmp_path):
        memories = tmp_path / "memories"
        obs = _record_obs(memories)
        first = record_knowledge_file(
            memory_dir=memories,
            project_id="P-test",
            memory_type=MemoryType.SEMANTIC,
            summary="Update test knowledge",
            knowledge="First version.",
            supporting_observation_ids=[obs["observation_id"]],
            scope=MemoryScope.GLOBAL,
        )
        path = memories / first["path"].removeprefix("/memories/")
        first_meta, _ = _parse_document(path)

        # Update with same id
        record_knowledge_file(
            memory_dir=memories,
            project_id="P-test",
            memory_type=MemoryType.SEMANTIC,
            summary="Update test knowledge",
            knowledge="Second version.",
            supporting_observation_ids=[obs["observation_id"]],
            scope=MemoryScope.GLOBAL,
            knowledge_id=first["knowledge_id"],
        )
        second_meta, _ = _parse_document(path)
        # created_at should be preserved
        assert second_meta["created_at"] == first_meta["created_at"]


# ---------------------------------------------------------------------------
# read_knowledge_file
# ---------------------------------------------------------------------------


class TestReadKnowledgeFile:
    def test_reads_existing_knowledge_by_id(self, tmp_path):
        memories = tmp_path / "memories"
        obs = _record_obs(memories)
        result = _record_knowledge(memories, obs_id=obs["observation_id"])

        read = read_knowledge_file(
            memory_dir=memories,
            project_id="P-test",
            knowledge_id=result["knowledge_id"],
        )
        assert read is not None
        assert read["knowledge_id"] == result["knowledge_id"]
        assert read["memory_type"] == MemoryType.SEMANTIC
        assert read["scope"] == MemoryScope.GLOBAL
        assert read["status"] == KnowledgeStatus.ACTIVE
        assert "text" in read
        assert isinstance(read["supporting_observation_ids"], list)

    def test_returns_none_for_missing_id(self, tmp_path):
        memories = tmp_path / "memories"
        result = read_knowledge_file(
            memory_dir=memories,
            project_id="P-test",
            knowledge_id="K-nonexistent1234567",
        )
        assert result is None

    def test_returns_none_for_empty_id(self, tmp_path):
        memories = tmp_path / "memories"
        result = read_knowledge_file(
            memory_dir=memories,
            project_id="P-test",
            knowledge_id="   ",
        )
        assert result is None


# ---------------------------------------------------------------------------
# read_memory_file
# ---------------------------------------------------------------------------


class TestReadMemoryFile:
    def test_reads_knowledge_by_k_prefix(self, tmp_path):
        memories = tmp_path / "memories"
        obs = _record_obs(memories)
        result = _record_knowledge(memories, obs_id=obs["observation_id"])

        read = read_memory_file(
            memory_dir=memories,
            project_id="P-test",
            memory_id=result["knowledge_id"],
        )
        assert read is not None
        assert read.get("knowledge_id") == result["knowledge_id"]

    def test_reads_observation_by_o_prefix(self, tmp_path):
        memories = tmp_path / "memories"
        obs = _record_obs(memories)

        read = read_memory_file(
            memory_dir=memories,
            project_id="P-test",
            memory_id=obs["observation_id"],
        )
        assert read is not None
        assert read.get("observation_id") == obs["observation_id"]

    def test_returns_none_for_empty_id(self, tmp_path):
        memories = tmp_path / "memories"
        assert read_memory_file(memory_dir=memories, project_id="P-test", memory_id="") is None

    def test_returns_none_for_unknown_prefix(self, tmp_path):
        memories = tmp_path / "memories"
        assert (
            read_memory_file(
                memory_dir=memories, project_id="P-test", memory_id="X-unknown"
            )
            is None
        )

    def test_returns_none_for_nonexistent_k_id(self, tmp_path):
        memories = tmp_path / "memories"
        result = read_memory_file(
            memory_dir=memories,
            project_id="P-test",
            memory_id="K-nonexistent1234567",
        )
        assert result is None


# ---------------------------------------------------------------------------
# knowledge_search_documents
# ---------------------------------------------------------------------------


class TestKnowledgeSearchDocuments:
    def test_returns_knowledge_documents(self, tmp_path):
        memories = tmp_path / "memories"
        obs = _record_obs(memories)
        result = _record_knowledge(
            memories,
            obs_id=obs["observation_id"],
            summary="GraphQL resolver finding",
        )

        docs = knowledge_search_documents(
            memory_dir=memories, project_id="P-test"
        )
        assert any(d.knowledge_id == result["knowledge_id"] for d in docs)

    def test_filters_by_status_active(self, tmp_path):
        memories = tmp_path / "memories"
        active_result = record_knowledge_file(
            memory_dir=memories,
            project_id="P-test",
            memory_type=MemoryType.SEMANTIC,
            summary="Active knowledge",
            knowledge="Active content.",
            supporting_observation_ids=["O-dummy"],
            scope=MemoryScope.GLOBAL,
            status=KnowledgeStatus.ARCHIVED,
        )
        active_obs = _record_obs(memories, summary="Active obs")
        active_knowledge = _record_knowledge(
            memories,
            obs_id=active_obs["observation_id"],
            summary="Active knowledge doc",
        )

        active_docs = knowledge_search_documents(
            memory_dir=memories,
            project_id="P-test",
            status=KnowledgeStatus.ACTIVE,
        )
        active_ids = {d.knowledge_id for d in active_docs}
        assert active_knowledge["knowledge_id"] in active_ids
        assert active_result["knowledge_id"] not in active_ids

    def test_filters_by_scope_global(self, tmp_path):
        memories = tmp_path / "memories"
        global_obs = _record_obs(memories, scope=MemoryScope.GLOBAL)
        project_obs = _record_obs(
            memories,
            scope=MemoryScope.PROJECT,
            summary="Project scoped obs",
            observation="Project obs.",
            why_it_matters="Project why.",
        )
        global_k = _record_knowledge(
            memories, obs_id=global_obs["observation_id"], scope=MemoryScope.GLOBAL
        )
        project_k = _record_knowledge(
            memories,
            obs_id=project_obs["observation_id"],
            scope=MemoryScope.PROJECT,
            summary="Project knowledge",
        )

        global_docs = knowledge_search_documents(
            memory_dir=memories,
            project_id="P-test",
            scope=MemoryScope.GLOBAL,
        )
        global_ids = {d.knowledge_id for d in global_docs}
        assert global_k["knowledge_id"] in global_ids
        assert project_k["knowledge_id"] not in global_ids

    def test_filters_by_memory_type(self, tmp_path):
        memories = tmp_path / "memories"
        sem_obs = _record_obs(memories, memory_type=MemoryType.SEMANTIC)
        proc_obs = _record_obs(
            memories,
            memory_type=MemoryType.PROCEDURAL,
            summary="Proc obs",
            observation="Proc finding.",
            why_it_matters="Proc why.",
        )
        sem_k = _record_knowledge(
            memories,
            obs_id=sem_obs["observation_id"],
            memory_type=MemoryType.SEMANTIC,
        )
        proc_k = _record_knowledge(
            memories,
            obs_id=proc_obs["observation_id"],
            memory_type=MemoryType.PROCEDURAL,
            summary="Procedural knowledge",
        )

        sem_docs = knowledge_search_documents(
            memory_dir=memories,
            project_id="P-test",
            memory_type=MemoryType.SEMANTIC,
        )
        sem_ids = {d.knowledge_id for d in sem_docs}
        assert sem_k["knowledge_id"] in sem_ids
        assert proc_k["knowledge_id"] not in sem_ids


# ---------------------------------------------------------------------------
# search_knowledge_files
# ---------------------------------------------------------------------------


class TestSearchKnowledgeFiles:
    def test_ranked_search_returns_hits(self, tmp_path):
        memories = tmp_path / "memories"
        obs = _record_obs(
            memories, summary="asyncio event loop closes unexpectedly"
        )
        _record_knowledge(
            memories,
            obs_id=obs["observation_id"],
            summary="asyncio event loop behavior",
            knowledge="Event loops close on finalization.",
        )

        hits = search_knowledge_files(
            memory_dir=memories,
            project_id="P-test",
            query="asyncio event loop",
            limit=5,
        )
        assert len(hits) >= 1
        assert all(h["level"] == "knowledge" for h in hits)

    def test_no_match_returns_empty(self, tmp_path):
        memories = tmp_path / "memories"
        obs = _record_obs(memories)
        _record_knowledge(memories, obs_id=obs["observation_id"])

        hits = search_knowledge_files(
            memory_dir=memories,
            project_id="P-test",
            query="completely_irrelevant_xyz",
        )
        assert hits == []

    def test_empty_query_returns_empty(self, tmp_path):
        memories = tmp_path / "memories"
        obs = _record_obs(memories)
        _record_knowledge(memories, obs_id=obs["observation_id"])

        hits = search_knowledge_files(
            memory_dir=memories, project_id="P-test", query="   "
        )
        assert hits == []

    def test_archived_excluded_by_default(self, tmp_path):
        memories = tmp_path / "memories"
        archived_k = record_knowledge_file(
            memory_dir=memories,
            project_id="P-test",
            memory_type=MemoryType.SEMANTIC,
            summary="archived semantic knowledge here",
            knowledge="Archived finding.",
            supporting_observation_ids=["O-dummy"],
            scope=MemoryScope.GLOBAL,
            status=KnowledgeStatus.ARCHIVED,
        )

        hits = search_knowledge_files(
            memory_dir=memories,
            project_id="P-test",
            query="archived semantic knowledge",
        )
        archived_ids = {h["memory_id"] for h in hits}
        assert archived_k["knowledge_id"] not in archived_ids

    def test_archived_included_when_flag_set(self, tmp_path):
        memories = tmp_path / "memories"
        archived_k = record_knowledge_file(
            memory_dir=memories,
            project_id="P-test",
            memory_type=MemoryType.SEMANTIC,
            summary="archived semantic knowledge here",
            knowledge="Archived finding.",
            supporting_observation_ids=["O-dummy"],
            scope=MemoryScope.GLOBAL,
            status=KnowledgeStatus.ARCHIVED,
        )

        hits = search_knowledge_files(
            memory_dir=memories,
            project_id="P-test",
            query="archived semantic knowledge",
            include_archived=True,
        )
        archived_ids = {h["memory_id"] for h in hits}
        assert archived_k["knowledge_id"] in archived_ids


# ---------------------------------------------------------------------------
# search_memory_files (combined knowledge + observation search)
# ---------------------------------------------------------------------------


class TestSearchMemoryFiles:
    def test_any_level_returns_both(self, tmp_path):
        memories = tmp_path / "memories"
        obs = _record_obs(memories, summary="python testing pytest")
        k = _record_knowledge(
            memories,
            obs_id=obs["observation_id"],
            summary="python testing pattern",
            knowledge="Use pytest for focused tests.",
        )

        hits = search_memory_files(
            memory_dir=memories,
            project_id="P-test",
            query="python testing",
            memory_level=MemoryLevelFilter.ANY,
            include_covered_observations=True,
        )
        ids = {h["memory_id"] for h in hits}
        # Both the knowledge and observation may appear
        assert k["knowledge_id"] in ids or obs["observation_id"] in ids

    def test_knowledge_level_only_returns_knowledge(self, tmp_path):
        memories = tmp_path / "memories"
        obs = _record_obs(memories, summary="asyncio knowledge finding")
        k = _record_knowledge(
            memories,
            obs_id=obs["observation_id"],
            summary="asyncio knowledge finding",
        )

        hits = search_memory_files(
            memory_dir=memories,
            project_id="P-test",
            query="asyncio knowledge finding",
            memory_level=MemoryLevelFilter.KNOWLEDGE,
        )
        ids = {h["memory_id"] for h in hits}
        assert k["knowledge_id"] in ids
        assert obs["observation_id"] not in ids

    def test_observation_level_only_returns_observations(self, tmp_path):
        memories = tmp_path / "memories"
        obs = _record_obs(memories, summary="asyncio observation finding")
        k = _record_knowledge(
            memories,
            obs_id=obs["observation_id"],
            summary="asyncio observation finding",
        )

        hits = search_memory_files(
            memory_dir=memories,
            project_id="P-test",
            query="asyncio observation finding",
            memory_level=MemoryLevelFilter.OBSERVATION,
        )
        ids = {h["memory_id"] for h in hits}
        assert obs["observation_id"] in ids
        assert k["knowledge_id"] not in ids

    def test_any_level_excludes_covered_observations_by_default(self, tmp_path):
        """Covered observations (cited by active knowledge) are hidden by default."""
        memories = tmp_path / "memories"
        obs = _record_obs(memories, summary="covered observation finding")
        k = _record_knowledge(
            memories,
            obs_id=obs["observation_id"],
            summary="covered observation finding",
        )

        hits = search_memory_files(
            memory_dir=memories,
            project_id="P-test",
            query="covered observation finding",
            memory_level=MemoryLevelFilter.ANY,
            include_covered_observations=False,
        )
        ids = {h["memory_id"] for h in hits}
        # observation should be excluded since it's covered by active knowledge
        assert obs["observation_id"] not in ids

    def test_empty_query_returns_empty(self, tmp_path):
        memories = tmp_path / "memories"
        obs = _record_obs(memories)
        _record_knowledge(memories, obs_id=obs["observation_id"])

        hits = search_memory_files(
            memory_dir=memories, project_id="P-test", query=""
        )
        assert hits == []

    def test_invalid_memory_level_returns_empty(self, tmp_path):
        memories = tmp_path / "memories"
        hits = search_memory_files(
            memory_dir=memories,
            project_id="P-test",
            query="test",
            memory_level="completely_invalid_level",
        )
        assert hits == []


# ---------------------------------------------------------------------------
# archive_knowledge_file
# ---------------------------------------------------------------------------


class TestArchiveKnowledgeFile:
    def test_archives_active_knowledge(self, tmp_path):
        memories = tmp_path / "memories"
        obs = _record_obs(memories)
        k = _record_knowledge(memories, obs_id=obs["observation_id"])

        result = archive_knowledge_file(
            memory_dir=memories,
            project_id="P-test",
            knowledge_id=k["knowledge_id"],
            reason="Test archival reason.",
        )

        assert result is not None
        assert result["knowledge_id"] == k["knowledge_id"]
        assert result["status"] == KnowledgeStatus.ARCHIVED

    def test_returns_none_for_nonexistent_knowledge(self, tmp_path):
        memories = tmp_path / "memories"
        result = archive_knowledge_file(
            memory_dir=memories,
            project_id="P-test",
            knowledge_id="K-nonexistent1234567",
            reason="Archiving nonexistent.",
        )
        assert result is None

    def test_archived_file_has_archive_fields(self, tmp_path):
        memories = tmp_path / "memories"
        obs = _record_obs(memories)
        k = _record_knowledge(memories, obs_id=obs["observation_id"])

        archive_knowledge_file(
            memory_dir=memories,
            project_id="P-test",
            knowledge_id=k["knowledge_id"],
            reason="Obsolete after new findings.",
        )

        # Verify the file content was updated
        read = read_knowledge_file(
            memory_dir=memories,
            project_id="P-test",
            knowledge_id=k["knowledge_id"],
        )
        assert read is not None
        assert read["status"] == KnowledgeStatus.ARCHIVED
        assert "Obsolete after new findings." in read["text"]

    def test_archive_preserves_original_content(self, tmp_path):
        memories = tmp_path / "memories"
        obs = _record_obs(memories)
        k = record_knowledge_file(
            memory_dir=memories,
            project_id="P-test",
            memory_type=MemoryType.PROCEDURAL,
            summary="Original summary",
            knowledge="Original procedural content.",
            supporting_observation_ids=[obs["observation_id"]],
            scope=MemoryScope.GLOBAL,
        )

        archive_knowledge_file(
            memory_dir=memories,
            project_id="P-test",
            knowledge_id=k["knowledge_id"],
            reason="Replaced.",
        )

        read = read_knowledge_file(
            memory_dir=memories,
            project_id="P-test",
            knowledge_id=k["knowledge_id"],
        )
        assert read is not None
        assert "Original procedural content." in read["text"]


# ---------------------------------------------------------------------------
# create_search_memory_tool
# ---------------------------------------------------------------------------


class TestCreateSearchMemoryTool:
    def test_returns_tool_with_correct_name(self, tmp_path):
        memories = tmp_path / "memories"
        tool = create_search_memory_tool(memory_dir=memories, project_id="P-test")
        assert tool.name == "search_memory"

    def test_tool_returns_json_results(self, tmp_path):
        memories = tmp_path / "memories"
        obs = _record_obs(memories, summary="pytest testing tool search")
        _record_knowledge(
            memories,
            obs_id=obs["observation_id"],
            summary="pytest testing tool search",
        )

        tool = create_search_memory_tool(memory_dir=memories, project_id="P-test")
        payload = json.loads(
            tool.run({"query": "pytest testing tool search", "limit": 5})
        )
        assert "results" in payload
        assert isinstance(payload["results"], list)

    def test_tool_filters_by_memory_level(self, tmp_path):
        memories = tmp_path / "memories"
        obs = _record_obs(memories, summary="memory level filter test")
        k = _record_knowledge(
            memories,
            obs_id=obs["observation_id"],
            summary="memory level filter test",
        )

        tool = create_search_memory_tool(memory_dir=memories, project_id="P-test")
        payload = json.loads(
            tool.run(
                {
                    "query": "memory level filter test",
                    "memory_level": "knowledge",
                    "limit": 5,
                }
            )
        )
        ids = {r["memory_id"] for r in payload["results"]}
        assert k["knowledge_id"] in ids
        assert obs["observation_id"] not in ids


# ---------------------------------------------------------------------------
# create_read_memory_tool
# ---------------------------------------------------------------------------


class TestCreateReadMemoryTool:
    def test_returns_tool_with_correct_name(self, tmp_path):
        memories = tmp_path / "memories"
        tool = create_read_memory_tool(memory_dir=memories, project_id="P-test")
        assert tool.name == "read_memory"

    def test_reads_knowledge_by_id(self, tmp_path):
        memories = tmp_path / "memories"
        obs = _record_obs(memories, summary="read memory tool test")
        k = _record_knowledge(
            memories,
            obs_id=obs["observation_id"],
            summary="read memory tool test",
        )

        tool = create_read_memory_tool(memory_dir=memories, project_id="P-test")
        payload = json.loads(tool.run({"memory_id": k["knowledge_id"]}))
        assert "text" in payload

    def test_reads_observation_by_id(self, tmp_path):
        memories = tmp_path / "memories"
        obs = _record_obs(memories)

        tool = create_read_memory_tool(memory_dir=memories, project_id="P-test")
        payload = json.loads(tool.run({"memory_id": obs["observation_id"]}))
        assert "text" in payload

    def test_returns_error_for_nonexistent_id(self, tmp_path):
        memories = tmp_path / "memories"
        tool = create_read_memory_tool(memory_dir=memories, project_id="P-test")
        payload = json.loads(tool.run({"memory_id": "K-nonexistent1234567"}))
        assert "error" in payload

    def test_returns_error_for_unknown_prefix(self, tmp_path):
        memories = tmp_path / "memories"
        tool = create_read_memory_tool(memory_dir=memories, project_id="P-test")
        payload = json.loads(tool.run({"memory_id": "X-unknown"}))
        assert "error" in payload