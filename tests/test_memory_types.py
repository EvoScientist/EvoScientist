"""Tests for EvoScientist.memory.types (changed/added types in this PR)."""

from __future__ import annotations

import pytest

from EvoScientist.memory.types import (
    KnowledgeReadResult,
    KnowledgeRecordResult,
    KnowledgeSearchDocument,
    KnowledgeStatus,
    MemoryLevel,
    MemoryLevelFilter,
    MemoryScope,
    MemorySearchDocument,
    MemorySearchHit,
    MemorySearchMode,
    MemorySourceType,
    MemoryType,
    ObservationRecordResult,
    ObservationSearchDocument,
)


# ---------------------------------------------------------------------------
# MemoryLevel
# ---------------------------------------------------------------------------


class TestMemoryLevel:
    def test_values(self):
        assert MemoryLevel.KNOWLEDGE == "knowledge"
        assert MemoryLevel.OBSERVATION == "observation"

    def test_is_str_enum(self):
        assert isinstance(MemoryLevel.KNOWLEDGE, str)


# ---------------------------------------------------------------------------
# MemoryLevelFilter
# ---------------------------------------------------------------------------


class TestMemoryLevelFilter:
    def test_any_value(self):
        assert MemoryLevelFilter.ANY == "any"

    def test_knowledge_matches_memory_level(self):
        assert MemoryLevelFilter.KNOWLEDGE == MemoryLevel.KNOWLEDGE.value

    def test_observation_matches_memory_level(self):
        assert MemoryLevelFilter.OBSERVATION == MemoryLevel.OBSERVATION.value

    def test_all_values_exist(self):
        values = {f.value for f in MemoryLevelFilter}
        assert values == {"any", "knowledge", "observation"}


# ---------------------------------------------------------------------------
# KnowledgeStatus
# ---------------------------------------------------------------------------


class TestKnowledgeStatus:
    def test_active_value(self):
        assert KnowledgeStatus.ACTIVE == "active"

    def test_archived_value(self):
        assert KnowledgeStatus.ARCHIVED == "archived"

    def test_is_str_enum(self):
        assert isinstance(KnowledgeStatus.ACTIVE, str)


# ---------------------------------------------------------------------------
# MemorySourceType (extended with SYNTHESIS)
# ---------------------------------------------------------------------------


class TestMemorySourceType:
    def test_synthesis_value(self):
        assert MemorySourceType.SYNTHESIS == "synthesis"

    def test_existing_values_unchanged(self):
        assert MemorySourceType.SUBAGENT == "subagent"
        assert MemorySourceType.TURN == "turn"


# ---------------------------------------------------------------------------
# MemorySearchMode (renamed from ObservationSearchMode)
# ---------------------------------------------------------------------------


class TestMemorySearchMode:
    def test_ranked_value(self):
        assert MemorySearchMode.RANKED == "ranked"

    def test_regex_value(self):
        assert MemorySearchMode.REGEX == "regex"


# ---------------------------------------------------------------------------
# ObservationSearchDocument (now a MemorySearchDocument subclass)
# ---------------------------------------------------------------------------


class TestObservationSearchDocument:
    def _make(self, **kwargs):
        defaults = dict(
            observation_id="O-abc1234567890123",
            path="/memories/observations/global/O-abc1234567890123.md",
            memory_type=MemoryType.SEMANTIC,
            scope=MemoryScope.GLOBAL,
            summary="Test summary",
            body="Test body content.",
        )
        defaults.update(kwargs)
        return ObservationSearchDocument(**defaults)

    def test_memory_id_returns_observation_id(self):
        doc = self._make(observation_id="O-deadbeef12345678")
        assert doc.memory_id == "O-deadbeef12345678"

    def test_memory_level_is_observation(self):
        doc = self._make()
        assert doc.memory_level == MemoryLevel.OBSERVATION

    def test_search_metadata_contains_type_scope_level(self):
        doc = self._make(memory_type=MemoryType.PROCEDURAL, scope=MemoryScope.PROJECT)
        meta = doc.search_metadata
        assert "procedural" in meta
        assert "project" in meta
        assert "observation" in meta

    def test_search_hit_extra_is_empty(self):
        doc = self._make()
        assert doc.search_hit_extra == {}

    def test_is_frozen_dataclass(self):
        doc = self._make()
        with pytest.raises((AttributeError, TypeError)):
            doc.summary = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# KnowledgeSearchDocument (new in this PR)
# ---------------------------------------------------------------------------


class TestKnowledgeSearchDocument:
    def _make(self, **kwargs):
        defaults = dict(
            knowledge_id="K-cafebabe12345678",
            path="/memories/knowledge/global/K-cafebabe12345678.md",
            memory_type=MemoryType.SEMANTIC,
            scope=MemoryScope.GLOBAL,
            summary="Synthesized knowledge summary",
            body="Detailed knowledge body.",
            status=KnowledgeStatus.ACTIVE,
            supporting_observation_ids=("O-obs1", "O-obs2"),
        )
        defaults.update(kwargs)
        return KnowledgeSearchDocument(**defaults)

    def test_memory_id_returns_knowledge_id(self):
        doc = self._make(knowledge_id="K-abc1234567890123")
        assert doc.memory_id == "K-abc1234567890123"

    def test_memory_level_is_knowledge(self):
        doc = self._make()
        assert doc.memory_level == MemoryLevel.KNOWLEDGE

    def test_search_metadata_includes_status(self):
        doc = self._make(status=KnowledgeStatus.ARCHIVED)
        assert "archived" in doc.search_metadata

    def test_search_metadata_includes_supporting_obs_ids(self):
        doc = self._make(supporting_observation_ids=("O-a", "O-b"))
        meta = doc.search_metadata
        assert "O-a" in meta
        assert "O-b" in meta

    def test_search_metadata_includes_base_fields(self):
        doc = self._make(
            memory_type=MemoryType.EPISODIC,
            scope=MemoryScope.PROJECT,
        )
        meta = doc.search_metadata
        assert "episodic" in meta
        assert "project" in meta
        assert "knowledge" in meta

    def test_search_hit_extra_has_status(self):
        doc = self._make(status=KnowledgeStatus.ACTIVE)
        extra = doc.search_hit_extra
        assert extra["status"] == KnowledgeStatus.ACTIVE

    def test_search_hit_extra_has_supporting_obs_ids(self):
        doc = self._make(supporting_observation_ids=("O-1", "O-2"))
        extra = doc.search_hit_extra
        assert extra["supporting_observation_ids"] == ["O-1", "O-2"]

    def test_is_frozen(self):
        doc = self._make()
        with pytest.raises((AttributeError, TypeError)):
            doc.summary = "mutated"  # type: ignore[misc]

    def test_is_subclass_of_memory_search_document(self):
        doc = self._make()
        assert isinstance(doc, MemorySearchDocument)


# ---------------------------------------------------------------------------
# KnowledgeRecordResult (TypedDict)
# ---------------------------------------------------------------------------


class TestKnowledgeRecordResult:
    def test_required_fields(self):
        result: KnowledgeRecordResult = {
            "knowledge_id": "K-abc",
            "path": "/memories/knowledge/global/K-abc.md",
            "created": True,
            "memory_type": MemoryType.SEMANTIC,
            "scope": MemoryScope.GLOBAL,
            "status": KnowledgeStatus.ACTIVE,
        }
        assert result["knowledge_id"] == "K-abc"
        assert result["status"] == KnowledgeStatus.ACTIVE

    def test_optional_project_id(self):
        result: KnowledgeRecordResult = {
            "knowledge_id": "K-proj",
            "path": "/memories/knowledge/projects/P-x/K-proj.md",
            "created": True,
            "memory_type": MemoryType.PROCEDURAL,
            "scope": MemoryScope.PROJECT,
            "status": KnowledgeStatus.ACTIVE,
            "project_id": "P-x",
        }
        assert result["project_id"] == "P-x"