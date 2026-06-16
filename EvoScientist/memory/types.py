"""Shared types for EvoMemory storage and search."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import NotRequired, TypedDict


class MemoryType(StrEnum):
    """Kinds of reusable memory an observation can represent."""

    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    EPISODIC = "episodic"


class MemoryScope(StrEnum):
    """Whether an observation is global or tied to the active project."""

    GLOBAL = "global"
    PROJECT = "project"


class MemoryLevel(StrEnum):
    """Storage levels searched by `search_memory`."""

    KNOWLEDGE = "knowledge"
    OBSERVATION = "observation"


class MemoryLevelFilter(StrEnum):
    """Tool-facing memory-level filter, including the aggregate `any` value."""

    ANY = "any"
    KNOWLEDGE = MemoryLevel.KNOWLEDGE.value
    OBSERVATION = MemoryLevel.OBSERVATION.value


class MemorySourceType(StrEnum):
    """Where a memory came from in the agent lifecycle."""

    SUBAGENT = "subagent"
    SYNTHESIS = "synthesis"
    TURN = "turn"


class MemorySearchMode(StrEnum):
    """Search modes supported by EvoMemory search tools."""

    RANKED = "ranked"
    REGEX = "regex"


class KnowledgeStatus(StrEnum):
    """Lifecycle status for synthesized knowledge records."""

    ACTIVE = "active"
    ARCHIVED = "archived"


class MemoryRecordResult(TypedDict):
    """Common fields returned when a memory file is written."""

    path: str
    created: bool
    memory_type: MemoryType
    scope: MemoryScope
    project_id: NotRequired[str]


class ObservationRecordResult(MemoryRecordResult):
    """Result returned by `record_observation`."""

    observation_id: str


class KnowledgeRecordResult(MemoryRecordResult):
    """Result returned when a knowledge file is created or updated."""

    knowledge_id: str
    status: KnowledgeStatus


class MemorySearchHitExtra(TypedDict, total=False):
    """Level-specific fields included on search hits when available."""

    status: KnowledgeStatus
    supporting_observation_ids: list[str]


@dataclass(frozen=True)
class MemorySearchDocument(ABC):
    """Common parsed memory document shape used by search internals."""

    path: str
    memory_type: MemoryType
    scope: MemoryScope
    summary: str
    body: str

    @property
    @abstractmethod
    def memory_id(self) -> str:
        """Canonical K-* or O-* id for search output."""

    @property
    @abstractmethod
    def memory_level(self) -> MemoryLevel:
        """Whether this document is synthesized knowledge or raw observation."""

    @property
    def search_metadata(self) -> tuple[str, ...]:
        """Metadata terms included in ranked and regex search haystacks."""
        return (self.memory_type.value, self.scope.value, self.memory_level.value)

    @property
    def search_hit_extra(self) -> MemorySearchHitExtra:
        """Level-specific fields to expose on the canonical search hit."""
        return {}


@dataclass(frozen=True)
class ObservationSearchDocument(MemorySearchDocument):
    """Parsed observation document ready for search."""

    observation_id: str

    @property
    def memory_id(self) -> str:
        return self.observation_id

    @property
    def memory_level(self) -> MemoryLevel:
        return MemoryLevel.OBSERVATION


@dataclass(frozen=True)
class KnowledgeSearchDocument(MemorySearchDocument):
    """Parsed knowledge document ready for search."""

    knowledge_id: str
    status: KnowledgeStatus
    supporting_observation_ids: tuple[str, ...]

    @property
    def memory_id(self) -> str:
        return self.knowledge_id

    @property
    def memory_level(self) -> MemoryLevel:
        return MemoryLevel.KNOWLEDGE

    @property
    def search_metadata(self) -> tuple[str, ...]:
        return (
            *super().search_metadata,
            self.status.value,
            *self.supporting_observation_ids,
        )

    @property
    def search_hit_extra(self) -> MemorySearchHitExtra:
        return {
            "status": self.status,
            "supporting_observation_ids": list(self.supporting_observation_ids),
        }


class MemorySearchHit(TypedDict):
    """One result returned by combined memory search."""

    memory_id: str
    level: MemoryLevel
    path: str
    memory_type: MemoryType
    scope: MemoryScope
    summary: str
    matches: list[str]
    score: NotRequired[float]
    status: NotRequired[KnowledgeStatus]
    supporting_observation_ids: NotRequired[list[str]]


class MemoryReadResult(TypedDict):
    """Common fields returned when reading a memory document."""

    path: str
    memory_type: MemoryType
    scope: MemoryScope
    summary: str
    text: str


class ObservationReadResult(MemoryReadResult):
    """Full observation document returned by `read_memory`."""

    observation_id: str


class KnowledgeReadResult(MemoryReadResult):
    """Full knowledge document returned by `read_memory`."""

    knowledge_id: str
    status: KnowledgeStatus
    supporting_observation_ids: list[str]
