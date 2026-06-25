"""Shared types for EvoMemory observation storage and search."""

from __future__ import annotations

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


class MemorySourceType(StrEnum):
    """Where a memory observation originated."""

    SUBAGENT = "subagent"
    TURN = "turn"


class ObservationSearchMode(StrEnum):
    """Search modes supported by `search_observations`."""

    RANKED = "ranked"
    REGEX = "regex"


class ObservationRelation(StrEnum):
    """Allowed relationship labels between observations."""

    COMPLEMENTS = "complements"
    CONTRADICTS = "contradicts"
    SUPERSEDES = "supersedes"


class ObservationRecordResult(TypedDict):
    """Result returned by `record_observation`."""

    observation_id: str
    path: str
    created: bool
    memory_type: MemoryType
    scope: MemoryScope
    project_id: NotRequired[str]


class RelatedObservationResult(TypedDict):
    """One resolved observation relationship exposed to memory tools."""

    observation_id: str
    path: str
    memory_type: MemoryType
    scope: MemoryScope
    summary: str
    relation: NotRequired[ObservationRelation]
    reason: NotRequired[str]


@dataclass(frozen=True)
class ObservationSearchDocument:
    """Parsed observation document ready for search."""

    observation_id: str
    path: str
    memory_type: MemoryType
    scope: MemoryScope
    summary: str
    body: str
    text: str
    related_observations: tuple[RelatedObservationResult, ...] = ()


class ObservationSearchHit(TypedDict):
    """One result returned by `search_observations`."""

    observation_id: str
    path: str
    memory_type: MemoryType
    scope: MemoryScope
    summary: str
    matches: list[str]
    related_observations: NotRequired[list[RelatedObservationResult]]
    score: NotRequired[float]


class ObservationReadResult(TypedDict):
    """Full observation document returned by `read_memory`."""

    observation_id: str
    path: str
    memory_type: MemoryType
    scope: MemoryScope
    summary: str
    text: str
    related_observations: NotRequired[list[RelatedObservationResult]]
