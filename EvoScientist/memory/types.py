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
    """Where an observation came from in the agent lifecycle."""

    SUBAGENT = "subagent"
    TURN = "turn"


class ObservationSearchMode(StrEnum):
    """Search modes supported by `search_observations`."""

    RANKED = "ranked"
    REGEX = "regex"


class ObservationRecordResult(TypedDict):
    """Result returned by `record_observation`."""

    observation_id: str
    path: str
    created: bool
    memory_type: MemoryType
    scope: MemoryScope
    project_id: NotRequired[str]


@dataclass(frozen=True)
class ObservationSearchDocument:
    """Parsed observation document ready for search."""

    observation_id: str
    path: str
    memory_type: MemoryType
    scope: MemoryScope
    summary: str
    body: str


class ObservationSearchHit(TypedDict):
    """One result returned by `search_observations`."""

    observation_id: str
    path: str
    memory_type: MemoryType
    scope: MemoryScope
    summary: str
    matches: list[str]
    score: NotRequired[float]


class ObservationReadResult(TypedDict):
    """Full observation document returned by `read_memory`."""

    observation_id: str
    path: str
    memory_type: MemoryType
    scope: MemoryScope
    summary: str
    text: str
