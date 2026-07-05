"""Observation graph candidate extraction for AutoSkills."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, deque
from pathlib import Path
from typing import TypedDict

from ..observations import list_observation_documents
from ..types import (
    MemoryScope,
    MemoryType,
    ObservationRelation,
    ObservationSearchDocument,
)
from .proposals import cluster_hashes_by_status, processed_cluster_hashes

MIN_CLUSTER_SIZE = 3
MIN_PROCEDURAL_OBSERVATIONS = 2


class AutoskillCandidateObservation(TypedDict):
    id: str
    memory_type: MemoryType
    scope: MemoryScope
    summary: str
    path: str


class AutoskillCandidateRelation(TypedDict):
    source: str
    target: str
    relation: ObservationRelation
    reason: str


class AutoskillCandidate(TypedDict):
    cluster_hash: str
    observation_ids: list[str]
    observation_count: int
    procedural_count: int
    semantic_count: int
    episodic_count: int
    observations: list[AutoskillCandidateObservation]
    relations: list[AutoskillCandidateRelation]
    existing_pending_proposal: bool
    already_processed: bool


def _stable_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _short_hash(value: object, *, n: int = 16) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()[:n]


def _ordered_pair(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)


def _graph_edges(
    documents: list[ObservationSearchDocument],
) -> tuple[set[tuple[str, str]], list[AutoskillCandidateRelation]]:
    graph_edges: set[tuple[str, str]] = set()
    relation_rows: list[AutoskillCandidateRelation] = []
    document_ids = {document.observation_id for document in documents}
    for document in documents:
        for related in document.related_observations:
            target = str(related["observation_id"])
            if target not in document_ids:
                continue
            relation = related.get("relation", ObservationRelation.COMPLEMENTS)
            relation_value = (
                relation.value
                if isinstance(relation, ObservationRelation)
                else str(relation)
            )
            try:
                normalized_relation = ObservationRelation(relation_value)
            except ValueError:
                continue
            relation_value = normalized_relation.value
            source = document.observation_id
            dest = target
            graph_edges.add(_ordered_pair(source, dest))
            # Cluster connectivity is undirected, but supersedes is meaningful
            # only in its original source-to-target direction.
            if relation_value != ObservationRelation.SUPERSEDES.value:
                source, dest = _ordered_pair(source, dest)
            row: AutoskillCandidateRelation = {
                "source": source,
                "target": dest,
                "relation": normalized_relation,
                "reason": str(related.get("reason", "")),
            }
            relation_rows.append(row)
    return graph_edges, relation_rows


def _dedupe_relation_rows(
    rows: list[AutoskillCandidateRelation],
) -> list[AutoskillCandidateRelation]:
    by_key: dict[tuple[str, str, str, str], AutoskillCandidateRelation] = {}
    for row in rows:
        key = (
            row["source"],
            row["target"],
            row["relation"].value,
            row["reason"],
        )
        by_key[key] = row
    return [by_key[key] for key in sorted(by_key)]


def _components(
    document_ids: set[str],
    edges: set[tuple[str, str]],
) -> list[set[str]]:
    adjacency: dict[str, set[str]] = {
        observation_id: set() for observation_id in document_ids
    }
    for source, target in edges:
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)

    seen: set[str] = set()
    components: list[set[str]] = []
    for observation_id in sorted(document_ids):
        if observation_id in seen:
            continue
        queue = deque([observation_id])
        seen.add(observation_id)
        component: set[str] = set()
        while queue:
            current = queue.popleft()
            component.add(current)
            for neighbor in sorted(adjacency.get(current, ())):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def autoskill_candidates(
    *,
    memory_dir: str | Path,
    project_id: str,
    workspace_dir: str | Path | None = None,
) -> list[AutoskillCandidate]:
    """Return observation graph components worth showing to the AutoSkills agent."""
    documents = list_observation_documents(memory_dir=memory_dir, project_id=project_id)
    documents_by_id = {document.observation_id: document for document in documents}
    edges, relation_rows = _graph_edges(documents)
    relation_rows = _dedupe_relation_rows(relation_rows)
    proposed_hashes = cluster_hashes_by_status(
        memory_dir,
        workspace_dir=workspace_dir,
    )
    processed_hashes = processed_cluster_hashes(memory_dir)
    candidates: list[AutoskillCandidate] = []

    for component in _components(set(documents_by_id), edges):
        component_docs = [
            documents_by_id[observation_id] for observation_id in sorted(component)
        ]
        memory_type_counts = Counter(
            document.memory_type for document in component_docs
        )
        procedural_count = memory_type_counts[MemoryType.PROCEDURAL]
        if len(component_docs) < MIN_CLUSTER_SIZE:
            continue
        if procedural_count < MIN_PROCEDURAL_OBSERVATIONS:
            continue
        component_relations = [
            row
            for row in relation_rows
            if row["source"] in component and row["target"] in component
        ]

        observation_rows: list[AutoskillCandidateObservation] = [
            {
                "id": document.observation_id,
                "memory_type": document.memory_type,
                "scope": document.scope,
                "summary": document.summary,
                "path": document.path,
            }
            for document in component_docs
        ]
        observation_ids = [row["id"] for row in observation_rows]
        cluster_hash = _short_hash({"observation_ids": observation_ids})
        candidates.append(
            {
                "cluster_hash": cluster_hash,
                "observation_ids": observation_ids,
                "observation_count": len(observation_rows),
                "procedural_count": procedural_count,
                "semantic_count": memory_type_counts[MemoryType.SEMANTIC],
                "episodic_count": memory_type_counts[MemoryType.EPISODIC],
                "observations": observation_rows,
                "relations": component_relations,
                "existing_pending_proposal": cluster_hash in proposed_hashes["pending"],
                "already_processed": (
                    cluster_hash in proposed_hashes["approved"]
                    or cluster_hash in proposed_hashes["rejected"]
                    or cluster_hash in processed_hashes
                ),
            }
        )

    return candidates
