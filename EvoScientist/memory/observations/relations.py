"""Frontmatter-native links between observation memory files."""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path

from ..types import ObservationRelation
from .store import (
    ObservationFrontmatter,
    RelatedObservationEntry,
    observation_document_by_id,
    related_observation_entries,
    write_observation_document,
)

_link_write_lock = threading.Lock()


def _relation_value(value: ObservationRelation | str) -> str:
    try:
        return ObservationRelation(value).value
    except ValueError as exc:
        allowed = ", ".join(relation.value for relation in ObservationRelation)
        raise ValueError(f"relation must be one of: {allowed}") from exc


def _can_write_reverse_relation(relation: str) -> bool:
    return relation != ObservationRelation.SUPERSEDES.value


def _upsert_related_observation(
    metadata: ObservationFrontmatter,
    *,
    target_observation_id: str,
    relation: str,
    reason: str,
    linked_at: str,
) -> bool:
    entries = related_observation_entries(metadata)
    new_entry = RelatedObservationEntry(
        id=target_observation_id,
        relation=ObservationRelation(relation),
        reason=reason,
        linked_at=linked_at,
    )
    for index, entry in enumerate(entries):
        if entry.id != target_observation_id:
            continue
        if (
            entry.id == new_entry.id
            and entry.relation == new_entry.relation
            and entry.reason == new_entry.reason
        ):
            return False
        entries[index] = new_entry
        metadata.related_observations = entries
        return True

    entries.append(new_entry)
    metadata.related_observations = entries
    return True


def link_observation_files(
    *,
    memory_dir: str | Path,
    project_id: str,
    source_observation_id: str,
    target_observation_id: str,
    reason: str,
    relation: ObservationRelation = ObservationRelation.COMPLEMENTS,
    bidirectional: bool = True,
) -> dict[str, object]:
    """Link two observations by amending their frontmatter metadata."""
    source_id = source_observation_id.strip()
    target_id = target_observation_id.strip()
    reason_text = reason.strip()
    relation_text = _relation_value(relation)
    if not source_id:
        raise ValueError("source_observation_id must not be empty")
    if not target_id:
        raise ValueError("target_observation_id must not be empty")
    if source_id == target_id:
        raise ValueError("source_observation_id and target_observation_id must differ")
    if not reason_text:
        raise ValueError("reason must not be empty")

    with _link_write_lock:
        source_document = observation_document_by_id(
            memory_dir=memory_dir,
            project_id=project_id,
            observation_id=source_id,
        )
        target_document = observation_document_by_id(
            memory_dir=memory_dir,
            project_id=project_id,
            observation_id=target_id,
        )
        missing = [
            observation_id
            for observation_id, document in (
                (source_id, source_document),
                (target_id, target_document),
            )
            if document is None
        ]
        if missing:
            return {
                "linked": False,
                "source_observation_id": source_id,
                "target_observation_id": target_id,
                "relation": relation_text,
                "updated_observation_ids": [],
                "missing_observation_ids": missing,
            }

        linked_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        updates: list[tuple[str, Path, ObservationFrontmatter, str]] = []
        assert source_document is not None
        source_path, source_metadata, source_body = source_document
        if _upsert_related_observation(
            source_metadata,
            target_observation_id=target_id,
            relation=relation_text,
            reason=reason_text,
            linked_at=linked_at,
        ):
            updates.append((source_id, source_path, source_metadata, source_body))

        if bidirectional and _can_write_reverse_relation(relation_text):
            assert target_document is not None
            target_path, target_metadata, target_body = target_document
            if _upsert_related_observation(
                target_metadata,
                target_observation_id=source_id,
                relation=relation_text,
                reason=reason_text,
                linked_at=linked_at,
            ):
                updates.append((target_id, target_path, target_metadata, target_body))

        for _observation_id, path, metadata, body in updates:
            write_observation_document(path, metadata=metadata, body=body)

        return {
            "linked": bool(updates),
            "source_observation_id": source_id,
            "target_observation_id": target_id,
            "relation": relation_text,
            "updated_observation_ids": [
                observation_id for observation_id, *_ in updates
            ],
            "missing_observation_ids": [],
        }
