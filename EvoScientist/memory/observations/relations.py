"""Frontmatter-native links between observation memory files."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import yaml

from ..types import ObservationRelation
from .store import observation_document_by_id


def _related_observation_entries(
    metadata: Mapping[str, object],
) -> list[dict[str, str]]:
    value = metadata.get("related_observations")
    if not isinstance(value, list):
        return []

    entries: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        related_id = str(item.get("id") or "").strip()
        if not related_id:
            continue
        entry = {"id": related_id}
        for key in ("relation", "reason", "linked_at", "linked_by"):
            field_value = item.get(key)
            if isinstance(field_value, str) and field_value.strip():
                entry[key] = field_value.strip()
        entries.append(entry)
    return entries


def _relation_value(value: ObservationRelation | str) -> str:
    try:
        return ObservationRelation(value).value
    except ValueError as exc:
        allowed = ", ".join(relation.value for relation in ObservationRelation)
        raise ValueError(f"relation must be one of: {allowed}") from exc


def _can_write_reverse_relation(relation: str) -> bool:
    return relation != ObservationRelation.SUPERSEDES.value


def _upsert_related_observation(
    metadata: dict[str, object],
    *,
    target_observation_id: str,
    relation: str,
    reason: str,
    linked_at: str,
    linked_by: str,
) -> bool:
    entries = _related_observation_entries(metadata)
    new_entry = {
        "id": target_observation_id,
        "relation": relation,
        "reason": reason,
        "linked_at": linked_at,
        "linked_by": linked_by,
    }
    for index, entry in enumerate(entries):
        if entry["id"] != target_observation_id:
            continue
        comparable = {
            key: entry.get(key) for key in ("id", "relation", "reason", "linked_by")
        }
        expected = {
            key: new_entry[key] for key in ("id", "relation", "reason", "linked_by")
        }
        if comparable == expected:
            return False
        entries[index] = new_entry
        metadata["related_observations"] = entries
        return True

    entries.append(new_entry)
    metadata["related_observations"] = entries
    return True


def _write_observation_document(
    path: Path,
    *,
    metadata: Mapping[str, object],
    body: str,
) -> None:
    frontmatter = yaml.safe_dump(
        dict(metadata),
        allow_unicode=True,
        sort_keys=False,
    )
    path.write_text(f"---\n{frontmatter}---\n{body}", encoding="utf-8")


def link_observation_files(
    *,
    memory_dir: str | Path,
    project_id: str,
    source_observation_id: str,
    target_observation_id: str,
    reason: str,
    relation: ObservationRelation = ObservationRelation.RELATED,
    bidirectional: bool = True,
    linked_by: str = "evomemory-observation-linker",
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
    updates: list[tuple[str, Path, dict[str, object], str]] = []
    assert source_document is not None
    source_path, source_metadata, source_body = source_document
    if _upsert_related_observation(
        source_metadata,
        target_observation_id=target_id,
        relation=relation_text,
        reason=reason_text,
        linked_at=linked_at,
        linked_by=linked_by,
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
            linked_by=linked_by,
        ):
            updates.append((target_id, target_path, target_metadata, target_body))

    for _observation_id, path, metadata, body in updates:
        _write_observation_document(path, metadata=metadata, body=body)

    return {
        "linked": bool(updates),
        "source_observation_id": source_id,
        "target_observation_id": target_id,
        "relation": relation_text,
        "updated_observation_ids": [observation_id for observation_id, *_ in updates],
        "missing_observation_ids": [],
    }
