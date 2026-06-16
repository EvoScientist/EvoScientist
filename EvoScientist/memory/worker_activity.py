"""Process-local activity tracking for EvoMemory workers."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class MemoryWorkerStatusSnapshot:
    """Completed memory writes shown in the status bar."""

    is_running: bool = False
    synthesis_running: bool = False
    profile_updates: int = 0
    observations_recorded: int = 0
    knowledge_created: int = 0
    knowledge_updated: int = 0
    knowledge_archived: int = 0


@dataclass(frozen=True)
class KnowledgeOutputSnapshot:
    digest: str
    status: str


@dataclass(frozen=True)
class MemoryOutputSnapshot:
    profile_files: dict[str, str]
    observation_files: frozenset[str]
    knowledge_files: dict[str, KnowledgeOutputSnapshot] = field(default_factory=dict)


@dataclass(frozen=True)
class _MemoryOutputDelta:
    profile_versions: frozenset[tuple[str, str, str]] = frozenset()
    observation_files: frozenset[tuple[str, str]] = frozenset()
    knowledge_created: frozenset[tuple[str, str, str]] = frozenset()
    knowledge_updated: frozenset[tuple[str, str, str]] = frozenset()
    knowledge_archived: frozenset[tuple[str, str, str]] = frozenset()

    @property
    def is_empty(self) -> bool:
        return not (
            self.profile_versions
            or self.observation_files
            or self.knowledge_created
            or self.knowledge_updated
            or self.knowledge_archived
        )


@dataclass(frozen=True)
class _ActiveMemoryWorker:
    memory_dir: Path
    before_outputs: MemoryOutputSnapshot


_active_runs: dict[tuple[str, str], _ActiveMemoryWorker] = {}
_active_synthesis_runs: dict[tuple[str, str], _ActiveMemoryWorker] = {}
_active_lock = threading.Lock()
_profile_updates = 0
_observations_recorded = 0
_knowledge_created = 0
_knowledge_updated = 0
_knowledge_archived = 0
_counted_profile_versions: set[tuple[str, str, str]] = set()
_counted_observation_files: set[tuple[str, str]] = set()
_counted_knowledge_versions: set[tuple[str, str, str]] = set()


def _file_digest(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _knowledge_status(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return "unknown"
    if not text.startswith("---\n"):
        return "unknown"
    try:
        frontmatter, _body = text.removeprefix("---\n").split("\n---\n", 1)
        metadata = yaml.safe_load(frontmatter)
    except (ValueError, yaml.YAMLError):
        return "unknown"
    if not isinstance(metadata, dict):
        return "unknown"
    if "status" in metadata:
        status = str(metadata.get("status") or "").strip().lower()
        return status or "unknown"
    return "active"


def snapshot_memory_outputs(memory_dir: str | Path) -> MemoryOutputSnapshot:
    root = Path(memory_dir).expanduser()
    profile_root = root / "profile"
    observation_root = root / "observations"
    knowledge_root = root / "knowledge"

    profile_files: dict[str, str] = {}
    if profile_root.exists():
        for path in profile_root.rglob("*.md"):
            if not path.is_file():
                continue
            digest = _file_digest(path)
            if digest is not None:
                profile_files[str(path.relative_to(root))] = digest

    observation_files: set[str] = set()
    if observation_root.exists():
        for path in observation_root.rglob("*.md"):
            if path.is_file():
                observation_files.add(str(path.relative_to(root)))

    knowledge_files: dict[str, KnowledgeOutputSnapshot] = {}
    if knowledge_root.exists():
        for path in knowledge_root.rglob("*.md"):
            if not path.is_file():
                continue
            digest = _file_digest(path)
            if digest is not None:
                knowledge_files[str(path.relative_to(root))] = KnowledgeOutputSnapshot(
                    digest=digest,
                    status=_knowledge_status(path),
                )

    return MemoryOutputSnapshot(
        profile_files=profile_files,
        observation_files=frozenset(observation_files),
        knowledge_files=knowledge_files,
    )


def _memory_output_delta(
    memory_dir: str | Path,
    before: MemoryOutputSnapshot,
    after: MemoryOutputSnapshot,
) -> _MemoryOutputDelta:
    root_key = str(Path(memory_dir).expanduser().resolve())
    profile_versions = {
        (root_key, path, digest)
        for path, digest in after.profile_files.items()
        if before.profile_files.get(path) != digest
    }
    observation_files = {
        (root_key, path) for path in after.observation_files - before.observation_files
    }
    knowledge_created: set[tuple[str, str, str]] = set()
    knowledge_updated: set[tuple[str, str, str]] = set()
    knowledge_archived: set[tuple[str, str, str]] = set()
    for path, after_state in after.knowledge_files.items():
        before_state = before.knowledge_files.get(path)
        if before_state is not None and before_state.digest == after_state.digest:
            continue
        key = (root_key, path, after_state.digest)
        if before_state is None:
            if after_state.status == "archived":
                knowledge_archived.add(key)
            else:
                knowledge_created.add(key)
            continue
        if before_state.status != "archived" and after_state.status == "archived":
            knowledge_archived.add(key)
        else:
            knowledge_updated.add(key)
    return _MemoryOutputDelta(
        profile_versions=frozenset(profile_versions),
        observation_files=frozenset(observation_files),
        knowledge_created=frozenset(knowledge_created),
        knowledge_updated=frozenset(knowledge_updated),
        knowledge_archived=frozenset(knowledge_archived),
    )


def _active_worker_output_delta(
    workers: list[_ActiveMemoryWorker],
) -> _MemoryOutputDelta:
    profile_versions: set[tuple[str, str, str]] = set()
    observation_files: set[tuple[str, str]] = set()
    knowledge_created: set[tuple[str, str, str]] = set()
    knowledge_updated: set[tuple[str, str, str]] = set()
    knowledge_archived: set[tuple[str, str, str]] = set()

    for worker in workers:
        after = snapshot_memory_outputs(worker.memory_dir)
        delta = _memory_output_delta(worker.memory_dir, worker.before_outputs, after)
        profile_versions.update(delta.profile_versions)
        observation_files.update(delta.observation_files)
        knowledge_created.update(delta.knowledge_created)
        knowledge_updated.update(delta.knowledge_updated)
        knowledge_archived.update(delta.knowledge_archived)

    return _MemoryOutputDelta(
        profile_versions=frozenset(profile_versions),
        observation_files=frozenset(observation_files),
        knowledge_created=frozenset(knowledge_created),
        knowledge_updated=frozenset(knowledge_updated),
        knowledge_archived=frozenset(knowledge_archived),
    )


def _record_memory_output_delta(
    memory_dir: str | Path,
    before: MemoryOutputSnapshot,
) -> None:
    global _knowledge_archived, _knowledge_created, _knowledge_updated
    global _observations_recorded, _profile_updates

    after = snapshot_memory_outputs(memory_dir)
    delta = _memory_output_delta(memory_dir, before, after)
    if delta.is_empty:
        return

    with _active_lock:
        new_profile_versions = delta.profile_versions - _counted_profile_versions
        new_observation_files = delta.observation_files - _counted_observation_files
        new_knowledge_created = delta.knowledge_created - _counted_knowledge_versions
        new_knowledge_updated = delta.knowledge_updated - _counted_knowledge_versions
        new_knowledge_archived = delta.knowledge_archived - _counted_knowledge_versions
        _counted_profile_versions.update(new_profile_versions)
        _counted_observation_files.update(new_observation_files)
        _counted_knowledge_versions.update(new_knowledge_created)
        _counted_knowledge_versions.update(new_knowledge_updated)
        _counted_knowledge_versions.update(new_knowledge_archived)
        _profile_updates += len(new_profile_versions)
        _observations_recorded += len(new_observation_files)
        _knowledge_created += len(new_knowledge_created)
        _knowledge_updated += len(new_knowledge_updated)
        _knowledge_archived += len(new_knowledge_archived)


def memory_worker_status() -> MemoryWorkerStatusSnapshot:
    with _active_lock:
        return MemoryWorkerStatusSnapshot(
            is_running=bool(_active_runs),
            synthesis_running=bool(_active_synthesis_runs),
            profile_updates=_profile_updates,
            observations_recorded=_observations_recorded,
            knowledge_created=_knowledge_created,
            knowledge_updated=_knowledge_updated,
            knowledge_archived=_knowledge_archived,
        )


def memory_worker_observed_outputs() -> MemoryWorkerStatusSnapshot:
    """Return completed counts plus already-written outputs from active workers."""
    with _active_lock:
        active_workers = list(_active_runs.values())
        active_synthesis_workers = list(_active_synthesis_runs.values())
        profile_updates = _profile_updates
        observations_recorded = _observations_recorded
        knowledge_created_count = _knowledge_created
        knowledge_updated_count = _knowledge_updated
        knowledge_archived_count = _knowledge_archived
        counted_profile_versions = set(_counted_profile_versions)
        counted_observation_files = set(_counted_observation_files)
        counted_knowledge_versions = set(_counted_knowledge_versions)

    delta = _active_worker_output_delta([*active_workers, *active_synthesis_workers])
    new_knowledge_created = delta.knowledge_created - counted_knowledge_versions
    new_knowledge_updated = delta.knowledge_updated - counted_knowledge_versions
    new_knowledge_archived = delta.knowledge_archived - counted_knowledge_versions
    profile_updates += len(delta.profile_versions - counted_profile_versions)
    observations_recorded += len(delta.observation_files - counted_observation_files)
    knowledge_created_count += len(new_knowledge_created)
    knowledge_updated_count += len(new_knowledge_updated)
    knowledge_archived_count += len(new_knowledge_archived)
    return MemoryWorkerStatusSnapshot(
        is_running=bool(active_workers),
        synthesis_running=bool(active_synthesis_workers),
        profile_updates=profile_updates,
        observations_recorded=observations_recorded,
        knowledge_created=knowledge_created_count,
        knowledge_updated=knowledge_updated_count,
        knowledge_archived=knowledge_archived_count,
    )


def mark_synthesis_started(
    *,
    project_id: str,
    context_digest: str,
    memory_dir: str | Path,
    before_outputs: MemoryOutputSnapshot | None = None,
) -> None:
    memory_root = Path(memory_dir).expanduser()
    before = before_outputs or snapshot_memory_outputs(memory_root)
    with _active_lock:
        _active_synthesis_runs[(project_id, context_digest)] = _ActiveMemoryWorker(
            memory_dir=memory_root,
            before_outputs=before,
        )


def mark_synthesis_finished(*, project_id: str, context_digest: str) -> None:
    with _active_lock:
        worker = _active_synthesis_runs.pop((project_id, context_digest), None)
    if worker is None:
        return
    _record_memory_output_delta(worker.memory_dir, worker.before_outputs)


def clear_memory_worker_saved_counts() -> None:
    """Clear completed memory-save counters while preserving active workers."""
    global _knowledge_archived, _knowledge_created, _knowledge_updated
    global _observations_recorded, _profile_updates

    with _active_lock:
        _profile_updates = 0
        _observations_recorded = 0
        _knowledge_created = 0
        _knowledge_updated = 0
        _knowledge_archived = 0


def mark_memory_worker_started(
    *,
    thread_id: str,
    run_id: str,
    memory_dir: str | Path,
    before_outputs: MemoryOutputSnapshot | None = None,
) -> None:
    memory_root = Path(memory_dir).expanduser()
    before = before_outputs or snapshot_memory_outputs(memory_root)
    with _active_lock:
        _active_runs[(thread_id, run_id)] = _ActiveMemoryWorker(
            memory_dir=memory_root,
            before_outputs=before,
        )


def forget_memory_worker(thread_id: str, run_id: str) -> None:
    """Stop tracking a worker without counting memory-output deltas."""
    with _active_lock:
        _active_runs.pop((thread_id, run_id), None)


def mark_memory_worker_finished(thread_id: str, run_id: str) -> None:
    with _active_lock:
        worker = _active_runs.pop((thread_id, run_id), None)
    if worker is None:
        return

    _record_memory_output_delta(worker.memory_dir, worker.before_outputs)


def reset_memory_worker_status_for_tests() -> None:
    global _knowledge_archived, _knowledge_created, _knowledge_updated
    global _observations_recorded, _profile_updates

    with _active_lock:
        _active_runs.clear()
        _active_synthesis_runs.clear()
        _counted_profile_versions.clear()
        _counted_observation_files.clear()
        _counted_knowledge_versions.clear()
        _profile_updates = 0
        _observations_recorded = 0
        _knowledge_created = 0
        _knowledge_updated = 0
        _knowledge_archived = 0
