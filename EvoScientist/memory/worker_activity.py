"""Process-local activity tracking for EvoMemory workers."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MemoryWorkerStatusSnapshot:
    """Completed memory writes shown in the status bar."""

    is_running: bool = False
    profile_updates: int = 0
    observations_recorded: int = 0


@dataclass(frozen=True)
class MemoryOutputSnapshot:
    profile_files: dict[str, str]
    observation_files: frozenset[str]


@dataclass(frozen=True)
class _ActiveMemoryWorker:
    memory_dir: Path
    before_outputs: MemoryOutputSnapshot


_active_runs: dict[tuple[str, str], _ActiveMemoryWorker] = {}
_active_lock = threading.Lock()
_profile_updates = 0
_observations_recorded = 0
_counted_profile_versions: set[tuple[str, str, str]] = set()
_counted_observation_files: set[tuple[str, str]] = set()


def _file_digest(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def snapshot_memory_outputs(memory_dir: str | Path) -> MemoryOutputSnapshot:
    root = Path(memory_dir).expanduser()
    profile_root = root / "profile"
    observation_root = root / "observations"

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

    return MemoryOutputSnapshot(
        profile_files=profile_files,
        observation_files=frozenset(observation_files),
    )


def _memory_output_delta(
    memory_dir: str | Path,
    before: MemoryOutputSnapshot,
    after: MemoryOutputSnapshot,
) -> tuple[set[tuple[str, str, str]], set[tuple[str, str]]]:
    root_key = str(Path(memory_dir).expanduser().resolve())
    profile_versions = {
        (root_key, path, digest)
        for path, digest in after.profile_files.items()
        if before.profile_files.get(path) != digest
    }
    observation_files = {
        (root_key, path) for path in after.observation_files - before.observation_files
    }
    return profile_versions, observation_files


def memory_worker_status() -> MemoryWorkerStatusSnapshot:
    with _active_lock:
        return MemoryWorkerStatusSnapshot(
            is_running=bool(_active_runs),
            profile_updates=_profile_updates,
            observations_recorded=_observations_recorded,
        )


def clear_memory_worker_saved_counts() -> None:
    """Clear completed memory-save counters while preserving active workers."""
    global _observations_recorded, _profile_updates

    with _active_lock:
        _profile_updates = 0
        _observations_recorded = 0


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
    global _observations_recorded, _profile_updates

    with _active_lock:
        worker = _active_runs.pop((thread_id, run_id), None)
    if worker is None:
        return

    after = snapshot_memory_outputs(worker.memory_dir)
    profile_versions, observation_files = _memory_output_delta(
        worker.memory_dir,
        worker.before_outputs,
        after,
    )
    if not profile_versions and not observation_files:
        return

    with _active_lock:
        new_profile_versions = profile_versions - _counted_profile_versions
        new_observation_files = observation_files - _counted_observation_files
        _counted_profile_versions.update(new_profile_versions)
        _counted_observation_files.update(new_observation_files)
        _profile_updates += len(new_profile_versions)
        _observations_recorded += len(new_observation_files)


def reset_memory_worker_status_for_tests() -> None:
    global _observations_recorded, _profile_updates

    with _active_lock:
        _active_runs.clear()
        _counted_profile_versions.clear()
        _counted_observation_files.clear()
        _profile_updates = 0
        _observations_recorded = 0
