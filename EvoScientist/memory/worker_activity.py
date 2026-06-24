"""Process-local activity tracking for EvoMemory workers."""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

MemoryActivityPhase = Literal["worker", "linker"]


@dataclass(frozen=True)
class MemoryWorkerStatusSnapshot:
    """Completed memory writes shown in the status bar."""

    is_running: bool = False
    profile_updates: int = 0
    observations_recorded: int = 0


@dataclass(frozen=True)
class ObservationLinkerStatusSnapshot:
    """Active observation-linking work shown in the status bar."""

    is_running: bool = False


@dataclass(frozen=True)
class MemoryOutputSnapshot:
    profile_files: dict[str, str]
    observation_files: frozenset[str]


@dataclass(frozen=True)
class MemoryOutputDelta:
    """Deduped memory writes credited when a worker finishes."""

    memory_dir: Path
    profile_paths: tuple[str, ...] = ()
    observation_paths: tuple[str, ...] = ()

    @property
    def profile_updates(self) -> int:
        return len(self.profile_paths)

    @property
    def observations_recorded(self) -> int:
        return len(self.observation_paths)

    @property
    def has_changes(self) -> bool:
        return bool(self.profile_paths or self.observation_paths)


@dataclass(frozen=True)
class _ActiveMemoryWorker:
    memory_dir: Path
    before_outputs: MemoryOutputSnapshot


_active_runs: dict[tuple[str, str], _ActiveMemoryWorker] = {}
_active_linker_runs: set[tuple[str, str]] = set()
_active_lock = threading.Lock()
_profile_updates = 0
_observations_recorded = 0
_counted_profile_versions: set[tuple[str, str, str]] = set()
_counted_observation_files: set[tuple[str, str]] = set()


def _memory_root_key(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


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


def _memory_output_delta_result(
    *,
    memory_dir: Path,
    profile_versions: set[tuple[str, str, str]],
    observation_files: set[tuple[str, str]],
) -> MemoryOutputDelta:
    return MemoryOutputDelta(
        memory_dir=memory_dir,
        profile_paths=tuple(
            sorted({path for _root_key, path, _digest in profile_versions})
        ),
        observation_paths=tuple(sorted(path for _root_key, path in observation_files)),
    )


def memory_worker_status() -> MemoryWorkerStatusSnapshot:
    with _active_lock:
        return MemoryWorkerStatusSnapshot(
            is_running=bool(_active_runs),
            profile_updates=_profile_updates,
            observations_recorded=_observations_recorded,
        )


def observation_linker_status() -> ObservationLinkerStatusSnapshot:
    with _active_lock:
        return ObservationLinkerStatusSnapshot(is_running=bool(_active_linker_runs))


def has_active_memory_workers(memory_dir: str | Path | None = None) -> bool:
    """Return whether any memory workers are still active."""
    with _active_lock:
        if memory_dir is None:
            return bool(_active_runs)
        root_key = _memory_root_key(memory_dir)
        return any(
            _memory_root_key(worker.memory_dir) == root_key
            for worker in _active_runs.values()
        )


def memory_worker_observed_outputs() -> MemoryWorkerStatusSnapshot:
    """Return completed counts plus already-written outputs from active workers."""
    with _active_lock:
        active_workers = list(_active_runs.values())
        profile_updates = _profile_updates
        observations_recorded = _observations_recorded
        counted_profile_versions = set(_counted_profile_versions)
        counted_observation_files = set(_counted_observation_files)

    profile_versions: set[tuple[str, str, str]] = set()
    observation_files: set[tuple[str, str]] = set()
    for worker in active_workers:
        after = snapshot_memory_outputs(worker.memory_dir)
        worker_profile_versions, worker_observation_files = _memory_output_delta(
            worker.memory_dir,
            worker.before_outputs,
            after,
        )
        profile_versions.update(worker_profile_versions)
        observation_files.update(worker_observation_files)

    profile_updates += len(profile_versions - counted_profile_versions)
    observations_recorded += len(observation_files - counted_observation_files)
    return MemoryWorkerStatusSnapshot(
        is_running=bool(active_workers),
        profile_updates=profile_updates,
        observations_recorded=observations_recorded,
    )


def wait_for_memory_pipeline_idle(
    *,
    timeout_seconds: float,
    poll_seconds: float,
    output_grace_seconds: float,
    on_saved: Callable[[MemoryWorkerStatusSnapshot], None] | None = None,
    on_waiting: Callable[[MemoryActivityPhase], None] | None = None,
    on_timeout: Callable[[MemoryActivityPhase], None] | None = None,
    get_worker_status: Callable[[], MemoryWorkerStatusSnapshot] = (
        memory_worker_observed_outputs
    ),
    get_linker_status: Callable[[], ObservationLinkerStatusSnapshot] = (
        observation_linker_status
    ),
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Poll until the memory worker/linker pipeline is idle.

    Returns ``True`` when all tracked memory work is idle, ``False`` when
    status polling fails or the active phase exceeds its timeout.
    """
    deadline = monotonic() + timeout_seconds
    saved_announced = False
    announced_saved_counts: tuple[int, int] | None = None
    output_seen_at: float | None = None
    observed_status: MemoryWorkerStatusSnapshot | None = None
    saw_active_memory_work = False
    idle_after_active_memory_work = False
    active_phase: MemoryActivityPhase | None = None

    def emit_saved(status: MemoryWorkerStatusSnapshot) -> None:
        nonlocal announced_saved_counts
        saved_counts = (status.observations_recorded, status.profile_updates)
        if saved_counts == (0, 0) or saved_counts == announced_saved_counts:
            return
        if on_saved is not None:
            on_saved(status)
        announced_saved_counts = saved_counts

    while True:
        now = monotonic()
        try:
            observed = get_worker_status()
            linker_status = get_linker_status()
        except Exception:
            return False

        memory_work_is_running = observed.is_running or linker_status.is_running
        if not memory_work_is_running:
            if saw_active_memory_work and not idle_after_active_memory_work:
                idle_after_active_memory_work = True
                sleep(poll_seconds)
                continue
            emit_saved(observed)
            return True

        saw_active_memory_work = True
        idle_after_active_memory_work = False
        current_phase: MemoryActivityPhase = (
            "worker" if observed.is_running else "linker"
        )
        if current_phase != active_phase:
            active_phase = current_phase
            deadline = now + timeout_seconds

        if observed.observations_recorded or observed.profile_updates:
            if output_seen_at is None:
                output_seen_at = now
            observed_status = observed
            if now - output_seen_at >= output_grace_seconds and not saved_announced:
                emit_saved(observed_status)
                saved_announced = True

        if now >= deadline:
            if on_timeout is not None:
                on_timeout(current_phase)
            return False

        if on_waiting is not None:
            on_waiting(current_phase)
        sleep(poll_seconds)


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


def mark_observation_linker_started(*, thread_id: str, run_id: str) -> None:
    with _active_lock:
        _active_linker_runs.add((thread_id, run_id))


def forget_observation_linker(thread_id: str, run_id: str) -> None:
    with _active_lock:
        _active_linker_runs.discard((thread_id, run_id))


def mark_memory_worker_finished(
    thread_id: str,
    run_id: str,
) -> MemoryOutputDelta | None:
    global _observations_recorded, _profile_updates

    with _active_lock:
        worker = _active_runs.pop((thread_id, run_id), None)
    if worker is None:
        return None

    after = snapshot_memory_outputs(worker.memory_dir)
    profile_versions, observation_files = _memory_output_delta(
        worker.memory_dir,
        worker.before_outputs,
        after,
    )
    if not profile_versions and not observation_files:
        return MemoryOutputDelta(memory_dir=worker.memory_dir)

    with _active_lock:
        new_profile_versions = profile_versions - _counted_profile_versions
        new_observation_files = observation_files - _counted_observation_files
        _counted_profile_versions.update(new_profile_versions)
        _counted_observation_files.update(new_observation_files)
        _profile_updates += len(new_profile_versions)
        _observations_recorded += len(new_observation_files)
    return _memory_output_delta_result(
        memory_dir=worker.memory_dir,
        profile_versions=new_profile_versions,
        observation_files=new_observation_files,
    )


def reset_memory_worker_status_for_tests() -> None:
    global _observations_recorded, _profile_updates

    with _active_lock:
        _active_runs.clear()
        _active_linker_runs.clear()
        _counted_profile_versions.clear()
        _counted_observation_files.clear()
        _profile_updates = 0
        _observations_recorded = 0
