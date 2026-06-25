"""Schedule memory follow-up work after memory workers finish."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from ..gateway.background_runs import BackgroundRun
from .observations import read_observation_id_from_path
from .worker_activity import (
    MemoryOutputDelta,
    has_active_memory_workers,
    mark_observation_linker_launch_finished,
    mark_observation_linker_launch_started,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ObservationLinkerContext:
    """Input for one batched observation-linking pass."""

    memory_dir: Path
    workspace_dir: Path
    project_id: str
    observation_ids: tuple[str, ...]


ObservationLinkerLauncher = Callable[[ObservationLinkerContext], BackgroundRun | None]
ActiveMemoryWorkerCheck = Callable[[str | Path], bool]


class _BatchKey(NamedTuple):
    memory_dir: str
    workspace_dir: str
    project_id: str


def _root_key(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def _batch_key_from_worker_run(
    run: BackgroundRun,
    delta: MemoryOutputDelta,
) -> _BatchKey | None:
    metadata = run.metadata
    workspace_dir = metadata.get("workspace_dir")
    project_id = metadata.get("project_id")
    if not workspace_dir or not project_id:
        logger.debug(
            "Skipping observation linker for run %s; missing worker metadata",
            run.run_id,
        )
        return None
    return _BatchKey(
        memory_dir=_root_key(delta.memory_dir),
        workspace_dir=_root_key(workspace_dir),
        project_id=project_id,
    )


class MemoryScheduler:
    """Batch memory worker outputs and launch ready follow-up workers."""

    def __init__(
        self,
        *,
        launch_linker: ObservationLinkerLauncher,
        has_active_workers: ActiveMemoryWorkerCheck = has_active_memory_workers,
    ) -> None:
        self._launch_linker = launch_linker
        self._has_active_workers = has_active_workers
        self._pending: dict[_BatchKey, set[str]] = {}
        self._lock = threading.Lock()

    def _launch_ready(
        self,
        contexts: tuple[ObservationLinkerContext, ...],
    ) -> None:
        for context in contexts:
            mark_observation_linker_launch_started()
            try:
                self._launch_linker(context)
            except Exception:
                logger.warning("Failed to launch observation linker", exc_info=True)
            finally:
                mark_observation_linker_launch_finished()

    def _observation_ids_for_paths(
        self,
        *,
        memory_dir: str,
        observation_paths: set[str],
    ) -> tuple[str, ...]:
        observation_ids = []
        for observation_path in sorted(observation_paths):
            observation_id = read_observation_id_from_path(
                Path(memory_dir) / observation_path
            )
            if observation_id is None:
                logger.debug(
                    "Skipping observation linker input without id: %s",
                    observation_path,
                )
                continue
            observation_ids.append(observation_id)
        return tuple(observation_ids)

    def record_observation_created(self, context: ObservationLinkerContext) -> None:
        """Queue directly written observations for the next ready linker batch."""
        key = _BatchKey(
            memory_dir=_root_key(context.memory_dir),
            workspace_dir=_root_key(context.workspace_dir),
            project_id=context.project_id,
        )
        with self._lock:
            self._pending.setdefault(key, set()).update(context.observation_ids)

    def flush_ready(self) -> None:
        """Launch any pending linker batches that are no longer blocked."""
        self._launch_ready(self._drain_ready())

    def record_worker_finished(
        self,
        run: BackgroundRun,
        delta: MemoryOutputDelta | None,
    ) -> None:
        """Record one finished memory worker and launch any ready linker batches."""
        contexts = self._record_finished_and_drain_ready(run=run, delta=delta)
        self._launch_ready(contexts)

    def record_worker_aborted(
        self,
        run: BackgroundRun,
        delta: MemoryOutputDelta | None,
    ) -> None:
        """Queue persisted observations from an abandoned worker."""
        contexts = self._record_finished_and_drain_ready(run=run, delta=delta)
        self._launch_ready(contexts)

    def _ready_batches_locked(self) -> list[tuple[_BatchKey, set[str]]]:
        ready_batches = []
        for key in list(self._pending):
            if not self._has_active_workers(key.memory_dir):
                ready_batches.append((key, self._pending.pop(key)))
        return ready_batches

    def _contexts_for_batches(
        self,
        ready_batches: list[tuple[_BatchKey, set[str]]],
    ) -> tuple[ObservationLinkerContext, ...]:
        ready_contexts = []
        for key, observation_ids in ready_batches:
            if not observation_ids:
                continue
            ready_contexts.append(
                ObservationLinkerContext(
                    memory_dir=Path(key.memory_dir),
                    workspace_dir=Path(key.workspace_dir),
                    project_id=key.project_id,
                    observation_ids=tuple(sorted(observation_ids)),
                )
            )

        return tuple(ready_contexts)

    def _drain_ready(self) -> tuple[ObservationLinkerContext, ...]:
        with self._lock:
            ready_batches = self._ready_batches_locked()
        return self._contexts_for_batches(ready_batches)

    def _record_finished_and_drain_ready(
        self,
        *,
        run: BackgroundRun,
        delta: MemoryOutputDelta | None,
    ) -> tuple[ObservationLinkerContext, ...]:
        key: _BatchKey | None = None
        observation_ids: tuple[str, ...] = ()
        if delta is not None and delta.observation_paths:
            key = _batch_key_from_worker_run(run, delta)
            if key is not None:
                observation_ids = self._observation_ids_for_paths(
                    memory_dir=key.memory_dir,
                    observation_paths=set(delta.observation_paths),
                )

        with self._lock:
            if key is not None and observation_ids:
                self._pending.setdefault(key, set()).update(observation_ids)

            ready_batches = self._ready_batches_locked()

        return self._contexts_for_batches(ready_batches)
