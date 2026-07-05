"""EvoMemory LangGraph launch adapter."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

from ..config import MemoryControls, get_effective_config
from ..gateway.background_runs import (
    BackgroundRun,
    BackgroundRunHooks,
    BackgroundRunPayload,
    BackgroundRunRequest,
    alaunch_background_run,
    launch_background_run,
)
from ..langgraph_dev.sdk import messages_input
from .observations import build_observation_linker_index_context
from .scheduler import ObservationLinkerContext
from .source_context import MemorySourceContext, _trajectory_for_prompt
from .types import MemorySourceType
from .worker_activity import (
    MemoryOutputDelta,
    MemoryOutputSnapshot,
    ObservationRelationSnapshot,
    forget_memory_worker,
    forget_observation_linker,
    mark_memory_worker_finished,
    mark_memory_worker_started,
    mark_observation_linker_finished,
    mark_observation_linker_started,
    snapshot_memory_outputs,
    snapshot_observation_relations,
)

SUBAGENT_MEMORY_WORKER_GRAPH_ID = "evomemory-subagent-worker"
TURN_MEMORY_WORKER_GRAPH_ID = "evomemory-turn-worker"
OBSERVATION_LINKER_GRAPH_ID = "evomemory-observation-linker"

MemoryWorkerFinishedHook = Callable[[BackgroundRun, MemoryOutputDelta | None], None]
MemoryWorkerAbortedHook = Callable[[BackgroundRun, MemoryOutputDelta | None], None]


def _observation_linking_enabled() -> bool:
    return MemoryControls.from_config(get_effective_config()).observations_enabled


def _memory_worker_graph_id(source_type: MemorySourceType) -> str:
    match source_type:
        case MemorySourceType.TURN:
            return TURN_MEMORY_WORKER_GRAPH_ID
        case MemorySourceType.SUBAGENT:
            return SUBAGENT_MEMORY_WORKER_GRAPH_ID
        case _:
            raise ValueError(f"Unsupported memory source type: {source_type!r}")


def _memory_worker_user_prompt(context: MemorySourceContext) -> str:
    match context.source_type:
        case MemorySourceType.TURN:
            return (
                "Review this completed orchestrator turn.\n\n"
                f"Source agent: {context.source_agent}\n"
                f"Source session: {context.session_id}\n\n"
                f"Turn trajectory:\n{_trajectory_for_prompt(context.trajectory)}"
            )
        case MemorySourceType.SUBAGENT:
            return (
                "Review this completed subagent run.\n\n"
                f"Source agent: {context.source_agent}\n"
                f"Source session: {context.session_id}\n\n"
                f"Trajectory:\n{_trajectory_for_prompt(context.trajectory)}"
            )
        case _:
            raise ValueError(f"Unsupported memory source type: {context.source_type!r}")


def _runs_create_kwargs(payload: BackgroundRunPayload) -> BackgroundRunPayload:
    try:
        from EvoScientist.llm.patches import _merge_runs_config_kwargs
    except Exception:
        return payload
    return cast("BackgroundRunPayload", _merge_runs_config_kwargs(dict(payload)))


def _worker_workspace_dir(workspace_dir: str | Path) -> str:
    return str(Path(workspace_dir).expanduser().resolve())


def _memory_worker_metadata(context: MemorySourceContext) -> dict[str, str]:
    return {
        "run_kind": f"evomemory_{context.source_type.value}_worker",
        "source_session_id": context.session_id,
        "source_agent": context.source_agent,
        "project_id": context.project_id,
        "trajectory_digest": context.trajectory_digest,
        "workspace_dir": _worker_workspace_dir(context.workspace_dir),
    }


def _memory_worker_run_payload(
    *,
    context: MemorySourceContext,
    thread_id: str,
) -> BackgroundRunPayload:
    """Build the LangGraph SDK run payload for a memory worker."""
    metadata = _memory_worker_metadata(context)
    payload: BackgroundRunPayload = {
        "assistant_id": _memory_worker_graph_id(context.source_type),
        "input": messages_input(_memory_worker_user_prompt(context)),
        "metadata": metadata,
        "config": {
            "configurable": {
                "thread_id": thread_id,
                "evomemory_source_session_id": context.session_id,
                "evomemory_source_agent": context.source_agent,
                "evomemory_project_id": context.project_id,
                "evomemory_trajectory_digest": context.trajectory_digest,
            }
        },
    }
    return _runs_create_kwargs(payload)


def memory_worker_launch_request(
    context: MemorySourceContext,
) -> BackgroundRunRequest:
    """Build the background run request for a memory worker."""
    metadata = _memory_worker_metadata(context)

    def run_payload(thread_id: str) -> BackgroundRunPayload:
        return _memory_worker_run_payload(context=context, thread_id=thread_id)

    return BackgroundRunRequest(
        graph_id=_memory_worker_graph_id(context.source_type),
        run_payload=run_payload,
        thread_metadata=metadata,
        name="EvoMemory worker",
    )


def _observation_linker_user_prompt(context: ObservationLinkerContext) -> str:
    payload = {
        "project_id": context.project_id,
        "new_observation_ids": sorted(context.observation_ids),
    }
    prompt = (
        "Link newly recorded observations when there is a strong reusable "
        "relationship.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)}"
    )
    observation_index = build_observation_linker_index_context(
        memory_dir=context.memory_dir,
        project_id=context.project_id,
        exclude_ids=context.observation_ids,
    )
    if observation_index:
        prompt += f"\n\n{observation_index}"
    return prompt


def _observation_linker_metadata(
    context: ObservationLinkerContext,
) -> dict[str, str]:
    return {
        "run_kind": "evomemory_observation_linker",
        "project_id": context.project_id,
        "observation_count": str(len(context.observation_ids)),
        "workspace_dir": str(context.workspace_dir.expanduser().resolve()),
    }


def _observation_linker_run_payload(
    *,
    context: ObservationLinkerContext,
    thread_id: str,
) -> BackgroundRunPayload:
    payload: BackgroundRunPayload = {
        "assistant_id": OBSERVATION_LINKER_GRAPH_ID,
        "input": messages_input(_observation_linker_user_prompt(context)),
        "metadata": _observation_linker_metadata(context),
        "config": {
            "configurable": {
                "thread_id": thread_id,
                "evomemory_project_id": context.project_id,
                "evomemory_observation_ids": json.dumps(
                    list(context.observation_ids),
                    ensure_ascii=False,
                ),
            }
        },
    }
    return _runs_create_kwargs(payload)


def observation_linker_launch_request(
    context: ObservationLinkerContext,
) -> BackgroundRunRequest:
    """Build the background run request for the observation linker."""

    def run_payload(thread_id: str) -> BackgroundRunPayload:
        return _observation_linker_run_payload(
            context=context,
            thread_id=thread_id,
        )

    return BackgroundRunRequest(
        graph_id=OBSERVATION_LINKER_GRAPH_ID,
        run_payload=run_payload,
        thread_metadata=_observation_linker_metadata(context),
        name="EvoMemory observation linker",
    )


def _observation_linker_launch_hooks(memory_dir: str | Path) -> BackgroundRunHooks:
    before_relations: dict[str, ObservationRelationSnapshot] = {}

    def on_before_run(_thread_id: str) -> None:
        before_relations["value"] = snapshot_observation_relations(memory_dir)

    def on_started(run: BackgroundRun) -> None:
        mark_observation_linker_started(
            thread_id=run.thread_id,
            run_id=run.run_id,
            before_relations=before_relations.get("value"),
        )

    def on_finished(run: BackgroundRun) -> None:
        mark_observation_linker_finished(
            run.thread_id,
            run.run_id,
            memory_dir=memory_dir,
        )

    def on_aborted(run: BackgroundRun) -> None:
        forget_observation_linker(run.thread_id, run.run_id)

    return BackgroundRunHooks(
        on_before_run=on_before_run,
        on_started=on_started,
        on_finished=on_finished,
        on_aborted=on_aborted,
        on_watcher_start_failed=on_aborted,
    )


def _memory_worker_launch_hooks(
    memory_dir: str | Path,
    *,
    on_worker_finished: MemoryWorkerFinishedHook | None = None,
    on_worker_aborted: MemoryWorkerAbortedHook | None = None,
) -> BackgroundRunHooks:
    before_outputs: dict[str, MemoryOutputSnapshot] = {}

    def on_before_run(_thread_id: str) -> None:
        before_outputs["value"] = snapshot_memory_outputs(memory_dir)

    def on_started(run: BackgroundRun) -> None:
        mark_memory_worker_started(
            thread_id=run.thread_id,
            run_id=run.run_id,
            memory_dir=memory_dir,
            before_outputs=before_outputs.get("value"),
        )

    def on_finished(run: BackgroundRun) -> None:
        delta = mark_memory_worker_finished(run.thread_id, run.run_id)
        if on_worker_finished is not None:
            on_worker_finished(run, delta)

    def on_aborted(run: BackgroundRun) -> None:
        delta = mark_memory_worker_finished(run.thread_id, run.run_id)
        if on_worker_aborted is not None:
            on_worker_aborted(run, delta)

    def on_status_unknown(run: BackgroundRun) -> None:
        forget_memory_worker(run.thread_id, run.run_id)

    return BackgroundRunHooks(
        on_before_run=on_before_run,
        on_started=on_started,
        on_finished=on_finished,
        on_aborted=on_aborted,
        on_status_unknown=on_status_unknown,
        on_watcher_start_failed=on_aborted,
    )


def launch_memory_worker(
    context: MemorySourceContext,
    *,
    on_worker_finished: MemoryWorkerFinishedHook | None = None,
    on_worker_aborted: MemoryWorkerAbortedHook | None = None,
) -> BackgroundRun | None:
    """Launch one synchronous EvoMemory worker for a source context."""
    return launch_background_run(
        memory_worker_launch_request(context),
        hooks=_memory_worker_launch_hooks(
            context.memory_dir,
            on_worker_finished=on_worker_finished,
            on_worker_aborted=on_worker_aborted,
        ),
    )


async def alaunch_memory_worker(
    context: MemorySourceContext,
    *,
    on_worker_finished: MemoryWorkerFinishedHook | None = None,
    on_worker_aborted: MemoryWorkerAbortedHook | None = None,
) -> BackgroundRun | None:
    """Launch one asynchronous EvoMemory worker for a source context."""
    return await alaunch_background_run(
        memory_worker_launch_request(context),
        hooks=_memory_worker_launch_hooks(
            context.memory_dir,
            on_worker_finished=on_worker_finished,
            on_worker_aborted=on_worker_aborted,
        ),
    )


def launch_observation_linker(
    context: ObservationLinkerContext,
) -> BackgroundRun | None:
    """Launch one synchronous observation-linking pass."""
    if not _observation_linking_enabled():
        return None
    return launch_background_run(
        observation_linker_launch_request(context),
        hooks=_observation_linker_launch_hooks(context.memory_dir),
    )


async def alaunch_observation_linker(
    context: ObservationLinkerContext,
) -> BackgroundRun | None:
    """Launch one asynchronous observation-linking pass."""
    if not _observation_linking_enabled():
        return None
    return await alaunch_background_run(
        observation_linker_launch_request(context),
        hooks=_observation_linker_launch_hooks(context.memory_dir),
    )
