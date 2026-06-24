"""EvoMemory LangGraph launch adapter."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

from ..gateway.background_runs import (
    BackgroundRun,
    BackgroundRunHooks,
    BackgroundRunPayload,
    BackgroundRunRequest,
    alaunch_background_run,
    launch_background_run,
)
from .source_context import MemorySourceContext, _trajectory_for_prompt
from .types import MemorySourceType
from .worker_activity import (
    MemoryOutputDelta,
    MemoryOutputSnapshot,
    forget_memory_worker,
    mark_memory_worker_finished,
    mark_memory_worker_started,
    snapshot_memory_outputs,
)

SUBAGENT_MEMORY_WORKER_GRAPH_ID = "evomemory-subagent-worker"
TURN_MEMORY_WORKER_GRAPH_ID = "evomemory-turn-worker"

MemoryWorkerFinishedHook = Callable[[BackgroundRun, MemoryOutputDelta | None], None]


def _memory_worker_graph_id(source_type: MemorySourceType) -> str:
    match source_type:
        case MemorySourceType.TURN:
            return TURN_MEMORY_WORKER_GRAPH_ID
        case MemorySourceType.SUBAGENT:
            return SUBAGENT_MEMORY_WORKER_GRAPH_ID


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
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": _memory_worker_user_prompt(context),
                }
            ]
        },
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


def _memory_worker_launch_hooks(
    memory_dir: str | Path,
    *,
    on_worker_finished: MemoryWorkerFinishedHook | None = None,
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
        forget_memory_worker(run.thread_id, run.run_id)

    return BackgroundRunHooks(
        on_before_run=on_before_run,
        on_started=on_started,
        on_finished=on_finished,
        on_aborted=on_aborted,
        on_watcher_start_failed=on_finished,
    )


def launch_memory_worker(
    context: MemorySourceContext,
    *,
    on_worker_finished: MemoryWorkerFinishedHook | None = None,
) -> BackgroundRun | None:
    """Launch one synchronous EvoMemory worker for a source context."""
    return launch_background_run(
        memory_worker_launch_request(context),
        hooks=_memory_worker_launch_hooks(
            context.memory_dir,
            on_worker_finished=on_worker_finished,
        ),
    )


async def alaunch_memory_worker(
    context: MemorySourceContext,
    *,
    on_worker_finished: MemoryWorkerFinishedHook | None = None,
) -> BackgroundRun | None:
    """Launch one asynchronous EvoMemory worker for a source context."""
    return await alaunch_background_run(
        memory_worker_launch_request(context),
        hooks=_memory_worker_launch_hooks(
            context.memory_dir,
            on_worker_finished=on_worker_finished,
        ),
    )
