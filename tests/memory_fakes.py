"""Runtimes and source contexts for the memory pipeline tests."""

from __future__ import annotations

from langgraph.runtime import ExecutionInfo, Runtime

from EvoScientist.memory import scheduler as memory_scheduler
from EvoScientist.memory import source_context
from EvoScientist.memory.observations import MemorySourceType


def fake_execution_info(thread_id: str | None = None) -> ExecutionInfo:
    return ExecutionInfo(
        checkpoint_id="checkpoint-1",
        checkpoint_ns="",
        task_id="task-1",
        thread_id=thread_id,
    )


def fake_runtime(thread_id: str | None = None) -> Runtime[None]:
    return Runtime(execution_info=fake_execution_info(thread_id))


def memory_source_context(
    *,
    memory_dir,
    workspace_dir,
    source_type: MemorySourceType = MemorySourceType.TURN,
    project_id: str = "P-project",
    source_agent: str = "EvoScientist",
    session_id: str = "thread-1",
    trajectory: list[source_context.CompactMessage] | None = None,
) -> source_context.MemorySourceContext:
    context_trajectory = trajectory or [{"role": "human", "content": "hi"}]
    return source_context.MemorySourceContext(
        source_type=source_type,
        memory_dir=memory_dir,
        workspace_dir=workspace_dir,
        project_id=project_id,
        source_agent=source_agent,
        session_id=session_id,
        trajectory=context_trajectory,
        trajectory_digest=source_context._trajectory_digest(context_trajectory),
    )


def linker_context(
    *,
    memory_dir,
    workspace_dir,
    observation_ids: tuple[str, ...],
    project_id: str = "P-project",
) -> memory_scheduler.ObservationLinkerContext:
    return memory_scheduler.ObservationLinkerContext(
        memory_dir=memory_dir,
        workspace_dir=workspace_dir,
        project_id=project_id,
        observation_ids=observation_ids,
    )
