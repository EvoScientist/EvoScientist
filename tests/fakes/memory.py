"""Runtimes, tool helpers and source contexts for the memory pipeline tests."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.runtime import ExecutionInfo, Runtime

from EvoScientist.memory import scheduler as memory_scheduler
from EvoScientist.memory import source_context
from EvoScientist.memory.observations import MemoryScope, MemorySourceType, MemoryType


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


def tool_runtime(
    tool: Any,
    *,
    config: RunnableConfig | None = None,
    thread_id: str | None = None,
    tool_call_id: str | None = None,
) -> ToolRuntime:
    runtime_config: RunnableConfig = config if config is not None else {}
    return ToolRuntime(
        state={},
        context=None,
        config=runtime_config,
        stream_writer=lambda _chunk: None,
        tool_call_id=tool_call_id,
        store=None,
        tools=[tool],
        execution_info=fake_execution_info(thread_id),
        server_info=None,
    )


def record_observation_payload(
    tool: Any,
    *,
    runtime: ToolRuntime,
    memory_type: MemoryType,
    summary: str,
    observation: str,
    why_it_matters: str,
    scope: MemoryScope,
) -> dict[str, Any]:
    payload = tool.run(
        {
            "memory_type": memory_type,
            "summary": summary,
            "observation": observation,
            "why_it_matters": why_it_matters,
            "scope": scope,
            "runtime": runtime,
        }
    )
    return json.loads(payload)


def tool_by_name(tools: Sequence[BaseTool], name: str) -> BaseTool:
    matches = [tool for tool in tools if tool.name == name]
    assert len(matches) == 1
    return matches[0]
