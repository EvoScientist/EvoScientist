"""Middleware that schedules post-run EvoMemory workers."""

from __future__ import annotations

import asyncio
import logging
from functools import cache
from pathlib import Path
from typing import Annotated, NotRequired

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    PrivateStateAttr,
)
from langgraph.runtime import Runtime

from .. import paths as _paths
from ..memory.launch import (
    alaunch_memory_worker,
    launch_memory_worker,
    launch_observation_linker,
)
from ..memory.scheduler import MemoryScheduler
from ..memory.source_context import build_memory_source_context
from ..memory.types import MemorySourceType

logger = logging.getLogger(__name__)


@cache
def default_memory_scheduler() -> MemoryScheduler:
    return MemoryScheduler(launch_linker=launch_observation_linker)


class _LifecycleState(AgentState):
    # Id of the thread's last message when a run last reached this hook.
    # Private: sub-agents carry this middleware too.
    _memory_last_exit: NotRequired[Annotated[str | None, PrivateStateAttr]]


def _last_message_id(state: AgentState[object]) -> str | None:
    messages = state.get("messages") or []
    message_id = getattr(messages[-1], "id", None) if messages else None
    return message_id if isinstance(message_id, str) and message_id else None


class EvoMemoryLifecycleMiddleware(AgentMiddleware):
    """Schedule post-turn and post-subagent memory workers.

    A run that added nothing to the conversation (it ended in a start hook)
    launches no worker: its context would be the previous turn again.
    """

    name = "evomemory_lifecycle"
    state_schema = _LifecycleState

    def __init__(
        self,
        *,
        memory_dir: str | Path,
        workspace_dir: str | Path,
        project_id: str,
        source_type: MemorySourceType,
        source_agent: str,
        memory_scheduler: MemoryScheduler | None = None,
    ) -> None:
        self._memory_dir = Path(memory_dir).expanduser()
        self._workspace_dir = Path(workspace_dir).expanduser()
        self._project_id = project_id
        self._source_type = source_type
        self._source_agent = source_agent
        self._memory_scheduler = (
            memory_scheduler
            if memory_scheduler is not None
            else default_memory_scheduler()
        )

    def after_agent(
        self,
        state: AgentState[object],
        runtime: Runtime,
    ) -> dict[str, object] | None:
        last_id = _last_message_id(state)
        if last_id is not None and state.get("_memory_last_exit") == last_id:
            self._memory_scheduler.flush_ready()
            return None
        context = build_memory_source_context(
            state=state,
            runtime=runtime,
            memory_dir=self._memory_dir,
            workspace_dir=self._workspace_dir,
            project_id=self._project_id,
            source_type=self._source_type,
            source_agent=self._source_agent,
        )
        if context is not None:
            try:
                run = launch_memory_worker(
                    context,
                    on_worker_finished=self._memory_scheduler.record_worker_finished,
                    on_worker_aborted=self._memory_scheduler.record_worker_aborted,
                )
                if run is None:
                    self._memory_scheduler.flush_ready()
            except Exception:
                logger.warning("Failed to launch EvoMemory worker", exc_info=True)
                self._memory_scheduler.flush_ready()
        else:
            self._memory_scheduler.flush_ready()
        return {"_memory_last_exit": last_id} if last_id is not None else None

    async def aafter_agent(
        self,
        state: AgentState[object],
        runtime: Runtime,
    ) -> dict[str, object] | None:
        last_id = _last_message_id(state)
        if last_id is not None and state.get("_memory_last_exit") == last_id:
            await asyncio.to_thread(self._memory_scheduler.flush_ready)
            return None
        context = build_memory_source_context(
            state=state,
            runtime=runtime,
            memory_dir=self._memory_dir,
            workspace_dir=self._workspace_dir,
            project_id=self._project_id,
            source_type=self._source_type,
            source_agent=self._source_agent,
        )
        if context is not None:
            try:
                run = await alaunch_memory_worker(
                    context,
                    on_worker_finished=self._memory_scheduler.record_worker_finished,
                    on_worker_aborted=self._memory_scheduler.record_worker_aborted,
                )
                if run is None:
                    await asyncio.to_thread(self._memory_scheduler.flush_ready)
            except Exception:
                logger.warning("Failed to launch EvoMemory worker", exc_info=True)
                await asyncio.to_thread(self._memory_scheduler.flush_ready)
        else:
            await asyncio.to_thread(self._memory_scheduler.flush_ready)
        return {"_memory_last_exit": last_id} if last_id is not None else None


def create_memory_lifecycle_middleware(
    memory_dir: str | None = None,
    *,
    workspace_dir: str | Path,
    project_id: str,
    source_type: MemorySourceType,
    source_agent: str,
    memory_scheduler: MemoryScheduler | None = None,
) -> EvoMemoryLifecycleMiddleware:
    """Build the post-run EvoMemory lifecycle middleware."""

    if memory_dir is None:
        memory_dir = str(_paths.MEMORIES_DIR)
    return EvoMemoryLifecycleMiddleware(
        memory_dir=memory_dir,
        workspace_dir=workspace_dir,
        project_id=project_id,
        source_type=source_type,
        source_agent=source_agent,
        memory_scheduler=memory_scheduler,
    )
