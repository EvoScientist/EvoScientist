"""A run that added nothing to the conversation does not launch a memory worker."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from EvoScientist.memory.scheduler import MemoryScheduler
from EvoScientist.memory.types import MemorySourceType
from EvoScientist.middleware import memory_lifecycle
from tests.fakes.memory import fake_runtime

TURN = [HumanMessage("hi", id="h1"), AIMessage("done", name="EvoScientist", id="a1")]


class _Recorder:
    """Stands in for the worker launch and counts scheduler flushes."""

    def __init__(self, monkeypatch) -> None:
        self.launched: list = []
        self.flushed = 0

        def _launch(context, **_kwargs):
            self.launched.append(context)
            return object()

        async def _alaunch(context, **_kwargs):
            self.launched.append(context)
            return object()

        monkeypatch.setattr(memory_lifecycle, "launch_memory_worker", _launch)
        monkeypatch.setattr(memory_lifecycle, "alaunch_memory_worker", _alaunch)
        scheduler = MemoryScheduler(launch_linker=lambda _context: None)
        original = scheduler.flush_ready

        def _flush():
            self.flushed += 1
            return original()

        scheduler.flush_ready = _flush
        self.scheduler = scheduler


def _middleware(tmp_path, recorder, source_type=MemorySourceType.TURN):
    return memory_lifecycle.EvoMemoryLifecycleMiddleware(
        memory_dir=tmp_path / "memories",
        workspace_dir=tmp_path / "workspace",
        project_id="P-project",
        source_type=source_type,
        source_agent="EvoScientist",
        memory_scheduler=recorder.scheduler,
    )


@pytest.fixture
def recorder(monkeypatch):
    return _Recorder(monkeypatch)


async def _exit(middleware, state, *, sync):
    runtime = fake_runtime("thread-1")
    if sync:
        return middleware.after_agent(state, runtime)
    return await middleware.aafter_agent(state, runtime)


@pytest.mark.parametrize("sync", [True, False])
async def test_a_turn_launches_a_worker_and_remembers_where_it_ended(
    tmp_path, recorder, sync
):
    middleware = _middleware(tmp_path, recorder)
    update = await _exit(middleware, {"messages": TURN}, sync=sync)
    assert len(recorder.launched) == 1
    assert update == {"_memory_last_exit": "a1"}


@pytest.mark.parametrize("sync", [True, False])
async def test_a_run_that_added_nothing_launches_no_worker(tmp_path, recorder, sync):
    """A ``/goal`` request that is cancelled, or one that only changes the goal's status."""
    middleware = _middleware(tmp_path, recorder)
    state = {"messages": TURN, "_memory_last_exit": "a1"}
    update = await _exit(middleware, state, sync=sync)
    assert recorder.launched == []
    assert update is None
    # The scheduler is still given its chance to run what is ready.
    assert recorder.flushed == 1


async def test_the_next_real_turn_launches_again(tmp_path, recorder):
    middleware = _middleware(tmp_path, recorder)
    messages = [
        *TURN,
        HumanMessage("more", id="h2"),
        AIMessage("ok", name="EvoScientist", id="a2"),
    ]
    update = await _exit(
        middleware, {"messages": messages, "_memory_last_exit": "a1"}, sync=False
    )
    assert len(recorder.launched) == 1
    assert update == {"_memory_last_exit": "a2"}


async def test_a_thread_written_before_the_mark_existed_launches_once(
    tmp_path, recorder
):
    middleware = _middleware(tmp_path, recorder)
    first = await _exit(middleware, {"messages": TURN}, sync=False)
    second = await _exit(middleware, {"messages": TURN, **first}, sync=False)
    assert len(recorder.launched) == 1
    assert second is None


async def test_the_mark_is_private_to_the_graph(tmp_path, recorder):
    """The middleware is on sync sub-agents too; a public key would come back
    from two of them into one step of the parent."""
    from typing import get_args, get_type_hints

    from langchain.agents.middleware.types import PrivateStateAttr

    schema = _middleware(tmp_path, recorder).state_schema
    annotation = get_type_hints(schema, include_extras=True)["_memory_last_exit"]
    while not hasattr(annotation, "__metadata__") and get_args(annotation):
        annotation = get_args(annotation)[0]
    assert PrivateStateAttr in annotation.__metadata__
