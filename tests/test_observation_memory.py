from __future__ import annotations

import json
import re
import threading
from collections.abc import Sequence
from datetime import datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml
from blockbuster import BlockBuster
from langchain.agents.middleware.types import AgentState
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.runtime import ExecutionInfo, Runtime
from pydantic import BaseModel

from EvoScientist.config import EvoScientistConfig, MemoryObservationWriter
from EvoScientist.gateway import background_runs
from EvoScientist.memory import (
    launch as memory_launch,
)
from EvoScientist.memory import (
    scheduler as memory_scheduler,
)
from EvoScientist.memory import (
    source_context,
    worker_activity,
)
from EvoScientist.memory.agents import memory_worker, observation_linker
from EvoScientist.memory.observations import (
    MemoryScope,
    MemorySourceType,
    MemoryType,
    ObservationSearchMode,
    create_link_observations_tool,
    create_read_memory_tool,
    create_search_observations_tool,
    link_observation_files,
    list_observation_documents,
    read_observation_document,
    read_observation_file,
    read_observation_id_from_path,
    record_observation_file,
    search_observation_files,
)
from EvoScientist.memory.types import ObservationRelation
from EvoScientist.middleware import memory_lifecycle


def _read_memory_document(path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    frontmatter, body = text.removeprefix("---\n").split("\n---\n", 1)
    metadata = yaml.safe_load(frontmatter)
    assert isinstance(metadata, dict)
    return metadata, body


def _stable_created_at(metadata: dict[str, Any]) -> dict[str, Any]:
    created_at = metadata.get("created_at")
    assert isinstance(created_at, str)
    datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
    return {**metadata, "created_at": "<created_at>"}


def _markdown_sections(body: str) -> dict[str, str]:
    matches = list(re.finditer(r"^## (?P<title>.+)$", body, flags=re.MULTILINE))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        sections[match.group("title")] = body[start:end].strip()
    return sections


def _execution_info(thread_id: str | None = None) -> ExecutionInfo:
    return ExecutionInfo(
        checkpoint_id="checkpoint-1",
        checkpoint_ns="",
        task_id="task-1",
        thread_id=thread_id,
    )


def _tool_runtime(
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
        execution_info=_execution_info(thread_id),
        server_info=None,
    )


def _runtime(thread_id: str | None = None) -> Runtime[None]:
    return Runtime(execution_info=_execution_info(thread_id))


def _memory_source_context(
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


def _memory_worker_run(
    *,
    thread_id: str = "worker-thread",
    run_id: str = "run-1",
    workspace_dir: str = "/tmp/ws",
    project_id: str = "P-project",
    source_agent: str = "EvoScientist",
    source_session_id: str = "thread-1",
    trajectory_digest: str = "digest-1",
) -> background_runs.BackgroundRun:
    return background_runs.BackgroundRun(
        name="EvoMemory worker",
        url="http://x",
        graph_id=memory_launch.TURN_MEMORY_WORKER_GRAPH_ID,
        thread_id=thread_id,
        run_id=run_id,
        assistant_id=memory_launch.TURN_MEMORY_WORKER_GRAPH_ID,
        metadata={
            "workspace_dir": workspace_dir,
            "project_id": project_id,
            "source_agent": source_agent,
            "source_session_id": source_session_id,
            "trajectory_digest": trajectory_digest,
        },
    )


def _observation_linker_run(
    *,
    thread_id: str = "linker-thread",
    run_id: str = "linker-run",
) -> background_runs.BackgroundRun:
    return background_runs.BackgroundRun(
        name="EvoMemory observation linker",
        url="http://x",
        graph_id=memory_launch.OBSERVATION_LINKER_GRAPH_ID,
        thread_id=thread_id,
        run_id=run_id,
        assistant_id=memory_launch.OBSERVATION_LINKER_GRAPH_ID,
        metadata={},
    )


def _linker_context(
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


def _mark_worker_started(
    memory_dir,
    *,
    thread_id: str = "worker-thread",
    run_id: str = "run-1",
    before_outputs: worker_activity.MemoryOutputSnapshot | None = None,
) -> None:
    worker_activity.mark_memory_worker_started(
        thread_id=thread_id,
        run_id=run_id,
        memory_dir=memory_dir,
        before_outputs=(
            before_outputs
            if before_outputs is not None
            else worker_activity.snapshot_memory_outputs(memory_dir)
        ),
    )


def _record_test_observation(
    memory_dir,
    *,
    summary: str = "Durable test observation.",
    observation: str = "A reusable test observation.",
    scope: MemoryScope = MemoryScope.GLOBAL,
) -> dict[str, Any]:
    return record_observation_file(
        memory_dir=memory_dir,
        project_id="P-project",
        memory_type=MemoryType.PROCEDURAL,
        summary=summary,
        observation=observation,
        why_it_matters=f"Future agents can use this test memory: {summary}",
        scope=scope,
        source_type=MemorySourceType.TURN,
        source_session_id="thread-1",
        source_agent="EvoScientist",
    )


def _memory_relative_path(record: dict[str, Any]) -> str:
    return record["path"].removeprefix("/memories/")


def _record_observation_payload(
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


def _tool_by_name(tools: Sequence[BaseTool], name: str) -> BaseTool:
    matches = [tool for tool in tools if tool.name == name]
    assert len(matches) == 1
    return matches[0]


def _fast_watcher_config(
    *, max_poll_failures: int = 3
) -> background_runs.BackgroundRunWatcherConfig:
    return background_runs.BackgroundRunWatcherConfig(
        poll_interval_seconds=0,
        max_poll_failures=max_poll_failures,
    )


@pytest.fixture(autouse=True)
def _reset_memory_activity():
    worker_activity.reset_memory_worker_status_for_tests()
    yield
    worker_activity.reset_memory_worker_status_for_tests()


def test_record_observation_file_writes_contract_and_dedupes(tmp_path):
    memories = tmp_path / "memories"
    summary = "Focused pytest catches local regressions before broader runs."
    observation = "Run pytest with the focused file before the full suite."
    why_it_matters = "This catches local regressions faster."
    evidence = "Command: uv run pytest tests/test_observation_memory.py"

    first = record_observation_file(
        memory_dir=memories,
        project_id="P-project",
        memory_type=MemoryType.PROCEDURAL,
        summary=summary,
        observation=observation,
        why_it_matters=why_it_matters,
        evidence=evidence,
        scope=MemoryScope.PROJECT,
        source_type=MemorySourceType.SUBAGENT,
        source_session_id="thread-1",
        source_agent="code-agent",
    )
    second = record_observation_file(
        memory_dir=memories,
        project_id="P-project",
        memory_type=MemoryType.PROCEDURAL,
        summary=summary,
        observation=observation,
        why_it_matters=why_it_matters,
        evidence=evidence,
        scope=MemoryScope.PROJECT,
        source_type=MemorySourceType.SUBAGENT,
        source_session_id="thread-1",
        source_agent="code-agent",
    )

    path = memories / first["path"].removeprefix("/memories/")
    metadata, body = _read_memory_document(path)

    assert first["created"] is True
    assert second == {**first, "created": False}
    assert first["path"] == (
        f"/memories/observations/projects/P-project/{first['observation_id']}.md"
    )
    assert _stable_created_at(metadata) == {
        "id": first["observation_id"],
        "created_at": "<created_at>",
        "summary": summary,
        "memory_type": "procedural",
        "scope": "project",
        "project_id": "P-project",
        "source": {
            "type": "subagent",
            "agent": "code-agent",
            "session_id": "thread-1",
        },
    }
    assert _markdown_sections(body) == {
        "Observation": observation,
        "Why It Matters": why_it_matters,
        "Evidence": evidence,
    }


def test_link_observation_files_writes_frontmatter_and_dedupes(tmp_path):
    memories = tmp_path / "memories"
    first = record_observation_file(
        memory_dir=memories,
        project_id="P-project",
        memory_type=MemoryType.PROCEDURAL,
        summary="Graph gateway launches background runs.",
        observation="Use the graph gateway background run service for workers.",
        why_it_matters="Future launchers avoid duplicating SDK plumbing.",
        scope=MemoryScope.PROJECT,
        source_type=MemorySourceType.TURN,
        source_session_id="thread-1",
        source_agent="EvoScientist",
    )
    second = record_observation_file(
        memory_dir=memories,
        project_id="P-project",
        memory_type=MemoryType.PROCEDURAL,
        summary="Memory linkers should update metadata.",
        observation="Observation links belong in frontmatter metadata.",
        why_it_matters="Future indexing can consume links without parsing prose.",
        scope=MemoryScope.PROJECT,
        source_type=MemorySourceType.TURN,
        source_session_id="thread-1",
        source_agent="EvoScientist",
    )
    first_path = memories / first["path"].removeprefix("/memories/")
    second_path = memories / second["path"].removeprefix("/memories/")
    _first_metadata, first_body_before = _read_memory_document(first_path)
    _second_metadata, second_body_before = _read_memory_document(second_path)

    result = link_observation_files(
        memory_dir=memories,
        project_id="P-project",
        source_observation_id=first["observation_id"],
        target_observation_id=second["observation_id"],
        relation=ObservationRelation.COMPLEMENTS,
        reason="Both observations describe the durable background-memory flow.",
    )
    duplicate = link_observation_files(
        memory_dir=memories,
        project_id="P-project",
        source_observation_id=first["observation_id"],
        target_observation_id=second["observation_id"],
        relation=ObservationRelation.COMPLEMENTS,
        reason="Both observations describe the durable background-memory flow.",
    )

    first_metadata, first_body_after = _read_memory_document(first_path)
    second_metadata, second_body_after = _read_memory_document(second_path)
    assert result == {
        "linked": True,
        "source_observation_id": first["observation_id"],
        "target_observation_id": second["observation_id"],
        "relation": "complements",
        "updated_observation_ids": [
            first["observation_id"],
            second["observation_id"],
        ],
        "missing_observation_ids": [],
    }
    assert duplicate == {
        **result,
        "linked": False,
        "updated_observation_ids": [],
    }
    assert first_body_after == first_body_before
    assert second_body_after == second_body_before
    first_links = first_metadata["related_observations"]
    second_links = second_metadata["related_observations"]
    assert len(first_links) == 1
    assert len(second_links) == 1
    assert first_links[0] == {
        "id": second["observation_id"],
        "relation": "complements",
        "reason": "Both observations describe the durable background-memory flow.",
        "linked_at": first_links[0]["linked_at"],
    }
    assert second_links[0] == {
        "id": first["observation_id"],
        "relation": "complements",
        "reason": "Both observations describe the durable background-memory flow.",
        "linked_at": first_links[0]["linked_at"],
    }
    datetime.strptime(first_links[0]["linked_at"], "%Y-%m-%dT%H:%M:%SZ")


def test_link_observation_files_serializes_concurrent_frontmatter_updates(tmp_path):
    memories = tmp_path / "memories"
    source = record_observation_file(
        memory_dir=memories,
        project_id="P-project",
        memory_type=MemoryType.SEMANTIC,
        summary="Source observation for concurrent links.",
        observation="Several linker workers may update this observation.",
        why_it_matters="Concurrent linkers must not lose frontmatter updates.",
        scope=MemoryScope.PROJECT,
        source_type=MemorySourceType.TURN,
        source_session_id="thread-1",
        source_agent="EvoScientist",
    )
    targets = [
        record_observation_file(
            memory_dir=memories,
            project_id="P-project",
            memory_type=MemoryType.SEMANTIC,
            summary=f"Target observation {index}.",
            observation=f"Concurrent target observation {index}.",
            why_it_matters=f"Target {index} should remain linked.",
            scope=MemoryScope.PROJECT,
            source_type=MemorySourceType.TURN,
            source_session_id="thread-1",
            source_agent="EvoScientist",
        )
        for index in range(12)
    ]
    barrier = threading.Barrier(len(targets))
    errors: list[Exception] = []

    def link_target(index: int, target: dict[str, Any]) -> None:
        try:
            barrier.wait(timeout=5)
            link_observation_files(
                memory_dir=memories,
                project_id="P-project",
                source_observation_id=source["observation_id"],
                target_observation_id=target["observation_id"],
                relation=ObservationRelation.COMPLEMENTS,
                reason=f"Target {index} is relevant to the shared source.",
                bidirectional=False,
            )
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=link_target, args=(index, target))
        for index, target in enumerate(targets)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    source_path = memories / source["path"].removeprefix("/memories/")
    source_metadata, _source_body = _read_memory_document(source_path)
    linked_ids = {entry["id"] for entry in source_metadata["related_observations"]}
    assert linked_ids == {target["observation_id"] for target in targets}


def test_read_and_search_surface_related_observations(tmp_path):
    memories = tmp_path / "memories"
    source = record_observation_file(
        memory_dir=memories,
        project_id="P-project",
        memory_type=MemoryType.PROCEDURAL,
        summary="Gateway memory workers preserve launch metadata.",
        observation="Use the gateway service when launching memory workers.",
        why_it_matters="Future launchers should reuse the same async-run plumbing.",
        scope=MemoryScope.PROJECT,
        source_type=MemorySourceType.TURN,
        source_session_id="thread-1",
        source_agent="EvoScientist",
    )
    target = record_observation_file(
        memory_dir=memories,
        project_id="P-project",
        memory_type=MemoryType.SEMANTIC,
        summary="Observation links should be visible during retrieval.",
        observation="Related observations need to surface in memory tool results.",
        why_it_matters="Future agents can use existing links without parsing YAML.",
        scope=MemoryScope.PROJECT,
        source_type=MemorySourceType.TURN,
        source_session_id="thread-1",
        source_agent="EvoScientist",
    )
    link_observation_files(
        memory_dir=memories,
        project_id="P-project",
        source_observation_id=source["observation_id"],
        target_observation_id=target["observation_id"],
        relation=ObservationRelation.COMPLEMENTS,
        reason="Launch metadata and retrieval visibility describe the same memory pipeline.",
    )
    read = read_observation_file(
        memory_dir=memories,
        project_id="P-project",
        observation_id=source["observation_id"],
    )
    hits = search_observation_files(
        memory_dir=memories,
        project_id="P-project",
        query="gateway memory workers launch metadata",
    )

    assert read is not None
    assert read["related_observations"][0]["observation_id"] == target["observation_id"]
    assert (
        read["related_observations"][0]["relation"] == ObservationRelation.COMPLEMENTS
    )
    assert hits[0]["observation_id"] == source["observation_id"]
    assert (
        hits[0]["related_observations"][0]["observation_id"] == target["observation_id"]
    )

    read_tool = create_read_memory_tool(memory_dir=memories, project_id="P-project")
    read_payload = json.loads(
        read_tool.run({"observation_id": source["observation_id"]})
    )
    search_tool = create_search_observations_tool(
        memory_dir=memories,
        project_id="P-project",
    )
    search_payload = json.loads(
        search_tool.run({"query": "gateway memory workers launch metadata"})
    )
    assert (
        read_payload["related_observations"][0]["observation_id"]
        == target["observation_id"]
    )
    assert (
        search_payload["results"][0]["related_observations"][0]["observation_id"]
        == target["observation_id"]
    )


def test_read_and_search_resolve_related_observations_from_other_projects(tmp_path):
    memories = tmp_path / "memories"
    source = record_observation_file(
        memory_dir=memories,
        project_id="P-current",
        memory_type=MemoryType.PROCEDURAL,
        summary="Global linker practice applies across projects.",
        observation="Global observations may link to project-specific follow-ups.",
        why_it_matters="Related observations should remain visible from other projects.",
        scope=MemoryScope.GLOBAL,
        source_type=MemorySourceType.TURN,
        source_session_id="thread-1",
        source_agent="EvoScientist",
    )
    target = record_observation_file(
        memory_dir=memories,
        project_id="P-other",
        memory_type=MemoryType.SEMANTIC,
        summary="Other project follow-up explains the linker practice.",
        observation="A separate project can hold the concrete follow-up observation.",
        why_it_matters="Global observations should surface explicitly linked project memories.",
        scope=MemoryScope.PROJECT,
        source_type=MemorySourceType.TURN,
        source_session_id="thread-2",
        source_agent="EvoScientist",
    )
    link_observation_files(
        memory_dir=memories,
        project_id="P-other",
        source_observation_id=source["observation_id"],
        target_observation_id=target["observation_id"],
        relation=ObservationRelation.COMPLEMENTS,
        reason="The other project gives a concrete follow-up for the global practice.",
        bidirectional=False,
    )

    read = read_observation_file(
        memory_dir=memories,
        project_id="P-current",
        observation_id=source["observation_id"],
    )
    hits = search_observation_files(
        memory_dir=memories,
        project_id="P-current",
        query="global linker practice",
    )

    assert read is not None
    assert read["related_observations"][0]["observation_id"] == target["observation_id"]
    assert (
        hits[0]["related_observations"][0]["observation_id"] == target["observation_id"]
    )


def test_malformed_observation_frontmatter_is_skipped(tmp_path):
    memories = tmp_path / "memories"
    valid = record_observation_file(
        memory_dir=memories,
        project_id="P-project",
        memory_type=MemoryType.SEMANTIC,
        summary="Valid observations remain searchable.",
        observation="A malformed neighboring observation file must not break search.",
        why_it_matters="One bad memory file should not hide the rest of memory.",
        scope=MemoryScope.PROJECT,
        source_type=MemorySourceType.TURN,
        source_session_id="thread-1",
        source_agent="EvoScientist",
    )
    global_dir = memories / "observations" / "global"
    global_dir.mkdir(parents=True, exist_ok=True)
    missing_id = global_dir / "missing-id.md"
    missing_id.write_text(
        "---\n"
        "summary: Missing id should skip this file\n"
        "memory_type: semantic\n"
        "scope: global\n"
        "---\n"
        "Body\n",
        encoding="utf-8",
    )
    bad_link = global_dir / "bad-link.md"
    bad_link.write_text(
        "---\n"
        'id: "O-bad-link"\n'
        'summary: "Invalid relation entries should be ignored"\n'
        "memory_type: semantic\n"
        "scope: global\n"
        "related_observations:\n"
        '  - id: "O-target"\n'
        '    relation: "unbounded"\n'
        '    reason: "not a supported relation"\n'
        '    linked_at: "2026-06-25T00:00:00Z"\n'
        "---\n"
        "Body\n",
        encoding="utf-8",
    )

    assert read_observation_id_from_path(missing_id) is None
    hits = search_observation_files(
        memory_dir=memories,
        project_id="P-project",
        query="malformed neighboring observation",
    )
    assert [hit["observation_id"] for hit in hits] == [valid["observation_id"]]
    assert worker_activity.snapshot_observation_relations(memories) == frozenset()


def test_legacy_observation_source_without_session_id_still_reads(tmp_path):
    memories = tmp_path / "memories"
    global_dir = memories / "observations" / "global"
    global_dir.mkdir(parents=True)
    legacy = global_dir / "O-legacy.md"
    legacy.write_text(
        "---\n"
        "id: O-legacy\n"
        "created_at: 2026-01-01T00:00:00Z\n"
        "summary: Legacy observation without session id.\n"
        "memory_type: procedural\n"
        "scope: global\n"
        "source:\n"
        "  type: turn\n"
        "  agent: EvoScientist\n"
        "---\n"
        "Legacy body text.\n",
        encoding="utf-8",
    )

    documents = list_observation_documents(
        memory_dir=memories,
        project_id="P-project",
    )
    read = read_observation_file(
        memory_dir=memories,
        project_id="P-project",
        observation_id="O-legacy",
    )
    hits = search_observation_files(
        memory_dir=memories,
        project_id="P-project",
        query="Legacy body",
    )

    assert [document.observation_id for document in documents] == ["O-legacy"]
    assert read is not None
    assert read["observation_id"] == "O-legacy"
    assert hits[0]["observation_id"] == "O-legacy"


def test_unquoted_naive_yaml_timestamp_does_not_claim_utc(tmp_path):
    observation = tmp_path / "O-naive.md"
    observation.write_text(
        "---\n"
        "id: O-naive\n"
        "created_at: 2026-01-01 12:30:00\n"
        "summary: Legacy observation with naive YAML timestamp.\n"
        "memory_type: procedural\n"
        "scope: global\n"
        "source:\n"
        "  type: turn\n"
        "  agent: EvoScientist\n"
        "  session_id: thread-1\n"
        "---\n"
        "Legacy body text.\n",
        encoding="utf-8",
    )

    document = read_observation_document(observation)

    assert document is not None
    metadata, _body = document
    assert metadata.created_at == "2026-01-01T12:30:00"


def test_unquoted_aware_yaml_timestamp_normalizes_to_utc(tmp_path):
    observation = tmp_path / "O-aware.md"
    observation.write_text(
        "---\n"
        "id: O-aware\n"
        "created_at: 2026-01-01 12:30:00+02:00\n"
        "summary: Legacy observation with aware YAML timestamp.\n"
        "memory_type: procedural\n"
        "scope: global\n"
        "source:\n"
        "  type: turn\n"
        "  agent: EvoScientist\n"
        "  session_id: thread-1\n"
        "---\n"
        "Legacy body text.\n",
        encoding="utf-8",
    )

    document = read_observation_document(observation)

    assert document is not None
    metadata, _body = document
    assert metadata.created_at == "2026-01-01T10:30:00Z"


def test_link_observation_files_keeps_supersedes_directional(tmp_path):
    memories = tmp_path / "memories"
    source = record_observation_file(
        memory_dir=memories,
        project_id="P-project",
        memory_type=MemoryType.SEMANTIC,
        summary="New memory replaces older guidance.",
        observation="Use the newer observation as the current guidance.",
        why_it_matters="Future agents should prefer the replacement guidance.",
        scope=MemoryScope.PROJECT,
        source_type=MemorySourceType.TURN,
        source_session_id="thread-1",
        source_agent="EvoScientist",
    )
    target = record_observation_file(
        memory_dir=memories,
        project_id="P-project",
        memory_type=MemoryType.SEMANTIC,
        summary="Older guidance is superseded.",
        observation="This older observation should no longer be preferred.",
        why_it_matters="Future agents need to avoid stale guidance.",
        scope=MemoryScope.PROJECT,
        source_type=MemorySourceType.TURN,
        source_session_id="thread-1",
        source_agent="EvoScientist",
    )

    result = link_observation_files(
        memory_dir=memories,
        project_id="P-project",
        source_observation_id=source["observation_id"],
        target_observation_id=target["observation_id"],
        relation=ObservationRelation.SUPERSEDES,
        reason="The source observation replaces the target observation.",
    )

    source_metadata, _source_body = _read_memory_document(
        memories / source["path"].removeprefix("/memories/")
    )
    target_metadata, _target_body = _read_memory_document(
        memories / target["path"].removeprefix("/memories/")
    )
    assert result["updated_observation_ids"] == [source["observation_id"]]
    assert source_metadata["related_observations"] == [
        {
            "id": target["observation_id"],
            "relation": "supersedes",
            "reason": "The source observation replaces the target observation.",
            "linked_at": source_metadata["related_observations"][0]["linked_at"],
        }
    ]
    assert "related_observations" not in target_metadata


def test_link_observation_files_rejects_unknown_relation(tmp_path):
    memories = tmp_path / "memories"

    with pytest.raises(ValueError, match="relation must be one of"):
        link_observation_files(
            memory_dir=memories,
            project_id="P-project",
            source_observation_id="O-source",
            target_observation_id="O-target",
            relation="overlaps",
            reason="This unsupported relation should be rejected.",
        )


def test_link_observations_tool_uses_runtime_project_id(tmp_path):
    memories = tmp_path / "memories"
    first = record_observation_file(
        memory_dir=memories,
        project_id="P-runtime",
        memory_type=MemoryType.SEMANTIC,
        summary="Runtime project id selects project memory.",
        observation="Linking tools should honor runtime project ids.",
        why_it_matters="Shared graph builds can still handle project memory.",
        scope=MemoryScope.PROJECT,
        source_type=MemorySourceType.TURN,
        source_session_id="thread-1",
        source_agent="EvoScientist",
    )
    second = record_observation_file(
        memory_dir=memories,
        project_id="P-runtime",
        memory_type=MemoryType.SEMANTIC,
        summary="Observation links live in frontmatter.",
        observation="Frontmatter links are machine-readable.",
        why_it_matters="Future status and search features can use metadata.",
        scope=MemoryScope.PROJECT,
        source_type=MemorySourceType.TURN,
        source_session_id="thread-1",
        source_agent="EvoScientist",
    )
    tool = create_link_observations_tool(
        memory_dir=memories,
        project_id="wrong-project",
    )
    runtime = _tool_runtime(
        tool,
        config={"configurable": {"evomemory_project_id": "P-runtime"}},
    )

    payload = json.loads(
        tool.run(
            {
                "source_observation_id": first["observation_id"],
                "target_observation_id": second["observation_id"],
                "reason": "Both validate frontmatter-native linker behavior.",
                "runtime": runtime,
            }
        )
    )

    assert payload["linked"] is True
    metadata, _body = _read_memory_document(
        memories / first["path"].removeprefix("/memories/")
    )
    assert metadata["related_observations"][0]["id"] == second["observation_id"]


def test_observation_linker_finish_counts_successful_relations_once(tmp_path):
    memories = tmp_path / "memories"
    first = record_observation_file(
        memory_dir=memories,
        project_id="P-project",
        memory_type=MemoryType.SEMANTIC,
        summary="Linked relation counts are status-bar outcomes.",
        observation="Successful link_observations calls should count relations.",
        why_it_matters="The status bar should report durable link outcomes.",
        scope=MemoryScope.PROJECT,
        source_type=MemorySourceType.TURN,
        source_session_id="thread-1",
        source_agent="EvoScientist",
    )
    second = record_observation_file(
        memory_dir=memories,
        project_id="P-project",
        memory_type=MemoryType.SEMANTIC,
        summary="Duplicate relation calls should be no-ops.",
        observation="Duplicate link_observations calls should not recount links.",
        why_it_matters="Relation counts should reflect actual metadata updates.",
        scope=MemoryScope.PROJECT,
        source_type=MemorySourceType.TURN,
        source_session_id="thread-1",
        source_agent="EvoScientist",
    )
    run = _observation_linker_run()
    hooks = memory_launch._observation_linker_launch_hooks(memories)

    assert hooks.on_before_run is not None
    assert hooks.on_started is not None
    assert hooks.on_finished is not None
    hooks.on_before_run(run.thread_id)
    hooks.on_started(run)
    first_payload = link_observation_files(
        memory_dir=memories,
        project_id="P-project",
        source_observation_id=first["observation_id"],
        target_observation_id=second["observation_id"],
        reason="Both validate status counting for durable links.",
    )
    duplicate_payload = link_observation_files(
        memory_dir=memories,
        project_id="P-project",
        source_observation_id=first["observation_id"],
        target_observation_id=second["observation_id"],
        reason="Both validate status counting for durable links.",
    )
    hooks.on_finished(run)
    status = worker_activity.observation_linker_status()

    assert first_payload["linked"] is True
    assert duplicate_payload["linked"] is False
    assert status.relations_linked == 1


def test_observation_linker_finish_does_not_count_reason_only_updates(tmp_path):
    memories = tmp_path / "memories"
    first = record_observation_file(
        memory_dir=memories,
        project_id="P-project",
        memory_type=MemoryType.SEMANTIC,
        summary="Relation identity ignores rationale text.",
        observation="Changing a relation reason is not a newly created link.",
        why_it_matters="The status bar should not inflate memory-link counts.",
        scope=MemoryScope.PROJECT,
        source_type=MemorySourceType.TURN,
        source_session_id="thread-1",
        source_agent="EvoScientist",
    )
    second = record_observation_file(
        memory_dir=memories,
        project_id="P-project",
        memory_type=MemoryType.SEMANTIC,
        summary="Existing relation can receive a better rationale.",
        observation="Linker reruns may refine the reason for an existing relation.",
        why_it_matters="Refined metadata should not look like a new edge.",
        scope=MemoryScope.PROJECT,
        source_type=MemorySourceType.TURN,
        source_session_id="thread-1",
        source_agent="EvoScientist",
    )
    link_observation_files(
        memory_dir=memories,
        project_id="P-project",
        source_observation_id=first["observation_id"],
        target_observation_id=second["observation_id"],
        reason="Initial rationale for the existing relation.",
    )
    run = _observation_linker_run()
    hooks = memory_launch._observation_linker_launch_hooks(memories)

    assert hooks.on_before_run is not None
    assert hooks.on_started is not None
    assert hooks.on_finished is not None
    hooks.on_before_run(run.thread_id)
    hooks.on_started(run)
    payload = link_observation_files(
        memory_dir=memories,
        project_id="P-project",
        source_observation_id=first["observation_id"],
        target_observation_id=second["observation_id"],
        reason="Updated rationale for the existing relation.",
    )
    hooks.on_finished(run)

    assert payload["linked"] is True
    assert worker_activity.observation_linker_status().relations_linked == 0


def test_read_and_search_observation_tools_use_runtime_project_id(tmp_path):
    memories = tmp_path / "memories"
    observation = record_observation_file(
        memory_dir=memories,
        project_id="P-runtime",
        memory_type=MemoryType.SEMANTIC,
        summary="Runtime project id selects observation reads.",
        observation="Read and search tools should honor runtime project ids.",
        why_it_matters="Shared graph builds can still inspect project memory.",
        scope=MemoryScope.PROJECT,
        source_type=MemorySourceType.TURN,
        source_session_id="thread-1",
        source_agent="EvoScientist",
    )

    search_tool = create_search_observations_tool(
        memory_dir=memories,
        project_id="wrong-project",
    )
    search_runtime = _tool_runtime(
        search_tool,
        config={"configurable": {"evomemory_project_id": "P-runtime"}},
    )
    search_payload = json.loads(
        search_tool.run(
            {
                "query": "runtime project observation reads",
                "scope": MemoryScope.PROJECT,
                "runtime": search_runtime,
            }
        )
    )
    assert [hit["observation_id"] for hit in search_payload["results"]] == [
        observation["observation_id"]
    ]

    read_tool = create_read_memory_tool(
        memory_dir=memories,
        project_id="wrong-project",
    )
    read_runtime = _tool_runtime(
        read_tool,
        config={"configurable": {"evomemory_project_id": "P-runtime"}},
    )
    read_payload = json.loads(
        read_tool.run(
            {
                "observation_id": observation["observation_id"],
                "runtime": read_runtime,
            }
        )
    )
    assert (
        "Read and search tools should honor runtime project ids."
        in read_payload["text"]
    )


def test_search_observation_files_returns_ranked_keyword_hits(tmp_path):
    memories = tmp_path / "memories"
    first = record_observation_file(
        memory_dir=memories,
        project_id="P-project",
        memory_type=MemoryType.PROCEDURAL,
        summary="GraphQL resolver aliases preserve userName fields.",
        observation=(
            "When GraphQL returns blank camelCase fields, inspect resolver "
            "aliases before changing the frontend query."
        ),
        why_it_matters="Future profile tasks can avoid frontend-only fixes.",
        scope=MemoryScope.GLOBAL,
        source_type=MemorySourceType.SUBAGENT,
        source_session_id="thread-1",
        source_agent="code-agent",
    )
    second = record_observation_file(
        memory_dir=memories,
        project_id="P-project",
        memory_type=MemoryType.SEMANTIC,
        summary="CSV date normalization can change ordering.",
        observation="Normalize date strings before sorting cross-source reports.",
        why_it_matters="Future data tasks should avoid lexicographic date sorting.",
        scope=MemoryScope.PROJECT,
        source_type=MemorySourceType.SUBAGENT,
        source_session_id="thread-1",
        source_agent="data-agent",
    )

    hits = search_observation_files(
        memory_dir=memories,
        project_id="P-project",
        query="GraphQL userName frontend",
        limit=5,
    )

    assert [hit["observation_id"] for hit in hits] == [first["observation_id"]]
    assert hits[0]["path"] == first["path"]
    assert hits[0]["memory_type"] == MemoryType.PROCEDURAL
    assert hits[0]["scope"] == MemoryScope.GLOBAL
    assert hits[0]["summary"] == "GraphQL resolver aliases preserve userName fields."
    assert hits[0]["matches"] == [
        (
            "When GraphQL returns blank camelCase fields, inspect resolver aliases "
            "before changing the frontend query."
        ),
        "Future profile tasks can avoid frontend-only fixes.",
    ]
    assert hits[0]["score"] > 0
    assert (
        search_observation_files(
            memory_dir=memories,
            project_id="P-project",
            query="date|sorting",
            scope=MemoryScope.PROJECT,
            memory_type=MemoryType.SEMANTIC,
        )[0]["observation_id"]
        == second["observation_id"]
    )

    tool = create_search_observations_tool(
        memory_dir=memories,
        project_id="P-project",
    )
    payload = json.loads(tool.run({"query": "GraphQL userName frontend", "limit": 5}))
    assert list(payload) == ["results"]
    assert payload["results"][0]["observation_id"] == first["observation_id"]


def test_read_memory_returns_full_observation_by_id(tmp_path):
    memories = tmp_path / "memories"
    observation = "Read the full observation before applying a partial snippet."
    result = record_observation_file(
        memory_dir=memories,
        project_id="P-project",
        memory_type=MemoryType.PROCEDURAL,
        summary="Full memory reads prevent acting on partial snippets.",
        observation=observation,
        why_it_matters="Future agents can inspect the full rationale before editing.",
        scope=MemoryScope.PROJECT,
        source_type=MemorySourceType.SUBAGENT,
        source_session_id="thread-1",
        source_agent="code-agent",
    )

    read = read_observation_file(
        memory_dir=memories,
        project_id="P-project",
        observation_id=result["observation_id"],
    )

    assert read is not None
    assert read["observation_id"] == result["observation_id"]
    assert read["path"] == result["path"]
    assert read["memory_type"] == MemoryType.PROCEDURAL
    assert read["scope"] == MemoryScope.PROJECT
    assert read["summary"] == "Full memory reads prevent acting on partial snippets."
    assert read["text"].startswith("---\n")
    assert observation in read["text"]

    tool = create_read_memory_tool(memory_dir=memories, project_id="P-project")
    payload = json.loads(tool.run({"observation_id": result["observation_id"]}))
    assert payload == {"text": read["text"]}

    missing = json.loads(tool.run({"observation_id": "../not-a-memory"}))
    assert missing == {
        "error": "No observation with that ID exists in global or current-project memory.",
    }


def test_search_observation_files_supports_keyword_or_regex_queries(tmp_path):
    memories = tmp_path / "memories"
    record_observation_file(
        memory_dir=memories,
        project_id="P-project",
        memory_type=MemoryType.SEMANTIC,
        summary="Build command exits with a generic error.",
        observation="The local build can fail with an error after dependency setup.",
        why_it_matters="Future agents should inspect command output.",
        scope=MemoryScope.GLOBAL,
        source_type=MemorySourceType.SUBAGENT,
        source_session_id="thread-1",
        source_agent="code-agent",
    )
    record_observation_file(
        memory_dir=memories,
        project_id="P-project",
        memory_type=MemoryType.SEMANTIC,
        summary="FastAPI version conflicts can block dependency resolution.",
        observation="FastAPI and pydantic version constraints can make installs fail.",
        why_it_matters="Future agents should inspect package constraints.",
        scope=MemoryScope.GLOBAL,
        source_type=MemorySourceType.SUBAGENT,
        source_session_id="thread-1",
        source_agent="code-agent",
    )
    relevant = record_observation_file(
        memory_dir=memories,
        project_id="P-project",
        memory_type=MemoryType.SEMANTIC,
        summary="Silent-failure: backend status handling hides API response errors.",
        observation=(
            "When HTTP response status handling treats server errors as success, "
            "frontend error states can collapse into ordinary empty data."
        ),
        why_it_matters="Future agents should audit both HTTP status and UI state.",
        scope=MemoryScope.GLOBAL,
        source_type=MemorySourceType.SUBAGENT,
        source_session_id="thread-1",
        source_agent="code-agent",
    )

    variant_hits = search_observation_files(
        memory_dir=memories,
        project_id="P-project",
        query="blank profile silent failure empty data not onboarded",
    )
    focused_hits = search_observation_files(
        memory_dir=memories,
        project_id="P-project",
        mode=ObservationSearchMode.REGEX,
        query="silent[- ]failure|status",
    )

    assert [hit["observation_id"] for hit in variant_hits] == [
        relevant["observation_id"]
    ]
    assert [hit["observation_id"] for hit in focused_hits] == [
        relevant["observation_id"]
    ]


def test_search_observation_files_handles_regex_like_literals(tmp_path):
    memories = tmp_path / "memories"
    relevant = record_observation_file(
        memory_dir=memories,
        project_id="P-project",
        memory_type=MemoryType.PROCEDURAL,
        summary="Literal [bracket token appears in build logs.",
        observation="When logs include [bracket tokens, search should not crash.",
        why_it_matters="Malformed model regex should still behave like literal grep.",
        scope=MemoryScope.GLOBAL,
        source_type=MemorySourceType.SUBAGENT,
        source_session_id="thread-1",
        source_agent="code-agent",
    )

    hits = search_observation_files(
        memory_dir=memories,
        project_id="P-project",
        query="[bracket",
    )
    regex_hits = search_observation_files(
        memory_dir=memories,
        project_id="P-project",
        query="[bracket",
        mode=ObservationSearchMode.REGEX,
    )

    assert [hit["observation_id"] for hit in hits] == [relevant["observation_id"]]
    assert [hit["observation_id"] for hit in regex_hits] == [relevant["observation_id"]]


def test_search_observation_files_ranks_bag_of_words_queries(tmp_path):
    memories = tmp_path / "memories"
    record_observation_file(
        memory_dir=memories,
        project_id="P-project",
        memory_type=MemoryType.SEMANTIC,
        summary="CSV header normalization needs whitespace stripping.",
        observation="Strip CSV headers before schema matching.",
        why_it_matters="Future revenue imports may have messy column names.",
        scope=MemoryScope.GLOBAL,
        source_type=MemorySourceType.SUBAGENT,
        source_session_id="thread-1",
        source_agent="data-agent",
    )
    relevant = record_observation_file(
        memory_dir=memories,
        project_id="P-project",
        memory_type=MemoryType.SEMANTIC,
        summary=(
            "Batch duplicate detection by batch_id grouping misses cross-ID "
            "imports; content fingerprinting is required."
        ),
        observation=(
            "Compute a stable content-fingerprint per batch from sorted "
            "(date, amount) pairs before revenue aggregation."
        ),
        why_it_matters=(
            "Future quarterly revenue reports should collapse duplicate import "
            "batches before totals are computed."
        ),
        scope=MemoryScope.GLOBAL,
        source_type=MemorySourceType.SUBAGENT,
        source_session_id="thread-1",
        source_agent="data-agent",
    )

    hits = search_observation_files(
        memory_dir=memories,
        project_id="P-project",
        query="CSV duplicate batch fingerprint revenue quarterly",
        limit=2,
    )

    assert hits[0]["observation_id"] == relevant["observation_id"]
    assert hits[0]["score"] > hits[1]["score"]


def test_search_observation_files_returns_no_low_confidence_fallback(tmp_path):
    memories = tmp_path / "memories"
    record_observation_file(
        memory_dir=memories,
        project_id="P-project",
        memory_type=MemoryType.SEMANTIC,
        summary="CSV header normalization needs whitespace stripping.",
        observation="Strip CSV headers before schema matching.",
        why_it_matters="Future imports may have messy column names.",
        scope=MemoryScope.GLOBAL,
        source_type=MemorySourceType.SUBAGENT,
        source_session_id="thread-1",
        source_agent="data-agent",
    )

    hits = search_observation_files(
        memory_dir=memories,
        project_id="P-project",
        query="quantum thermostat",
    )

    assert hits == []


def test_record_observation_tool_can_use_worker_config_source(tmp_path):
    from EvoScientist.middleware.memory import create_memory_middleware

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    middleware = create_memory_middleware(
        str(tmp_path / "memories"),
        workspace_dir=workspace,
        source_type=MemorySourceType.SUBAGENT,
        source_agent="evomemory-subagent-worker",
    )
    tool = _tool_by_name(middleware.tools, "record_observation")
    payload = _record_observation_payload(
        tool,
        runtime=_tool_runtime(
            tool,
            tool_call_id="tool-1",
            config={
                "configurable": {
                    "evomemory_project_id": "P-project",
                    "evomemory_source_agent": "writing-agent",
                    "evomemory_source_session_id": "thread-source",
                    "evomemory_trajectory_digest": "digest-source",
                }
            },
        ),
        memory_type=MemoryType.PROCEDURAL,
        summary="Worker observations retain source run attribution.",
        observation="The worker should attribute observations to the source run.",
        why_it_matters="Later debugging needs the original agent and thread.",
        scope=MemoryScope.PROJECT,
    )
    path = tmp_path / "memories" / payload["path"].removeprefix("/memories/")
    metadata, _body = _read_memory_document(path)

    assert payload["project_id"] == "P-project"
    assert _stable_created_at(metadata) == {
        "id": payload["observation_id"],
        "created_at": "<created_at>",
        "summary": "Worker observations retain source run attribution.",
        "memory_type": "procedural",
        "scope": "project",
        "project_id": "P-project",
        "source": {
            "type": "subagent",
            "agent": "writing-agent",
            "session_id": "thread-source",
        },
    }


def test_record_observation_tool_schema_hides_runtime(tmp_path):
    from EvoScientist.middleware.memory import create_memory_middleware

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    middleware = create_memory_middleware(
        str(tmp_path / "memories"),
        workspace_dir=workspace,
    )

    tool = _tool_by_name(middleware.tools, "record_observation")
    assert "runtime" in tool.get_input_schema().model_fields
    schema = tool.tool_call_schema
    assert isinstance(schema, type)
    assert issubclass(schema, BaseModel)
    assert sorted(schema.model_json_schema()["properties"]) == [
        "evidence",
        "memory_type",
        "observation",
        "scope",
        "summary",
        "why_it_matters",
    ]


def test_record_observation_tool_keeps_injected_runtime_through_validation(tmp_path):
    from EvoScientist.middleware.memory import create_memory_middleware

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    middleware = create_memory_middleware(
        str(tmp_path / "memories"),
        workspace_dir=workspace,
        source_type=MemorySourceType.TURN,
        source_agent="EvoScientist",
    )
    tool = _tool_by_name(middleware.tools, "record_observation")
    payload = _record_observation_payload(
        tool,
        runtime=_tool_runtime(
            tool,
            config={"configurable": {"thread_id": "thread-from-runtime"}},
            tool_call_id="tool-1",
        ),
        memory_type=MemoryType.SEMANTIC,
        summary="Injected runtime metadata survives tool validation.",
        observation="Runtime survives validation.",
        why_it_matters="Observation metadata should keep the live thread.",
        scope=MemoryScope.GLOBAL,
    )
    path = tmp_path / "memories" / payload["path"].removeprefix("/memories/")
    metadata, _body = _read_memory_document(path)

    assert _stable_created_at(metadata) == {
        "id": payload["observation_id"],
        "created_at": "<created_at>",
        "summary": "Injected runtime metadata survives tool validation.",
        "memory_type": "semantic",
        "scope": "global",
        "source": {
            "type": "turn",
            "agent": "EvoScientist",
            "session_id": "thread-from-runtime",
        },
    }


def test_record_observation_tool_skips_without_runtime_thread_id(tmp_path):
    from EvoScientist.middleware.memory import create_memory_middleware

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    launched: list[memory_scheduler.ObservationLinkerContext] = []
    coordinator = memory_scheduler.MemoryScheduler(launch_linker=launched.append)
    middleware = create_memory_middleware(
        str(tmp_path / "memories"),
        workspace_dir=workspace,
        memory_scheduler=coordinator,
    )
    tool = _tool_by_name(middleware.tools, "record_observation")

    payload = _record_observation_payload(
        tool,
        runtime=_tool_runtime(tool),
        memory_type=MemoryType.SEMANTIC,
        summary="Unthreaded observations are skipped.",
        observation="Observation recording needs source session provenance.",
        why_it_matters="Durable memory should not persist unknown source sessions.",
        scope=MemoryScope.GLOBAL,
    )

    assert payload == {
        "error": "Cannot record observation without a source session id.",
    }
    assert launched == []
    assert list((tmp_path / "memories").glob("observations/**/*.md")) == []


def test_direct_record_observation_queues_linking_until_worker_finish(tmp_path):
    from EvoScientist.middleware.memory import create_memory_middleware

    memory_dir = tmp_path / "memories"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    launched: list[memory_scheduler.ObservationLinkerContext] = []
    coordinator = memory_scheduler.MemoryScheduler(launch_linker=launched.append)
    middleware = create_memory_middleware(
        str(memory_dir),
        workspace_dir=workspace,
        memory_scheduler=coordinator,
    )
    tool = _tool_by_name(middleware.tools, "record_observation")

    payload = _record_observation_payload(
        tool,
        runtime=_tool_runtime(tool, thread_id="source-thread"),
        memory_type=MemoryType.PROCEDURAL,
        summary="Direct observations link after worker finish.",
        observation=(
            "Direct main-agent observations should be linked after the "
            "post-turn memory worker phase finishes."
        ),
        why_it_matters=(
            "The linker should see direct writes even when the worker "
            "does not create another observation."
        ),
        scope=MemoryScope.PROJECT,
    )
    assert payload["created"] is True
    assert launched == []

    hooks = memory_launch._memory_worker_launch_hooks(
        memory_dir,
        on_worker_finished=coordinator.record_worker_finished,
    )
    assert hooks.on_before_run is not None
    assert hooks.on_started is not None
    assert hooks.on_finished is not None
    hooks.on_before_run("worker-thread")
    worker_run = _memory_worker_run(
        thread_id="worker-thread",
        run_id="worker-run",
        workspace_dir=str(workspace),
        project_id=middleware.project_id,
    )
    hooks.on_started(worker_run)
    assert launched == []

    hooks.on_finished(worker_run)

    assert launched == [
        _linker_context(
            memory_dir=memory_dir,
            workspace_dir=workspace,
            project_id=middleware.project_id,
            observation_ids=(payload["observation_id"],),
        )
    ]
    status = worker_activity.memory_worker_status()
    assert status.observations_recorded == 0


def test_turn_compaction_hides_task_call_and_keeps_orchestrator_response():
    messages = [
        HumanMessage("please delegate"),
        AIMessage(
            content="",
            name="EvoScientist",
            tool_calls=[
                {
                    "name": "task",
                    "id": "task-1",
                    "args": {"subagent_type": "code-agent", "description": "debug"},
                }
            ],
        ),
        ToolMessage("raw subagent result body", tool_call_id="task-1"),
        AIMessage(
            "final orchestrator text with summarized finding", name="EvoScientist"
        ),
    ]

    compact = source_context._compact_turn_messages(
        messages,
        source_agent="EvoScientist",
    )

    assert compact == [
        {"role": "human", "content": "please delegate"},
        {
            "role": "ai",
            "content": "final orchestrator text with summarized finding",
            "name": "EvoScientist",
        },
    ]


def test_turn_compaction_keeps_direct_tool_results_with_tool_names():
    messages = [
        HumanMessage("run a check"),
        AIMessage(
            content="",
            name="EvoScientist",
            tool_calls=[
                {
                    "name": "execute",
                    "id": "exec-1",
                    "args": {"command": "pytest -q"},
                },
                {
                    "name": "task",
                    "id": "task-1",
                    "args": {"subagent_type": "code-agent", "description": "debug"},
                },
            ],
        ),
        ToolMessage("pytest passed", tool_call_id="exec-1", name="execute"),
        ToolMessage("raw subagent result body", tool_call_id="task-1", name="task"),
        AIMessage("final answer", name="EvoScientist"),
    ]

    compact = source_context._compact_turn_messages(
        messages,
        source_agent="EvoScientist",
    )

    assert compact == [
        {"role": "human", "content": "run a check"},
        {
            "role": "ai",
            "content": "",
            "name": "EvoScientist",
            "tool_calls": [
                {
                    "name": "execute",
                    "id": "exec-1",
                    "args": {"command": "pytest -q"},
                    "type": "tool_call",
                },
            ],
        },
        {
            "role": "tool",
            "content": "pytest passed",
            "name": "execute",
            "tool_call_id": "exec-1",
            "status": "success",
        },
        {"role": "ai", "content": "final answer", "name": "EvoScientist"},
    ]


def test_turn_compaction_uses_latest_user_turn_only():
    messages = [
        HumanMessage("old request"),
        AIMessage("old answer", name="EvoScientist"),
        HumanMessage("current request"),
        AIMessage("current answer", name="EvoScientist"),
    ]

    compact = source_context._compact_turn_messages(
        messages,
        source_agent="EvoScientist",
    )

    assert compact == [
        {"role": "human", "content": "current request"},
        {"role": "ai", "content": "current answer", "name": "EvoScientist"},
    ]


def test_lifecycle_schedules_turn_worker_without_awaiting(
    tmp_path, monkeypatch, run_async
):
    memory_dir = tmp_path / "memories"
    workspace_dir = tmp_path / "workspace"
    calls = []
    launched: list[memory_scheduler.ObservationLinkerContext] = []
    coordinator = memory_scheduler.MemoryScheduler(launch_linker=launched.append)

    async def fake_launch(request, **kwargs):
        calls.append((request, kwargs["hooks"]))

    monkeypatch.setattr(
        memory_launch,
        "alaunch_background_run",
        fake_launch,
    )
    middleware = memory_lifecycle.EvoMemoryLifecycleMiddleware(
        memory_dir=memory_dir,
        workspace_dir=workspace_dir,
        project_id="P-project",
        source_type=MemorySourceType.TURN,
        source_agent="EvoScientist",
        memory_scheduler=coordinator,
    )
    runtime = _runtime("thread-1")

    async def run():
        state: AgentState[object] = {
            "messages": [
                HumanMessage("previous turn"),
                AIMessage("previous answer"),
                HumanMessage("hi"),
                AIMessage("done"),
            ]
        }
        await middleware.aafter_agent(
            state,
            runtime,
        )

    run_async(run())

    assert len(calls) == 1
    request, hooks = calls[0]
    assert request.graph_id == memory_launch.TURN_MEMORY_WORKER_GRAPH_ID
    assert request.name == "EvoMemory worker"
    assert hooks.on_before_run is not None
    assert hooks.on_started is not None
    assert hooks.on_finished is not None
    hooks.on_before_run("worker-thread")
    worker_run = _memory_worker_run(workspace_dir=str(workspace_dir))
    hooks.on_started(worker_run)
    observation = _record_test_observation(memory_dir)
    hooks.on_finished(worker_run)
    assert launched == [
        _linker_context(
            memory_dir=memory_dir,
            workspace_dir=workspace_dir,
            observation_ids=(observation["observation_id"],),
        )
    ]


def test_lifecycle_skips_memory_worker_without_runtime_thread_id(tmp_path, monkeypatch):
    def fail_launch(*_args, **_kwargs):
        raise AssertionError("worker should not launch without a source thread id")

    monkeypatch.setattr(memory_lifecycle, "launch_memory_worker", fail_launch)
    middleware = memory_lifecycle.EvoMemoryLifecycleMiddleware(
        memory_dir=tmp_path / "memories",
        workspace_dir=tmp_path / "workspace",
        project_id="P-project",
        source_type=MemorySourceType.TURN,
        source_agent="EvoScientist",
    )

    middleware.after_agent(
        {"messages": [HumanMessage("hi"), AIMessage("done", name="EvoScientist")]},
        _runtime(),
    )


def test_subagent_summary_writer_uses_worker_metadata(tmp_path, monkeypatch):
    summary = "Completed the analysis."
    monkeypatch.setattr(
        memory_worker,
        "_current_configurable",
        lambda: {
            "evomemory_source_session_id": "thread-1",
            "evomemory_source_agent": "writing-agent",
            "evomemory_project_id": "P-project",
            "evomemory_trajectory_digest": "digest-1",
        },
    )
    middleware = memory_worker._SubagentSummaryWriterMiddleware(
        memory_dir=tmp_path / "memories"
    )

    state: AgentState[object] = {
        "messages": [],
        "structured_response": memory_worker.SubagentMemoryDecision(summary=summary),
    }
    middleware.after_agent(
        state,
        _runtime(),
    )

    paths = list((tmp_path / "memories" / "executions" / "thread-1").glob("*.md"))
    assert len(paths) == 1
    metadata, body = _read_memory_document(paths[0])
    assert _stable_created_at(metadata) == {
        "id": memory_worker._execution_summary_id(
            session_id="thread-1",
            source_agent="writing-agent",
            trajectory_digest="digest-1",
        ),
        "created_at": "<created_at>",
        "source": {
            "type": "subagent",
            "session_id": "thread-1",
            "agent": "writing-agent",
        },
        "project_id": "P-project",
    }
    assert _markdown_sections(body) == {"Summary": summary}


def test_memory_worker_run_payload_use_server_thread_id_and_source_metadata(
    monkeypatch,
):
    monkeypatch.setattr(
        memory_launch,
        "_worker_workspace_dir",
        lambda _workspace_dir: "/tmp/ws",
    )
    trajectory: list[source_context.CompactMessage] = [
        {"role": "human", "content": "hi"}
    ]
    context = _memory_source_context(
        memory_dir="/memories",
        workspace_dir="/active/workspace",
        source_type=MemorySourceType.SUBAGENT,
        source_agent="writing-agent",
        trajectory=trajectory,
    )

    kwargs = memory_launch._memory_worker_run_payload(
        context=context,
        thread_id="worker-thread",
    )

    assert kwargs["assistant_id"] == memory_launch.SUBAGENT_MEMORY_WORKER_GRAPH_ID
    assert kwargs["metadata"] == {
        "run_kind": "evomemory_subagent_worker",
        "source_session_id": "thread-1",
        "source_agent": "writing-agent",
        "project_id": "P-project",
        "trajectory_digest": source_context._trajectory_digest(trajectory),
        "workspace_dir": "/tmp/ws",
    }
    configurable = kwargs["config"]["configurable"]
    assert configurable["thread_id"] == "worker-thread"
    assert {
        key: value
        for key, value in configurable.items()
        if key.startswith("evomemory_")
    } == {
        "evomemory_source_session_id": "thread-1",
        "evomemory_source_agent": "writing-agent",
        "evomemory_project_id": "P-project",
        "evomemory_trajectory_digest": source_context._trajectory_digest(trajectory),
    }


def test_memory_worker_finish_launches_linker_for_new_observations(
    tmp_path,
):
    memory_dir = tmp_path / "memories"
    workspace_dir = tmp_path / "workspace"
    launched: list[memory_scheduler.ObservationLinkerContext] = []
    coordinator = memory_scheduler.MemoryScheduler(launch_linker=launched.append)

    _mark_worker_started(memory_dir)
    observation = _record_test_observation(memory_dir)

    hooks = memory_launch._memory_worker_launch_hooks(
        memory_dir,
        on_worker_finished=coordinator.record_worker_finished,
    )
    assert hooks.on_finished is not None
    hooks.on_finished(
        _memory_worker_run(workspace_dir=str(workspace_dir), run_id="run-1")
    )

    assert launched == [
        _linker_context(
            memory_dir=memory_dir,
            workspace_dir=workspace_dir,
            observation_ids=(observation["observation_id"],),
        )
    ]
    assert worker_activity.memory_worker_status().observations_recorded == 1


def test_memory_worker_linker_waits_for_active_workers_and_batches_observations(
    tmp_path,
):
    memory_dir = tmp_path / "memories"
    workspace_dir = tmp_path / "workspace"
    launched: list[memory_scheduler.ObservationLinkerContext] = []
    coordinator = memory_scheduler.MemoryScheduler(launch_linker=launched.append)

    before = worker_activity.snapshot_memory_outputs(memory_dir)
    _mark_worker_started(
        memory_dir,
        thread_id="thread-1",
        run_id="run-1",
        before_outputs=before,
    )
    _mark_worker_started(
        memory_dir,
        thread_id="thread-2",
        run_id="run-2",
        before_outputs=before,
    )
    first_observation = _record_test_observation(
        memory_dir,
        summary="First durable observation.",
        observation="The first reusable observation for linking.",
    )

    hooks = memory_launch._memory_worker_launch_hooks(
        memory_dir,
        on_worker_finished=coordinator.record_worker_finished,
    )
    assert hooks.on_finished is not None
    hooks.on_finished(
        _memory_worker_run(
            thread_id="thread-1",
            run_id="run-1",
            workspace_dir=str(workspace_dir),
            source_agent="subagent-a",
            source_session_id="session-a",
            trajectory_digest="digest-a",
        )
    )
    assert launched == []

    second_observation = _record_test_observation(
        memory_dir,
        summary="Second durable observation.",
        observation="The second reusable observation for linking.",
        scope=MemoryScope.PROJECT,
    )
    hooks.on_finished(
        _memory_worker_run(
            thread_id="thread-2",
            run_id="run-2",
            workspace_dir=str(workspace_dir),
            source_agent="EvoScientist",
            source_session_id="session-b",
            trajectory_digest="digest-b",
        )
    )

    assert len(launched) == 1
    assert launched[0].memory_dir == memory_dir
    assert launched[0].workspace_dir == workspace_dir
    assert launched[0].project_id == "P-project"
    assert set(launched[0].observation_ids) == {
        first_observation["observation_id"],
        second_observation["observation_id"],
    }
    assert worker_activity.memory_worker_status().observations_recorded == 2


def test_memory_worker_linker_flushes_when_last_worker_has_no_observations(tmp_path):
    memory_dir = tmp_path / "memories"
    workspace_dir = tmp_path / "workspace"
    launched: list[memory_scheduler.ObservationLinkerContext] = []
    coordinator = memory_scheduler.MemoryScheduler(launch_linker=launched.append)

    before = worker_activity.snapshot_memory_outputs(memory_dir)
    _mark_worker_started(
        memory_dir,
        thread_id="thread-1",
        run_id="run-1",
        before_outputs=before,
    )
    _mark_worker_started(
        memory_dir,
        thread_id="thread-2",
        run_id="run-2",
        before_outputs=before,
    )
    observation = _record_test_observation(memory_dir)

    hooks = memory_launch._memory_worker_launch_hooks(
        memory_dir,
        on_worker_finished=coordinator.record_worker_finished,
    )
    assert hooks.on_finished is not None
    hooks.on_finished(
        _memory_worker_run(
            thread_id="thread-1",
            run_id="run-1",
            workspace_dir=str(workspace_dir),
        )
    )
    assert launched == []

    profile_path = memory_dir / "profile" / "USER_PROFILE.md"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text("# User profile\n\n- remembered\n", encoding="utf-8")
    hooks.on_finished(
        _memory_worker_run(
            thread_id="thread-2",
            run_id="run-2",
            workspace_dir=str(workspace_dir),
        )
    )

    assert len(launched) == 1
    assert launched[0].observation_ids == (observation["observation_id"],)
    status = worker_activity.memory_worker_status()
    assert status.profile_updates == 1
    assert status.observations_recorded == 1


def test_memory_worker_linker_flushes_pending_batch_when_last_worker_aborts(tmp_path):
    memory_dir = tmp_path / "memories"
    workspace_dir = tmp_path / "workspace"
    launched: list[memory_scheduler.ObservationLinkerContext] = []
    coordinator = memory_scheduler.MemoryScheduler(launch_linker=launched.append)

    before = worker_activity.snapshot_memory_outputs(memory_dir)
    _mark_worker_started(
        memory_dir,
        thread_id="thread-1",
        run_id="run-1",
        before_outputs=before,
    )
    _mark_worker_started(
        memory_dir,
        thread_id="thread-2",
        run_id="run-2",
        before_outputs=before,
    )
    observation = _record_test_observation(memory_dir)

    hooks = memory_launch._memory_worker_launch_hooks(
        memory_dir,
        on_worker_finished=coordinator.record_worker_finished,
        on_worker_aborted=coordinator.record_worker_aborted,
    )
    assert hooks.on_finished is not None
    assert hooks.on_aborted is not None
    hooks.on_finished(
        _memory_worker_run(
            thread_id="thread-1",
            run_id="run-1",
            workspace_dir=str(workspace_dir),
        )
    )
    assert launched == []

    hooks.on_aborted(
        _memory_worker_run(
            thread_id="thread-2",
            run_id="run-2",
            workspace_dir=str(workspace_dir),
        )
    )

    assert launched == [
        _linker_context(
            memory_dir=memory_dir,
            workspace_dir=workspace_dir,
            observation_ids=(observation["observation_id"],),
        )
    ]
    status = worker_activity.memory_worker_status()
    assert status.is_running is False
    assert status.observations_recorded == 1


def test_memory_worker_finish_does_not_launch_linker_for_profile_only_delta(
    tmp_path,
):
    memory_dir = tmp_path / "memories"
    launched: list[memory_scheduler.ObservationLinkerContext] = []
    coordinator = memory_scheduler.MemoryScheduler(launch_linker=launched.append)

    _mark_worker_started(memory_dir)
    profile_path = memory_dir / "profile" / "USER_PROFILE.md"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text("# User profile\n\n- remembered\n", encoding="utf-8")

    hooks = memory_launch._memory_worker_launch_hooks(
        memory_dir,
        on_worker_finished=coordinator.record_worker_finished,
    )
    assert hooks.on_finished is not None
    hooks.on_finished(_memory_worker_run(run_id="run-1"))

    assert launched == []
    status = worker_activity.memory_worker_status()
    assert status.profile_updates == 1
    assert status.observations_recorded == 0


def test_memory_worker_abort_queues_written_observations_for_linking(tmp_path):
    memory_dir = tmp_path / "memories"
    workspace_dir = tmp_path / "workspace"
    launched: list[memory_scheduler.ObservationLinkerContext] = []
    coordinator = memory_scheduler.MemoryScheduler(launch_linker=launched.append)

    _mark_worker_started(memory_dir)
    observation = _record_test_observation(memory_dir)

    hooks = memory_launch._memory_worker_launch_hooks(
        memory_dir,
        on_worker_aborted=coordinator.record_worker_aborted,
    )
    assert hooks.on_aborted is not None
    hooks.on_aborted(
        _memory_worker_run(
            run_id="run-1",
            workspace_dir=str(workspace_dir),
        )
    )

    assert launched == [
        _linker_context(
            memory_dir=memory_dir,
            workspace_dir=workspace_dir,
            observation_ids=(observation["observation_id"],),
        )
    ]
    status = worker_activity.memory_worker_status()
    assert status.is_running is False
    assert status.observations_recorded == 1


def test_memory_worker_watcher_start_failure_queues_written_observations_for_linking(
    tmp_path,
):
    memory_dir = tmp_path / "memories"
    workspace_dir = tmp_path / "workspace"
    launched: list[memory_scheduler.ObservationLinkerContext] = []
    coordinator = memory_scheduler.MemoryScheduler(launch_linker=launched.append)

    hooks = memory_launch._memory_worker_launch_hooks(
        memory_dir,
        on_worker_aborted=coordinator.record_worker_aborted,
    )
    worker_run = _memory_worker_run(
        thread_id="worker-thread",
        run_id="run-1",
        workspace_dir=str(workspace_dir),
    )
    assert hooks.on_before_run is not None
    assert hooks.on_started is not None
    assert hooks.on_watcher_start_failed is not None
    hooks.on_before_run(worker_run.thread_id)
    hooks.on_started(worker_run)
    observation = _record_test_observation(memory_dir)

    hooks.on_watcher_start_failed(worker_run)

    assert launched == [
        _linker_context(
            memory_dir=memory_dir,
            workspace_dir=workspace_dir,
            observation_ids=(observation["observation_id"],),
        )
    ]
    status = worker_activity.memory_worker_status()
    assert status.is_running is False
    assert status.observations_recorded == 1


def test_observation_linker_launch_request_encodes_batch_context(tmp_path):
    context = _linker_context(
        memory_dir=tmp_path / "memories",
        workspace_dir=tmp_path / "workspace",
        observation_ids=("O-2", "O-1"),
    )

    request = memory_launch.observation_linker_launch_request(context)
    kwargs = request.run_payload("linker-thread")

    assert request.graph_id == memory_launch.OBSERVATION_LINKER_GRAPH_ID
    assert request.name == "EvoMemory observation linker"
    configurable = kwargs["config"]["configurable"]
    assert configurable["thread_id"] == "linker-thread"
    assert configurable["evomemory_project_id"] == "P-project"
    assert json.loads(configurable["evomemory_observation_ids"]) == [
        "O-2",
        "O-1",
    ]


def test_observation_linker_does_not_launch_when_observations_disabled(
    tmp_path,
    monkeypatch,
):
    context = _linker_context(
        memory_dir=tmp_path / "memories",
        workspace_dir=tmp_path / "workspace",
        observation_ids=("O-1",),
    )
    monkeypatch.setattr(
        memory_launch,
        "get_effective_config",
        lambda: EvoScientistConfig(memory_observations_enabled=False),
    )
    launch_call = MagicMock()
    monkeypatch.setattr(memory_launch, "launch_background_run", launch_call)

    run = memory_launch.launch_observation_linker(context)

    assert run is None
    launch_call.assert_not_called()


def test_async_observation_linker_does_not_launch_when_observations_disabled(
    tmp_path,
    monkeypatch,
    run_async,
):
    context = _linker_context(
        memory_dir=tmp_path / "memories",
        workspace_dir=tmp_path / "workspace",
        observation_ids=("O-1",),
    )
    monkeypatch.setattr(
        memory_launch,
        "get_effective_config",
        lambda: EvoScientistConfig(memory_observations_enabled=False),
    )
    launch_call = MagicMock()
    monkeypatch.setattr(memory_launch, "alaunch_background_run", launch_call)

    run = run_async(memory_launch.alaunch_observation_linker(context))

    assert run is None
    launch_call.assert_not_called()


def test_observation_linker_launch_hooks_track_running_status(tmp_path):
    run = _observation_linker_run()

    hooks = memory_launch._observation_linker_launch_hooks(tmp_path / "memories")
    assert hooks.on_started is not None
    assert hooks.on_finished is not None
    hooks.on_started(run)
    assert worker_activity.observation_linker_status().is_running is True

    hooks.on_finished(run)
    assert worker_activity.observation_linker_status().is_running is False


def test_observation_linker_uses_read_search_memory_and_link_tool(tmp_path):
    tools = observation_linker._observation_linker_tools(
        memory_dir=tmp_path / "memories",
        workspace_dir=tmp_path / "workspace",
    )

    assert [tool.name for tool in tools] == [
        "search_observations",
        "read_memory",
        "link_observations",
    ]
    assert "record_observation" not in {tool.name for tool in tools}


def test_memory_worker_accepts_roots_at_build_time(tmp_path, monkeypatch):
    calls = []

    def fake_build(**kwargs):
        calls.append(kwargs)
        return MagicMock()

    monkeypatch.setattr(memory_worker, "_build_memory_worker_agent", fake_build)

    memory_worker.build_memory_worker_graph(
        MemorySourceType.TURN,
        memory_dir=tmp_path / "memories",
        workspace_dir=tmp_path / "workspace",
    )

    assert calls[0]["memory_dir"] == tmp_path / "memories"
    assert calls[0]["workspace_dir"] == tmp_path / "workspace"


def _memory_tool_names(middleware) -> list[str]:
    memory_middleware = next(item for item in middleware if getattr(item, "tools", ()))
    return [tool.name for tool in memory_middleware.tools]


@pytest.mark.parametrize(
    ("source_type", "observation_writer", "expected_tools"),
    [
        (
            MemorySourceType.SUBAGENT,
            MemoryObservationWriter.AGENT,
            ["search_observations", "read_memory"],
        ),
        (
            MemorySourceType.SUBAGENT,
            MemoryObservationWriter.WORKER,
            ["search_observations", "read_memory", "record_observation"],
        ),
        (
            MemorySourceType.TURN,
            MemoryObservationWriter.WORKER,
            ["search_observations", "read_memory", "record_observation"],
        ),
        (
            MemorySourceType.TURN,
            MemoryObservationWriter.ALL,
            ["search_observations", "read_memory", "record_observation"],
        ),
        (
            MemorySourceType.SUBAGENT,
            MemoryObservationWriter.ALL,
            ["search_observations", "read_memory", "record_observation"],
        ),
    ],
)
def test_memory_worker_observation_writer_modes(
    tmp_path,
    source_type: MemorySourceType,
    observation_writer: MemoryObservationWriter,
    expected_tools: list[str],
):
    middleware = memory_worker._memory_worker_middleware(
        memory_dir=tmp_path / "memories",
        workspace_dir=tmp_path / "workspace",
        source_type=source_type,
        observation_writer=observation_writer,
    )

    assert type(middleware[0]).__name__ == "ToolErrorHandlerMiddleware"
    assert _memory_tool_names(middleware) == expected_tools


def test_sync_memory_worker_watcher_untracks_without_counting_on_poll_abort(
    tmp_path, monkeypatch
):
    memory_dir = tmp_path / "memories"
    _mark_worker_started(memory_dir)
    profile_path = memory_dir / "profile" / "USER_PROFILE.md"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text("# User profile\n\n- later update\n", encoding="utf-8")

    class _Runs:
        def get(self, **_kwargs):
            raise RuntimeError("poll failed")

    monkeypatch.setattr(
        "langgraph_sdk.get_sync_client",
        lambda **_kwargs: SimpleNamespace(runs=_Runs()),
    )

    background_runs.watch_background_run_sync(
        url="http://x",
        thread_id="worker-thread",
        run_id="run-1",
        hooks=memory_launch._memory_worker_launch_hooks(memory_dir),
        watcher_config=_fast_watcher_config(max_poll_failures=1),
    )
    status = worker_activity.memory_worker_status()
    assert status.is_running is False
    assert status.profile_updates == 0
    assert status.observations_recorded == 0


def test_async_memory_worker_watcher_untracks_without_counting_on_poll_abort(
    tmp_path, monkeypatch, run_async
):
    memory_dir = tmp_path / "memories"
    _mark_worker_started(memory_dir)
    observation_path = memory_dir / "observations" / "global" / "O-1.md"
    observation_path.parent.mkdir(parents=True)
    observation_path.write_text("# Observation\n", encoding="utf-8")

    class _Runs:
        async def get(self, **_kwargs):
            raise RuntimeError("poll failed")

    run_async(
        background_runs.awatch_background_run(
            SimpleNamespace(runs=_Runs()),
            thread_id="worker-thread",
            run_id="run-1",
            hooks=memory_launch._memory_worker_launch_hooks(memory_dir),
            watcher_config=_fast_watcher_config(max_poll_failures=1),
        )
    )
    status = worker_activity.memory_worker_status()
    assert status.is_running is False
    assert status.profile_updates == 0
    assert status.observations_recorded == 0


def test_async_memory_worker_watcher_counts_completion_under_blockbuster(
    tmp_path, run_async
):
    memory_dir = tmp_path / "memories"
    _mark_worker_started(memory_dir)
    profile_path = memory_dir / "profile" / "USER_PROFILE.md"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text("# User profile\n\n- later update\n", encoding="utf-8")

    class _Runs:
        async def get(self, **_kwargs):
            return {"status": "success"}

    async def run():
        blocker = BlockBuster(scanned_modules=[memory_worker, worker_activity])
        blocker.activate()
        try:
            await background_runs.awatch_background_run(
                SimpleNamespace(runs=_Runs()),
                thread_id="worker-thread",
                run_id="run-1",
                hooks=memory_launch._memory_worker_launch_hooks(memory_dir),
            )
        finally:
            blocker.deactivate()

    run_async(run())
    status = worker_activity.memory_worker_status()
    assert status.is_running is False
    assert status.profile_updates == 1
    assert status.observations_recorded == 0


def test_memory_worker_watcher_untracks_when_client_creation_fails(
    tmp_path, monkeypatch
):
    memory_dir = tmp_path / "memories"
    _mark_worker_started(memory_dir)
    profile_path = memory_dir / "profile" / "USER_PROFILE.md"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text("# User profile\n\n- later update\n", encoding="utf-8")

    monkeypatch.setattr(
        "langgraph_sdk.get_sync_client",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("client failed")),
    )

    with pytest.raises(RuntimeError, match="client failed"):
        background_runs.watch_background_run_sync(
            url="http://x",
            thread_id="worker-thread",
            run_id="run-1",
            hooks=memory_launch._memory_worker_launch_hooks(memory_dir),
        )
    status = worker_activity.memory_worker_status()
    assert status.is_running is False
    assert status.profile_updates == 0
    assert status.observations_recorded == 0


def test_memory_worker_skips_when_langgraph_dev_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(
        background_runs, "default_background_run_url", lambda: "http://x"
    )
    monkeypatch.setattr(
        "EvoScientist.langgraph_dev.manager.is_langgraph_dev_running",
        lambda **_kwargs: False,
    )

    def fail_get_sync_client(*_args, **_kwargs):
        raise AssertionError("client should not be created")

    monkeypatch.setattr("langgraph_sdk.get_sync_client", fail_get_sync_client)

    middleware = memory_lifecycle.EvoMemoryLifecycleMiddleware(
        memory_dir=tmp_path / "memories",
        workspace_dir=tmp_path / "workspace",
        project_id="P-project",
        source_type=MemorySourceType.TURN,
        source_agent="EvoScientist",
    )
    middleware.after_agent(
        {"messages": [HumanMessage("hi"), AIMessage("done", name="EvoScientist")]},
        _runtime("thread-1"),
    )


def test_memory_worker_marks_active_status(tmp_path, monkeypatch):
    monkeypatch.setattr(
        background_runs, "default_background_run_url", lambda: "http://x"
    )
    monkeypatch.setattr(
        memory_launch,
        "_worker_workspace_dir",
        lambda _workspace_dir: "/tmp/ws",
    )
    monkeypatch.setattr(
        "EvoScientist.langgraph_dev.manager.is_langgraph_dev_running",
        lambda **_kwargs: True,
    )

    fake_client = MagicMock()
    fake_client.threads.create.return_value = {"thread_id": "worker-thread"}
    fake_client.runs.create.return_value = {"run_id": "run-1", "status": "pending"}
    monkeypatch.setattr("langgraph_sdk.get_sync_client", lambda **_kwargs: fake_client)

    spawned: list[background_runs.BackgroundRun] = []

    trajectory: list[source_context.CompactMessage] = [
        {"role": "human", "content": "hi"}
    ]

    memory_dir = tmp_path / "memories"
    context = _memory_source_context(
        memory_dir=memory_dir,
        workspace_dir=tmp_path / "workspace",
        trajectory=trajectory,
    )
    request = memory_launch.memory_worker_launch_request(context)
    background_runs.launch_background_run(
        request,
        hooks=memory_launch._memory_worker_launch_hooks(memory_dir),
        spawn_status_watcher=spawned.append,
    )

    assert worker_activity.memory_worker_status().is_running is True
    expected_metadata = {
        "run_kind": "evomemory_turn_worker",
        "source_session_id": "thread-1",
        "source_agent": "EvoScientist",
        "project_id": "P-project",
        "trajectory_digest": source_context._trajectory_digest(trajectory),
        "workspace_dir": "/tmp/ws",
    }
    fake_client.threads.create.assert_called_once_with(
        graph_id=memory_launch.TURN_MEMORY_WORKER_GRAPH_ID,
        metadata=expected_metadata,
    )
    fake_client.runs.create.assert_called_once()
    run_kwargs = fake_client.runs.create.call_args.kwargs
    assert run_kwargs["thread_id"] == "worker-thread"
    assert run_kwargs["metadata"] == expected_metadata
    assert run_kwargs["config"]["configurable"]["thread_id"] == "worker-thread"
    assert [(run.url, run.thread_id, run.run_id) for run in spawned] == [
        ("http://x", "worker-thread", "run-1")
    ]
    profile_path = memory_dir / "profile" / "USER_PROFILE.md"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text("# User profile\n\n- remembered\n", encoding="utf-8")
    observation_path = memory_dir / "observations" / "global" / "O-1.md"
    observation_path.parent.mkdir(parents=True)
    observation_path.write_text("# Observation\n", encoding="utf-8")
    delta = worker_activity.mark_memory_worker_finished("worker-thread", "run-1")
    status = worker_activity.memory_worker_status()
    assert delta == worker_activity.MemoryOutputDelta(
        memory_dir=memory_dir,
        profile_paths=("profile/USER_PROFILE.md",),
        observation_paths=("observations/global/O-1.md",),
    )
    assert status.is_running is False
    assert status.profile_updates == 1
    assert status.observations_recorded == 1


def test_async_memory_worker_offloads_blocking_work(tmp_path, monkeypatch, run_async):
    monkeypatch.setattr(
        background_runs, "default_background_run_url", lambda: "http://x"
    )
    monkeypatch.setattr(
        memory_launch,
        "_worker_workspace_dir",
        lambda _workspace_dir: "/tmp/ws",
    )

    call_threads: list[tuple[str, int]] = []

    def fake_is_running(**_kwargs):
        call_threads.append(("health", threading.get_ident()))
        return True

    def fake_snapshot(_memory_dir):
        call_threads.append(("snapshot", threading.get_ident()))
        return worker_activity.MemoryOutputSnapshot(
            profile_files={},
            observation_files=frozenset(),
        )

    monkeypatch.setattr(
        "EvoScientist.langgraph_dev.manager.is_langgraph_dev_running",
        fake_is_running,
    )
    monkeypatch.setattr(memory_launch, "snapshot_memory_outputs", fake_snapshot)

    class _Threads:
        async def create(self, **_kwargs):
            return {"thread_id": "worker-thread"}

    class _Runs:
        async def create(self, **_kwargs):
            return {"run_id": "run-1", "status": "pending"}

    fake_client = SimpleNamespace(threads=_Threads(), runs=_Runs())
    monkeypatch.setattr("langgraph_sdk.get_client", lambda **_kwargs: fake_client)

    spawned: list[background_runs.BackgroundRun] = []

    async def run():
        event_loop_thread = threading.get_ident()
        context = _memory_source_context(
            memory_dir=tmp_path / "memories",
            workspace_dir=tmp_path / "workspace",
            trajectory=[{"role": "human", "content": "hi"}],
        )
        request = memory_launch.memory_worker_launch_request(context)
        await background_runs.alaunch_background_run(
            request,
            hooks=memory_launch._memory_worker_launch_hooks(tmp_path / "memories"),
            spawn_status_watcher=spawned.append,
        )
        return event_loop_thread

    event_loop_thread = run_async(run())
    assert [name for name, _thread_id in call_threads] == ["health", "snapshot"]
    assert all(thread_id != event_loop_thread for _name, thread_id in call_threads)
    assert worker_activity.memory_worker_status().is_running is True
    assert [(run.url, run.thread_id, run.run_id) for run in spawned] == [
        ("http://x", "worker-thread", "run-1")
    ]


def test_completed_memory_activity_clear_preserves_pending_worker_delta(tmp_path):
    memory_dir = tmp_path / "memories"
    before = worker_activity.snapshot_memory_outputs(memory_dir)
    _mark_worker_started(
        memory_dir,
        thread_id="finished-thread",
        run_id="finished-run",
        before_outputs=before,
    )
    profile_path = memory_dir / "profile" / "USER_PROFILE.md"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text("# User profile\n\n- remembered\n", encoding="utf-8")
    worker_activity.mark_memory_worker_finished("finished-thread", "finished-run")
    _mark_worker_started(
        memory_dir,
        thread_id="active-thread",
        run_id="active-run",
    )
    worker_activity.mark_observation_relations_linked(1)

    worker_activity.clear_completed_memory_activity_counts()
    assert worker_activity.memory_worker_status().is_running is True
    assert worker_activity.observation_linker_status().relations_linked == 0
    observation_path = memory_dir / "observations" / "global" / "O-1.md"
    observation_path.parent.mkdir(parents=True)
    observation_path.write_text("# Observation\n", encoding="utf-8")
    worker_activity.mark_memory_worker_finished("active-thread", "active-run")
    status = worker_activity.memory_worker_status()

    assert status.is_running is False
    assert status.profile_updates == 0
    assert status.observations_recorded == 1


def test_memory_worker_observed_outputs_includes_active_worker_delta(tmp_path):
    memory_dir = tmp_path / "memories"
    before = worker_activity.snapshot_memory_outputs(memory_dir)
    _mark_worker_started(
        memory_dir,
        thread_id="active-thread",
        run_id="active-run",
        before_outputs=before,
    )
    record_observation_file(
        memory_dir=memory_dir,
        project_id="P-project",
        memory_type=MemoryType.SEMANTIC,
        summary="Active worker observation.",
        observation="The active worker has already written an observation.",
        why_it_matters="One-shot CLI waits can detect persisted worker output.",
        scope=MemoryScope.PROJECT,
        source_type=MemorySourceType.TURN,
        source_session_id="thread-1",
        source_agent="EvoScientist",
    )

    status = worker_activity.memory_worker_observed_outputs()
    assert status.is_running is True
    assert status.observations_recorded == 1
    assert status.profile_updates == 0
    assert worker_activity.memory_worker_status().observations_recorded == 0


def test_memory_pipeline_wait_keeps_polling_after_observed_memory_output():
    now = 0.0
    saved_counts = []
    observed_calls = 0

    def monotonic():
        return now

    def sleep(seconds):
        nonlocal now
        now += seconds

    def get_worker_status():
        nonlocal observed_calls
        observed_calls += 1
        if observed_calls < 8:
            return worker_activity.MemoryWorkerStatusSnapshot(
                is_running=True,
                observations_recorded=1,
            )
        return worker_activity.MemoryWorkerStatusSnapshot(
            is_running=False,
            observations_recorded=1,
            profile_updates=1,
        )

    waited_until_idle = worker_activity.wait_for_memory_pipeline_idle(
        timeout_seconds=10,
        poll_seconds=0.5,
        output_grace_seconds=3,
        get_worker_status=get_worker_status,
        get_linker_status=worker_activity.ObservationLinkerStatusSnapshot,
        monotonic=monotonic,
        sleep=sleep,
        on_saved=lambda status: saved_counts.append(
            (status.observations_recorded, status.profile_updates)
        ),
    )

    assert waited_until_idle is True
    assert observed_calls == 9
    assert saved_counts == [(1, 0), (1, 1)]


def test_memory_pipeline_wait_reports_fast_worker_output():
    saved_counts = []

    waited_until_idle = worker_activity.wait_for_memory_pipeline_idle(
        timeout_seconds=10,
        poll_seconds=0.5,
        output_grace_seconds=3,
        get_worker_status=lambda: worker_activity.MemoryWorkerStatusSnapshot(
            is_running=False,
            observations_recorded=1,
        ),
        get_linker_status=worker_activity.ObservationLinkerStatusSnapshot,
        on_saved=lambda status: saved_counts.append(
            (status.observations_recorded, status.profile_updates)
        ),
    )

    assert waited_until_idle is True
    assert saved_counts == [(1, 0)]


def test_memory_pipeline_waits_for_observation_linker():
    now = 0.0
    waiting_phases = []
    linker_calls = 0

    def monotonic():
        return now

    def sleep(seconds):
        nonlocal now
        now += seconds

    def get_linker_status():
        nonlocal linker_calls
        linker_calls += 1
        return worker_activity.ObservationLinkerStatusSnapshot(
            is_running=linker_calls < 3
        )

    waited_until_idle = worker_activity.wait_for_memory_pipeline_idle(
        timeout_seconds=10,
        poll_seconds=0.5,
        output_grace_seconds=3,
        get_worker_status=lambda: worker_activity.MemoryWorkerStatusSnapshot(
            is_running=False
        ),
        get_linker_status=get_linker_status,
        monotonic=monotonic,
        sleep=sleep,
        on_waiting=waiting_phases.append,
    )

    assert waited_until_idle is True
    assert linker_calls >= 3
    assert waiting_phases == ["linker", "linker"]


def test_memory_pipeline_waits_while_observation_linker_is_launching(tmp_path):
    entered_launch = threading.Event()
    release_launch = threading.Event()
    waiting_phases: list[worker_activity.MemoryActivityPhase] = []

    def launch_linker(_context: memory_scheduler.ObservationLinkerContext):
        entered_launch.set()
        release_launch.wait(timeout=5)

    def on_waiting(phase: worker_activity.MemoryActivityPhase) -> None:
        waiting_phases.append(phase)
        release_launch.set()

    coordinator = memory_scheduler.MemoryScheduler(launch_linker=launch_linker)
    coordinator.record_observation_created(
        _linker_context(
            memory_dir=tmp_path / "memories",
            workspace_dir=tmp_path / "workspace",
            observation_ids=("O-1",),
        )
    )
    flush_thread = threading.Thread(target=coordinator.flush_ready)
    flush_thread.start()

    try:
        assert entered_launch.wait(timeout=1)
        waited_until_idle = worker_activity.wait_for_memory_pipeline_idle(
            timeout_seconds=1,
            poll_seconds=0.01,
            output_grace_seconds=0,
            on_waiting=on_waiting,
        )

        assert waited_until_idle is True
        assert waiting_phases == ["linker"]
    finally:
        release_launch.set()
        flush_thread.join(timeout=1)


def test_memory_pipeline_wait_gives_linker_its_own_timeout_after_worker():
    now = 0.0
    timed_out_phases = []
    worker_calls = 0
    linker_calls = 0

    def monotonic():
        return now

    def sleep(seconds):
        nonlocal now
        now += seconds

    def get_worker_status():
        nonlocal worker_calls
        worker_calls += 1
        return worker_activity.MemoryWorkerStatusSnapshot(is_running=worker_calls < 20)

    def get_linker_status():
        nonlocal linker_calls
        if worker_calls < 20:
            return worker_activity.ObservationLinkerStatusSnapshot(is_running=False)
        linker_calls += 1
        return worker_activity.ObservationLinkerStatusSnapshot(
            is_running=linker_calls < 4
        )

    waited_until_idle = worker_activity.wait_for_memory_pipeline_idle(
        timeout_seconds=10,
        poll_seconds=0.5,
        output_grace_seconds=3,
        get_worker_status=get_worker_status,
        get_linker_status=get_linker_status,
        monotonic=monotonic,
        sleep=sleep,
        on_timeout=timed_out_phases.append,
    )

    assert waited_until_idle is True
    assert worker_calls >= 20
    assert linker_calls >= 4
    assert timed_out_phases == []


def test_memory_worker_status_dedupes_overlapping_observation_deltas(tmp_path):
    memory_dir = tmp_path / "memories"
    before = worker_activity.snapshot_memory_outputs(memory_dir)
    _mark_worker_started(
        memory_dir,
        thread_id="thread-1",
        run_id="run-1",
        before_outputs=before,
    )
    _mark_worker_started(
        memory_dir,
        thread_id="thread-2",
        run_id="run-2",
        before_outputs=before,
    )
    observation_path = memory_dir / "observations" / "global" / "O-1.md"
    observation_path.parent.mkdir(parents=True)
    observation_path.write_text("# Observation\n", encoding="utf-8")

    first_delta = worker_activity.mark_memory_worker_finished("thread-1", "run-1")
    second_delta = worker_activity.mark_memory_worker_finished("thread-2", "run-2")
    status = worker_activity.memory_worker_status()

    assert first_delta == worker_activity.MemoryOutputDelta(
        memory_dir=memory_dir,
        observation_paths=("observations/global/O-1.md",),
    )
    assert second_delta == worker_activity.MemoryOutputDelta(memory_dir=memory_dir)
    assert status.is_running is False
    assert status.observations_recorded == 1


def test_memory_output_snapshot_uses_posix_relative_paths(tmp_path):
    memory_dir = tmp_path / "memories"
    profile_path = memory_dir / "profile" / "USER_PROFILE.md"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text("# User profile\n", encoding="utf-8")
    observation_path = memory_dir / "observations" / "global" / "O-1.md"
    observation_path.parent.mkdir(parents=True)
    observation_path.write_text("# Observation\n", encoding="utf-8")

    snapshot = worker_activity.snapshot_memory_outputs(memory_dir)

    assert set(snapshot.profile_files) == {"profile/USER_PROFILE.md"}
    assert snapshot.observation_files == frozenset({"observations/global/O-1.md"})


def test_memory_worker_clear_does_not_recount_already_credited_file(tmp_path):
    memory_dir = tmp_path / "memories"
    before = worker_activity.snapshot_memory_outputs(memory_dir)
    _mark_worker_started(
        memory_dir,
        thread_id="thread-1",
        run_id="run-1",
        before_outputs=before,
    )
    _mark_worker_started(
        memory_dir,
        thread_id="thread-2",
        run_id="run-2",
        before_outputs=before,
    )
    observation_path = memory_dir / "observations" / "global" / "O-1.md"
    observation_path.parent.mkdir(parents=True)
    observation_path.write_text("# Observation\n", encoding="utf-8")

    first_delta = worker_activity.mark_memory_worker_finished("thread-1", "run-1")
    assert worker_activity.memory_worker_status().observations_recorded == 1
    worker_activity.clear_completed_memory_activity_counts()
    second_delta = worker_activity.mark_memory_worker_finished("thread-2", "run-2")
    status = worker_activity.memory_worker_status()

    assert first_delta == worker_activity.MemoryOutputDelta(
        memory_dir=memory_dir,
        observation_paths=("observations/global/O-1.md",),
    )
    assert second_delta == worker_activity.MemoryOutputDelta(memory_dir=memory_dir)
    assert status.is_running is False
    assert status.observations_recorded == 0
