from __future__ import annotations

import asyncio
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

from EvoScientist.config import MemoryObservationWriter
from EvoScientist.memory import worker_activity
from EvoScientist.memory.observations import (
    MemoryScope,
    MemorySourceType,
    MemoryType,
    ObservationSearchMode,
    create_read_memory_tool,
    create_search_observations_tool,
    read_observation_file,
    record_observation_file,
    search_observation_files,
)
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
        source_tool_call_id="tool-1",
        record_worker_agent="evomemory-subagent-worker",
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
        source_tool_call_id="tool-1",
        record_worker_agent="evomemory-subagent-worker",
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
        "source": {"type": "subagent", "agent": "code-agent"},
    }
    assert _markdown_sections(body) == {
        "Observation": observation,
        "Why It Matters": why_it_matters,
        "Evidence": evidence,
    }


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
        "source": {"type": "subagent", "agent": "writing-agent"},
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
        "source": {"type": "turn", "agent": "EvoScientist"},
    }


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

    compact = memory_lifecycle._compact_turn_messages(
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

    compact = memory_lifecycle._compact_turn_messages(
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

    compact = memory_lifecycle._compact_turn_messages(
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
    calls = []

    async def fake_launch(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        memory_lifecycle,
        "_alaunch_memory_worker",
        fake_launch,
    )
    middleware = memory_lifecycle.EvoMemoryLifecycleMiddleware(
        memory_dir=tmp_path / "memories",
        workspace_dir=tmp_path / "workspace",
        project_id="P-project",
        role=memory_lifecycle.MemoryLifecycleRole.TURN,
        source_agent="EvoScientist",
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
        await asyncio.sleep(0)

    run_async(run())

    assert len(calls) == 1
    assert calls[0]["role"] == memory_lifecycle.MemoryLifecycleRole.TURN
    assert calls[0]["session_id"] == "thread-1"
    assert calls[0]["source_agent"] == "EvoScientist"
    assert calls[0]["project_id"] == "P-project"
    assert calls[0]["trajectory"] == [
        {"role": "human", "content": "hi"},
        {"role": "ai", "content": "done"},
    ]


def test_subagent_summary_writer_uses_worker_metadata(tmp_path, monkeypatch):
    summary = "Completed the analysis."
    monkeypatch.setattr(
        memory_lifecycle,
        "_current_configurable",
        lambda: {
            "evomemory_source_session_id": "thread-1",
            "evomemory_source_agent": "writing-agent",
            "evomemory_project_id": "P-project",
            "evomemory_trajectory_digest": "digest-1",
        },
    )
    middleware = memory_lifecycle._SubagentSummaryWriterMiddleware(
        memory_dir=tmp_path / "memories"
    )

    state: AgentState[object] = {
        "messages": [],
        "structured_response": memory_lifecycle.SubagentMemoryDecision(summary=summary),
    }
    middleware.after_agent(
        state,
        _runtime(),
    )

    paths = list((tmp_path / "memories" / "executions" / "thread-1").glob("*.md"))
    assert len(paths) == 1
    metadata, body = _read_memory_document(paths[0])
    assert _stable_created_at(metadata) == {
        "id": memory_lifecycle._execution_summary_id(
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


def test_memory_worker_run_kwargs_use_server_thread_id_and_source_metadata(monkeypatch):
    monkeypatch.setattr(
        memory_lifecycle,
        "_worker_workspace_dir",
        lambda _workspace_dir: "/tmp/ws",
    )
    trajectory: list[memory_lifecycle.CompactMessage] = [
        {"role": "human", "content": "hi"}
    ]

    kwargs = memory_lifecycle._memory_worker_run_kwargs(
        role=memory_lifecycle.MemoryLifecycleRole.SUBAGENT,
        thread_id="worker-thread",
        workspace_dir="/active/workspace",
        project_id="P-project",
        source_agent="writing-agent",
        session_id="thread-1",
        trajectory=trajectory,
    )

    assert kwargs["assistant_id"] == memory_lifecycle.SUBAGENT_MEMORY_WORKER_GRAPH_ID
    assert kwargs["metadata"] == {
        "run_kind": "evomemory_subagent_worker",
        "source_session_id": "thread-1",
        "source_agent": "writing-agent",
        "project_id": "P-project",
        "trajectory_digest": memory_lifecycle._trajectory_digest(trajectory),
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
        "evomemory_trajectory_digest": memory_lifecycle._trajectory_digest(trajectory),
    }


def test_memory_worker_graph_accepts_roots_at_build_time(tmp_path, monkeypatch):
    calls = []

    def fake_build(**kwargs):
        calls.append(kwargs)
        return MagicMock()

    monkeypatch.setattr(memory_lifecycle, "_build_memory_worker_agent", fake_build)

    memory_lifecycle.build_memory_worker_graph(
        memory_lifecycle.MemoryLifecycleRole.TURN,
        memory_dir=tmp_path / "memories",
        workspace_dir=tmp_path / "workspace",
    )

    assert calls[0]["memory_dir"] == tmp_path / "memories"
    assert calls[0]["workspace_dir"] == tmp_path / "workspace"


def test_all_mode_gives_memory_workers_observation_tool(tmp_path):
    turn_middleware = memory_lifecycle._memory_worker_middleware(
        memory_dir=tmp_path / "memories",
        workspace_dir=tmp_path / "workspace",
        role=memory_lifecycle.MemoryLifecycleRole.TURN,
        observation_writer=MemoryObservationWriter.ALL,
    )
    subagent_middleware = memory_lifecycle._memory_worker_middleware(
        memory_dir=tmp_path / "memories",
        workspace_dir=tmp_path / "workspace",
        role=memory_lifecycle.MemoryLifecycleRole.SUBAGENT,
        observation_writer=MemoryObservationWriter.ALL,
    )

    assert [tool.name for tool in turn_middleware[0].tools] == [
        "search_observations",
        "read_memory",
        "record_observation",
    ]
    assert [tool.name for tool in subagent_middleware[0].tools] == [
        "search_observations",
        "read_memory",
        "record_observation",
    ]


def test_memory_worker_observation_writer_modes(tmp_path):
    agent_only = memory_lifecycle._memory_worker_middleware(
        memory_dir=tmp_path / "memories",
        workspace_dir=tmp_path / "workspace",
        role=memory_lifecycle.MemoryLifecycleRole.SUBAGENT,
        observation_writer=MemoryObservationWriter.AGENT,
    )
    worker_subagent = memory_lifecycle._memory_worker_middleware(
        memory_dir=tmp_path / "memories",
        workspace_dir=tmp_path / "workspace",
        role=memory_lifecycle.MemoryLifecycleRole.SUBAGENT,
        observation_writer=MemoryObservationWriter.WORKER,
    )
    worker_turn = memory_lifecycle._memory_worker_middleware(
        memory_dir=tmp_path / "memories",
        workspace_dir=tmp_path / "workspace",
        role=memory_lifecycle.MemoryLifecycleRole.TURN,
        observation_writer=MemoryObservationWriter.WORKER,
    )

    assert [tool.name for tool in agent_only[0].tools] == [
        "search_observations",
        "read_memory",
    ]
    assert [tool.name for tool in worker_subagent[0].tools] == [
        "search_observations",
        "read_memory",
        "record_observation",
    ]
    assert [tool.name for tool in worker_turn[0].tools] == [
        "search_observations",
        "read_memory",
        "record_observation",
    ]


def test_memory_worker_prompts_match_observation_tool_availability():
    turn_profile_only = memory_lifecycle._memory_worker_system_prompt(
        memory_lifecycle.MemoryLifecycleRole.TURN,
        enable_profile_memory=True,
        enable_observation_tool=False,
    )
    turn_with_observation_flag = memory_lifecycle._memory_worker_system_prompt(
        memory_lifecycle.MemoryLifecycleRole.TURN,
        enable_profile_memory=True,
        enable_observation_tool=True,
    )
    subagent_profile_only = memory_lifecycle._memory_worker_system_prompt(
        memory_lifecycle.MemoryLifecycleRole.SUBAGENT,
        enable_profile_memory=True,
        enable_observation_tool=False,
    )
    subagent_with_observations = memory_lifecycle._memory_worker_system_prompt(
        memory_lifecycle.MemoryLifecycleRole.SUBAGENT,
        enable_profile_memory=True,
        enable_observation_tool=True,
    )
    subagent_observations_only = memory_lifecycle._memory_worker_system_prompt(
        memory_lifecycle.MemoryLifecycleRole.SUBAGENT,
        enable_profile_memory=False,
        enable_observation_tool=True,
    )
    turn_observations_only = memory_lifecycle._memory_worker_system_prompt(
        memory_lifecycle.MemoryLifecycleRole.TURN,
        enable_profile_memory=False,
        enable_observation_tool=True,
    )

    assert "record_observation" not in turn_profile_only
    assert "record_observation" in turn_with_observation_flag
    assert "record_observation" not in subagent_profile_only
    assert "record_observation" in subagent_with_observations
    assert "record_observation" in subagent_observations_only
    assert "/memories/profile/" not in subagent_observations_only
    assert "record_observation" in turn_observations_only
    assert "/memories/profile/" not in turn_observations_only


def test_sync_memory_worker_watcher_untracks_without_counting_on_poll_abort(
    tmp_path, monkeypatch
):
    worker_activity.reset_memory_worker_status_for_tests()
    memory_dir = tmp_path / "memories"
    worker_activity.mark_memory_worker_started(
        thread_id="worker-thread",
        run_id="run-1",
        memory_dir=memory_dir,
    )
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
    monkeypatch.setattr(memory_lifecycle, "_MEMORY_WORKER_POLL_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(memory_lifecycle, "_MEMORY_WORKER_MAX_POLL_FAILURES", 1)

    try:
        memory_lifecycle._watch_memory_worker_run_sync(
            url="http://x",
            thread_id="worker-thread",
            run_id="run-1",
        )
        status = worker_activity.memory_worker_status()
        assert status.is_running is False
        assert status.profile_updates == 0
        assert status.observations_recorded == 0
    finally:
        worker_activity.reset_memory_worker_status_for_tests()


def test_async_memory_worker_watcher_untracks_without_counting_on_poll_abort(
    tmp_path, monkeypatch, run_async
):
    worker_activity.reset_memory_worker_status_for_tests()
    memory_dir = tmp_path / "memories"
    worker_activity.mark_memory_worker_started(
        thread_id="worker-thread",
        run_id="run-1",
        memory_dir=memory_dir,
    )
    observation_path = memory_dir / "observations" / "global" / "O-1.md"
    observation_path.parent.mkdir(parents=True)
    observation_path.write_text("# Observation\n", encoding="utf-8")

    class _Runs:
        async def get(self, **_kwargs):
            raise RuntimeError("poll failed")

    monkeypatch.setattr(memory_lifecycle, "_MEMORY_WORKER_POLL_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(memory_lifecycle, "_MEMORY_WORKER_MAX_POLL_FAILURES", 1)

    try:
        run_async(
            memory_lifecycle._watch_memory_worker_run_async(
                SimpleNamespace(runs=_Runs()),
                thread_id="worker-thread",
                run_id="run-1",
            )
        )
        status = worker_activity.memory_worker_status()
        assert status.is_running is False
        assert status.profile_updates == 0
        assert status.observations_recorded == 0
    finally:
        worker_activity.reset_memory_worker_status_for_tests()


def test_async_memory_worker_watcher_counts_completion_under_blockbuster(
    tmp_path, run_async
):
    worker_activity.reset_memory_worker_status_for_tests()
    memory_dir = tmp_path / "memories"
    worker_activity.mark_memory_worker_started(
        thread_id="worker-thread",
        run_id="run-1",
        memory_dir=memory_dir,
        before_outputs=worker_activity.snapshot_memory_outputs(memory_dir),
    )
    profile_path = memory_dir / "profile" / "USER_PROFILE.md"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text("# User profile\n\n- later update\n", encoding="utf-8")

    class _Runs:
        async def get(self, **_kwargs):
            return {"status": "success"}

    async def run():
        blocker = BlockBuster(scanned_modules=[memory_lifecycle, worker_activity])
        blocker.activate()
        try:
            await memory_lifecycle._watch_memory_worker_run_async(
                SimpleNamespace(runs=_Runs()),
                thread_id="worker-thread",
                run_id="run-1",
            )
        finally:
            blocker.deactivate()

    try:
        run_async(run())
        status = worker_activity.memory_worker_status()
        assert status.is_running is False
        assert status.profile_updates == 1
        assert status.observations_recorded == 0
    finally:
        worker_activity.reset_memory_worker_status_for_tests()


def test_memory_worker_watcher_untracks_when_client_creation_fails(
    tmp_path, monkeypatch
):
    worker_activity.reset_memory_worker_status_for_tests()
    memory_dir = tmp_path / "memories"
    worker_activity.mark_memory_worker_started(
        thread_id="worker-thread",
        run_id="run-1",
        memory_dir=memory_dir,
    )
    profile_path = memory_dir / "profile" / "USER_PROFILE.md"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text("# User profile\n\n- later update\n", encoding="utf-8")

    monkeypatch.setattr(
        "langgraph_sdk.get_sync_client",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("client failed")),
    )

    try:
        with pytest.raises(RuntimeError, match="client failed"):
            memory_lifecycle._watch_memory_worker_run_sync(
                url="http://x",
                thread_id="worker-thread",
                run_id="run-1",
            )
        status = worker_activity.memory_worker_status()
        assert status.is_running is False
        assert status.profile_updates == 0
        assert status.observations_recorded == 0
    finally:
        worker_activity.reset_memory_worker_status_for_tests()


def test_memory_worker_watcher_finishes_on_terminal_status(tmp_path, monkeypatch):
    worker_activity.reset_memory_worker_status_for_tests()
    worker_activity.mark_memory_worker_started(
        thread_id="worker-thread",
        run_id="run-1",
        memory_dir=tmp_path / "memories",
    )

    class _Runs:
        def get(self, **_kwargs):
            return {"status": "success"}

    monkeypatch.setattr(
        "langgraph_sdk.get_sync_client",
        lambda **_kwargs: SimpleNamespace(runs=_Runs()),
    )
    monkeypatch.setattr(memory_lifecycle, "_MEMORY_WORKER_POLL_INTERVAL_SECONDS", 0)

    try:
        memory_lifecycle._watch_memory_worker_run_sync(
            url="http://x",
            thread_id="worker-thread",
            run_id="run-1",
        )
        assert worker_activity.memory_worker_status().is_running is False
    finally:
        worker_activity.reset_memory_worker_status_for_tests()


def test_sync_watcher_deletes_worker_thread_on_terminal_status(tmp_path, monkeypatch):
    """Finished workers leave no checkpoint residue: thread is deleted."""
    worker_activity.reset_memory_worker_status_for_tests()
    worker_activity.mark_memory_worker_started(
        thread_id="worker-thread",
        run_id="run-1",
        memory_dir=tmp_path / "memories",
    )
    deleted: list[str] = []

    class _Runs:
        def get(self, **_kwargs):
            return {"status": "success"}

    class _Threads:
        def delete(self, thread_id):
            deleted.append(thread_id)

    monkeypatch.setattr(
        "langgraph_sdk.get_sync_client",
        lambda **_kwargs: SimpleNamespace(runs=_Runs(), threads=_Threads()),
    )
    monkeypatch.setattr(memory_lifecycle, "_MEMORY_WORKER_POLL_INTERVAL_SECONDS", 0)

    try:
        memory_lifecycle._watch_memory_worker_run_sync(
            url="http://x",
            thread_id="worker-thread",
            run_id="run-1",
        )
        assert deleted == ["worker-thread"]
        assert worker_activity.memory_worker_status().is_running is False
    finally:
        worker_activity.reset_memory_worker_status_for_tests()


def test_sync_watcher_delete_failure_still_marks_finished(tmp_path, monkeypatch):
    """Thread deletion is best-effort: a failure must not break accounting."""
    worker_activity.reset_memory_worker_status_for_tests()
    worker_activity.mark_memory_worker_started(
        thread_id="worker-thread",
        run_id="run-1",
        memory_dir=tmp_path / "memories",
    )

    class _Runs:
        def get(self, **_kwargs):
            return {"status": "success"}

    class _Threads:
        def delete(self, thread_id):
            raise RuntimeError("delete failed")

    monkeypatch.setattr(
        "langgraph_sdk.get_sync_client",
        lambda **_kwargs: SimpleNamespace(runs=_Runs(), threads=_Threads()),
    )
    monkeypatch.setattr(memory_lifecycle, "_MEMORY_WORKER_POLL_INTERVAL_SECONDS", 0)

    try:
        memory_lifecycle._watch_memory_worker_run_sync(
            url="http://x",
            thread_id="worker-thread",
            run_id="run-1",
        )
        assert worker_activity.memory_worker_status().is_running is False
    finally:
        worker_activity.reset_memory_worker_status_for_tests()


def test_sync_watcher_does_not_delete_thread_on_poll_abort(tmp_path, monkeypatch):
    """A run we lost track of may still be live — never delete its thread."""
    worker_activity.reset_memory_worker_status_for_tests()
    worker_activity.mark_memory_worker_started(
        thread_id="worker-thread",
        run_id="run-1",
        memory_dir=tmp_path / "memories",
    )
    deleted: list[str] = []

    class _Runs:
        def get(self, **_kwargs):
            raise RuntimeError("poll failed")

    class _Threads:
        def delete(self, thread_id):
            deleted.append(thread_id)

    monkeypatch.setattr(
        "langgraph_sdk.get_sync_client",
        lambda **_kwargs: SimpleNamespace(runs=_Runs(), threads=_Threads()),
    )
    monkeypatch.setattr(memory_lifecycle, "_MEMORY_WORKER_POLL_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(memory_lifecycle, "_MEMORY_WORKER_MAX_POLL_FAILURES", 1)

    try:
        memory_lifecycle._watch_memory_worker_run_sync(
            url="http://x",
            thread_id="worker-thread",
            run_id="run-1",
        )
        assert deleted == []
    finally:
        worker_activity.reset_memory_worker_status_for_tests()


def test_async_watcher_deletes_worker_thread_on_terminal_status(
    tmp_path, monkeypatch, run_async
):
    worker_activity.reset_memory_worker_status_for_tests()
    worker_activity.mark_memory_worker_started(
        thread_id="worker-thread",
        run_id="run-1",
        memory_dir=tmp_path / "memories",
    )
    deleted: list[str] = []

    class _Runs:
        async def get(self, **_kwargs):
            return {"status": "success"}

    class _Threads:
        async def delete(self, thread_id):
            # Accounting must complete BEFORE the best-effort deletion —
            # cancellation mid-deletion must never leave the worker
            # stuck as "running" (CodeRabbit on #279).
            assert worker_activity.memory_worker_status().is_running is False
            deleted.append(thread_id)

    monkeypatch.setattr(memory_lifecycle, "_MEMORY_WORKER_POLL_INTERVAL_SECONDS", 0)

    try:
        run_async(
            memory_lifecycle._watch_memory_worker_run_async(
                SimpleNamespace(runs=_Runs(), threads=_Threads()),
                thread_id="worker-thread",
                run_id="run-1",
            )
        )
        assert deleted == ["worker-thread"]
        assert worker_activity.memory_worker_status().is_running is False
    finally:
        worker_activity.reset_memory_worker_status_for_tests()


def test_memory_worker_skips_when_langgraph_dev_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_lifecycle, "_memory_worker_url", lambda: "http://x")
    monkeypatch.setattr(
        "EvoScientist.langgraph_dev.manager.is_langgraph_dev_running",
        lambda **_kwargs: False,
    )

    def fail_get_sync_client(*_args, **_kwargs):
        raise AssertionError("client should not be created")

    monkeypatch.setattr("langgraph_sdk.get_sync_client", fail_get_sync_client)

    trajectory: list[memory_lifecycle.CompactMessage] = [
        {"role": "human", "content": "hi"}
    ]

    memory_lifecycle._launch_memory_worker(
        role=memory_lifecycle.MemoryLifecycleRole.TURN,
        memory_dir=tmp_path / "memories",
        workspace_dir=tmp_path / "workspace",
        project_id="P-project",
        source_agent="EvoScientist",
        session_id="thread-1",
        trajectory=trajectory,
    )


def test_memory_worker_launch_marks_active_status(tmp_path, monkeypatch):
    worker_activity.reset_memory_worker_status_for_tests()
    monkeypatch.setattr(memory_lifecycle, "_memory_worker_url", lambda: "http://x")
    monkeypatch.setattr(
        memory_lifecycle,
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

    spawned = []
    monkeypatch.setattr(
        memory_lifecycle,
        "_spawn_memory_worker_status_thread",
        lambda **kwargs: spawned.append(kwargs),
    )

    trajectory: list[memory_lifecycle.CompactMessage] = [
        {"role": "human", "content": "hi"}
    ]

    memory_dir = tmp_path / "memories"
    memory_lifecycle._launch_memory_worker(
        role=memory_lifecycle.MemoryLifecycleRole.TURN,
        memory_dir=memory_dir,
        workspace_dir=tmp_path / "workspace",
        project_id="P-project",
        source_agent="EvoScientist",
        session_id="thread-1",
        trajectory=trajectory,
    )

    try:
        assert worker_activity.memory_worker_status().is_running is True
        expected_metadata = {
            "run_kind": "evomemory_turn_worker",
            "source_session_id": "thread-1",
            "source_agent": "EvoScientist",
            "project_id": "P-project",
            "trajectory_digest": memory_lifecycle._trajectory_digest(trajectory),
            "workspace_dir": "/tmp/ws",
        }
        fake_client.threads.create.assert_called_once_with(
            graph_id=memory_lifecycle.TURN_MEMORY_WORKER_GRAPH_ID,
            metadata=expected_metadata,
        )
        fake_client.runs.create.assert_called_once()
        run_kwargs = fake_client.runs.create.call_args.kwargs
        assert run_kwargs["thread_id"] == "worker-thread"
        assert run_kwargs["metadata"] == expected_metadata
        assert run_kwargs["config"]["configurable"]["thread_id"] == "worker-thread"
        assert spawned == [
            {"url": "http://x", "thread_id": "worker-thread", "run_id": "run-1"}
        ]
        profile_path = memory_dir / "profile" / "USER_PROFILE.md"
        profile_path.parent.mkdir(parents=True)
        profile_path.write_text("# User profile\n\n- remembered\n", encoding="utf-8")
        observation_path = memory_dir / "observations" / "global" / "O-1.md"
        observation_path.parent.mkdir(parents=True)
        observation_path.write_text("# Observation\n", encoding="utf-8")
    finally:
        worker_activity.mark_memory_worker_finished("worker-thread", "run-1")
    status = worker_activity.memory_worker_status()
    assert status.is_running is False
    assert status.profile_updates == 1
    assert status.observations_recorded == 1
    worker_activity.reset_memory_worker_status_for_tests()


def test_async_memory_worker_launch_offloads_blocking_work(
    tmp_path, monkeypatch, run_async
):
    worker_activity.reset_memory_worker_status_for_tests()
    monkeypatch.setattr(memory_lifecycle, "_memory_worker_url", lambda: "http://x")
    monkeypatch.setattr(
        memory_lifecycle,
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
    monkeypatch.setattr(memory_lifecycle, "snapshot_memory_outputs", fake_snapshot)

    class _Threads:
        async def create(self, **_kwargs):
            return {"thread_id": "worker-thread"}

    class _Runs:
        async def create(self, **_kwargs):
            return {"run_id": "run-1", "status": "pending"}

    fake_client = SimpleNamespace(threads=_Threads(), runs=_Runs())
    monkeypatch.setattr("langgraph_sdk.get_client", lambda **_kwargs: fake_client)

    spawned = []
    monkeypatch.setattr(
        memory_lifecycle,
        "_spawn_memory_worker_status_thread",
        lambda **kwargs: spawned.append(kwargs),
    )

    async def run():
        event_loop_thread = threading.get_ident()
        await memory_lifecycle._alaunch_memory_worker(
            role=memory_lifecycle.MemoryLifecycleRole.TURN,
            memory_dir=tmp_path / "memories",
            workspace_dir=tmp_path / "workspace",
            project_id="P-project",
            source_agent="EvoScientist",
            session_id="thread-1",
            trajectory=[{"role": "human", "content": "hi"}],
        )
        return event_loop_thread

    try:
        event_loop_thread = run_async(run())
        assert [name for name, _thread_id in call_threads] == ["health", "snapshot"]
        assert all(thread_id != event_loop_thread for _name, thread_id in call_threads)
        assert worker_activity.memory_worker_status().is_running is True
        assert spawned == [
            {"url": "http://x", "thread_id": "worker-thread", "run_id": "run-1"}
        ]
    finally:
        worker_activity.reset_memory_worker_status_for_tests()


def test_memory_worker_saved_counts_clear_preserves_pending_worker_delta(tmp_path):
    worker_activity.reset_memory_worker_status_for_tests()
    memory_dir = tmp_path / "memories"
    before = worker_activity.snapshot_memory_outputs(memory_dir)
    worker_activity.mark_memory_worker_started(
        thread_id="finished-thread",
        run_id="finished-run",
        memory_dir=memory_dir,
        before_outputs=before,
    )
    profile_path = memory_dir / "profile" / "USER_PROFILE.md"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text("# User profile\n\n- remembered\n", encoding="utf-8")
    worker_activity.mark_memory_worker_finished("finished-thread", "finished-run")
    worker_activity.mark_memory_worker_started(
        thread_id="active-thread",
        run_id="active-run",
        memory_dir=memory_dir,
    )

    worker_activity.clear_memory_worker_saved_counts()
    assert worker_activity.memory_worker_status().is_running is True
    observation_path = memory_dir / "observations" / "global" / "O-1.md"
    observation_path.parent.mkdir(parents=True)
    observation_path.write_text("# Observation\n", encoding="utf-8")
    worker_activity.mark_memory_worker_finished("active-thread", "active-run")
    status = worker_activity.memory_worker_status()

    try:
        assert status.is_running is False
        assert status.profile_updates == 0
        assert status.observations_recorded == 1
    finally:
        worker_activity.reset_memory_worker_status_for_tests()


def test_memory_worker_observed_outputs_includes_active_worker_delta(tmp_path):
    worker_activity.reset_memory_worker_status_for_tests()
    memory_dir = tmp_path / "memories"
    before = worker_activity.snapshot_memory_outputs(memory_dir)
    worker_activity.mark_memory_worker_started(
        thread_id="active-thread",
        run_id="active-run",
        memory_dir=memory_dir,
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

    try:
        status = worker_activity.memory_worker_observed_outputs()
        assert status.is_running is True
        assert status.observations_recorded == 1
        assert status.profile_updates == 0
        assert worker_activity.memory_worker_status().observations_recorded == 0
    finally:
        worker_activity.reset_memory_worker_status_for_tests()


def test_one_shot_cli_wait_keeps_polling_after_observed_memory_output(monkeypatch):
    from EvoScientist.cli import interactive

    now = 0.0
    printed = []
    observed_calls = 0

    def fake_monotonic():
        return now

    def fake_sleep(seconds):
        nonlocal now
        now += seconds

    def fake_observed_outputs():
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

    monkeypatch.setattr(interactive.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(interactive.time, "sleep", fake_sleep)
    monkeypatch.setattr(interactive.console, "print", lambda text: printed.append(text))
    monkeypatch.setattr(
        worker_activity,
        "memory_worker_observed_outputs",
        fake_observed_outputs,
    )

    interactive._wait_for_memory_workers_before_exit(timeout_seconds=10)

    assert observed_calls == 8
    assert any("EvoMemory saved 1 observation(s)." in str(line) for line in printed)
    assert any(
        "EvoMemory saved 1 observation(s), 1 profile update(s)." in str(line)
        for line in printed
    )
    assert not any("still running" in str(line) for line in printed)


def test_one_shot_cli_wait_reports_fast_worker_output(monkeypatch):
    from EvoScientist.cli import interactive

    printed = []

    monkeypatch.setattr(interactive.console, "print", lambda text: printed.append(text))
    monkeypatch.setattr(
        worker_activity,
        "memory_worker_observed_outputs",
        lambda: worker_activity.MemoryWorkerStatusSnapshot(
            is_running=False,
            observations_recorded=1,
        ),
    )

    interactive._wait_for_memory_workers_before_exit(timeout_seconds=10)

    assert printed == ["[dim]EvoMemory saved 1 observation(s).[/dim]"]


def test_memory_worker_status_dedupes_overlapping_observation_deltas(tmp_path):
    worker_activity.reset_memory_worker_status_for_tests()
    memory_dir = tmp_path / "memories"
    before = worker_activity.snapshot_memory_outputs(memory_dir)
    worker_activity.mark_memory_worker_started(
        thread_id="thread-1",
        run_id="run-1",
        memory_dir=memory_dir,
        before_outputs=before,
    )
    worker_activity.mark_memory_worker_started(
        thread_id="thread-2",
        run_id="run-2",
        memory_dir=memory_dir,
        before_outputs=before,
    )
    observation_path = memory_dir / "observations" / "global" / "O-1.md"
    observation_path.parent.mkdir(parents=True)
    observation_path.write_text("# Observation\n", encoding="utf-8")

    worker_activity.mark_memory_worker_finished("thread-1", "run-1")
    worker_activity.mark_memory_worker_finished("thread-2", "run-2")
    status = worker_activity.memory_worker_status()

    try:
        assert status.is_running is False
        assert status.observations_recorded == 1
    finally:
        worker_activity.reset_memory_worker_status_for_tests()


def test_memory_worker_clear_does_not_recount_already_credited_file(tmp_path):
    worker_activity.reset_memory_worker_status_for_tests()
    memory_dir = tmp_path / "memories"
    before = worker_activity.snapshot_memory_outputs(memory_dir)
    worker_activity.mark_memory_worker_started(
        thread_id="thread-1",
        run_id="run-1",
        memory_dir=memory_dir,
        before_outputs=before,
    )
    worker_activity.mark_memory_worker_started(
        thread_id="thread-2",
        run_id="run-2",
        memory_dir=memory_dir,
        before_outputs=before,
    )
    observation_path = memory_dir / "observations" / "global" / "O-1.md"
    observation_path.parent.mkdir(parents=True)
    observation_path.write_text("# Observation\n", encoding="utf-8")

    worker_activity.mark_memory_worker_finished("thread-1", "run-1")
    assert worker_activity.memory_worker_status().observations_recorded == 1
    worker_activity.clear_memory_worker_saved_counts()
    worker_activity.mark_memory_worker_finished("thread-2", "run-2")
    status = worker_activity.memory_worker_status()

    try:
        assert status.is_running is False
        assert status.observations_recorded == 0
    finally:
        worker_activity.reset_memory_worker_status_for_tests()
