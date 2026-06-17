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
from langchain.agents.middleware.types import AgentState
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.runtime import ExecutionInfo, Runtime
from pydantic import BaseModel

from EvoScientist.config import MemoryObservationWriter
from EvoScientist.memory import synthesis as memory_synthesis
from EvoScientist.memory import worker_activity
from EvoScientist.memory.knowledge import (
    create_read_memory_tool,
    create_search_memory_tool,
    record_knowledge_file,
    search_memory_files,
)
from EvoScientist.memory.observations import (
    read_observation_file,
    record_observation_file,
    search_observation_files,
)
from EvoScientist.memory.synthesis import (
    SynthesisAction,
    SynthesisArchiveDecision,
    SynthesisCreateDecision,
    SynthesisReviewDecision,
    SynthesisUpdateDecision,
    apply_synthesis_review_decision,
)
from EvoScientist.memory.types import (
    KnowledgeStatus,
    MemoryLevel,
    MemoryScope,
    MemorySearchMode,
    MemorySourceType,
    MemoryType,
)
from EvoScientist.middleware import memory_lifecycle


def _read_memory_document(path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    frontmatter, body = text.removeprefix("---\n").split("\n---\n", 1)
    metadata = yaml.safe_load(frontmatter)
    assert isinstance(metadata, dict)
    return metadata, body


def _write_knowledge_status_file(
    memory_dir,
    *,
    status: str = "active",
    body: str = "Knowledge body.",
) -> None:
    path = memory_dir / "knowledge" / "projects" / "P-project" / "K-1.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "---",
                "id: K-1",
                "summary: Status test knowledge.",
                "memory_type: semantic",
                "scope: project",
                f"status: {status}",
                "---",
                body,
            ]
        ),
        encoding="utf-8",
    )


def _stable_created_at(metadata: dict[str, Any]) -> dict[str, Any]:
    created_at = metadata.get("created_at")
    assert isinstance(created_at, str)
    datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
    return {**metadata, "created_at": "<created_at>"}


def _stable_memory_timestamps(metadata: dict[str, Any]) -> dict[str, Any]:
    out = dict(metadata)
    for key in ("created_at", "updated_at", "archived_at"):
        if key not in out:
            continue
        value = out[key]
        assert isinstance(value, str)
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
        out[key] = f"<{key}>"
    return out


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

    assert [hit["memory_id"] for hit in hits] == [first["observation_id"]]
    assert hits[0]["level"] == MemoryLevel.OBSERVATION
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
        )[0]["memory_id"]
        == second["observation_id"]
    )

    tool = create_search_memory_tool(
        memory_dir=memories,
        project_id="P-project",
    )
    payload = json.loads(
        tool.run(
            {
                "query": "GraphQL userName frontend",
                "memory_level": "observation",
                "limit": 5,
            }
        )
    )
    assert list(payload) == ["results"]
    assert payload["results"][0]["memory_id"] == first["observation_id"]


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
    payload = json.loads(tool.run({"memory_id": result["observation_id"]}))
    assert payload == {"text": read["text"]}

    missing = json.loads(tool.run({"memory_id": "../not-a-memory"}))
    assert missing == {
        "error": "No memory with that ID exists in global or current-project memory.",
    }


def test_knowledge_memory_writes_support_and_combined_search_prefers_it(tmp_path):
    memories = tmp_path / "memories"
    observation = record_observation_file(
        memory_dir=memories,
        project_id="P-project",
        memory_type=MemoryType.PROCEDURAL,
        summary="Focused pytest should run before broad suites.",
        observation="Run focused pytest for the changed test file before the suite.",
        why_it_matters="Future agents catch local regressions faster.",
        scope=MemoryScope.PROJECT,
        source_type=MemorySourceType.SUBAGENT,
        source_session_id="thread-1",
        source_agent="code-agent",
    )

    result = record_knowledge_file(
        memory_dir=memories,
        project_id="P-project",
        memory_type=MemoryType.PROCEDURAL,
        summary="Run focused pytest before broad regression suites.",
        knowledge=(
            "When editing a narrow area, first run the focused pytest file that "
            "covers the change before escalating to the broader suite."
        ),
        when_to_use="Use during coding or debugging changes with a focused test.",
        supporting_observation_ids=[observation["observation_id"]],
        scope=MemoryScope.PROJECT,
    )

    path = memories / result["path"].removeprefix("/memories/")
    metadata, body = _read_memory_document(path)

    assert result["created"] is True
    assert result["path"].startswith("/memories/knowledge/projects/P-project/K-")
    assert _stable_memory_timestamps(metadata) == {
        "id": result["knowledge_id"],
        "created_at": "<created_at>",
        "updated_at": "<updated_at>",
        "summary": "Run focused pytest before broad regression suites.",
        "memory_type": "procedural",
        "scope": "project",
        "project_id": "P-project",
        "status": "active",
        "supporting_observation_ids": [observation["observation_id"]],
        "source": {"type": "synthesis", "agent": "evomemory-synthesizer"},
    }
    assert _markdown_sections(body) == {
        "Knowledge": (
            "When editing a narrow area, first run the focused pytest file that "
            "covers the change before escalating to the broader suite."
        ),
        "When To Use": "Use during coding or debugging changes with a focused test.",
        "Support": (
            f"- {observation['observation_id']}: "
            "Focused pytest should run before broad suites."
        ),
    }

    hits = search_memory_files(
        memory_dir=memories,
        project_id="P-project",
        query="focused pytest regression suite",
    )
    assert [hit["memory_id"] for hit in hits] == [result["knowledge_id"]]
    raw_hits = search_memory_files(
        memory_dir=memories,
        project_id="P-project",
        query="focused pytest regression suite",
        memory_level="observation",
    )
    assert [hit["memory_id"] for hit in raw_hits] == [observation["observation_id"]]

    tool = create_read_memory_tool(
        memory_dir=memories,
        project_id="P-project",
    )
    knowledge_payload = json.loads(tool.run({"memory_id": result["knowledge_id"]}))
    assert result["knowledge_id"] in knowledge_payload["text"]
    observation_payload = json.loads(
        tool.run({"memory_id": observation["observation_id"]})
    )
    assert observation["observation_id"] in observation_payload["text"]

    search_tool = create_search_memory_tool(
        memory_dir=memories,
        project_id="P-project",
    )
    schema = search_tool.tool_call_schema
    assert isinstance(schema, type)
    assert issubclass(schema, BaseModel)
    assert sorted(schema.model_json_schema()["properties"]) == [
        "include_archived_knowledge",
        "limit",
        "memory_level",
        "memory_type",
        "mode",
        "query",
        "scope",
    ]


def test_build_synthesis_context_seeds_uncovered_observations_only(tmp_path):
    memories = tmp_path / "memories"
    covered = record_observation_file(
        memory_dir=memories,
        project_id="P-project",
        memory_type=MemoryType.PROCEDURAL,
        summary="Covered observation already supports knowledge.",
        observation="Covered observation body " + ("a" * 900),
        why_it_matters="Existing knowledge already cites this observation.",
        scope=MemoryScope.PROJECT,
        source_type=MemorySourceType.SUBAGENT,
        source_session_id="thread-1",
        source_agent="code-agent",
    )
    uncovered = record_observation_file(
        memory_dir=memories,
        project_id="P-project",
        memory_type=MemoryType.PROCEDURAL,
        summary="Uncovered observation should seed synthesis.",
        observation="Uncovered observation body " + ("b" * 900),
        why_it_matters="The synthesis worker should inspect it with read_memory.",
        scope=MemoryScope.PROJECT,
        source_type=MemorySourceType.SUBAGENT,
        source_session_id="thread-1",
        source_agent="code-agent",
    )
    record_knowledge_file(
        memory_dir=memories,
        project_id="P-project",
        memory_type=MemoryType.PROCEDURAL,
        summary="Existing knowledge covers one observation.",
        knowledge="Existing knowledge should be found through search_memory.",
        supporting_observation_ids=[covered["observation_id"]],
        scope=MemoryScope.PROJECT,
    )

    context = memory_synthesis.build_synthesis_context(
        memory_dir=memories,
        project_id="P-project",
        seed_observation_ids=(
            f" {covered['observation_id']} ",
            uncovered["observation_id"],
            uncovered["observation_id"],
            "",
        ),
    )

    assert context is not None
    assert set(context) == {
        "project_id",
        "uncovered_observations",
        "memory_inventory",
    }
    assert [obs["id"] for obs in context["uncovered_observations"]] == [
        uncovered["observation_id"]
    ]
    seed = context["uncovered_observations"][0]
    assert set(seed) == {"id", "path", "memory_type", "scope", "summary", "snippet"}
    assert "body" not in seed
    assert (
        len(seed["snippet"])
        <= memory_synthesis.SYNTHESIS_CONTEXT_OBSERVATION_SNIPPET_CHARS
    )
    assert context["memory_inventory"] == {
        "active_knowledge_count": 1,
        "seed_observation_count": 2,
    }


def test_build_synthesis_context_shrinks_detail_without_dropping_seed_ids(
    monkeypatch, tmp_path
):
    seed_ids = tuple(f"O-{index}" for index in range(6))

    def fake_read_observation_file(**kwargs):
        observation_id = kwargs["observation_id"]
        return {
            "observation_id": observation_id,
            "path": f"/memories/observations/projects/P-project/{observation_id}.md",
            "memory_type": MemoryType.SEMANTIC,
            "scope": MemoryScope.PROJECT,
            "summary": f"Observation {observation_id} " + ("x" * 800),
            "text": f"Observation body {observation_id} " + ("y" * 1600),
        }

    monkeypatch.setattr(
        memory_synthesis,
        "knowledge_search_documents",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        memory_synthesis,
        "read_observation_file",
        fake_read_observation_file,
    )

    context = memory_synthesis.build_synthesis_context(
        memory_dir=tmp_path / "memories",
        project_id="P-project",
        seed_observation_ids=seed_ids,
        max_chars=2200,
    )

    assert context is not None
    assert len(memory_synthesis._pretty_json(context)) <= 2200
    assert [obs["id"] for obs in context["uncovered_observations"]] == list(seed_ids)
    assert {obs["snippet"] for obs in context["uncovered_observations"]} == {""}
    assert context["memory_inventory"]["seed_observation_count"] == len(seed_ids)


def test_build_synthesis_context_keeps_seed_ids_when_still_oversized(
    monkeypatch, tmp_path
):
    seed_ids = tuple(f"O-{index}" for index in range(4))

    def fake_read_observation_file(**kwargs):
        observation_id = kwargs["observation_id"]
        return {
            "observation_id": observation_id,
            "path": f"/memories/observations/projects/P-project/{observation_id}.md",
            "memory_type": MemoryType.SEMANTIC,
            "scope": MemoryScope.PROJECT,
            "summary": "summary",
            "text": "body",
        }

    monkeypatch.setattr(
        memory_synthesis,
        "knowledge_search_documents",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        memory_synthesis,
        "read_observation_file",
        fake_read_observation_file,
    )

    context = memory_synthesis.build_synthesis_context(
        memory_dir=tmp_path / "memories",
        project_id="P-project",
        seed_observation_ids=seed_ids,
        max_chars=100,
    )

    assert context is not None
    assert len(memory_synthesis._pretty_json(context)) > 100
    assert [obs["id"] for obs in context["uncovered_observations"]] == list(seed_ids)
    assert {obs["summary"] for obs in context["uncovered_observations"]} == {""}
    assert {obs["snippet"] for obs in context["uncovered_observations"]} == {""}


def test_synthesis_memory_tools_resolve_project_from_run_config(tmp_path, monkeypatch):
    memories = tmp_path / "memories"
    observation = record_observation_file(
        memory_dir=memories,
        project_id="P-project",
        memory_type=MemoryType.PROCEDURAL,
        summary="Synthesis tools read configured project memory.",
        observation="Configured synthesis memory tools can search and read records.",
        why_it_matters="The synthesis graph is shared across projects.",
        scope=MemoryScope.PROJECT,
        source_type=MemorySourceType.SUBAGENT,
        source_session_id="thread-1",
        source_agent="code-agent",
    )
    monkeypatch.setattr(
        memory_synthesis,
        "_current_configurable",
        lambda: {"evomemory_project_id": "P-project"},
    )
    tools = memory_synthesis._synthesis_memory_tools(memory_dir=memories)
    by_name = {tool.name: tool for tool in tools}

    search_payload = json.loads(
        by_name["search_memory"].run(
            {
                "query": "configured synthesis memory",
                "memory_level": "observation",
            }
        )
    )
    read_payload = json.loads(
        by_name["read_memory"].run({"memory_id": observation["observation_id"]})
    )

    assert search_payload["results"][0]["memory_id"] == observation["observation_id"]
    assert observation["observation_id"] in read_payload["text"]


def test_apply_synthesis_review_handles_create_update_and_archive(tmp_path):
    memories = tmp_path / "memories"
    observation = record_observation_file(
        memory_dir=memories,
        project_id="P-project",
        memory_type=MemoryType.SEMANTIC,
        summary="Resolver aliases preserve camelCase GraphQL fields.",
        observation="Check resolver aliases before changing camelCase queries.",
        why_it_matters="Future agents can avoid frontend-only fixes.",
        scope=MemoryScope.GLOBAL,
        source_type=MemorySourceType.SUBAGENT,
        source_session_id="thread-1",
        source_agent="code-agent",
    )
    review = SynthesisReviewDecision(
        decisions=[
            SynthesisCreateDecision(
                action=SynthesisAction.CREATE,
                rationale="The observation is a reusable debugging rule.",
                summary="Check GraphQL resolver aliases before frontend query fixes.",
                memory_type=MemoryType.SEMANTIC,
                scope=MemoryScope.GLOBAL,
                knowledge=(
                    "Blank camelCase GraphQL fields can come from resolver alias "
                    "mismatches; inspect resolver aliases before changing the "
                    "frontend query."
                ),
                when_to_use="Use when GraphQL camelCase fields render blank.",
                supporting_observation_ids=[observation["observation_id"]],
            )
        ]
    )

    created = apply_synthesis_review_decision(
        memory_dir=memories,
        project_id="P-project",
        review=review,
    )
    target_id = created[0]["knowledge_id"]
    updated = apply_synthesis_review_decision(
        memory_dir=memories,
        project_id="P-project",
        review=SynthesisReviewDecision(
            decisions=[
                SynthesisUpdateDecision(
                    action=SynthesisAction.UPDATE,
                    target_knowledge_id=target_id,
                    rationale="Tighten the reusable wording.",
                    summary="Audit GraphQL resolver aliases before changing queries.",
                    memory_type=MemoryType.SEMANTIC,
                    knowledge=(
                        "When GraphQL camelCase fields are blank, audit backend "
                        "resolver aliases before changing frontend query names."
                    ),
                    when_to_use="Use during GraphQL blank-field debugging.",
                    supporting_observation_ids=[observation["observation_id"]],
                )
            ]
        ),
    )
    archived = apply_synthesis_review_decision(
        memory_dir=memories,
        project_id="P-project",
        review=SynthesisReviewDecision(
            decisions=[
                SynthesisArchiveDecision(
                    action=SynthesisAction.ARCHIVE,
                    target_knowledge_id=target_id,
                    rationale="The record is superseded.",
                    archive_reason="Superseded by a later rule.",
                )
            ]
        ),
    )

    assert created[0]["created"] is True
    assert updated[0]["knowledge_id"] == target_id
    assert updated[0]["created"] is False
    assert archived[0]["knowledge_id"] == target_id
    assert archived[0]["status"] == KnowledgeStatus.ARCHIVED
    metadata, body = _read_memory_document(
        memories / archived[0]["path"].removeprefix("/memories/")
    )
    assert _stable_memory_timestamps(metadata)["status"] == "archived"
    assert _markdown_sections(body)["Archive Reason"] == "Superseded by a later rule."


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
        mode=MemorySearchMode.REGEX,
        query="silent[- ]failure|status",
    )

    assert [hit["memory_id"] for hit in variant_hits] == [relevant["observation_id"]]
    assert [hit["memory_id"] for hit in focused_hits] == [relevant["observation_id"]]


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
        mode=MemorySearchMode.REGEX,
    )

    assert [hit["memory_id"] for hit in hits] == [relevant["observation_id"]]
    assert [hit["memory_id"] for hit in regex_hits] == [relevant["observation_id"]]


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

    assert hits[0]["memory_id"] == relevant["observation_id"]
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
    assert "runtime" in tool.get_input_schema().model_fields  # type: ignore[possibly-missing-attribute]
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


def test_created_observation_ids_from_record_observation_results():
    messages = [
        AIMessage(
            content="",
            name="EvoScientist",
            tool_calls=[
                {
                    "name": "record_observation",
                    "id": "record-1",
                    "args": {},
                },
                {
                    "name": "execute",
                    "id": "exec-1",
                    "args": {},
                },
            ],
        ),
        ToolMessage(
            json.dumps({"created": True, "observation_id": "O-live"}),
            tool_call_id="record-1",
            name="record_observation",
        ),
        ToolMessage(
            json.dumps({"created": True, "observation_id": "O-ignore"}),
            tool_call_id="exec-1",
            name="execute",
        ),
        ToolMessage(
            json.dumps({"created": False, "observation_id": "O-duplicate"}),
            tool_call_id="record-1",
            name="record_observation",
        ),
    ]

    assert memory_lifecycle._created_observation_ids_from_messages(messages) == (
        "O-live",
    )


def test_turn_lifecycle_seed_ids_ignore_prior_turn_observations(tmp_path):
    middleware = memory_lifecycle.EvoMemoryLifecycleMiddleware(
        memory_dir=tmp_path / "memories",
        workspace_dir=tmp_path / "workspace",
        project_id="P-project",
        role=memory_lifecycle.MemoryLifecycleRole.TURN,
        source_agent="EvoScientist",
        launch_memory_worker=False,
        launch_synthesis_worker=True,
    )
    messages = [
        HumanMessage("old turn"),
        AIMessage(
            content="",
            name="EvoScientist",
            tool_calls=[
                {
                    "name": "record_observation",
                    "id": "old-record",
                    "args": {},
                }
            ],
        ),
        ToolMessage(
            json.dumps({"created": True, "observation_id": "O-old"}),
            tool_call_id="old-record",
            name="record_observation",
        ),
        AIMessage("old done", name="EvoScientist"),
        HumanMessage("current turn"),
        AIMessage("current done", name="EvoScientist"),
    ]

    assert middleware._seed_observation_ids(messages) == ()


def test_turn_lifecycle_launches_worker_with_latest_turn_only(
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
        launch_memory_worker=True,
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
    assert calls[0]["seed_observation_ids"] == ()


def test_turn_lifecycle_launches_synthesis_for_live_agent_observation(
    tmp_path, monkeypatch, run_async
):
    worker_calls = []
    synthesis_calls = []

    async def fake_worker(**kwargs):
        worker_calls.append(kwargs)

    monkeypatch.setattr(
        memory_lifecycle,
        "_alaunch_memory_worker",
        fake_worker,
    )
    monkeypatch.setattr(
        memory_lifecycle,
        "_launch_synthesis_worker",
        lambda **kwargs: synthesis_calls.append(kwargs),
    )
    middleware = memory_lifecycle.EvoMemoryLifecycleMiddleware(
        memory_dir=tmp_path / "memories",
        workspace_dir=tmp_path / "workspace",
        project_id="P-project",
        role=memory_lifecycle.MemoryLifecycleRole.TURN,
        source_agent="EvoScientist",
        launch_memory_worker=False,
        launch_synthesis_worker=True,
    )

    async def run():
        state: AgentState[object] = {
            "messages": [
                HumanMessage("remember this"),
                AIMessage(
                    content="",
                    name="EvoScientist",
                    tool_calls=[
                        {
                            "name": "record_observation",
                            "id": "record-1",
                            "args": {},
                        }
                    ],
                ),
                ToolMessage(
                    json.dumps({"created": True, "observation_id": "O-live"}),
                    tool_call_id="record-1",
                    name="record_observation",
                ),
                AIMessage("done", name="EvoScientist"),
            ]
        }
        await middleware.aafter_agent(state, _runtime("thread-1"))
        await asyncio.sleep(0)

    run_async(run())

    assert worker_calls == []
    assert synthesis_calls == [
        {
            "memory_dir": tmp_path / "memories",
            "project_id": "P-project",
            "seed_observation_ids": ("O-live",),
            "trigger": "turn_agent",
        }
    ]


def test_turn_lifecycle_defers_live_observation_synthesis_to_memory_worker(
    tmp_path, monkeypatch, run_async
):
    worker_calls = []
    synthesis_calls = []

    async def fake_worker(**kwargs):
        worker_calls.append(kwargs)

    monkeypatch.setattr(
        memory_lifecycle,
        "_alaunch_memory_worker",
        fake_worker,
    )
    monkeypatch.setattr(
        memory_lifecycle,
        "_launch_synthesis_worker",
        lambda **kwargs: synthesis_calls.append(kwargs),
    )
    middleware = memory_lifecycle.EvoMemoryLifecycleMiddleware(
        memory_dir=tmp_path / "memories",
        workspace_dir=tmp_path / "workspace",
        project_id="P-project",
        role=memory_lifecycle.MemoryLifecycleRole.TURN,
        source_agent="EvoScientist",
        launch_memory_worker=True,
        launch_synthesis_worker=True,
    )

    async def run():
        state: AgentState[object] = {
            "messages": [
                HumanMessage("remember this"),
                AIMessage(
                    content="",
                    name="EvoScientist",
                    tool_calls=[
                        {
                            "name": "record_observation",
                            "id": "record-1",
                            "args": {},
                        }
                    ],
                ),
                ToolMessage(
                    json.dumps({"created": True, "observation_id": "O-live"}),
                    tool_call_id="record-1",
                    name="record_observation",
                ),
                AIMessage("done", name="EvoScientist"),
            ]
        }
        await middleware.aafter_agent(state, _runtime("thread-1"))
        await asyncio.sleep(0)

    run_async(run())

    assert len(worker_calls) == 1
    assert worker_calls[0]["seed_observation_ids"] == ("O-live",)
    assert synthesis_calls == []


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


def test_memory_worker_run_payload_uses_graph_id_and_source_metadata_only():
    trajectory: list[memory_lifecycle.CompactMessage] = [
        {"role": "human", "content": "hi"}
    ]

    kwargs = memory_lifecycle._memory_worker_run_kwargs(
        role=memory_lifecycle.MemoryLifecycleRole.SUBAGENT,
        project_id="P-project",
        source_agent="writing-agent",
        session_id="thread-1",
        trajectory=trajectory,
    )

    assert kwargs["assistant_id"] == memory_lifecycle.SUBAGENT_MEMORY_WORKER_GRAPH_ID
    assert kwargs["metadata"] == {
        "agent_name": "EvoScientist",
        "run_kind": "evomemory_subagent_worker",
        "source_session_id": "thread-1",
        "source_agent": "writing-agent",
        "project_id": "P-project",
        "trajectory_digest": memory_lifecycle._trajectory_digest(trajectory),
    }
    configurable = kwargs["config"]["configurable"]
    assert configurable["thread_id"] == memory_lifecycle._worker_thread_id(
        role=memory_lifecycle.MemoryLifecycleRole.SUBAGENT,
        session_id="thread-1",
        source_agent="writing-agent",
        trajectory=trajectory,
    )
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


@pytest.mark.parametrize(
    ("writer", "role", "expected_tools"),
    [
        (
            MemoryObservationWriter.AGENT,
            memory_lifecycle.MemoryLifecycleRole.SUBAGENT,
            ["search_memory", "read_memory"],
        ),
        (
            MemoryObservationWriter.WORKER,
            memory_lifecycle.MemoryLifecycleRole.SUBAGENT,
            ["search_memory", "read_memory", "record_observation"],
        ),
        (
            MemoryObservationWriter.WORKER,
            memory_lifecycle.MemoryLifecycleRole.TURN,
            ["search_memory", "read_memory", "record_observation"],
        ),
        (
            MemoryObservationWriter.ALL,
            memory_lifecycle.MemoryLifecycleRole.SUBAGENT,
            ["search_memory", "read_memory", "record_observation"],
        ),
        (
            MemoryObservationWriter.ALL,
            memory_lifecycle.MemoryLifecycleRole.TURN,
            ["search_memory", "read_memory", "record_observation"],
        ),
    ],
)
def test_memory_worker_observation_writer_modes(
    tmp_path,
    writer,
    role,
    expected_tools,
):
    middleware = memory_lifecycle._memory_worker_middleware(
        memory_dir=tmp_path / "memories",
        workspace_dir=tmp_path / "workspace",
        role=role,
        observation_writer=writer,
    )

    assert [tool.name for tool in middleware[0].tools] == expected_tools


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
    monkeypatch.setattr(
        memory_lifecycle, "_memory_worker_thread_cleanup_enabled", lambda: True
    )

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


def test_sync_memory_worker_watcher_preserves_thread_when_cleanup_disabled(
    tmp_path, monkeypatch
):
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
    monkeypatch.setattr(
        memory_lifecycle, "_memory_worker_thread_cleanup_enabled", lambda: False
    )

    try:
        memory_lifecycle._watch_memory_worker_run_sync(
            url="http://x",
            thread_id="worker-thread",
            run_id="run-1",
        )
        assert deleted == []
        assert worker_activity.memory_worker_status().is_running is False
    finally:
        worker_activity.reset_memory_worker_status_for_tests()


def test_sync_memory_worker_watcher_launches_synthesis_after_terminal_status(
    tmp_path, monkeypatch
):
    worker_activity.reset_memory_worker_status_for_tests()
    memory_dir = tmp_path / "memories"
    worker_activity.mark_memory_worker_started(
        thread_id="worker-thread",
        run_id="run-1",
        memory_dir=memory_dir,
    )
    observation = record_observation_file(
        memory_dir=memory_dir,
        project_id="P-project",
        memory_type=MemoryType.SEMANTIC,
        summary="Worker output seeds synthesis.",
        observation="The completed worker wrote this observation.",
        why_it_matters="Synthesis should inspect observations from the worker delta.",
        scope=MemoryScope.PROJECT,
        source_type=MemorySourceType.TURN,
        source_session_id="thread-1",
        source_agent="EvoScientist",
    )
    calls = []

    class _Runs:
        def get(self, **_kwargs):
            return {"status": "success"}

    class _Threads:
        def delete(self, _thread_id):
            pass

    monkeypatch.setattr(
        "langgraph_sdk.get_sync_client",
        lambda **_kwargs: SimpleNamespace(runs=_Runs(), threads=_Threads()),
    )
    monkeypatch.setattr(memory_lifecycle, "_MEMORY_WORKER_POLL_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(
        memory_lifecycle,
        "_launch_synthesis_worker",
        lambda **kwargs: calls.append(kwargs),
    )
    synthesis_after = memory_lifecycle.WorkerSynthesisLaunchArgs(
        memory_dir=memory_dir,
        project_id="P-project",
        trigger="turn_memory_worker",
    )

    try:
        memory_lifecycle._watch_memory_worker_run_sync(
            url="http://x",
            thread_id="worker-thread",
            run_id="run-1",
            synthesis_after=synthesis_after,
        )
        assert calls == [
            {
                "memory_dir": memory_dir,
                "project_id": "P-project",
                "seed_observation_ids": (observation["observation_id"],),
                "trigger": "turn_memory_worker",
            }
        ]
    finally:
        worker_activity.reset_memory_worker_status_for_tests()


def test_sync_memory_worker_watcher_skips_synthesis_without_worker_observations(
    tmp_path, monkeypatch
):
    worker_activity.reset_memory_worker_status_for_tests()
    memory_dir = tmp_path / "memories"
    worker_activity.mark_memory_worker_started(
        thread_id="worker-thread",
        run_id="run-1",
        memory_dir=memory_dir,
    )
    calls = []

    class _Runs:
        def get(self, **_kwargs):
            return {"status": "success"}

    class _Threads:
        def delete(self, _thread_id):
            pass

    monkeypatch.setattr(
        "langgraph_sdk.get_sync_client",
        lambda **_kwargs: SimpleNamespace(runs=_Runs(), threads=_Threads()),
    )
    monkeypatch.setattr(memory_lifecycle, "_MEMORY_WORKER_POLL_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(
        memory_lifecycle,
        "_launch_synthesis_worker",
        lambda **kwargs: calls.append(kwargs),
    )
    synthesis_after = memory_lifecycle.WorkerSynthesisLaunchArgs(
        memory_dir=memory_dir,
        project_id="P-project",
        trigger="turn_memory_worker",
    )

    try:
        memory_lifecycle._watch_memory_worker_run_sync(
            url="http://x",
            thread_id="worker-thread",
            run_id="run-1",
            synthesis_after=synthesis_after,
        )
        assert calls == []
    finally:
        worker_activity.reset_memory_worker_status_for_tests()


def test_sync_memory_worker_watcher_combines_deferred_live_and_worker_observations(
    tmp_path, monkeypatch
):
    worker_activity.reset_memory_worker_status_for_tests()
    memory_dir = tmp_path / "memories"
    worker_activity.mark_memory_worker_started(
        thread_id="worker-thread",
        run_id="run-1",
        memory_dir=memory_dir,
    )
    worker_observation = record_observation_file(
        memory_dir=memory_dir,
        project_id="P-project",
        memory_type=MemoryType.SEMANTIC,
        summary="Worker output also seeds synthesis.",
        observation="The worker wrote an additional related observation.",
        why_it_matters="Live and worker observations should synthesize together.",
        scope=MemoryScope.PROJECT,
        source_type=MemorySourceType.TURN,
        source_session_id="thread-1",
        source_agent="EvoScientist",
    )
    calls = []

    class _Runs:
        def get(self, **_kwargs):
            return {"status": "success"}

    class _Threads:
        def delete(self, _thread_id):
            pass

    monkeypatch.setattr(
        "langgraph_sdk.get_sync_client",
        lambda **_kwargs: SimpleNamespace(runs=_Runs(), threads=_Threads()),
    )
    monkeypatch.setattr(memory_lifecycle, "_MEMORY_WORKER_POLL_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(
        memory_lifecycle,
        "_launch_synthesis_worker",
        lambda **kwargs: calls.append(kwargs),
    )
    synthesis_after = memory_lifecycle.WorkerSynthesisLaunchArgs(
        memory_dir=memory_dir,
        project_id="P-project",
        trigger="turn_memory_worker",
        seed_observation_ids=("O-live", worker_observation["observation_id"]),
    )

    try:
        memory_lifecycle._watch_memory_worker_run_sync(
            url="http://x",
            thread_id="worker-thread",
            run_id="run-1",
            synthesis_after=synthesis_after,
        )
        assert calls == [
            {
                "memory_dir": memory_dir,
                "project_id": "P-project",
                "seed_observation_ids": (
                    "O-live",
                    worker_observation["observation_id"],
                ),
                "trigger": "turn_memory_worker",
            }
        ]
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


def _empty_synthesis_context() -> memory_synthesis.SynthesisContext:
    return {
        "project_id": "P-project",
        "uncovered_observations": [],
        "memory_inventory": {
            "active_knowledge_count": 0,
            "seed_observation_count": 0,
        },
    }


def test_synthesis_run_preserves_thread_when_cleanup_disabled(monkeypatch):
    deleted: list[str] = []

    class _Runs:
        def create(self, **_kwargs):
            return {"run_id": "run-1", "status": "pending"}

        def get(self, **_kwargs):
            return {"status": "error"}

    class _Threads:
        def create(self, **_kwargs):
            return {"thread_id": "synthesis-thread"}

        def delete(self, thread_id):
            deleted.append(thread_id)

    monkeypatch.setattr(
        "langgraph_sdk.get_sync_client",
        lambda **_kwargs: SimpleNamespace(runs=_Runs(), threads=_Threads()),
    )
    monkeypatch.setattr(memory_synthesis, "_SYNTHESIS_POLL_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(
        memory_synthesis, "_memory_worker_thread_cleanup_enabled", lambda: False
    )

    outcome = memory_synthesis._submit_and_watch_synthesis_run(
        url="http://x",
        project_id="P-project",
        context=_empty_synthesis_context(),
        trigger="t",
    )

    assert outcome is memory_synthesis._SynthesisRunOutcome.FAILED
    assert deleted == []


def test_synthesis_runner_releases_context_after_exhausting_retries(
    tmp_path, monkeypatch
):
    worker_activity.reset_memory_worker_status_for_tests()
    memory_dir = tmp_path / "memories"
    worker_activity.mark_synthesis_started(
        project_id="P-project",
        context_digest="digest-1",
        memory_dir=memory_dir,
        before_outputs=worker_activity.snapshot_memory_outputs(memory_dir),
    )

    submits = 0

    class _Runs:
        def create(self, **_kwargs):
            nonlocal submits
            submits += 1
            return {"run_id": "run-1", "status": "pending"}

        def get(self, **_kwargs):
            raise RuntimeError("poll failed")

    class _Threads:
        def create(self, **_kwargs):
            return {"thread_id": "synthesis-thread"}

        def delete(self, thread_id):
            pass

    monkeypatch.setattr(
        "langgraph_sdk.get_sync_client",
        lambda **_kwargs: SimpleNamespace(runs=_Runs(), threads=_Threads()),
    )
    monkeypatch.setattr(memory_synthesis, "_SYNTHESIS_POLL_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(memory_synthesis, "_SYNTHESIS_RETRY_BACKOFF_SECONDS", 0)
    monkeypatch.setattr(memory_synthesis, "_SYNTHESIS_MAX_POLL_FAILURES", 1)

    try:
        # Tracked while the retry loop is in flight.
        assert worker_activity.memory_worker_status().synthesis_running is True
        memory_synthesis._run_synthesis_with_retries(
            url="http://x",
            project_id="P-project",
            context=_empty_synthesis_context(),
            trigger="t",
            active_key=("P-project", "digest-1"),
            max_attempts=2,
        )
        # Retried up to the bound, then fell back to (a): released, no leak.
        assert submits == 2
        status = worker_activity.memory_worker_status()
        assert status.synthesis_running is False
        assert status.knowledge_created == 0
    finally:
        worker_activity.reset_memory_worker_status_for_tests()


def test_synthesis_runner_retries_then_stops_on_success(tmp_path, monkeypatch):
    worker_activity.reset_memory_worker_status_for_tests()
    memory_dir = tmp_path / "memories"
    worker_activity.mark_synthesis_started(
        project_id="P-project",
        context_digest="digest-1",
        memory_dir=memory_dir,
        before_outputs=worker_activity.snapshot_memory_outputs(memory_dir),
    )

    submits = 0

    class _Runs:
        def create(self, **_kwargs):
            nonlocal submits
            submits += 1
            return {"run_id": "run-1", "status": "pending"}

        def get(self, **_kwargs):
            # First run errors out, the retry succeeds.
            return {"status": "error" if submits == 1 else "success"}

    class _Threads:
        def create(self, **_kwargs):
            return {"thread_id": "synthesis-thread"}

        def delete(self, thread_id):
            pass

    monkeypatch.setattr(
        "langgraph_sdk.get_sync_client",
        lambda **_kwargs: SimpleNamespace(runs=_Runs(), threads=_Threads()),
    )
    monkeypatch.setattr(memory_synthesis, "_SYNTHESIS_POLL_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(memory_synthesis, "_SYNTHESIS_RETRY_BACKOFF_SECONDS", 0)

    try:
        memory_synthesis._run_synthesis_with_retries(
            url="http://x",
            project_id="P-project",
            context=_empty_synthesis_context(),
            trigger="t",
            active_key=("P-project", "digest-1"),
            max_attempts=3,
        )
        # Retried once after the error, then stopped on success (no third run).
        assert submits == 2
        assert worker_activity.memory_worker_status().synthesis_running is False
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
        project_id="P-project",
        source_agent="EvoScientist",
        session_id="thread-1",
        trajectory=trajectory,
    )


def test_memory_worker_launch_marks_active_status(tmp_path, monkeypatch):
    worker_activity.reset_memory_worker_status_for_tests()
    monkeypatch.setattr(memory_lifecycle, "_memory_worker_url", lambda: "http://x")
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
    memory_dir = tmp_path / "memories"
    synthesis_after = memory_lifecycle.WorkerSynthesisLaunchArgs(
        memory_dir=memory_dir,
        project_id="P-project",
        trigger="turn_memory_worker",
    )
    monkeypatch.setattr(
        memory_lifecycle,
        "_synthesis_after_worker_args",
        lambda **_kwargs: synthesis_after,
    )

    trajectory: list[memory_lifecycle.CompactMessage] = [
        {"role": "human", "content": "hi"}
    ]

    memory_lifecycle._launch_memory_worker(
        role=memory_lifecycle.MemoryLifecycleRole.TURN,
        memory_dir=memory_dir,
        project_id="P-project",
        source_agent="EvoScientist",
        session_id="thread-1",
        trajectory=trajectory,
    )

    try:
        assert worker_activity.memory_worker_status().is_running is True
        assert spawned == [
            {
                "url": "http://x",
                "thread_id": "worker-thread",
                "run_id": "run-1",
                "synthesis_after": synthesis_after,
            }
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
    synthesis_after = memory_lifecycle.WorkerSynthesisLaunchArgs(
        memory_dir=tmp_path / "memories",
        project_id="P-project",
        trigger="turn_memory_worker",
    )
    monkeypatch.setattr(
        memory_lifecycle,
        "_synthesis_after_worker_args",
        lambda **_kwargs: synthesis_after,
    )

    async def run():
        event_loop_thread = threading.get_ident()
        await memory_lifecycle._alaunch_memory_worker(
            role=memory_lifecycle.MemoryLifecycleRole.TURN,
            memory_dir=tmp_path / "memories",
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
            {
                "url": "http://x",
                "thread_id": "worker-thread",
                "run_id": "run-1",
                "synthesis_after": synthesis_after,
            }
        ]
    finally:
        worker_activity.reset_memory_worker_status_for_tests()


def test_synthesis_worker_launch_spawns_runner_thread(tmp_path, monkeypatch):
    worker_activity.reset_memory_worker_status_for_tests()
    memory_dir = tmp_path / "memories"
    context: memory_synthesis.SynthesisContext = {
        "project_id": "P-project",
        "uncovered_observations": [
            {
                "id": "O-1",
                "path": "/memories/observations/project/P-project/O-1.md",
                "memory_type": MemoryType.SEMANTIC.value,
                "scope": MemoryScope.PROJECT.value,
                "summary": "Durable observation.",
                "snippet": "Durable observation.",
            }
        ],
        "memory_inventory": {
            "active_knowledge_count": 0,
            "seed_observation_count": 1,
        },
    }
    context_digest = memory_synthesis._synthesis_context_digest(context)

    monkeypatch.setattr(memory_synthesis, "_synthesis_worker_url", lambda: "http://x")
    monkeypatch.setattr(
        memory_synthesis,
        "build_synthesis_context",
        lambda **_kwargs: context,
    )
    monkeypatch.setattr(
        memory_synthesis,
        "snapshot_memory_outputs",
        lambda _memory_dir: worker_activity.MemoryOutputSnapshot(
            profile_files={},
            observation_files=frozenset(),
        ),
    )
    monkeypatch.setattr(
        "EvoScientist.langgraph_dev.manager.is_langgraph_dev_running",
        lambda **_kwargs: True,
    )

    spawned = []

    def fake_spawn(**kwargs):
        spawned.append(kwargs)
        assert worker_activity.memory_worker_status().synthesis_running is True
        active_key = kwargs["active_key"]
        memory_synthesis._release_synthesis_context(
            project_id=active_key[0],
            context_digest=active_key[1],
        )

    monkeypatch.setattr(memory_synthesis, "_spawn_synthesis_runner_thread", fake_spawn)

    try:
        memory_synthesis._launch_synthesis_worker(
            memory_dir=memory_dir,
            project_id="P-project",
            seed_observation_ids=("O-1",),
            trigger="turn_memory_worker",
        )
        assert spawned == [
            {
                "url": "http://x",
                "project_id": "P-project",
                "context": context,
                "trigger": "turn_memory_worker",
                "active_key": ("P-project", context_digest),
            }
        ]
        assert worker_activity.memory_worker_status().synthesis_running is False
    finally:
        worker_activity.reset_memory_worker_status_for_tests()


def test_clear_memory_worker_saved_counts_preserves_pending_worker_delta(tmp_path):
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


def test_memory_activity_tracks_active_synthesis_knowledge_create(tmp_path):
    worker_activity.reset_memory_worker_status_for_tests()
    memory_dir = tmp_path / "memories"
    before = worker_activity.snapshot_memory_outputs(memory_dir)
    worker_activity.mark_synthesis_started(
        project_id="P-project",
        context_digest="digest-1",
        memory_dir=memory_dir,
        before_outputs=before,
    )

    try:
        assert worker_activity.memory_worker_status().synthesis_running is True
        _write_knowledge_status_file(memory_dir, body="Created knowledge.")
        observed = worker_activity.memory_worker_observed_outputs()
        assert observed.is_running is True
        assert observed.synthesis_running is True
        assert observed.knowledge_created == 1
        assert worker_activity.memory_worker_status().knowledge_created == 0

        worker_activity.mark_synthesis_finished(
            project_id="P-project",
            context_digest="digest-1",
        )
        status = worker_activity.memory_worker_status()
        assert status.synthesis_running is False
        assert status.knowledge_created == 1
        assert status.knowledge_updated == 0
        assert status.knowledge_archived == 0
    finally:
        worker_activity.reset_memory_worker_status_for_tests()


def test_memory_activity_counts_knowledge_update_and_archive(tmp_path):
    worker_activity.reset_memory_worker_status_for_tests()
    memory_dir = tmp_path / "memories"
    _write_knowledge_status_file(memory_dir, body="Initial knowledge.")

    try:
        before_update = worker_activity.snapshot_memory_outputs(memory_dir)
        worker_activity.mark_memory_worker_started(
            thread_id="thread-1",
            run_id="run-1",
            memory_dir=memory_dir,
            before_outputs=before_update,
        )
        _write_knowledge_status_file(memory_dir, body="Updated knowledge.")
        worker_activity.mark_memory_worker_finished("thread-1", "run-1")
        status = worker_activity.memory_worker_status()
        assert status.knowledge_created == 0
        assert status.knowledge_updated == 1
        assert status.knowledge_archived == 0

        worker_activity.clear_memory_worker_saved_counts()
        before_archive = worker_activity.snapshot_memory_outputs(memory_dir)
        worker_activity.mark_memory_worker_started(
            thread_id="thread-2",
            run_id="run-2",
            memory_dir=memory_dir,
            before_outputs=before_archive,
        )
        _write_knowledge_status_file(
            memory_dir,
            status="archived",
            body="Archived knowledge.",
        )
        worker_activity.mark_memory_worker_finished("thread-2", "run-2")
        status = worker_activity.memory_worker_status()
        assert status.knowledge_created == 0
        assert status.knowledge_updated == 0
        assert status.knowledge_archived == 1
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


def test_one_shot_cli_wait_treats_synthesis_as_running(monkeypatch):
    from EvoScientist.cli import interactive

    printed = []
    observed_calls = 0

    def fake_sleep(_seconds):
        pass

    def fake_observed_outputs():
        nonlocal observed_calls
        observed_calls += 1
        if observed_calls < 3:
            return worker_activity.MemoryWorkerStatusSnapshot(
                is_running=True,
                synthesis_running=True,
            )
        return worker_activity.MemoryWorkerStatusSnapshot(
            is_running=False,
            synthesis_running=False,
            knowledge_created=1,
        )

    monkeypatch.setattr(interactive.time, "sleep", fake_sleep)
    monkeypatch.setattr(interactive.console, "print", lambda text: printed.append(text))
    monkeypatch.setattr(
        worker_activity,
        "memory_worker_observed_outputs",
        fake_observed_outputs,
    )

    interactive._wait_for_memory_workers_before_exit(timeout_seconds=10)

    assert observed_calls == 3
    assert not any("still running" in str(line) for line in printed)


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


def test_clear_memory_worker_saved_counts_does_not_recount_credited_file(tmp_path):
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
