"""Write guardrails of the EvoMemory background workers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from EvoScientist.config import MemoryControls, MemoryObservationWriter
from EvoScientist.memory.agents import memory_worker
from EvoScientist.memory.observations.index import (
    DEFAULT_INLINE_ROWS_PER_SESSION,
    build_observation_index_context,
    build_observation_linker_index_context,
)
from EvoScientist.memory.observations.store import (
    ObservationFrontmatter,
    ObservationSourceFrontmatter,
    list_observation_documents,
    read_observation_document,
    record_observation_file,
    write_observation_document,
)
from EvoScientist.memory.types import MemoryScope, MemorySourceType, MemoryType
from EvoScientist.paths import Workspace
from tests.fakes.memory import (
    record_observation_payload,
    tool_by_name,
    tool_runtime,
)

SESSION = "01a1test-0000-7000-8000-000000000001"
LOGGER = "EvoScientist.memory.agents.memory_worker"
BUDGET = memory_worker.MEMORY_WORKER_MODEL_CALL_BUDGET
SYNC_AND_ASYNC = pytest.mark.parametrize(
    "use_async", [False, True], ids=["sync", "async"]
)


def _text(content: Any) -> str:
    """Flatten message content (a string or content blocks) to its text."""
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


class _ScriptedModel(BaseChatModel):
    """Replays one AIMessage per call; the last one repeats."""

    scripts: list[AIMessage]
    model_name: str | None = None
    bound: list[list[str]] = Field(default_factory=list)
    systems: list[str] = Field(default_factory=list)
    lasts: list[str] = Field(default_factory=list)
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> _ScriptedModel:
        self.bound.append([getattr(t, "name", None) or t.get("name") for t in tools])
        return self

    def _generate(
        self,
        messages: Any,
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        index = min(self.calls, len(self.scripts) - 1)
        n = self.calls
        self.calls += 1
        first = messages[0] if messages else None
        self.systems.append(_text(getattr(first, "content", "")))
        self.lasts.append(_text(getattr(messages[-1], "content", "")))
        script = self.scripts[index]
        # Fresh ids per call: the messages reducer replaces a repeated id.
        message = script.model_copy(
            update={
                "id": f"ai-{n}",
                "tool_calls": [
                    {**call, "id": f"{call['id']}-{n}"} for call in script.tool_calls
                ],
            }
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def _record_call(
    call_id: str, summary: str, *, scope: str = "project", body: str | None = None
) -> dict[str, Any]:
    return {
        "name": "record_observation",
        "id": call_id,
        "args": {
            "memory_type": "procedural",
            "summary": summary,
            "observation": body or f"{summary} body",
            "why_it_matters": "future runs repeat it",
            "scope": scope,
        },
    }


def _ai(*tool_calls: dict[str, Any], text: str = "") -> AIMessage:
    return AIMessage(content=text, tool_calls=list(tool_calls))


def _finish() -> AIMessage:
    return _ai(
        {"name": "SubagentMemoryDecision", "id": "final", "args": {"summary": "done"}}
    )


def _search() -> AIMessage:
    return _ai({"name": "search_observations", "id": "s", "args": {"query": "any"}})


def _worker_config() -> dict[str, Any]:
    return {
        "configurable": {
            "thread_id": "worker-thread",
            "evomemory_source_session_id": SESSION,
            "evomemory_source_agent": "general-purpose",
            "evomemory_project_id": "P-test",
            "evomemory_trajectory_digest": "digest",
        }
    }


def _build(
    tmp_path: Path,
    monkeypatch,
    scripts: list[AIMessage],
    *,
    source_type: MemorySourceType = MemorySourceType.SUBAGENT,
) -> tuple[Any, _ScriptedModel]:
    model = _ScriptedModel(scripts=scripts, bound=[])
    monkeypatch.setattr(
        "EvoScientist.EvoScientist._ensure_auxiliary_chat_model", lambda: model
    )
    graph = memory_worker.build_memory_worker_graph(
        source_type,
        workspace=Workspace(tmp_path / "workspace"),
        memory_dir=tmp_path / "memories",
    )
    return graph, model


async def _run(graph: Any, *, use_async: bool) -> dict[str, Any]:
    payload = {"messages": [HumanMessage("Review the run.")]}
    if use_async:
        return await graph.ainvoke(payload, config=_worker_config())
    return graph.invoke(payload, config=_worker_config())


def _run_worker(
    tmp_path: Path,
    monkeypatch,
    scripts: list[AIMessage],
    *,
    source_type: MemorySourceType = MemorySourceType.SUBAGENT,
) -> tuple[dict[str, Any], _ScriptedModel]:
    graph, model = _build(tmp_path, monkeypatch, scripts, source_type=source_type)
    result = graph.invoke(
        {"messages": [HumanMessage("Review the run.")]}, config=_worker_config()
    )
    return result, model


def _observation_files(tmp_path: Path) -> list[Path]:
    return sorted((tmp_path / "memories" / "observations").rglob("O-*.md"))


def _execution_files(tmp_path: Path) -> list[Path]:
    return list((tmp_path / "memories" / "executions").rglob("*.md"))


def _record_messages(result: dict[str, Any]) -> list[ToolMessage]:
    return [
        m
        for m in result["messages"]
        if isinstance(m, ToolMessage) and m.name == "record_observation"
    ]


def _record_results(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [json.loads(m.content) for m in _record_messages(result)]


def _warnings(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING and r.name == LOGGER
    ]


@SYNC_AND_ASYNC
async def test_sequential_writes_stop_at_the_limit(tmp_path, monkeypatch, use_async):
    scripts = [_ai(_record_call(f"c{i}", f"lesson {i}")) for i in range(8)]
    graph, _ = _build(tmp_path, monkeypatch, [*scripts, _finish()])
    result = await _run(graph, use_async=use_async)

    assert len(_observation_files(tmp_path)) == 3
    results = _record_results(result)
    assert [r["created"] for r in results] == [True] * 3 + [False] * 5
    assert "observation budget is used up" in results[3]["note"]
    # The execution summary is still written after the cap.
    assert _execution_files(tmp_path)


@SYNC_AND_ASYNC
async def test_parallel_writes_keep_the_first_three(tmp_path, monkeypatch, use_async):
    calls = [_record_call(f"p{i}", f"parallel lesson {i}") for i in range(6)]
    graph, _ = _build(tmp_path, monkeypatch, [_ai(*calls), _finish()])
    result = await _run(graph, use_async=use_async)

    assert len(_observation_files(tmp_path)) == 3
    created = {
        m.tool_call_id.rsplit("-", 1)[0]: json.loads(m.content)["created"]
        for m in _record_messages(result)
    }
    assert created == {
        "p0": True,
        "p1": True,
        "p2": True,
        "p3": False,
        "p4": False,
        "p5": False,
    }


@SYNC_AND_ASYNC
async def test_parallel_same_summary_writes_land_once(tmp_path, monkeypatch, use_async):
    calls = [
        _record_call("a", "one lesson", scope="project", body="project body"),
        _record_call("b", "one lesson", scope="global", body="global body"),
    ]
    graph, _ = _build(tmp_path, monkeypatch, [_ai(*calls), _finish()])
    result = await _run(graph, use_async=use_async)

    assert len(_observation_files(tmp_path)) == 1
    results = _record_results(result)
    assert sorted(r["created"] for r in results) == [False, True]
    rejected = next(r for r in results if not r["created"])
    assert "this summary already exists" in rejected["note"]


def test_record_tool_leaves_the_tool_list_once_the_limit_is_reached(
    tmp_path, monkeypatch
):
    scripts = [_ai(_record_call(f"c{i}", f"lesson {i}")) for i in range(3)]
    _, model = _run_worker(tmp_path, monkeypatch, [*scripts, _search(), _finish()])

    assert "record_observation" in model.bound[0]
    assert "record_observation" in model.bound[2]
    assert "record_observation" not in model.bound[3]
    assert "search_observations" in model.bound[3]


@SYNC_AND_ASYNC
async def test_hard_stop_ends_a_run_that_never_wraps_up(
    tmp_path, monkeypatch, caplog, use_async
):
    # A budgeted run spends about 64 of the 100 supersteps; another hook node
    # per cycle would add 20, so this test also guards that margin.
    graph, model = _build(tmp_path, monkeypatch, [_search()])
    with caplog.at_level(logging.WARNING):
        result = await _run(graph, use_async=use_async)

    # The wrap-up call is retried once, so 21 API calls for 20 AI messages.
    assert model.calls == BUDGET + 1
    assert sum(isinstance(m, AIMessage) for m in result["messages"]) == BUDGET
    assert _warnings(caplog) == [
        f"evomemory-subagent-worker is making the last of its {BUDGET} model calls "
        f"with 0 observations written (source general-purpose, session {SESSION})",
        f"evomemory-subagent-worker stopped at the budget of its {BUDGET} model "
        f"calls with 0 observations written (source general-purpose, session "
        f"{SESSION})",
        "Subagent memory worker returned no structured summary",
    ]


def test_last_budgeted_call_offers_only_the_summary_tool(tmp_path, monkeypatch, caplog):
    scripts = [_search()] * (BUDGET - 1) + [_finish()]
    with caplog.at_level(logging.WARNING):
        result, model = _run_worker(tmp_path, monkeypatch, scripts)

    assert model.calls == BUDGET
    assert "search_observations" in model.bound[BUDGET - 2]
    assert model.bound[BUDGET - 1] == ["SubagentMemoryDecision"]
    assert "your last model call" not in model.systems[BUDGET - 2]
    assert "your last model call" in model.systems[BUDGET - 1]
    assert result["structured_response"] is not None
    assert _execution_files(tmp_path)
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert warnings[0].startswith(
        f"evomemory-subagent-worker is making the last of its {BUDGET} model calls"
    )


def _run_with_model(tmp_path, monkeypatch, model):
    monkeypatch.setattr(
        "EvoScientist.EvoScientist._ensure_auxiliary_chat_model", lambda: model
    )
    graph = memory_worker.build_memory_worker_graph(
        MemorySourceType.SUBAGENT,
        workspace=Workspace(tmp_path / "workspace"),
        memory_dir=tmp_path / "memories",
    )
    return graph.invoke(
        {"messages": [HumanMessage("Review the run.")]}, config=_worker_config()
    )


def test_wrap_up_is_skipped_on_a_provider_strategy_run(tmp_path, monkeypatch, caplog):
    model = _ScriptedModel(
        scripts=[_search()], bound=[], profile={"structured_output": True}
    )
    with caplog.at_level(logging.WARNING):
        result = _run_with_model(tmp_path, monkeypatch, model)

    # Such a run never forced a tool choice, so its last call keeps every
    # tool and the hard stop ends it.
    assert model.calls == BUDGET
    assert len(model.bound) == BUDGET
    assert "search_observations" in model.bound[BUDGET - 1]
    assert result.get("structured_response") is None
    assert _warnings(caplog) == [
        f"evomemory-subagent-worker stopped at the budget of its {BUDGET} model "
        f"calls with 0 observations written (source general-purpose, session "
        f"{SESSION})",
        "Subagent memory worker returned no structured summary",
    ]


@pytest.mark.parametrize(
    ("profile", "model_name", "tool_strategy"),
    [
        (None, None, True),
        (None, "anthropic/claude-haiku-5.5", True),
        ({"structured_output": False}, "claude-haiku-5-5", True),
        ({"structured_output": True}, "claude-fable-5-1", False),
        (None, "openai/gpt-5.5", False),
        # A context-window patch leaves a profile without structured_output;
        # the name list still decides, as it does in langchain.
        ({"max_input_tokens": 400000}, "openai/gpt-6-sol", False),
        ({"max_input_tokens": 200000}, "claude-fable-5", False),
    ],
)
def test_strategy_replica_agrees_with_langchain(profile, model_name, tool_strategy):
    from types import SimpleNamespace

    from langchain.agents.factory import _supports_provider_strategy

    model = _ScriptedModel(scripts=[_finish()], profile=profile, model_name=model_name)
    request = SimpleNamespace(
        model=model, response_format=memory_worker.SubagentMemoryDecision
    )

    assert memory_worker._tool_strategy_run(request) is tool_strategy
    assert _supports_provider_strategy(model) is (not tool_strategy)


@SYNC_AND_ASYNC
async def test_wrap_up_call_gets_one_retry_when_it_names_a_tool(
    tmp_path, monkeypatch, caplog, use_async
):
    stray = _ai(_record_call("stray", "late lesson"))
    scripts = [_search()] * (BUDGET - 1) + [stray, _finish()]
    graph, model = _build(tmp_path, monkeypatch, scripts)
    with caplog.at_level(logging.WARNING):
        result = await _run(graph, use_async=use_async)

    # 21 API calls, 20 AI messages: the retry replaces the stray reply.
    assert model.calls == BUDGET + 1
    assert sum(isinstance(m, AIMessage) for m in result["messages"]) == BUDGET
    assert model.bound[BUDGET - 1] == ["SubagentMemoryDecision"]
    assert model.bound[BUDGET] == ["SubagentMemoryDecision"]
    assert model.lasts[BUDGET] == memory_worker._WRAP_UP_RETRY
    assert result["structured_response"] is not None
    assert _execution_files(tmp_path)
    assert _observation_files(tmp_path) == []
    assert len(_warnings(caplog)) == 1


def test_turn_worker_keeps_its_tools_and_hard_stops(tmp_path, monkeypatch, caplog):
    with caplog.at_level(logging.WARNING):
        result, model = _run_worker(
            tmp_path, monkeypatch, [_search()], source_type=MemorySourceType.TURN
        )

    assert model.calls == BUDGET
    assert len(model.bound) == BUDGET
    assert "search_observations" in model.bound[BUDGET - 1]
    assert sum(isinstance(m, AIMessage) for m in result["messages"]) == BUDGET
    assert _warnings(caplog) == [
        f"evomemory-turn-worker stopped at the budget of its {BUDGET} model calls "
        f"with 0 observations written (source general-purpose, session {SESSION})"
    ]


def test_turn_worker_writes_are_capped_too(tmp_path, monkeypatch):
    scripts = [_ai(_record_call(f"t{i}", f"turn lesson {i}")) for i in range(5)]
    result, _ = _run_worker(
        tmp_path,
        monkeypatch,
        [*scripts, _ai(text="Nothing else to record.")],
        source_type=MemorySourceType.TURN,
    )

    assert len(_observation_files(tmp_path)) == 3
    assert [r["created"] for r in _record_results(result)] == [True] * 3 + [False] * 2


def test_profile_only_worker_still_has_the_budget(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(
        MemoryControls,
        "from_config",
        classmethod(
            lambda cls, config: MemoryControls(
                profile_enabled=True,
                observations_enabled=True,
                observation_writer=MemoryObservationWriter.AGENT,
                workers_enabled=True,
            )
        ),
    )
    with caplog.at_level(logging.WARNING):
        result, model = _run_worker(
            tmp_path, monkeypatch, [_search()], source_type=MemorySourceType.TURN
        )

    assert all("record_observation" not in names for names in model.bound)
    assert sum(isinstance(m, AIMessage) for m in result["messages"]) == BUDGET
    assert _warnings(caplog) == [
        f"evomemory-turn-worker stopped at the budget of its {BUDGET} model calls "
        f"with 0 observations written (source general-purpose, session {SESSION})"
    ]


def test_replayed_flood_lands_no_placeholder(tmp_path, monkeypatch):
    """A flood: distinct entries, one restated under another body, then
    self-referential placeholders that share one summary."""
    scripts = [
        _ai(_record_call("a", "panel review contribution seat")),
        _ai(
            _record_call(
                "a2", "panel review contribution seat", scope="global", body="reworded"
            )
        ),
        _ai(_record_call("b", "panel review method seat", scope="global")),
        _ai(_record_call("c", "panel review testing seat")),
        _ai(_record_call("d", "panel review results seat")),
    ]
    for i in range(10):
        scripts.append(
            _ai(
                _record_call(
                    f"ph{i}",
                    "panel review method seat duplicate placeholder: ignore",
                    scope="global" if i % 2 else "project",
                    body=f"Duplicate placeholder {i}. No unique content.",
                )
            )
        )
    result, _ = _run_worker(tmp_path, monkeypatch, [*scripts, _finish()])

    files = _observation_files(tmp_path)
    assert len(files) == 3
    assert not [f for f in files if "placeholder" in f.read_text(encoding="utf-8")]
    results = _record_results(result)
    assert [r["created"] for r in results[:4]] == [True, False, True, True]
    assert "this summary already exists" in results[1]["note"]


def test_record_call_limit_counts_rejected_attempts(tmp_path, monkeypatch):
    same = [
        _record_call(f"r{i}", "one lesson", body="the same lesson") for i in range(8)
    ]
    scripts = [_ai(call) for call in same] + [_finish()]
    result, model = _run_worker(tmp_path, monkeypatch, scripts)

    results = _record_results(result)
    assert len(_observation_files(tmp_path)) == 1
    assert [r["created"] for r in results] == [True] + [False] * 7
    assert all("identical observation" in r["note"] for r in results[1:6])
    assert all("budget is used up" in r["note"] for r in results[6:])
    assert "record_observation" in model.bound[5]
    assert "record_observation" not in model.bound[6]


def test_record_call_limit_counts_error_results_too(tmp_path, monkeypatch):
    bad = {**_record_call("bad", "lesson"), "args": {"memory_type": "nonsense"}}
    scripts = [_ai(bad)] * 6 + [_ai(_record_call("ok", "a real lesson")), _finish()]
    result, _ = _run_worker(tmp_path, monkeypatch, scripts)

    record_messages = _record_messages(result)
    assert len(record_messages) == 7
    assert "budget is used up" in record_messages[6].content
    assert _observation_files(tmp_path) == []


def test_worker_prompt_states_the_limit_and_the_ordering():
    prompt = memory_worker._memory_worker_system_prompt(
        MemorySourceType.SUBAGENT,
        enable_profile_memory=True,
        enable_observation_tool=True,
    )
    assert "Record at most 3 observations in this run" in prompt
    assert "most runs need none or one" in prompt
    assert "in order of importance" in prompt


def _store_write(
    tmp_path: Path,
    *,
    summary: str,
    body: str,
    scope: MemoryScope = MemoryScope.PROJECT,
):
    return record_observation_file(
        memory_dir=tmp_path / "memories",
        project_id="P-test",
        memory_type=MemoryType.PROCEDURAL,
        summary=summary,
        observation=body,
        why_it_matters="it repeats",
        scope=scope,
        source_type=MemorySourceType.SUBAGENT,
        source_session_id=SESSION,
        source_agent="general-purpose",
    )


def test_store_rejects_a_second_observation_with_the_same_summary(tmp_path):
    first = _store_write(tmp_path, summary="grep widens to the whole tree", body="one")
    second = _store_write(
        tmp_path, summary="  Grep widens  to the whole TREE ", body="another body"
    )
    other_scope = _store_write(
        tmp_path,
        summary="grep widens to the whole tree",
        body="global body",
        scope=MemoryScope.GLOBAL,
    )

    assert first["created"] is True
    assert second["created"] is False
    assert second["observation_id"] == first["observation_id"]
    assert "do not retry" in second["note"]
    assert other_scope["created"] is False
    assert other_scope["observation_id"] == first["observation_id"]
    assert len(_observation_files(tmp_path)) == 1


def test_store_refuses_placeholder_fields(tmp_path):
    """A field made only of filler never lands as an observation."""
    stub_body = _store_write(
        tmp_path,
        summary="chunk-3 picks were edge-relevant",
        body="Placeholder — see below.",
    )
    stub_summary = _store_write(
        tmp_path, summary="placeholder: ignore", body="a real body"
    )
    stub_why = record_observation_file(
        memory_dir=tmp_path / "memories",
        project_id="P-test",
        memory_type=MemoryType.PROCEDURAL,
        summary="a real lesson",
        observation="a real body",
        why_it_matters="TBD",
        scope=MemoryScope.PROJECT,
        source_type=MemorySourceType.SUBAGENT,
        source_session_id=SESSION,
        source_agent="general-purpose",
    )
    dashes = _store_write(tmp_path, summary="another lesson", body="---")

    assert [r["created"] for r in (stub_body, stub_summary, stub_why, dashes)] == [
        False
    ] * 4
    assert "The `observation` field is a placeholder" in stub_body["note"]
    assert "The `summary` field is a placeholder" in stub_summary["note"]
    assert "The `why_it_matters` field is a placeholder" in stub_why["note"]
    assert stub_why["note"] == (
        "The `why_it_matters` field is a placeholder; nothing was written. "
        "Record the observation with its real content, or leave it out."
    )
    assert "nothing was written" in dashes["note"]
    assert "observation_id" not in stub_body
    assert "path" not in stub_body
    assert _observation_files(tmp_path) == []


@pytest.mark.parametrize(
    "text",
    [
        "Placeholder — see below.",
        "Placeholder observation.",
        "Placeholder body",
        "placeholder text",
        "Placeholder — see summary above.",
        "placeholder: ignore",
        "placeholder_1",
        "placeholder for now",
        "Details TBD.",
        "TODO: add details",
        "Not applicable",
        "Same as above.",
        "To be determined",
        "No content.",
        "N/A",
        "___",
        "...",
    ],
)
def test_is_stub_matches_model_filler(text):
    from EvoScientist.memory.observations.store import _is_stub

    assert _is_stub(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "Placeholder rows break the parser.",
        "No placeholders in the body.",
        "No TODO.",
        "Body text.",
        "None → null",
        "a in [None]",
        "42",
        "占位行会让解析器出错。",
        "TBD 字段要先填默认值",
        "n_a is the column name",
        "v0.3.7 drops the fallback",
        "Same as the 10-09 run: 31 entries",
        "No content-type header on the retry",
    ],
)
def test_is_stub_keeps_real_text(text):
    from EvoScientist.memory.observations.store import _is_stub

    assert _is_stub(text) is False


def test_store_keeps_real_text_that_mentions_placeholders(tmp_path):
    english = _store_write(
        tmp_path,
        summary="placeholder rows break the parser",
        body="Placeholder rows break the parser.",
    )
    chinese = _store_write(
        tmp_path, summary="占位行会让解析器出错", body="占位行会让解析器出错。"
    )

    assert english["created"] is True
    assert chinese["created"] is True
    assert len(_observation_files(tmp_path)) == 2


def test_worker_stub_attempt_is_refused_and_counted(tmp_path, monkeypatch):
    """Six refused stubs use up the run's record calls like any other attempt."""
    stubs = [
        _ai(_record_call(f"s{i}", f"lesson {i}", body="placeholder")) for i in range(6)
    ]
    scripts = [
        *stubs,
        _ai(_record_call("ok", "a real lesson", body="the real body")),
        _finish(),
    ]
    result, model = _run_worker(tmp_path, monkeypatch, scripts)

    results = _record_results(result)
    assert [r["created"] for r in results] == [False] * 7
    assert all("placeholder" in r["note"] for r in results[:6])
    assert "budget is used up" in results[6]["note"]
    assert "record_observation" not in model.bound[6]
    assert _observation_files(tmp_path) == []


def test_observation_files_are_replaced_in_one_step(tmp_path, monkeypatch):
    """A rewrite (the linker) or a new record never leaves a partial file
    where the summary scan, the index or a search could read it."""
    _store_write(tmp_path, summary="lesson", body="one")
    [path] = _observation_files(tmp_path)
    before = path.read_text(encoding="utf-8")
    metadata, body = read_observation_document(path)

    writes: list[tuple[str, str | None]] = []
    original = Path.write_text

    def spy(self: Path, data: str, *args: Any, **kwargs: Any) -> int:
        target = path.read_text(encoding="utf-8") if path.exists() else None
        writes.append((self.name, target))
        return original(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", spy)
    write_observation_document(path, metadata=metadata, body=body.replace("one", "two"))
    _store_write(tmp_path, summary="another lesson", body="three")

    assert [name.endswith(".tmp") for name, _ in writes] == [True, True]
    assert writes[0][1] == before
    assert "two" in path.read_text(encoding="utf-8")
    assert len(_observation_files(tmp_path)) == 2
    leftovers = [p for p in path.parent.iterdir() if p.suffix != ".md"]
    assert leftovers == []


def test_observation_rewrite_drops_the_cached_parse(tmp_path, monkeypatch):
    """Even where stat cannot tell the files apart, the writing process must
    not serve the old parse."""
    from EvoScientist.memory.observations import store

    _store_write(tmp_path, summary="lesson", body="one")
    [path] = _observation_files(tmp_path)
    memory_dir = tmp_path / "memories"
    [document] = list_observation_documents(memory_dir=memory_dir, project_id="P-test")
    assert document.summary == "lesson"
    metadata, body = read_observation_document(path)
    metadata.summary = "lesson, restated"
    frozen = path.stat()
    original_stat = Path.stat
    monkeypatch.setattr(
        store.Path,
        "stat",
        lambda self, *a, **k: frozen if self == path else original_stat(self, *a, **k),
    )

    write_observation_document(path, metadata=metadata, body=body)
    [document] = list_observation_documents(memory_dir=memory_dir, project_id="P-test")

    assert document.summary == "lesson, restated"


def test_observation_rewrite_outruns_a_parse_in_flight(tmp_path, monkeypatch):
    """A reader that parsed the old file before a rewrite must not put that
    parse into the cache afterwards."""
    from EvoScientist.memory.observations import store

    _store_write(tmp_path, summary="lesson", body="one")
    [path] = _observation_files(tmp_path)
    memory_dir = tmp_path / "memories"
    metadata, body = read_observation_document(path)
    metadata.summary = "lesson, restated"
    frozen = path.stat()
    original_stat = Path.stat
    monkeypatch.setattr(
        store.Path,
        "stat",
        lambda self, *a, **k: frozen if self == path else original_stat(self, *a, **k),
    )
    original_parse = store._parse_observation_search_document
    raced = {"done": False}

    def parse_then_lose_the_race(**kwargs: Any) -> Any:
        parsed = original_parse(**kwargs)
        if not raced["done"]:
            raced["done"] = True
            write_observation_document(path, metadata=metadata, body=body)
        return parsed

    monkeypatch.setattr(
        store, "_parse_observation_search_document", parse_then_lose_the_race
    )

    [stale] = list_observation_documents(memory_dir=memory_dir, project_id="P-test")
    [fresh] = list_observation_documents(memory_dir=memory_dir, project_id="P-test")

    assert stale.summary == "lesson"
    assert fresh.summary == "lesson, restated"


def test_observation_rewrite_retries_a_busy_target(tmp_path, monkeypatch):
    """On Windows a reader's open handle fails the rename for a moment."""
    from EvoScientist.memory.observations import store

    _store_write(tmp_path, summary="lesson", body="one")
    [path] = _observation_files(tmp_path)
    before = path.read_text(encoding="utf-8")
    metadata, body = read_observation_document(path)
    failures = {"left": 2}
    original_replace = store._os_replace
    naps: list[float] = []

    def busy_replace(source: Any, target: Any) -> None:
        if failures["left"]:
            failures["left"] -= 1
            raise PermissionError(13, "busy", str(target))
        original_replace(source, target)

    monkeypatch.setattr(store, "_os_replace", busy_replace)
    monkeypatch.setattr(store, "_sleep", naps.append)

    write_observation_document(path, metadata=metadata, body=body.replace("one", "two"))
    assert naps == [store._REPLACE_RETRY_SECONDS] * 2
    assert "two" in path.read_text(encoding="utf-8")

    failures["left"] = store._REPLACE_ATTEMPTS + 1
    with pytest.raises(PermissionError):
        write_observation_document(
            path, metadata=metadata, body=body.replace("one", "three")
        )
    assert "two" in path.read_text(encoding="utf-8")
    assert "three" not in path.read_text(encoding="utf-8")
    assert [p for p in path.parent.iterdir() if p.suffix != ".md"] == []
    assert before != path.read_text(encoding="utf-8")


def test_main_agent_record_tool_itself_has_no_per_run_limit(tmp_path):
    """The cap lives in the worker middleware; the tool shared with the main
    agent writes as often as it is called."""
    from EvoScientist.middleware.memory import create_memory_middleware

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    middleware = create_memory_middleware(
        str(tmp_path / "memories"),
        workspace_dir=workspace,
        source_type=MemorySourceType.TURN,
        source_agent="EvoScientist",
    )
    tool = tool_by_name(middleware.tools, "record_observation")
    runtime = tool_runtime(
        tool,
        tool_call_id="tool-1",
        config={"configurable": {"evomemory_source_session_id": SESSION}},
    )
    payloads = [
        record_observation_payload(
            tool,
            runtime=runtime,
            memory_type=MemoryType.PROCEDURAL,
            summary=f"lesson {i}",
            observation=f"body {i}",
            why_it_matters="it repeats",
            scope=MemoryScope.PROJECT,
        )
        for i in range(5)
    ]

    assert all(p["created"] for p in payloads)
    assert len(_observation_files(tmp_path)) == 5


def test_main_agent_middleware_carries_no_guardrail(tmp_path):
    from EvoScientist.EvoScientist import _get_default_middleware

    middleware = _get_default_middleware(workspace=Workspace(tmp_path / "workspace"))

    assert not any(
        isinstance(m, memory_worker._MemoryWorkerGuardrailMiddleware)
        for m in middleware
    )


def test_search_documents_carry_the_source_session(tmp_path):
    _store_write(tmp_path, summary="lesson", body="lesson body")
    documents = list_observation_documents(
        memory_dir=tmp_path / "memories", project_id="P-test"
    )

    assert [d.source_session_id for d in documents] == [SESSION]


def _write_dated(
    tmp_path: Path,
    *,
    observation_id: str,
    summary: str,
    created_at: str | None,
    session: str | None,
    scope: MemoryScope = MemoryScope.PROJECT,
) -> None:
    observations = tmp_path / "memories" / "observations"
    folder = (
        observations / "projects" / "P-test"
        if scope == MemoryScope.PROJECT
        else observations / "global"
    )
    folder.mkdir(parents=True, exist_ok=True)
    source = (
        ObservationSourceFrontmatter(
            type=MemorySourceType.SUBAGENT,
            agent="general-purpose",
            session_id=session,
        )
        if session
        else None
    )
    write_observation_document(
        folder / f"{observation_id}.md",
        metadata=ObservationFrontmatter(
            id=observation_id,
            created_at=created_at,
            summary=summary,
            memory_type=MemoryType.PROCEDURAL,
            scope=scope,
            project_id="P-test" if scope == MemoryScope.PROJECT else None,
            source=source,
        ),
        body=f"\n## Observation\n\n{summary}\n\n## Why It Matters\n\nx\n",
    )


def _index_rows(index: str) -> list[str]:
    return [line.split()[1] for line in index.splitlines() if line.startswith("- O-")]


SHARE_NOTE = "observations are listed before its older ones"


def _write_sessions(tmp_path: Path, sessions: list[tuple[str, int]]) -> None:
    """Write rows of equal length; the first session listed is the newest."""
    minute = 0
    for session, count in reversed(sessions):
        for _ in range(count):
            _write_dated(
                tmp_path,
                observation_id=f"O-{session}{minute:015d}",
                summary=f"entry {minute:03d}",
                created_at=f"2026-10-15T{minute // 60:02d}:{minute % 60:02d}:00Z",
                session=f"session-{session}",
            )
            minute += 1


def _ids(session: str, minutes: range) -> list[str]:
    return [f"O-{session}{minute:015d}" for minute in minutes]


def _index(tmp_path: Path, **kwargs: Any) -> str:
    return build_observation_index_context(
        memory_dir=tmp_path / "memories", project_id="P-test", **kwargs
    )


TRUNCATED_LINE = "Observation index truncated to entries that fit."


def _frame_and_rows(full: str) -> tuple[int, list[str]]:
    """Split an index that fits into the length of its frame and its rows."""
    rows = [line for line in full.splitlines() if line.startswith("- O-")]
    return len(full) - sum(len(row) + 1 for row in rows), rows


def _write_overflowing_global_memory(tmp_path: Path) -> None:
    """26 sessions with 3 rows each, then one session with 70 newer rows."""
    minute = 0
    sessions = [(f"session-{n:02d}", 3) for n in range(26)] + [("session-x", 70)]
    for session, count in sessions:
        for _ in range(count):
            _write_dated(
                tmp_path,
                observation_id=f"O-{minute:016d}",
                summary=f"entry {minute:03d}",
                created_at=f"2026-10-15T{minute // 60:02d}:{minute % 60:02d}:00Z",
                session=session,
                scope=MemoryScope.GLOBAL,
            )
            minute += 1


def test_inline_index_lists_every_row_when_it_fits(tmp_path):
    _write_sessions(tmp_path, [("a", 10), ("b", 2)])
    _write_dated(
        tmp_path,
        observation_id="O-c000000000000000",
        summary="legacy entry",
        created_at="2026-09-01T00:00:00Z",
        session=None,
    )

    index = _index(tmp_path)

    assert len(_index_rows(index)) == 13
    assert "Counts: total=13;" in index
    assert "truncated" not in index
    assert SHARE_NOTE not in index
    assert _index(tmp_path, max_inline_chars=len(index)) == index
    assert TRUNCATED_LINE in _index(tmp_path, max_inline_chars=len(index) - 1)


def test_overflowing_index_gives_each_session_its_newest_three_first(tmp_path):
    """Session a wrote ten rows; b, c and d keep their rows before a's fourth."""
    _write_sessions(tmp_path, [("a", 10), ("b", 3), ("c", 3), ("d", 2)])
    first_pass = [
        *_ids("a", range(17, 14, -1)),
        *_ids("b", range(7, 4, -1)),
        *_ids("c", range(4, 1, -1)),
        *_ids("d", range(1, -1, -1)),
    ]
    probe = _index(tmp_path, max_inline_chars=len(_index(tmp_path)) - 1)
    row = next(line for line in probe.splitlines() if line.startswith("- O-"))
    per_row = len(row) + 1
    base = len(probe) - len(_index_rows(probe)) * per_row

    eleven = _index(tmp_path, max_inline_chars=base + 11 * per_row)
    thirteen = _index(tmp_path, max_inline_chars=base + 13 * per_row)
    six = _index(tmp_path, max_inline_chars=base + 6 * per_row)

    for built, budget in ((eleven, 11), (thirteen, 13), (six, 6)):
        assert len(built) <= base + budget * per_row
    assert _index_rows(eleven) == first_pass
    assert _index_rows(thirteen) == [
        *_ids("a", range(17, 12, -1)),
        *first_pass[3:],
    ]
    assert _index_rows(six) == first_pass[:6]
    assert "Counts: total=18;" in six
    lines = eleven.splitlines()
    note_at = next(i for i, line in enumerate(lines) if SHARE_NOTE in line)
    assert lines[note_at] == (
        "Each session's newest 3 observations are listed before its older ones; "
        "search_observations finds the rest."
    )
    assert lines[note_at - 1] == "Indexed observations:"
    assert lines[note_at + 1].startswith("- O-")

    by_time = _index(
        tmp_path, max_inline_chars=base + 11 * per_row, rows_per_session=None
    )
    newest = [
        *_ids("a", range(17, 7, -1)),
        *_ids("b", range(7, 4, -1)),
        *_ids("c", range(4, 1, -1)),
        *_ids("d", range(1, -1, -1)),
    ]
    assert _index_rows(by_time) == newest[: len(_index_rows(by_time))]
    assert SHARE_NOTE not in by_time

    linker = build_observation_linker_index_context(
        memory_dir=tmp_path / "memories",
        project_id="P-test",
        exclude_ids=(),
        max_inline_chars=base + 11 * per_row,
    )
    assert _index_rows(linker) == newest[: len(_index_rows(linker))]
    assert SHARE_NOTE not in linker

    no_tools = _index(
        tmp_path,
        include_search_hints=False,
        max_inline_chars=len(_index(tmp_path, include_search_hints=False)) - 1,
    )
    assert "newest 3 observations are listed before its older ones." in no_tools
    assert "search_observations" not in no_tools

    too_large = _index(tmp_path, max_inline_chars=100)
    assert "too large to inline" in too_large
    assert _index_rows(too_large) == []
    assert SHARE_NOTE not in too_large


def test_one_session_overflow_fills_exactly_like_time_order(tmp_path):
    """With nobody to crowd out, the share changes nothing and adds no note."""
    _write_sessions(tmp_path, [("a", 30)])
    budget = len(_index(tmp_path)) // 2

    shared = _index(tmp_path, max_inline_chars=budget)

    assert shared == _index(tmp_path, max_inline_chars=budget, rows_per_session=None)
    assert "truncated" in shared
    assert len(_index_rows(shared)) > 3
    assert SHARE_NOTE not in shared


def test_one_session_keeps_time_order_when_a_long_row_is_skipped(tmp_path):
    """A long row that no longer fits is skipped by size alone; the share must
    not read that as a session giving way, add a note or drop another row."""
    for i, length in enumerate([10, 10, 10, 200, 10]):
        _write_dated(
            tmp_path,
            observation_id=f"O-a{i:015d}",
            summary="s" * length,
            created_at=f"2026-10-15T17:{59 - i:02d}:00Z",
            session="session-a",
        )
    frame, rows = _frame_and_rows(_index(tmp_path))
    short = len(rows[0]) + 1
    budget = frame + len(TRUNCATED_LINE) + 1 + 4 * short

    shared = _index(tmp_path, max_inline_chars=budget)

    assert shared == _index(tmp_path, max_inline_chars=budget, rows_per_session=None)
    assert _index_rows(shared) == [*_ids("a", range(0, 3)), "O-a000000000000004"]
    assert len(shared) == budget
    assert SHARE_NOTE not in shared


def test_rows_without_a_session_are_never_deferred(tmp_path):
    rows = [("b", "session-b"), ("c", None)] + [("a", "session-a")] * 6
    for minute, (prefix, session) in enumerate(rows):
        _write_dated(
            tmp_path,
            observation_id=f"O-{prefix}{minute:015d}",
            summary=f"entry {minute:03d}",
            created_at=f"2026-10-15T17:{minute:02d}:00Z",
            session=session,
        )
    frame, lines = _frame_and_rows(_index(tmp_path))
    per_row = len(lines[0]) + 1
    note = (
        "Each session's newest 3 observations are listed before its older ones; "
        "search_observations finds the rest."
    )
    fits_five = frame + len(TRUNCATED_LINE) + 1 + len(note) + 1 + 5 * per_row

    index = _index(tmp_path, max_inline_chars=fits_five + per_row - 1)

    assert _index_rows(index) == [
        *_ids("a", range(7, 4, -1)),
        "O-c000000000000001",
        "O-b000000000000000",
    ]
    assert note in index
    assert len(index) == fits_five


def test_many_rows_without_a_session_still_go_by_time(tmp_path):
    """Rows without a session are not one session: five of them keep their
    place ahead of an older named session."""
    rows = [("b", "session-b")] * 2 + [("c", None)] * 5
    for minute, (prefix, session) in enumerate(rows):
        _write_dated(
            tmp_path,
            observation_id=f"O-{prefix}{minute:015d}",
            summary=f"entry {minute:03d}",
            created_at=f"2026-10-15T17:{minute:02d}:00Z",
            session=session,
        )
    frame, lines = _frame_and_rows(_index(tmp_path))
    per_row = len(lines[0]) + 1
    fits_five = frame + len(TRUNCATED_LINE) + 1 + 5 * per_row

    index = _index(tmp_path, max_inline_chars=fits_five + per_row - 1)

    assert TRUNCATED_LINE in index
    assert _index_rows(index) == _ids("c", range(6, 1, -1))
    assert SHARE_NOTE not in index
    assert len(index) == fits_five


def test_share_drops_its_note_when_the_note_leaves_no_room(tmp_path):
    """Room for one row: the share still picks b's row, without the note."""
    rows = [("b", "session-b", 10), ("a", "session-a", 10)] + [
        ("a", "session-a", 200)
    ] * 3
    for minute, (prefix, session, length) in enumerate(rows):
        _write_dated(
            tmp_path,
            observation_id=f"O-{prefix}{minute:015d}",
            summary="s" * length,
            created_at=f"2026-10-15T17:{minute:02d}:00Z",
            session=session,
        )
    frame, lines = _frame_and_rows(_index(tmp_path))
    short = min(len(line) for line in lines) + 1
    budget = frame + len(TRUNCATED_LINE) + 1 + short

    by_time = _index(tmp_path, max_inline_chars=budget, rows_per_session=None)
    shared = _index(tmp_path, max_inline_chars=budget)

    assert _index_rows(by_time) == ["O-a000000000000001"]
    assert _index_rows(shared) == ["O-b000000000000000"]
    assert SHARE_NOTE not in shared
    assert len(shared) == budget


def test_overflow_without_a_session_past_its_share_adds_no_note(tmp_path):
    _write_sessions(tmp_path, [("a", 3), ("b", 3), ("c", 3), ("d", 3)])
    index = _index(tmp_path, max_inline_chars=len(_index(tmp_path)) // 2)

    assert "truncated" in index
    assert _index_rows(index) == _ids("a", range(11, -1, -1))[: len(_index_rows(index))]
    assert SHARE_NOTE not in index


def test_session_share_counts_a_session_across_scopes(tmp_path):
    scopes = [
        MemoryScope.PROJECT,
        MemoryScope.GLOBAL,
        MemoryScope.PROJECT,
        MemoryScope.GLOBAL,
    ]
    for i, scope in enumerate(scopes):
        _write_dated(
            tmp_path,
            observation_id=f"O-a{i:015d}",
            summary=f"entry {i}",
            created_at=f"2026-10-15T17:{i:02d}:00Z",
            session="session-a",
            scope=scope,
        )
    for i in range(3):
        _write_dated(
            tmp_path,
            observation_id=f"O-b{i:015d}",
            summary=f"entry {i}",
            created_at=f"2026-10-14T10:{i:02d}:00Z",
            session="session-b",
        )

    index = _index(tmp_path, max_inline_chars=len(_index(tmp_path)) - 1)
    rows = _index_rows(index)

    assert "O-a000000000000000" not in rows
    assert rows[:3] == [
        "O-a000000000000003",
        "O-a000000000000002",
        "O-a000000000000001",
    ]
    assert "Counts: total=7; scope global=2, project=5;" in index
    assert SHARE_NOTE in index


def test_inline_index_orders_rows_by_instant_not_by_string(tmp_path):
    stamps = {
        "O-a000000000000000": "2026-10-15T12:00:00+02:00",
        "O-a000000000000001": "2026-10-15T10:10:00Z",
        "O-a000000000000002": "2026-10-15T10:20:00Z",
        "O-a000000000000003": "2026-10-15T10:30:00Z",
    }
    for observation_id, created_at in stamps.items():
        _write_dated(
            tmp_path,
            observation_id=observation_id,
            summary=f"entry {observation_id[-1]}",
            created_at=created_at,
            session="session-a",
        )
    _write_dated(
        tmp_path,
        observation_id="O-c000000000000000",
        summary="unreadable date",
        created_at="yesterday",
        session=None,
    )
    _write_dated(
        tmp_path,
        observation_id="O-b000000000000000",
        summary="undated entry",
        created_at=None,
        session=None,
    )

    index = build_observation_index_context(
        memory_dir=tmp_path / "memories", project_id="P-test"
    )

    # 12:00+02:00 is 10:00Z, the oldest, although it sorts first as a string;
    # rows without a readable date go last, in id order.
    assert _index_rows(index) == [
        "O-a000000000000003",
        "O-a000000000000002",
        "O-a000000000000001",
        "O-a000000000000000",
        "O-b000000000000000",
        "O-c000000000000000",
    ]


@pytest.mark.parametrize("source_type", list(MemorySourceType))
def test_worker_index_fills_by_time_alone(tmp_path, monkeypatch, source_type):
    """The share serves the agents doing work; a worker checks what its own,
    newest session already recorded, so its overflowing index goes by time."""
    _write_overflowing_global_memory(tmp_path)
    script = _finish() if source_type == MemorySourceType.SUBAGENT else _ai(text="done")
    _, model = _run_worker(tmp_path, monkeypatch, [script], source_type=source_type)

    worker_index = model.systems[0]
    hints = "Search hints:" in worker_index
    by_time = _index(tmp_path, rows_per_session=None, include_search_hints=hints)
    assert TRUNCATED_LINE in worker_index
    assert _index_rows(worker_index) == _index_rows(by_time)
    assert _index_rows(worker_index) != _index_rows(
        _index(tmp_path, include_search_hints=hints)
    )
    assert SHARE_NOTE not in worker_index


def test_working_agents_share_the_index_by_session(tmp_path, monkeypatch):
    """The real wiring: the main agent, the async sub-agent graphs and the
    sync sub-agents all share an overflowing index by session."""
    from EvoScientist import paths
    from EvoScientist.EvoScientist import (
        _get_default_middleware,
        _inject_subagent_middleware,
    )
    from EvoScientist.middleware.memory import EvoMemoryMiddleware

    _write_overflowing_global_memory(tmp_path)
    shared = _index_rows(_index(tmp_path))
    assert shared != _index_rows(_index(tmp_path, rows_per_session=None))
    monkeypatch.setattr(paths, "MEMORIES_DIR", tmp_path / "memories")
    workspace = Workspace(tmp_path / "workspace")
    subagents = [{"name": "research-agent"}]
    _inject_subagent_middleware(subagents, workspace=workspace)
    stacks = [
        _get_default_middleware(workspace=workspace),
        _get_default_middleware(
            workspace=workspace,
            for_async_subagent=True,
            memory_source_agent="scheduler",
        ),
        subagents[0]["middleware"],
    ]

    for stack in stacks:
        memory = next(m for m in stack if isinstance(m, EvoMemoryMiddleware))
        assert memory._inline_rows_per_session == DEFAULT_INLINE_ROWS_PER_SESSION
        index = memory._refresh_observation_index_context()
        assert _index_rows(index) == shared
        assert SHARE_NOTE in index


@pytest.mark.parametrize("source_type", list(MemorySourceType))
def test_guardrail_is_mounted_last_on_both_workers(tmp_path, source_type):
    middleware = memory_worker._memory_worker_middleware(
        memory_dir=tmp_path / "memories",
        workspace_dir=tmp_path / "workspace",
        source_type=source_type,
        observation_writer=MemoryObservationWriter.ALL,
    )
    guardrail = middleware[-1]
    assert isinstance(guardrail, memory_worker._MemoryWorkerGuardrailMiddleware)
    assert guardrail.observation_limit == 3
    assert guardrail.record_call_limit == 6
    assert guardrail.model_call_budget == BUDGET
    expected_schema = (
        memory_worker.SubagentMemoryDecision
        if source_type == MemorySourceType.SUBAGENT
        else None
    )
    assert guardrail._summary_schema is expected_schema


def test_strategy_replica_skips_the_wrap_up_without_the_name_list(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(memory_worker, "_STRUCTURED_OUTPUT_MODEL_PATTERNS", None)
    for model_name in ("openai/gpt-5.5", "anthropic/claude-haiku-5.5", None):
        model = _ScriptedModel(scripts=[_finish()], model_name=model_name)
        request = SimpleNamespace(
            model=model, response_format=memory_worker.SubagentMemoryDecision
        )
        # Unsure which strategy langchain binds: never force a tool choice.
        assert memory_worker._tool_strategy_run(request) is False


_PROVIDER_KEY_ENV = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "DEEPSEEK_API_KEY",
    "MINIMAX_API_KEY",
    "KIMI_API_KEY",
    "MIMO_API_KEY",
    "MIMO_TOKEN_PLAN_API_KEY",
    "CUSTOM_ANTHROPIC_API_KEY",
    "CUSTOM_OPENAI_API_KEY",
    "MOONSHOT_API_KEY",
    "ZHIPUAI_API_KEY",
    "DASHSCOPE_API_KEY",
    "GOOGLE_API_KEY",
    "SILICONFLOW_API_KEY",
    "VOLCENGINE_API_KEY",
    "ARK_API_KEY",
    "NOVITA_API_KEY",
    "REQUESTY_API_KEY",
    "ATLASCLOUD_API_KEY",
    "NVIDIA_API_KEY",
)


def test_strategy_replica_never_claims_a_tool_strategy_langchain_lacks(monkeypatch):
    """Across the registry the replica may only err towards skipping a wrap-up:
    a run it calls tool-strategy must be one langchain binds that way too."""
    from types import SimpleNamespace

    from langchain.agents.factory import _supports_provider_strategy
    from langchain.agents.structured_output import ToolStrategy
    from langchain_core.tools import tool
    from langchain_nvidia_ai_endpoints._common import _NVIDIABaseClient

    from EvoScientist.llm import get_chat_model
    from EvoScientist.llm.registry import _MODEL_ENTRIES
    from EvoScientist.middleware.utils import structured_output_for

    for key in _PROVIDER_KEY_ENV:
        monkeypatch.setenv(key, "sk-test")
    monkeypatch.setenv("CUSTOM_ANTHROPIC_BASE_URL", "http://127.0.0.1:9/anthropic")
    monkeypatch.setenv("CUSTOM_OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    # ChatNVIDIA asks the API about model names outside its static table.
    monkeypatch.setattr(
        _NVIDIABaseClient, "available_models", property(lambda self: [])
    )

    @tool
    def search_observations(query: str) -> str:
        """Search."""
        return ""

    compared = 0
    unsafe: list[str] = []
    for _short, model_id, provider in _MODEL_ENTRIES:
        try:
            model = get_chat_model(model=model_id, provider=provider)
            model, response_format = structured_output_for(
                model, memory_worker.SubagentMemoryDecision
            )
        except Exception:
            continue
        compared += 1
        replica = memory_worker._tool_strategy_run(
            SimpleNamespace(model=model, response_format=response_format)
        )
        langchain_tool = isinstance(
            response_format, ToolStrategy
        ) or not _supports_provider_strategy(model, tools=[search_observations])
        if replica and not langchain_tool:
            unsafe.append(f"{provider}:{model_id}")

    assert compared > 100
    assert unsafe == []
