"""Dedicated async agent for observation-to-knowledge synthesis."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    TypeAlias,
    TypedDict,
    TypeVar,
    cast,
)

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain.agents.structured_output import ToolStrategy
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.config import get_config
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from .. import paths as _paths
from .knowledge import (
    SYNTHESIS_AGENT_NAME,
    ReadMemoryArgs,
    SearchMemoryArgs,
    archive_knowledge_file,
    knowledge_search_documents,
    read_knowledge_file,
    read_memory_file,
    record_knowledge_file,
    search_memory_files,
)
from .observations import read_observation_file
from .types import (
    KnowledgeRecordResult,
    KnowledgeStatus,
    MemoryLevelFilter,
    MemoryScope,
    MemorySearchMode,
    MemoryType,
)
from .worker_activity import (
    mark_synthesis_finished,
    mark_synthesis_started,
    snapshot_memory_outputs,
)

if TYPE_CHECKING:
    from langgraph_sdk.schema import Config, Input, Run

logger = logging.getLogger(__name__)

SYNTHESIS_GRAPH_ID = "evomemory-synthesizer"
SYNTHESIS_RECURSION_LIMIT = 100
SYNTHESIS_CONTEXT_MAX_CHARS = 32_000
SYNTHESIS_CONTEXT_OBSERVATION_SUMMARY_CHARS = 320
SYNTHESIS_CONTEXT_OBSERVATION_CLAMPED_SUMMARY_CHARS = 120
SYNTHESIS_CONTEXT_OBSERVATION_SNIPPET_CHARS = 500
_SYNTHESIS_TERMINAL_STATUSES = frozenset({"success", "error", "timeout", "interrupted"})
_SYNTHESIS_POLL_INTERVAL_SECONDS = 1.0
_SYNTHESIS_MAX_POLL_FAILURES = 3
# Bounded retries for runs that fail transiently (run error/timeout/interrupted
# or abandoned polling). After this many attempts the context is released anyway
# so a dead LangGraph-dev server cannot wedge synthesis tracking.
_SYNTHESIS_MAX_RUN_ATTEMPTS = 3
_SYNTHESIS_RETRY_BACKOFF_SECONDS = 2.0
_active_synthesis_lock = threading.Lock()
_active_synthesis_contexts: set[tuple[str, str]] = set()


class SynthesisAction(StrEnum):
    """Actions the synthesizer may ask deterministic code to apply."""

    SKIP = "skip"
    CREATE = "create"
    UPDATE = "update"
    ARCHIVE = "archive"


class _SynthesisDecisionBase(BaseModel):
    """Common fields for one synthesis maintenance decision."""

    rationale: str = Field(
        min_length=1,
        description="Brief reason for this decision.",
    )


class SynthesisSkipDecision(_SynthesisDecisionBase):
    """Decision to leave the current context unchanged."""

    action: Literal[SynthesisAction.SKIP] = Field(
        description="Skip when the context does not justify durable knowledge changes.",
    )


class SynthesisCreateDecision(_SynthesisDecisionBase):
    """Decision to create a new synthesized knowledge record."""

    action: Literal[SynthesisAction.CREATE] = Field(
        description="Create a new reusable knowledge record.",
    )
    summary: str = Field(
        min_length=1,
        description="One-line summary for create/update decisions.",
    )
    memory_type: MemoryType = Field(
        description="Knowledge memory type for create/update decisions.",
    )
    scope: MemoryScope = Field(
        description="Knowledge scope for create/update decisions.",
    )
    knowledge: str = Field(
        min_length=1,
        description="Compact synthesized knowledge body for create/update decisions.",
    )
    when_to_use: str | None = Field(
        default=None,
        description="Short guidance describing when future agents should use it.",
    )
    supporting_observation_ids: list[str] = Field(
        min_length=1,
        description="Minimal O-* evidence IDs supporting create/update decisions.",
    )


class SynthesisUpdateDecision(_SynthesisDecisionBase):
    """Decision to update an existing synthesized knowledge record."""

    action: Literal[SynthesisAction.UPDATE] = Field(
        description="Update an existing reusable knowledge record.",
    )
    target_knowledge_id: str = Field(
        min_length=1,
        description="Existing K-* id for update/archive decisions.",
    )
    summary: str = Field(
        min_length=1,
        description="One-line summary for create/update decisions.",
    )
    memory_type: MemoryType = Field(
        description="Knowledge memory type for create/update decisions.",
    )
    knowledge: str = Field(
        min_length=1,
        description="Compact synthesized knowledge body for create/update decisions.",
    )
    when_to_use: str | None = Field(
        default=None,
        description="Short guidance describing when future agents should use it.",
    )
    supporting_observation_ids: list[str] = Field(
        min_length=1,
        description="Minimal O-* evidence IDs supporting create/update decisions.",
    )


class SynthesisArchiveDecision(_SynthesisDecisionBase):
    """Decision to archive an existing synthesized knowledge record."""

    action: Literal[SynthesisAction.ARCHIVE] = Field(
        description="Archive an obsolete, contradicted, or superseded knowledge record.",
    )
    target_knowledge_id: str = Field(
        min_length=1,
        description="Existing K-* id for update/archive decisions.",
    )
    archive_reason: str | None = Field(
        default=None,
        description="Reason to store when archiving an existing knowledge record.",
    )


SynthesisDecision: TypeAlias = Annotated[
    SynthesisSkipDecision
    | SynthesisCreateDecision
    | SynthesisUpdateDecision
    | SynthesisArchiveDecision,
    Field(discriminator="action"),
]


class SynthesisReviewDecision(BaseModel):
    """Structured synthesis review returned by the synthesis agent."""

    decisions: list[SynthesisDecision] = Field(
        default_factory=list,
        description=(
            "Maintenance decisions. Return an empty list when no durable "
            "knowledge should be created, updated, or archived."
        ),
    )
    no_op_reason: str | None = Field(
        default=None,
        description="Reason no decisions were made, when decisions is empty.",
    )


class SynthesisRunPayload(TypedDict):
    """Typed payload submitted to LangGraph SDK runs.create."""

    assistant_id: str
    input: Input
    metadata: dict[str, str]
    config: Config


class SynthesisObservationContext(TypedDict):
    """Compact seed observation included in a bounded synthesis prompt."""

    id: str
    path: str
    memory_type: str
    scope: str
    summary: str
    snippet: str


class SynthesisMemoryInventory(TypedDict):
    """Compact counts that tell the synthesizer what memory exists to inspect."""

    active_knowledge_count: int
    seed_observation_count: int


class SynthesisContext(TypedDict):
    """Bounded seed context submitted to synthesis."""

    project_id: str
    uncovered_observations: list[SynthesisObservationContext]
    memory_inventory: SynthesisMemoryInventory


T = TypeVar("T", bound=BaseModel)


def _agent_result_model(result: Mapping[str, object], model_type: type[T]) -> T | None:
    value = result.get("structured_response")
    if isinstance(value, model_type):
        return value
    if isinstance(value, dict):
        try:
            return model_type.model_validate(value)
        except Exception:
            return None
    return None


def _current_configurable() -> Mapping[str, object]:
    try:
        config = get_config()
    except RuntimeError:
        return {}
    configurable = config.get("configurable", {})
    return configurable if isinstance(configurable, dict) else {}


def _config_str(configurable: Mapping[str, object], key: str) -> str | None:
    value = configurable.get(key)
    return value if isinstance(value, str) and value else None


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _stable_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _pretty_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def _truncate_text(text: str, max_chars: int) -> str:
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    return stripped[: max_chars - 20].rstrip() + "\n[truncated]"


def _shrink_synthesis_context(
    context: SynthesisContext,
    *,
    max_chars: int,
) -> SynthesisContext:
    """Shrink seed context detail without dropping seed observation ids."""
    encoded = _pretty_json(context)
    if len(encoded) <= max_chars:
        return context

    for observation in context["uncovered_observations"]:
        observation["snippet"] = ""
    encoded = _pretty_json(context)
    if len(encoded) <= max_chars:
        return context

    for observation in context["uncovered_observations"]:
        observation["summary"] = _truncate_text(
            observation["summary"],
            SYNTHESIS_CONTEXT_OBSERVATION_CLAMPED_SUMMARY_CHARS,
        )
    encoded = _pretty_json(context)
    if len(encoded) <= max_chars:
        return context

    for observation in context["uncovered_observations"]:
        observation["summary"] = ""
    encoded = _pretty_json(context)

    if len(encoded) > max_chars:
        logger.warning(
            "Submitting oversized EvoMemory synthesis context: %d chars for "
            "%d seed observations after stripping snippets and summaries",
            len(encoded),
            len(context["uncovered_observations"]),
        )
    return context


def _normalize_seed_observation_ids(
    observation_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """Strip blanks and dedupe seed observation ids in first-seen order."""
    normalized = []
    seen = set()
    for observation_id in observation_ids:
        clean_id = observation_id.strip()
        if not clean_id or clean_id in seen:
            continue
        normalized.append(clean_id)
        seen.add(clean_id)
    return tuple(normalized)


def build_synthesis_context(
    *,
    memory_dir: str | Path,
    project_id: str,
    seed_observation_ids: tuple[str, ...],
    max_chars: int = SYNTHESIS_CONTEXT_MAX_CHARS,
) -> SynthesisContext | None:
    """Build bounded seed context for the synthesis agent."""
    seed_ids = _normalize_seed_observation_ids(seed_observation_ids)
    if not seed_ids:
        return None

    knowledge_documents = knowledge_search_documents(
        memory_dir=memory_dir,
        project_id=project_id,
        status=KnowledgeStatus.ACTIVE,
    )
    covered_observation_ids = {
        observation_id
        for document in knowledge_documents
        for observation_id in document.supporting_observation_ids
    }

    observations: list[SynthesisObservationContext] = []
    for observation_id in seed_ids:
        if observation_id in covered_observation_ids:
            continue
        document = read_observation_file(
            memory_dir=memory_dir,
            project_id=project_id,
            observation_id=observation_id,
        )
        if document is None:
            continue
        observations.append(
            {
                "id": document["observation_id"],
                "path": document["path"],
                "memory_type": document["memory_type"].value,
                "scope": document["scope"].value,
                "summary": _truncate_text(
                    document["summary"],
                    SYNTHESIS_CONTEXT_OBSERVATION_SUMMARY_CHARS,
                ),
                "snippet": _truncate_text(
                    document["text"],
                    SYNTHESIS_CONTEXT_OBSERVATION_SNIPPET_CHARS,
                ),
            }
        )
    if not observations:
        return None

    context: SynthesisContext = {
        "project_id": project_id,
        "uncovered_observations": observations,
        "memory_inventory": {
            "active_knowledge_count": len(knowledge_documents),
            "seed_observation_count": len(seed_ids),
        },
    }
    return _shrink_synthesis_context(context, max_chars=max_chars)


def _tool_error(message: str) -> str:
    return json.dumps({"error": message}, ensure_ascii=False, sort_keys=True)


def _synthesis_memory_tools(*, memory_dir: str | Path) -> list[BaseTool]:
    """Build read-side memory tools for the synthesis worker graph."""
    worker_memory_dir = Path(memory_dir).expanduser()

    def _project_id() -> str | None:
        return _config_str(_current_configurable(), "evomemory_project_id")

    def _search_memory(
        query: str,
        mode: MemorySearchMode = MemorySearchMode.RANKED,
        memory_level: MemoryLevelFilter = MemoryLevelFilter.ANY,
        scope: MemoryScope | None = None,
        memory_type: MemoryType | None = None,
        include_archived_knowledge: bool = False,
        limit: int = 8,
    ) -> str:
        project_id = _project_id()
        if project_id is None:
            return _tool_error("EvoMemory synthesizer missing project id.")
        results = search_memory_files(
            memory_dir=worker_memory_dir,
            project_id=project_id,
            query=query,
            memory_level=memory_level,
            scope=scope,
            memory_type=memory_type,
            include_archived_knowledge=include_archived_knowledge,
            limit=limit,
            mode=mode,
        )
        return json.dumps({"results": results}, ensure_ascii=False, sort_keys=True)

    def _read_memory(memory_id: str) -> str:
        project_id = _project_id()
        if project_id is None:
            return _tool_error("EvoMemory synthesizer missing project id.")
        result = read_memory_file(
            memory_dir=worker_memory_dir,
            project_id=project_id,
            memory_id=memory_id.strip(),
        )
        if result is None:
            return _tool_error(
                "No memory with that ID exists in global or current-project memory."
            )
        return json.dumps({"text": result["text"]}, ensure_ascii=False, sort_keys=True)

    return [
        StructuredTool.from_function(
            func=_search_memory,
            name="search_memory",
            description=(
                "Search EvoMemory from the synthesis worker. Use memory_level="
                "knowledge before creating or updating knowledge, and "
                "memory_level=observation when related raw evidence could affect "
                "the synthesis decision. Read promising K-* or O-* hits with "
                "read_memory."
            ),
            args_schema=SearchMemoryArgs,
            infer_schema=False,
        ),
        StructuredTool.from_function(
            func=_read_memory,
            name="read_memory",
            description=(
                "Read the full markdown for an EvoMemory knowledge or observation "
                "record by exact K-* or O-* ID. Read every observation you cite "
                "in a synthesis decision."
            ),
            args_schema=ReadMemoryArgs,
            infer_schema=False,
        ),
    ]


def _synthesis_system_prompt() -> str:
    return "\n\n".join(
        [
            "You are the EvoMemory synthesis agent.",
            (
                "Your job is to convert durable observations into compact "
                "knowledge records. The prompt contains a bounded seed list of "
                "uncovered observations, not a complete memory snapshot. Treat "
                "seed summaries and snippets as leads to inspect with memory "
                "tools before relying on them."
            ),
            (
                "Use `read_memory` to read every O-* observation you cite. Use "
                "`search_memory(memory_level=knowledge)` before creating new "
                "knowledge so you can update or archive existing K-* records "
                "instead of duplicating them. Use "
                "`search_memory(memory_level=observation)` when neighboring raw "
                "evidence or conflicts could change the decision."
            ),
            (
                "Do not continue the user task, infer from missing trajectory "
                "context, or create memories from vibes. Base decisions only on "
                "memory records you were given or retrieved."
            ),
            (
                "Knowledge is an abstraction over evidence, not a restatement "
                "of each observation. Prefer fewer, higher-value records that "
                "merge related observations. Skip when a candidate would only "
                "promote one narrow observation without clear future decision "
                "value."
            ),
            (
                "Use a high evidence bar. Global semantic or empirical claims "
                "normally need at least two independent supporting observations. "
                "A single observation can justify a create only when it captures "
                "a concrete procedural/tooling fact, explicit project convention, "
                "reproducible failure mode, or explicit user preference. Every "
                "create/update decision must cite the minimal supporting O-* IDs."
            ),
            (
                "Do not widen scope beyond the evidence. Project-scoped "
                "observations should produce project knowledge. Global knowledge "
                "must be supported only by global observations. Preserve "
                "uncertainty and qualifiers from the evidence, including words "
                "like `appears`, version ranges, benchmark names, workspace "
                "constraints, and environment details."
            ),
            (
                "Prefer update over duplicate create when existing knowledge "
                "already captures the same idea. Archive only when existing "
                "knowledge is contradicted, obsolete, or fully superseded by a "
                "better active record."
            ),
            (
                "Return exactly one structured response by calling the "
                "`SynthesisReviewDecision` tool. Use `skip` or an empty decision "
                "list when the observations are too narrow, redundant, "
                "unsupported, routine, or merely session progress."
            ),
        ]
    )


def _synthesis_user_prompt(context: SynthesisContext, *, trigger: str) -> str:
    return "\n\n".join(
        [
            "New or still-uncovered EvoMemory observations are ready for "
            "synthesis review. Decide whether to create, update, archive, or "
            "skip synthesized knowledge.",
            f"Trigger: {trigger}",
            (
                "Use memory tools to inspect full observation records and "
                "nearby existing memory before deciding. The seed context is "
                "bounded; absence from it does not mean absence from memory."
            ),
            "Seed context JSON:",
            _pretty_json(context),
        ]
    )


def apply_synthesis_review_decision(
    *,
    memory_dir: str | Path,
    project_id: str,
    review: SynthesisReviewDecision,
    source_agent: str = SYNTHESIS_AGENT_NAME,
) -> list[KnowledgeRecordResult]:
    """Apply validated synthesis decisions to knowledge markdown files."""
    results: list[KnowledgeRecordResult] = []
    for decision in review.decisions:
        try:
            match decision:
                case SynthesisSkipDecision():
                    continue
                case SynthesisCreateDecision():
                    results.append(
                        record_knowledge_file(
                            memory_dir=memory_dir,
                            project_id=project_id,
                            memory_type=decision.memory_type,
                            summary=decision.summary,
                            knowledge=decision.knowledge,
                            supporting_observation_ids=decision.supporting_observation_ids,
                            scope=decision.scope,
                            when_to_use=decision.when_to_use,
                            source_agent=source_agent,
                        )
                    )
                case SynthesisUpdateDecision():
                    existing = read_knowledge_file(
                        memory_dir=memory_dir,
                        project_id=project_id,
                        knowledge_id=decision.target_knowledge_id,
                    )
                    if existing is None:
                        continue
                    results.append(
                        record_knowledge_file(
                            memory_dir=memory_dir,
                            project_id=project_id,
                            knowledge_id=decision.target_knowledge_id,
                            memory_type=decision.memory_type,
                            summary=decision.summary,
                            knowledge=decision.knowledge,
                            supporting_observation_ids=decision.supporting_observation_ids,
                            scope=existing["scope"],
                            when_to_use=decision.when_to_use,
                            source_agent=source_agent,
                        )
                    )
                case SynthesisArchiveDecision():
                    result = archive_knowledge_file(
                        memory_dir=memory_dir,
                        project_id=project_id,
                        knowledge_id=decision.target_knowledge_id,
                        reason=decision.archive_reason or decision.rationale,
                        source_agent=source_agent,
                    )
                    if result is not None:
                        results.append(result)
        except ValueError:
            logger.warning(
                "Skipping invalid EvoMemory synthesis decision: %s",
                decision,
                exc_info=True,
            )
    return results


class _SynthesisApplyMiddleware(AgentMiddleware):
    """Apply structured synthesis decisions after the agent finishes."""

    name = "evomemory_synthesis_apply"

    def __init__(self, *, memory_dir: str | Path) -> None:
        self._memory_dir = Path(memory_dir).expanduser()

    def _apply(self, state: AgentState[object]) -> None:
        review = _agent_result_model(state, SynthesisReviewDecision)
        if review is None:
            logger.warning("EvoMemory synthesizer returned no structured decision")
            return
        project_id = _config_str(_current_configurable(), "evomemory_project_id")
        if not project_id:
            logger.warning("EvoMemory synthesizer missing project id")
            return
        apply_synthesis_review_decision(
            memory_dir=self._memory_dir,
            project_id=project_id,
            review=review,
        )

    async def _aapply(self, state: AgentState[object]) -> None:
        await asyncio.to_thread(self._apply, state)

    def after_agent(
        self,
        state: AgentState[object],
        runtime: Runtime,
    ) -> dict[str, object] | None:
        self._apply(state)
        return None

    async def aafter_agent(
        self,
        state: AgentState[object],
        runtime: Runtime,
    ) -> dict[str, object] | None:
        await self._aapply(state)
        return None


def build_synthesis_agent_graph(
    *,
    memory_dir: str | Path | None = None,
) -> CompiledStateGraph:
    """Build the registered LangGraph synthesis worker."""
    from deepagents import create_deep_agent

    from ..EvoScientist import _ensure_auxiliary_chat_model

    worker_memory_dir = Path(
        _paths.MEMORIES_DIR if memory_dir is None else memory_dir
    ).expanduser()
    agent = create_deep_agent(
        name=SYNTHESIS_AGENT_NAME,
        model=_ensure_auxiliary_chat_model(),
        system_prompt=_synthesis_system_prompt(),
        tools=_synthesis_memory_tools(memory_dir=worker_memory_dir),
        middleware=[_SynthesisApplyMiddleware(memory_dir=worker_memory_dir)],
        subagents=[],
        response_format=ToolStrategy(
            schema=SynthesisReviewDecision,
            tool_message_content="Synthesis review accepted.",
        ),
    )
    return agent.with_config({"recursion_limit": SYNTHESIS_RECURSION_LIMIT})


def _runs_create_kwargs(kwargs: SynthesisRunPayload) -> SynthesisRunPayload:
    try:
        from EvoScientist.llm.patches import _merge_runs_config_kwargs
    except Exception:
        return kwargs
    return cast("SynthesisRunPayload", _merge_runs_config_kwargs(dict(kwargs)))


def _synthesis_thread_id(*, project_id: str, context: SynthesisContext) -> str:
    return f"evomemory-synth:{_short_hash(project_id + _stable_json(context))}"


def _synthesis_context_digest(context: SynthesisContext) -> str:
    return _short_hash(_stable_json(context))


def _claim_synthesis_context(*, project_id: str, context_digest: str) -> bool:
    key = (project_id, context_digest)
    with _active_synthesis_lock:
        if key in _active_synthesis_contexts:
            return False
        _active_synthesis_contexts.add(key)
        return True


def _release_synthesis_context(*, project_id: str, context_digest: str) -> None:
    with _active_synthesis_lock:
        _active_synthesis_contexts.discard((project_id, context_digest))
    mark_synthesis_finished(project_id=project_id, context_digest=context_digest)


def _synthesis_run_kwargs(
    *,
    project_id: str,
    context: SynthesisContext,
    trigger: str,
) -> SynthesisRunPayload:
    context_digest = _synthesis_context_digest(context)
    thread_id = _synthesis_thread_id(project_id=project_id, context=context)
    payload: SynthesisRunPayload = {
        "assistant_id": SYNTHESIS_GRAPH_ID,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": _synthesis_user_prompt(context, trigger=trigger),
                }
            ]
        },
        "metadata": {
            "agent_name": "EvoScientist",
            "run_kind": "evomemory_synthesis_worker",
            "project_id": project_id,
            "context_digest": context_digest,
            "trigger": trigger,
        },
        "config": {
            "configurable": {
                "thread_id": thread_id,
                "evomemory_project_id": project_id,
                "evomemory_context_digest": context_digest,
                "evomemory_trigger": trigger,
            }
        },
    }
    return _runs_create_kwargs(payload)


def _synthesis_worker_url() -> str:
    from ..EvoScientist import _ensure_config

    cfg = _ensure_config()
    port = int(getattr(cfg, "langgraph_dev_port", 6174))
    return f"http://localhost:{port}"


def _run_id_from_response(run: Run) -> str | None:
    run_id = run["run_id"].strip()
    return run_id or None


def _status_from_run_response(run: Run) -> str:
    return str(run["status"]).strip().lower()


def _delete_synthesis_thread(client: Any, thread_id: str) -> None:
    if not _memory_worker_thread_cleanup_enabled():
        logger.debug(
            "Preserving EvoMemory synthesis thread %s because cleanup is disabled",
            thread_id,
        )
        return
    try:
        client.threads.delete(thread_id)
    except Exception:
        logger.debug(
            "Failed to delete EvoMemory synthesis thread %s",
            thread_id,
            exc_info=True,
        )


def _memory_worker_thread_cleanup_enabled() -> bool:
    try:
        from ..config import get_effective_config

        return bool(get_effective_config().memory_worker_thread_cleanup_enabled)
    except Exception:
        logger.debug(
            "Failed to resolve EvoMemory worker thread cleanup config; "
            "defaulting to enabled",
            exc_info=True,
        )
        return True


class _SynthesisRunOutcome(StrEnum):
    """Result of submitting and polling a single synthesis run."""

    SUCCESS = "success"  # reached the success terminal status (writes or skip)
    FAILED = "failed"  # non-success terminal status, or could not be submitted
    ABORTED = "aborted"  # polling abandoned before terminal; run state unknown


def _poll_synthesis_run(
    client: Any,
    *,
    thread_id: str,
    run_id: str,
) -> _SynthesisRunOutcome:
    """Poll one synthesis run until it is terminal or polling is abandoned."""
    failures = 0
    while True:
        try:
            run = client.runs.get(thread_id=thread_id, run_id=run_id)
            failures = 0
        except Exception:
            failures += 1
            if failures >= _SYNTHESIS_MAX_POLL_FAILURES:
                logger.warning(
                    "Stopping EvoMemory synthesis status watch for %s after "
                    "%d failed polls",
                    run_id,
                    failures,
                    exc_info=True,
                )
                return _SynthesisRunOutcome.ABORTED
            time.sleep(_SYNTHESIS_POLL_INTERVAL_SECONDS)
            continue
        status = _status_from_run_response(run)
        if status in _SYNTHESIS_TERMINAL_STATUSES:
            return (
                _SynthesisRunOutcome.SUCCESS
                if status == "success"
                else _SynthesisRunOutcome.FAILED
            )
        time.sleep(_SYNTHESIS_POLL_INTERVAL_SECONDS)


def _submit_and_watch_synthesis_run(
    *,
    url: str,
    project_id: str,
    context: SynthesisContext,
    trigger: str,
) -> _SynthesisRunOutcome:
    """Submit one synthesis run and poll it to a terminal status."""
    from langgraph_sdk import get_sync_client

    try:
        client = get_sync_client(url=url, headers={"x-auth-scheme": "langsmith"})
    except Exception:
        logger.warning("Failed to create EvoMemory synthesis client", exc_info=True)
        return _SynthesisRunOutcome.ABORTED

    thread_id: str | None = None
    try:
        thread = client.threads.create(graph_id=SYNTHESIS_GRAPH_ID)
        thread_id = str(thread["thread_id"])
        payload = _synthesis_run_kwargs(
            project_id=project_id,
            context=context,
            trigger=trigger,
        )
        run = client.runs.create(
            thread_id=thread_id,
            assistant_id=payload["assistant_id"],
            input=payload["input"],
            metadata=payload["metadata"],
            config=payload["config"],
        )
        run_id = _run_id_from_response(run)
    except Exception:
        logger.warning("Failed to submit EvoMemory synthesis run", exc_info=True)
        # No live run is attached, so an empty thread is safe to drop.
        if thread_id is not None:
            _delete_synthesis_thread(client, thread_id)
        return _SynthesisRunOutcome.FAILED

    if run_id is None:
        _delete_synthesis_thread(client, thread_id)
        return _SynthesisRunOutcome.FAILED

    outcome = _poll_synthesis_run(client, thread_id=thread_id, run_id=run_id)
    # Only delete once the run is confirmed terminal — deleting a thread with a
    # live run would break it, and an aborted poll leaves the state unknown.
    if outcome is not _SynthesisRunOutcome.ABORTED:
        _delete_synthesis_thread(client, thread_id)
    return outcome


def _run_synthesis_with_retries(
    *,
    url: str,
    project_id: str,
    context: SynthesisContext,
    trigger: str,
    active_key: tuple[str, str],
    max_attempts: int = _SYNTHESIS_MAX_RUN_ATTEMPTS,
) -> None:
    """Submit synthesis runs until one succeeds or attempts run out, then release.

    A ``success`` terminal status — including a deliberate skip with no writes —
    stops immediately and is never retried. Transient failures (run
    error/timeout/interrupted, or abandoned polling) are retried up to
    ``max_attempts``. Once attempts are exhausted the context is released anyway:
    the seed observations stay on disk and searchable as raw evidence, they are
    simply not promoted into knowledge this round.
    """
    release_project_id, context_digest = active_key
    try:
        for attempt in range(1, max_attempts + 1):
            outcome = _submit_and_watch_synthesis_run(
                url=url,
                project_id=project_id,
                context=context,
                trigger=trigger,
            )
            if outcome is _SynthesisRunOutcome.SUCCESS:
                return
            if attempt < max_attempts:
                logger.info(
                    "Retrying EvoMemory synthesis for project %s "
                    "(attempt %d/%d) after %s",
                    project_id,
                    attempt + 1,
                    max_attempts,
                    outcome.value,
                )
                time.sleep(_SYNTHESIS_RETRY_BACKOFF_SECONDS)
        logger.warning(
            "Giving up EvoMemory synthesis for project %s after %d attempts; "
            "seed observations remain unsynthesized but searchable",
            project_id,
            max_attempts,
        )
    finally:
        _release_synthesis_context(
            project_id=release_project_id,
            context_digest=context_digest,
        )


def _spawn_synthesis_runner_thread(
    *,
    url: str,
    project_id: str,
    context: SynthesisContext,
    trigger: str,
    active_key: tuple[str, str],
) -> None:
    thread = threading.Thread(
        target=_run_synthesis_with_retries,
        kwargs={
            "url": url,
            "project_id": project_id,
            "context": context,
            "trigger": trigger,
            "active_key": active_key,
        },
        name="evomemory-synthesis-status",
        daemon=True,
    )
    thread.start()


def _launch_synthesis_worker(
    *,
    memory_dir: str | Path,
    project_id: str,
    seed_observation_ids: tuple[str, ...],
    trigger: str,
) -> None:
    """Submit a background synthesis run if uncovered seed observations exist."""
    from ..langgraph_dev.manager import is_langgraph_dev_running

    context = build_synthesis_context(
        memory_dir=memory_dir,
        project_id=project_id,
        seed_observation_ids=seed_observation_ids,
    )
    if context is None:
        return

    url = _synthesis_worker_url()
    if not is_langgraph_dev_running(base_url=url):
        logger.info("Skipping EvoMemory synthesis launch; LangGraph dev is unavailable")
        return

    context_digest = _synthesis_context_digest(context)
    if not _claim_synthesis_context(
        project_id=project_id,
        context_digest=context_digest,
    ):
        logger.debug(
            "Skipping duplicate EvoMemory synthesis launch for context %s",
            context_digest,
        )
        return

    mark_synthesis_started(
        project_id=project_id,
        context_digest=context_digest,
        memory_dir=memory_dir,
        before_outputs=snapshot_memory_outputs(memory_dir),
    )
    try:
        _spawn_synthesis_runner_thread(
            url=url,
            project_id=project_id,
            context=context,
            trigger=trigger,
            active_key=(project_id, context_digest),
        )
    except Exception:
        logger.warning("Failed to spawn EvoMemory synthesis runner", exc_info=True)
        _release_synthesis_context(
            project_id=project_id,
            context_digest=context_digest,
        )
