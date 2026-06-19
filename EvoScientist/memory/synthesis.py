"""Dedicated async agent for observation-to-knowledge synthesis."""

from __future__ import annotations

import asyncio
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
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from .. import paths as _paths
from ._common import (
    config_str,
    current_configurable,
    dedupe_ids,
    document_body,
    pretty_json,
    short_hash,
    stable_json,
)
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
from .observations import candidate_observation_documents, read_observation_file
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
SYNTHESIS_TOOL_DEFAULT_LIMIT = 20
SYNTHESIS_TOOL_MAX_LIMIT = 100
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

    explored_queries: list[str] = Field(
        default_factory=list,
        description=(
            "Memory search queries used while deciding the synthesis boundary."
        ),
    )
    read_memory_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Exact K-* and O-* IDs read with read_memory before making decisions. "
            "Every supporting O-* ID in a create/update decision must appear here."
        ),
    )
    boundary_rationale: str | None = Field(
        default=None,
        description=(
            "Brief explanation of how the exploration boundary was chosen, or "
            "why more exploration was unnecessary."
        ),
    )
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
    """Compact trigger observation included in a bounded synthesis prompt."""

    id: str
    path: str
    memory_type: str
    scope: str
    summary: str
    snippet: str
    covered_by_knowledge_ids: list[str]


class SynthesisMemoryInventory(TypedDict):
    """Compact counts that tell the synthesizer what memory exists to inspect."""

    active_knowledge_count: int
    archived_knowledge_count: int
    observation_count: int
    uncovered_observation_count: int
    covered_observation_count: int
    seed_observation_count: int


class SynthesisContext(TypedDict):
    """Bounded starting context submitted to synthesis."""

    project_id: str
    review_goal: str
    trigger_observation_ids: list[str]
    starting_observations: list[SynthesisObservationContext]
    memory_inventory: SynthesisMemoryInventory


class ListSynthesisObservationsArgs(BaseModel):
    """Model-facing arguments for paginating observation memory."""

    include_covered_observations: bool = Field(
        default=False,
        description=(
            "Whether to include observations already cited by active knowledge. "
            "Use true when auditing or revising an existing knowledge boundary."
        ),
    )
    scope: MemoryScope | None = Field(
        default=None,
        description=(
            "Optional scope filter. Omit to list both global and current-project "
            "observations."
        ),
    )
    memory_type: MemoryType | None = Field(
        default=None,
        description="Optional memory-type filter.",
    )
    limit: int = Field(
        default=SYNTHESIS_TOOL_DEFAULT_LIMIT,
        ge=1,
        le=SYNTHESIS_TOOL_MAX_LIMIT,
        description="Maximum number of observations to return in this page.",
    )
    offset: int = Field(
        default=0,
        ge=0,
        le=5000,
        description="Number of observations to skip before returning this page.",
    )


class ListSynthesisKnowledgeArgs(BaseModel):
    """Model-facing arguments for paginating knowledge memory."""

    include_archived_knowledge: bool = Field(
        default=False,
        description="Whether archived knowledge records should be listed.",
    )
    scope: MemoryScope | None = Field(
        default=None,
        description=(
            "Optional scope filter. Omit to list both global and current-project "
            "knowledge."
        ),
    )
    memory_type: MemoryType | None = Field(
        default=None,
        description="Optional memory-type filter.",
    )
    limit: int = Field(
        default=SYNTHESIS_TOOL_DEFAULT_LIMIT,
        ge=1,
        le=SYNTHESIS_TOOL_MAX_LIMIT,
        description="Maximum number of knowledge records to return in this page.",
    )
    offset: int = Field(
        default=0,
        ge=0,
        le=5000,
        description="Number of knowledge records to skip before returning this page.",
    )


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
    """Shrink starting-point detail without dropping trigger observation ids."""
    encoded = pretty_json(context)
    if len(encoded) <= max_chars:
        return context

    for observation in context["starting_observations"]:
        observation["snippet"] = ""
    encoded = pretty_json(context)
    if len(encoded) <= max_chars:
        return context

    for observation in context["starting_observations"]:
        observation["summary"] = _truncate_text(
            observation["summary"],
            SYNTHESIS_CONTEXT_OBSERVATION_CLAMPED_SUMMARY_CHARS,
        )
    encoded = pretty_json(context)
    if len(encoded) <= max_chars:
        return context

    for observation in context["starting_observations"]:
        observation["summary"] = ""
    encoded = pretty_json(context)

    if len(encoded) > max_chars:
        logger.warning(
            "Submitting oversized EvoMemory synthesis context: %d chars for "
            "%d trigger observations after stripping snippets and summaries",
            len(encoded),
            len(context["starting_observations"]),
        )
    return context


def _covered_observation_index(
    knowledge_documents: list[Any],
) -> dict[str, list[str]]:
    covered: dict[str, list[str]] = {}
    for document in knowledge_documents:
        for observation_id in document.supporting_observation_ids:
            covered.setdefault(observation_id, []).append(document.knowledge_id)
    return covered


def _observation_context(
    document: Mapping[str, Any],
    *,
    covered_by: Mapping[str, list[str]],
) -> SynthesisObservationContext:
    observation_id = str(document["observation_id"])
    return {
        "id": observation_id,
        "path": str(document["path"]),
        "memory_type": MemoryType(document["memory_type"]).value,
        "scope": MemoryScope(document["scope"]).value,
        "summary": _truncate_text(
            str(document["summary"]),
            SYNTHESIS_CONTEXT_OBSERVATION_SUMMARY_CHARS,
        ),
        "snippet": _truncate_text(
            document_body(str(document["text"])),
            SYNTHESIS_CONTEXT_OBSERVATION_SNIPPET_CHARS,
        ),
        "covered_by_knowledge_ids": list(covered_by.get(observation_id, [])),
    }


def build_synthesis_context(
    *,
    memory_dir: str | Path,
    project_id: str,
    seed_observation_ids: tuple[str, ...],
    max_chars: int = SYNTHESIS_CONTEXT_MAX_CHARS,
) -> SynthesisContext | None:
    """Build bounded starting context for a pull-based synthesis review."""
    seed_ids = dedupe_ids(seed_observation_ids)

    active_knowledge_documents = knowledge_search_documents(
        memory_dir=memory_dir,
        project_id=project_id,
        status=KnowledgeStatus.ACTIVE,
    )
    archived_knowledge_documents = knowledge_search_documents(
        memory_dir=memory_dir,
        project_id=project_id,
        status=KnowledgeStatus.ARCHIVED,
    )
    observation_documents = candidate_observation_documents(
        memory_dir=memory_dir,
        project_id=project_id,
    )
    covered_by = _covered_observation_index(active_knowledge_documents)
    covered_observation_count = sum(
        1 for document in observation_documents if document.observation_id in covered_by
    )

    starting_observations: list[SynthesisObservationContext] = []
    for observation_id in seed_ids:
        document = read_observation_file(
            memory_dir=memory_dir,
            project_id=project_id,
            observation_id=observation_id,
        )
        if document is None:
            continue
        starting_observations.append(
            _observation_context(document, covered_by=covered_by)
        )

    uncovered_observation_count = len(observation_documents) - covered_observation_count
    if (
        not starting_observations
        and uncovered_observation_count == 0
        and not active_knowledge_documents
        and not archived_knowledge_documents
    ):
        return None

    context: SynthesisContext = {
        "project_id": project_id,
        "review_goal": (
            "Use trigger observations as starting points, then pull from "
            "visible EvoMemory with search/list/read tools until the synthesis "
            "boundary is clear."
        ),
        "trigger_observation_ids": list(seed_ids),
        "starting_observations": starting_observations,
        "memory_inventory": {
            "active_knowledge_count": len(active_knowledge_documents),
            "archived_knowledge_count": len(archived_knowledge_documents),
            "observation_count": len(observation_documents),
            "uncovered_observation_count": uncovered_observation_count,
            "covered_observation_count": covered_observation_count,
            "seed_observation_count": len(seed_ids),
        },
    }
    return _shrink_synthesis_context(context, max_chars=max_chars)


def _tool_error(message: str) -> str:
    return json.dumps({"error": message}, ensure_ascii=False, sort_keys=True)


def _page_payload(
    *,
    results: list[dict[str, object]],
    total: int,
    limit: int,
    offset: int,
) -> dict[str, object]:
    next_offset = offset + len(results)
    return {
        "results": results,
        "total": total,
        "limit": limit,
        "offset": offset,
        "next_offset": next_offset if next_offset < total else None,
    }


def _synthesis_memory_tools(*, memory_dir: str | Path) -> list[BaseTool]:
    """Build read-side memory tools for the synthesis worker graph."""
    worker_memory_dir = Path(memory_dir).expanduser()

    def _project_id() -> str | None:
        return config_str(current_configurable(), "evomemory_project_id")

    def _search_memory(
        query: str,
        mode: MemorySearchMode = MemorySearchMode.RANKED,
        memory_level: MemoryLevelFilter = MemoryLevelFilter.ANY,
        scope: MemoryScope | None = None,
        memory_type: MemoryType | None = None,
        include_archived_knowledge: bool = False,
        include_covered_observations: bool = False,
        limit: int = SYNTHESIS_TOOL_DEFAULT_LIMIT,
        offset: int = 0,
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
            include_covered_observations=include_covered_observations,
            limit=limit,
            offset=offset,
            mode=mode,
        )
        return json.dumps(
            {
                "results": results,
                "limit": limit,
                "offset": offset,
                "next_offset": offset + len(results) if len(results) == limit else None,
            },
            ensure_ascii=False,
            sort_keys=True,
        )

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

    def _list_observations(
        include_covered_observations: bool = False,
        scope: MemoryScope | None = None,
        memory_type: MemoryType | None = None,
        limit: int = SYNTHESIS_TOOL_DEFAULT_LIMIT,
        offset: int = 0,
    ) -> str:
        project_id = _project_id()
        if project_id is None:
            return _tool_error("EvoMemory synthesizer missing project id.")
        active_knowledge = knowledge_search_documents(
            memory_dir=worker_memory_dir,
            project_id=project_id,
            status=KnowledgeStatus.ACTIVE,
        )
        covered_by = _covered_observation_index(active_knowledge)
        observations = candidate_observation_documents(
            memory_dir=worker_memory_dir,
            project_id=project_id,
            scope=scope,
            memory_type=memory_type,
        )
        if not include_covered_observations:
            observations = [
                document
                for document in observations
                if document.observation_id not in covered_by
            ]
        page = observations[offset : offset + limit]
        results: list[dict[str, object]] = [
            {
                "memory_id": document.observation_id,
                "path": document.path,
                "memory_type": document.memory_type,
                "scope": document.scope,
                "summary": document.summary,
                "covered_by_knowledge_ids": list(
                    covered_by.get(document.observation_id, [])
                ),
            }
            for document in page
        ]
        return json.dumps(
            _page_payload(
                results=results,
                total=len(observations),
                limit=limit,
                offset=offset,
            ),
            ensure_ascii=False,
            sort_keys=True,
        )

    def _list_knowledge(
        include_archived_knowledge: bool = False,
        scope: MemoryScope | None = None,
        memory_type: MemoryType | None = None,
        limit: int = SYNTHESIS_TOOL_DEFAULT_LIMIT,
        offset: int = 0,
    ) -> str:
        project_id = _project_id()
        if project_id is None:
            return _tool_error("EvoMemory synthesizer missing project id.")
        knowledge = knowledge_search_documents(
            memory_dir=worker_memory_dir,
            project_id=project_id,
            scope=scope,
            memory_type=memory_type,
            status=None if include_archived_knowledge else KnowledgeStatus.ACTIVE,
        )
        page = knowledge[offset : offset + limit]
        results: list[dict[str, object]] = [
            {
                "memory_id": document.knowledge_id,
                "path": document.path,
                "memory_type": document.memory_type,
                "scope": document.scope,
                "summary": document.summary,
                "status": document.status,
                "supporting_observation_ids": list(document.supporting_observation_ids),
            }
            for document in page
        ]
        return json.dumps(
            _page_payload(
                results=results,
                total=len(knowledge),
                limit=limit,
                offset=offset,
            ),
            ensure_ascii=False,
            sort_keys=True,
        )

    return [
        StructuredTool.from_function(
            func=_search_memory,
            name="search_memory",
            description=(
                "Search EvoMemory from the synthesis worker. Results are one "
                "page; increase offset to keep exploring beyond the first page. "
                "Use memory_level=knowledge before creating or updating "
                "knowledge, and memory_level=observation when related raw "
                "evidence could affect the synthesis decision. Set "
                "include_covered_observations=true when auditing an existing "
                "knowledge boundary. Read promising K-* or O-* hits with "
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
        StructuredTool.from_function(
            func=_list_observations,
            name="list_observations",
            description=(
                "List visible observation memory as a paginated index. By "
                "default this lists uncovered observations, which are not yet "
                "cited by active knowledge. Increase offset to inspect more "
                "observations, or set include_covered_observations=true to "
                "audit existing knowledge support."
            ),
            args_schema=ListSynthesisObservationsArgs,
            infer_schema=False,
        ),
        StructuredTool.from_function(
            func=_list_knowledge,
            name="list_knowledge",
            description=(
                "List visible synthesized knowledge as a paginated index. Use "
                "this to inspect existing K-* records before creating new "
                "knowledge, and increase offset to keep exploring."
            ),
            args_schema=ListSynthesisKnowledgeArgs,
            infer_schema=False,
        ),
    ]


def _synthesis_system_prompt() -> str:
    return "\n\n".join(
        [
            "You are the EvoMemory synthesis agent.",
            (
                "Your job is to explore EvoMemory and decide whether durable "
                "observations justify compact knowledge records. Trigger "
                "observations are starting points, not the synthesis boundary. "
                "You choose the boundary by pulling memory with tools."
            ),
            (
                "Use `list_observations` to page through uncovered observations "
                "or audit covered ones. Use `list_knowledge` and "
                "`search_memory(memory_level=knowledge)` before creating new "
                "knowledge so you can update or archive existing K-* records "
                "instead of duplicating them. Use "
                "`search_memory(memory_level=observation)` for content-based "
                "neighborhoods, conflicts, and older supporting evidence. "
                "Increase offset and search alternate phrasings when the first "
                "page is not enough."
            ),
            (
                "Use `read_memory` to read every O-* observation you cite and "
                "every K-* record you update or archive. In the final structured "
                "response, include the queries you explored, the exact memory "
                "IDs you read, and a short boundary_rationale. Create/update "
                "decisions citing O-* IDs that were not read are ignored."
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
            "EvoMemory is ready for synthesis review. Decide whether to create, "
            "update, archive, or skip synthesized knowledge after exploring the "
            "memory base.",
            f"Trigger: {trigger}",
            (
                "The context below is only a starting point. Use memory tools "
                "to pull nearby observations and existing knowledge until you "
                "can explain the synthesis boundary. Absence from this context "
                "does not mean absence from memory."
            ),
            "Starting context JSON:",
            pretty_json(context),
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
    read_memory_ids = set(dedupe_ids(review.read_memory_ids))
    for decision in review.decisions:
        try:
            match decision:
                case SynthesisSkipDecision():
                    continue
                case SynthesisCreateDecision():
                    unread = [
                        observation_id
                        for observation_id in decision.supporting_observation_ids
                        if observation_id not in read_memory_ids
                    ]
                    if unread:
                        logger.warning(
                            "Skipping EvoMemory synthesis CREATE because support "
                            "was not read: %s",
                            ", ".join(unread),
                        )
                        continue
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
                        logger.warning(
                            "Skipping EvoMemory synthesis UPDATE for missing "
                            "knowledge %s",
                            decision.target_knowledge_id,
                        )
                        continue
                    unread = [
                        observation_id
                        for observation_id in decision.supporting_observation_ids
                        if observation_id not in read_memory_ids
                    ]
                    if unread:
                        logger.warning(
                            "Skipping EvoMemory synthesis UPDATE for %s because "
                            "support was not read: %s",
                            decision.target_knowledge_id,
                            ", ".join(unread),
                        )
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
        project_id = config_str(current_configurable(), "evomemory_project_id")
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
    return f"evomemory-synth:{short_hash(project_id + stable_json(context))}"


def _synthesis_context_digest(context: SynthesisContext) -> str:
    return short_hash(stable_json(context))


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
    """Extract a LangGraph run id from the SDK response."""
    run_id = run.get("run_id", "").strip()
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
        return _SynthesisRunOutcome.FAILED

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
    """Submit synthesis runs until one succeeds, aborts, or attempts run out.

    A ``success`` terminal status — including a deliberate skip with no writes —
    stops immediately and is never retried. Confirmed terminal failures
    (error/timeout/interrupted) are retried up to ``max_attempts``. Abandoned
    polling is not retried and does not release the active context because the
    previous run may still be live and may still write knowledge.
    """
    release_project_id, context_digest = active_key
    release_context = True
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
            if outcome is _SynthesisRunOutcome.ABORTED:
                release_context = False
                logger.warning(
                    "Abandoning EvoMemory synthesis retry for project %s; "
                    "previous run state is unknown",
                    project_id,
                )
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
        if release_context:
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
    """Submit a background synthesis run when there is visible memory to inspect."""
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

    try:
        before_outputs = snapshot_memory_outputs(memory_dir)
        mark_synthesis_started(
            project_id=project_id,
            context_digest=context_digest,
            memory_dir=memory_dir,
            before_outputs=before_outputs,
        )
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
