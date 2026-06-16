"""Dedicated async agent for observation-to-knowledge synthesis."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
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
from langgraph.config import get_config
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from .. import paths as _paths
from .knowledge import (
    SYNTHESIS_AGENT_NAME,
    archive_knowledge_file,
    knowledge_search_documents,
    read_knowledge_file,
    record_knowledge_file,
)
from .observations import candidate_observation_documents
from .types import (
    KnowledgeRecordResult,
    KnowledgeStatus,
    MemoryScope,
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
SYNTHESIS_CONTEXT_OBSERVATION_LIMIT = 16
SYNTHESIS_CONTEXT_KNOWLEDGE_LIMIT = 12
SYNTHESIS_CONTEXT_MAX_CHARS = 32_000
_SYNTHESIS_TERMINAL_STATUSES = frozenset({"success", "error", "timeout", "interrupted"})
_SYNTHESIS_POLL_INTERVAL_SECONDS = 1.0
_SYNTHESIS_MAX_POLL_FAILURES = 3
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
    """Observation payload included in a bounded synthesis prompt."""

    id: str
    path: str
    memory_type: str
    scope: str
    summary: str
    body: str


class SynthesisKnowledgeContext(TypedDict):
    """Knowledge payload included in a bounded synthesis prompt."""

    id: str
    path: str
    memory_type: str
    scope: str
    summary: str
    supporting_observation_ids: list[str]
    body: str


class SynthesisContext(TypedDict):
    """Bounded observation/knowledge context submitted to synthesis."""

    project_id: str
    observations: list[SynthesisObservationContext]
    existing_knowledge: list[SynthesisKnowledgeContext]
    covered_observation_ids: list[str]


@dataclass(frozen=True)
class SynthesisLaunchArgs:
    """Arguments needed to submit one background synthesis run."""

    memory_dir: str | Path
    project_id: str
    trigger: str


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


def build_synthesis_context(
    *,
    memory_dir: str | Path,
    project_id: str,
    observation_limit: int = SYNTHESIS_CONTEXT_OBSERVATION_LIMIT,
    knowledge_limit: int = SYNTHESIS_CONTEXT_KNOWLEDGE_LIMIT,
    max_chars: int = SYNTHESIS_CONTEXT_MAX_CHARS,
) -> SynthesisContext | None:
    """Build bounded observation/knowledge context for the synthesis agent."""
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
    observation_documents = [
        document
        for document in candidate_observation_documents(
            memory_dir=memory_dir,
            project_id=project_id,
        )
        if document.observation_id not in covered_observation_ids
    ]
    if not observation_documents:
        return None

    selected_observations = observation_documents[:observation_limit]
    selected_knowledge = knowledge_documents[:knowledge_limit]
    observations: list[SynthesisObservationContext] = [
        {
            "id": document.observation_id,
            "path": document.path,
            "memory_type": document.memory_type.value,
            "scope": document.scope.value,
            "summary": document.summary,
            "body": _truncate_text(document.body, 1800),
        }
        for document in selected_observations
    ]
    existing_knowledge: list[SynthesisKnowledgeContext] = [
        {
            "id": document.knowledge_id,
            "path": document.path,
            "memory_type": document.memory_type.value,
            "scope": document.scope.value,
            "summary": document.summary,
            "supporting_observation_ids": list(document.supporting_observation_ids),
            "body": _truncate_text(document.body, 1200),
        }
        for document in selected_knowledge
    ]
    context: SynthesisContext = {
        "project_id": project_id,
        "observations": observations,
        "existing_knowledge": existing_knowledge,
        "covered_observation_ids": sorted(covered_observation_ids),
    }
    encoded = _pretty_json(context)
    if len(encoded) > max_chars:
        context["observations"] = observations[: max(1, observation_limit // 2)]
        encoded = _pretty_json(context)
    if len(encoded) > max_chars:
        context["existing_knowledge"] = []
    return context


def _synthesis_system_prompt() -> str:
    return "\n\n".join(
        [
            "You are the EvoMemory synthesis agent.",
            (
                "Your job is to convert durable observations into compact "
                "knowledge records. Operate only on the observations and "
                "existing knowledge provided in the prompt. Do not continue the "
                "user task, infer from missing trajectory context, or create "
                "memories from vibes."
            ),
            (
                "Knowledge is an abstraction over evidence. Create or update "
                "knowledge only when multiple observations or a single strong "
                "observation support a reusable rule, fact, procedure, failure "
                "mode, or project convention. Every create/update decision must "
                "cite the minimal supporting O-* IDs."
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
            "Review this bounded EvoMemory context and decide whether synthesis "
            "maintenance is warranted.",
            f"Trigger: {trigger}",
            "Context JSON:",
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
    from deepagents.backends import FilesystemBackend

    from ..EvoScientist import _ensure_auxiliary_chat_model

    worker_memory_dir = Path(
        _paths.MEMORIES_DIR if memory_dir is None else memory_dir
    ).expanduser()
    agent = create_deep_agent(
        name=SYNTHESIS_AGENT_NAME,
        model=_ensure_auxiliary_chat_model(),
        system_prompt=_synthesis_system_prompt(),
        tools=[],
        backend=FilesystemBackend(root_dir=str(worker_memory_dir), virtual_mode=True),
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


def _watch_synthesis_run_sync(
    *,
    url: str,
    thread_id: str,
    run_id: str,
    active_key: tuple[str, str] | None = None,
) -> None:
    from langgraph_sdk import get_sync_client

    failures = 0
    confirmed_finished = False
    client = None
    try:
        client = get_sync_client(url=url, headers={"x-auth-scheme": "langsmith"})
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
                    return
                time.sleep(_SYNTHESIS_POLL_INTERVAL_SECONDS)
                continue
            if _status_from_run_response(run) in _SYNTHESIS_TERMINAL_STATUSES:
                confirmed_finished = True
                return
            time.sleep(_SYNTHESIS_POLL_INTERVAL_SECONDS)
    finally:
        if confirmed_finished and client is not None:
            _delete_synthesis_thread(client, thread_id)
        if active_key is not None:
            project_id, context_digest = active_key
            _release_synthesis_context(
                project_id=project_id,
                context_digest=context_digest,
            )


def _spawn_synthesis_status_thread(
    *,
    url: str,
    thread_id: str,
    run_id: str,
    active_key: tuple[str, str] | None = None,
) -> None:
    thread = threading.Thread(
        target=_watch_synthesis_run_sync,
        kwargs={
            "url": url,
            "thread_id": thread_id,
            "run_id": run_id,
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
    trigger: str,
) -> None:
    """Submit a background synthesis run if uncovered observations exist."""
    from langgraph_sdk import get_sync_client

    from ..langgraph_dev.manager import is_langgraph_dev_running

    context = build_synthesis_context(memory_dir=memory_dir, project_id=project_id)
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
    active_key: tuple[str, str] | None = (project_id, context_digest)
    try:
        client = get_sync_client(url=url, headers={"x-auth-scheme": "langsmith"})
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
        if run_id := _run_id_from_response(run):
            _spawn_synthesis_status_thread(
                url=url,
                thread_id=thread_id,
                run_id=run_id,
                active_key=active_key,
            )
            active_key = None
    finally:
        if active_key is not None:
            _release_synthesis_context(
                project_id=active_key[0],
                context_digest=active_key[1],
            )


async def _alaunch_synthesis_worker(
    *,
    memory_dir: str | Path,
    project_id: str,
    trigger: str,
) -> None:
    """Submit a background synthesis run without blocking the live agent."""
    from langgraph_sdk import get_client

    from ..langgraph_dev.manager import is_langgraph_dev_running

    context = await asyncio.to_thread(
        build_synthesis_context,
        memory_dir=memory_dir,
        project_id=project_id,
    )
    if context is None:
        return

    url = _synthesis_worker_url()
    if not await asyncio.to_thread(is_langgraph_dev_running, base_url=url):
        logger.info("Skipping EvoMemory synthesis launch; LangGraph dev is unavailable")
        return

    context_digest = _synthesis_context_digest(context)
    if not await asyncio.to_thread(
        _claim_synthesis_context,
        project_id=project_id,
        context_digest=context_digest,
    ):
        logger.debug(
            "Skipping duplicate EvoMemory synthesis launch for context %s",
            context_digest,
        )
        return

    before_outputs = await asyncio.to_thread(snapshot_memory_outputs, memory_dir)
    await asyncio.to_thread(
        mark_synthesis_started,
        project_id=project_id,
        context_digest=context_digest,
        memory_dir=memory_dir,
        before_outputs=before_outputs,
    )
    active_key: tuple[str, str] | None = (project_id, context_digest)
    try:
        client = get_client(url=url, headers={"x-auth-scheme": "langsmith"})
        thread = await client.threads.create(graph_id=SYNTHESIS_GRAPH_ID)
        thread_id = str(thread["thread_id"])
        payload = _synthesis_run_kwargs(
            project_id=project_id,
            context=context,
            trigger=trigger,
        )
        run = await client.runs.create(
            thread_id=thread_id,
            assistant_id=payload["assistant_id"],
            input=payload["input"],
            metadata=payload["metadata"],
            config=payload["config"],
        )
        if run_id := _run_id_from_response(run):
            _spawn_synthesis_status_thread(
                url=url,
                thread_id=thread_id,
                run_id=run_id,
                active_key=active_key,
            )
            active_key = None
    finally:
        if active_key is not None:
            await asyncio.to_thread(
                _release_synthesis_context,
                project_id=active_key[0],
                context_digest=active_key[1],
            )
