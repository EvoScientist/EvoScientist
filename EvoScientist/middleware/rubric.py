"""Acceptance grading shared by the main agent and the scheduler graph.

A run whose input carries ``rubric`` is graded when the agent stops and sent
back to the model until the grader is satisfied or the pass limit is reached.
On the main agent the run's config must also carry the ``rubric_run`` marker,
whose ``mode`` says where the checklist comes from: the input, the checklists
``/rubric`` saved on the thread, or a grading the thread deferred while
background work was running. Without a checklist the middleware does nothing
and builds no grader model.
"""

from __future__ import annotations

import asyncio
import logging
import warnings
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Annotated, Any, NotRequired

import httpx
from deepagents.middleware.rubric import (
    GRADER_SYSTEM_PROMPT,
    RUBRIC_GRADER_MESSAGE_SOURCE,
    GraderResponse,
    RubricMiddleware,
    RubricState,
)
from langchain.agents import create_agent
from langchain.agents.middleware.types import (
    AgentMiddleware,
    PrivateStateAttr,
    hook_config,
)
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from langchain_core.messages import AIMessage
from langgraph.errors import GraphBubbleUp
from langgraph_sdk.errors import NotFoundError

from ..message_meta import message_source
from ..rubric_session import thread_goal

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol
    from deepagents.middleware.rubric import RubricEvaluation
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


# Read-only slice of the filesystem tools handed to the grader. A rubric names
# its deliverables, so reading them is enough; ``grep``/``glob`` invited
# whole-workspace scans (15s timeouts per call on large workspaces).
GRADER_TOOLS = ("ls", "read_file")

# Structured-output strategy per OpenRouter model family. langchain picks the
# grader's strategy from the model profile plus a model-name regex table, and
# OpenRouter breaks each family differently: Gemini tool schemas lose the
# criteria ``oneOf`` (every entry comes back null), Anthropic JSON mode returns
# non-JSON, OpenAI's strict JSON mode rejects that ``oneOf`` with a 400.
# Verified live 2026-09-04 and 2026-09-26. Passed explicitly because a profile
# pin loses to the name regex (``anthropic/claude-fable-5``).
_OPENROUTER_GRADER_STRATEGY: dict[str, type[ProviderStrategy] | type[ToolStrategy]] = {
    "google/": ProviderStrategy,
    "anthropic/": ToolStrategy,
    "openai/": ToolStrategy,
}

# Model calls one grader attempt may spend before the run fails closed with
# ``grader_error``. Without it a parse-error retry loop inherits the graph's
# recursion limit and spins for minutes. Sized for a 3-5 bullet rubric over a
# few files: ``ls`` + one ``read_file`` per file + the verdict call.
GRADER_MAX_CALLS = 12

# Grading passes per run: the main agent, and the cheaper unattended scheduler.
MAIN_RUBRIC_MAX_ITERATIONS = 3
SCHEDULER_RUBRIC_MAX_ITERATIONS = 2

# Background work still counts as unfinished only while the live status is one
# of these; anything else, or a status that cannot be read, never holds a
# grading back.
ACTIVE_STATUSES = frozenset({"pending", "running"})
# One budget for all the status reads before a grading pass.
DEFER_QUERY_TIMEOUT_SECONDS = 10.0


async def live_task_status(thread_id: str, run_id: str) -> str:
    """Live status of a background sub-agent run on the dev server."""
    from EvoScientist.langgraph_dev import sdk

    async with sdk.langgraph_async_client_scoped(
        sdk.configured_langgraph_dev_url()
    ) as client:
        run = await client.runs.get(thread_id, run_id)
    return str(run.get("status") or "")


def _connection_refused(exc: BaseException) -> bool:
    """Whether an error chain bottoms out in a refused connection: nobody listening.

    httpx wraps the OS error two levels deep, and the async client folds the
    attempts for every address under one more ``OSError``.
    """
    pending = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, ConnectionRefusedError):
            return True
        pending.extend(getattr(current, "exceptions", None) or ())
        nested = current.__cause__ or current.__context__
        if nested is not None:
            pending.append(nested)
    return False


async def _probe_dev_server(
    url: str, *, transport: httpx.AsyncBaseTransport | None = None
) -> tuple[bool, str | None]:
    """``(answers, reason)``: a refused connection has no reason to report.

    Async like the status reads: the sync client keeps only the last address's
    error, so a refusal on an earlier address would hide behind a later failure.
    """
    try:
        # An explicit transport keeps env proxies off this loopback probe.
        async with httpx.AsyncClient(
            transport=transport or httpx.AsyncHTTPTransport(retries=0),
            timeout=1.0,
        ) as client:
            status = (await client.get(f"{url}/ok")).status_code
    except (httpx.TransportError, OSError) as exc:
        if _connection_refused(exc):
            return False, None
        name = type(exc).__name__
        return False, f"{name}: {exc}" if str(exc) else name
    return status == 200, None if status == 200 else f"HTTP {status}"


async def dev_server_reachable() -> bool:
    """Whether task status can be read at all: the server was started and answers."""
    from EvoScientist.langgraph_dev import manager, sdk

    if not manager.is_async_subagents_available():
        return False
    ok, reason = await _probe_dev_server(sdk.configured_langgraph_dev_url())
    if reason:
        logger.warning(
            "rubric: dev server health check failed (%s); grading without waiting",
            reason,
        )
    return ok


# Appended to the upstream grader prompt. Without it the grader either fails a
# file criterion the transcript never mentions, or walks the whole workspace
# looking for a missing file until the call budget ends the pass.
GRADER_FILE_CHECK_PROMPT = (
    "\n\nYou have read-only `ls` and `read_file` tools rooted at the workspace (`/`). "
    "The transcript is bounded and can omit or shorten earlier messages, so when a "
    "criterion is about a file, check that file on disk before you decide it either "
    "way: do not fail it only because the transcript does not show the file, and do "
    "not pass it only because the transcript says it was written. Open only files "
    "that a criterion or the transcript names. A file that is not at the named path "
    "is missing: fail that criterion and return your verdict without looking through "
    "other files or directories."
)

# OpenRouter ids for which neither strategy yields a verdict (re-probed 2026-09-22,
# sonnet-5.5 probed 2026-09-29): the model rejects forced ``tool_choice`` and JSON
# mode rejects the grader schema's ``oneOf``. Exact ids, not families:
# ``anthropic/claude-fable-5`` and the native ``claude-fable-5-1`` /
# ``claude-opus-5-5`` grade fine.
_OPENROUTER_UNGRADABLE_IDS = (
    "anthropic/claude-fable-5.1",
    "anthropic/claude-opus-5.5",
    "anthropic/claude-sonnet-5.5",
)


def _warn_if_grader_unsupported(model: BaseChatModel) -> None:
    from EvoScientist.llm.errors import _provider_from_model

    if _provider_from_model(model) != "openrouter":
        return
    model_id = (getattr(model, "model_name", None) or "").lower()
    if model_id in _OPENROUTER_UNGRADABLE_IDS:
        logger.warning(
            "rubric: grader model %s via OpenRouter cannot return "
            "structured verdicts (this route rejects forced tool_choice and its "
            "JSON mode rejects the grader schema); rubric runs will end in "
            "grader_error. Use the native anthropic provider for this model, or "
            "set auxiliary_model to another model (claude-fable-5, claude-sonnet-5, "
            "Haiku, GPT and Gemini all grade through OpenRouter).",
            model_id,
        )


def _grader_strategy(model: BaseChatModel) -> ProviderStrategy | ToolStrategy | None:
    """Explicit grader strategy on OpenRouter routes; ``None`` defers to langchain."""
    from EvoScientist.llm.errors import _provider_from_model

    if _provider_from_model(model) != "openrouter":
        return None
    model_id = (getattr(model, "model_name", None) or "").lower()
    for prefix, strategy in _OPENROUTER_GRADER_STRATEGY.items():
        if model_id.startswith(prefix):
            return strategy(GraderResponse)
    return None


# How a marked run picks its checklist: the input text, the checklists saved
# on the thread, or only a grading the thread already deferred.
RUBRIC_MODES = ("input", "thread", "followup")


def rubric_run_config() -> dict[str, Any] | None:
    """The ``rubric_run`` dict of the current run, or ``None`` when unmarked."""
    from langgraph.config import get_config

    from EvoScientist.backends import RUBRIC_RUN_KEY

    try:
        configurable = get_config().get("configurable") or {}
    except RuntimeError:
        return None
    marker = configurable.get(RUBRIC_RUN_KEY)
    if isinstance(marker, dict) and marker.get("mode") in RUBRIC_MODES:
        return marker
    return None


class EvoRubricState(RubricState):
    """Checklists saved on the thread and the grading that has no verdict yet."""

    _sticky_rubric: NotRequired[Annotated[dict[str, Any] | None, PrivateStateAttr]]
    _next_rubric: NotRequired[Annotated[dict[str, Any] | None, PrivateStateAttr]]
    # Highest applied operation sequence number per client session.
    _rubric_applied: NotRequired[Annotated[dict[str, int], PrivateStateAttr]]
    # {"source", "id", "start", "deferred", "skip", "skip_runs"} while a grading is open.
    _rubric_grading: NotRequired[Annotated[dict[str, Any] | None, PrivateStateAttr]]


def _checklist_id(operation: Mapping[str, Any]) -> str:
    return f"{operation['session']}:{operation['seq']}"


def _apply_operations(state: Any, changes: list[Any]) -> dict[str, Any]:
    """State update for the ``/rubric`` operations not applied yet, in order."""
    applied = dict(state.get("_rubric_applied") or {})
    update: dict[str, Any] = {}
    for operation in changes:
        if not isinstance(operation, Mapping):
            continue
        session, seq = operation.get("session"), operation.get("seq")
        if not isinstance(session, str) or not isinstance(seq, int):
            continue
        if seq <= applied.get(session, 0):
            continue
        for slot in ("sticky", "next"):
            if slot in operation:
                text = operation[slot]
                update[f"_{slot}_rubric"] = (
                    {"text": text, "id": _checklist_id(operation)} if text else None
                )
        applied[session] = seq
    if applied != (state.get("_rubric_applied") or {}):
        update["_rubric_applied"] = applied
    return update


def _message_index(messages: list, message_id: Any) -> int | None:
    if not message_id:
        return None
    for index, message in enumerate(messages):
        if getattr(message, "id", None) == message_id:
            return index
    return None


def _current_thread_id() -> str | None:
    from langgraph.config import get_config

    try:
        return (get_config().get("configurable") or {}).get("thread_id")
    except RuntimeError:
        return None


class EvoRubricMiddleware(RubricMiddleware):
    """``RubricMiddleware`` adapted to checklists that belong to a run or a thread.

    A ``"input"`` run grades the checklist in its input and starts a new
    attempt; a ``"thread"`` run applies the ``/rubric`` operations it carries,
    picks ``next`` before ``sticky`` and starts a new attempt; a
    ``"followup"`` run only finishes a grading the thread deferred, keeping
    its pass count. A final verdict clears the run's checklist, and the
    ``next`` checklist only when that grading used it. With
    ``require_run_marker`` an unmarked invocation drops an open grading,
    unless it is deferred and still waits for its own turn.
    ``_ensure_grader`` mirrors upstream except for the model and
    ``response_format``: a bare ``GraderResponse`` there lets langchain choose
    the strategy, which is wrong on OpenRouter (see
    ``_OPENROUTER_GRADER_STRATEGY``) and native DeepSeek (see
    ``structured_output_for``).
    """

    state_schema = EvoRubricState
    _warned_unsupported = False

    def __init__(
        self,
        *,
        require_run_marker: bool = False,
        prepare_messages: Callable[[list], list] | None = None,
        task_status: Callable[[str, str], Awaitable[str]] | None = None,
        process_status: Callable[[str], str] | None = None,
        server_available: Callable[[], Awaitable[bool]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._require_run_marker = require_run_marker
        # Applied here, not by upstream: a grading that spans turns is read
        # from its first message, which upstream's hook cannot see.
        self._prepare_run_messages = prepare_messages
        self._task_status = task_status
        self._process_status = process_status
        self._server_available = server_available

    def _warn_unsupported_once(self, state: Any) -> None:
        if self._warned_unsupported or not state.get("rubric"):
            return
        self._warned_unsupported = True
        _warn_if_grader_unsupported(self._model)

    def _ungraded_run_update(self, state: Any) -> tuple[bool, dict[str, Any] | None]:
        """Whether this run must not be graded, and the update that drops its rubric."""
        if not self._require_run_marker or rubric_run_config() is not None:
            return False, None
        update: dict[str, Any] = {}
        if state.get("rubric"):
            logger.debug(
                "rubric: dropped from a run whose config carries no rubric_run "
                "marker; the run is not graded"
            )
            update["rubric"] = None
        grading = state.get("_rubric_grading")
        # A deferred grading waits for its own turn; this run only passes by.
        if grading and not grading.get("deferred"):
            update["_rubric_grading"] = None
        return True, (update or None)

    # -- start of a run -----------------------------------------------------

    def _start_of_run(self, state: Any) -> dict[str, Any] | None:
        if not self._require_run_marker:
            self._warn_unsupported_once(state)
            return super()._reset_for_new_rubric({**state, "_active_rubric": None})
        config = rubric_run_config()
        if config is None:
            return self._ungraded_run_update(state)[1]
        update = _apply_operations(state, list(config.get("changes") or []))
        view = {**state, **update}
        if state.get("_goal") and thread_goal(view) is None:
            # A checklist command replaced the criteria the goal stood on.
            update["_goal"] = None
            self._goal_event("cleared", note="The checklist was replaced.")
        mode = config.get("mode")
        if mode == "input":
            if state.get("_rubric_grading"):
                update["_rubric_grading"] = None
            update.update(
                super()._reset_for_new_rubric({**state, "_active_rubric": None}) or {}
            )
            update["_rubric_evaluations"] = []
        elif mode == "thread":
            update.update(self._select_for_thread(state, view))
        else:
            update.update(self._select_for_followup(view))
        self._warn_unsupported_once({**view, **update})
        return update or None

    def _select_for_thread(self, state: Any, view: Any) -> dict[str, Any]:
        view = self._without_idle_goal_checklist(view)
        source = next(
            (slot for slot in ("next", "sticky") if view.get(f"_{slot}_rubric")),
            None,
        )
        if source is None:
            update: dict[str, Any] = {}
            if state.get("rubric"):
                update["rubric"] = None
            if state.get("_rubric_grading"):
                update["_rubric_grading"] = None
            return update
        picked = view[f"_{source}_rubric"]
        update = {"rubric": picked["text"]}
        # Reset against the picked text: upstream reads ``rubric`` from state,
        # where it is not written yet.
        update.update(
            super()._reset_for_new_rubric(
                {**state, "rubric": picked["text"], "_active_rubric": None}
            )
            or {}
        )
        # A new attempt: the previous grading's passes are not this one's.
        update["_rubric_evaluations"] = []
        messages = list(state.get("messages") or [])
        current = getattr(messages[-1], "id", None) if messages else None
        existing = state.get("_rubric_grading") or None
        if (
            existing
            and existing.get("source") == source
            and existing.get("id") == picked["id"]
        ):
            start = existing.get("start")
            if _message_index(messages, start) is None:
                start = current
            grading = {**existing, "start": start}
        else:
            grading = {
                "source": source,
                "id": picked["id"],
                "start": current,
                "deferred": False,
                # Sub-agent runs already going belong to earlier work.
                "skip_runs": self._active_task_runs(state),
            }
        grading["skip"] = self._running_own_processes(state)
        update["_rubric_grading"] = grading
        return update

    def _select_for_followup(self, view: Any) -> dict[str, Any]:
        update: dict[str, Any] = {}
        grading = view.get("_rubric_grading") or None
        if grading:
            slot = view.get(f"_{grading.get('source')}_rubric")
            in_place = bool(slot and slot.get("id") == grading.get("id"))
            if in_place and grading.get("deferred"):
                update["rubric"] = slot["text"]
                return update
            if not in_place:
                update["_rubric_grading"] = None
        if view.get("rubric"):
            update["rubric"] = None
        return update

    @staticmethod
    def _active_task_runs(state: Any) -> list[str]:
        """Run ids of the sub-agent tasks the thread records as unfinished."""
        return [
            task["run_id"]
            for task in (state.get("async_tasks") or {}).values()
            if task.get("status") in ACTIVE_STATUSES and task.get("run_id")
        ]

    # -- the thread's goal ----------------------------------------------------

    @staticmethod
    def _goal_of_grading(state: Any) -> dict[str, Any] | None:
        """The thread's goal, when the grading record is of its criteria."""
        goal = thread_goal(state)
        if goal is None:
            return None
        grading = state.get("_rubric_grading") or {}
        if grading.get("source") != "sticky" or grading.get("id") != goal["rubric_id"]:
            return None
        return goal

    def _graded_goal(self, state: Any) -> dict[str, Any] | None:
        """The thread's goal, when this run grades its criteria.

        The run's own marker decides: a turn parked mid-way can be continued
        by a run marked otherwise, and the start hook that would drop the
        grading record does not run again.
        """
        from .goal import goal_in_force

        return self._goal_of_grading(state) if goal_in_force() else None

    @staticmethod
    def _without_idle_goal_checklist(view: Any) -> Any:
        """A paused or completed goal's criteria are not graded."""
        goal = thread_goal(view)
        if goal is not None and goal.get("status") in ("paused", "complete"):
            return {**view, "_sticky_rubric": None}
        return view

    @staticmethod
    def _goal_event(change: str, **fields: Any) -> None:
        from langgraph.config import get_stream_writer

        from .goal import emit_goal_event

        try:
            writer = get_stream_writer()
        except RuntimeError:
            return
        emit_goal_event(writer, change, **fields)

    def _goal_ungraded_update(self, state: Any) -> dict[str, Any] | None:
        """The update that ends this run ungraded because of the thread's goal.

        The grading on the thread is of the goal's criteria, and either the
        agent said it needs the user (grading would only push it back), or
        goals do not apply on this run (it continues a turn that was parked).
        """
        from .goal import goal_in_force

        goal = self._goal_of_grading(state)
        if goal is None:
            return None
        if goal_in_force() and goal.get("status") != "blocked":
            return None
        return {"rubric": None, "_rubric_grading": None}

    def _goal_verdict_update(self, state: Any, result: str) -> dict[str, Any]:
        """What a final verdict does to the thread's goal."""
        from .goal import goal_in_force

        goal = self._graded_goal(state)
        if goal is None:
            # Another checklist's verdict: evidence the agent staged in this
            # turn was about this turn, so it does not wait for a later one.
            other = thread_goal(state) if goal_in_force() else None
            if other and other.get("status") == "active" and other.get("evidence"):
                return {"_goal": {"evidence": None}}
            return {}
        if goal.get("status") != "active":
            return {}
        if result == "satisfied":
            evidence = goal.get("evidence") or ""
            # The verdict line already says the criteria passed; the event
            # adds only what the agent itself reported.
            self._goal_event("complete", status="complete", note=evidence)
            note = evidence or "Acceptance criteria satisfied."
            return {"_goal": {"status": "complete", "note": note, "evidence": None}}
        self._goal_event("unmet", status="active")
        return {"_goal": {"evidence": None}}

    def _running_own_processes(self, state: Any) -> list[str]:
        """Ids of this thread's background processes that are still running."""
        if self._process_status is None:
            return []
        thread_id = _current_thread_id()
        running: list[str] = []
        for process_id, record in (state.get("bg_processes") or {}).items():
            origin = record.get("origin_thread_id")
            if record.get("status") != "running" or (origin and origin != thread_id):
                continue
            if self._process_status(process_id) == "running":
                running.append(process_id)
        return running

    async def _pending_work(self, state: Any) -> tuple[int, int]:
        """Live counts of this thread's unfinished sub-agent tasks and processes."""
        grading = state.get("_rubric_grading") or {}
        skip = set(grading.get("skip") or [])
        processes = [p for p in self._running_own_processes(state) if p not in skip]
        tasks = 0
        skip_runs = set(grading.get("skip_runs") or [])
        candidates = [
            task
            for task in (state.get("async_tasks") or {}).values()
            if task.get("status") in ACTIVE_STATUSES
            and task.get("run_id")
            and task["run_id"] not in skip_runs
        ]
        if self._task_status is None or not candidates:
            return tasks, len(processes)
        try:
            async with asyncio.timeout(DEFER_QUERY_TIMEOUT_SECONDS):
                if self._server_available is not None and not (
                    await self._server_available()
                ):
                    # Nothing can be read; a task that cannot be read never blocks.
                    return tasks, len(processes)
                for task in candidates:
                    try:
                        status = await self._task_status(
                            task["task_id"], task["run_id"]
                        )
                    except NotFoundError:
                        continue  # the run is gone: nothing to wait for
                    except Exception as exc:
                        if _connection_refused(exc):
                            continue  # nobody listening: not a reason to warn
                        logger.warning(
                            "rubric: could not read the status of task %s; "
                            "grading without waiting for it",
                            task.get("task_id"),
                            exc_info=True,
                        )
                        continue
                    tasks += status in ACTIVE_STATUSES
        except TimeoutError:
            logger.warning(
                "rubric: task status reads exceeded %ss; grading without waiting",
                DEFER_QUERY_TIMEOUT_SECONDS,
            )
        return tasks, len(processes)

    @staticmethod
    def _emit_deferred(runtime: Any, tasks: int, processes: int) -> None:
        writer = getattr(runtime, "stream_writer", None)
        if writer is None:
            return
        try:
            writer(
                {
                    "type": "rubric_evaluation_deferred",
                    "tasks": tasks,
                    "processes": processes,
                }
            )
        except Exception:
            logger.debug("rubric: stream_writer raised; ignoring", exc_info=True)

    async def _defer_or_fail(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """The update that ends this hook early, or ``None`` to grade now."""
        if not state.get("rubric") or not self._require_run_marker:
            return None
        mode = (rubric_run_config() or {}).get("mode")
        if mode not in ("thread", "followup"):
            return None
        grading = state.get("_rubric_grading") or {}
        messages = list(state.get("messages") or [])
        if (
            mode == "followup"
            and _message_index(messages, grading.get("start")) is None
        ):
            # Nothing to show the grader: the task's own messages are gone.
            prep = self._prepare_evaluation(state, runtime)
            if prep is None:
                return None
            return self._end_ungraded(
                runtime,
                state,
                *prep,
                "the message that started this task is no longer in the thread",
            )
        tasks, processes = await self._pending_work(state)
        if not tasks and not processes:
            return None
        self._emit_deferred(runtime, tasks, processes)
        return {"_rubric_grading": {**grading, "deferred": True}}

    def before_agent(self, state, runtime):
        return self._start_of_run(state)

    async def abefore_agent(self, state, runtime):
        return self._start_of_run(state)

    # Checked again before grading: a run parked mid-way can be continued by a
    # client that never asked for it to be graded.
    @hook_config(can_jump_to=["model"])
    def after_agent(self, state, runtime):
        ungraded, update = self._ungraded_run_update(state)
        if ungraded:
            return update
        ungraded_goal = self._goal_ungraded_update(state)
        if ungraded_goal is not None:
            return ungraded_goal
        return super().after_agent(state, runtime)

    @hook_config(can_jump_to=["model"])
    async def aafter_agent(self, state, runtime):
        ungraded, update = self._ungraded_run_update(state)
        if ungraded:
            return update
        ungraded_goal = self._goal_ungraded_update(state)
        if ungraded_goal is not None:
            return ungraded_goal
        early = await self._defer_or_fail(state, runtime)
        if early is not None:
            return early
        return await super().aafter_agent(state, runtime)

    # -- grading ------------------------------------------------------------

    def _grader_input(self, state, iteration, correction=None):
        messages = list(state.get("messages") or [])
        grading = state.get("_rubric_grading") or {}
        index = _message_index(messages, grading.get("start"))
        goal = self._graded_goal(state)
        first = _message_index(messages, goal.get("start")) if goal else None
        if first is not None:
            # A goal spans turns: read from its first message, without the
            # feedback of gradings that came before this one.
            messages = [
                message
                for position, message in enumerate(messages)
                if position >= first
                and not (
                    index is not None
                    and position < index
                    and message_source(message) == RUBRIC_GRADER_MESSAGE_SOURCE
                )
            ]
        elif index is not None:
            messages = messages[index:]
        elif self._prepare_run_messages is not None:
            messages = self._prepare_run_messages(messages)
        return super()._grader_input(
            {**state, "messages": messages}, iteration, correction
        )

    def _final_cleanup(self, state: Any) -> dict[str, Any]:
        """The checklist belongs to the run that carried it; ``next`` to its own grading."""
        update: dict[str, Any] = {"rubric": None, "_rubric_grading": None}
        grading = state.get("_rubric_grading") or None
        if grading and grading.get("source") == "next":
            slot = state.get("_next_rubric")
            if slot and slot.get("id") == grading.get("id"):
                update["_next_rubric"] = None
        return update

    @staticmethod
    def _current_grading_only(update: dict[str, Any], grading_run_id: str) -> None:
        # Upstream appends every evaluation for the thread's lifetime.
        update["_rubric_evaluations"] = [
            e
            for e in update.get("_rubric_evaluations", [])
            if e.get("grading_run_id") == grading_run_id
        ]

    def _compose_update(self, state, evaluation):
        update = dict(super()._compose_update(state, evaluation))
        self._current_grading_only(update, evaluation["grading_run_id"])
        if evaluation["result"] == "needs_revision":
            grading = state.get("_rubric_grading") or None
            if grading and grading.get("deferred"):
                update["_rubric_grading"] = {**grading, "deferred": False}
            return update
        return {
            **update,
            **self._final_cleanup(state),
            **self._goal_verdict_update(state, evaluation["result"]),
        }

    def _handle_grader_exception(self, runtime, state, grading_run_id, iteration, exc):
        update = dict(
            super()._handle_grader_exception(
                runtime, state, grading_run_id, iteration, exc
            )
        )
        self._current_grading_only(update, grading_run_id)
        return {**update, **self._final_cleanup(state)}

    def _end_ungraded(self, runtime, state, grading_run_id, iteration, reason):
        """End a grading the grader was never called for, as a ``grader_error``."""
        logger.warning("rubric: grading ended without a verdict: %s", reason)
        evaluation: RubricEvaluation = {
            "grading_run_id": grading_run_id,
            "iteration": iteration,
            "result": "grader_error",
            "explanation": f"Not graded: {reason}.",
            "criteria": [],
            "unverified": False,
        }
        self._emit(
            runtime, "rubric_evaluation_end", grading_run_id, iteration, evaluation
        )
        if self._on_evaluation is not None:
            try:
                self._on_evaluation(evaluation)
            except GraphBubbleUp:
                raise
            except Exception:
                logger.exception("rubric: on_evaluation callback raised")
        update: dict[str, Any] = {
            "_rubric_evaluations": [
                *state.get("_rubric_evaluations", []),
                evaluation,
            ],
            "_rubric_iterations": iteration + 1,
            "_rubric_status": "grader_error",
        }
        self._current_grading_only(update, grading_run_id)
        return {**update, **self._final_cleanup(state)}

    @staticmethod
    def _grader_invocation_config(metadata):
        # Upstream hands the grader the parent's metadata, agent name included,
        # so its retry notice would be labelled like the main agent's.
        config = RubricMiddleware._grader_invocation_config(metadata)
        return {
            "metadata": {
                **config["metadata"],
                "lc_agent_name": RUBRIC_GRADER_MESSAGE_SOURCE,
            }
        }

    def _ensure_grader(self) -> Any:
        if self._grader is not None:
            return self._grader
        from deepagents._models import resolve_model

        from EvoScientist.middleware.utils import structured_output_for

        resolved_model = resolve_model(self._model)
        self._resolved_model = resolved_model
        model, response_format = resolved_model, _grader_strategy(resolved_model)
        if response_format is None:
            model, response_format = structured_output_for(
                resolved_model, GraderResponse
            )
        self._grader = create_agent(
            model=model,
            system_prompt=self._system_prompt,
            tools=self._tools,
            middleware=self._grader_middleware,
            name=RUBRIC_GRADER_MESSAGE_SOURCE,
            response_format=response_format,
            state_schema=self._grader_state_schema,
            context_schema=self._grader_context_schema,
        )
        return self._grader


class _GraderCallBudget(AgentMiddleware):
    """Fail closed once one grader attempt has made ``max_calls`` model calls.

    Counts the ``AIMessage``s already in the request, so the budget is per
    grader invocation by construction and no per-run state is needed. Raised
    on the first attempt it surfaces as ``grader_error``; raised on the
    coverage retry, upstream ``_grade`` swallows it and downgrades the first
    (unusable) verdict to ``needs_revision`` instead. Both terminate.
    """

    name = "rubric_grader_budget"

    def __init__(
        self,
        *,
        max_calls: int,
        who: str = "rubric grader",
        missing: str = "a verdict",
    ) -> None:
        self.max_calls = max_calls
        self._who = who
        self._missing = missing

    def _check(self, request: Any) -> None:
        spent = sum(isinstance(m, AIMessage) for m in request.messages)
        if spent >= self.max_calls:
            msg = (
                f"{self._who} exceeded {self.max_calls} model calls "
                f"without {self._missing}"
            )
            raise RuntimeError(msg)

    def wrap_model_call(self, request, handler):
        self._check(request)
        return handler(request)

    async def awrap_model_call(self, request, handler):
        self._check(request)
        return await handler(request)


def _log_rubric_evaluation(evaluation: RubricEvaluation) -> None:
    logger.info(
        "rubric iteration %s: %s — %s",
        evaluation.get("iteration"),
        evaluation.get("result"),
        evaluation.get("explanation"),
    )


def create_rubric_middleware(
    *,
    model: BaseChatModel,
    backend: BackendProtocol,
    max_iterations: int,
    prepare_messages: Callable[[list], list] | None = None,
    require_run_marker: bool = False,
    task_status: Callable[[str, str], Awaitable[str]] | None = None,
    process_status: Callable[[str], str] | None = None,
    server_available: Callable[[], Awaitable[bool]] | None = None,
    model_retry: AgentMiddleware | None = None,
) -> EvoRubricMiddleware:
    """Acceptance grading for a run that carries ``rubric`` (no-op otherwise).

    Must be the last of our middleware with an end hook: those run in reverse
    list order, so the grader sees the finished run first and a
    ``needs_revision`` verdict jumps back to the model before EvoScientist's
    other ``after_agent`` hooks fire (a harness profile's tail hooks still run
    first). The grader reads the same backend because the deliverables are
    files the transcript alone cannot prove exist.
    """
    from deepagents import FilesystemMiddleware
    from deepagents.middleware import UnsupportedContentMiddleware
    from langchain_core._api import LangChainBetaWarning

    from .model_retry import ModelCallBoundaryMiddleware

    # Eviction thresholds off: both eviction paths write files through the
    # backend, which would let the grader touch the shared workspace.
    grader_fs = FilesystemMiddleware(
        backend=backend,
        tools=list(GRADER_TOOLS),
        tool_token_limit_before_evict=None,
        human_message_token_limit_before_evict=None,
    )
    with warnings.catch_warnings():
        # Beta API; graphs build at langgraph dev import, keep the log clean.
        warnings.simplefilter("ignore", LangChainBetaWarning)
        return EvoRubricMiddleware(
            model=model,
            system_prompt=GRADER_SYSTEM_PROMPT + GRADER_FILE_CHECK_PROMPT,
            grader_middleware=[
                *([model_retry] if model_retry is not None else []),
                grader_fs,
                _GraderCallBudget(max_calls=GRADER_MAX_CALLS),
                # create_agent does not add it; after every middleware that
                # edits the request, so it sees the final one.
                UnsupportedContentMiddleware(),
                # The call marks come with the retry layer: it is what rejects
                # an error-finished grader response the marks flag.
                *([ModelCallBoundaryMiddleware()] if model_retry is not None else []),
            ],
            prepare_messages=prepare_messages,
            max_iterations=max_iterations,
            on_evaluation=_log_rubric_evaluation,
            require_run_marker=require_run_marker,
            task_status=task_status,
            process_status=process_status,
            server_available=server_available,
        )
