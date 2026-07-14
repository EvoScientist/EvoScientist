"""CodeInterpreterMiddleware configuration for EvoScientist.

Wraps ``langchain-quickjs``'s ``CodeInterpreterMiddleware`` with project-specific
defaults: a PTC allowlist scoped to read-only, batch-friendly tools relevant to
the scientific research workflow (search, sub-agent dispatch, file inspection),
a longer per-eval timeout suitable for LLM-authored algorithms, a larger result
budget for returning structured JSON, and a user-facing tool name that LLMs
recognize from ChatGPT Code Interpreter training data.

Excluded from PTC by design:
    - ``task`` (sub-agent dispatch) — reserved by langchain-quickjs >=0.3; it
      is always the top-level ``task()`` REPL global (with ``responseSchema``),
      so a ``tools.task`` variant would be a conflicting, degraded duplicate
    - ``execute`` (shell) — would bypass ``HumanInTheLoopMiddleware`` approval
    - ``write_file`` / ``edit_file`` — side-effectful, no batch benefit
    - ``think_tool`` — reflection is not batchable
    - ``tavily_search`` — only mounted on the ``research-agent`` sub-agent,
      not on the main agent; main agent reaches search via ``task`` dispatch
    - MCP tools — dynamic at runtime; add manually if a specific server needs PTC

Usage::

    from EvoScientist.middleware import create_code_interpreter_middleware

    middleware = create_code_interpreter_middleware(
        timeout=60.0, max_result_chars=10000
    )
"""

from __future__ import annotations

import contextlib
import weakref

from langchain.agents.middleware.types import ModelRequest
from langchain_quickjs import CodeInterpreterMiddleware

# Defaults match the historical hardcoded values. Callers (the agent
# builder in ``EvoScientist.py``) pass the resolved ``EvoScientistConfig``
# values; tests / ad-hoc callers can omit and get sensible defaults.
_DEFAULT_TIMEOUT_SECONDS: float = 60.0
_DEFAULT_MAX_RESULT_CHARS: int = 10000

_MEMORY_FIRST_INTERPRETER_PROMPT = (
    "\n\nWhen memory tools (search_observations, read_memory) are available, use "
    "them before `code_interpreter` for workspace inspection or implementation work."
)

_live_interpreters: weakref.WeakSet[EvoCodeInterpreterMiddleware]


class EvoCodeInterpreterMiddleware(CodeInterpreterMiddleware):
    """Code interpreter middleware with EvoScientist's memory preflight hint.

    ``after_agent`` / ``aafter_agent`` are intentionally NOT overridden. An
    earlier "conditional snapshot" gate that skipped ``after_agent`` on turns
    where ``code_interpreter`` wasn't called saved ~50 ms/turn of
    ``create_snapshot()`` work, but also skipped the slot eviction upstream
    performs in the same hook (``finally: self._registry.evict(thread_id)``
    in ``langchain_quickjs.middleware.CodeInterpreterMiddleware.after_agent``).
    ``before_agent`` restores the REPL on every turn that follows a touched
    one via ``self._registry.get(thread_id)`` (get-or-create), so skipping
    eviction leaked one ``ThreadWorker`` + QuickJS Runtime per persistent
    ``thread_id`` that ever went touched → quiet. The regression test
    ``test_after_agent_evicts_slot_on_untouched_turn`` guards against
    reintroducing the gate.
    """

    def _prepare_for_call(self, request: ModelRequest) -> str:
        return super()._prepare_for_call(request) + _MEMORY_FIRST_INTERPRETER_PROMPT

    async def aclose(self) -> None:
        """Evict active REPLs on their worker loops before event-loop shutdown."""
        registry = self._registry
        with registry._lock:
            thread_ids = tuple(registry._slots)
        for thread_id in thread_ids:
            with contextlib.suppress(Exception):
                await registry.aevict(thread_id)
        self._ptc_tools_by_thread.clear()


_live_interpreters = weakref.WeakSet()


async def aclose_code_interpreters() -> None:
    """Close all live EvoScientist QuickJS middleware instances."""
    for middleware in tuple(_live_interpreters):
        await middleware.aclose()


# Read-only, batchable tools that benefit from being callable inside JS.
# Multi-agent orchestration is the killer use case: ``Promise.all`` over
# ``start_async_task`` fans out experiments / writing / data-analysis in
# parallel without each dispatch costing a separate LLM round-trip. Names
# that don't exist at runtime (e.g. async tools when langgraph dev isn't
# reachable) are silently skipped by ``filter_tools_for_ptc``.
_DEFAULT_PTC_ALLOWLIST: list[str] = [
    # Memory lookup (read-only, should precede workspace inspection)
    "search_observations",
    "read_memory",
    # Async sub-agent dispatch (langgraph dev). `task` is excluded — see docstring.
    "start_async_task",
    "check_async_task",
    "update_async_task",
    "cancel_async_task",
    "list_async_tasks",
    # Workspace inspection (read-only, batchable)
    "read_file",
    "grep",
    "glob",
    "ls",
]


def create_code_interpreter_middleware(
    *,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    max_result_chars: int = _DEFAULT_MAX_RESULT_CHARS,
) -> CodeInterpreterMiddleware:
    """Build a project-tuned CodeInterpreterMiddleware instance.

    Args:
        timeout: Per-eval timeout in seconds. Defaults to 60s — long enough
            for LLM-authored algorithms that touch async sub-agent dispatch
            (``start_async_task`` + ``check_async_task`` polling).
        max_result_chars: Maximum characters of JS eval output passed back
            to the LLM. Defaults to 10k — fits structured JSON aggregations
            of file reads / sub-agent results without truncating useful
            payloads. Larger values trade tokens for completeness.

    Returns:
        Configured ``CodeInterpreterMiddleware`` ready to append to an agent's
        middleware stack.
    """
    middleware = EvoCodeInterpreterMiddleware(
        ptc=_DEFAULT_PTC_ALLOWLIST,
        timeout=timeout,
        max_result_chars=max_result_chars,
        tool_name="code_interpreter",
    )
    _live_interpreters.add(middleware)
    return middleware
