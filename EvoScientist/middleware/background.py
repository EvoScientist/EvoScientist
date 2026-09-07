"""``BackgroundExecutionMiddleware`` — background-process tools for the main agent.

Mirrors deepagents' ``AsyncSubAgentMiddleware`` shape (a middleware that owns a set of
tools). The tools are wrappers over :mod:`EvoScientist.background`, which holds the live,
process-level registry. They reuse the sandbox's ``validate_command`` so a background
launch cannot bypass the same safety checks as ``execute``.

Lifecycle mirror: each tool also writes a snapshot of the affected process(es) into the
``bg_processes`` thread-state channel (a merge-reducer dict keyed by ``process_id``,
modelled on deepagents' ``async_tasks``). The client detects a process exit by reading
that state through the gateway and polling ``gateway.get_process_status`` — the same
state-driven read path the async-task reader uses — so completion notifications work
across the CLI↔server process boundary. There is no in-process push: the daemon watcher
(``background._watch``) still records the exit in the registry, but nothing is enqueued
from it.

Naming: these manage OS *processes* (never "job" — that word is reserved-free; async
sub-agents are *tasks*, future cron is *schedules*).

This module intentionally does NOT use ``from __future__ import annotations``: the tools
inject ``ToolRuntime`` and return ``Command``, and ``StructuredTool`` inspects the raw
signature to wire the injected runtime — PEP 563 stringized annotations break that
detection (same reason as ``middleware/expert_async_subagent.py``).
"""

from typing import Annotated, Any, NotRequired, TypedDict

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import AgentState
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command

from .. import background, paths
from ..backends import prepare_sandbox_command


def _bg_processes_reducer(
    existing: "dict[str, BgProcessRecord] | None",
    update: "dict[str, BgProcessRecord]",
) -> "dict[str, BgProcessRecord]":
    """Merge process-record updates into the existing registry (last write per key)."""
    merged = dict(existing or {})
    merged.update(update)
    return merged


class BgProcessRecord(TypedDict, total=False):
    """One process's mirror in the ``bg_processes`` state channel."""

    process_id: str
    name: str
    command: str
    pid: int
    status: str  # "running" | terminal (success/error/interrupted)
    returncode: int | None
    started_at: str
    origin_thread_id: str | None


class BackgroundState(AgentState):
    """State extension registering the ``bg_processes`` channel + its merge reducer."""

    bg_processes: Annotated[
        NotRequired[dict[str, BgProcessRecord]], _bg_processes_reducer
    ]


def _origin_thread_id(runtime: ToolRuntime | None) -> str | None:
    """Best-effort current CLI thread_id, used to route the completion notification."""
    try:
        return (runtime.config or {}).get("configurable", {}).get("thread_id")
    except Exception:
        return None


def _bg_command(
    text: str,
    records: list[dict[str, Any] | None],
    runtime: ToolRuntime | None,
) -> "str | Command":
    """Return a ``Command`` carrying the tool message + a ``bg_processes`` state update.

    Falls back to the plain ``text`` string when there is nothing to mirror or no
    ``tool_call_id`` is reachable (e.g. a direct unit-test call with no runtime), so the
    tool stays usable outside a graph.
    """
    tool_call_id = getattr(runtime, "tool_call_id", None)
    valid = [r for r in records if r is not None]
    if not valid or tool_call_id is None:
        return text
    return Command(
        update={
            "messages": [ToolMessage(text, tool_call_id=tool_call_id)],
            "bg_processes": {r["process_id"]: r for r in valid},
        }
    )


def _make_run_in_background(dangerous: bool, guard_dangerous: bool = False):
    """Build the ``run_in_background`` tool bound to the sandbox policy.

    ``dangerous`` is captured from ``cfg.dangerous_mode`` at assembly (the agent is
    rebuilt when config changes, so the captured value never goes stale).
    ``guard_dangerous`` mirrors ``execute``'s backstop: with no interactive approval
    reachable (``auto_approve``), refuse the narrow dangerous set instead of running it
    unattended.
    """

    @tool(parse_docstring=True)
    def run_in_background(
        command: str, name: str | None = None, runtime: ToolRuntime = None
    ) -> "str | Command":
        """Launch a long-running shell command in the background and return immediately.

        Use for unbounded or very long tasks (model training, large downloads, servers)
        that should not block the conversation. Output streams to a log file; poll it with
        check_process and stop it with stop_process. For a bounded command that just needs
        more time, prefer execute(..., timeout=N) instead of backgrounding.

        Args:
            command: The shell command to run in the background.
            name: Optional short label to recognize the process later.
        """
        cwd = str(paths.resolve_virtual_path("/"))
        # Same path-rewriting + validation as execute (shared helper) so virtual paths
        # resolve to the workspace and the command can't bypass the sandbox checks.
        command, error = prepare_sandbox_command(
            command,
            cwd,
            virtual_mode=not dangerous,
            dangerous=dangerous,
            guard_dangerous=guard_dangerous,
        )
        if error:
            return error
        tid = _origin_thread_id(runtime)
        process_id = background.launch(command, cwd, name, origin_thread_id=tid)
        label = f" (name={name!r})" if name else ""
        # In dangerous mode `/` is the real root, so advertise the real log path;
        # in virtual mode `/.bg_processes/...` correctly maps to the workspace.
        log_path = (
            f"{cwd}/.bg_processes/{process_id}.log"
            if dangerous
            else f"/.bg_processes/{process_id}.log"
        )
        text = (
            f"Started background process {process_id}{label}. "
            f"Output -> {log_path}. "
            f"Poll with check_process('{process_id}'), stop with stop_process('{process_id}')."
        )
        return _bg_command(text, [background.state_record(process_id)], runtime)

    return run_in_background


@tool(parse_docstring=True)
def check_process(process_id: str, runtime: ToolRuntime = None) -> "str | Command":
    """Check a background process's status and recent output.

    Args:
        process_id: The id returned by run_in_background.
    """
    text = background.status(process_id, thread_id=_origin_thread_id(runtime))
    return _bg_command(text, [background.state_record(process_id)], runtime)


@tool(parse_docstring=True)
def stop_process(process_id: str, runtime: ToolRuntime = None) -> "str | Command":
    """Stop (kill) a running background process and its child process group.

    Args:
        process_id: The id returned by run_in_background.
    """
    text = background.stop(process_id)
    return _bg_command(text, [background.state_record(process_id)], runtime)


@tool(parse_docstring=True)
def list_processes(
    all_threads: bool = False, runtime: ToolRuntime = None
) -> "str | Command":
    """List background processes launched this session with their live statuses.

    Args:
        all_threads: List processes from every session, not just the current one.
    """
    tid = _origin_thread_id(runtime)
    text = background.list_all(tid, include_all=all_threads)
    return _bg_command(
        text, background.list_records(tid, include_all=all_threads), runtime
    )


class BackgroundExecutionMiddleware(AgentMiddleware):
    """Adds run_in_background / check_process / stop_process / list_processes.

    Modelled on ``AsyncSubAgentMiddleware``: exposes the tool set and declares the
    ``bg_processes`` state channel the tools write into. Attached to the main agent only
    (async sub-agents must not spawn local processes).
    """

    state_schema = BackgroundState

    def __init__(
        self,
        *,
        dangerous: bool = False,
        guard_dangerous: bool = False,
    ) -> None:
        super().__init__()
        self.tools = [
            _make_run_in_background(dangerous, guard_dangerous),
            check_process,
            stop_process,
            list_processes,
        ]
