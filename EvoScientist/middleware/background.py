"""``BackgroundExecutionMiddleware`` — background-process tools for the main agent.

Mirrors deepagents' ``AsyncSubAgentMiddleware`` shape (a middleware that owns a set of
tools). The tools are stateless wrappers over :mod:`EvoScientist.background`, which holds
the live, process-level registry. They reuse the sandbox's ``validate_command`` so a
background launch cannot bypass the same safety checks as ``execute``.

Naming: these manage OS *processes* (never "job" — that word is reserved-free; async
sub-agents are *tasks*, future cron is *schedules*).
"""

from __future__ import annotations

from langchain.agents.middleware import AgentMiddleware
from langchain_core.tools import tool

from .. import background, paths
from ..backends import prepare_sandbox_command


@tool(parse_docstring=True)
def run_in_background(command: str, name: str | None = None) -> str:
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
    command, error = prepare_sandbox_command(command, cwd)
    if error:
        return error
    process_id = background.launch(command, cwd, name)
    label = f" (name={name!r})" if name else ""
    return (
        f"Started background process {process_id}{label}. "
        f"Output -> /.bg_processes/{process_id}.log. "
        f"Poll with check_process('{process_id}'), stop with stop_process('{process_id}')."
    )


@tool(parse_docstring=True)
def check_process(process_id: str) -> str:
    """Check a background process's status and recent output.

    Args:
        process_id: The id returned by run_in_background.
    """
    return background.status(process_id)


@tool(parse_docstring=True)
def stop_process(process_id: str) -> str:
    """Stop (kill) a running background process and its child process group.

    Args:
        process_id: The id returned by run_in_background.
    """
    return background.stop(process_id)


@tool(parse_docstring=True)
def list_processes() -> str:
    """List all background processes launched this session with their live statuses."""
    return background.list_all()


class BackgroundExecutionMiddleware(AgentMiddleware):
    """Adds run_in_background / check_process / stop_process / list_processes.

    Modelled on ``AsyncSubAgentMiddleware``: the middleware simply exposes the tool set.
    Attached to the main agent only (async sub-agents must not spawn local processes).
    """

    def __init__(self) -> None:
        super().__init__()
        self.tools = [run_in_background, check_process, stop_process, list_processes]
