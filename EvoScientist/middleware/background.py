"""``BackgroundExecutionMiddleware`` — background-process tools for the main agent.

Mirrors deepagents' ``AsyncSubAgentMiddleware`` shape (a middleware that owns a set of
tools). The tools are stateless wrappers over :mod:`EvoScientist.background`, which holds
the live, process-level registry. They reuse the sandbox's ``validate_command`` so a
background launch cannot bypass the same safety checks as ``execute``.

Naming: these manage OS *processes* (never "job" — that word is reserved-free; async
sub-agents are *tasks*, future cron is *schedules*).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain.agents.middleware import AgentMiddleware
from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from .. import background, paths
from ..backends import prepare_sandbox_command

if TYPE_CHECKING:
    from .notifier import NotifierPort


def _origin_thread_id(runtime: ToolRuntime | None) -> str | None:
    """Best-effort current CLI thread_id, used to route the completion notification."""
    try:
        return (runtime.config or {}).get("configurable", {}).get("thread_id")
    except Exception:
        return None


def _notify_done(
    proc: background.BgProcess,
    origin_thread_id: str | None,
    notifier: NotifierPort,
) -> None:
    """Watcher ``on_exit`` hook: enqueue a completion notification via the port.

    Skipped for user-stopped processes (the user already knows). The notifier
    port owns the notification type, so this module never imports the CLI layer.
    """
    if proc.stopped:
        return
    rc = proc.returncode
    if rc == 0:
        status = "success"
    elif rc is not None and rc < 0:
        status = "interrupted"  # terminated by a signal
    else:
        status = "error"
    notifier.enqueue_bg_process_notification(
        task_id=proc.process_id,
        agent_name=proc.name,
        status=status,
        prompt=proc.command,
        origin_cli_thread_id=origin_thread_id,
    )


def _make_run_in_background(notifier: NotifierPort, dangerous: bool):
    """Build the ``run_in_background`` tool bound to an injected notifier + policy.

    ``dangerous`` is captured from ``cfg.dangerous_mode`` at assembly (the agent
    is rebuilt when config changes, so the captured value never goes stale), and
    the notifier is the injected port used for the completion notification.
    """

    @tool(parse_docstring=True)
    def run_in_background(
        command: str, name: str | None = None, runtime: ToolRuntime = None
    ) -> str:
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
            command, cwd, virtual_mode=not dangerous, dangerous=dangerous
        )
        if error:
            return error
        tid = _origin_thread_id(runtime)
        process_id = background.launch(
            command,
            cwd,
            name,
            origin_thread_id=tid,
            on_exit=lambda p: _notify_done(p, tid, notifier),
        )
        label = f" (name={name!r})" if name else ""
        # In dangerous mode `/` is the real root, so advertise the real log path;
        # in virtual mode `/.bg_processes/...` correctly maps to the workspace.
        log_path = (
            f"{cwd}/.bg_processes/{process_id}.log"
            if dangerous
            else f"/.bg_processes/{process_id}.log"
        )
        return (
            f"Started background process {process_id}{label}. "
            f"Output -> {log_path}. "
            f"Poll with check_process('{process_id}'), stop with stop_process('{process_id}')."
        )

    return run_in_background


@tool(parse_docstring=True)
def check_process(process_id: str, runtime: ToolRuntime = None) -> str:
    """Check a background process's status and recent output.

    Args:
        process_id: The id returned by run_in_background.
    """
    return background.status(process_id, thread_id=_origin_thread_id(runtime))


@tool(parse_docstring=True)
def stop_process(process_id: str) -> str:
    """Stop (kill) a running background process and its child process group.

    Args:
        process_id: The id returned by run_in_background.
    """
    return background.stop(process_id)


@tool(parse_docstring=True)
def list_processes(all_threads: bool = False, runtime: ToolRuntime = None) -> str:
    """List background processes launched this session with their live statuses.

    Args:
        all_threads: List processes from every session, not just the current one.
    """
    return background.list_all(_origin_thread_id(runtime), include_all=all_threads)


class BackgroundExecutionMiddleware(AgentMiddleware):
    """Adds run_in_background / check_process / stop_process / list_processes.

    Modelled on ``AsyncSubAgentMiddleware``: the middleware simply exposes the tool set.
    Attached to the main agent only (async sub-agents must not spawn local processes).
    """

    def __init__(self, notifier: NotifierPort, *, dangerous: bool = False) -> None:
        super().__init__()
        self.tools = [
            _make_run_in_background(notifier, dangerous),
            check_process,
            stop_process,
            list_processes,
        ]
