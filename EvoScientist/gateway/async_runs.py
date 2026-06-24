"""On-demand background LangGraph agent runs.

This module owns the generic mechanics for launching short-lived background
agent graphs through the local ``langgraph dev`` server:

* check that the server is reachable
* create a worker thread
* submit a run
* poll run status without blocking the caller
* delete finished worker threads

Domain-specific callers, such as EvoMemory, provide payload builders and hooks
for their own accounting.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypedDict

if TYPE_CHECKING:
    from langgraph_sdk.schema import Config, Input, Run, Thread

logger = logging.getLogger(__name__)

DEFAULT_BACKGROUND_AGENT_TERMINAL_STATUSES = frozenset(
    {"success", "error", "timeout", "interrupted"}
)
DEFAULT_BACKGROUND_AGENT_POLL_INTERVAL_SECONDS = 1.0
DEFAULT_BACKGROUND_AGENT_MAX_POLL_FAILURES = 3
DEFAULT_BACKGROUND_AGENT_HEADERS = {"x-auth-scheme": "langsmith"}

_background_agent_watcher_tasks: set[asyncio.Task[None]] = set()


class BackgroundAgentRunPayload(TypedDict):
    """Typed payload submitted to LangGraph SDK ``runs.create``."""

    assistant_id: str
    input: Input
    metadata: dict[str, str]
    config: Config


class _SyncThreadsClient(Protocol):
    def create(
        self,
        *,
        graph_id: str,
        metadata: dict[str, str],
    ) -> Thread: ...

    def delete(self, thread_id: str) -> object: ...


class _SyncRunsClient(Protocol):
    def create(
        self,
        thread_id: str,
        assistant_id: str,
        *,
        input: Input,
        metadata: dict[str, str],
        config: Config,
    ) -> Run: ...

    def get(self, thread_id: str, run_id: str) -> Run: ...


class SyncLangGraphClient(Protocol):
    """Sync subset of the LangGraph SDK used by background runs."""

    threads: _SyncThreadsClient
    runs: _SyncRunsClient


class _AsyncThreadsClient(Protocol):
    async def create(
        self,
        *,
        graph_id: str,
        metadata: dict[str, str],
    ) -> Thread: ...

    async def delete(self, thread_id: str) -> object: ...


class _AsyncRunsClient(Protocol):
    async def create(
        self,
        thread_id: str,
        assistant_id: str,
        *,
        input: Input,
        metadata: dict[str, str],
        config: Config,
    ) -> Run: ...

    async def get(self, thread_id: str, run_id: str) -> Run: ...


class AsyncLangGraphClient(Protocol):
    """Async subset of the LangGraph SDK used by background runs."""

    threads: _AsyncThreadsClient
    runs: _AsyncRunsClient


BackgroundAgentRunPayloadBuilder = Callable[[str], BackgroundAgentRunPayload]


@dataclass(frozen=True)
class BackgroundAgentLaunchRequest:
    """Description of one on-demand background agent run."""

    graph_id: str
    run_payload: BackgroundAgentRunPayloadBuilder
    thread_metadata: Mapping[str, str] | None = None
    url: str | None = None
    headers: Mapping[str, str] | None = None
    name: str = "background agent"


@dataclass(frozen=True)
class BackgroundAgentRun:
    """Identifiers for a submitted background agent run."""

    name: str
    url: str
    graph_id: str
    thread_id: str
    run_id: str
    assistant_id: str
    metadata: Mapping[str, str]


@dataclass(frozen=True)
class BackgroundAgentLaunchHooks:
    """Lifecycle hooks for caller-specific accounting."""

    on_before_run: Callable[[str], None] | None = None
    on_started: Callable[[BackgroundAgentRun], None] | None = None
    on_finished: Callable[[BackgroundAgentRun], None] | None = None
    on_aborted: Callable[[BackgroundAgentRun], None] | None = None
    on_watcher_start_failed: Callable[[BackgroundAgentRun], None] | None = None


@dataclass(frozen=True)
class BackgroundAgentStatusWatcherConfig:
    """Polling behavior for a background agent run."""

    terminal_statuses: frozenset[str] = DEFAULT_BACKGROUND_AGENT_TERMINAL_STATUSES
    poll_interval_seconds: float = DEFAULT_BACKGROUND_AGENT_POLL_INTERVAL_SECONDS
    max_poll_failures: int = DEFAULT_BACKGROUND_AGENT_MAX_POLL_FAILURES
    delete_thread_on_finish: bool = True


def default_background_agent_url() -> str:
    """Return the configured local ``langgraph dev`` URL."""
    from ..EvoScientist import _ensure_config

    cfg = _ensure_config()
    port = int(getattr(cfg, "langgraph_dev_port", 6174))
    return f"http://localhost:{port}"


def _headers(headers: Mapping[str, str] | None) -> dict[str, str]:
    return dict(DEFAULT_BACKGROUND_AGENT_HEADERS if headers is None else headers)


def _call_hook(
    callback: Callable[[BackgroundAgentRun], None] | None,
    run: BackgroundAgentRun,
    *,
    hook_name: str,
) -> None:
    if callback is None:
        return
    try:
        callback(run)
    except Exception:
        logger.warning(
            "%s hook failed for %s run %s",
            hook_name,
            run.name,
            run.run_id,
            exc_info=True,
        )


def _call_before_run_hook(
    callback: Callable[[str], None] | None,
    thread_id: str,
    *,
    name: str,
) -> None:
    if callback is None:
        return
    try:
        callback(thread_id)
    except Exception:
        logger.warning(
            "on_before_run hook failed for %s thread %s",
            name,
            thread_id,
            exc_info=True,
        )
        raise


async def _acall_hook(
    callback: Callable[[BackgroundAgentRun], None] | None,
    run: BackgroundAgentRun,
    *,
    hook_name: str,
) -> None:
    if callback is None:
        return
    try:
        await asyncio.to_thread(callback, run)
    except Exception:
        logger.warning(
            "%s hook failed for %s run %s",
            hook_name,
            run.name,
            run.run_id,
            exc_info=True,
        )


async def _acall_before_run_hook(
    callback: Callable[[str], None] | None,
    thread_id: str,
    *,
    name: str,
) -> None:
    if callback is None:
        return
    try:
        await asyncio.to_thread(callback, thread_id)
    except Exception:
        logger.warning(
            "on_before_run hook failed for %s thread %s",
            name,
            thread_id,
            exc_info=True,
        )
        raise


def delete_background_agent_thread(
    client: SyncLangGraphClient,
    thread_id: str,
    *,
    name: str = "background agent",
) -> None:
    """Best-effort delete of a finished worker thread."""
    try:
        client.threads.delete(thread_id)
    except Exception:
        logger.debug("Failed to delete %s thread %s", name, thread_id, exc_info=True)


async def adelete_background_agent_thread(
    client: AsyncLangGraphClient,
    thread_id: str,
    *,
    name: str = "background agent",
) -> None:
    """Async variant of :func:`delete_background_agent_thread`."""
    try:
        await client.threads.delete(thread_id)
    except Exception:
        logger.debug("Failed to delete %s thread %s", name, thread_id, exc_info=True)


def launch_background_agent(
    request: BackgroundAgentLaunchRequest,
    *,
    hooks: BackgroundAgentLaunchHooks | None = None,
    watcher_config: BackgroundAgentStatusWatcherConfig | None = None,
    spawn_status_watcher: Callable[[BackgroundAgentRun], None] | None = None,
) -> BackgroundAgentRun | None:
    """Submit a background agent run to the local LangGraph server."""
    from langgraph_sdk import get_sync_client

    from ..langgraph_dev.manager import is_langgraph_dev_running

    hooks = hooks or BackgroundAgentLaunchHooks()
    watcher_config = watcher_config or BackgroundAgentStatusWatcherConfig()
    url = request.url or default_background_agent_url()
    if not is_langgraph_dev_running(base_url=url):
        logger.info("Skipping %s launch; LangGraph dev is unavailable", request.name)
        return None

    client: SyncLangGraphClient = get_sync_client(
        url=url,
        headers=_headers(request.headers),
    )
    thread_metadata = dict(request.thread_metadata or {})
    thread = client.threads.create(
        graph_id=request.graph_id,
        metadata=thread_metadata,
    )
    thread_id = thread["thread_id"]
    _call_before_run_hook(
        hooks.on_before_run,
        thread_id,
        name=request.name,
    )
    payload = request.run_payload(thread_id)
    run = client.runs.create(
        thread_id=thread_id,
        assistant_id=payload["assistant_id"],
        input=payload["input"],
        metadata=payload["metadata"],
        config=payload["config"],
    )
    run_id = run["run_id"]

    handle = BackgroundAgentRun(
        name=request.name,
        url=url,
        graph_id=request.graph_id,
        thread_id=thread_id,
        run_id=run_id,
        assistant_id=payload["assistant_id"],
        metadata=dict(payload["metadata"]),
    )
    _call_hook(hooks.on_started, handle, hook_name="on_started")
    try:
        if spawn_status_watcher is None:
            spawn_background_agent_status_thread(
                handle,
                headers=request.headers,
                hooks=hooks,
                watcher_config=watcher_config,
            )
        else:
            spawn_status_watcher(handle)
    except Exception:
        failed_hook = hooks.on_watcher_start_failed or hooks.on_aborted
        _call_hook(failed_hook, handle, hook_name="on_watcher_start_failed")
        logger.warning("Failed to start %s status watcher", request.name, exc_info=True)
    return handle


async def alaunch_background_agent(
    request: BackgroundAgentLaunchRequest,
    *,
    hooks: BackgroundAgentLaunchHooks | None = None,
    watcher_config: BackgroundAgentStatusWatcherConfig | None = None,
    spawn_status_watcher: Callable[[BackgroundAgentRun], None] | None = None,
) -> BackgroundAgentRun | None:
    """Async variant of :func:`launch_background_agent`."""
    from langgraph_sdk import get_client

    from ..langgraph_dev.manager import is_langgraph_dev_running

    hooks = hooks or BackgroundAgentLaunchHooks()
    watcher_config = watcher_config or BackgroundAgentStatusWatcherConfig()
    url = request.url or default_background_agent_url()
    if not await asyncio.to_thread(is_langgraph_dev_running, base_url=url):
        logger.info("Skipping %s launch; LangGraph dev is unavailable", request.name)
        return None

    client: AsyncLangGraphClient = get_client(
        url=url,
        headers=_headers(request.headers),
    )
    thread_metadata = dict(request.thread_metadata or {})
    thread = await client.threads.create(
        graph_id=request.graph_id,
        metadata=thread_metadata,
    )
    thread_id = thread["thread_id"]
    await _acall_before_run_hook(
        hooks.on_before_run,
        thread_id,
        name=request.name,
    )
    payload = request.run_payload(thread_id)
    run = await client.runs.create(
        thread_id=thread_id,
        assistant_id=payload["assistant_id"],
        input=payload["input"],
        metadata=payload["metadata"],
        config=payload["config"],
    )
    run_id = run["run_id"]

    handle = BackgroundAgentRun(
        name=request.name,
        url=url,
        graph_id=request.graph_id,
        thread_id=thread_id,
        run_id=run_id,
        assistant_id=payload["assistant_id"],
        metadata=dict(payload["metadata"]),
    )
    await _acall_hook(hooks.on_started, handle, hook_name="on_started")
    try:
        if spawn_status_watcher is None:
            spawn_background_agent_status_thread(
                handle,
                headers=request.headers,
                hooks=hooks,
                watcher_config=watcher_config,
            )
        else:
            spawn_status_watcher(handle)
    except Exception:
        failed_hook = hooks.on_watcher_start_failed or hooks.on_aborted
        await _acall_hook(failed_hook, handle, hook_name="on_watcher_start_failed")
        logger.warning("Failed to start %s status watcher", request.name, exc_info=True)
    return handle


def spawn_background_agent_status_thread(
    run: BackgroundAgentRun,
    *,
    headers: Mapping[str, str] | None = None,
    hooks: BackgroundAgentLaunchHooks | None = None,
    watcher_config: BackgroundAgentStatusWatcherConfig | None = None,
) -> None:
    """Poll a background agent run from a daemon thread."""
    thread = threading.Thread(
        target=watch_background_agent_run_sync,
        kwargs={
            "url": run.url,
            "thread_id": run.thread_id,
            "run_id": run.run_id,
            "graph_id": run.graph_id,
            "assistant_id": run.assistant_id,
            "metadata": run.metadata,
            "name": run.name,
            "headers": headers,
            "hooks": hooks,
            "watcher_config": watcher_config,
        },
        name="evosci-background-agent-status",
        daemon=True,
    )
    thread.start()


def watch_background_agent_run_sync(
    *,
    url: str,
    thread_id: str,
    run_id: str,
    graph_id: str = "",
    assistant_id: str = "",
    metadata: Mapping[str, str] | None = None,
    name: str = "background agent",
    headers: Mapping[str, str] | None = None,
    hooks: BackgroundAgentLaunchHooks | None = None,
    watcher_config: BackgroundAgentStatusWatcherConfig | None = None,
) -> None:
    """Poll a submitted background agent run until it finishes or polling aborts."""
    from langgraph_sdk import get_sync_client

    hooks = hooks or BackgroundAgentLaunchHooks()
    watcher_config = watcher_config or BackgroundAgentStatusWatcherConfig()
    run_ref = BackgroundAgentRun(
        name=name,
        url=url,
        graph_id=graph_id,
        thread_id=thread_id,
        run_id=run_id,
        assistant_id=assistant_id,
        metadata=dict(metadata or {}),
    )
    failures = 0
    confirmed_finished = False
    client: SyncLangGraphClient | None = None
    try:
        client = get_sync_client(url=url, headers=_headers(headers))
        while True:
            try:
                run = client.runs.get(thread_id=thread_id, run_id=run_id)
                failures = 0
                status = run["status"]
            except Exception:
                failures += 1
                if failures >= watcher_config.max_poll_failures:
                    logger.warning(
                        "Stopping %s status watch for %s after %d failed polls",
                        name,
                        run_id,
                        failures,
                        exc_info=True,
                    )
                    return
                time.sleep(watcher_config.poll_interval_seconds)
                continue

            if status in watcher_config.terminal_statuses:
                confirmed_finished = True
                return
            time.sleep(watcher_config.poll_interval_seconds)
    finally:
        if confirmed_finished:
            _call_hook(hooks.on_finished, run_ref, hook_name="on_finished")
            if watcher_config.delete_thread_on_finish and client is not None:
                delete_background_agent_thread(client, thread_id, name=name)
        else:
            _call_hook(hooks.on_aborted, run_ref, hook_name="on_aborted")


def spawn_background_agent_status_task(
    client: AsyncLangGraphClient,
    run: BackgroundAgentRun,
    *,
    hooks: BackgroundAgentLaunchHooks | None = None,
    watcher_config: BackgroundAgentStatusWatcherConfig | None = None,
) -> None:
    """Poll a background agent run without blocking the event loop."""
    task = asyncio.create_task(
        awatch_background_agent_run(
            client,
            thread_id=run.thread_id,
            run_id=run.run_id,
            graph_id=run.graph_id,
            assistant_id=run.assistant_id,
            metadata=run.metadata,
            name=run.name,
            hooks=hooks,
            watcher_config=watcher_config,
        )
    )
    _background_agent_watcher_tasks.add(task)
    task.add_done_callback(_background_agent_watcher_tasks.discard)


async def awatch_background_agent_run(
    client: AsyncLangGraphClient,
    *,
    thread_id: str,
    run_id: str,
    graph_id: str = "",
    assistant_id: str = "",
    metadata: Mapping[str, str] | None = None,
    name: str = "background agent",
    hooks: BackgroundAgentLaunchHooks | None = None,
    watcher_config: BackgroundAgentStatusWatcherConfig | None = None,
) -> None:
    """Async status watcher for callers that already hold an async SDK client."""
    hooks = hooks or BackgroundAgentLaunchHooks()
    watcher_config = watcher_config or BackgroundAgentStatusWatcherConfig()
    run_ref = BackgroundAgentRun(
        name=name,
        url="",
        graph_id=graph_id,
        thread_id=thread_id,
        run_id=run_id,
        assistant_id=assistant_id,
        metadata=dict(metadata or {}),
    )
    failures = 0
    confirmed_finished = False
    try:
        while True:
            try:
                run = await client.runs.get(thread_id=thread_id, run_id=run_id)
                failures = 0
                status = run["status"]
            except asyncio.CancelledError:
                raise
            except Exception:
                failures += 1
                if failures >= watcher_config.max_poll_failures:
                    logger.warning(
                        "Stopping %s status watch for %s after %d failed polls",
                        name,
                        run_id,
                        failures,
                        exc_info=True,
                    )
                    return
                await asyncio.sleep(watcher_config.poll_interval_seconds)
                continue

            if status in watcher_config.terminal_statuses:
                confirmed_finished = True
                return
            await asyncio.sleep(watcher_config.poll_interval_seconds)
    finally:
        if confirmed_finished:
            await _acall_hook(hooks.on_finished, run_ref, hook_name="on_finished")
            if watcher_config.delete_thread_on_finish:
                await adelete_background_agent_thread(client, thread_id, name=name)
        else:
            await _acall_hook(hooks.on_aborted, run_ref, hook_name="on_aborted")
