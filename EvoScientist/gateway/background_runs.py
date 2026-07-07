"""On-demand background LangGraph runs.

This module owns the generic mechanics for launching short-lived background
graphs through the local ``langgraph dev`` server:

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

from ..langgraph_dev.sdk import (
    configured_langgraph_dev_url,
    langgraph_dev_headers,
)

if TYPE_CHECKING:
    from langgraph_sdk.schema import Config, Input, Run, Thread

logger = logging.getLogger(__name__)

DEFAULT_BACKGROUND_RUN_TERMINAL_STATUSES = frozenset(
    {"success", "error", "timeout", "interrupted"}
)
DEFAULT_BACKGROUND_RUN_POLL_INTERVAL_SECONDS = 1.0
DEFAULT_BACKGROUND_RUN_MAX_POLL_FAILURES = 3

_background_run_watcher_tasks: set[asyncio.Task[None]] = set()


class BackgroundRunPayload(TypedDict):
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


BackgroundRunPayloadBuilder = Callable[[str], BackgroundRunPayload]


@dataclass(frozen=True)
class BackgroundRunRequest:
    """Description of one on-demand background run."""

    graph_id: str
    run_payload: BackgroundRunPayloadBuilder
    thread_metadata: Mapping[str, str] | None = None
    url: str | None = None
    headers: Mapping[str, str] | None = None
    name: str = "background run"


@dataclass(frozen=True)
class BackgroundRun:
    """Identifiers for a submitted background run."""

    name: str
    url: str
    graph_id: str
    thread_id: str
    run_id: str
    assistant_id: str
    metadata: Mapping[str, str]


@dataclass(frozen=True)
class BackgroundRunHooks:
    """Lifecycle hooks for caller-specific accounting."""

    on_before_run: Callable[[str], None] | None = None
    on_started: Callable[[BackgroundRun], None] | None = None
    on_finished: Callable[[BackgroundRun], None] | None = None
    on_aborted: Callable[[BackgroundRun], None] | None = None
    on_status_unknown: Callable[[BackgroundRun], None] | None = None
    on_watcher_start_failed: Callable[[BackgroundRun], None] | None = None


@dataclass(frozen=True)
class BackgroundRunWatcherConfig:
    """Polling behavior for a background run."""

    terminal_statuses: frozenset[str] = DEFAULT_BACKGROUND_RUN_TERMINAL_STATUSES
    poll_interval_seconds: float = DEFAULT_BACKGROUND_RUN_POLL_INTERVAL_SECONDS
    max_poll_failures: int = DEFAULT_BACKGROUND_RUN_MAX_POLL_FAILURES
    delete_thread_on_finish: bool = True


def default_background_run_url() -> str:
    """Return the configured local ``langgraph dev`` URL."""
    return configured_langgraph_dev_url()


def _headers(headers: Mapping[str, str] | None) -> dict[str, str]:
    return langgraph_dev_headers(headers)


def _create_thread(
    client: SyncLangGraphClient,
    *,
    graph_id: str,
    metadata: dict[str, str],
) -> str:
    thread = client.threads.create(graph_id=graph_id, metadata=metadata)
    return thread["thread_id"]


async def _acreate_thread(
    client: AsyncLangGraphClient,
    *,
    graph_id: str,
    metadata: dict[str, str],
) -> str:
    thread = await client.threads.create(graph_id=graph_id, metadata=metadata)
    return thread["thread_id"]


def _create_run(
    client: SyncLangGraphClient,
    *,
    thread_id: str,
    payload: BackgroundRunPayload,
) -> str:
    run = client.runs.create(
        thread_id=thread_id,
        assistant_id=payload["assistant_id"],
        input=payload["input"],
        metadata=payload["metadata"],
        config=payload["config"],
    )
    return run["run_id"]


async def _acreate_run(
    client: AsyncLangGraphClient,
    *,
    thread_id: str,
    payload: BackgroundRunPayload,
) -> str:
    run = await client.runs.create(
        thread_id=thread_id,
        assistant_id=payload["assistant_id"],
        input=payload["input"],
        metadata=payload["metadata"],
        config=payload["config"],
    )
    return run["run_id"]


def _get_run_status(
    client: SyncLangGraphClient,
    *,
    thread_id: str,
    run_id: str,
) -> str:
    run = client.runs.get(thread_id=thread_id, run_id=run_id)
    return run["status"]


async def _aget_run_status(
    client: AsyncLangGraphClient,
    *,
    thread_id: str,
    run_id: str,
) -> str:
    run = await client.runs.get(thread_id=thread_id, run_id=run_id)
    return run["status"]


def _delete_thread(
    client: SyncLangGraphClient,
    thread_id: str,
    *,
    name: str,
) -> None:
    try:
        client.threads.delete(thread_id)
    except Exception:
        logger.debug("Failed to delete %s thread %s", name, thread_id, exc_info=True)


async def _adelete_thread(
    client: AsyncLangGraphClient,
    thread_id: str,
    *,
    name: str,
) -> None:
    try:
        await client.threads.delete(thread_id)
    except Exception:
        logger.debug("Failed to delete %s thread %s", name, thread_id, exc_info=True)


def _background_run_handle(
    *,
    request: BackgroundRunRequest,
    url: str,
    thread_id: str,
    run_id: str,
    payload: BackgroundRunPayload,
) -> BackgroundRun:
    return BackgroundRun(
        name=request.name,
        url=url,
        graph_id=request.graph_id,
        thread_id=thread_id,
        run_id=run_id,
        assistant_id=payload["assistant_id"],
        metadata=dict(payload["metadata"]),
    )


def _call_hook(
    callback: Callable[[BackgroundRun], None] | None,
    run: BackgroundRun,
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


def _terminal_status_succeeded(status: str | None) -> bool:
    return str(status or "").strip().lower() == "success"


async def _acall_hook(
    callback: Callable[[BackgroundRun], None] | None,
    run: BackgroundRun,
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


def launch_background_run(
    request: BackgroundRunRequest,
    *,
    hooks: BackgroundRunHooks | None = None,
    watcher_config: BackgroundRunWatcherConfig | None = None,
    spawn_status_watcher: Callable[[BackgroundRun], None] | None = None,
) -> BackgroundRun | None:
    """Submit a background run to the local LangGraph server."""
    from langgraph_sdk import get_sync_client

    from ..langgraph_dev.manager import is_langgraph_dev_running

    hooks = hooks or BackgroundRunHooks()
    watcher_config = watcher_config or BackgroundRunWatcherConfig()
    url = request.url or default_background_run_url()
    if not is_langgraph_dev_running(base_url=url):
        logger.info("Skipping %s launch; LangGraph dev is unavailable", request.name)
        return None

    client: SyncLangGraphClient = get_sync_client(
        url=url,
        headers=_headers(request.headers),
    )
    thread_id = _create_thread(
        client,
        graph_id=request.graph_id,
        metadata=dict(request.thread_metadata or {}),
    )
    try:
        _call_before_run_hook(
            hooks.on_before_run,
            thread_id,
            name=request.name,
        )
        payload = request.run_payload(thread_id)
        run_id = _create_run(
            client,
            thread_id=thread_id,
            payload=payload,
        )
    except Exception:
        _delete_thread(client, thread_id, name=request.name)
        raise

    handle = _background_run_handle(
        request=request,
        url=url,
        thread_id=thread_id,
        run_id=run_id,
        payload=payload,
    )
    _call_hook(hooks.on_started, handle, hook_name="on_started")
    try:
        if spawn_status_watcher is None:
            spawn_background_run_status_thread(
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


async def alaunch_background_run(
    request: BackgroundRunRequest,
    *,
    hooks: BackgroundRunHooks | None = None,
    watcher_config: BackgroundRunWatcherConfig | None = None,
    spawn_status_watcher: Callable[[BackgroundRun], None] | None = None,
) -> BackgroundRun | None:
    """Async variant of :func:`launch_background_run`."""
    from langgraph_sdk import get_client

    from ..langgraph_dev.manager import is_langgraph_dev_running

    hooks = hooks or BackgroundRunHooks()
    watcher_config = watcher_config or BackgroundRunWatcherConfig()
    url = request.url or default_background_run_url()
    if not await asyncio.to_thread(is_langgraph_dev_running, base_url=url):
        logger.info("Skipping %s launch; LangGraph dev is unavailable", request.name)
        return None

    client: AsyncLangGraphClient = get_client(
        url=url,
        headers=_headers(request.headers),
    )
    thread_id = await _acreate_thread(
        client,
        graph_id=request.graph_id,
        metadata=dict(request.thread_metadata or {}),
    )
    try:
        await _acall_before_run_hook(
            hooks.on_before_run,
            thread_id,
            name=request.name,
        )
        payload = request.run_payload(thread_id)
        run_id = await _acreate_run(
            client,
            thread_id=thread_id,
            payload=payload,
        )
    except Exception:
        await _adelete_thread(client, thread_id, name=request.name)
        raise

    handle = _background_run_handle(
        request=request,
        url=url,
        thread_id=thread_id,
        run_id=run_id,
        payload=payload,
    )
    await _acall_hook(hooks.on_started, handle, hook_name="on_started")
    try:
        if spawn_status_watcher is None:
            spawn_background_run_status_thread(
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


def spawn_background_run_status_thread(
    run: BackgroundRun,
    *,
    headers: Mapping[str, str] | None = None,
    hooks: BackgroundRunHooks | None = None,
    watcher_config: BackgroundRunWatcherConfig | None = None,
) -> None:
    """Poll a background run from a daemon thread."""
    thread = threading.Thread(
        target=watch_background_run_sync,
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
        name="evosci-background-run-status",
        daemon=True,
    )
    thread.start()


def watch_background_run_sync(
    *,
    url: str,
    thread_id: str,
    run_id: str,
    graph_id: str = "",
    assistant_id: str = "",
    metadata: Mapping[str, str] | None = None,
    name: str = "background run",
    headers: Mapping[str, str] | None = None,
    hooks: BackgroundRunHooks | None = None,
    watcher_config: BackgroundRunWatcherConfig | None = None,
) -> None:
    """Poll a submitted background run until it finishes or polling aborts."""
    from langgraph_sdk import get_sync_client

    hooks = hooks or BackgroundRunHooks()
    watcher_config = watcher_config or BackgroundRunWatcherConfig()
    run_ref = BackgroundRun(
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
    final_status: str | None = None
    client: SyncLangGraphClient | None = None
    try:
        client = get_sync_client(url=url, headers=_headers(headers))
        while True:
            try:
                status = _get_run_status(
                    client,
                    thread_id=thread_id,
                    run_id=run_id,
                )
                failures = 0
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
                final_status = status
                return
            time.sleep(watcher_config.poll_interval_seconds)
    finally:
        if confirmed_finished:
            if _terminal_status_succeeded(final_status):
                _call_hook(hooks.on_finished, run_ref, hook_name="on_finished")
            else:
                _call_hook(hooks.on_aborted, run_ref, hook_name="on_aborted")
            if watcher_config.delete_thread_on_finish and client is not None:
                _delete_thread(client, thread_id, name=name)
        else:
            _call_hook(
                hooks.on_status_unknown or hooks.on_aborted,
                run_ref,
                hook_name="on_status_unknown",
            )


def spawn_background_run_status_task(
    client: AsyncLangGraphClient,
    run: BackgroundRun,
    *,
    hooks: BackgroundRunHooks | None = None,
    watcher_config: BackgroundRunWatcherConfig | None = None,
) -> None:
    """Poll a background run without blocking the event loop."""
    task = asyncio.create_task(
        awatch_background_run(
            client,
            url=run.url,
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
    _background_run_watcher_tasks.add(task)
    task.add_done_callback(_background_run_watcher_tasks.discard)


async def awatch_background_run(
    client: AsyncLangGraphClient,
    *,
    url: str = "",
    thread_id: str,
    run_id: str,
    graph_id: str = "",
    assistant_id: str = "",
    metadata: Mapping[str, str] | None = None,
    name: str = "background run",
    hooks: BackgroundRunHooks | None = None,
    watcher_config: BackgroundRunWatcherConfig | None = None,
) -> None:
    """Async status watcher for callers that already hold an async SDK client."""
    hooks = hooks or BackgroundRunHooks()
    watcher_config = watcher_config or BackgroundRunWatcherConfig()
    run_ref = BackgroundRun(
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
    final_status: str | None = None
    try:
        while True:
            try:
                status = await _aget_run_status(
                    client,
                    thread_id=thread_id,
                    run_id=run_id,
                )
                failures = 0
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
                final_status = status
                return
            await asyncio.sleep(watcher_config.poll_interval_seconds)
    finally:
        if confirmed_finished:
            if _terminal_status_succeeded(final_status):
                await _acall_hook(hooks.on_finished, run_ref, hook_name="on_finished")
            else:
                await _acall_hook(hooks.on_aborted, run_ref, hook_name="on_aborted")
            if watcher_config.delete_thread_on_finish:
                await _adelete_thread(client, thread_id, name=name)
        else:
            await _acall_hook(
                hooks.on_status_unknown or hooks.on_aborted,
                run_ref,
                hook_name="on_status_unknown",
            )
