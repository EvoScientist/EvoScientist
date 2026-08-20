"""Runtime selection for streaming UI backends."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..gateway import GraphGateway
from ..runtime import AsyncRuntimeError
from ..stream.console import console
from .tui_backends import RichStreamingBackend, StreamingTUIBackend

if TYPE_CHECKING:
    from ..runtime import AsyncRuntime

DEFAULT_UI_BACKEND = "cli"
STREAM_CANCEL_SETTLE_TIMEOUT = 5.0
# "webui" launches the browser front-end instead of an in-terminal UI; it is
# intercepted earlier (cli/commands.py:_main_callback) and never reaches the
# streaming backends, but is listed here so normalize/resolve preserve it
# rather than falling back to "cli".
SUPPORTED_UI_BACKENDS = ("cli", "tui", "webui")
_LEGACY_BACKEND_MAP = {"textual": "tui", "rich": "cli"}


class StreamCancellationTimeout(RuntimeError):
    """A blocking renderer did not settle after its turn was cancelled."""


def _consume_late_worker_result(worker: asyncio.Task[Any]) -> None:
    """Retrieve a detached worker result so eventual failure is not unhandled."""
    try:
        worker.exception()
    except asyncio.CancelledError:
        pass


async def settle_cancelled_worker(
    worker: asyncio.Task[Any],
    *,
    on_cancel: Callable[[], Any],
) -> Any:
    """Request cooperative cancellation and wait a bounded time for settlement."""
    on_cancel()
    done, _ = await asyncio.wait(
        {worker},
        timeout=STREAM_CANCEL_SETTLE_TIMEOUT,
    )
    if not done:
        worker.add_done_callback(_consume_late_worker_result)
        raise StreamCancellationTimeout(
            "The active turn did not stop within "
            f"{STREAM_CANCEL_SETTLE_TIMEOUT:g} seconds after cancellation."
        )
    try:
        return worker.result()
    except Exception:
        return ""


def normalize_ui_backend(value: str | None) -> str:
    """Normalize user-provided backend name with a safe default."""
    if not value:
        return DEFAULT_UI_BACKEND
    normalized = value.strip().lower()
    normalized = _LEGACY_BACKEND_MAP.get(normalized, normalized)
    if normalized in SUPPORTED_UI_BACKENDS:
        return normalized
    return DEFAULT_UI_BACKEND


def _has_textual_support() -> bool:
    try:
        import textual  # noqa: F401

        return True
    except Exception:
        return False


def resolve_ui_backend(value: str | None, *, warn_fallback: bool = False) -> str:
    """Resolve requested backend and fallback safely when unavailable."""
    requested = normalize_ui_backend(value)
    if requested == "tui" and not _has_textual_support():
        if warn_fallback:
            console.print(
                "[yellow]TUI is unavailable (missing textual package). "
                "Falling back to CLI.[/yellow]"
            )
        return DEFAULT_UI_BACKEND
    return requested


def get_backend(
    name: str | None, *, warn_fallback: bool = False
) -> StreamingTUIBackend:
    """Instantiate a streaming backend by name.

    Note: The Textual TUI is now a full interactive app (tui_interactive.py),
    not a streaming backend.  The streaming backend is always Rich.
    """
    resolve_ui_backend(name, warn_fallback=warn_fallback)
    return RichStreamingBackend()


def run_streaming(
    *,
    ui_backend: str | None,
    agent: Any,
    message: str,
    thread_id: str,
    show_thinking: bool,
    interactive: bool,
    on_thinking: Callable[[str], None] | None = None,
    on_todo: Callable[[list[dict]], None] | None = None,
    on_file_write: Callable[[str], None] | None = None,
    on_stream_event: Callable[[str, Any], Any] | None = None,
    status_footer_builder: Callable[[], Any] | None = None,
    metadata: dict | None = None,
    configurable_extra: dict[str, Any] | None = None,
    hitl_prompt_fn: Callable[[list], list[dict] | None] | None = None,
    ask_user_prompt_fn: Callable[[dict], dict] | None = None,
    cancel_scope: str | None = None,
    gateway: GraphGateway,
    runtime: AsyncRuntime | None = None,
) -> str:
    """Run streaming with the selected backend."""
    backend = get_backend(ui_backend, warn_fallback=True)
    try:
        return backend.run_streaming(
            agent=agent,
            message=message,
            thread_id=thread_id,
            show_thinking=show_thinking,
            interactive=interactive,
            on_thinking=on_thinking,
            on_todo=on_todo,
            on_file_write=on_file_write,
            on_stream_event=on_stream_event,
            status_footer_builder=status_footer_builder,
            metadata=metadata,
            configurable_extra=configurable_extra,
            hitl_prompt_fn=hitl_prompt_fn,
            ask_user_prompt_fn=ask_user_prompt_fn,
            cancel_scope=cancel_scope,
            gateway=gateway,
            runtime=runtime,
        )
    except AsyncRuntimeError:
        raise
    except RuntimeError:
        requested = normalize_ui_backend(ui_backend)
        if requested == "tui":
            console.print(
                "[yellow]TUI failed at runtime. Falling back to CLI for this request.[/yellow]"
            )
            return RichStreamingBackend().run_streaming(
                agent=agent,
                message=message,
                thread_id=thread_id,
                show_thinking=show_thinking,
                interactive=interactive,
                on_thinking=on_thinking,
                on_todo=on_todo,
                on_file_write=on_file_write,
                on_stream_event=on_stream_event,
                status_footer_builder=status_footer_builder,
                metadata=metadata,
                configurable_extra=configurable_extra,
                hitl_prompt_fn=hitl_prompt_fn,
                ask_user_prompt_fn=ask_user_prompt_fn,
                cancel_scope=cancel_scope,
                gateway=gateway,
                runtime=runtime,
            )
        raise


async def run_streaming_async(
    *,
    recover_on_cancel: bool = False,
    **kwargs: Any,
) -> str:
    """Run the synchronous Rich renderer without blocking a frontend loop.

    Cancellation requests the matching stream scope and gives the worker a
    bounded interval to unwind.  Foreground interactive turns may opt into
    recovering the frontend task after cleanup so Ctrl+C returns to the prompt.
    """
    from ..stream.display import request_stream_cancel

    worker = asyncio.create_task(asyncio.to_thread(run_streaming, **kwargs))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        try:
            response = await settle_cancelled_worker(
                worker,
                on_cancel=lambda: request_stream_cancel(kwargs.get("cancel_scope")),
            )
        finally:
            from ..middleware.code_interpreter import aclose_code_interpreters

            await aclose_code_interpreters()
        if recover_on_cancel:
            current = asyncio.current_task()
            if current is not None and current.uncancel() > 0:
                raise
            return response
        raise
