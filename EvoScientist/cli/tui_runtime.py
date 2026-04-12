"""Runtime selection for streaming UI backends."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..stream.display import console
from .tui_backends import RichStreamingBackend, StreamingTUIBackend

DEFAULT_UI_BACKEND = "cli"
SUPPORTED_UI_BACKENDS = ("cli", "tui")
_LEGACY_BACKEND_MAP = {"textual": "tui", "rich": "cli"}


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
    hitl_prompt_fn: Callable[[list], list[dict] | None] | None = None,
    ask_user_prompt_fn: Callable[[dict], dict] | None = None,
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
            hitl_prompt_fn=hitl_prompt_fn,
            ask_user_prompt_fn=ask_user_prompt_fn,
        )
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
                hitl_prompt_fn=hitl_prompt_fn,
                ask_user_prompt_fn=ask_user_prompt_fn,
            )
        raise
