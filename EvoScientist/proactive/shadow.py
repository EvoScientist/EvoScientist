"""In-memory shadow runner for proactive turns (issue #263).

A proactive "shadow" turn runs the *real* config-neutered agent graph over a
real chat's message history and produces a push/no-push text decision. Two
orthogonal properties make it safe to run against a live, populated setup:

- **Isolation** — the shadow uses an ``InMemorySaver`` checkpointer, so its
  thread state never touches ``sessions.db``; and a side-effect-neutered config
  (``_shadow_cfg``) disables the scheduler, the memory worker, and ask_user, so
  no writer survives outside the tool surface (an audit of the default
  middleware chain established that the tool surface plus these three flags is
  the whole side-effect axis: every other writer is either tool-backed and thus
  killed by the strip, or gated by one of these flags).
- **Tool-strip** — the turn runs with ``configurable.proactive_mode`` set, so
  ``ProactiveModeMiddleware`` strips every tool from the model request. This is
  the entire safety story, so it is enforced, not trusted: the shadow model is
  wrapped with a fail-closed spy that raises if any tool is ever bound.

The reasoning axis (system prompt, memory injection, model, context management)
is left intact — that is what makes the shadow's push/no-push judgment faithful
to what the real agent would decide.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any

from .. import sessions as session_store

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from ..config.settings import EvoScientistConfig

logger = logging.getLogger(__name__)


# --- fail-closed tool-strip interlock ----------------------------------------


class ProactiveToolLeakError(RuntimeError):
    """Raised when a proactive shadow turn binds a tool to the model.

    The proactive tool-strip is the whole safety story: a shadow turn must be a
    pure reasoning turn. This is the fail-closed interlock. On
    the correctly-stripped path langchain never calls ``bind_tools`` (it calls
    ``model.bind`` when the tool list is empty), so this only fires if the strip
    ever regresses — and when it does, we raise rather than let a tool execute
    against the real sandbox.
    """


class _ToolStripSpyMixin:
    """Mixin whose ``bind_tools`` refuses any non-empty tool list.

    Installed on the shadow chat model via ``__class__`` reassignment so the
    check rides on the exact model object langchain binds against
    (``request.model.bind_tools(...)``), independent of middleware ordering.
    """

    def bind_tools(self, tools, *args, **kwargs):  # type: ignore[override]
        if tools:
            names = [_tool_label(t) for t in tools]
            raise ProactiveToolLeakError(
                f"proactive shadow turn attempted to bind {len(tools)} tool(s): {names}"
            )
        return super().bind_tools(tools, *args, **kwargs)  # type: ignore[misc]


def _tool_label(tool: Any) -> str:
    """Best-effort human label for a tool in a leak error message."""
    name = getattr(tool, "name", None)
    if name:
        return str(name)
    if isinstance(tool, dict):
        return str(tool.get("name") or tool.get("type") or "dict-tool")
    return type(tool).__name__


def _install_tool_strip_spy(model: Any) -> Any:
    """Reassign ``model.__class__`` to a spy subclass and return the same model.

    The spy subclass adds only a ``bind_tools`` override (no new fields), so the
    reassignment is layout-compatible with the pydantic model and
    ``isinstance(model, <original class>)`` stays true.
    """
    spy_cls = type(
        f"ToolStripSpy_{type(model).__name__}",
        (_ToolStripSpyMixin, type(model)),
        {},
    )
    model.__class__ = spy_cls
    return model


# --- config neutering --------------------------------------------------------

# The side-effect axis: every flag here gates a writer identified by the
# middleware audit. The reasoning axis (prompt, memory injection, model, context) is left
# untouched. ``auto_mode=True`` declares "no human in the loop": it disables the
# first-contact profile bootstrap (which would ask a consent question AND write
# session/intro bookkeeping into the real profile file from every shadow turn)
# and ask_user. ``auto_approve=True`` is belt-and-braces (``__post_init__`` forces
# it under auto mode anyway).
_SHADOW_FLAGS: dict[str, Any] = {
    "enable_scheduler": False,
    "memory_workers_enabled": False,
    "enable_ask_user": False,
    "auto_mode": True,
    "auto_approve": True,
    "model_fallbacks": "",
}


def _shadow_cfg(cfg: EvoScientistConfig) -> EvoScientistConfig:
    """Return a side-effect-neutered copy of *cfg* for a shadow turn."""
    return dataclasses.replace(cfg, **_SHADOW_FLAGS)


# --- shadow graph + turn -----------------------------------------------------


def build_shadow_graph(
    cfg: EvoScientistConfig,
    chat_model: Any,
    workspace_dir: str,
) -> CompiledStateGraph:
    """Build the config-neutered shadow graph with the tool-strip spy installed.

    ``chat_model`` is spied in place, then passed with ``_shadow_cfg(cfg)`` to
    ``create_cli_agent`` on its pure path (both ``config`` and ``chat_model``
    non-None → no module globals are written). ``workspace_dir`` MUST be the
    live setup's REAL workspace so profile-file seeding on the memory read path
    is a no-op. Build once per process and reuse across checks.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    from ..EvoScientist import create_cli_agent

    _install_tool_strip_spy(chat_model)
    return create_cli_agent(
        workspace_dir=workspace_dir,
        checkpointer=InMemorySaver(),
        config=_shadow_cfg(cfg),
        chat_model=chat_model,
    )


async def run_shadow_turn(
    graph: CompiledStateGraph,
    source_messages: list[Any],
    trigger: str,
    *,
    thread_id: str | None = None,
) -> str | None:
    """Run one proactive shadow turn and return the terminal assistant text.

    Seeds ``source_messages`` into a fresh in-memory thread, then drains a
    ``proactive_mode`` turn driven by ``trigger``. Returns the terminal ``done``
    text (possibly empty — the NO_PUSH/empty decision belongs to the caller).
    Returns ``None`` on any error, including a ``ProactiveToolLeakError`` from
    the spy, and when the model did not finish cleanly (a truncated, errored, or
    filtered generation is never a message to send): the caller treats ``None``
    as "abort, do not push" (fail-closed). The shadow thread is deleted from the
    in-memory checkpointer afterwards to bound memory across reused checks.
    """
    from ..stream.events import stream_agent_events

    tid = thread_id or session_store.generate_thread_id()
    final = ""
    try:
        await graph.aupdate_state(
            {"configurable": {"thread_id": tid}},
            {"messages": source_messages},
        )
        async for event in stream_agent_events(
            graph,
            trigger,
            tid,
            configurable_extra={"proactive_mode": True},
        ):
            etype = event.get("type")
            if etype == "text":
                final += event.get("content", "")
            elif etype == "done":
                final = event.get("content", "") or final
        last = await _last_message(graph, tid)
        if not final.strip():
            _log_empty_reply(last)
        if not _finished_cleanly(last):
            return None
    except ProactiveToolLeakError:
        # The interlock fired — a tool reached the model. Never deliver.
        logger.warning(
            "proactive shadow turn aborted: tool-strip regressed (spy fired)",
            exc_info=True,
        )
        return None
    except Exception:
        logger.warning("proactive shadow turn failed", exc_info=True)
        return None
    finally:
        await _delete_shadow_thread(graph, tid)
    return final


# ``finish_reason`` values that mean the generation is complete. Anything else
# (``length``, ``error``, ``content_filter``, provider-specific codes) is not a
# message the user should receive. Absent metadata is treated as clean, so
# providers that do not report a finish reason still work.
_CLEAN_FINISH_REASONS = frozenset({"stop", "end_turn", "STOP"})


async def _last_message(graph: CompiledStateGraph, thread_id: str) -> Any | None:
    """The shadow thread's last message, or None if the state cannot be read."""
    try:
        state = await graph.aget_state({"configurable": {"thread_id": thread_id}})
        values = getattr(state, "values", None) or {}
        messages = values.get("messages") or []
        return messages[-1] if messages else None
    except Exception:
        logger.debug("could not read shadow state after the turn", exc_info=True)
        return None


def _finish_reason(message: Any | None) -> str | None:
    metadata = getattr(message, "response_metadata", None) or {}
    reason = metadata.get("finish_reason") if isinstance(metadata, dict) else None
    return str(reason) if reason else None


def _finished_cleanly(message: Any | None) -> bool:
    """False when the model reports a non-clean finish (truncated / errored)."""
    reason = _finish_reason(message)
    if reason is None or reason in _CLEAN_FINISH_REASONS:
        return True
    logger.warning(
        "proactive shadow turn discarded: model finish_reason=%r (not a clean "
        "stop); response_metadata=%s",
        reason,
        {
            k: str(v)[:200]
            for k, v in (getattr(message, "response_metadata", None) or {}).items()
        },
    )
    return False


def _log_empty_reply(last: Any | None) -> None:
    """INFO-log the raw last message when the shadow produced no text.

    The shadow thread is deleted right after the turn, so this is the only place
    a reasoning-only response, a refusal, or a provider quirk can be told apart
    from a genuine NO_PUSH.
    """
    logger.info(
        "proactive shadow turn returned no text; last message=%s content=%r "
        "additional_kwargs=%s response_metadata=%s",
        type(last).__name__,
        str(getattr(last, "content", ""))[:300],
        {
            k: str(v)[:200]
            for k, v in (getattr(last, "additional_kwargs", None) or {}).items()
        },
        {
            k: str(v)[:200]
            for k, v in (getattr(last, "response_metadata", None) or {}).items()
        },
    )


async def _delete_shadow_thread(graph: CompiledStateGraph, thread_id: str) -> None:
    """Best-effort deletion of the shadow thread from the in-memory saver."""
    checkpointer = getattr(graph, "checkpointer", None)
    adelete = getattr(checkpointer, "adelete_thread", None)
    if adelete is None:
        return
    try:
        await adelete(thread_id)
    except Exception:
        logger.debug("failed to delete shadow thread %s", thread_id, exc_info=True)
