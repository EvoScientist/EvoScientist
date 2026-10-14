"""The shadow turn: the real agent, in memory, with every tool stripped.

A shadow turn runs the configured agent over a copy of a conversation's
messages and returns the text it would send. Three properties make that safe to
run against a live setup:

- **In memory, outside the calling run.** The shadow graph has its own
  ``InMemorySaver`` and its own ``thread_id``, and it runs in a fresh
  ``contextvars.Context``. A child graph that inherits the context of the cron
  run calling it streams its model tokens on that run (``stream_mode=
  "messages"``), and, invoked without a config, writes its checkpoints into
  the parent's checkpointer under a sub-namespace, which would copy the source
  conversation into the cron thread.
- **No side effects.** ``_shadow_cfg`` turns off the scheduler, the memory
  workers and ``ask_user``, and sets ``auto_mode`` (no profile bootstrap, no
  approval prompts). Every other writer is a tool.
- **No tools.** The run carries ``configurable.proactive_mode``, so
  ``ProactiveModeMiddleware`` strips every tool from the model request. The
  shadow model also refuses to bind any tool, so a regression in the strip
  fails the turn instead of running a tool against the real sandbox.

Prompt, memory injection, model and context management are the real ones: the
point is to judge as the real agent would.
"""

from __future__ import annotations

import asyncio
import contextvars
import dataclasses
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from ..config.settings import EvoScientistConfig
    from ..paths import SessionDirs

logger = logging.getLogger(__name__)


class ProactiveToolLeakError(RuntimeError):
    """A shadow turn tried to bind a tool to the model.

    On the stripped path langchain never calls ``bind_tools`` (an empty tool
    list skips it), so this only fires if the strip regresses.
    """


class _ToolStripSpyMixin:
    """``bind_tools`` that refuses any non-empty tool list."""

    def bind_tools(self, tools, *args, **kwargs):  # type: ignore[override]
        if tools:
            names = [_tool_label(t) for t in tools]
            raise ProactiveToolLeakError(
                f"proactive shadow turn tried to bind {len(tools)} tool(s): {names}"
            )
        return super().bind_tools(tools, *args, **kwargs)  # type: ignore[misc]


def _tool_label(tool: Any) -> str:
    name = getattr(tool, "name", None)
    if name:
        return str(name)
    if isinstance(tool, dict):
        return str(tool.get("name") or tool.get("type") or "dict-tool")
    return type(tool).__name__


def install_tool_strip_spy(model: Any) -> Any:
    """Make *model* refuse to bind tools, in place, and return it.

    Reassigns ``model.__class__`` to a subclass that only overrides
    ``bind_tools``, so the pydantic layout is unchanged and
    ``isinstance(model, <original class>)`` still holds.
    """
    spy_cls = type(
        f"ToolStripSpy_{type(model).__name__}",
        (_ToolStripSpyMixin, type(model)),
        {},
    )
    model.__class__ = spy_cls
    return model


# The config flags a shadow turn overrides. The first three turn off the
# writers that are not tools; ``auto_mode`` / ``auto_approve`` declare that no
# human is present (no profile bootstrap question, no approval prompt), and
# ``model_fallbacks`` pins the turn to the primary model.
_SHADOW_FLAGS: dict[str, Any] = {
    "enable_scheduler": False,
    "memory_workers_enabled": False,
    "enable_ask_user": False,
    "auto_mode": True,
    "auto_approve": True,
    "model_fallbacks": "",
}


def shadow_cfg(cfg: EvoScientistConfig) -> EvoScientistConfig:
    """A copy of *cfg* with the shadow turn's overrides."""
    return dataclasses.replace(cfg, **_SHADOW_FLAGS)


def build_shadow_graph(
    cfg: EvoScientistConfig,
    chat_model: Any,
    dirs: SessionDirs,
) -> CompiledStateGraph:
    """Build the shadow agent for a conversation's folders.

    Blocking (the agent build reads the filesystem and may load MCP tools):
    call it off the event loop. *chat_model* gets the tool-strip spy, so pass
    a model built for the shadow, not one shared with other agents. Both
    ``config`` and ``chat_model`` are explicit, so ``create_cli_agent`` writes
    none of its cached module globals.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    from ..EvoScientist import create_cli_agent

    install_tool_strip_spy(chat_model)
    return create_cli_agent(
        work_dir=dirs.work_dir,
        checkpointer=InMemorySaver(),
        config=shadow_cfg(cfg),
        chat_model=chat_model,
        workspace=dirs.workspace,
    )


async def run_shadow_turn(
    graph: CompiledStateGraph,
    source_messages: list[Any],
    trigger: str,
) -> str | None:
    """Run one shadow turn and return the text of its final assistant message.

    Returns ``None``, meaning "do not push", when the model answered but not
    with a message to send: the turn ends without a final assistant message,
    that message asks for a tool, the model did not finish cleanly (truncated,
    errored or filtered), or a tool reached the model. Any other failure (a
    provider error, a timeout) is raised: the model gave no answer, so the
    caller must not count the turn as an evaluation. The returned text may be
    empty or ``NO_PUSH``; judging it is the caller's job.
    """
    task = asyncio.get_running_loop().create_task(
        _run_shadow_turn(graph, flatten_tool_history(source_messages), trigger),
        context=contextvars.Context(),
    )
    return await task


def flatten_tool_history(messages: list[Any]) -> list[Any]:
    """Rewrite tool calls and tool results in *messages* as plain text.

    The shadow model is called with no tools. Some providers (Anthropic among
    them) refuse a request whose history holds ``tool_use`` / ``tool_result``
    blocks but defines no tools, so every shadow turn on them would fail. As
    text, the history keeps what was done and what came back, and no provider
    sees a tool block.
    """
    flattened: list[Any] = []
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            lines = [message.text] if message.text.strip() else []
            for call in message.tool_calls:
                args = json.dumps(
                    call.get("args") or {}, ensure_ascii=False, default=str
                )
                lines.append(f"[called tool `{call.get('name')}` with {args}]")
            flattened.append(
                AIMessage(
                    content="\n".join(lines),
                    id=message.id,
                    additional_kwargs={
                        k: v
                        for k, v in message.additional_kwargs.items()
                        if k not in ("tool_calls", "function_call")
                    },
                )
            )
        elif isinstance(message, ToolMessage):
            name = message.name or "tool"
            flattened.append(
                HumanMessage(
                    content=f"[result of tool `{name}`]\n{_text_of(message)}",
                    id=message.id,
                )
            )
        else:
            flattened.append(message)
    return flattened


def _text_of(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return message.text or json.dumps(content, ensure_ascii=False, default=str)


async def _run_shadow_turn(
    graph: CompiledStateGraph,
    source_messages: list[Any],
    trigger: str,
) -> str | None:
    thread_id = f"proactive-shadow-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id, "proactive_mode": True}}
    try:
        result = await graph.ainvoke(
            {"messages": [*source_messages, HumanMessage(content=trigger)]},
            config,
        )
        messages = (result or {}).get("messages") or []
        last = messages[-1] if messages else None
        if not isinstance(last, AIMessage) or last.tool_calls:
            logger.warning(
                "proactive shadow turn discarded: it did not end on a final "
                "assistant message (last=%s)",
                type(last).__name__,
            )
            return None
        if not _finished_cleanly(last):
            return None
        text = last.text
        if not text.strip():
            _log_empty_reply(last)
        return text
    except ProactiveToolLeakError:
        logger.warning(
            "proactive shadow turn aborted: the tool strip regressed", exc_info=True
        )
        return None
    finally:
        await _delete_shadow_thread(graph, thread_id)


# Finish reasons that mean the generation completed. Anything else (``length``,
# ``max_tokens``, ``error``, ``content_filter``, ...) is not a message to send.
# A model that reports no finish reason counts as clean.
_CLEAN_FINISH_REASONS = frozenset({"stop", "end_turn", "STOP", "stop_sequence"})


def _finish_reason(message: Any) -> str | None:
    # OpenAI-style providers write ``finish_reason``, Anthropic ``stop_reason``;
    # reading only one would let an Anthropic ``max_tokens`` cut pass as clean.
    metadata = getattr(message, "response_metadata", None) or {}
    if not isinstance(metadata, dict):
        return None
    reason = metadata.get("finish_reason") or metadata.get("stop_reason")
    return str(reason) if reason else None


def _finished_cleanly(message: Any) -> bool:
    reason = _finish_reason(message)
    if reason is None or reason in _CLEAN_FINISH_REASONS:
        return True
    metadata = getattr(message, "response_metadata", None) or {}
    logger.warning(
        "proactive shadow turn discarded: finish reason %r; model=%s",
        reason,
        metadata.get("model_name") if isinstance(metadata, dict) else None,
    )
    return False


def _log_empty_reply(last: Any) -> None:
    """Log the shape of an empty reply (never its content).

    The shadow thread is deleted after the turn, so this is the only trace that
    tells a reasoning-only response or a provider quirk from a plain empty one.
    """
    metadata = getattr(last, "response_metadata", None) or {}
    logger.info(
        "proactive shadow turn returned no text; content_chars=%d "
        "additional_kwargs_keys=%s finish_reason=%s model=%s",
        len(str(getattr(last, "content", "") or "")),
        sorted((getattr(last, "additional_kwargs", None) or {}).keys()),
        _finish_reason(last),
        metadata.get("model_name") if isinstance(metadata, dict) else None,
    )


async def _delete_shadow_thread(graph: CompiledStateGraph, thread_id: str) -> None:
    """Drop the shadow thread from the in-memory saver, best effort."""
    adelete = getattr(getattr(graph, "checkpointer", None), "adelete_thread", None)
    if adelete is None:
        return
    try:
        await adelete(thread_id)
    except Exception:
        logger.debug("could not delete shadow thread %s", thread_id, exc_info=True)
