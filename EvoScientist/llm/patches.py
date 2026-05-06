"""Monkey-patches and utilities for third-party LangChain provider quirks.

All patches follow the same pattern: wrap an existing method/function to
fix upstream bugs, applied at import time or on first use.

Patches:
    - _patch_anthropic_proxy_compat: ccproxy dict→Pydantic model mismatch
    - _patch_openai_compat_content: list content→string for strict APIs
    - _patch_ccproxy_codex_compat: ccproxy model fixes + langchain None guard
    - _patch_ccproxy_system_to_developer: system→developer role for ccproxy
    - _patch_openai_capture_reasoning_content: capture provider
      reasoning_content into AIMessage.additional_kwargs (module-level,
      applied at import)
    - _patch_deepseek_reasoning_passback: re-inject reasoning_content into
      outgoing DeepSeek assistant messages for thinking-mode multi-turn /
      tool_use scenarios

Utilities:
    - _is_ccproxy_codex: detect ccproxy Codex OAuth adapter
    - _flatten_message_content: convert content blocks to plain string
"""

from __future__ import annotations

import os
from typing import Any


# ---------------------------------------------------------------------------
# Patch: langchain-anthropic (>=1.3.4) calls .model_dump() on
# context_management / container objects returned by the Anthropic SDK.
# Proxies like ccproxy may return plain dicts which lack that method.
# We wrap the class method to pre-convert dicts before the original runs.
# ---------------------------------------------------------------------------
def _patch_anthropic_proxy_compat() -> None:
    try:
        import types as _types

        from langchain_anthropic.chat_models import ChatAnthropic as _CA

        _orig = _CA._make_message_chunk_from_anthropic_event

        def _safe(self: Any, event: Any, *args: Any, **kwargs: Any) -> Any:
            for obj, attrs in [
                (event, ("context_management",)),
                (getattr(event, "delta", None), ("container",)),
            ]:
                if obj is None:
                    continue
                for attr in attrs:
                    val = getattr(obj, attr, None)
                    if isinstance(val, dict):
                        d = val.copy()
                        setattr(
                            obj,
                            attr,
                            _types.SimpleNamespace(model_dump=lambda d=d, **kw: d),
                        )
            return _orig(self, event, *args, **kwargs)

        _CA._make_message_chunk_from_anthropic_event = _safe
    except Exception:
        pass


_patch_anthropic_proxy_compat()


# ---------------------------------------------------------------------------
# Patch: ccproxy-api 0.2.7 Codex compatibility.
#
# 1) ResponseObject.output is required but upstream may omit it → 502.
#    Fix: make output default to [].
# 2) CodexMessage.role only allows "user"/"assistant" → 400 on system msgs.
#    Fix: widen to also accept "system" and "developer".
# 3) langchain-openai iterates response.output which can be None after the
#    proxy strips it.  Fix: guard in _construct_lc_result_from_responses_api.
# ---------------------------------------------------------------------------
def _patch_ccproxy_codex_compat() -> None:
    """Patch ccproxy-api models for Responses API compatibility."""
    # 1) Make ResponseObject.output optional (default=[])
    try:
        import ccproxy.llms.models.openai as _oai_mod

        _OrigResponse = _oai_mod.ResponseObject

        from pydantic import Field as _PydanticField

        class _PatchedResponseObject(_OrigResponse):  # type: ignore[misc]
            output: list = _PydanticField(default_factory=list)  # type: ignore[assignment]

            model_config = _OrigResponse.model_config.copy()

        _PatchedResponseObject.__name__ = "ResponseObject"
        _PatchedResponseObject.__qualname__ = "ResponseObject"
        _oai_mod.ResponseObject = _PatchedResponseObject  # type: ignore[misc]

        # Also patch modules that import ResponseObject directly
        for _mod_path in (
            "ccproxy.llms.formatters.openai_to_openai.responses",
            "ccproxy.llms.formatters.anthropic_to_openai.responses",
        ):
            try:
                import importlib

                _mod = importlib.import_module(_mod_path)
                if hasattr(_mod, "ResponseObject"):
                    _mod.ResponseObject = _PatchedResponseObject  # type: ignore[attr-defined]
            except Exception:
                pass
    except Exception:
        pass

    # 2) Widen CodexMessage.role to accept system/developer
    try:
        from typing import Annotated, Literal

        import ccproxy.plugins.codex.models as _codex_mod

        _OrigMessage = _codex_mod.CodexMessage

        from pydantic import Field as _Field

        class _PatchedCodexMessage(_OrigMessage):  # type: ignore[misc]
            role: Annotated[  # type: ignore[assignment]
                Literal["user", "assistant", "system", "developer"],
                _Field(description="Message role"),
            ]

        _PatchedCodexMessage.__name__ = "CodexMessage"
        _PatchedCodexMessage.__qualname__ = "CodexMessage"
        _codex_mod.CodexMessage = _PatchedCodexMessage  # type: ignore[misc]
    except Exception:
        pass

    # 3) Fix StreamingBufferService returning response.completed event
    #    whose output is None/empty, instead of using accumulated outputs.
    try:
        from ccproxy.llms.streaming.accumulators import ResponsesAccumulator

        _orig_get = ResponsesAccumulator.get_completed_response

        def _patched_get(self: Any) -> dict | None:
            result = _orig_get(self)
            if result is not None:
                output = result.get("output")
                if output is None:
                    # output field lost — force rebuild from accumulated items
                    return None
            return result

        ResponsesAccumulator.get_completed_response = _patched_get  # type: ignore[assignment]
    except Exception:
        pass

    # 4) Guard langchain-openai against None output (final safety net)
    try:
        import langchain_openai.chat_models.base as _base

        _orig_construct = _base._construct_lc_result_from_responses_api

        def _safe(response: Any, *args: Any, **kwargs: Any) -> Any:
            if response.output is None:
                response.output = []
            return _orig_construct(response, *args, **kwargs)

        _base._construct_lc_result_from_responses_api = _safe
    except Exception:
        pass


_patch_ccproxy_codex_compat()


# ---------------------------------------------------------------------------
# Utility: detect ccproxy's Codex adapter (as opposed to generic localhost).
# ---------------------------------------------------------------------------
def _is_ccproxy_codex() -> bool:
    """Return True if the OpenAI endpoint is ccproxy's Codex adapter.

    Checks for the ccproxy-specific markers set by ``setup_codex_env()``
    in ``ccproxy_manager.py``: the sentinel API key and the ``/codex/v1``
    path.  Plain localhost endpoints (vLLM, Ollama, etc.) are not affected.
    """
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    return (
        ("127.0.0.1" in base_url or "localhost" in base_url)
        and api_key == "ccproxy-oauth"
        and "/codex/" in base_url
    )


# ---------------------------------------------------------------------------
# Utility + Patch: Flatten list content to strings for OpenAI-compatible APIs.
# DeepSeek, SiliconFlow, etc. reject assistant messages whose content is a
# list rather than a string.
# ---------------------------------------------------------------------------
_SKIP_CONTENT_TYPES = frozenset({"thinking", "reasoning", "reasoning_content"})


def _flatten_message_content(content: Any) -> str | Any:
    """Convert list-of-blocks content to a plain string.

    Args:
        content: Message content — either a string, a list of content blocks
            (dicts with ``type`` and ``text`` keys), or another type.

    Returns:
        A plain string with text blocks joined by double newlines.
        Thinking/reasoning blocks are skipped.  Non-list input is
        returned unchanged.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") in _SKIP_CONTENT_TYPES:
                continue
            text = block.get("text")
            if text:
                parts.append(text)
        elif isinstance(block, str):
            parts.append(block)
    return "\n\n".join(parts) if parts else ""


def _patch_openai_compat_content(model: Any) -> None:
    """Flatten list content to strings before OpenAI-compatible API calls.

    Wraps ``_generate`` / ``_agenerate`` to prevent "invalid type: sequence,
    expected a string" errors from strict APIs like DeepSeek.

    Args:
        model: A LangChain chat model instance to patch in-place.
    """
    import copy
    import functools

    from langchain_core.messages import BaseMessage

    def _sanitize_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
        out: list[BaseMessage] = []
        for msg in messages:
            if isinstance(msg.content, list):
                msg = copy.copy(msg)
                msg.content = _flatten_message_content(msg.content)
            out.append(msg)
        return out

    orig_generate = getattr(model, "_generate", None)
    if orig_generate is None:
        return

    @functools.wraps(orig_generate)
    def _patched_generate(
        messages: list[BaseMessage], *args: Any, **kwargs: Any
    ) -> Any:
        return orig_generate(_sanitize_messages(messages), *args, **kwargs)

    model._generate = _patched_generate

    orig_agenerate = getattr(model, "_agenerate", None)
    if orig_agenerate is not None:

        @functools.wraps(orig_agenerate)
        async def _patched_agenerate(
            messages: list[BaseMessage], *args: Any, **kwargs: Any
        ) -> Any:
            return await orig_agenerate(_sanitize_messages(messages), *args, **kwargs)

        model._agenerate = _patched_agenerate

    # Also patch streaming paths — CLI/agent uses _stream/_astream, so without
    # these the content flattening is bypassed during normal streaming calls.
    orig_stream = getattr(model, "_stream", None)
    if orig_stream is not None:

        @functools.wraps(orig_stream)
        def _patched_stream(
            messages: list[BaseMessage], *args: Any, **kwargs: Any
        ) -> Any:
            return orig_stream(_sanitize_messages(messages), *args, **kwargs)

        model._stream = _patched_stream

    orig_astream = getattr(model, "_astream", None)
    if orig_astream is not None:

        @functools.wraps(orig_astream)
        async def _patched_astream(
            messages: list[BaseMessage], *args: Any, **kwargs: Any
        ) -> Any:
            async for chunk in orig_astream(
                _sanitize_messages(messages), *args, **kwargs
            ):
                yield chunk

        model._astream = _patched_astream


# ---------------------------------------------------------------------------
# Patch: ccproxy Codex Responses API rejects "system" role messages.
# Convert SystemMessage to use "developer" role via langchain-openai's
# __openai_role__ mechanism.
# ---------------------------------------------------------------------------
def _patch_ccproxy_system_to_developer(model: Any) -> None:
    """Convert SystemMessage role from 'system' to 'developer' for ccproxy.

    ccproxy's Responses API endpoint rejects system role messages with
    400 "System messages are not allowed".  LangChain's ``langchain_openai``
    checks ``additional_kwargs["__openai_role__"]`` and uses that value as
    the message role when serializing to the API.

    Args:
        model: A LangChain chat model instance to patch in-place.
    """
    import copy
    import functools

    from langchain_core.messages import BaseMessage, SystemMessage

    def _system_to_developer(messages: list[BaseMessage]) -> list[BaseMessage]:
        out: list[BaseMessage] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                if msg.additional_kwargs.get("__openai_role__") != "developer":
                    msg = copy.copy(msg)
                    msg.additional_kwargs = {
                        **msg.additional_kwargs,
                        "__openai_role__": "developer",
                    }
            out.append(msg)
        return out

    orig_generate = getattr(model, "_generate", None)
    if orig_generate is None:
        return

    @functools.wraps(orig_generate)
    def _patched_generate(
        messages: list[BaseMessage], *args: Any, **kwargs: Any
    ) -> Any:
        return orig_generate(_system_to_developer(messages), *args, **kwargs)

    model._generate = _patched_generate

    orig_agenerate = getattr(model, "_agenerate", None)
    if orig_agenerate is not None:

        @functools.wraps(orig_agenerate)
        async def _patched_agenerate(
            messages: list[BaseMessage], *args: Any, **kwargs: Any
        ) -> Any:
            return await orig_agenerate(_system_to_developer(messages), *args, **kwargs)

        model._agenerate = _patched_agenerate

    orig_stream = getattr(model, "_stream", None)
    if orig_stream is not None:

        @functools.wraps(orig_stream)
        def _patched_stream(
            messages: list[BaseMessage], *args: Any, **kwargs: Any
        ) -> Any:
            return orig_stream(_system_to_developer(messages), *args, **kwargs)

        model._stream = _patched_stream

    orig_astream = getattr(model, "_astream", None)
    if orig_astream is not None:

        @functools.wraps(orig_astream)
        async def _patched_astream(
            messages: list[BaseMessage], *args: Any, **kwargs: Any
        ) -> Any:
            async for chunk in orig_astream(
                _system_to_developer(messages), *args, **kwargs
            ):
                yield chunk

        model._astream = _patched_astream


# ---------------------------------------------------------------------------
# Patch (module-level): langchain-openai's _convert_dict_to_message and
# _convert_delta_to_message_chunk discard provider-specific fields like
# `reasoning_content`. We monkey-patch them to capture reasoning_content
# into AIMessage.additional_kwargs so downstream code (incl. our passback
# patch) can find it. Benign for non-DeepSeek providers — they just don't
# return this field, so the patch is a no-op for them.
# ---------------------------------------------------------------------------
_openai_capture_patched = False


def _patch_openai_capture_reasoning_content() -> None:
    global _openai_capture_patched
    if _openai_capture_patched:
        return
    try:
        import langchain_openai.chat_models.base as _base

        _orig_dict_to_msg = _base._convert_dict_to_message
        _orig_delta_to_chunk = _base._convert_delta_to_message_chunk

        def _patched_dict_to_msg(_dict, *args, **kwargs):
            msg = _orig_dict_to_msg(_dict, *args, **kwargs)
            rc = _dict.get("reasoning_content") if isinstance(_dict, dict) else None
            if isinstance(rc, str) and rc and hasattr(msg, "additional_kwargs"):
                msg.additional_kwargs["reasoning_content"] = rc
            return msg

        def _patched_delta_to_chunk(_dict, *args, **kwargs):
            chunk = _orig_delta_to_chunk(_dict, *args, **kwargs)
            rc = _dict.get("reasoning_content") if isinstance(_dict, dict) else None
            if isinstance(rc, str) and rc and hasattr(chunk, "additional_kwargs"):
                # Per-chunk: stash this delta's reasoning_content on the chunk.
                # Cross-chunk accumulation is handled by AIMessageChunk.__add__
                # via merge_dicts (string values in additional_kwargs concatenate).
                chunk.additional_kwargs["reasoning_content"] = (
                    chunk.additional_kwargs.get("reasoning_content", "") + rc
                )
            return chunk

        _base._convert_dict_to_message = _patched_dict_to_msg
        _base._convert_delta_to_message_chunk = _patched_delta_to_chunk
        _openai_capture_patched = True
    except Exception:
        pass


_patch_openai_capture_reasoning_content()


# ---------------------------------------------------------------------------
# Patch: DeepSeek thinking mode requires reasoning_content to be passed back
# in all assistant messages for multi-turn + tool_use scenarios.
# langchain-openai's _convert_message_to_dict drops this field, causing
# HTTP 400 "The reasoning_content in the thinking mode must be passed back".
# Mirrors langchain-ai/langchain PR #34516 (which patches langchain-deepseek;
# we apply equivalent logic to a langchain-openai ChatOpenAI instance).
# ---------------------------------------------------------------------------
def _patch_deepseek_reasoning_passback(model: Any) -> None:
    """Inject reasoning_content into outgoing payload assistant messages.

    DeepSeek V4 thinking mode + tool_use requires every historical assistant
    message to carry its reasoning_content as a top-level field (sibling to
    content / tool_calls).  Without this, multi-turn requests fail with 400.

    For assistant messages where no reasoning_content was captured (e.g.
    history left over from another provider, from DeepSeek Flash, or from an
    older EvoSci version that ran before the capture patch landed), we
    inject an empty string.  This satisfies DeepSeek's format requirement
    in thinking mode.  Non-thinking DeepSeek endpoints are believed to
    accept the extra field without complaint based on observed behavior,
    but this has not been independently audited; if a future DeepSeek
    release rejects empty reasoning_content on non-thinking models, this
    fallback would need a per-call thinking-mode check instead of a blanket
    inject.  The check is intentionally not gated on model name: this
    function is only mounted when provider == "deepseek" (see
    EvoScientist/llm/models.py), so all callers are DeepSeek endpoints.

    Args:
        model: A langchain-openai ChatOpenAI instance configured for DeepSeek.
    """
    import functools

    from langchain_core.messages import AIMessage

    orig = getattr(model, "_get_request_payload", None)
    if orig is None:
        return

    import logging as _logging

    _logger = _logging.getLogger(__name__)

    @functools.wraps(orig)
    def _patched(input_: Any, *, stop: Any = None, **kwargs: Any) -> dict:
        try:
            lc_messages = model._convert_input(input_).to_messages()
        except Exception:
            _logger.warning(
                "DeepSeek passback patch: _convert_input failed, "
                "falling back to unpatched payload (reasoning_content "
                "will not be injected)",
                exc_info=True,
            )
            return orig(input_, stop=stop, **kwargs)

        ai_rcs: list[str | None] = [
            m.additional_kwargs.get("reasoning_content")
            for m in lc_messages
            if isinstance(m, AIMessage)
        ]

        payload = orig(input_, stop=stop, **kwargs)
        msgs = payload.get("messages")
        if not isinstance(msgs, list):
            return payload

        ai_idx = 0
        for msg in msgs:
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            rc = ai_rcs[ai_idx] if ai_idx < len(ai_rcs) else None
            if rc:
                msg["reasoning_content"] = rc
            elif "reasoning_content" not in msg:
                # Empty-string fallback for ALL DeepSeek models (not just
                # reasoner). Required when history contains AI messages that
                # came from a different provider (Anthropic / OpenAI /
                # DeepSeek Flash) or from an older EvoSci that didn't capture
                # reasoning_content. Empirically tolerated by non-thinking
                # DeepSeek endpoints; see docstring for the audit caveat.
                msg["reasoning_content"] = ""
            ai_idx += 1

        return payload

    model._get_request_payload = _patched


# ---------------------------------------------------------------------------
# Patch: deepagents _build_start_tool to spawn an EvoSci watcher per launch.
# The watcher monitors the async subagent's run stream and enqueues a
# notification when it reaches a terminal state.
# ---------------------------------------------------------------------------
_async_watcher_patched = False


def _patch_deepagents_async_watcher() -> None:
    """Wrap deepagents._build_start_tool to spawn an EvoSci watcher per launch.

    The patch is idempotent (applied once per process). It preserves the original
    StructuredTool's return shape (Command); it only adds a side-effect: after
    `runs.create()` returns, spawn `async_notifier.watch_run_and_notify()` so
    the supervisor receives a push notification when the sub-agent terminates.

    The watcher runs in the agent's asyncio event loop. It watches the run's
    event stream asynchronously and enqueues a notification to the thread-safe
    queue when the run reaches a terminal state (success, error, cancelled, etc).
    """
    global _async_watcher_patched
    if _async_watcher_patched:
        return

    try:
        import deepagents.middleware.async_subagents as ds_mod

        orig_build = ds_mod._build_start_tool

        def patched_build_start_tool(agent_map, clients, tool_description):
            # Build the original tool
            tool = orig_build(agent_map, clients, tool_description)
            orig_coro = tool.coroutine

            async def wrapped_coro(description, subagent_type, runtime):
                # Call the original coroutine to launch the subagent
                result = await orig_coro(description, subagent_type, runtime)

                # If successful (returned a Command), extract task IDs and spawn watchers
                if result is not None:
                    from langgraph.types import Command

                    if isinstance(result, Command):
                        # The Command.update dict contains "async_tasks" keyed by task_id
                        tasks_update = (result.update or {}).get(
                            "async_tasks", {}
                        ) or {}
                        for task_id, task in tasks_update.items():
                            try:
                                from EvoScientist.cli import async_notifier

                                # Get the async client for this subagent type
                                client = clients.get_async(subagent_type)
                                # Spawn the watcher (non-blocking, returns immediately).
                                # Pass `description` so the watcher captures the
                                # original prompt for friendlier completion display.
                                async_notifier.spawn_watcher(
                                    client,
                                    task_id,
                                    task["run_id"],
                                    task["agent_name"],
                                    prompt=description,
                                )
                            except Exception:
                                import logging

                                logging.getLogger(__name__).warning(
                                    "Failed to spawn watcher for task %s",
                                    task_id,
                                    exc_info=True,
                                )

                return result

            # Replace the tool's coroutine with the wrapped version
            tool.coroutine = wrapped_coro
            return tool

        ds_mod._build_start_tool = patched_build_start_tool

        # Also wrap _build_update_tool so that update_async_task (which creates
        # a new run_id on the same thread_id) also spawns a replacement watcher.
        orig_build_update = ds_mod._build_update_tool

        def patched_build_update_tool(agent_map, clients):
            tool = orig_build_update(agent_map, clients)
            orig_coro = tool.coroutine

            async def wrapped_coro(task_id, message, runtime):
                # Cancel the existing watcher for this thread_id BEFORE
                # awaiting orig_coro.  The update creates a new run on the
                # same thread_id via runs.create(..., multitask_strategy=
                # "interrupt"), which causes the old run's join_stream to
                # close cleanly.  Without pre-cancellation the old watcher
                # would observe a clean stream exit and default to
                # status="success", enqueuing a STALE notification before
                # spawn_watcher (called after orig_coro) can cancel it.
                # Pre-cancellation is safe because task_id IS the thread_id
                # in the deepagents async subagent protocol.
                try:
                    from EvoScientist.cli import async_notifier as _an

                    old_task = _an._watcher_by_thread.get(task_id)
                    if old_task is not None and not old_task.done():
                        old_task.cancel()
                except Exception:
                    pass

                result = await orig_coro(task_id, message, runtime)
                from langgraph.types import Command

                if isinstance(result, Command):
                    tasks_update = (result.update or {}).get("async_tasks", {}) or {}
                    for tid, task in tasks_update.items():
                        try:
                            from EvoScientist.cli import async_notifier

                            client = clients.get_async(task["agent_name"])
                            async_notifier.spawn_watcher(
                                client,
                                tid,
                                task["run_id"],
                                task["agent_name"],
                                prompt=message,
                            )
                        except Exception:
                            import logging

                            logging.getLogger(__name__).warning(
                                "Failed to spawn update watcher for task %s",
                                tid,
                                exc_info=True,
                            )
                return result

            tool.coroutine = wrapped_coro
            return tool

        ds_mod._build_update_tool = patched_build_update_tool
        _async_watcher_patched = True
    except Exception:
        pass
