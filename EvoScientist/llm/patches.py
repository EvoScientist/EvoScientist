"""Monkey-patches and utilities for third-party LangChain provider quirks.

All patches follow the same pattern: wrap an existing method/function to
fix upstream bugs, applied at import time or on first use.

Patches:
    - _patch_anthropic_proxy_compat: ccproxy dict→Pydantic model mismatch
    - _patch_openai_compat_content: list content→string for strict APIs
    - _patch_openai_capture_reasoning_content: capture provider
      reasoning_content into AIMessage.additional_kwargs (module-level,
      applied at import)
    - _patch_openai_responses_whole_tool_args: keep tool-call arguments the
      ChatGPT Codex backend sends only in ``function_call_arguments.done``
      (installed by the first sign-in model, then applies to every Responses stream)
    - _patch_langgraph_goto_none_crash: handle Command(goto=None)
      defensively in langgraph and langgraph-api (langgraph 1.2.9 workaround)
    - _patch_deepagents_bound_reasoning_claude_only: after a summary, drop
      kept ``anthropic``-tagged thinking for Claude only (installed by
      ``create_per_run_summarization_middleware``)

Utilities:
    - _flatten_message_content: convert content blocks to plain string
"""

from __future__ import annotations

import dataclasses
import functools
import sys
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    Mapping,
    Sequence,
)
from contextvars import ContextVar
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage


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
# Utility + Patch: Flatten list content to strings for OpenAI-compatible APIs.
# DeepSeek, SiliconFlow, etc. reject assistant messages whose content is a
# list rather than a string.  Image and file (PDF/document) blocks are
# preserved, not flattened away.
# ---------------------------------------------------------------------------
_SKIP_CONTENT_TYPES = frozenset({"thinking", "reasoning", "reasoning_content"})

# Media block types preserved when flattening (positive allowlist;
# thinking/reasoning still dropped).  Images + files (PDF/documents): both
# serialize on OpenAI-compatible APIs and capable models read them.  `video` is
# deliberately EXCLUDED (langchain-openai raises ValueError on it); `audio` is
# omitted (almost no model support).  is_data_content_block is unreliable
# (False for OpenAI image_url / Anthropic image-source).  Kept as separate
# image/file sets so the no-media fallback can gate per modality.
_IMAGE_CONTENT_TYPES = frozenset({"image", "image_url", "input_image"})
_FILE_CONTENT_TYPES = frozenset({"file", "input_file", "document"})
_MEDIA_CONTENT_TYPES = _IMAGE_CONTENT_TYPES | _FILE_CONTENT_TYPES


def _flatten_message_content(content: Any) -> str | list[Any] | Any:
    """Convert list-of-blocks content to a string, preserving media blocks.

    Thinking/reasoning blocks are dropped.  When a media block (image or file)
    is present, returns a list in the ORIGINAL order — consecutive text is
    joined into a single text block, media blocks kept as-is — so captions stay
    next to the attachment they describe.  Otherwise returns a plain string.
    Non-list input is returned unchanged.

    Args:
        content: Message content — a string, a list of content blocks, or
            another type.

    Returns:
        A plain string for text-only content, a list of blocks when media is
        present, or the input unchanged for non-list input.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return content
    parts: list[str] = []
    ordered_blocks: list[Any] = []
    saw_media = False

    def _flush_text() -> None:
        if parts:
            ordered_blocks.append({"type": "text", "text": "\n\n".join(parts)})
            parts.clear()

    for block in content:
        if isinstance(block, dict):
            btype = block.get("type")
            if btype in _MEDIA_CONTENT_TYPES:
                # Keep media as-is (never mutate; upstream copy.copy is shallow)
                # and preserve its position relative to surrounding text.
                _flush_text()
                ordered_blocks.append(block)
                saw_media = True
                continue
            if btype in _SKIP_CONTENT_TYPES:
                continue
            text = block.get("text")
            if text:
                parts.append(text)
        elif isinstance(block, str):
            parts.append(block)
    if saw_media:
        _flush_text()
        return ordered_blocks
    return "\n\n".join(parts) if parts else ""


def _sanitize_messages(
    messages: list[BaseMessage], hoist_tool_media: bool = True
) -> list[BaseMessage]:
    """Flatten list content for OpenAI-compatible APIs, preserving media.

    Text/reasoning content is flattened to a string; image blocks are
    preserved.  Tool messages cannot carry media
    on OpenAI-compatible APIs (content must be a string), so when
    ``hoist_tool_media`` is set the media in a tool result is moved into a
    HumanMessage emitted after that turn's (possibly parallel) tool messages.
    Anthropic-routed providers accept tool-result media natively and pass
    ``hoist_tool_media=False`` to keep it inline.
    """
    import copy

    from langchain_core.messages import HumanMessage

    out: list[BaseMessage] = []
    pending_media: list[Any] = []  # media hoisted out of a run of tool messages

    def _flush() -> None:
        if pending_media:
            out.append(HumanMessage(content=list(pending_media)))
            pending_media.clear()

    for msg in messages:
        is_tool = getattr(msg, "type", None) == "tool"
        if not is_tool:
            _flush()  # emit hoisted media before any non-tool message
        if not isinstance(msg.content, list):
            out.append(msg)
            continue
        flat = _flatten_message_content(msg.content)
        if hoist_tool_media and is_tool and isinstance(flat, list):
            text_blocks = [
                b for b in flat if isinstance(b, dict) and b.get("type") == "text"
            ]
            media_blocks = [
                b for b in flat if not (isinstance(b, dict) and b.get("type") == "text")
            ]
            tool_msg = copy.copy(msg)
            # Join ALL text runs (interleaved content can yield more than one)
            # so no text is lost; tool content must be a string on OpenAI-compat.
            tool_msg.content = (
                "\n\n".join(b["text"] for b in text_blocks)
                if text_blocks
                else "[media content provided in the following message]"
            )
            out.append(tool_msg)
            pending_media.extend(media_blocks)
        else:
            msg = copy.copy(msg)
            msg.content = flat
            out.append(msg)
    _flush()  # conversation may end with tool messages
    return out


# Fallback for models that reject some media input (no vision / no file
# support): replace blocks of the unsupported types with a placeholder so the
# conversation can keep going instead of erroring every turn that re-sends them.
_UNSUPPORTED_MEDIA_PLACEHOLDER = (
    "[attachment omitted: this model does not support this input type]"
)
# Markers that identify WHICH modality an error rejects, so only the implicated
# type is remembered (not all media).  Generic markers map to all media.
_IMAGE_ERROR_MARKERS = ("image_url", "image input", "support image")
_FILE_ERROR_MARKERS = ("file input", "pdf input", "document input")
# `expected \`text\`` keeps the backticks — the bare "expected text" phrase is
# too broad (matches non-media schema errors like "... expected text").
_GENERIC_MEDIA_MARKERS = ("multimodal", "expected `text`")


def _media_types_in(messages: list[BaseMessage]) -> set[str]:
    """Set of preserved-media block types present across the messages."""
    found: set[str] = set()
    for msg in messages:
        content = getattr(msg, "content", None)
        if isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") in _MEDIA_CONTENT_TYPES:
                    found.add(b["type"])
    return found


def _media_error_types(exc: Exception) -> set[str]:
    """Media modalities the error text implicates (empty if not media-specific).

    Used to remember ONLY the rejected modality, so e.g. a PDF rejection never
    disables images.
    """
    text = str(exc).lower()
    types: set[str] = set()
    if any(m in text for m in _IMAGE_ERROR_MARKERS):
        types |= _IMAGE_CONTENT_TYPES
    if any(m in text for m in _FILE_ERROR_MARKERS):
        types |= _FILE_CONTENT_TYPES
    if any(m in text for m in _GENERIC_MEDIA_MARKERS):
        types |= _MEDIA_CONTENT_TYPES
    return types


def _is_http_400(exc: Exception) -> bool:
    """True if the error carries an HTTP 400 status (bad request)."""
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    return status == 400


def _strip_media_types(
    messages: list[BaseMessage], types: set[str]
) -> list[BaseMessage]:
    """Replace blocks of the given types with a placeholder text block.

    Each stripped block is replaced IN PLACE (consecutive ones collapse into one
    placeholder), so surrounding text/media keep their original positions and a
    model that rejects only one modality still receives the others in order.
    """
    import copy

    out: list[BaseMessage] = []
    for msg in messages:
        content = getattr(msg, "content", None)
        if not isinstance(content, list) or not any(
            isinstance(b, dict) and b.get("type") in types for b in content
        ):
            out.append(msg)
            continue
        kept: list[Any] = []
        last_was_placeholder = False
        for b in content:
            if isinstance(b, dict) and b.get("type") in types:
                if not last_was_placeholder:
                    kept.append(
                        {"type": "text", "text": _UNSUPPORTED_MEDIA_PLACEHOLDER}
                    )
                    last_was_placeholder = True
                continue
            kept.append(b)
            last_was_placeholder = False
        msg = copy.copy(msg)
        msg.content = kept
        out.append(msg)
    return out


class _OpenAICompatContent:
    """Apply OpenAI-compatible content normalization without owning a model."""

    def __init__(
        self,
        profile: Mapping[str, object] | None,
        hoist_tool_media: bool,
    ) -> None:
        self.hoist_tool_media = hoist_tool_media
        self.blocked: set[str] = set()
        if profile is not None:
            if profile.get("image_inputs") is False:
                self.blocked |= _IMAGE_CONTENT_TYPES
            if profile.get("pdf_inputs") is False:
                self.blocked |= _FILE_CONTENT_TYPES

    def _prepare(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        prepared = (
            _strip_media_types(messages, self.blocked) if self.blocked else messages
        )
        return _sanitize_messages(prepared, self.hoist_tool_media)

    def _stripped(
        self,
        messages: list[BaseMessage],
        suspects: set[str],
    ) -> list[BaseMessage]:
        return _sanitize_messages(
            _strip_media_types(messages, self.blocked | suspects),
            self.hoist_tool_media,
        )

    def invoke(
        self,
        call: Callable[..., Any],
        messages: list[BaseMessage],
        *args,
        **kwargs,
    ) -> Any:
        suspects = _media_types_in(messages) - self.blocked
        prepared = self._prepare(messages)
        if not suspects:
            return call(prepared, *args, **kwargs)
        try:
            return call(prepared, *args, **kwargs)
        except Exception as exc:
            culprit = _media_error_types(exc) & suspects
            if not culprit and not _is_http_400(exc):
                raise
            try:
                result = call(self._stripped(messages, suspects), *args, **kwargs)
            except Exception:
                raise exc from None
            self.blocked.update(culprit)
            return result

    async def ainvoke(
        self,
        call: Callable[..., Awaitable[Any]],
        messages: list[BaseMessage],
        *args,
        **kwargs,
    ) -> Any:
        suspects = _media_types_in(messages) - self.blocked
        prepared = self._prepare(messages)
        if not suspects:
            return await call(prepared, *args, **kwargs)
        try:
            return await call(prepared, *args, **kwargs)
        except Exception as exc:
            culprit = _media_error_types(exc) & suspects
            if not culprit and not _is_http_400(exc):
                raise
            try:
                result = await call(self._stripped(messages, suspects), *args, **kwargs)
            except Exception:
                # stripping didn't help — surface original, don't cache
                raise exc from None
            self.blocked.update(culprit)
            return result

    def stream(
        self,
        call: Callable[..., Iterator[Any]],
        messages: list[BaseMessage],
        *args,
        **kwargs,
    ) -> Iterator[Any]:
        suspects = _media_types_in(messages) - self.blocked
        prepared = self._prepare(messages)
        if not suspects:
            yield from call(prepared, *args, **kwargs)
            return
        started = False
        try:
            for chunk in call(prepared, *args, **kwargs):
                started = True
                yield chunk
            return
        except Exception as exc:
            culprit = _media_error_types(exc) & suspects
            if started or (not culprit and not _is_http_400(exc)):
                raise
            media_exc = exc
        retry_started = False
        try:
            for chunk in call(self._stripped(messages, suspects), *args, **kwargs):
                if not retry_started:
                    retry_started = True
                    self.blocked.update(culprit)
                yield chunk
        except Exception:
            if retry_started:
                raise
            raise media_exc from None
        if not retry_started:
            raise media_exc from None

    async def astream(
        self,
        call: Callable[..., AsyncIterator[Any]],
        messages: list[BaseMessage],
        *args,
        **kwargs,
    ) -> AsyncIterator[Any]:
        suspects = _media_types_in(messages) - self.blocked
        prepared = self._prepare(messages)
        if not suspects:
            async for chunk in call(prepared, *args, **kwargs):
                yield chunk
            return
        started = False
        try:
            async for chunk in call(prepared, *args, **kwargs):
                started = True
                yield chunk
            return
        except Exception as exc:
            culprit = _media_error_types(exc) & suspects
            if started or (not culprit and not _is_http_400(exc)):
                raise
            media_exc = exc
        retry_started = False
        try:
            async for chunk in call(
                self._stripped(messages, suspects), *args, **kwargs
            ):
                if not retry_started:
                    retry_started = True
                    self.blocked.update(culprit)
                yield chunk
        except Exception:
            if retry_started:
                raise
            raise media_exc from None
        if not retry_started:
            raise media_exc from None


def _patch_openai_compat_content(model: Any, hoist_tool_media: bool = True) -> None:
    """Normalize content for OpenAI-compatible models lacking a native adapter."""
    import functools

    profile = getattr(model, "profile", None)
    compat = _OpenAICompatContent(
        profile if isinstance(profile, Mapping) else None,
        hoist_tool_media,
    )

    orig_generate = getattr(model, "_generate", None)
    if orig_generate is None:
        return

    @functools.wraps(orig_generate)
    def _patched_generate(
        messages: list[BaseMessage], *args: Any, **kwargs: Any
    ) -> Any:
        return compat.invoke(orig_generate, messages, *args, **kwargs)

    model._generate = _patched_generate

    orig_agenerate = getattr(model, "_agenerate", None)
    if orig_agenerate is not None:

        @functools.wraps(orig_agenerate)
        async def _patched_agenerate(
            messages: list[BaseMessage], *args: Any, **kwargs: Any
        ) -> Any:
            return await compat.ainvoke(orig_agenerate, messages, *args, **kwargs)

        model._agenerate = _patched_agenerate

    # Also patch streaming paths — CLI/agent uses _stream/_astream, so without
    # these the content flattening is bypassed during normal streaming calls.
    orig_stream = getattr(model, "_stream", None)
    if orig_stream is not None:

        @functools.wraps(orig_stream)
        def _patched_stream(
            messages: list[BaseMessage], *args: Any, **kwargs: Any
        ) -> Any:
            yield from compat.stream(orig_stream, messages, *args, **kwargs)

        model._stream = _patched_stream

    orig_astream = getattr(model, "_astream", None)
    if orig_astream is not None:

        @functools.wraps(orig_astream)
        async def _patched_astream(
            messages: list[BaseMessage], *args: Any, **kwargs: Any
        ) -> Any:
            async for chunk in compat.astream(orig_astream, messages, *args, **kwargs):
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

        def _reasoning_text(_dict) -> str | None:
            # vLLM-style servers stream the same text under `reasoning`.
            if not isinstance(_dict, dict):
                return None
            rc = _dict.get("reasoning_content") or _dict.get("reasoning")
            return rc if isinstance(rc, str) and rc else None

        def _patched_dict_to_msg(_dict, *args, **kwargs):
            msg = _orig_dict_to_msg(_dict, *args, **kwargs)
            rc = _reasoning_text(_dict)
            if rc and hasattr(msg, "additional_kwargs"):
                msg.additional_kwargs["reasoning_content"] = rc
            return msg

        def _patched_delta_to_chunk(_dict, *args, **kwargs):
            chunk = _orig_delta_to_chunk(_dict, *args, **kwargs)
            rc = _reasoning_text(_dict)
            if rc and hasattr(chunk, "additional_kwargs"):
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
# Patch (module-level): silence langgraph_api's OpenAPI schema-generation
# warnings for endpoints whose docstrings aren't valid YAML.
#
# Upstream ``langgraph_api.utils.SchemaGenerator.get_schema`` calls
# ``parse_docstring`` (inherited from Starlette's ``BaseSchemaGenerator``)
# on every registered endpoint. When the docstring is prose with stray
# ``:`` characters, ``yaml.safe_load`` raises and upstream logs the
# failure + full traceback at WARNING level. It then falls back to
# ``{"description": docstring}`` — the endpoint still ends up in the
# schema with its prose as the description, just without structured
# ``parameters``/``responses``/``tags`` fields.
#
# The fallback path is fine; the warning + traceback is just noise. And
# it's only triggered for our deploy because mounting any custom Starlette
# app (``EvoScientist/langgraph_dev/http.py``) makes upstream call
# ``update_openapi_spec`` at startup — which iterates EVERY route,
# including upstream's own endpoints whose prose docstrings predate the
# YAML convention.
#
# Fix: wrap ``parse_docstring`` itself and absorb ``yaml.YAMLError`` by
# returning the same fallback shape upstream's except branch produces.
# Non-YAML exceptions are deliberately left to propagate — upstream's
# ``get_schema`` already catches them and logs WARNING + traceback, so
# unexpected failures remain debuggable. Patching ``parse_docstring`` (a
# small, stable method) instead of ``get_schema`` (the larger loop body)
# minimizes our exposure to upstream churn.
# ---------------------------------------------------------------------------
_langgraph_schema_silenced_patched = False


def _patch_langgraph_schema_generator_silence_warnings() -> None:
    global _langgraph_schema_silenced_patched
    if _langgraph_schema_silenced_patched:
        return
    try:
        import langgraph_api.utils as _lgapi_utils
        import yaml

        _SchemaGenerator = _lgapi_utils.SchemaGenerator
        _orig_parse_docstring = _SchemaGenerator.parse_docstring

        def _patched_parse_docstring(self: Any, func: Any) -> dict[str, Any]:
            try:
                return _orig_parse_docstring(self, func)
            except yaml.YAMLError:
                return {"description": getattr(func, "__doc__", None) or ""}

        _SchemaGenerator.parse_docstring = _patched_parse_docstring
        _langgraph_schema_silenced_patched = True
    except Exception:
        # Patches are loader-safe: never crash the import. Silent failure
        # here just leaves the upstream warnings visible in deploy logs,
        # which is a benign fallback.
        pass


_patch_langgraph_schema_generator_silence_warnings()


# ---------------------------------------------------------------------------
# Patch (lazy, Anthropic protocol): strip foreign reasoning blocks from
# outgoing assistant messages.
#
# History produced on OpenAI-compatible providers (Qwen/DashScope, DeepSeek, …)
# can carry `reasoning_content` content blocks in AI messages. When a session
# switches to an Anthropic-protocol endpoint (native Anthropic, custom-anthropic,
# kimi-coding, minimax), langchain-anthropic's `_format_messages` passes unknown
# block types through verbatim (`else: content.append(block)`), and strict
# Anthropic-compatible backends (e.g. Moonshot's Kimi For Coding endpoint at
# api.kimi.com) reject the request with an opaque HTTP 400. Upstream already
# skips the OpenAI-Responses `reasoning` / `function_call` types the same way;
# `reasoning_content` is simply missing from that list. These blocks are display
# metadata — no Anthropic-protocol block type has this name — so stripping is
# always safe. Anthropic `thinking` blocks are preserved (Kimi backends require
# them on tool-call turns, even unsigned), but a missing `signature` is
# defaulted to "": Anthropic-compatible backends stream thinking without a
# signature_delta, so the aggregated block lacks the key, and strict edges
# (e.g. OpenRouter's Anthropic schema) reject replay with 400
# ("signature: expected string, received undefined"). Native Anthropic thinking
# always carries a signature string, so the default never fires there.
# ---------------------------------------------------------------------------
_anthropic_foreign_reasoning_patched = False
_FOREIGN_REASONING_BLOCK_TYPES = frozenset({"reasoning_content"})


def _normalize_anthropic_replay_messages(
    messages: Sequence[BaseMessage],
) -> Sequence[BaseMessage]:
    """Sanitize AI-message list content for Anthropic-protocol replay."""
    cleaned: list[BaseMessage] = []
    changed = False
    for message in messages:
        if isinstance(message, AIMessage) and isinstance(message.content, list):
            blocks: list[Any] = []
            block_changed = False
            for block in message.content:
                if isinstance(block, dict):
                    btype = block.get("type")
                    if btype in _FOREIGN_REASONING_BLOCK_TYPES:
                        block_changed = True
                        continue
                    if btype == "thinking" and not isinstance(
                        block.get("signature"), str
                    ):
                        block = {**block, "signature": ""}
                        block_changed = True
                blocks.append(block)
            if block_changed:
                message = message.model_copy(update={"content": blocks})
                changed = True
        cleaned.append(message)
    return cleaned if changed else messages


def _patch_anthropic_strip_foreign_reasoning() -> None:
    global _anthropic_foreign_reasoning_patched
    if _anthropic_foreign_reasoning_patched:
        return
    try:
        import langchain_anthropic.chat_models as _mod

        _orig = _mod._format_messages

        # Forward extra args: langchain-anthropic 1.7.3 added a required `model=` kwarg.
        @functools.wraps(_orig)
        def _patched(messages: Sequence[BaseMessage], *args: Any, **kwargs: Any) -> Any:
            return _orig(
                _normalize_anthropic_replay_messages(messages), *args, **kwargs
            )

        _mod._format_messages = _patched
        _anthropic_foreign_reasoning_patched = True
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Patch (lazy, OpenRouter only): default structured output to json_schema for
# Moonshot's always-thinking models (the kimi-k3 family).
#
# Moonshot rejects a forced tool choice while thinking is enabled with HTTP 400:
#   "tool_choice 'specified' is incompatible with thinking enabled"
# and kimi-k3's thinking cannot be disabled, so the function_calling default of
# ChatOpenRouter.with_structured_output fails every structured-output call
# (LLMToolSelectorMiddleware included). These endpoints support
# response_format json_schema, which needs no tool_choice — route them there.
# Moonshot-specific: other mandatory-reasoning models keep function_calling.
# Gated on the instance's model_name, so copies behave correctly and other
# models keep the function_calling default.
# ---------------------------------------------------------------------------
_openrouter_structured_output_patched = False


def _patch_openrouter_structured_output() -> None:
    global _openrouter_structured_output_patched
    if _openrouter_structured_output_patched:
        return
    try:
        from langchain_openrouter import ChatOpenRouter

        _orig = ChatOpenRouter.with_structured_output

        def _patched(
            self: Any,
            schema: Any = None,
            *,
            method: str = "function_calling",
            **kwargs: Any,
        ) -> Any:
            if method == "function_calling":
                from .models import _openrouter_needs_json_schema

                if _openrouter_needs_json_schema(self.model_name):
                    method = "json_schema"
            return _orig(self, schema, method=method, **kwargs)

        ChatOpenRouter.with_structured_output = _patched
        _openrouter_structured_output_patched = True
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Patch (lazy, Anthropic protocol): default structured output to json_schema
# for mandatory-thinking Kimi models (K3 / Kimi For Coding).
#
# These endpoints run with thinking always on and reject a forced tool choice
# with HTTP 400 ("tool_choice 'specified' is incompatible with thinking
# enabled") — ChatAnthropic's function_calling default forces tool_choice, so
# every structured-output call (LLMToolSelectorMiddleware included) fails.
# With thinking declared, langchain-anthropic falls back to tool_choice auto,
# but the model then calls the tool only sometimes (OutputParserException).
# These endpoints accept output_config.format (verified live via OpenRouter's
# Anthropic-compatible edge), which needs no tool_choice — route them there.
# Gated on the instance's model, so copies behave correctly and Claude models
# keep the function_calling default.
# ---------------------------------------------------------------------------
_anthropic_structured_output_patched = False


def _patch_anthropic_structured_output() -> None:
    global _anthropic_structured_output_patched
    if _anthropic_structured_output_patched:
        return
    try:
        from langchain_anthropic import ChatAnthropic

        _orig = ChatAnthropic.with_structured_output

        def _patched(
            self: Any,
            schema: Any = None,
            *,
            method: str = "function_calling",
            **kwargs: Any,
        ) -> Any:
            if method == "function_calling":
                from .models import _is_mandatory_thinking_kimi

                if _is_mandatory_thinking_kimi(self.model):
                    method = "json_schema"
            return _orig(self, schema, method=method, **kwargs)

        ChatAnthropic.with_structured_output = _patched
        _anthropic_structured_output_patched = True
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Explicit client composition for EvoAsyncSubAgentMiddleware and memory runs.
# Inject the live model into runs.create without mutating upstream factories.
#
# The failed-run reason (``_RunErrors``) retires once deepagents' check tool
# reads the thread's error: it reads ``run["error"]``, which langgraph-api
# never fills.
# ---------------------------------------------------------------------------

# The launching run's per-run (model, provider), set by the async-task tools
# from ``runtime.config`` for the duration of a single ``runs.create`` call.
# ``_merge_runs_config_kwargs`` reads it as the highest-precedence source.
#
# Why a ContextVar rather than passing ``config=`` through each tool: the fix
# has to hold at every ``runs.create`` site (``start_async_task`` and
# ``update_async_task``), but only the tool functions can see the caller's
# per-run model (via ``runtime.config``, the config langgraph's ToolNode
# injects into tool calls). Threading it through a ContextVar
# lets start and update share a single merge point.
_caller_configurable: ContextVar[dict[str, str] | None] = ContextVar(
    "_evo_caller_configurable", default=None
)


def _extract_caller_configurable(config: Any) -> dict[str, str]:
    """Pull ``(model, model_provider)`` out of a launching run's ``config``.

    ``config`` is the tool's ``runtime.config`` (a ``RunnableConfig``-shaped
    dict). Returns a dict suitable for ``configurable``; empty when the caller
    carries no model override, so the config-default fallback still applies.
    ``model_provider`` is only forwarded alongside a model — a bare provider
    without a model is meaningless to the deployed graph's resolver.
    """
    configurable = (config or {}).get("configurable") or {}
    model = configurable.get("model")
    provider = configurable.get("model_provider")
    out: dict[str, str] = {}
    if isinstance(model, str) and model:
        out["model"] = model
        if isinstance(provider, str) and provider:
            out["model_provider"] = provider
    return out


def _read_cfg_configurable() -> dict[str, str]:
    """Read live ``(model, provider)`` from EvoScientist config.

    Returns a dict suitable for inserting under
    ``RunnableConfig.configurable``. Empty dict on any failure (so the
    proxy degrades to a no-op rather than breaking async tool calls).
    """
    try:
        from EvoScientist.EvoScientist import _ensure_config

        cfg = _ensure_config()
    except Exception:
        return {}

    out: dict[str, str] = {}
    model = getattr(cfg, "model", None)
    provider = getattr(cfg, "provider", None)
    if isinstance(model, str) and model:
        out["model"] = model
    if isinstance(provider, str) and provider:
        out["model_provider"] = provider
    return out


def _merge_runs_config_kwargs(
    kwargs: dict, folders: dict[str, str] | None = None
) -> dict:
    """Merge the live model override into ``kwargs`` for ``runs.create``.

    ``folders`` are the launching graph's ``workspace_dir`` / ``run_dir``;
    they replace any the caller passed. A run named by its graph goes to that
    graph's assistant for the folders' workspace, which stands for the
    project; the server reads the run folder from ``run_dir``.

    Model source, in increasing precedence:

    1. ``_read_cfg_configurable()`` — the effective EvoScientist config. On the
       in-process backend this is the CLI's live per-run model; on the
       ``langgraph_server`` backend this proxy runs inside the dev-server
       process, where it reports the *server's* config-default model, not the
       CLI's per-run choice.
    2. ``_caller_configurable`` — the launching run's actual model, forwarded
       by the async-task tools from ``runtime.config``. This wins so that a
       sub-agent launched (or continued) while the caller runs on, say, a free
       model does not silently fall back to a billed config-default.

    Any unrelated caller-supplied ``config.configurable`` keys are preserved.
    """
    overrides = {
        **(folders or {}),
        **_read_cfg_configurable(),
        **(_caller_configurable.get() or {}),
    }
    if not overrides:
        return kwargs
    existing = kwargs.get("config")
    if not isinstance(existing, dict):
        existing = {}
    existing_configurable = existing.get("configurable")
    if not isinstance(existing_configurable, dict):
        existing_configurable = {}
    if folders:
        # The launching graph's folders are authoritative; in daemon mode they
        # carry no run_dir, so a caller's stale one must not survive.
        existing_configurable = {
            key: value
            for key, value in existing_configurable.items()
            if key not in ("workspace_dir", "run_dir")
        }
    merged_configurable = {**existing_configurable, **overrides}
    kwargs = dict(kwargs)
    kwargs["config"] = {**existing, "configurable": merged_configurable}
    if isinstance(kwargs.get("assistant_id"), str):
        from EvoScientist.langgraph_dev.assistants import launch_assistant_id

        kwargs["assistant_id"] = launch_assistant_id(
            kwargs["assistant_id"], (folders or {}).get("workspace_dir")
        )
    return kwargs


def _thread_error_message(thread: Any) -> str | None:
    """The error message langgraph-api records on a thread whose run failed."""
    error = thread.get("error") if isinstance(thread, dict) else None
    if isinstance(error, dict):
        # langgraph-api stores {"error": <type>, "message": <text>}; for most
        # exception types the text is a placeholder, so keep the type.
        kind, message = error.get("error"), error.get("message")
        error = f"{kind}: {message}" if kind and message else (message or kind)
    return str(error) if error else None


class _RunErrors:
    """Why failed runs failed, looked up once from their thread.

    langgraph-api records a failed run's error on its thread, not on the run,
    so the async-task check would otherwise report no reason. A failed run's
    error never changes, so each is looked up once.
    """

    def __init__(self) -> None:
        self._by_run: dict[str, str] = {}

    @staticmethod
    def _missing(run: Any) -> bool:
        return (
            isinstance(run, dict)
            and run.get("status") == "error"
            and not run.get("error")
        )

    def needs_lookup(self, run: Any) -> bool:
        return self._missing(run) and run.get("run_id") not in self._by_run

    def attach(self, run: Any, thread: Any = None) -> Any:
        run_id = run.get("run_id") if self._missing(run) else None
        if run_id is None:
            return run
        message = _thread_error_message(thread)
        if message:
            self._by_run[run_id] = message
        message = self._by_run.get(run_id)
        return {**run, "error": message} if message else run


class _SyncRunsProxy:
    """Wraps a sync ``RunsClient``: injects config into ``create`` and adds
    the failure reason to failed runs returned by ``get``."""

    def __init__(
        self,
        real: Any,
        folders: dict[str, str] | None = None,
        threads: Any = None,
        errors: _RunErrors | None = None,
    ) -> None:
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_folders", folders)
        object.__setattr__(self, "_threads", threads)
        object.__setattr__(self, "_errors", errors or _RunErrors())

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    def create(self, **kwargs: Any) -> Any:
        return self._real.create(**_merge_runs_config_kwargs(kwargs, self._folders))

    def get(self, *args: Any, **kwargs: Any) -> Any:
        run = self._real.get(*args, **kwargs)
        thread = None
        if self._threads is not None and self._errors.needs_lookup(run):
            try:
                thread = self._threads.get(run["thread_id"])
            except Exception:
                thread = None
        return self._errors.attach(run, thread)


class _AsyncRunsProxy:
    """Wraps an async ``RunsClient``: injects config into ``create`` and adds
    the failure reason to failed runs returned by ``get``."""

    def __init__(
        self,
        real: Any,
        folders: dict[str, str] | None = None,
        threads: Any = None,
        errors: _RunErrors | None = None,
    ) -> None:
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_folders", folders)
        object.__setattr__(self, "_threads", threads)
        object.__setattr__(self, "_errors", errors or _RunErrors())

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    async def create(self, **kwargs: Any) -> Any:
        return await self._real.create(
            **_merge_runs_config_kwargs(kwargs, self._folders)
        )

    async def get(self, *args: Any, **kwargs: Any) -> Any:
        run = await self._real.get(*args, **kwargs)
        thread = None
        if self._threads is not None and self._errors.needs_lookup(run):
            try:
                thread = await self._threads.get(run["thread_id"])
            except Exception:
                thread = None
        return self._errors.attach(run, thread)


class _ClientProxy:
    """Lightweight proxy that swaps ``client.runs`` for a runs proxy.

    Read-only forwarding: only ``__getattr__`` is overridden. Attribute
    *writes* on the proxy land on the proxy itself, NOT on the wrapped real
    client. Current deepagents only reads ``.threads`` / ``.runs``, so this
    is safe — but if a future caller tries ``client.foo = bar`` through the
    proxy, the write will be silently lost. ``object.__setattr__`` in
    ``__init__`` is used solely to avoid infinite recursion when seeding
    the internal slots.

    ``client.runs`` is a stable attribute set in
    ``langgraph_sdk.client.LangGraphClient.__init__`` (``self.runs =
    RunsClient(...)``) — not a property or lazy initializer — so the wrapped
    runs proxy is built once at ``__init__`` time and reused on every
    ``proxy.runs`` access. This avoids the previous behavior of selecting
    sync/async and instantiating a new ``_RunsProxy`` per attribute access.
    """

    def __init__(
        self,
        real: Any,
        *,
        is_async: bool,
        folders: dict[str, str] | None = None,
        errors: _RunErrors | None = None,
    ) -> None:
        runs_proxy_cls = _AsyncRunsProxy if is_async else _SyncRunsProxy
        object.__setattr__(self, "_real", real)
        object.__setattr__(
            self,
            "_runs_proxy",
            runs_proxy_cls(real.runs, folders, real.threads, errors),
        )

    def __getattr__(self, name: str) -> Any:
        if name == "runs":
            return self._runs_proxy
        return getattr(self._real, name)


class _ClientCacheProxy:
    """Proxy a ``_ClientCache`` so callers receive config-injecting clients.

    ``folders`` (``SessionDirs.metadata()`` of the launching graph) go into the
    ``configurable`` of every run these clients create.
    """

    def __init__(self, real: Any, folders: dict[str, str] | None = None) -> None:
        self._real = real
        self._folders = folders
        self._errors = _RunErrors()

    def get_sync(self, name: str) -> Any:
        return _ClientProxy(
            self._real.get_sync(name),
            is_async=False,
            folders=self._folders,
            errors=self._errors,
        )

    def get_async(self, name: str) -> Any:
        return _ClientProxy(
            self._real.get_async(name),
            is_async=True,
            folders=self._folders,
            errors=self._errors,
        )


# ---------------------------------------------------------------------------
# Patch: LangGraph 1.2.9 _control_branch crashes with TypeError when
# Command.goto is None (e.g. from a resume command). We patch both the
# generator (map_cmd) and the consumer (_control_branch).
# ---------------------------------------------------------------------------
def _patch_langgraph_goto_none_crash() -> None:
    """Handle Command(goto=None) defensively in langgraph and langgraph-api.

    Either patch alone prevents the production crash; both together cover
    all code paths (a Command(goto=None) could in principle be created
    by code other than map_cmd).

    Matches upstream issue langchain-ai/langgraph#5656.
    """
    # ── 1. map_cmd — prevent goto=None from being created ────────────────
    try:
        import langgraph_api.command as _cmd_mod
        from langgraph.types import Command as _Command

        _orig_map_cmd = getattr(_cmd_mod, "map_cmd", None)
        if _orig_map_cmd and not hasattr(_orig_map_cmd, "_evosci_goto_none_fix"):

            @functools.wraps(_orig_map_cmd)
            def _patched_map_cmd(cmd: Any) -> _Command:
                result = _orig_map_cmd(cmd)
                if result.goto is None:
                    return dataclasses.replace(result, goto=[])
                return result

            _patched_map_cmd._evosci_goto_none_fix = True  # type: ignore[attr-defined]
            _cmd_mod.map_cmd = _patched_map_cmd
            # Rebind on already-loaded consumers that grabbed map_cmd via
            # ``from langgraph_api.command import map_cmd`` — those hold the
            # original reference and wouldn't see the module-level rebind.
            for _name in ("langgraph_api.stream", "langgraph_api.grpc.ops.threads"):
                _mod = sys.modules.get(_name)
                if _mod is not None and getattr(_mod, "map_cmd", None) is _orig_map_cmd:
                    _mod.map_cmd = _patched_map_cmd
    except ImportError:
        pass

    # ── 2. _control_branch — handle goto=None defensively ───────────────
    # Wrap (not reimplement) the upstream function so only the goto=None→[]
    # normalization is ours; all other routing logic delegates to the
    # original, avoiding drift as upstream evolves.
    try:
        import langgraph.graph.state as _state_mod
        from langgraph.types import Command as _Command

        _orig_ctrl = getattr(_state_mod, "_control_branch", None)
        if _orig_ctrl and not hasattr(_orig_ctrl, "_evosci_goto_none_fix"):

            def _fix_goto(cmd: Any) -> Any:
                if isinstance(cmd, _Command) and cmd.goto is None:
                    return dataclasses.replace(cmd, goto=[])
                return cmd

            @functools.wraps(_orig_ctrl)
            def _patched_ctrl(value: Any) -> Sequence[tuple[str, Any]]:
                if isinstance(value, (list, tuple)):
                    value = [_fix_goto(v) for v in value]
                else:
                    value = _fix_goto(value)
                return _orig_ctrl(value)

            _patched_ctrl._evosci_goto_none_fix = True  # type: ignore[attr-defined]
            _state_mod._control_branch = _patched_ctrl
    except (ImportError, AttributeError):
        pass


_patch_langgraph_goto_none_crash()


# ---------------------------------------------------------------------------
# Patch: deepagents (0.7.30+) drops the ``thinking`` blocks of messages kept
# across a summary whenever they are tagged ``model_provider == "anthropic"``.
# Only Claude binds them to the earlier history; every Anthropic-protocol reply
# carries that tag, and Kimi backends reject a tool-call turn without thinking.
# ---------------------------------------------------------------------------
def _patch_deepagents_bound_reasoning_claude_only() -> None:
    """Keep thinking of models named other than Claude (upstream's Bedrock rule).

    A reply without a model name keeps upstream's drop.
    """
    try:
        import deepagents.middleware.summarization as _summ_mod
    except ImportError:
        return

    _orig = getattr(_summ_mod, "_bound_reasoning_types", None)
    if _orig is None or hasattr(_orig, "_evosci_claude_only"):
        return

    @functools.wraps(_orig)
    def _patched(message: AIMessage) -> frozenset[str]:
        metadata = message.response_metadata
        model = str(metadata.get("model_name") or "").lower()
        if (
            metadata.get("model_provider") == "anthropic"
            and model
            and "claude" not in model
            and "anthropic" not in model
        ):
            return frozenset()
        return _orig(message)

    _patched._evosci_claude_only = True  # type: ignore[attr-defined]
    _summ_mod._bound_reasoning_types = _patched


# ---------------------------------------------------------------------------
# Patch: the ChatGPT Codex backend usually sends parallel tool calls whole
# (only ``response.function_call_arguments.done``), while langchain-openai
# builds tool-call arguments from ``.delta`` events alone, leaving them empty.
# Hand the full arguments through as a single delta when none was streamed.
# ---------------------------------------------------------------------------
# Calls whose delivered arguments are remembered at once (across all streams).
_WHOLE_TOOL_ARGS_RECORD = 256


def _patch_openai_responses_whole_tool_args() -> None:
    """Keep tool-call arguments that arrive only in the ``done`` event.

    Process-wide: every ``ChatOpenAI`` Responses stream passes through it, so it
    changes nothing unless a call's arguments would otherwise be lost.
    """
    try:
        import langchain_openai.chat_models.base as _base
        from openai.types.responses import ResponseFunctionCallArgumentsDeltaEvent

        _orig = _base._convert_responses_chunk_to_generation_chunk
    except (ImportError, AttributeError):
        return
    if hasattr(_orig, "_evosci_whole_tool_args"):
        return
    # Arguments delivered so far per call; insertion-ordered so a stream cut
    # before ``done`` cannot grow it forever. A call pushed out is left to upstream;
    # only with more than twice the record's calls open at once is one forgotten.
    delivered: dict[str, str] = {}
    dropped: dict[str, None] = {}

    def _add(item_id: str, text: str) -> None:
        if item_id in dropped:
            return
        delivered[item_id] = delivered.get(item_id, "") + text
        if len(delivered) > _WHOLE_TOOL_ARGS_RECORD:
            oldest = next(iter(delivered))
            del delivered[oldest]
            dropped[oldest] = None
            if len(dropped) > _WHOLE_TOOL_ARGS_RECORD:
                del dropped[next(iter(dropped))]

    @functools.wraps(_orig)
    def _convert(chunk: Any, *args: Any, **kwargs: Any) -> Any:
        kind = getattr(chunk, "type", None)
        if kind == "response.output_item.added":
            item = getattr(chunk, "item", None)
            # Some servers open the call with (part of) its arguments filled in.
            if getattr(item, "type", None) == "function_call":
                # A new call: whatever an earlier, cut stream left under this id is stale.
                delivered.pop(item.id, None)
                dropped.pop(item.id, None)
                if item.arguments:
                    _add(item.id, item.arguments)
        elif kind == "response.function_call_arguments.delta":
            _add(chunk.item_id, chunk.delta)
        elif kind == "response.function_call_arguments.done":
            sent = delivered.pop(chunk.item_id, "")
            full = chunk.arguments or ""
            missing = full[len(sent) :] if full.startswith(sent) else ""
            if chunk.item_id in dropped:
                del dropped[chunk.item_id]
                missing = ""
            if missing:
                chunk = ResponseFunctionCallArgumentsDeltaEvent.construct(
                    type="response.function_call_arguments.delta",
                    delta=missing,
                    item_id=chunk.item_id,
                    output_index=chunk.output_index,
                    sequence_number=chunk.sequence_number,
                )
        return _orig(chunk, *args, **kwargs)

    _convert._evosci_whole_tool_args = True  # type: ignore[attr-defined]
    _base._convert_responses_chunk_to_generation_chunk = _convert
