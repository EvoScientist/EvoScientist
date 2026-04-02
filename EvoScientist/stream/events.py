"""Stream event generator and chunk processing helpers.

Async generator that streams events from an agent graph,
plus helpers for processing AI message chunks and tool results.
"""

import asyncio
import base64
import mimetypes
import os
import re
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import (  # type: ignore[import-untyped]
    AIMessage,
    AIMessageChunk,
)

from .emitter import StreamEventEmitter
from .tracker import ToolCallTracker
from .utils import DisplayLimits, is_success

# Safety net: older ccproxy versions may embed thinking as XML tags in content
# strings.  Strip them so they never leak to users or channels.
_THINKING_TAG_RE = re.compile(r"<thinking>.*?</thinking>", re.DOTALL)


def _strip_legacy_thinking_tags(content: str) -> str:
    """Remove ``<thinking>...</thinking>`` tags from content strings."""
    return _THINKING_TAG_RE.sub("", content)


# Image media types returned by DeepAgents read_file
_IMAGE_MEDIA_TYPES = {
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
    "image/bmp",
    "image/svg+xml",
}


def _extract_tool_content(msg) -> tuple[str, bool]:
    """Extract display-safe content from a ToolMessage.

    DeepAgents ``read_file`` returns image content as
    ``ToolMessage(content=[ImageContentBlock])`` with
    ``additional_kwargs["read_file_media_type"]`` set.
    Stringifying that would dump huge base64 data into the display.

    Returns:
        (content_string, is_image) — a short summary for images,
        or the raw string content for normal results.
    """
    additional = getattr(msg, "additional_kwargs", None) or {}
    media_type = additional.get("read_file_media_type", "")
    if media_type and media_type in _IMAGE_MEDIA_TYPES:
        # Extract path from the tool call args if available
        file_path = additional.get("read_file_path", "")
        if not file_path:
            file_path = getattr(msg, "name", "image")
        return f"[OK] Image displayed: {file_path} ({media_type})", True

    content = getattr(msg, "content", "")
    # Guard against list-type content (image content blocks without metadata)
    if isinstance(content, list):
        # Check if any block looks like image data
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "image" or "base64" in block:
                    return "[OK] Image displayed", True
        # Non-image list content — join text blocks
        parts = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text", "")
                if text:
                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts) if parts else str(content), False

    return str(content), False


def _extract_summarization_text(msg: Any) -> str:
    """Extract plain text from a summarization chunk.

    The summarization LLM streams ``AIMessageChunk`` objects whose
    ``content`` may be a plain string **or** a list of content blocks
    (e.g. ``[{'type': 'text', 'text': '...', 'index': 1}]``) depending
    on the provider.  This helper normalises both forms to a plain string.
    """
    if not hasattr(msg, "content"):
        return ""
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""


async def stream_agent_events(
    agent: Any,
    message: Any,
    thread_id: str,
    metadata: dict | None = None,
    media: list[str] | None = None,
) -> AsyncIterator[dict]:
    """Stream events from the agent graph using async iteration.

    Uses agent.astream() with subgraphs=True to see sub-agent activity.

    Args:
        agent: Compiled state graph from create_deep_agent()
        message: User message
        thread_id: Thread ID for conversation persistence
        metadata: Optional metadata dict merged into the LangGraph config
            (e.g. agent_name, updated_at for checkpoint persistence).
        media: Optional list of local file paths for attachments.

    Yields:
        Event dicts: thinking, text, tool_call, tool_result,
                     subagent_start, subagent_tool_call, subagent_tool_result, subagent_end,
                     done, error
    """
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    if metadata:
        config["metadata"] = metadata
    emitter = StreamEventEmitter()
    main_tracker = ToolCallTracker()
    full_response = ""

    # Track sub-agent names
    _key_to_name: dict[str, str] = {}  # subagent_key -> display name (cache)
    _announced_names: list[str] = []  # ordered queue of announced task names
    _assigned_names: set[str] = set()  # names already assigned to a namespace
    _announced_task_ids: list[str] = []  # ordered task tool_call_ids
    _task_id_to_name: dict[str, str] = {}  # tool_call_id -> sub-agent name
    _subagent_trackers: dict[str, ToolCallTracker] = {}  # namespace_key -> tracker

    def _register_task_tool_call(tc_data: dict) -> str | None:
        """Register or update a task tool call, return subagent name if started/updated."""
        tool_id = tc_data.get("id", "")
        if not tool_id:
            return None
        args = tc_data.get("args", {}) or {}
        desc = str(args.get("description", "")).strip()
        sa_name = str(args.get("subagent_type", "")).strip()
        if not sa_name:
            # Fallback to description snippet (may be empty during streaming)
            sa_name = desc.split("\n")[0].strip()
            sa_name = sa_name[:30] + "\u2026" if len(sa_name) > 30 else sa_name
        if not sa_name:
            sa_name = "sub-agent"

        if tool_id not in _announced_task_ids:
            _announced_task_ids.append(tool_id)
            _announced_names.append(sa_name)
            _task_id_to_name[tool_id] = sa_name
            return sa_name

        # Update mapping if we learned a better name later
        current = _task_id_to_name.get(tool_id, "sub-agent")
        if sa_name != "sub-agent" and current != sa_name:
            _task_id_to_name[tool_id] = sa_name
            try:
                idx = _announced_task_ids.index(tool_id)
                if idx < len(_announced_names):
                    _announced_names[idx] = sa_name
            except ValueError:
                pass
            return sa_name
        return None

    def _extract_task_id(namespace: tuple) -> tuple[str | None, str | None]:
        """Extract task tool_call_id from namespace if present.

        Returns (task_id, task_ns_element) or (None, None).
        """
        for part in namespace:
            part_str = str(part)
            if "task:" in part_str:
                tail = part_str.split("task:", 1)[1]
                task_id = tail.split(":", 1)[0] if tail else ""
                if task_id:
                    return task_id, part_str
        return None, None

    def _next_announced_name() -> str | None:
        """Get next announced name that hasn't been assigned yet."""
        for announced in _announced_names:
            if announced not in _assigned_names:
                _assigned_names.add(announced)
                return announced
        return None

    def _find_task_id_from_metadata(metadata: dict | None) -> str | None:
        """Try to find a task tool_call_id in metadata."""
        if not metadata:
            return None
        candidates = (
            "tool_call_id",
            "task_id",
            "parent_run_id",
            "root_run_id",
            "run_id",
        )
        for key in candidates:
            val = metadata.get(key)
            if val and val in _task_id_to_name:
                return val
        return None

    def _get_subagent_key(namespace: tuple, metadata: dict | None) -> str | None:
        """Stable key for tracker/mapping per sub-agent namespace."""
        if not namespace:
            return None
        _task_id, task_ns = _extract_task_id(namespace)
        if task_ns:
            return task_ns
        meta_task_id = _find_task_id_from_metadata(metadata)
        if meta_task_id:
            return f"task:{meta_task_id}"
        if metadata:
            for key in (
                "parent_run_id",
                "root_run_id",
                "run_id",
                "graph_id",
                "node_id",
            ):
                val = metadata.get(key)
                if val:
                    return f"{key}:{val}"
        return str(namespace)

    def _get_subagent_name(namespace: tuple, metadata: dict | None) -> str | None:
        """Resolve sub-agent name from namespace, or None if main agent.

        Priority:
        0) metadata["lc_agent_name"] -- most reliable, set by DeepAgents framework.
        1) Match task_id embedded in namespace to announced tool_call_id.
        2) Use cached key mapping (only real names, never "sub-agent").
        3) Queue-based: assign next announced name to this key.
        4) Fallback: return "sub-agent" WITHOUT caching.
        """
        if not namespace:
            return None

        key = _get_subagent_key(namespace, metadata) or str(namespace)

        # 0) lc_agent_name from metadata -- the REAL sub-agent name
        #    set by the DeepAgents framework on every namespace event.
        if metadata:
            lc_name = metadata.get("lc_agent_name", "")
            if isinstance(lc_name, str):
                lc_name = lc_name.strip()
            # Filter out generic/framework names
            if lc_name and lc_name not in (
                "sub-agent",
                "agent",
                "tools",
                "EvoScientist",
                "LangGraph",
                "",
            ):
                _key_to_name[key] = lc_name
                return lc_name

        # 1) Resolve by task_id if present in namespace
        task_id, _task_ns = _extract_task_id(namespace)
        if task_id and task_id in _task_id_to_name:
            name = _task_id_to_name[task_id]
            if name and name != "sub-agent":
                _assigned_names.add(name)
                _key_to_name[key] = name
                return name

        meta_task_id = _find_task_id_from_metadata(metadata)
        if meta_task_id and meta_task_id in _task_id_to_name:
            name = _task_id_to_name[meta_task_id]
            if name and name != "sub-agent":
                _assigned_names.add(name)
                _key_to_name[key] = name
                return name

        # 2) Cached real name for this key (skip if it's "sub-agent")
        cached = _key_to_name.get(key)
        if cached and cached != "sub-agent":
            return cached

        # 3) Assign next announced name from queue (skip "sub-agent" entries)
        for announced in _announced_names:
            if announced not in _assigned_names and announced != "sub-agent":
                _assigned_names.add(announced)
                _key_to_name[key] = announced
                return announced

        # 4) No real names available yet -- return generic WITHOUT caching
        return "sub-agent"

    # Build input for agent.astream()
    if isinstance(message, str):
        # Build user message content: text + inline images + file path references
        user_content: str | list[dict[str, Any]] = message
        if media:
            _IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"})
            _MAX_INLINE_SIZE = 5 * 1024 * 1024  # 5 MB
            content_blocks: list[dict[str, Any]] = []
            if message:
                content_blocks.append({"type": "text", "text": message})

            def _read_file_b64(path: str) -> str:
                with open(path, "rb") as fh:
                    return base64.b64encode(fh.read()).decode("ascii")

            file_refs: list[str] = []
            for path in media:
                ext = os.path.splitext(path)[1].lower()
                is_image = ext in _IMAGE_EXTS and await asyncio.to_thread(
                    os.path.isfile, path
                )
                if is_image:
                    fsize = await asyncio.to_thread(os.path.getsize, path)
                    if fsize <= _MAX_INLINE_SIZE:
                        mime = mimetypes.guess_type(path)[0] or "image/png"
                        b64 = await asyncio.to_thread(_read_file_b64, path)
                        content_blocks.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime};base64,{b64}",
                                },
                            }
                        )
                    else:
                        file_refs.append(path)
                else:
                    file_refs.append(path)
            if file_refs:
                ref_text = "\n".join(
                    f"[attached file: {os.path.basename(p)}] path: {p}"
                    for p in file_refs
                )
                content_blocks.append({"type": "text", "text": ref_text})
            if content_blocks:
                user_content = content_blocks
        astream_input: Any = {"messages": [{"role": "user", "content": user_content}]}
    else:
        # HITL resume: Command object passed directly to agent
        astream_input = message

    _summarization_in_progress = False
    _tool_selection_suppressing = False  # True while buffering selector JSON
    _tool_selection_buffer = ""  # accumulates JSON chunks for parse attempt
    _tool_selection_was_active = False  # True after suppression, triggers Panel

    try:
        async for chunk in agent.astream(
            astream_input,
            config=config,
            stream_mode=["messages", "updates"],
            subgraphs=True,
        ):
            # Multi-mode + subgraphs: 3-tuple (namespace, mode, data)
            # Single-mode + subgraphs: 2-tuple (namespace, data) — fallback
            if not isinstance(chunk, tuple):
                continue

            namespace: tuple = ()
            data: Any
            mode_str: str

            if len(chunk) == 3:
                namespace, mode_str, data = chunk
                if not isinstance(namespace, tuple):
                    namespace = ()
            elif len(chunk) == 2:
                first = chunk[0]
                if isinstance(first, tuple):
                    namespace = first
                    data = chunk[1]
                else:
                    data = chunk
                mode_str = "messages"
            else:
                continue

            # Parse HITL / ask_user interrupts from updates mode
            if mode_str == "updates":
                if isinstance(data, dict) and "__interrupt__" in data:
                    for interrupt_obj in data["__interrupt__"]:
                        if isinstance(interrupt_obj, dict):
                            interrupt_value = interrupt_obj.get("value", {})
                        else:
                            interrupt_value = getattr(interrupt_obj, "value", {})

                        # Discriminate ask_user vs HITL interrupts
                        iv_type = (
                            interrupt_value.get("type")
                            if isinstance(interrupt_value, dict)
                            else getattr(interrupt_value, "type", None)
                        )
                        if iv_type == "ask_user":
                            questions = (
                                interrupt_value.get("questions", [])
                                if isinstance(interrupt_value, dict)
                                else getattr(interrupt_value, "questions", [])
                            )
                            tc_id = (
                                interrupt_value.get("tool_call_id", "")
                                if isinstance(interrupt_value, dict)
                                else getattr(interrupt_value, "tool_call_id", "")
                            )
                            ns_parts = (
                                interrupt_obj.get("ns", [""])
                                if isinstance(interrupt_obj, dict)
                                else getattr(interrupt_obj, "ns", [""])
                            )
                            interrupt_id = str(ns_parts[0]) if ns_parts else "default"
                            yield emitter.ask_user_interrupt(
                                interrupt_id, questions, tc_id
                            ).data
                            continue

                        # Standard HITL approval interrupt
                        if isinstance(interrupt_value, dict):
                            action_reqs = interrupt_value.get("action_requests", [])
                            review_cfgs = interrupt_value.get("review_configs", [])
                        else:
                            action_reqs = getattr(
                                interrupt_value, "action_requests", []
                            )
                            review_cfgs = getattr(interrupt_value, "review_configs", [])
                        if action_reqs:
                            ns_parts = (
                                interrupt_obj.get("ns", [""])
                                if isinstance(interrupt_obj, dict)
                                else getattr(interrupt_obj, "ns", [""])
                            )
                            interrupt_id = str(ns_parts[0]) if ns_parts else "default"
                            yield emitter.interrupt(
                                interrupt_id, action_reqs, review_cfgs
                            ).data
                continue
            if mode_str != "messages":
                continue

            # Unpack message + metadata from data
            msg: Any
            metadata: dict = {}
            if isinstance(data, tuple) and len(data) >= 2:
                msg = data[0]
                metadata = data[1] or {}
            else:
                msg = data

            # Accumulate summarization middleware chunks and emit text incrementally.
            # The summarization LLM streams AIMessageChunks; content may be a
            # plain string or a list of content blocks (provider-dependent).
            if (
                isinstance(metadata, dict)
                and metadata.get("lc_source") == "summarization"
            ):
                if not _summarization_in_progress:
                    _summarization_in_progress = True
                chunk_text = _extract_summarization_text(msg)
                if chunk_text:
                    yield emitter.summarization(chunk_text).data
                continue

            # Suppress LLMToolSelectorMiddleware streaming output.
            # The selector streams JSON like '{"tools":[...]}' via ainvoke
            # which gets captured by astream.  We detect it by content since
            # the _selector_active flag is not visible in the streaming loop.
            # Uses _tool_selection_suppressing to track suppression state
            # and emits a tool_selection event from the tracker ContextVar.
            if isinstance(msg, (AIMessageChunk, AIMessage)):
                _raw = msg.content
                _text = (
                    _raw
                    if isinstance(_raw, str)
                    else "".join(
                        b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in _raw
                    )
                    if isinstance(_raw, list)
                    else ""
                )

                # Suppress Anthropic's structured output tool calls.
                # Anthropic implements with_structured_output via tool calls
                # named "ToolSelectionResponse" (exact LangChain convention).
                # After the initial tool call, Anthropic streams input_json_delta
                # chunks with the JSON arguments — suppress those too.
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    _tc_names = [tc.get("name", "") for tc in msg.tool_calls]
                    if any(n == "ToolSelectionResponse" for n in _tc_names):
                        _tool_selection_was_active = True
                        continue
                if _tool_selection_was_active and isinstance(_raw, list):
                    if any(
                        isinstance(b, dict) and b.get("type") == "input_json_delta"
                        for b in _raw
                    ):
                        continue  # still streaming selector tool call args

                # Universal selector JSON detection via buffering.
                # Buffer chunks starting with '{', try JSON parse,
                # suppress if contains "tools". If not selector JSON,
                # stop buffering and let the chunk through (don't drop it).
                if _tool_selection_suppressing:
                    _tool_selection_buffer += _text
                    try:
                        import json as _json

                        _parsed = _json.loads(_tool_selection_buffer.strip())
                        if isinstance(_parsed, dict) and "tools" in _parsed:
                            _tool_selection_was_active = True
                            _tool_selection_suppressing = False
                            _tool_selection_buffer = ""
                            continue
                        # Valid JSON but not selector — stop buffering,
                        # fall through so this chunk is processed normally.
                        _tool_selection_suppressing = False
                        _tool_selection_buffer = ""
                    except (ValueError, TypeError):
                        _buf = _tool_selection_buffer.strip()
                        # Concatenated JSONs: {"tools":[...]}{"tools":[...]}
                        if '"tools"' in _buf and _buf.endswith("}"):
                            _tool_selection_was_active = True
                            _tool_selection_suppressing = False
                            _tool_selection_buffer = ""
                            continue
                        if len(_tool_selection_buffer) > 10000:
                            _tool_selection_suppressing = False
                            _tool_selection_buffer = ""
                            # Fall through — don't drop content
                        else:
                            continue  # keep buffering
                if (
                    not _tool_selection_suppressing
                    and _text.lstrip().startswith("{")
                    and ('"tools"' in _text or len(_text.strip()) <= 10)
                ):
                    # Try immediate parse (ccproxy returns full JSON in one chunk)
                    _stripped_text = _text.strip()
                    try:
                        import json as _json2

                        _parsed2 = _json2.loads(_stripped_text)
                        if isinstance(_parsed2, dict) and "tools" in _parsed2:
                            _tool_selection_was_active = True
                            continue
                    except (ValueError, TypeError):
                        # Could be concatenated JSONs: {"tools":[...]}{"tools":[...]}
                        # or incomplete JSON from streamed provider.
                        if '"tools"' in _stripped_text and _stripped_text.endswith("}"):
                            _tool_selection_was_active = True
                            continue
                    # Incomplete JSON — start buffering for streamed providers
                    _tool_selection_suppressing = True
                    _tool_selection_buffer = _text
                    continue

                # Emit tool_selection event on first non-empty chunk after
                # suppression.  Empty chunks arrive before the tracker has
                # captured the selected tools, so we skip them.
                if _tool_selection_was_active:
                    import EvoScientist.middleware.tool_selector as _ts_mod

                    if _ts_mod._current_selected_tools:
                        _tool_selection_was_active = False
                        selected = _ts_mod._current_selected_tools
                        # Only show Panel when:
                        # 1. Tools were actually filtered (not all selected)
                        # 2. Selection changed from last time
                        if len(selected) < _ts_mod._total_tools_count and sorted(
                            selected
                        ) != sorted(_ts_mod._last_emitted_tools):
                            yield emitter.tool_selection(list(selected)).data
                            _ts_mod._last_emitted_tools = list(selected)
                        _ts_mod._current_selected_tools = []
                    elif _text or (hasattr(msg, "tool_calls") and msg.tool_calls):
                        # Non-empty content arrived but no selected tools —
                        # tracker didn't run (shouldn't happen). Give up.
                        _tool_selection_was_active = False

            subagent = _get_subagent_name(namespace, metadata)
            subagent_tracker = None
            if subagent:
                tracker_key = _get_subagent_key(namespace, metadata) or str(namespace)
                subagent_tracker = _subagent_trackers.setdefault(
                    tracker_key, ToolCallTracker()
                )

            # Extract token usage from main-agent AIMessages
            if isinstance(msg, (AIMessageChunk, AIMessage)) and not subagent:
                usage = getattr(msg, "usage_metadata", None)
                if usage:
                    inp = (
                        usage.get("input_tokens", 0)
                        if isinstance(usage, dict)
                        else getattr(usage, "input_tokens", 0)
                    )
                    out = (
                        usage.get("output_tokens", 0)
                        if isinstance(usage, dict)
                        else getattr(usage, "output_tokens", 0)
                    )
                    if inp or out:
                        yield emitter.usage_stats(inp, out).data

            # Process AIMessageChunk / AIMessage
            if isinstance(msg, (AIMessageChunk, AIMessage)):
                if subagent:
                    # Sub-agent content -- emit sub-agent events
                    for ev in _process_chunk_content(msg, emitter, subagent_tracker):
                        if ev.type == "tool_call":
                            yield emitter.subagent_tool_call(
                                subagent,
                                ev.data["name"],
                                ev.data["args"],
                                ev.data.get("id", ""),
                            ).data
                        # Emit sub-agent text for fallback extraction
                        # (not displayed in TUI, but available to consumers)
                        elif ev.type == "text":
                            yield emitter.subagent_text(
                                subagent,
                                ev.data.get("content", ""),
                                instance_id=tracker_key,
                            ).data

                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            name = tc.get("name", "")
                            args = tc.get("args", {})
                            tool_id = tc.get("id", "")
                            # Skip empty-name chunks (incomplete streaming fragments)
                            if not name and not tool_id:
                                continue
                            yield emitter.subagent_tool_call(
                                subagent,
                                name,
                                args if isinstance(args, dict) else {},
                                tool_id,
                            ).data
                else:
                    # Main agent content
                    for ev in _process_chunk_content(msg, emitter, main_tracker):
                        if ev.type == "text":
                            full_response += ev.data.get("content", "")
                        yield ev.data

                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for ev in _process_tool_calls(
                            msg.tool_calls, emitter, main_tracker
                        ):
                            yield ev.data
                            # Detect task tool calls -> announce sub-agent
                            tc_data = ev.data
                            if tc_data.get("name") == "task":
                                started_name = _register_task_tool_call(tc_data)
                                if started_name:
                                    desc = str(
                                        tc_data.get("args", {}).get("description", "")
                                    ).strip()
                                    yield emitter.subagent_start(
                                        started_name, desc
                                    ).data

            # Process ToolMessage (tool execution result)
            elif hasattr(msg, "type") and msg.type == "tool":
                if subagent:
                    if subagent_tracker:
                        subagent_tracker.finalize_all()
                        for info in subagent_tracker.emit_all_pending():
                            yield emitter.subagent_tool_call(
                                subagent,
                                info.name,
                                info.args,
                                info.id,
                            ).data
                    name = getattr(msg, "name", "unknown")
                    raw_content, _is_img = _extract_tool_content(msg)
                    content = raw_content[: DisplayLimits.TOOL_RESULT_MAX]
                    success = is_success(content)
                    yield emitter.subagent_tool_result(
                        subagent, name, content, success
                    ).data
                else:
                    for ev in _process_tool_result(msg, emitter, main_tracker):
                        yield ev.data
                        # Tool result can re-emit tool_call with full args; update task mapping
                        if ev.type == "tool_call" and ev.data.get("name") == "task":
                            started_name = _register_task_tool_call(ev.data)
                            if started_name:
                                desc = str(
                                    ev.data.get("args", {}).get("description", "")
                                ).strip()
                                yield emitter.subagent_start(started_name, desc).data
                    # Check if this is a task result -> sub-agent ended
                    name = getattr(msg, "name", "")
                    if name == "task":
                        tool_call_id = getattr(msg, "tool_call_id", "")
                        # Find the sub-agent name via tool_call_id map
                        sa_name = _task_id_to_name.get(tool_call_id, "sub-agent")
                        yield emitter.subagent_end(sa_name).data

    except Exception as e:
        yield emitter.error(str(e)).data
        raise

    yield emitter.done(full_response).data


def _process_chunk_content(
    chunk, emitter: StreamEventEmitter, tracker: ToolCallTracker
):
    """Process content blocks from an AI message chunk."""
    content = chunk.content

    if isinstance(content, str):
        if content:
            cleaned = _strip_legacy_thinking_tags(content)
            if cleaned:
                yield emitter.text(cleaned)
            return

    blocks = None
    if hasattr(chunk, "content_blocks"):
        try:
            blocks = chunk.content_blocks
        except Exception:
            blocks = None

    if blocks is None:
        if isinstance(content, dict):
            blocks = [content]
        elif isinstance(content, list):
            blocks = content
        else:
            return

    for raw_block in blocks:
        block = raw_block
        if not isinstance(block, dict):
            if hasattr(block, "model_dump"):
                block = block.model_dump()
            elif hasattr(block, "dict"):
                block = block.dict()
            else:
                continue

        block_type = block.get("type")

        if block_type in ("thinking", "reasoning"):
            thinking_text = block.get("thinking") or block.get("reasoning") or ""
            if thinking_text:
                yield emitter.thinking(thinking_text)

        elif block_type == "text":
            text = block.get("text") or block.get("content") or ""
            if text:
                text = _strip_legacy_thinking_tags(text)
                if text:
                    yield emitter.text(text)

        elif block_type in ("tool_use", "tool_call"):
            tool_id = block.get("id", "")
            name = block.get("name", "")
            args = block.get("input") if block_type == "tool_use" else block.get("args")
            args_payload = args if isinstance(args, dict) else {}

            if tool_id:
                tracker.update(tool_id, name=name, args=args_payload)
                if tracker.is_ready(tool_id):
                    tracker.mark_emitted(tool_id)
                    yield emitter.tool_call(name, args_payload, tool_id)

        elif block_type == "input_json_delta":
            partial_json = block.get("partial_json", "")
            if partial_json:
                tracker.append_json_delta(partial_json, block.get("index", 0))

        elif block_type == "tool_call_chunk":
            tool_id = block.get("id", "")
            name = block.get("name", "")
            if tool_id:
                tracker.update(tool_id, name=name)
            partial_args = block.get("args", "")
            if isinstance(partial_args, str) and partial_args:
                tracker.append_json_delta(partial_args, block.get("index", 0))


def _process_tool_calls(
    tool_calls: list, emitter: StreamEventEmitter, tracker: ToolCallTracker
):
    """Process tool_calls from chunk.tool_calls attribute."""
    for tc in tool_calls:
        tool_id = tc.get("id", "")
        if tool_id:
            name = tc.get("name", "")
            args = tc.get("args", {})
            args_payload = args if isinstance(args, dict) else {}

            tracker.update(tool_id, name=name, args=args_payload)
            if tracker.is_ready(tool_id):
                tracker.mark_emitted(tool_id)
                yield emitter.tool_call(name, args_payload, tool_id)


def _process_tool_result(chunk, emitter: StreamEventEmitter, tracker: ToolCallTracker):
    """Process a ToolMessage result."""
    tracker.finalize_all()

    # Re-emit all tool calls with complete args
    for info in tracker.get_all():
        yield emitter.tool_call(info.name, info.args, info.id)

    name = getattr(chunk, "name", "unknown")
    raw_content, _is_img = _extract_tool_content(chunk)
    content = raw_content[: DisplayLimits.TOOL_RESULT_MAX]
    if len(raw_content) > DisplayLimits.TOOL_RESULT_MAX:
        content += "\n... (truncated)"

    success = is_success(content)
    yield emitter.tool_result(name, content, success)
