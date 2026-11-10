"""Middleware that lets the file tools take Windows drive paths.

On Windows the agent's shell and Python print host paths (``C:\\Users\\me\\...``),
and the agent passes them to the file tools. deepagents validates the path
inside each tool and refuses every drive path, even one inside the workspace.
Rewriting the path argument to the ``/C:/...`` alias before the tool runs lets
the sandbox backend map it into the workspace, or refuse it when it lies
outside. Other platforms never decode that alias, so the rewrite runs on
Windows only.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any

from deepagents.backends.utils import validate_path
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage

from .. import paths

if TYPE_CHECKING:
    from langchain.agents.middleware.types import ToolCallRequest
    from langgraph.types import Command

# deepagents file tools and the argument that holds their path.
_PATH_ARGS = {
    "read_file": "file_path",
    "write_file": "file_path",
    "edit_file": "file_path",
    "delete": "file_path",
    "ls": "path",
    "glob": "path",
    "grep": "path",
}

# deepagents refuses a second call of these on one path within one reply.
_MUTATIONS = frozenset({"write_file", "edit_file", "delete"})


class WindowsFilePathMiddleware(AgentMiddleware):
    """Spell a drive path in a file-tool call as the ``/C:/...`` alias."""

    name = "windows_file_paths"

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        if (error := _parallel_mutation_error(request)) is not None:
            return error
        return handler(_with_file_tool_path(request))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        if (error := _parallel_mutation_error(request)) is not None:
            return error
        return await handler(_with_file_tool_path(request))


def _with_file_tool_path(request: ToolCallRequest) -> ToolCallRequest:
    if os.name != "nt":
        return request
    call = request.tool_call
    arg = _PATH_ARGS.get(call.get("name", ""))
    args = call.get("args") or {}
    value = args.get(arg) if arg is not None else None
    if not isinstance(value, str):
        return request
    alias = paths.file_tool_path(value)
    if alias == value:
        return request
    return request.override(tool_call={**call, "args": {**args, arg: alias}})


def _parallel_mutation_error(request: ToolCallRequest) -> ToolMessage | None:
    """deepagents' same-path check runs before this middleware and skips drive
    paths it cannot validate, so repeat it on their aliases."""
    call = request.tool_call
    if os.name != "nt" or call.get("name") not in _MUTATIONS:
        return None
    path = _mutation_path(call)
    if path is None:
        return None
    state = request.state if isinstance(request.state, Mapping) else {}
    reply = next(
        (m for m in reversed(state.get("messages") or []) if isinstance(m, AIMessage)),
        None,
    )
    for earlier in reply.tool_calls if reply else []:
        if earlier["id"] == call["id"]:
            return None
        if earlier["name"] in _MUTATIONS and _mutation_path(earlier) == path:
            return ToolMessage(
                "Error: parallel file mutations to the same path are not allowed.",
                name=call["name"],
                tool_call_id=call["id"],
                status="error",
            )
    return None


def _mutation_path(call: Mapping[str, Any]) -> str | None:
    value = (call.get("args") or {}).get("file_path")
    if not isinstance(value, str):
        return None
    try:
        return validate_path(paths.file_tool_path(value))
    except ValueError:
        return None
