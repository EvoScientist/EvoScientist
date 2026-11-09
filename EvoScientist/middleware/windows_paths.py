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
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware

from .. import paths

if TYPE_CHECKING:
    from langchain.agents.middleware.types import ToolCallRequest
    from langchain_core.messages import ToolMessage
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


class WindowsFilePathMiddleware(AgentMiddleware):
    """Spell a drive path in a file-tool call as the ``/C:/...`` alias."""

    name = "windows_file_paths"

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        return handler(_with_file_tool_path(request))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
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
