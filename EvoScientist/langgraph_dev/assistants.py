"""Workspace assistants: one assistant per served graph and project.

A run names the project it works in by its assistant. Every graph in
``langgraph.json`` gets one assistant per project, with an ID computed from
the graph and the project's key, so every client (the CLI, the WebUI, an SDK
script) lands on the same assistant without looking it up. The WebUI creates
the same assistants (``src/lib/workspaceAssistant.ts``); both sides create
them with ``if_exists="do_nothing"``, so their fields must match.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

from ..paths import Workspace

# ``uuid5(NAMESPACE_URL, "https://github.com/EvoScientist/EvoScientist")``.
WORKSPACE_ASSISTANT_NAMESPACE = uuid.UUID("35e045d5-e927-554e-9f04-4ba98f84c54c")

# The graph the chat runs.
CHAT_GRAPH_ID = "EvoScientist"

# Read at import: inside the server, blockbuster refuses file reads on the
# event loop once it serves requests.
_SERVED_GRAPH_IDS: tuple[str, ...] = tuple(
    json.loads(Path(__file__).with_name("langgraph.json").read_text(encoding="utf-8"))[
        "graphs"
    ]
)


def served_graph_ids() -> tuple[str, ...]:
    """The graphs the server serves, as ``langgraph.json`` lists them."""
    return _SERVED_GRAPH_IDS


def _key(workspace: Workspace | str) -> str:
    return workspace.key if isinstance(workspace, Workspace) else workspace


def workspace_assistant_id(graph_id: str, workspace: Workspace | str) -> str:
    """The ID of *graph_id*'s assistant for *workspace*.

    The newline keeps the graph ID and the path apart.
    """
    return str(
        uuid.uuid5(WORKSPACE_ASSISTANT_NAMESPACE, f"{graph_id}\n{_key(workspace)}")
    )


def workspace_assistants(workspace: Workspace | str) -> dict[str, str]:
    """*workspace*'s assistants, one per served graph: ID to graph."""
    return {
        workspace_assistant_id(graph_id, workspace): graph_id
        for graph_id in _SERVED_GRAPH_IDS
    }


def workspace_assistant_ids(workspace: Workspace | str) -> frozenset[str]:
    """The IDs of *workspace*'s assistants, one per served graph."""
    return frozenset(workspace_assistants(workspace))


def launch_assistant_id(graph_id: str, workspace: Workspace | str | None) -> str:
    """The assistant a run named *graph_id* is launched on for *workspace*:
    the graph's workspace assistant; *graph_id* itself without a workspace,
    or when it names no served graph (an assistant ID, say)."""
    if workspace and graph_id in _SERVED_GRAPH_IDS:
        return workspace_assistant_id(graph_id, workspace)
    return graph_id


def folder_name(key: str) -> str:
    """The last segment of a folder key, for labels; the key when it has none."""
    parts = [part for part in re.split(r"[\\/]", key) if part]
    return parts[-1] if parts else key


def workspace_assistant_fields(
    graph_id: str, workspace: Workspace | str
) -> dict[str, Any]:
    """What a workspace assistant is created with, on both the core and the WebUI.

    No ``context``: the runtime mirrors ``configurable`` into it, and sending
    both is refused.
    """
    key = _key(workspace)
    return {
        "graph_id": graph_id,
        "config": {"configurable": {"workspace_dir": key}},
        "metadata": {"workspace_dir": key},
        "name": f"{graph_id} · {folder_name(key)}",
    }


def chat_assistant_id(workspace: Workspace | str) -> str:
    """The ID of the chat graph's assistant for *workspace*."""
    return workspace_assistant_id(CHAT_GRAPH_ID, workspace)


async def ensure_workspace_assistants(workspace: Workspace) -> None:
    """Create *workspace*'s assistants, and repair one created with other fields.

    Runs inside the server process, through the runtime's own store, as
    langgraph-api registers its system assistants. An assistant that already
    exists under a computed ID with other fields (an old build, or anyone on
    the API) would make every run of the project fail the binding check, so
    it is reset to the expected fields.
    """
    from langgraph_runtime.database import connect
    from langgraph_runtime.ops import Assistants

    async with connect() as conn:
        for graph_id in served_graph_ids():
            assistant_id = workspace_assistant_id(graph_id, workspace)
            fields = workspace_assistant_fields(graph_id, workspace)
            # Yields the existing assistant unchanged, or the new one.
            (assistant,) = [
                a
                async for a in await Assistants.put(
                    conn,
                    assistant_id,
                    context={},
                    if_exists="do_nothing",
                    **fields,
                )
            ]
            if _matches(assistant, fields):
                continue
            async for _ in await Assistants.patch(
                conn, uuid.UUID(assistant_id), **fields
            ):
                pass


def _matches(assistant: dict[str, Any], fields: dict[str, Any]) -> bool:
    key = fields["metadata"]["workspace_dir"]
    configurable = (assistant.get("config") or {}).get("configurable") or {}
    return (
        assistant.get("graph_id") == fields["graph_id"]
        and configurable.get("workspace_dir") == key
        and (assistant.get("metadata") or {}).get("workspace_dir") == key
        and assistant.get("name") == fields["name"]
    )
