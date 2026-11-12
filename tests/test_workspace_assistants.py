"""Workspace assistants: the IDs and fields the WebUI shares, and their upkeep."""

from __future__ import annotations

import uuid

import pytest

from EvoScientist.langgraph_dev.assistants import (
    ensure_workspace_assistants,
    folder_name,
    served_graph_ids,
    workspace_assistant_fields,
    workspace_assistant_id,
)
from EvoScientist.paths import Workspace


def test_workspace_assistant_id_matches_the_webui():
    # Computed with the WebUI's own ``uuid`` package (``workspaceAssistantId``).
    assert (
        workspace_assistant_id("EvoScientist", "/home/you/projA")
        == "39dcbcc1-957e-51c1-8e52-c294d4031162"
    )
    assert (
        workspace_assistant_id("scheduler", "/tmp/projÄ ü")
        == "17766371-eab6-54fd-940c-f639d8dc3d97"
    )


def test_workspace_assistant_id_uses_the_workspace_key(tmp_path):
    workspace = Workspace(tmp_path)
    assert workspace_assistant_id("EvoScientist", workspace) == workspace_assistant_id(
        "EvoScientist", workspace.key
    )


@pytest.mark.parametrize(
    ("key", "name"),
    [("/home/you/projA", "projA"), ("C:/Users/you/proj", "proj"), ("/", "/")],
)
def test_folder_name_takes_the_last_segment(key, name):
    assert folder_name(key) == name


def test_workspace_assistant_fields_match_the_webui():
    assert workspace_assistant_fields("scheduler", "/home/you/projA") == {
        "graph_id": "scheduler",
        "config": {"configurable": {"workspace_dir": "/home/you/projA"}},
        "metadata": {"workspace_dir": "/home/you/projA"},
        "name": "scheduler · projA",
    }


def _assistant(store, assistant_id):
    (found,) = [
        a for a in store["assistants"] if a["assistant_id"] == uuid.UUID(assistant_id)
    ]
    return found


async def test_ensure_workspace_assistants_creates_one_per_graph(
    runtime_store, tmp_path
):
    workspace = Workspace(tmp_path)
    await ensure_workspace_assistants(workspace)
    await ensure_workspace_assistants(workspace)

    assert len(runtime_store["assistants"]) == len(served_graph_ids())
    main = _assistant(runtime_store, workspace_assistant_id("EvoScientist", workspace))
    assert main["config"] == {"configurable": {"workspace_dir": workspace.key}}
    assert main["metadata"] == {"workspace_dir": workspace.key}


@pytest.mark.parametrize("folder", ["/elsewhere", None])
async def test_ensure_workspace_assistants_repairs_other_fields(
    runtime_store, tmp_path, folder
):
    """Another folder, or the right folder with a stale name and metadata."""
    workspace = Workspace(tmp_path)
    folder = folder or workspace.key
    assistant_id = workspace_assistant_id("EvoScientist", workspace)
    runtime_store["assistants"].append(
        {
            "assistant_id": uuid.UUID(assistant_id),
            "graph_id": "EvoScientist",
            "config": {"configurable": {"workspace_dir": folder}},
            "context": {"workspace_dir": folder},
            "metadata": {},
            "name": "planted",
            "version": 1,
            "description": None,
        }
    )
    runtime_store["assistant_versions"].append(
        {"assistant_id": uuid.UUID(assistant_id), "version": 1}
    )

    await ensure_workspace_assistants(workspace)

    repaired = _assistant(runtime_store, assistant_id)
    assert repaired["config"] == {"configurable": {"workspace_dir": workspace.key}}
    assert repaired["metadata"] == {"workspace_dir": workspace.key}
    assert repaired["name"] == f"EvoScientist · {tmp_path.name}"


def test_server_lifespan_runs_set_up_at_start(monkeypatch):
    from starlette.testclient import TestClient

    from EvoScientist.langgraph_dev import workspace_setup
    from EvoScientist.langgraph_dev.http import app

    started = []

    async def record():
        started.append(True)

    monkeypatch.setattr(workspace_setup, "set_up_at_start", record)
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000)):
        pass
    assert started == [True]


def test_server_start_survives_project_setup_failure(monkeypatch, tmp_path, caplog):
    """One project never stops a server that serves the others; the failure
    is logged and the project stays registered."""
    from starlette.testclient import TestClient

    from EvoScientist.langgraph_dev import assistants, registry
    from EvoScientist.langgraph_dev.http import app

    async def fail(_workspace):
        raise ValueError("store unavailable")

    registry.REGISTRY.set_registered(Workspace(tmp_path).key, True)
    monkeypatch.setattr(assistants, "ensure_workspace_assistants", fail)
    with TestClient(app, base_url="http://127.0.0.1"):
        pass

    assert "store unavailable" in caplog.text
    assert registry.REGISTRY.is_registered(Workspace(tmp_path).key)
