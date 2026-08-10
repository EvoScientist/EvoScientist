"""Smoke tests for the /api/models and /api/teams routes mounted via
langgraph.json's ``http`` field. We test the Starlette app directly — no
need to spin up langgraph dev.
"""

from __future__ import annotations

from unittest.mock import patch

from starlette.testclient import TestClient

from EvoScientist.config import EvoScientistConfig
from EvoScientist.langgraph_dev.http import app

client = TestClient(app)


def test_get_models_returns_entries_and_default():
    mock_cfg = EvoScientistConfig(
        model="claude-sonnet-4-6", provider="custom-anthropic"
    )
    with patch(
        "EvoScientist.langgraph_dev.http.get_effective_config", return_value=mock_cfg
    ):
        resp = client.get("/api/models")
    assert resp.status_code == 200
    body = resp.json()
    assert "entries" in body
    assert "default" in body
    assert body["default"] == {
        "name": "claude-sonnet-4-6",
        "provider": "custom-anthropic",
    }
    assert isinstance(body["entries"], list)
    assert len(body["entries"]) > 0
    # Every entry has the three required keys
    for entry in body["entries"]:
        assert set(entry.keys()) == {"name", "model_id", "provider"}
        assert isinstance(entry["name"], str)
        assert entry["name"]
        assert isinstance(entry["model_id"], str)
        assert entry["model_id"]
        assert isinstance(entry["provider"], str)
        assert entry["provider"]


def test_entries_preserve_registry_order():
    """The picker uses position-in-list to rank providers per short name —
    the JSON must preserve the order returned by ``list_models_by_provider``.

    Stubs ``get_effective_config`` to keep the assertion focused on
    registry order rather than implicitly depending on the ambient
    deploy config.
    """
    from EvoScientist.llm.models import list_models_by_provider

    expected = [
        {"name": n, "model_id": m, "provider": p}
        for n, m, p in list_models_by_provider()
    ]
    mock_cfg = EvoScientistConfig()
    with patch(
        "EvoScientist.langgraph_dev.http.get_effective_config", return_value=mock_cfg
    ):
        resp = client.get("/api/models")
    assert resp.json()["entries"] == expected


def test_default_passes_through_arbitrary_config_pair():
    """If config.yaml names a (name, provider) pair that isn't in the
    registry (typo, retired model), still report it as default — the
    picker labels it as the active selection regardless.
    """
    mock_cfg = EvoScientistConfig(model="some-retired-name", provider="some-provider")
    with patch(
        "EvoScientist.langgraph_dev.http.get_effective_config", return_value=mock_cfg
    ):
        resp = client.get("/api/models")
    assert resp.json()["default"] == {
        "name": "some-retired-name",
        "provider": "some-provider",
    }


def test_ollama_models_appended_when_base_url_configured():
    """Mirrors the TUI ``/model`` picker: when ``ollama_base_url`` is set,
    locally-pulled Ollama models are appended after the static registry
    as ``provider: "ollama"`` entries.
    """
    mock_cfg = EvoScientistConfig(
        model="claude-sonnet-4-6",
        provider="custom-anthropic",
        ollama_base_url="http://localhost:11434",
    )

    async def fake_discover(_base_url, *, timeout):
        return ["llama3:8b", "mistral:7b"]

    with (
        patch(
            "EvoScientist.langgraph_dev.http.get_effective_config",
            return_value=mock_cfg,
        ),
        patch(
            "EvoScientist.llm.ollama_discovery.discover_ollama_models",
            new=fake_discover,
        ),
    ):
        body = client.get("/api/models").json()

    # Assert the response is the static registry followed by the discovered
    # Ollama suffix — robust to future static Ollama entries in the registry.
    from EvoScientist.llm.models import list_models_by_provider

    static_entries = [
        {"name": n, "model_id": m, "provider": p}
        for n, m, p in list_models_by_provider()
    ]
    discovered_entries = [
        {"name": "llama3:8b", "model_id": "llama3:8b", "provider": "ollama"},
        {"name": "mistral:7b", "model_id": "mistral:7b", "provider": "ollama"},
    ]
    assert body["entries"][: len(static_entries)] == static_entries
    assert body["entries"][len(static_entries) :] == discovered_entries
    # TUI's "Custom Ollama model…" sentinel is a widget-specific affordance —
    # it must not appear on the HTTP surface.
    assert not any(e["model_id"] == "__custom_ollama__" for e in body["entries"])


def test_ollama_discovery_skipped_when_base_url_absent():
    """No Ollama discovery should happen when ``ollama_base_url`` is unset —
    matches the ``/model`` picker's gating. The probe function should never
    be called in that case.
    """
    mock_cfg = EvoScientistConfig(
        model="claude-sonnet-4-6", provider="custom-anthropic"
    )
    calls: list[str | None] = []

    async def spy_discover(base_url, *, timeout):
        calls.append(base_url)
        return []

    with (
        patch(
            "EvoScientist.langgraph_dev.http.get_effective_config",
            return_value=mock_cfg,
        ),
        patch(
            "EvoScientist.llm.ollama_discovery.discover_ollama_models",
            new=spy_discover,
        ),
    ):
        body = client.get("/api/models").json()

    assert calls == []
    # Response is exactly the static registry — no Ollama additions whatsoever.
    from EvoScientist.llm.models import list_models_by_provider

    assert body["entries"] == [
        {"name": n, "model_id": m, "provider": p}
        for n, m, p in list_models_by_provider()
    ]


# ---- /api/teams -----------------------------------------------------------


def _expert_info(
    name: str,
    *,
    description: str = "",
    byline: str = "",
    capability_tags: list[str] | None = None,
    avatar_hint: str = "",
):
    """Build a SkillInfo for an expert skill (agent-teams v1)."""
    from pathlib import Path

    from EvoScientist.tools.skills_manager import SkillInfo

    return SkillInfo(
        name=name,
        description=description or f"{name} description",
        path=Path(f"/skills/{name}"),
        source="builtin",
        type="expert",
        byline=byline,
        capability_tags=list(capability_tags or []),
        avatar_hint=avatar_hint,
    )


def test_get_teams_returns_installed_expert_skills():
    experts = [
        _expert_info("expert-a", description="First expert"),
        _expert_info("expert-b", description="Second expert"),
    ]
    with patch(
        "EvoScientist.tools.skills_manager.list_expert_skills",
        return_value=experts,
    ):
        resp = client.get("/api/teams")
    assert resp.status_code == 200
    body = resp.json()
    assert "teams" in body
    names = [t["name"] for t in body["teams"]]
    assert names == ["expert-a", "expert-b"]


def test_get_teams_omits_backend_implementation_fields():
    """Never leak SKILL.md body / role / dispatch / source / path / etc.
    onto the gallery endpoint — those are backend-only."""
    experts = [_expert_info("expert-a")]
    with patch(
        "EvoScientist.tools.skills_manager.list_expert_skills",
        return_value=experts,
    ):
        body = client.get("/api/teams").json()
    entry = body["teams"][0]
    forbidden = {
        "system_prompt",
        "role",
        "default_dispatch",
        "type",
        "source",
        "path",
        "tools",
        "skills",
        "tags",
        "_async",
    }
    assert not (set(entry.keys()) & forbidden), (
        f"leaked backend fields: {set(entry.keys()) & forbidden}"
    )


def test_get_teams_projects_optional_gallery_metadata_when_present():
    experts = [
        _expert_info(
            "idea-brainstorm",
            description="Multi-round brainstorm",
            byline="Research idea brainstormer",
            capability_tags=["Iteration", "ELO ranking"],
            avatar_hint="lightbulb",
        ),
    ]
    with patch(
        "EvoScientist.tools.skills_manager.list_expert_skills",
        return_value=experts,
    ):
        body = client.get("/api/teams").json()
    entry = body["teams"][0]
    assert entry["name"] == "idea-brainstorm"
    assert entry["description"] == "Multi-round brainstorm"
    assert entry["byline"] == "Research idea brainstormer"
    assert entry["capability_tags"] == ["Iteration", "ELO ranking"]
    assert entry["avatar_hint"] == "lightbulb"


def test_get_teams_omits_optional_fields_when_absent():
    """Gallery card should degrade gracefully when an expert declares
    only the minimum (name, description, type: expert)."""
    experts = [_expert_info("minimal-expert")]  # no byline / tags / avatar
    with patch(
        "EvoScientist.tools.skills_manager.list_expert_skills",
        return_value=experts,
    ):
        body = client.get("/api/teams").json()
    entry = body["teams"][0]
    assert set(entry.keys()) == {"name", "description"}


def test_get_teams_returns_empty_list_when_no_experts_installed():
    with patch(
        "EvoScientist.tools.skills_manager.list_expert_skills",
        return_value=[],
    ):
        body = client.get("/api/teams").json()
    assert body == {"teams": []}


def test_get_teams_calls_loader_with_include_system_true():
    """First-party experts ship as builtin skills; the endpoint must
    include the builtin tier or the gallery will be empty on a fresh
    workspace with no user-installed experts."""
    calls = []

    def spy(include_system=False):
        calls.append(include_system)
        return []

    with patch(
        "EvoScientist.tools.skills_manager.list_expert_skills",
        new=spy,
    ):
        client.get("/api/teams")
    assert calls == [True]
