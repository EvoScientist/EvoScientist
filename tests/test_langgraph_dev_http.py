"""Smoke test for the /api/models route mounted via langgraph.json's
``http`` field. We test the FastAPI app directly — no need to spin up
langgraph dev.
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
