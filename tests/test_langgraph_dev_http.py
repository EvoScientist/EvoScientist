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
    with patch("EvoScientist.langgraph_dev.http.load_config", return_value=mock_cfg):
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
    """
    from EvoScientist.llm.models import list_models_by_provider

    expected = [
        {"name": n, "model_id": m, "provider": p}
        for n, m, p in list_models_by_provider()
    ]
    resp = client.get("/api/models")
    assert resp.json()["entries"] == expected


def test_default_passes_through_arbitrary_config_pair():
    """If config.yaml names a (name, provider) pair that isn't in the
    registry (typo, retired model), still report it as default — the
    picker labels it as the active selection regardless.
    """
    mock_cfg = EvoScientistConfig(model="some-retired-name", provider="some-provider")
    with patch("EvoScientist.langgraph_dev.http.load_config", return_value=mock_cfg):
        resp = client.get("/api/models")
    assert resp.json()["default"] == {
        "name": "some-retired-name",
        "provider": "some-provider",
    }


def test_langgraph_json_registers_http_app():
    """Backend wiring: the langgraph dev config must point at our
    FastAPI app or the route won't be served.
    """
    import json
    from pathlib import Path

    cfg_path = (
        Path(__file__).parent.parent / "EvoScientist/langgraph_dev/langgraph.json"
    )
    with open(cfg_path) as f:
        cfg = json.load(f)
    assert cfg.get("http") == {"app": "EvoScientist.langgraph_dev.http:app"}
