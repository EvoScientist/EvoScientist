"""The agent's ``execute`` backends get the research environment overrides."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import EvoScientist.EvoScientist as es_mod
from EvoScientist.setup import research_env

_PROBE = {"EVOSCI_SHELL_PROBE": "venv"}


@pytest.fixture
def overrides(monkeypatch):
    monkeypatch.setattr(research_env, "research_env_overrides", lambda: dict(_PROBE))


def test_default_backend_gets_the_overrides(overrides):
    backend = es_mod._get_default_backend()
    assert backend.default._env["EVOSCI_SHELL_PROBE"] == "venv"


def test_cli_agent_backend_gets_the_overrides(overrides, tmp_path):
    backends = []

    def fake_build_kwargs(backend, *args, **kwargs):
        backends.append(backend)
        return {"name": "x"}

    cfg = MagicMock()
    cfg.dangerous_mode = False
    cfg.sandbox_execute_timeout = 300
    cfg.recursion_limit = 100
    with (
        patch("deepagents.create_deep_agent", return_value=MagicMock()),
        patch.object(es_mod, "_apply_env_from_config"),
        patch.object(es_mod, "_get_default_middleware", return_value=[]),
        patch.object(
            es_mod, "load_mcp_and_build_kwargs", side_effect=fake_build_kwargs
        ),
    ):
        es_mod.create_cli_agent(
            workspace_dir=str(tmp_path), config=cfg, chat_model=MagicMock()
        )
    assert backends[0].default._env["EVOSCI_SHELL_PROBE"] == "venv"


def test_every_sandbox_backend_construction_passes_env():
    """A new agent backend that skips the overrides would give that agent a
    different ``python`` from the others."""
    root = Path(es_mod.__file__).parent
    sites = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        # The class or a subclass of it (e.g. AutoskillProposalSandboxBackend).
        for match in re.finditer(
            r"=\s*\w*SandboxBackend\((.*?)\n\s*\)", text, re.DOTALL
        ):
            sites.append((path.name, match.group(1)))
    assert len(sites) == 3
    for name, args in sites:
        assert "env=research_env_overrides()" in args, name


def test_autoskill_backend_gets_the_overrides(overrides, tmp_path):
    from EvoScientist.backends import build_autoskill_agent_backend

    backend = build_autoskill_agent_backend(
        memory_dir=tmp_path / "memories", proposals_dir=tmp_path / "proposals"
    )
    assert backend.default._env["EVOSCI_SHELL_PROBE"] == "venv"
