"""Tests for ``cli/agent``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import EvoScientist.EvoScientist as evo_mod
from EvoScientist.cli.agent import _load_agent
from EvoScientist.paths import Workspace


def _server_gateway():
    from EvoScientist.gateway.server import LangGraphServerGateway

    return object.__new__(LangGraphServerGateway)


@pytest.fixture
def builds(monkeypatch):
    calls: list[dict] = []

    def _create_cli_agent(**kwargs):
        calls.append(kwargs)
        return "AGENT"

    monkeypatch.setattr(evo_mod, "create_cli_agent", _create_cli_agent)
    monkeypatch.setattr(evo_mod, "_config", None)
    monkeypatch.setattr(evo_mod, "apply_config_to_env", lambda _cfg: None)
    return calls


@pytest.mark.parametrize(
    "gateway",
    [_server_gateway(), SimpleNamespace(execute_gateway=_server_gateway())],
    ids=["server", "composite"],
)
def test_no_agent_is_built_for_a_gateway_the_server_runs(builds, tmp_path, gateway):
    config = SimpleNamespace(model="m")
    work_dir = tmp_path / "runs" / "r1"

    agent = _load_agent(
        work_dir=str(work_dir),
        config=config,
        workspace=Workspace(tmp_path),
        graph_gateway=gateway,
    )

    assert agent is None
    assert builds == []
    # The server gateway sends the active config with each run.
    assert evo_mod._ensure_config() is config
    assert work_dir.is_dir()


@pytest.mark.parametrize(
    "gateway",
    [None, object(), SimpleNamespace(execute_gateway=object())],
    ids=["none", "local", "composite-fallback"],
)
def test_the_agent_is_built_for_an_in_process_gateway(builds, tmp_path, gateway):
    config = SimpleNamespace(model="m")

    agent = _load_agent(
        config=config, workspace=Workspace(tmp_path), graph_gateway=gateway
    )

    assert agent == "AGENT"
    assert len(builds) == 1
    assert builds[0]["config"] is config
