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


def test_the_workspace_root_is_created_when_no_work_folder_is_given(builds, tmp_path):
    """Serve loads without a work folder, for a ``--workdir`` that may not exist."""
    root = tmp_path / "new-workspace"

    _load_agent(
        config=SimpleNamespace(model="m"),
        workspace=Workspace(root),
        graph_gateway=_server_gateway(),
    )

    assert builds == []
    assert root.is_dir()


@pytest.mark.parametrize(
    "flags", [{"auto_approve": True}, {"auto_mode": True}], ids=lambda f: next(iter(f))
)
def test_cli_flags_reach_the_server_run_without_the_build(builds, tmp_path, flags):
    """``--auto-approve`` / ``--auto-mode`` and ``--model`` reach the server run
    though no agent was built: the run config is read from the active config."""
    from EvoScientist.backends import HITL_SUPPRESSED_KEY, MODEL_RETRY_RUN_KEY
    from EvoScientist.config import EvoScientistConfig

    gateway = _server_gateway()
    config = EvoScientistConfig(model="flag-model", provider="anthropic", **flags)

    _load_agent(config=config, workspace=Workspace(tmp_path), graph_gateway=gateway)
    run = gateway._resolve_run_config("t1", None)["configurable"]

    assert builds == []
    assert run[HITL_SUPPRESSED_KEY] is True
    assert run["model"] == "flag-model"
    if flags.get("auto_mode"):
        assert run[MODEL_RETRY_RUN_KEY]["unattended"] is True


@pytest.mark.parametrize("gateway", [None, object()], ids=["none", "local"])
def test_the_agent_is_built_for_an_in_process_gateway(builds, tmp_path, gateway):
    config = SimpleNamespace(model="m")

    agent = _load_agent(
        config=config, workspace=Workspace(tmp_path), graph_gateway=gateway
    )

    assert agent == "AGENT"
    assert len(builds) == 1
    assert builds[0]["config"] is config


def test_the_agent_is_built_when_the_server_did_not_come_up(
    builds, monkeypatch, tmp_path
):
    """A surface set to the server backend falls back to the in-process
    gateway, and then needs the agent."""
    import EvoScientist.langgraph_dev.manager as manager
    from EvoScientist.gateway import create_runtime_gateways_for_config

    monkeypatch.setattr(manager, "is_async_subagents_available", lambda: False)
    gateways = create_runtime_gateways_for_config(
        SimpleNamespace(), backend="langgraph_server"
    )

    agent = _load_agent(
        config=SimpleNamespace(model="m"),
        workspace=Workspace(tmp_path),
        graph_gateway=gateways.graph_gateway,
    )

    assert agent == "AGENT"
    assert len(builds) == 1
