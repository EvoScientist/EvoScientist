"""Tests for EvoScientist.setup.agent_env: the agent shell's environment."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from EvoScientist.setup import agent_env, node, research_env
from EvoScientist.setup._install import tools_dir


def _node_dir(directory: Path) -> Path:
    """A folder holding ``node`` and ``npx`` the PATH lookup accepts here."""
    directory.mkdir(parents=True)
    names = ["node.exe", "npx.cmd"] if os.name == "nt" else ["node", "npx"]
    for name in names:
        (directory / name).write_text("")
        (directory / name).chmod(0o755)
    return directory


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """Private Node recorded and first on PATH, then the user's Node; mirror
    off; no research environment; no NODE_USE_SYSTEM_CA of the user's."""
    install = tools_dir() / "node-v24.21.0"
    private = _node_dir(node._bin_dir(install))
    monkeypatch.setattr(node, "_read_record", lambda: {"path": str(install)})
    user = _node_dir(tmp_path / "nvm" / "v22" / "bin")
    monkeypatch.setenv("PATH", os.pathsep.join([str(private), str(user)]))
    monkeypatch.delenv("NODE_USE_SYSTEM_CA", raising=False)
    state = {"usable": True, "mirror": "default", "registry": None, "asked": []}
    agent_env._user_node_usable.cache_clear()
    monkeypatch.setattr(
        node, "_system_node", lambda: object() if state["usable"] else None
    )
    monkeypatch.setattr(node, "configured_mirror", lambda: state["mirror"])

    def configured(_env, node_exe):
        state["asked"].append(Path(node_exe).parent)
        return state["registry"]

    monkeypatch.setattr(node, "configured_npm_registry", configured)
    monkeypatch.setattr(research_env, "research_env_overrides", lambda: None)
    yield {"private": str(private), "user": str(user), "state": state}
    agent_env._user_node_usable.cache_clear()


def test_private_node_goes_last_when_the_user_has_a_usable_node(setup):
    before = os.environ["PATH"]
    overrides = agent_env.agent_shell_overrides()
    assert overrides["PATH"].split(os.pathsep) == [setup["user"], setup["private"]]
    assert os.environ["PATH"] == before


@pytest.mark.parametrize(
    "extra",
    [
        "bin",  # relative: resolves against the working directory
        pytest.param(
            "",
            marks=pytest.mark.skipif(
                os.name == "nt", reason="Windows ignores empty PATH entries"
            ),
        ),
    ],
)
def test_private_node_stays_first_when_path_searches_the_workspace(
    setup, monkeypatch, extra
):
    """Moved last, the private Node would come after the working directory,
    where a ``node`` in the agent's workspace could win."""
    path = os.pathsep.join([setup["private"], extra, setup["user"]])
    monkeypatch.setenv("PATH", path)
    assert "PATH" not in (agent_env.agent_shell_overrides() or {})


def test_private_node_stays_first_when_the_users_node_was_rejected(setup):
    """A Node EvoScientist did not accept (too old, no npx) must not win. The
    private one then trusts the system's certificates, as its other
    children do."""
    setup["state"]["usable"] = False
    assert agent_env.agent_shell_overrides() == {"NODE_USE_SYSTEM_CA": "1"}


def test_the_users_system_ca_choice_is_kept(setup, monkeypatch):
    setup["state"]["usable"] = False
    monkeypatch.setenv("NODE_USE_SYSTEM_CA", "0")
    assert agent_env.agent_shell_overrides() is None


def test_the_users_node_gets_no_system_ca_switch(setup):
    assert "NODE_USE_SYSTEM_CA" not in agent_env.agent_shell_overrides()


def test_research_environment_stays_first(setup, monkeypatch, tmp_path):
    # Absolute on every OS: on Windows, Python 3.13 no longer counts a path
    # without a drive as absolute.
    venv = tmp_path / "envs" / "default"
    venv_bin = str(venv / "bin")
    monkeypatch.setattr(
        research_env,
        "research_env_overrides",
        lambda: {
            "PATH": os.pathsep.join([venv_bin, os.environ["PATH"]]),
            "VIRTUAL_ENV": str(venv),
        },
    )
    overrides = agent_env.agent_shell_overrides()
    assert overrides["PATH"].split(os.pathsep) == [
        venv_bin,
        setup["user"],
        setup["private"],
    ]
    assert overrides["VIRTUAL_ENV"] == str(venv)


def test_mirror_cn_adds_npmmirror_for_npx(setup):
    setup["state"].update(mirror="cn", usable=False)
    assert agent_env.agent_shell_overrides()["npm_config_registry"] == (
        "https://registry.npmmirror.com/"
    )


def test_the_registry_is_asked_of_the_agents_own_node(setup):
    """The npm next to the first ``node`` on the agent's PATH: the user's
    after the move."""
    setup["state"]["mirror"] = "cn"
    agent_env.agent_shell_overrides()
    assert setup["state"]["asked"] == [Path(setup["user"])]


def test_a_registry_the_user_set_wins_under_cn(setup):
    setup["state"].update(
        mirror="cn", usable=False, registry="https://registry.yarnpkg.com/"
    )
    assert "npm_config_registry" not in agent_env.agent_shell_overrides()


def test_nothing_to_change_without_a_private_node(tmp_path, monkeypatch):
    user = _node_dir(tmp_path / "usr" / "bin")
    monkeypatch.setenv("PATH", str(user))
    monkeypatch.setattr(node, "_read_record", lambda: None)
    monkeypatch.setattr(node, "configured_mirror", lambda: "default")
    monkeypatch.setattr(research_env, "research_env_overrides", lambda: None)
    agent_env._user_node_usable.cache_clear()
    assert agent_env.agent_shell_overrides() is None


def test_the_usable_node_check_runs_once_per_process(setup, monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(node, "_system_node", lambda: calls.append(1) or object())
    agent_env.agent_shell_overrides()
    agent_env.agent_shell_overrides()
    assert calls == [1]
