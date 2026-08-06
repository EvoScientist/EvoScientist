"""Tests for the expert-registry rebuild in ``_get_default_agent``.

The module-level agent is rebuilt when ``dispatchable_experts_token`` moves,
so an expert installed mid-session becomes dispatchable without ``/new``. The
WebUI graph factory calls this per HTTP request, which is what makes the
failure branch load-bearing: a build that raises must not take a working
deployment down.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import EvoScientist.EvoScientist as agent_module
from EvoScientist.subagents import expert_container
from EvoScientist.tools.skills_manager import SkillInfo

_SKILLS = "EvoScientist.tools.skills_manager.list_expert_skills"


def _expert(name: str) -> SkillInfo:
    return SkillInfo(
        name=name,
        description=f"{name} description",
        path=Path("/tmp/nope") / name,
        source="builtin",
        type="expert",
        role=f"{name} role",
        body="persona body\n",
    )


@pytest.fixture(autouse=True)
def _isolate_agent_globals():
    """Snapshot and restore the module globals this suite mutates."""
    saved = (
        agent_module._EvoScientist_agent,
        agent_module._EvoScientist_agent_expert_token,
    )
    agent_module._EvoScientist_agent = None
    agent_module._EvoScientist_agent_expert_token = None
    try:
        yield
    finally:
        (
            agent_module._EvoScientist_agent,
            agent_module._EvoScientist_agent_expert_token,
        ) = saved


def _seat(agent: object, token: str) -> None:
    """Put a pre-built agent in the module cache, as a successful build would."""
    agent_module._EvoScientist_agent = agent
    agent_module._EvoScientist_agent_expert_token = token


@pytest.fixture
def build(monkeypatch):
    """Make the build fail at the kwargs step, counting attempts.

    The three calls ahead of it are stubbed because ``_get_default_middleware``
    reaches ``_ensure_chat_model`` and would raise on the missing model
    configuration before the interesting failure — these tests stay runnable
    without credentials, unlike the factory tests that construct a real agent.

    The failure is injected rather than provoked with a malformed expert,
    because expert content cannot fail a build: empty bodies, colliding names
    and unresolved tools are each skipped with a warning. What can fail is the
    surrounding construction a rebuild re-runs — backends against live paths,
    and the chat model — which is what this stands in for.
    """
    calls: list[int] = []

    def _fail(*_args, **_kwargs):
        calls.append(1)
        raise RuntimeError("backend construction failed")

    monkeypatch.setattr(agent_module, "_ensure_config", lambda: _StubConfig())
    monkeypatch.setattr(agent_module, "_get_default_backend", lambda: object())
    monkeypatch.setattr(agent_module, "_get_default_middleware", lambda: [])
    monkeypatch.setattr(agent_module, "load_mcp_and_build_kwargs", _fail)
    monkeypatch.setattr(agent_module, "_build_base_kwargs", _fail)
    return calls


class _StubConfig:
    auto_approve = False
    recursion_limit = 100


class TestFailedRebuildKeepsTheSeatedAgent:
    """A build that raises must degrade, not kill the deployment.

    The rebuild trigger is agent-authored content — the workspace skills tier
    is writable so the agent can be asked to write an expert — so a
    half-written file reaches this path in normal use.
    """

    def test_a_failed_rebuild_returns_the_previous_agent(self, build, monkeypatch):
        monkeypatch.setattr(
            expert_container, "dispatchable_experts_token", lambda **_k: "new"
        )
        seated = object()
        _seat(seated, "old")

        assert agent_module._get_default_agent() is seated
        assert agent_module._EvoScientist_agent is seated

    def test_a_failed_rebuild_is_not_retried_on_every_call(self, build, monkeypatch):
        """The token of the *attempt* is stamped, not the old one.

        Without that, the WebUI would re-enter the failing build once per
        HTTP request — including for read-only state polls.
        """
        monkeypatch.setattr(
            expert_container, "dispatchable_experts_token", lambda **_k: "new"
        )
        _seat(object(), "old")

        for _ in range(3):
            agent_module._get_default_agent()

        assert len(build) == 1

    def test_a_failed_rebuild_leaves_the_cue_on_the_seated_set(self, build):
        """The expert prompt and the ``/expert`` popup must not offer an
        expert ``task()`` cannot route to.

        Drives the real ``dispatchable_experts_token`` and the real memo,
        patching only the skills listing underneath them — a stand-in for
        either would let a scheme that writes the memo before the build and
        undoes it afterwards pass while failing in production.
        """
        seated = object()
        with patch(_SKILLS, return_value=[_expert("alpha")]):
            # The set the seated agent was built from.
            expert_container.list_dispatchable_experts()
            _seat(seated, expert_container.dispatchable_experts_token())

        with patch(_SKILLS, return_value=[_expert("alpha"), _expert("beta")]):
            assert agent_module._get_default_agent() is seated
            names = [s.name for s in expert_container.list_dispatchable_experts()]
            assert names == ["alpha"]

    def test_a_failed_first_build_still_raises(self, build, monkeypatch):
        """With no previous agent there is nothing to degrade to.

        Also the ``_replace_chat_model`` path: it nulls the agent global
        itself, so ``previous`` is ``None`` and a failed rebuild cannot
        re-seat an agent still bound to the superseded chat model.
        """
        monkeypatch.setattr(
            expert_container, "dispatchable_experts_token", lambda **_k: "new"
        )
        agent_module._EvoScientist_agent = None

        with pytest.raises(RuntimeError, match="backend construction failed"):
            agent_module._get_default_agent()

    def test_a_failed_rebuild_warns(self, build, monkeypatch, caplog):
        """A silent fallback would leave nobody able to tell it is still in effect."""
        monkeypatch.setattr(
            expert_container, "dispatchable_experts_token", lambda **_k: "new"
        )
        _seat(object(), "old")

        with caplog.at_level("WARNING"):
            agent_module._get_default_agent()

        assert any("Agent rebuild failed" in r.message for r in caplog.records)


class TestUnchangedRegistryDoesNotRebuild:
    def test_a_matching_token_returns_the_cached_agent(self, monkeypatch):
        monkeypatch.setattr(
            expert_container, "dispatchable_experts_token", lambda **_k: "same"
        )
        seated = object()
        _seat(seated, "same")

        # No build stub needed: reaching the build at all would raise on the
        # missing model configuration, so returning cleanly is the assertion.
        assert agent_module._get_default_agent() is seated
