"""Tests for the expert-registry rebuild in ``_get_default_agent``.

The module-level agent is rebuilt when ``dispatchable_experts_token`` moves,
so an expert installed mid-session becomes dispatchable without ``/new``. The
WebUI graph factory calls this per HTTP request, which is what makes the
failure branch load-bearing: a build that raises must not take a working
deployment down.
"""

from __future__ import annotations

import pytest

import EvoScientist.EvoScientist as agent_module
from EvoScientist.subagents import expert_container


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

    Failing at the kwargs step is also where a malformed expert actually bites:
    ``_fold_expert_subagents`` runs inside ``_build_base_kwargs``.
    """
    calls: list[int] = []

    def _fail(*_args, **_kwargs):
        calls.append(1)
        raise RuntimeError("bad expert spec")

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

    def test_a_failed_rebuild_rolls_the_cue_back(self, build, monkeypatch):
        """The staleness probe restamps the memo before the build is tried.

        If the build then fails, the expert prompt and the ``/expert`` popup
        must not go on offering an expert ``task()`` cannot route.
        """
        monkeypatch.setattr(
            expert_container, "_reset_dispatchable_experts_cache", lambda: None
        )
        expert_container._dispatchable_cache_key = (0, True)
        expert_container._dispatchable_cache_value = ["seated-set"]
        expert_container._dispatchable_memo_prev = ((0, True), ["seated-set"])

        def _token_that_restamps(**_kwargs):
            expert_container._dispatchable_memo_prev = (
                expert_container._dispatchable_cache_key,
                expert_container._dispatchable_cache_value,
            )
            expert_container._dispatchable_cache_value = ["seated-set", "not-built"]
            return "new"

        monkeypatch.setattr(
            expert_container, "dispatchable_experts_token", _token_that_restamps
        )
        _seat(object(), "old")

        agent_module._get_default_agent()

        assert expert_container._dispatchable_cache_value == ["seated-set"]

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

        with pytest.raises(RuntimeError, match="bad expert spec"):
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
