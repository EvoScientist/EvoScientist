"""Tests for the deployed graph's background expert-registry refresh.

The request path must stay inert — a staleness check there is blocking work on
the event loop, which langgraph-dev's blockbuster guard raises on rather than
merely slows. These tests pin that the watching happens off-request, that a
skill install wakes it without waiting out the poll, and that a bad pass cannot
end the watch.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from EvoScientist.langgraph_dev import registry_refresh
from EvoScientist.subagents.expert_container import (
    dispatchable_experts_token as _REAL_TOKEN,
)
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
def _deploy_mode(monkeypatch):
    """Run as the deployed main agent unless a test says otherwise."""
    monkeypatch.setenv("EVOSCIENTIST_DEPLOY_MODE", "full")


@pytest.fixture(autouse=True)
def _fast_poll(monkeypatch):
    """Keep the poll backstop far below any test timeout."""
    monkeypatch.setattr(registry_refresh, "POLL_INTERVAL_SECONDS", 0.01)


@pytest.fixture(autouse=True)
def _isolate_skills_callbacks(monkeypatch):
    """The callback list is process-scoped and never unregistered."""
    from EvoScientist.tools import skills_manager

    monkeypatch.setattr(skills_manager, "_skills_changed_callbacks", [])


class _Registry:
    """Stand-in for the token source and the rebuild."""

    def __init__(self, token: str = "t0") -> None:
        self.token = token
        self.rebuilds = 0
        self.token_reads = 0
        self.fail_next = False

    def read_token(self, *, include_system: bool = True) -> str:
        self.token_reads += 1
        return self.token

    def rebuild(self):
        self.rebuilds += 1
        if self.fail_next:
            raise RuntimeError("backend construction failed")
        return object()


@pytest.fixture
def registry(monkeypatch):
    reg = _Registry()
    monkeypatch.setattr(
        "EvoScientist.subagents.expert_container.dispatchable_experts_token",
        reg.read_token,
    )
    monkeypatch.setattr(
        "EvoScientist.langgraph_dev.main_graph.refresh_main_graph", reg.rebuild
    )
    return reg


async def _settle(predicate, timeout: float = 2.0) -> None:
    """Await a condition without pinning the test to a sleep duration."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition not reached within timeout")


class TestRefreshLoop:
    async def test_builds_once_at_startup(self, registry):
        async with registry_refresh.expert_registry_refresher():
            await _settle(lambda: registry.rebuilds >= 1)
        assert registry.rebuilds == 1

    async def test_unchanged_registry_does_not_rebuild_again(self, registry):
        async with registry_refresh.expert_registry_refresher():
            await _settle(lambda: registry.rebuilds >= 1)
            # Several poll intervals with a stable token.
            await asyncio.sleep(0.1)
        assert registry.rebuilds == 1

    async def test_a_changed_token_rebuilds(self, registry):
        async with registry_refresh.expert_registry_refresher():
            await _settle(lambda: registry.rebuilds >= 1)
            registry.token = "t1"
            await _settle(lambda: registry.rebuilds >= 2)
        assert registry.rebuilds == 2

    async def test_detection_never_stamps_the_cue_memo(self, registry, monkeypatch):
        """Stamping at detection time would have the active-expert cue name
        the new expert for the whole rebuild, before ``task()`` can route to it.

        Restores the real token over the fixture's stand-in and asserts
        against the real memo, so this still holds if the refresher starts
        reading it some other way. The stubbed rebuild publishes nothing, so
        anything in the memo afterwards came from detection.
        """
        from EvoScientist.subagents import expert_container

        monkeypatch.setattr(
            expert_container,
            "dispatchable_experts_token",
            _REAL_TOKEN,
        )
        with patch(_SKILLS, return_value=[_expert("alpha")]):
            async with registry_refresh.expert_registry_refresher():
                await _settle(lambda: registry.rebuilds >= 1)
        assert expert_container._dispatchable_cache_value is None

    async def test_a_failed_rebuild_does_not_end_the_watch(self, registry, caplog):
        """One bad build must not disable refresh for the rest of the process."""
        registry.fail_next = True
        async with registry_refresh.expert_registry_refresher():
            await _settle(lambda: registry.rebuilds >= 1)
            registry.fail_next = False
            registry.token = "t2"
            await _settle(lambda: registry.rebuilds >= 2)
        assert "refresh pass failed" in caplog.text

    async def test_a_failed_pass_is_retried_rather_than_accepted(self, registry):
        """The failed token must not be recorded as current, or the retry
        would be skipped and the expert would never become reachable."""
        registry.fail_next = True
        async with registry_refresh.expert_registry_refresher():
            await _settle(lambda: registry.rebuilds >= 2)
        assert registry.rebuilds >= 2


class TestPushWakeup:
    async def test_a_skill_install_wakes_the_loop(self, registry, monkeypatch):
        """Installs must not wait out the poll interval.

        Set the poll far beyond the test's patience, so only the push hook can
        drive the second rebuild.
        """
        monkeypatch.setattr(registry_refresh, "POLL_INTERVAL_SECONDS", 3600.0)
        from EvoScientist.tools import skills_manager

        async with registry_refresh.expert_registry_refresher():
            await _settle(lambda: registry.rebuilds >= 1)
            registry.token = "t1"
            # Fires on a worker thread in production; the hop back to the loop
            # is the loader's job, so exercise it from a thread here too.
            await asyncio.to_thread(skills_manager._notify_skills_changed)
            await _settle(lambda: registry.rebuilds >= 2)
        assert registry.rebuilds == 2


class TestDeployModeGate:
    async def test_no_watch_outside_deploy_mode(self, registry, monkeypatch):
        """Under ``EvoSci`` / ``EvoSci serve`` the deployed main agent is dead
        code, so refreshing it would be pure waste."""
        monkeypatch.setenv("EVOSCIENTIST_DEPLOY_MODE", "stripped")
        async with registry_refresh.expert_registry_refresher():
            await asyncio.sleep(0.1)
        assert registry.rebuilds == 0

    async def test_no_watch_when_unset(self, registry, monkeypatch):
        monkeypatch.delenv("EVOSCIENTIST_DEPLOY_MODE", raising=False)
        async with registry_refresh.expert_registry_refresher():
            await asyncio.sleep(0.1)
        assert registry.rebuilds == 0


class TestShutdown:
    async def test_exiting_cancels_the_task(self, registry):
        async with registry_refresh.expert_registry_refresher():
            await _settle(lambda: registry.rebuilds >= 1)
        # No pending task should survive the context manager.
        pending = [
            t
            for t in asyncio.all_tasks()
            if t.get_name() == "evoscientist-expert-registry-refresh"
        ]
        assert pending == []

    async def test_shutdown_is_clean_before_the_first_pass(self, registry):
        """Entering and immediately leaving must not raise or warn."""
        async with registry_refresh.expert_registry_refresher():
            pass
