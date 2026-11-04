"""Tests for MCP tool caching in EvoScientist.EvoScientist."""

from __future__ import annotations

import EvoScientist.EvoScientist as agent_module


def _reset_mcp_cache() -> None:
    agent_module._MCP_TOOLS_CACHE_KEY = None
    agent_module._MCP_TOOLS_CACHE_VALUE = None


class TestMcpToolCaching:
    def setup_method(self) -> None:
        _reset_mcp_cache()

    def test_reuses_cached_tools_when_config_unchanged(self, monkeypatch):
        calls = {"load": 0}
        tool = object()

        monkeypatch.setattr(
            "EvoScientist.mcp.client.load_mcp_config",
            lambda: {"srv": {"transport": "stdio", "command": "demo"}},
        )

        def fake_load_mcp_tools(config=None, **_kwargs):
            calls["load"] += 1
            return {"main": [tool]}

        monkeypatch.setattr("EvoScientist.mcp.load_mcp_tools", fake_load_mcp_tools)

        first = agent_module._load_mcp_tools_cached()
        second = agent_module._load_mcp_tools_cached()

        assert calls["load"] == 1
        assert first == second
        assert first is not second
        assert first["main"] is not second["main"]

    def test_reload_when_config_changes(self, monkeypatch):
        calls = {"load": 0}
        state = {"cfg": {"srv": {"transport": "stdio", "command": "v1"}}}

        def fake_load_config():
            return state["cfg"]

        def fake_load_mcp_tools(config=None, **_kwargs):
            calls["load"] += 1
            return {"main": [f"tool-v{calls['load']}"]}

        monkeypatch.setattr("EvoScientist.mcp.client.load_mcp_config", fake_load_config)
        monkeypatch.setattr("EvoScientist.mcp.load_mcp_tools", fake_load_mcp_tools)

        first = agent_module._load_mcp_tools_cached()
        state["cfg"] = {"srv": {"transport": "stdio", "command": "v2"}}
        second = agent_module._load_mcp_tools_cached()

        assert calls["load"] == 2
        assert first != second

    def test_load_mcp_config_called_once_per_cache_miss(self, monkeypatch):
        """load_mcp_config should be called exactly once per cache miss,
        not twice (once for the signature and once inside load_mcp_tools)."""
        calls = {"config": 0}

        def counting_load_config():
            calls["config"] += 1
            return {"srv": {"transport": "stdio", "command": "demo"}}

        monkeypatch.setattr(
            "EvoScientist.mcp.client.load_mcp_config", counting_load_config
        )
        monkeypatch.setattr(
            "EvoScientist.mcp.load_mcp_tools",
            lambda config=None, **_kwargs: {"main": []},
        )

        agent_module._load_mcp_tools_cached()
        assert calls["config"] == 1

    def test_cached_config_passed_to_load_mcp_tools(self, monkeypatch):
        """load_mcp_tools should receive the pre-loaded config dict."""
        received = {}

        def fake_load_config():
            return {"srv": {"transport": "stdio", "command": "demo"}}

        def fake_load_mcp_tools(config=None, **_kwargs):
            received["config"] = config
            return {"main": []}

        monkeypatch.setattr("EvoScientist.mcp.client.load_mcp_config", fake_load_config)
        monkeypatch.setattr("EvoScientist.mcp.load_mcp_tools", fake_load_mcp_tools)

        agent_module._load_mcp_tools_cached()
        assert received["config"] == {"srv": {"transport": "stdio", "command": "demo"}}


def _reset_mcp_failure_state() -> None:
    _reset_mcp_cache()
    agent_module._MCP_TOOLS_CACHE_FAILED = set()
    agent_module._MCP_TOOLS_CACHE_LAST_ATTEMPT = 0.0
    agent_module._MCP_TOOLS_CACHE_GENERATION = 0


class TestMcpFailedServerRetry:
    """A server whose load fails is retried later instead of staying
    tool-less until the process restarts (issue #696)."""

    def setup_method(self) -> None:
        _reset_mcp_failure_state()

    def _install(self, monkeypatch, behavior):
        """behavior: dict server name -> "ok" | "fail" | "whole-fail".

        Returns the list of configs ``load_mcp_tools`` was called with.
        """
        calls: list[dict] = []
        config = {
            name: {"transport": "stdio", "command": f"demo-{name}"}
            for name in behavior
        }
        monkeypatch.setattr(
            "EvoScientist.mcp.client.load_mcp_config", lambda: dict(config)
        )

        def fake_load_mcp_tools(config=None, on_progress=None, **_kwargs):
            config = config or {}
            calls.append(dict(config))
            if any(behavior[n] == "whole-fail" for n in config):
                return {}
            routed: dict[str, list] = {}
            for name in config:
                if on_progress is not None:
                    on_progress("start", name, "")
                if behavior[name] == "fail":
                    if on_progress is not None:
                        on_progress("error", name, "timed out after 40s")
                    continue
                if on_progress is not None:
                    on_progress("success", name, "1")
                routed.setdefault("main", []).append(f"tool-from-{name}")
            return routed

        monkeypatch.setattr("EvoScientist.mcp.load_mcp_tools", fake_load_mcp_tools)
        return calls

    def test_failed_server_tools_appear_once_it_recovers(self, monkeypatch):
        behavior = {"good": "ok", "flaky": "fail"}
        calls = self._install(monkeypatch, behavior)
        monkeypatch.setattr(agent_module, "_MCP_FAILED_SERVER_RETRY_SECONDS", 0.0)

        first = agent_module._load_mcp_tools_cached()
        assert first == {"main": ["tool-from-good"]}

        behavior["flaky"] = "ok"
        second = agent_module._load_mcp_tools_cached()

        assert second == {"main": ["tool-from-good", "tool-from-flaky"]}
        # The retry loaded only the server that had failed.
        assert [sorted(c) for c in calls] == [["flaky", "good"], ["flaky"]]

        # Nothing is left failed, so the cache is served without loading.
        third = agent_module._load_mcp_tools_cached()
        assert third == second
        assert len(calls) == 2

    def test_failed_server_is_not_retried_before_the_delay(self, monkeypatch):
        behavior = {"good": "ok", "flaky": "fail"}
        calls = self._install(monkeypatch, behavior)
        monkeypatch.setattr(
            agent_module, "_MCP_FAILED_SERVER_RETRY_SECONDS", 3600.0
        )

        agent_module._load_mcp_tools_cached()
        behavior["flaky"] = "ok"
        second = agent_module._load_mcp_tools_cached()

        assert second == {"main": ["tool-from-good"]}
        assert len(calls) == 1

    def test_still_failing_server_is_retried_again_later(self, monkeypatch):
        behavior = {"good": "ok", "flaky": "fail"}
        calls = self._install(monkeypatch, behavior)
        monkeypatch.setattr(agent_module, "_MCP_FAILED_SERVER_RETRY_SECONDS", 0.0)

        agent_module._load_mcp_tools_cached()
        agent_module._load_mcp_tools_cached()
        third = agent_module._load_mcp_tools_cached()

        assert third == {"main": ["tool-from-good"]}
        assert [sorted(c) for c in calls] == [
            ["flaky", "good"],
            ["flaky"],
            ["flaky"],
        ]
        assert agent_module._MCP_TOOLS_CACHE_FAILED == {"flaky"}

    def test_whole_load_failure_is_retried_with_every_server(self, monkeypatch):
        behavior = {"good": "whole-fail", "flaky": "whole-fail"}
        calls = self._install(monkeypatch, behavior)
        monkeypatch.setattr(agent_module, "_MCP_FAILED_SERVER_RETRY_SECONDS", 0.0)

        first = agent_module._load_mcp_tools_cached()
        assert first == {}
        assert agent_module._MCP_TOOLS_CACHE_FAILED == {"good", "flaky"}

        behavior.update({"good": "ok", "flaky": "ok"})
        second = agent_module._load_mcp_tools_cached()

        # Order within the merged list follows the retry's subset order.
        assert sorted(second["main"]) == ["tool-from-flaky", "tool-from-good"]
        assert [sorted(c) for c in calls] == [["flaky", "good"], ["flaky", "good"]]

    def test_recovery_bumps_the_cache_generation(self, monkeypatch):
        behavior = {"good": "ok", "flaky": "fail"}
        self._install(monkeypatch, behavior)
        monkeypatch.setattr(agent_module, "_MCP_FAILED_SERVER_RETRY_SECONDS", 0.0)

        agent_module._load_mcp_tools_cached()
        generation = agent_module._mcp_tools_cache_generation()

        behavior["flaky"] = "ok"
        agent_module._load_mcp_tools_cached()

        assert agent_module._mcp_tools_cache_generation() > generation

    def test_config_change_reloads_instead_of_retrying(self, monkeypatch):
        behavior = {"good": "ok", "flaky": "fail"}
        calls = self._install(monkeypatch, behavior)
        monkeypatch.setattr(agent_module, "_MCP_FAILED_SERVER_RETRY_SECONDS", 0.0)

        agent_module._load_mcp_tools_cached()
        assert agent_module._MCP_TOOLS_CACHE_FAILED == {"flaky"}

        new_config = {"solo": {"transport": "stdio", "command": "demo-solo"}}
        monkeypatch.setattr(
            "EvoScientist.mcp.client.load_mcp_config", lambda: dict(new_config)
        )

        def fake_load_mcp_tools(config=None, on_progress=None, **_kwargs):
            calls.append(dict(config or {}))
            return {"main": ["tool-from-solo"]}

        monkeypatch.setattr("EvoScientist.mcp.load_mcp_tools", fake_load_mcp_tools)

        result = agent_module._load_mcp_tools_cached()

        assert result == {"main": ["tool-from-solo"]}
        assert calls[-1] == new_config
        assert agent_module._MCP_TOOLS_CACHE_FAILED == set()
