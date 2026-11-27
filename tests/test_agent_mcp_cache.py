"""Tests for MCP tool caching in EvoScientist.EvoScientist."""

from __future__ import annotations

import EvoScientist.EvoScientist as agent_module


def _reset_mcp_cache() -> None:
    agent_module._MCP_TOOLS_CACHE_KEY = None
    agent_module._MCP_TOOLS_CACHE_VALUE = None
    agent_module._MCP_RETRY_AT = 0.0
    agent_module._MCP_LOAD_EPOCH = 0


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


class TestFailedMcpLoadRetry:
    def setup_method(self) -> None:
        _reset_mcp_cache()

    @staticmethod
    def _config(monkeypatch) -> None:
        monkeypatch.setattr(
            "EvoScientist.mcp.client.load_mcp_config",
            lambda: {"srv": {"transport": "stdio", "command": "demo"}},
        )

    def test_a_failed_load_is_retried_after_the_backoff(self, monkeypatch):
        self._config(monkeypatch)
        now = [0.0]
        monkeypatch.setattr(agent_module, "_mcp_now", lambda: now[0])
        calls = []

        def fake_load(config=None, on_progress=None, **_kwargs):
            calls.append(1)
            if on_progress is not None:
                on_progress("error", "srv", "down")
            return {}

        monkeypatch.setattr("EvoScientist.mcp.load_mcp_tools", fake_load)

        assert agent_module._load_mcp_tools_cached() == {}
        assert len(calls) == 1
        # Inside the backoff the failure is replayed, not retried on every
        # graph build.
        assert agent_module._load_mcp_tools_cached() == {}
        assert len(calls) == 1
        now[0] = agent_module._MCP_RETRY_AFTER_SECONDS + 1
        assert agent_module._load_mcp_tools_cached() == {}
        assert len(calls) == 2

    def test_recovery_bumps_the_graph_epoch_and_stops_retrying(self, monkeypatch):
        self._config(monkeypatch)
        now = [0.0]
        monkeypatch.setattr(agent_module, "_mcp_now", lambda: now[0])
        state = {"failed": True}
        calls = []

        def fake_load(config=None, on_progress=None, **_kwargs):
            calls.append(1)
            if on_progress is not None:
                if state["failed"]:
                    on_progress("error", "srv", "down")
                else:
                    on_progress("success", "srv", "1")
            return {} if state["failed"] else {"main": ["tool"]}

        monkeypatch.setattr("EvoScientist.mcp.load_mcp_tools", fake_load)

        assert agent_module._load_mcp_tools_cached() == {}
        assert agent_module.mcp_tools_epoch() == 0
        state["failed"] = False  # the server recovers while we are away
        now[0] = agent_module._MCP_RETRY_AFTER_SECONDS + 1
        assert agent_module._load_mcp_tools_cached() == {"main": ["tool"]}
        assert agent_module.mcp_tools_epoch() == 1

        now[0] += 100
        assert agent_module._load_mcp_tools_cached() == {"main": ["tool"]}
        assert len(calls) == 2  # the successful load stays cached
        assert agent_module.mcp_tools_epoch() == 1
