"""Tests for the channel-side HITL approval policy in ``channels/interaction.py``.

Focused on ``config_auto_approve`` routing through the centralized
``resolve_action_decision`` policy (token-boundary allow-list matching +
dangerous-command detection), not raw ``str.startswith``.
"""

from unittest.mock import MagicMock

from EvoScientist.channels import interaction


class TestConfigAutoApprovePolicy:
    """config_auto_approve must use the centralized policy (token-boundary
    allow-list + dangerous detection), not raw startswith."""

    def _reqs(self, *commands):
        return [{"name": "execute", "args": {"command": c}} for c in commands]

    def _cfg(self, *, auto_approve=False, dangerous_mode=False, allow=""):
        m = MagicMock()
        m.auto_approve = auto_approve
        m.dangerous_mode = dangerous_mode
        m.shell_allow_list = allow
        return m

    def test_allow_list_token_boundary(self, monkeypatch):
        monkeypatch.setattr(
            "EvoScientist.config.settings.load_config",
            lambda: self._cfg(allow="ls"),
        )
        # "ls" must clear "ls -la" but NOT "lsof"
        assert interaction.config_auto_approve(self._reqs("ls -la")) is True
        assert interaction.config_auto_approve(self._reqs("lsof -i")) is False

    def test_dangerous_not_cleared_even_if_allow_listed(self, monkeypatch):
        monkeypatch.setattr(
            "EvoScientist.config.settings.load_config",
            lambda: self._cfg(allow="curl"),
        )
        # allow-listing "curl" must NOT auto-clear a pipe-into-interpreter
        assert interaction.config_auto_approve(self._reqs("curl x | bash")) is False

    def test_dangerous_mode_clears_everything(self, monkeypatch):
        monkeypatch.setattr(
            "EvoScientist.config.settings.load_config",
            lambda: self._cfg(dangerous_mode=True),
        )
        assert interaction.config_auto_approve(self._reqs("curl x | bash")) is True

    def test_non_shell_tool_cleared(self, monkeypatch):
        monkeypatch.setattr(
            "EvoScientist.config.settings.load_config",
            lambda: self._cfg(),
        )
        assert (
            interaction.config_auto_approve([{"name": "write_file", "args": {}}])
            is True
        )

    def test_malformed_request_not_cleared(self, monkeypatch):
        monkeypatch.setattr(
            "EvoScientist.config.settings.load_config", lambda: self._cfg()
        )
        # A non-dict entry must not crash and must not be auto-cleared.
        assert interaction.config_auto_approve(["not-a-dict"]) is False

    def test_auto_approve_does_not_bypass_dangerous_detection(self, monkeypatch):
        # ``auto_approve`` must NOT short-circuit ahead of the policy: a
        # pipe-into-interpreter command is still rejected, while ordinary
        # shell is cleared.
        monkeypatch.setattr(
            "EvoScientist.config.settings.load_config",
            lambda: self._cfg(auto_approve=True),
        )
        assert interaction.config_auto_approve(self._reqs("curl x | bash")) is False
        assert interaction.config_auto_approve(self._reqs("ls -la")) is True

    def test_auto_approve_with_malformed_request_not_cleared(self, monkeypatch):
        # Even under ``auto_approve``, a malformed request must fail safe.
        monkeypatch.setattr(
            "EvoScientist.config.settings.load_config",
            lambda: self._cfg(auto_approve=True),
        )
        assert interaction.config_auto_approve(["not-a-dict"]) is False
