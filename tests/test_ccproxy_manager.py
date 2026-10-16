"""Tests for ccproxy_manager module."""

import os
from unittest.mock import MagicMock, patch

import pytest

from EvoScientist.ccproxy_manager import (
    _CCPROXY_AUTH_TIMEOUT_SECONDS,
    _CCPROXY_HEALTH_TIMEOUT_SECONDS,
    check_ccproxy_auth,
    ensure_ccproxy,
    is_ccproxy_available,
    is_ccproxy_running,
    maybe_start_ccproxy,
    setup_ccproxy_env,
    start_ccproxy,
    stop_ccproxy,
)

# =============================================================================
# is_ccproxy_available
# =============================================================================


class TestIsCcproxyAvailable:
    @patch("shutil.which", return_value="/usr/local/bin/ccproxy")
    def test_found(self, mock_which):
        assert is_ccproxy_available() is True
        mock_which.assert_called_once_with("ccproxy")

    @patch("os.access", return_value=False)
    @patch("os.path.isfile", return_value=False)
    @patch("shutil.which", return_value=None)
    def test_not_found(self, mock_which, mock_isfile, mock_access):
        assert is_ccproxy_available() is False


# =============================================================================
# check_ccproxy_auth
# =============================================================================


class TestCheckCcproxyAuth:
    @patch("subprocess.run")
    def test_valid_auth(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="Authenticated as user@example.com", stderr=""
        )
        valid, msg = check_ccproxy_auth()
        assert valid is True
        assert "Authenticated" in msg
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[1:] == ["auth", "status", "claude_api"]
        # ccproxy CLI cold start takes ~10s; timeout must leave headroom
        assert mock_run.call_args[1]["timeout"] == _CCPROXY_AUTH_TIMEOUT_SECONDS

    @patch("subprocess.run")
    def test_invalid_auth(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="Not authenticated"
        )
        valid, msg = check_ccproxy_auth()
        assert valid is False
        assert "Not authenticated" in msg

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_missing_binary(self, mock_run):
        valid, msg = check_ccproxy_auth()
        assert valid is False
        assert "not found" in msg


# =============================================================================
# is_ccproxy_running
# =============================================================================


class TestIsCcproxyRunning:
    @patch("httpx.get")
    def test_running(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        assert is_ccproxy_running(8000) is True

    @patch("httpx.get")
    def test_uses_health_live_endpoint(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        is_ccproxy_running(8000)
        url = mock_get.call_args[0][0]
        assert url == "http://127.0.0.1:8000/health/live"

    @patch("httpx.get")
    def test_uses_custom_port(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        is_ccproxy_running(7777)
        url = mock_get.call_args[0][0]
        assert url == "http://127.0.0.1:7777/health/live"

    @patch("httpx.get")
    def test_not_running(self, mock_get):
        import httpx

        mock_get.side_effect = httpx.ConnectError("Connection refused")
        assert is_ccproxy_running(8000) is False

    @patch("httpx.get")
    def test_non_200_means_not_running(self, mock_get):
        mock_get.return_value = MagicMock(status_code=404)
        assert is_ccproxy_running(8000) is False


# =============================================================================
# start_ccproxy
# =============================================================================


class TestStartCcproxy:
    @patch("EvoScientist.ccproxy_manager.logger.warning")
    @patch("EvoScientist.ccproxy_manager.is_ccproxy_running")
    @patch("subprocess.Popen")
    def test_success(self, mock_popen, mock_running, mock_warning):
        proc = MagicMock()
        proc.poll.return_value = None
        mock_popen.return_value = proc
        # First call: not running, second call: running
        mock_running.side_effect = [True]

        result = start_ccproxy(8000)
        assert result is proc
        mock_warning.assert_called_once_with(
            "Starting ccproxy on port %d; first startup may take up to %d seconds",
            8000,
            _CCPROXY_HEALTH_TIMEOUT_SECONDS,
        )

    @patch("EvoScientist.ccproxy_manager.is_ccproxy_running", return_value=False)
    @patch("EvoScientist.ccproxy_manager.time")
    @patch("subprocess.Popen")
    def test_timeout(self, mock_popen, mock_time, mock_running):
        proc = MagicMock()
        proc.poll.return_value = None
        mock_popen.return_value = proc
        # Simulate time passing beyond deadline
        mock_time.monotonic.side_effect = [
            0,
            0,
            _CCPROXY_HEALTH_TIMEOUT_SECONDS + 1,
        ]
        mock_time.sleep = MagicMock()

        with pytest.raises(RuntimeError, match="did not become healthy"):
            start_ccproxy(8000)

    @patch("subprocess.Popen", side_effect=FileNotFoundError)
    def test_missing_binary(self, mock_popen):
        with pytest.raises(FileNotFoundError):
            start_ccproxy(8000)

    @patch("EvoScientist.ccproxy_manager.is_ccproxy_running")
    @patch("subprocess.Popen")
    def test_start_passes_no_codex_overrides(self, mock_popen, mock_running):
        proc = MagicMock()
        proc.poll.return_value = None
        mock_popen.return_value = proc
        mock_running.side_effect = [True]

        start_ccproxy(8000)

        env = mock_popen.call_args[1].get("env") or {}
        assert not [key for key in env if key.startswith("PLUGINS__CODEX__")]


# =============================================================================
# ensure_ccproxy
# =============================================================================


class TestEnsureCcproxy:
    @patch("EvoScientist.ccproxy_manager.is_ccproxy_running", return_value=True)
    def test_already_running(self, mock_running):
        result = ensure_ccproxy(8000)
        assert result is None

    @patch("EvoScientist.ccproxy_manager.start_ccproxy")
    @patch("EvoScientist.ccproxy_manager.is_ccproxy_running", return_value=False)
    def test_needs_start(self, mock_running, mock_start):
        proc = MagicMock()
        mock_start.return_value = proc
        result = ensure_ccproxy(8000)
        assert result is proc
        mock_start.assert_called_once_with(8000)


# =============================================================================
# setup_ccproxy_env
# =============================================================================


class TestSetupCcproxyEnv:
    def test_sets_vars(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        setup_ccproxy_env(8000)

        assert os.environ["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8000/claude"
        assert os.environ["ANTHROPIC_API_KEY"] == "ccproxy-oauth"

    def test_overrides_existing(self, monkeypatch):
        """Force-sets vars even if already configured (oauth takes priority)."""
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://custom:9999")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-real-key")

        setup_ccproxy_env(8000)

        assert os.environ["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8000/claude"
        assert os.environ["ANTHROPIC_API_KEY"] == "ccproxy-oauth"


# =============================================================================
# stop_ccproxy
# =============================================================================


class TestStopCcproxy:
    def test_none_is_noop(self):
        stop_ccproxy(None)  # Should not raise

    def test_terminates_process(self):
        proc = MagicMock()
        stop_ccproxy(proc)
        proc.terminate.assert_called_once()
        proc.wait.assert_called_once_with(timeout=5)


# =============================================================================
# maybe_start_ccproxy
# =============================================================================


class TestMaybeStartCcproxy:
    def test_api_key_mode_noop(self):
        config = MagicMock()
        config.anthropic_auth_mode = "api_key"
        config.openai_auth_mode = "api_key"
        assert maybe_start_ccproxy(config) is None

    @patch("EvoScientist.ccproxy_manager.setup_ccproxy_env")
    @patch("EvoScientist.ccproxy_manager.ensure_ccproxy")
    @patch("EvoScientist.ccproxy_manager.check_ccproxy_auth", return_value=(True, "OK"))
    @patch("EvoScientist.ccproxy_manager.is_ccproxy_available", return_value=True)
    def test_oauth_mode_starts(self, mock_avail, mock_auth, mock_ensure, mock_env):
        proc = MagicMock()
        mock_ensure.return_value = proc
        config = MagicMock()
        config.anthropic_auth_mode = "oauth"
        config.openai_auth_mode = "api_key"
        config.ccproxy_port = 8000

        result = maybe_start_ccproxy(config)
        assert result is proc
        mock_env.assert_called_once()

    @patch("EvoScientist.ccproxy_manager.is_ccproxy_available", return_value=False)
    def test_oauth_mode_raises_no_binary(self, mock_avail):
        config = MagicMock()
        config.anthropic_auth_mode = "oauth"
        config.openai_auth_mode = "api_key"
        with pytest.raises(RuntimeError, match="not found"):
            maybe_start_ccproxy(config)

    @patch(
        "EvoScientist.ccproxy_manager.check_ccproxy_auth",
        return_value=(False, "expired"),
    )
    @patch("EvoScientist.ccproxy_manager.is_ccproxy_available", return_value=True)
    def test_oauth_mode_raises_no_auth(self, mock_avail, mock_auth):
        config = MagicMock()
        config.anthropic_auth_mode = "oauth"
        config.openai_auth_mode = "api_key"
        with pytest.raises(RuntimeError, match="not authenticated"):
            maybe_start_ccproxy(config)

    @patch("EvoScientist.ccproxy_manager.ensure_ccproxy")
    @patch("EvoScientist.ccproxy_manager.is_ccproxy_available")
    def test_openai_oauth_alone_needs_no_proxy(self, mock_avail, mock_ensure):
        config = MagicMock()
        config.anthropic_auth_mode = "api_key"
        config.openai_auth_mode = "oauth"

        assert maybe_start_ccproxy(config) is None
        mock_avail.assert_not_called()
        mock_ensure.assert_not_called()

    @patch("EvoScientist.ccproxy_manager.setup_ccproxy_env")
    @patch("EvoScientist.ccproxy_manager.ensure_ccproxy")
    @patch("EvoScientist.ccproxy_manager.check_ccproxy_auth", return_value=(True, "OK"))
    @patch("EvoScientist.ccproxy_manager.is_ccproxy_available", return_value=True)
    def test_both_oauth_starts_the_proxy_for_anthropic_only(
        self, mock_avail, mock_auth, mock_ensure, mock_env, monkeypatch
    ):
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        proc = MagicMock()
        mock_ensure.return_value = proc
        config = MagicMock()
        config.anthropic_auth_mode = "oauth"
        config.openai_auth_mode = "oauth"
        config.ccproxy_port = 8000

        assert maybe_start_ccproxy(config) is proc
        mock_auth.assert_called_once_with("claude_api")
        mock_env.assert_called_once()
        assert "OPENAI_BASE_URL" not in os.environ
        assert "OPENAI_API_KEY" not in os.environ

    @patch("EvoScientist.ccproxy_manager._is_editable_install", return_value=True)
    @patch("EvoScientist.ccproxy_manager.is_ccproxy_available", return_value=False)
    def test_oauth_raises_no_binary_editable(self, mock_avail, mock_edit):
        config = MagicMock()
        config.anthropic_auth_mode = "oauth"
        config.openai_auth_mode = "api_key"
        with pytest.raises(RuntimeError) as exc_info:
            maybe_start_ccproxy(config)
        msg = str(exc_info.value)
        assert "ccproxy is required for OAuth mode but not found" in msg
        assert "uv sync --extra oauth" in msg
        assert "pip install -e '.[oauth]'" in msg

    @patch("EvoScientist.ccproxy_manager._is_editable_install", return_value=False)
    @patch("EvoScientist.ccproxy_manager.is_ccproxy_available", return_value=False)
    def test_oauth_raises_no_binary_pip(self, mock_avail, mock_edit):
        config = MagicMock()
        config.anthropic_auth_mode = "oauth"
        config.openai_auth_mode = "api_key"
        with pytest.raises(RuntimeError) as exc_info:
            maybe_start_ccproxy(config)
        msg = str(exc_info.value)
        assert "ccproxy is required for OAuth mode but not found" in msg
        assert "pip install 'evoscientist[oauth]'" in msg

    @patch("EvoScientist.ccproxy_manager.setup_ccproxy_env")
    @patch("EvoScientist.ccproxy_manager.ensure_ccproxy")
    @patch("EvoScientist.ccproxy_manager.check_ccproxy_auth", return_value=(True, "OK"))
    @patch("EvoScientist.ccproxy_manager.is_ccproxy_available", return_value=True)
    def test_uses_config_ccproxy_port(
        self, mock_avail, mock_auth, mock_ensure, mock_env
    ):
        """maybe_start_ccproxy passes config.ccproxy_port to ensure_ccproxy."""
        proc = MagicMock()
        mock_ensure.return_value = proc
        config = MagicMock()
        config.anthropic_auth_mode = "oauth"
        config.openai_auth_mode = "api_key"
        config.ccproxy_port = 7777

        maybe_start_ccproxy(config)
        mock_ensure.assert_called_once_with(7777)

    @patch("EvoScientist.ccproxy_manager.check_ccproxy_auth", return_value=(True, "OK"))
    @patch("EvoScientist.ccproxy_manager.is_ccproxy_available", return_value=True)
    def test_invalid_port_raises(self, mock_avail, mock_auth):
        """maybe_start_ccproxy raises ValueError for out-of-range port."""
        config = MagicMock()
        config.anthropic_auth_mode = "oauth"
        config.openai_auth_mode = "api_key"
        config.ccproxy_port = 0

        with pytest.raises(ValueError, match="Invalid ccproxy port"):
            maybe_start_ccproxy(config)

    @patch("EvoScientist.ccproxy_manager.check_ccproxy_auth", return_value=(True, "OK"))
    @patch("EvoScientist.ccproxy_manager.is_ccproxy_available", return_value=True)
    def test_port_too_large_raises(self, mock_avail, mock_auth):
        config = MagicMock()
        config.anthropic_auth_mode = "oauth"
        config.openai_auth_mode = "api_key"
        config.ccproxy_port = 99999

        with pytest.raises(ValueError, match="Invalid ccproxy port"):
            maybe_start_ccproxy(config)
