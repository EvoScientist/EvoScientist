"""Tests for ccproxy_manager module."""

import os
from unittest.mock import MagicMock, patch

import pytest

from EvoScientist.ccproxy_manager import (
    is_ccproxy_available,
    check_ccproxy_auth,
    is_ccproxy_running,
    start_ccproxy,
    stop_ccproxy,
    ensure_ccproxy,
    setup_ccproxy_env,
    setup_codex_env,
    maybe_start_ccproxy,
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

    @patch("subprocess.run")
    def test_valid_auth_codex(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="Authenticated", stderr=""
        )
        valid, msg = check_ccproxy_auth("codex")
        assert valid is True
        cmd = mock_run.call_args[0][0]
        assert cmd[1:] == ["auth", "status", "codex"]

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
    @patch("EvoScientist.ccproxy_manager.is_ccproxy_running")
    @patch("subprocess.Popen")
    def test_success(self, mock_popen, mock_running):
        proc = MagicMock()
        proc.poll.return_value = None
        mock_popen.return_value = proc
        # First call: not running, second call: running
        mock_running.side_effect = [True]

        result = start_ccproxy(8000)
        assert result is proc

    @patch("EvoScientist.ccproxy_manager.is_ccproxy_running", return_value=False)
    @patch("EvoScientist.ccproxy_manager.time")
    @patch("subprocess.Popen")
    def test_timeout(self, mock_popen, mock_time, mock_running):
        proc = MagicMock()
        proc.poll.return_value = None
        mock_popen.return_value = proc
        # Simulate time passing beyond deadline
        mock_time.monotonic.side_effect = [0, 0, 11]
        mock_time.sleep = MagicMock()

        with pytest.raises(RuntimeError, match="did not become healthy"):
            start_ccproxy(8000)

    @patch("subprocess.Popen", side_effect=FileNotFoundError)
    def test_missing_binary(self, mock_popen):
        with pytest.raises(FileNotFoundError):
            start_ccproxy(8000)


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
# setup_codex_env
# =============================================================================


class TestSetupCodexEnv:
    def test_sets_vars(self, monkeypatch):
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        setup_codex_env(8000)

        assert os.environ["OPENAI_BASE_URL"] == "http://127.0.0.1:8000/codex/v1"
        assert os.environ["OPENAI_API_KEY"] == "ccproxy-oauth"

    def test_overrides_existing(self, monkeypatch):
        """Force-sets vars even if already configured (oauth takes priority)."""
        monkeypatch.setenv("OPENAI_BASE_URL", "http://custom:9999")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key")

        setup_codex_env(8000)

        assert os.environ["OPENAI_BASE_URL"] == "http://127.0.0.1:8000/codex/v1"
        assert os.environ["OPENAI_API_KEY"] == "ccproxy-oauth"


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

    @patch("EvoScientist.ccproxy_manager.setup_codex_env")
    @patch("EvoScientist.ccproxy_manager.ensure_ccproxy")
    @patch("EvoScientist.ccproxy_manager.check_ccproxy_auth", return_value=(True, "OK"))
    @patch("EvoScientist.ccproxy_manager.is_ccproxy_available", return_value=True)
    def test_openai_oauth_mode_starts(
        self, mock_avail, mock_auth, mock_ensure, mock_env
    ):
        proc = MagicMock()
        mock_ensure.return_value = proc
        config = MagicMock()
        config.anthropic_auth_mode = "api_key"
        config.openai_auth_mode = "oauth"
        config.ccproxy_port = 8000

        result = maybe_start_ccproxy(config)
        assert result is proc
        mock_auth.assert_called_once_with("codex")
        mock_env.assert_called_once()

    @patch("EvoScientist.ccproxy_manager.setup_codex_env")
    @patch("EvoScientist.ccproxy_manager.setup_ccproxy_env")
    @patch("EvoScientist.ccproxy_manager.ensure_ccproxy")
    @patch("EvoScientist.ccproxy_manager.check_ccproxy_auth", return_value=(True, "OK"))
    @patch("EvoScientist.ccproxy_manager.is_ccproxy_available", return_value=True)
    def test_both_oauth_starts_both(
        self, mock_avail, mock_auth, mock_ensure, mock_anthropic_env, mock_codex_env
    ):
        proc = MagicMock()
        mock_ensure.return_value = proc
        config = MagicMock()
        config.anthropic_auth_mode = "oauth"
        config.openai_auth_mode = "oauth"
        config.ccproxy_port = 8000

        result = maybe_start_ccproxy(config)
        assert result is proc
        # Auth checked for both providers
        assert mock_auth.call_count == 2
        mock_anthropic_env.assert_called_once()
        mock_codex_env.assert_called_once()

    @patch("EvoScientist.ccproxy_manager.is_ccproxy_available", return_value=False)
    def test_openai_oauth_raises_no_binary(self, mock_avail):
        config = MagicMock()
        config.anthropic_auth_mode = "api_key"
        config.openai_auth_mode = "oauth"
        with pytest.raises(RuntimeError, match="not found"):
            maybe_start_ccproxy(config)

    @patch(
        "EvoScientist.ccproxy_manager.check_ccproxy_auth",
        return_value=(False, "expired"),
    )
    @patch("EvoScientist.ccproxy_manager.is_ccproxy_available", return_value=True)
    def test_openai_oauth_raises_no_auth(self, mock_avail, mock_auth):
        config = MagicMock()
        config.anthropic_auth_mode = "api_key"
        config.openai_auth_mode = "oauth"
        with pytest.raises(RuntimeError, match="Codex OAuth not authenticated"):
            maybe_start_ccproxy(config)

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
