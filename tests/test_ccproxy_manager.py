"""Tests for ccproxy_manager module."""

import os
from unittest.mock import MagicMock, patch

import pytest

from EvoScientist.ccproxy_manager import (
    _CCPROXY_AUTH_TIMEOUT_SECONDS,
    _CCPROXY_HEALTH_TIMEOUT_SECONDS,
    check_ccproxy_auth,
    check_ccproxy_ready,
    is_ccproxy_available,
    is_ccproxy_running,
    setup_ccproxy_env,
    spawn_ccproxy,
    stop_ccproxy,
    wait_for_ccproxy,
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
    def test_probe_bypasses_proxies(self, mock_get):
        """A system or environment proxy would swallow the loopback probe."""
        mock_get.return_value = MagicMock(status_code=200)
        is_ccproxy_running(8000)
        assert mock_get.call_args[1]["trust_env"] is False

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
    def test_listener_without_http_means_not_running(self, mock_get):
        import httpx

        mock_get.side_effect = httpx.RemoteProtocolError("Server disconnected")
        assert is_ccproxy_running(8000) is False

    @patch("httpx.get")
    def test_non_200_means_not_running(self, mock_get):
        mock_get.return_value = MagicMock(status_code=404)
        assert is_ccproxy_running(8000) is False


# =============================================================================
# spawn_ccproxy / wait_for_ccproxy
# =============================================================================


def _start(port):
    proc = spawn_ccproxy(port)
    wait_for_ccproxy(proc, port)
    return proc


class TestSpawnAndWaitForCcproxy:
    @patch("EvoScientist.ccproxy_manager.logger.warning")
    @patch("EvoScientist.ccproxy_manager.is_ccproxy_running")
    @patch("subprocess.Popen")
    def test_success(self, mock_popen, mock_running, mock_warning):
        proc = MagicMock()
        proc.poll.return_value = None
        mock_popen.return_value = proc
        # First call: not running, second call: running
        mock_running.side_effect = [True]

        result = _start(8000)
        assert result is proc
        mock_warning.assert_called_once_with(
            "Starting ccproxy on port %d; first startup may take up to %d seconds",
            8000,
            _CCPROXY_HEALTH_TIMEOUT_SECONDS,
        )

    @patch("psutil.Process")
    @patch("EvoScientist.ccproxy_manager.end_tree")
    @patch("EvoScientist.ccproxy_manager.is_ccproxy_running", return_value=False)
    @patch("EvoScientist.ccproxy_manager.time")
    @patch("subprocess.Popen")
    def test_timeout(
        self, mock_popen, mock_time, mock_running, mock_end_tree, mock_process
    ):
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
            _start(8000)
        # With what it started: on Windows ccproxy.exe is a launcher.
        mock_end_tree.assert_called_once()

    @pytest.mark.parametrize("raised", [KeyboardInterrupt, OSError])
    @patch("psutil.Process")
    @patch("EvoScientist.ccproxy_manager.end_tree")
    @patch("EvoScientist.ccproxy_manager.is_ccproxy_running")
    @patch("subprocess.Popen")
    def test_interrupted_wait_stops_the_process(
        self, mock_popen, mock_running, mock_end_tree, mock_process, raised
    ):
        proc = MagicMock()
        proc.poll.return_value = None
        mock_popen.return_value = proc
        mock_running.side_effect = raised

        with pytest.raises(raised):
            _start(8000)
        mock_end_tree.assert_called_once()
        proc.wait.assert_called_once()

    @patch("EvoScientist.ccproxy_manager.is_ccproxy_running", return_value=False)
    @patch("subprocess.Popen")
    def test_exit_before_answering(self, mock_popen, mock_running):
        """It may die late in the wait, not only at once."""
        proc = MagicMock()
        proc.poll.return_value = 1
        proc.returncode = 1
        mock_popen.return_value = proc

        with pytest.raises(RuntimeError, match="before it answered"):
            _start(8000)

    @patch("subprocess.Popen", side_effect=FileNotFoundError)
    def test_missing_binary(self, mock_popen):
        with pytest.raises(FileNotFoundError):
            _start(8000)

    @patch("EvoScientist.ccproxy_manager.is_ccproxy_running")
    @patch("subprocess.Popen")
    def test_start_passes_no_codex_overrides(self, mock_popen, mock_running):
        proc = MagicMock()
        proc.poll.return_value = None
        mock_popen.return_value = proc
        mock_running.side_effect = [True]

        _start(8000)

        env = mock_popen.call_args[1].get("env") or {}
        assert not [key for key in env if key.startswith("PLUGINS__CODEX__")]


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
# check_ccproxy_ready
# =============================================================================


class TestCheckCcproxyReady:
    @patch("EvoScientist.ccproxy_manager.is_ccproxy_available", return_value=False)
    def test_check_ccproxy_ready_raises_without_binary(self, mock_avail):
        with pytest.raises(RuntimeError, match="not found"):
            check_ccproxy_ready()

    @patch(
        "EvoScientist.ccproxy_manager.check_ccproxy_auth",
        return_value=(False, "expired"),
    )
    @patch("EvoScientist.ccproxy_manager.is_ccproxy_available", return_value=True)
    def test_check_ccproxy_ready_raises_without_sign_in(self, mock_avail, mock_auth):
        with pytest.raises(RuntimeError, match="not authenticated"):
            check_ccproxy_ready()
        mock_auth.assert_called_once_with("claude_api")

    @patch("EvoScientist.ccproxy_manager._is_editable_install", return_value=True)
    @patch("EvoScientist.ccproxy_manager.is_ccproxy_available", return_value=False)
    def test_check_ccproxy_ready_names_editable_install(self, mock_avail, mock_edit):
        with pytest.raises(RuntimeError) as exc_info:
            check_ccproxy_ready()
        msg = str(exc_info.value)
        assert "uv sync --extra oauth" in msg
        assert "pip install -e '.[oauth]'" in msg

    @patch("EvoScientist.ccproxy_manager._is_editable_install", return_value=False)
    @patch("EvoScientist.ccproxy_manager.is_ccproxy_available", return_value=False)
    def test_check_ccproxy_ready_names_package_install(self, mock_avail, mock_edit):
        with pytest.raises(RuntimeError) as exc_info:
            check_ccproxy_ready()
        assert "pip install 'evoscientist[oauth]'" in str(exc_info.value)


def test_stop_recorded_ccproxy_leaves_other_process(monkeypatch, tmp_path):
    """A recorded pid reused by another program is not stopped."""
    import psutil

    import EvoScientist.ccproxy_manager as ccproxy_mod
    from EvoScientist import paths

    class _OtherProgram:
        pid = 4242

        def cmdline(self):
            return ["python", "-c", "import time; time.sleep(60)"]

        def terminate(self):
            raise AssertionError("another program was stopped")

        kill = terminate

    monkeypatch.setattr(psutil, "Process", lambda pid: _OtherProgram())
    monkeypatch.setattr(paths, "SERVER_HOME", tmp_path)
    (tmp_path / "ccproxy.pid").write_text("4242")

    assert ccproxy_mod.stop_recorded_ccproxy() is None
    assert not (tmp_path / "ccproxy.pid").exists()


@pytest.mark.parametrize(
    ("anthropic", "openai", "pointed"),
    [
        ("oauth", "api_key", True),
        ("api_key", "api_key", False),
        ("api_key", "oauth", False),
    ],
)
def test_use_server_ccproxy_only_with_anthropic_oauth(
    monkeypatch, anthropic, openai, pointed
):
    """The server starts ccproxy; a client only points at it. OpenAI's OAuth
    signs in to ChatGPT directly."""
    from types import SimpleNamespace

    import EvoScientist.ccproxy_manager as ccproxy_mod

    ports: list[int] = []
    monkeypatch.setattr(ccproxy_mod, "setup_ccproxy_env", ports.append)
    config = SimpleNamespace(
        anthropic_auth_mode=anthropic, openai_auth_mode=openai, ccproxy_port=8000
    )

    ccproxy_mod.use_server_ccproxy(config)

    assert ports == ([8000] if pointed else [])
