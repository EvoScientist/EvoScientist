"""Happy-path tests for langgraph_dev.manager.

Mocks httpx, psutil, subprocess.Popen, and module-level state so the tests
run on CI without requiring the langgraph CLI to be installed or any port
to be available.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from EvoScientist.config.settings import EvoScientistConfig
from EvoScientist.langgraph_dev import manager


@pytest.fixture(autouse=True)
def reset_module_state():
    """Reset manager module globals before each test for isolation."""
    manager._PROCESS = None
    manager._PROCESS_WORKSPACE = None
    manager._ASYNC_SUBAGENTS_AVAILABLE = False
    yield
    manager._PROCESS = None
    manager._PROCESS_WORKSPACE = None
    manager._ASYNC_SUBAGENTS_AVAILABLE = False


# =============================================================================
# is_langgraph_dev_running
# =============================================================================


class TestIsLanggraphDevRunning:
    @patch("EvoScientist.langgraph_dev.manager.httpx.get")
    def test_returns_false_on_connect_error(self, mock_get):
        mock_get.side_effect = httpx.ConnectError("refused")
        assert manager.is_langgraph_dev_running(port=6174) is False

    @patch("EvoScientist.langgraph_dev.manager.httpx.get")
    def test_returns_false_on_timeout(self, mock_get):
        mock_get.side_effect = httpx.TimeoutException("slow")
        assert manager.is_langgraph_dev_running(port=6174) is False

    @patch("EvoScientist.langgraph_dev.manager.httpx.get")
    def test_returns_true_on_200(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        assert manager.is_langgraph_dev_running(port=6174) is True
        # Verify it probed /ok at the configured port.
        called_url = mock_get.call_args[0][0]
        assert called_url == "http://localhost:6174/ok"

    @patch("EvoScientist.langgraph_dev.manager.httpx.get")
    def test_returns_false_on_non_200(self, mock_get):
        mock_get.return_value = MagicMock(status_code=503)
        assert manager.is_langgraph_dev_running(port=6174) is False


# =============================================================================
# _list_pids_on_port
# =============================================================================


class TestListPidsOnPort:
    @patch("EvoScientist.langgraph_dev.manager.psutil.net_connections")
    def test_empty_when_no_connections(self, mock_net):
        mock_net.return_value = []
        assert manager._list_pids_on_port(6174) == []

    @patch("EvoScientist.langgraph_dev.manager.psutil.net_connections")
    def test_returns_pid_for_matching_port(self, mock_net):
        mock_net.return_value = [
            SimpleNamespace(laddr=SimpleNamespace(port=6174), pid=12345),
            SimpleNamespace(laddr=SimpleNamespace(port=8080), pid=99999),
        ]
        result = manager._list_pids_on_port(6174)
        assert result == [12345]

    @patch("EvoScientist.langgraph_dev.manager.psutil.net_connections")
    def test_filters_none_pid(self, mock_net):
        mock_net.return_value = [
            SimpleNamespace(laddr=SimpleNamespace(port=6174), pid=None),
            SimpleNamespace(laddr=SimpleNamespace(port=6174), pid=12345),
        ]
        result = manager._list_pids_on_port(6174)
        assert result == [12345]

    def test_returns_empty_on_access_denied(self):
        with patch.object(
            manager.psutil,
            "net_connections",
            side_effect=manager.psutil.AccessDenied(),
        ):
            assert manager._list_pids_on_port(6174) == []


# =============================================================================
# _kill_owned_stale_process
# =============================================================================


class TestKillOwnedStaleProcess:
    def test_returns_false_if_no_pid_file(self, tmp_path):
        with patch.object(manager, "_PID_FILE", tmp_path / "missing.pid"):
            assert manager._kill_owned_stale_process(6174) is False

    def test_returns_false_if_pid_file_unreadable(self, tmp_path):
        pid_file = tmp_path / "bad.pid"
        pid_file.write_text("not-a-number")
        with patch.object(manager, "_PID_FILE", pid_file):
            assert manager._kill_owned_stale_process(6174) is False

    def test_returns_false_if_pid_not_in_occupiers(self, tmp_path):
        pid_file = tmp_path / "lg.pid"
        pid_file.write_text("12345")
        with (
            patch.object(manager, "_PID_FILE", pid_file),
            patch.object(manager, "_list_pids_on_port", return_value=[99999]),
        ):
            assert manager._kill_owned_stale_process(6174) is False
            # PID file should be left intact — the port is held by someone
            # else, not a stale ours.
            assert pid_file.exists()

    def test_refuses_to_kill_recycled_pid(self, tmp_path):
        """PID matches but cmdline doesn't contain 'langgraph' → don't kill."""
        pid_file = tmp_path / "lg.pid"
        pid_file.write_text("12345")
        fake_proc = MagicMock()
        fake_proc.cmdline.return_value = ["bash", "-c", "echo hi"]
        with (
            patch.object(manager, "_PID_FILE", pid_file),
            patch.object(manager, "_list_pids_on_port", return_value=[12345]),
            patch.object(manager.psutil, "Process", return_value=fake_proc),
        ):
            assert manager._kill_owned_stale_process(6174) is False
            fake_proc.kill.assert_not_called()
            # PID file should be removed — the entry is stale (our process is
            # gone, PID was recycled by an unrelated process).
            assert not pid_file.exists()

    def test_kills_when_cmdline_matches_langgraph(self, tmp_path):
        """Owned PID + cmdline contains 'langgraph' → kill + cleanup PID file."""
        pid_file = tmp_path / "lg.pid"
        pid_file.write_text("12345")
        fake_proc = MagicMock()
        fake_proc.cmdline.return_value = [
            "/usr/bin/python",
            "/usr/bin/langgraph",
            "dev",
        ]
        with (
            patch.object(manager, "_PID_FILE", pid_file),
            patch.object(manager, "_list_pids_on_port", return_value=[12345]),
            patch.object(manager.psutil, "Process", return_value=fake_proc),
        ):
            assert manager._kill_owned_stale_process(6174) is True
            fake_proc.kill.assert_called_once()
            assert not pid_file.exists()

    def test_handles_dead_pid(self, tmp_path):
        """PID file claims a PID but the process is gone → cleanup PID file, no error."""
        pid_file = tmp_path / "lg.pid"
        pid_file.write_text("12345")
        with (
            patch.object(manager, "_PID_FILE", pid_file),
            patch.object(manager, "_list_pids_on_port", return_value=[12345]),
            patch.object(
                manager.psutil,
                "Process",
                side_effect=manager.psutil.NoSuchProcess(12345),
            ),
        ):
            assert manager._kill_owned_stale_process(6174) is False
            assert not pid_file.exists()


# =============================================================================
# ensure_langgraph_dev — high-level orchestration
# =============================================================================


class TestEnsureLanggraphDev:
    def test_starts_when_async_disabled_but_memory_workers_enabled(self, tmp_path):
        """EvoMemory workers can require langgraph dev even without async subagents."""
        cfg = EvoScientistConfig()
        cfg.enable_async_subagents = False
        cfg.memory_workers_enabled = True
        cfg.langgraph_dev_port = 6174
        cfg.langgraph_dev_file_persistence = True
        proc = MagicMock()
        with (
            patch.object(manager, "is_langgraph_dev_running", return_value=False),
            patch.object(manager, "start_langgraph_dev", return_value=proc) as start,
            patch.object(manager, "_FILE_LOCK_PATH", tmp_path / "lg.lock"),
            patch.object(manager, "_PID_DIR", tmp_path / "pids"),
        ):
            result = manager.ensure_langgraph_dev(cfg, workspace_dir=tmp_path)

        assert result is proc
        start.assert_called_once()
        assert manager.is_async_subagents_available() is True

    def test_skips_when_async_and_memory_workers_disabled(self, tmp_path):
        """No background server is needed without async subagents or workers."""
        cfg = EvoScientistConfig()
        cfg.enable_async_subagents = False
        cfg.memory_workers_enabled = False
        cfg.langgraph_dev_port = 6174
        cfg.langgraph_dev_file_persistence = True
        with (
            patch.object(manager, "is_langgraph_dev_running") as mock_running,
            patch.object(manager, "start_langgraph_dev") as start,
            patch.object(manager, "_FILE_LOCK_PATH", tmp_path / "lg.lock"),
            patch.object(manager, "_PID_DIR", tmp_path / "pids"),
        ):
            result = manager.ensure_langgraph_dev(cfg, workspace_dir=tmp_path)

        assert result is None
        mock_running.assert_not_called()
        start.assert_not_called()
        assert manager.is_async_subagents_available() is False

    def test_reuses_existing_healthy_subprocess(self, tmp_path):
        """When the subprocess is already running, no new Popen call."""
        cfg = EvoScientistConfig()
        cfg.enable_async_subagents = True
        cfg.langgraph_dev_port = 6174
        cfg.langgraph_dev_file_persistence = True
        with (
            patch.object(
                manager, "is_langgraph_dev_running", return_value=True
            ) as mock_running,
            patch.object(manager, "start_langgraph_dev") as mock_start,
            patch.object(manager, "_FILE_LOCK_PATH", tmp_path / "lg.lock"),
            # Isolate from real ``~/.config/evoscientist/`` — without this
            # patch, the FileLock setup would mkdir the user's actual config
            # dir as a test side-effect.
            patch.object(manager, "_PID_DIR", tmp_path / "pids"),
        ):
            result = manager.ensure_langgraph_dev(cfg, workspace_dir=tmp_path)
            # We didn't spawn anything — there's already a healthy server.
            mock_start.assert_not_called()
            # Reuse path returns None (we don't own the existing process).
            assert result is None
            # is_async_subagents_available was flipped True.
            assert manager.is_async_subagents_available() is True
            # Health check was called at least once.
            assert mock_running.called


# =============================================================================
# is_async_subagents_available — module state
# =============================================================================


class TestIsAsyncSubagentsAvailable:
    def test_starts_false(self):
        assert manager.is_async_subagents_available() is False

    def test_reflects_module_state(self):
        manager._ASYNC_SUBAGENTS_AVAILABLE = True
        assert manager.is_async_subagents_available() is True
        manager._ASYNC_SUBAGENTS_AVAILABLE = False
        assert manager.is_async_subagents_available() is False
