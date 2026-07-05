"""Happy-path tests for langgraph_dev.manager.

Mocks httpx, psutil, subprocess.Popen, and module-level state so the tests
run on CI without requiring the langgraph CLI to be installed or any port
to be available.
"""

from __future__ import annotations

import dataclasses
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
    manager._LOG_OFFSET_AT_START = 0
    yield
    manager._PROCESS = None
    manager._PROCESS_WORKSPACE = None
    manager._ASYNC_SUBAGENTS_AVAILABLE = False
    manager._LOG_OFFSET_AT_START = 0


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
    def test_returns_false_if_no_pid_file(self, tmp_path, runtime_paths):
        with patch.object(
            manager,
            "RUNTIME",
            dataclasses.replace(runtime_paths, pid_file=tmp_path / "missing.pid"),
        ):
            assert manager._kill_owned_stale_process(6174) is False

    def test_returns_false_if_pid_file_unreadable(self, tmp_path, runtime_paths):
        pid_file = tmp_path / "bad.pid"
        pid_file.write_text("not-a-number")
        with patch.object(
            manager, "RUNTIME", dataclasses.replace(runtime_paths, pid_file=pid_file)
        ):
            assert manager._kill_owned_stale_process(6174) is False

    def test_returns_false_if_pid_not_in_occupiers(self, tmp_path, runtime_paths):
        pid_file = tmp_path / "lg.pid"
        pid_file.write_text("12345")
        with (
            patch.object(
                manager,
                "RUNTIME",
                dataclasses.replace(runtime_paths, pid_file=pid_file),
            ),
            patch.object(manager, "_list_pids_on_port", return_value=[99999]),
        ):
            assert manager._kill_owned_stale_process(6174) is False
            # PID file should be left intact — the port is held by someone
            # else, not a stale ours.
            assert pid_file.exists()

    def test_refuses_to_kill_recycled_pid(self, tmp_path, runtime_paths):
        """PID matches but cmdline doesn't contain 'langgraph' → don't kill."""
        pid_file = tmp_path / "lg.pid"
        pid_file.write_text("12345")
        fake_proc = MagicMock()
        fake_proc.cmdline.return_value = ["bash", "-c", "echo hi"]
        with (
            patch.object(
                manager,
                "RUNTIME",
                dataclasses.replace(runtime_paths, pid_file=pid_file),
            ),
            patch.object(manager, "_list_pids_on_port", return_value=[12345]),
            patch.object(manager.psutil, "Process", return_value=fake_proc),
        ):
            assert manager._kill_owned_stale_process(6174) is False
            fake_proc.kill.assert_not_called()
            # PID file should be removed — the entry is stale (our process is
            # gone, PID was recycled by an unrelated process).
            assert not pid_file.exists()

    def test_kills_when_cmdline_matches_langgraph(self, tmp_path, runtime_paths):
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
            patch.object(
                manager,
                "RUNTIME",
                dataclasses.replace(runtime_paths, pid_file=pid_file),
            ),
            patch.object(manager, "_list_pids_on_port", return_value=[12345]),
            patch.object(manager.psutil, "Process", return_value=fake_proc),
        ):
            assert manager._kill_owned_stale_process(6174) is True
            fake_proc.kill.assert_called_once()
            assert not pid_file.exists()

    def test_handles_dead_pid(self, tmp_path, runtime_paths):
        """PID file claims a PID but the process is gone → cleanup PID file, no error."""
        pid_file = tmp_path / "lg.pid"
        pid_file.write_text("12345")
        with (
            patch.object(
                manager,
                "RUNTIME",
                dataclasses.replace(runtime_paths, pid_file=pid_file),
            ),
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
    def test_starts_when_async_disabled_but_memory_workers_enabled(
        self, tmp_path, runtime_paths
    ):
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
            patch.object(
                manager,
                "RUNTIME",
                dataclasses.replace(
                    manager.LanggraphRuntimePaths.for_directory(tmp_path / "pids"),
                    lock_file=tmp_path / "lg.lock",
                ),
            ),
        ):
            result = manager.ensure_langgraph_dev(cfg, workspace_dir=tmp_path)

        assert result is proc
        start.assert_called_once()
        assert manager.is_async_subagents_available() is True

    def test_skips_when_async_and_memory_workers_disabled(
        self, tmp_path, runtime_paths
    ):
        """No background server is needed without async subagents, workers, or scheduler."""
        cfg = EvoScientistConfig()
        cfg.enable_async_subagents = False
        cfg.memory_workers_enabled = False
        cfg.enable_scheduler = False  # scheduler crons also require the backend
        cfg.memory_skill_synthesis_enabled = False
        cfg.langgraph_dev_port = 6174
        cfg.langgraph_dev_file_persistence = True
        with (
            patch.object(manager, "is_langgraph_dev_running") as mock_running,
            patch.object(manager, "start_langgraph_dev") as start,
            patch.object(
                manager,
                "RUNTIME",
                dataclasses.replace(
                    manager.LanggraphRuntimePaths.for_directory(tmp_path / "pids"),
                    lock_file=tmp_path / "lg.lock",
                ),
            ),
        ):
            result = manager.ensure_langgraph_dev(cfg, workspace_dir=tmp_path)

        assert result is None
        mock_running.assert_not_called()
        start.assert_not_called()
        assert manager.is_async_subagents_available() is False

    def test_needs_langgraph_dev_for_scheduler_only(self):
        """enable_scheduler alone requires the backend (crons fire inside it)."""
        cfg = EvoScientistConfig()
        cfg.enable_async_subagents = False
        cfg.memory_workers_enabled = False
        cfg.memory_skill_synthesis_enabled = False
        cfg.enable_scheduler = True
        assert manager.needs_langgraph_dev(cfg) is True
        cfg.enable_scheduler = False
        assert manager.needs_langgraph_dev(cfg) is False

    def test_reuses_existing_healthy_subprocess(self, tmp_path, runtime_paths):
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
            patch.object(
                manager,
                "RUNTIME",
                dataclasses.replace(
                    manager.LanggraphRuntimePaths.for_directory(tmp_path / "pids"),
                    lock_file=tmp_path / "lg.lock",
                ),
            ),
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


# =============================================================================
# _rotate_log_if_needed — log rotation for langgraph_dev.log
# =============================================================================


class TestRotateLogIfNeeded:
    """``_rotate_log_if_needed`` implements the single-backup rollover
    policy from #209. When ``RUNTIME.log_file`` exceeds the module's
    ``_LOG_ROTATION_BYTES`` threshold, rename to ``<log>.1`` (overwriting
    any existing backup) so the next open() starts fresh. Threshold
    is patched to a small value per-test to keep the fixtures tiny.
    """

    def test_no_existing_file_is_noop(self, tmp_path):
        log = tmp_path / "langgraph_dev.log"
        manager._rotate_log_if_needed(log)
        assert not log.exists()
        assert not (tmp_path / "langgraph_dev.log.1").exists()

    def test_file_smaller_than_threshold_is_not_rotated(self, tmp_path, monkeypatch):
        monkeypatch.setattr(manager, "_LOG_ROTATION_BYTES", 1024)
        log = tmp_path / "langgraph_dev.log"
        log.write_bytes(b"x" * 100)
        manager._rotate_log_if_needed(log)
        assert log.exists()
        assert log.stat().st_size == 100
        assert not (tmp_path / "langgraph_dev.log.1").exists()

    def test_file_exactly_at_threshold_is_not_rotated(self, tmp_path, monkeypatch):
        """Off-by-one: rotation triggers only on strict greater-than.
        A log sitting at the threshold size is left alone — the next
        session that pushes it over triggers the rollover.
        """
        monkeypatch.setattr(manager, "_LOG_ROTATION_BYTES", 1024)
        log = tmp_path / "langgraph_dev.log"
        log.write_bytes(b"x" * 1024)
        manager._rotate_log_if_needed(log)
        assert log.exists()
        assert log.stat().st_size == 1024
        assert not (tmp_path / "langgraph_dev.log.1").exists()

    def test_file_over_threshold_is_rotated(self, tmp_path, monkeypatch):
        monkeypatch.setattr(manager, "_LOG_ROTATION_BYTES", 1024)
        log = tmp_path / "langgraph_dev.log"
        log.write_bytes(b"x" * 1025)
        manager._rotate_log_if_needed(log)
        # After rotation: the original path was moved to ``<log>.1``.
        # The next ``open(log, "ab")`` will re-create the active file
        # at offset 0 (append mode creates if missing). Verify both
        # halves of the contract: the backup holds the previous content,
        # and the active log is writable from scratch.
        assert not log.exists()
        backup = tmp_path / "langgraph_dev.log.1"
        assert backup.exists()
        assert backup.stat().st_size == 1025
        with open(log, "ab") as fh:
            fh.write(b"new")
        assert log.stat().st_size == 3  # just "new", not appended to backup

    def test_rotation_overwrites_existing_backup(self, tmp_path, monkeypatch):
        """A pre-existing ``<log>.1`` from an earlier rotation must be
        clobbered by the new rollover — single-backup policy means we
        never keep more than one historical copy.
        """
        monkeypatch.setattr(manager, "_LOG_ROTATION_BYTES", 1024)
        log = tmp_path / "langgraph_dev.log"
        backup = tmp_path / "langgraph_dev.log.1"
        log.write_bytes(b"x" * 2000)
        backup.write_bytes(b"OLD_BACKUP_PAYLOAD_THAT_SHOULD_BE_GONE_NOW")
        original_backup_size = backup.stat().st_size
        manager._rotate_log_if_needed(log)
        assert backup.exists()
        assert backup.stat().st_size != original_backup_size
        assert backup.stat().st_size == 2000  # now holds the just-rotated log

    def test_failed_rotation_does_not_raise(self, tmp_path, monkeypatch):
        """If ``os.replace`` fails (e.g. permission denied on Windows
        when another process holds the backup open), the helper logs a
        warning and returns — the caller can still open the un-rotated
        log and proceed. Failing rotation is non-fatal: the next
        ``start_langgraph_dev`` invocation will try again.
        """
        monkeypatch.setattr(manager, "_LOG_ROTATION_BYTES", 0)  # always rotate
        log = tmp_path / "langgraph_dev.log"
        log.write_bytes(b"x" * 10)
        with patch(
            "EvoScientist.langgraph_dev.manager.os.replace",
            side_effect=PermissionError("denied"),
        ):
            # Must not raise.
            manager._rotate_log_if_needed(log)
        # Original log is left intact (we failed to rotate, didn't corrupt).
        assert log.exists()
        assert log.stat().st_size == 10


class TestStartLanggraphDevRotatesLog:
    """``start_langgraph_dev`` must call ``_rotate_log_if_needed`` before
    opening the log handle, so each session starts with either an
    existing-but-fresh log or a brand-new file. Verifying the call site
    directly (vs. mocking the entire subprocess spawn) keeps the test
    cheap while still guarding the integration point.
    """

    def test_rotate_called_before_open(self, tmp_path, monkeypatch):
        log = tmp_path / "langgraph_dev.log"
        log.write_bytes(b"x" * 2048)  # contents don't matter for the check
        # Build a fully temp-rooted runtime bundle via
        # ``for_directory`` so *every* path (pid_dir, pid_file,
        # workspace_sidecar, lock_file) is rooted under ``tmp_path``.
        # ``dataclasses.replace(runtime_paths, …)`` would still carry
        # ``pid_file`` / ``workspace_sidecar`` / ``lock_file`` from the
        # production object pointing at ``~/.config/evoscientist/``.
        pid_dir = tmp_path / "pids"
        monkeypatch.setattr(
            manager,
            "RUNTIME",
            dataclasses.replace(
                manager.LanggraphRuntimePaths.for_directory(pid_dir),
                log_file=log,
            ),
        )
        monkeypatch.setattr(manager, "_LOG_ROTATION_BYTES", 1024)
        # This test only verifies log rotation — we must not touch real
        # sockets.  Patch ``_can_bind_port`` so the bind-poll loop in
        # ``_wait_for_port_bindable`` passes immediately regardless of
        # whether port 6174 is in use on the dev machine.
        monkeypatch.setattr(manager, "_can_bind_port", lambda port: True)
        # Make ``_packaged_langgraph_config`` point at a real file so
        # ``start_langgraph_dev`` doesn't bail at the existence check
        # before reaching the rotation call.
        fake_config = tmp_path / "langgraph.json"
        fake_config.write_text("{}")
        # Don't actually start a subprocess — just verify the rotation
        # call happens. We mock the spawn to raise immediately so the
        # rest of start_langgraph_dev aborts before doing anything else.
        with (
            patch.object(manager, "_langgraph_exe", return_value="/fake/langgraph"),
            patch.object(
                manager, "_packaged_langgraph_config", return_value=fake_config
            ),
            patch(
                "EvoScientist.langgraph_dev.manager.subprocess.Popen",
                side_effect=FileNotFoundError("subprocess not available"),
            ),
        ):
            try:
                manager.start_langgraph_dev(workspace_dir=tmp_path)
            except FileNotFoundError:
                pass  # expected — we just need rotation to have happened
        # After start attempt, the oversize log must have been rotated.
        assert (tmp_path / "langgraph_dev.log.1").exists()
        # And the redirect held — nothing leaked into the real
        # ``~/.config/evoscientist/`` (we'd have observed a
        # ``langgraph_dev.log.1`` *next* to the user's real log, not
        # under ``tmp_path``). The ``pid_dir`` we redirected to must
        # exist, proving the function reached past the mkdir prelude.
        assert pid_dir.is_dir()


class TestStartLanggraphDevCapturesLogOffset:
    """``start_langgraph_dev`` must capture ``_LOG_OFFSET_AT_START`` at the
    right moment — after ``_rotate_log_if_needed`` + ``open('ab')`` but
    before ``subprocess.Popen`` — so ``read_tunnel_url`` scans only this
    session's bytes. The existing ``TestReadTunnelUrl`` tests monkeypatch
    the offset directly (consumer side); these guard the producer side, so
    a regression moving the capture line would actually be caught.
    """

    def _patch_prereqs(self, tmp_path, monkeypatch, log):
        """Mock everything up to (but not including) Popen, redirecting all
        runtime paths under ``tmp_path``."""
        pid_dir = tmp_path / "pids"
        monkeypatch.setattr(
            manager,
            "RUNTIME",
            dataclasses.replace(
                manager.LanggraphRuntimePaths.for_directory(pid_dir),
                log_file=log,
            ),
        )
        monkeypatch.setattr(manager, "_can_bind_port", lambda port: True)
        fake_config = tmp_path / "langgraph.json"
        fake_config.write_text("{}")
        monkeypatch.setattr(manager, "_langgraph_exe", lambda: "/fake/langgraph")
        monkeypatch.setattr(manager, "_packaged_langgraph_config", lambda: fake_config)

    def test_offset_equals_existing_log_size(self, tmp_path, monkeypatch):
        """No rotation → offset is the pre-existing (appended-to) log size,
        so a stale URL above that offset is never re-read."""
        log = tmp_path / "langgraph_dev.log"
        log.write_bytes(b"x" * 512)
        # Keep the log well under the rotation threshold so it is NOT rotated.
        monkeypatch.setattr(manager, "_LOG_ROTATION_BYTES", 10**9)
        self._patch_prereqs(tmp_path, monkeypatch, log)

        captured: dict = {}

        def _fake_popen(args, **kwargs):
            # Read the global at the instant Popen is invoked — this is
            # strictly after the capture line in start_langgraph_dev.
            captured["offset"] = manager._LOG_OFFSET_AT_START
            raise FileNotFoundError("stop before real spawn")

        monkeypatch.setattr(
            "EvoScientist.langgraph_dev.manager.subprocess.Popen", _fake_popen
        )
        try:
            manager.start_langgraph_dev(workspace_dir=tmp_path)
        except FileNotFoundError:
            pass
        assert captured["offset"] == 512

    def test_offset_zero_after_forced_rotation(self, tmp_path, monkeypatch):
        """Forced rotation moves the old log away; the fresh ``open('ab')``
        starts empty → offset 0 (scan the whole new file)."""
        log = tmp_path / "langgraph_dev.log"
        log.write_bytes(b"x" * 4096)
        monkeypatch.setattr(manager, "_LOG_ROTATION_BYTES", 1024)
        self._patch_prereqs(tmp_path, monkeypatch, log)

        captured: dict = {}

        def _fake_popen(args, **kwargs):
            captured["offset"] = manager._LOG_OFFSET_AT_START
            raise FileNotFoundError("stop before real spawn")

        monkeypatch.setattr(
            "EvoScientist.langgraph_dev.manager.subprocess.Popen", _fake_popen
        )
        try:
            manager.start_langgraph_dev(workspace_dir=tmp_path)
        except FileNotFoundError:
            pass
        assert (tmp_path / "langgraph_dev.log.1").exists()  # rotation happened
        assert captured["offset"] == 0


# =============================================================================
# read_tunnel_url
# =============================================================================


class TestReadTunnelUrl:
    """``read_tunnel_url`` scrapes the Cloudflare tunnel URL from the log,
    scanning only bytes written after the current subprocess started."""

    def test_returns_url_when_present(self, tmp_path, runtime_paths, monkeypatch):
        log = tmp_path / "langgraph_dev.log"
        log.write_text(
            "INFO server up\n"
            "[cloudflared] Your quick Tunnel has been created! Visit it at:\n"
            "[cloudflared] https://happy-tiger-demo.trycloudflare.com\n"
        )
        monkeypatch.setattr(
            manager, "RUNTIME", dataclasses.replace(runtime_paths, log_file=log)
        )
        monkeypatch.setattr(manager, "_LOG_OFFSET_AT_START", 0)

        assert (
            manager.read_tunnel_url(timeout=1.0)
            == "https://happy-tiger-demo.trycloudflare.com"
        )

    def test_returns_none_on_timeout(self, tmp_path, runtime_paths, monkeypatch):
        log = tmp_path / "langgraph_dev.log"
        log.write_text("INFO server up — but no tunnel line ever printed\n")
        monkeypatch.setattr(
            manager, "RUNTIME", dataclasses.replace(runtime_paths, log_file=log)
        )
        monkeypatch.setattr(manager, "_LOG_OFFSET_AT_START", 0)

        assert manager.read_tunnel_url(timeout=0.2, poll_interval=0.05) is None

    def test_ignores_stale_url_before_offset(
        self, tmp_path, runtime_paths, monkeypatch
    ):
        """A URL from a previous session (before the offset) must be skipped;
        only this session's bytes count."""
        stale = "[cloudflared] https://old-stale-url.trycloudflare.com\n"
        log = tmp_path / "langgraph_dev.log"
        log.write_text(stale)
        monkeypatch.setattr(
            manager, "RUNTIME", dataclasses.replace(runtime_paths, log_file=log)
        )
        # Offset points past the stale line — nothing fresh yet → None.
        monkeypatch.setattr(manager, "_LOG_OFFSET_AT_START", len(stale.encode()))
        assert manager.read_tunnel_url(timeout=0.2, poll_interval=0.05) is None

        # Now this session appends its own fresh URL → returned.
        with open(log, "a") as fh:
            fh.write("[cloudflared] https://fresh-new-url.trycloudflare.com\n")
        assert (
            manager.read_tunnel_url(timeout=1.0)
            == "https://fresh-new-url.trycloudflare.com"
        )
