"""Happy-path tests for langgraph_dev.manager.

Mocks httpx, psutil, subprocess.Popen, and module-level state so the tests
run on CI without requiring the langgraph CLI to be installed or any port
to be available.
"""

from __future__ import annotations

import dataclasses
import sys
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
# Bind host vs. probe host
# =============================================================================


class TestProbeHost:
    """``_probe_host`` is the seam that makes host support safe: only bind()
    uses the configured interface, every client falls back to something
    actually reachable."""

    @pytest.mark.parametrize("wildcard", ["0.0.0.0", "::", ""])
    def test_wildcards_map_to_loopback(self, wildcard):
        assert manager._probe_host(wildcard) == "127.0.0.1"

    @pytest.mark.parametrize(
        "host", ["127.0.0.1", "192.168.1.5", "::1", "example.test"]
    )
    def test_specific_hosts_pass_through(self, host):
        assert manager._probe_host(host) == host

    def test_defaults_to_loopback(self):
        assert manager._probe_host() == "127.0.0.1"


class TestIsLoopbackHost:
    """Drives the public-bind warning, so it must be conservative: only
    provable loopback suppresses the banner."""

    @pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost", "  LOCALHOST  "])
    def test_loopback_recognized(self, host):
        assert manager._is_loopback_host(host) is True

    @pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.5", "example.test"])
    def test_exposed_hosts_rejected(self, host):
        assert manager._is_loopback_host(host) is False


class TestBaseUrl:
    def test_default_is_loopback(self):
        assert manager._base_url(6174) == "http://127.0.0.1:6174"

    def test_wildcard_renders_as_loopback(self):
        assert manager._base_url(6174, "0.0.0.0") == "http://127.0.0.1:6174"

    def test_specific_host_preserved(self):
        assert manager._base_url(6174, "192.168.1.5") == "http://192.168.1.5:6174"

    def test_ipv6_literal_is_bracketed(self):
        """Unbracketed ``::1:6174`` is not a parseable authority (RFC 3986)."""
        assert manager._base_url(6174, "::1") == "http://[::1]:6174"


class TestCanBindPort:
    def test_binds_literal_host_not_probe_host(self, monkeypatch):
        """``_can_bind_port`` must replicate the server's own bind. Probing
        loopback while the server claims 0.0.0.0 would give false confidence
        when another process holds a single non-loopback interface."""
        import socket as _socket

        bound: list[tuple] = []

        class _FakeSocket:
            def __init__(self, family, type_):
                self.family = family

            def bind(self, addr):
                bound.append((self.family, addr))

            def close(self):
                pass

        monkeypatch.setattr(_socket, "socket", _FakeSocket)

        assert manager._can_bind_port(6174, "0.0.0.0") is True
        assert bound == [(_socket.AF_INET, ("0.0.0.0", 6174))]

    def test_ipv6_host_uses_ipv6_family(self, monkeypatch):
        import socket as _socket

        bound: list[tuple] = []

        class _FakeSocket:
            def __init__(self, family, type_):
                self.family = family

            def bind(self, addr):
                bound.append((self.family, addr))

            def close(self):
                pass

        monkeypatch.setattr(_socket, "socket", _FakeSocket)

        assert manager._can_bind_port(6174, "::1") is True
        assert bound == [(_socket.AF_INET6, ("::1", 6174))]


# =============================================================================
# langgraph CLI resolution
# =============================================================================


class TestLanggraphCliResolution:
    def _make_executable(self, path):
        path.write_text("#!/bin/sh\n", encoding="utf-8")
        path.chmod(0o755)

    def test_prefers_current_python_environment_over_path(self, tmp_path, monkeypatch):
        local_bin = tmp_path / "local" / "bin"
        local_bin.mkdir(parents=True)
        local_langgraph = local_bin / "langgraph"
        self._make_executable(local_langgraph)

        path_bin = tmp_path / "path" / "bin"
        path_bin.mkdir(parents=True)
        path_langgraph = path_bin / "langgraph"
        self._make_executable(path_langgraph)

        monkeypatch.setattr(sys, "executable", str(local_bin / "python"))
        monkeypatch.setattr(
            manager.shutil,
            "which",
            lambda command: str(path_langgraph) if command == "langgraph" else None,
        )

        assert manager._langgraph_exe() == str(local_langgraph)

    def test_falls_back_to_path_when_environment_binary_missing(
        self, tmp_path, monkeypatch
    ):
        path_bin = tmp_path / "path" / "bin"
        path_bin.mkdir(parents=True)
        path_langgraph = path_bin / "langgraph"
        self._make_executable(path_langgraph)

        monkeypatch.setattr(
            sys, "executable", str(tmp_path / "local" / "bin" / "python")
        )
        monkeypatch.setattr(
            manager.shutil,
            "which",
            lambda command: str(path_langgraph) if command == "langgraph" else None,
        )

        assert manager._langgraph_exe() == str(path_langgraph)

    def test_checks_windows_suffix_next_to_current_python(self, tmp_path, monkeypatch):
        scripts_dir = tmp_path / "Scripts"
        scripts_dir.mkdir()
        local_langgraph = scripts_dir / "langgraph.exe"
        self._make_executable(local_langgraph)

        path_bin = tmp_path / "path" / "bin"
        path_bin.mkdir(parents=True)
        path_langgraph = path_bin / "langgraph.exe"
        self._make_executable(path_langgraph)

        monkeypatch.setattr(sys, "executable", str(scripts_dir / "python.exe"))
        monkeypatch.setattr(manager.os, "name", "nt", raising=False)
        monkeypatch.setattr(
            manager.shutil,
            "which",
            lambda command: str(path_langgraph) if command == "langgraph" else None,
        )

        assert manager._langgraph_exe() == str(local_langgraph)


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
        # Verify it probed /ok at the configured port. 127.0.0.1 rather than
        # "localhost" on purpose: the latter can resolve to ::1 first, which
        # never reaches a server bound to an IPv4 interface.
        called_url = mock_get.call_args[0][0]
        assert called_url == "http://127.0.0.1:6174/ok"

    @patch("EvoScientist.langgraph_dev.manager.httpx.get")
    def test_returns_false_on_non_200(self, mock_get):
        mock_get.return_value = MagicMock(status_code=503)
        assert manager.is_langgraph_dev_running(port=6174) is False

    @patch("EvoScientist.langgraph_dev.manager.httpx.get")
    def test_wildcard_bind_probed_over_loopback(self, mock_get):
        """A server bound to 0.0.0.0 also listens on loopback, and you cannot
        meaningfully connect to 0.0.0.0 itself — probe 127.0.0.1."""
        mock_get.return_value = MagicMock(status_code=200)
        assert manager.is_langgraph_dev_running(port=6174, host="0.0.0.0") is True
        assert mock_get.call_args[0][0] == "http://127.0.0.1:6174/ok"

    @patch("EvoScientist.langgraph_dev.manager.httpx.get")
    def test_specific_host_probed_verbatim(self, mock_get):
        """Loopback would not reach a server pinned to one interface."""
        mock_get.return_value = MagicMock(status_code=200)
        assert manager.is_langgraph_dev_running(port=6174, host="192.168.1.5") is True
        assert mock_get.call_args[0][0] == "http://192.168.1.5:6174/ok"

    @patch("EvoScientist.langgraph_dev.manager.httpx.get")
    def test_explicit_base_url_still_wins(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        assert (
            manager.is_langgraph_dev_running(
                base_url="http://example.test:1234", port=6174, host="0.0.0.0"
            )
            is True
        )
        assert mock_get.call_args[0][0] == "http://example.test:1234/ok"


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
        monkeypatch.setattr(manager, "_can_bind_port", lambda port, *_a, **_kw: True)
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


@pytest.fixture
def start_langgraph_dev_capture(tmp_path, monkeypatch):
    """Prereq patches + capturing ``_fake_popen`` for ``start_langgraph_dev``
    tests. Mocks everything up to (but not including) ``Popen``, redirecting
    all runtime paths under ``tmp_path``, then installs a ``_fake_popen``
    that records the argv, env, and ``_LOG_OFFSET_AT_START`` at the instant
    ``Popen`` is invoked and raises ``FileNotFoundError`` to stop before the
    real spawn. Callers may seed the log file (available as ``env.log``)
    before invoking ``start_langgraph_dev``.
    """
    pid_dir = tmp_path / "pids"
    log = tmp_path / "langgraph_dev.log"
    monkeypatch.setattr(
        manager,
        "RUNTIME",
        dataclasses.replace(
            manager.LanggraphRuntimePaths.for_directory(pid_dir),
            log_file=log,
        ),
    )
    monkeypatch.setattr(manager, "_can_bind_port", lambda port, *_a, **_kw: True)
    fake_config = tmp_path / "langgraph.json"
    fake_config.write_text("{}")
    monkeypatch.setattr(manager, "_langgraph_exe", lambda: "/fake/langgraph")
    monkeypatch.setattr(manager, "_packaged_langgraph_config", lambda: fake_config)

    captured: dict = {}

    def _fake_popen(args, **kwargs):
        # Read the offset global at the instant Popen is invoked — this is
        # strictly after the capture line in start_langgraph_dev.
        captured["args"] = args
        captured["env"] = kwargs["env"]
        captured["offset"] = manager._LOG_OFFSET_AT_START
        raise FileNotFoundError("stop before real spawn")

    monkeypatch.setattr(
        "EvoScientist.langgraph_dev.manager.subprocess.Popen", _fake_popen
    )
    return SimpleNamespace(tmp_path=tmp_path, log=log, captured=captured)


class TestStartLanggraphDevCapturesLogOffset:
    """``start_langgraph_dev`` must capture ``_LOG_OFFSET_AT_START`` at the
    right moment — after ``_rotate_log_if_needed`` + ``open('ab')`` but
    before ``subprocess.Popen`` — so ``read_tunnel_url`` scans only this
    session's bytes. The existing ``TestReadTunnelUrl`` tests monkeypatch
    the offset directly (consumer side); these guard the producer side, so
    a regression moving the capture line would actually be caught.
    """

    def test_offset_equals_existing_log_size(
        self, start_langgraph_dev_capture, monkeypatch
    ):
        """No rotation → offset is the pre-existing (appended-to) log size,
        so a stale URL above that offset is never re-read."""
        env = start_langgraph_dev_capture
        env.log.write_bytes(b"x" * 512)
        # Keep the log well under the rotation threshold so it is NOT rotated.
        monkeypatch.setattr(manager, "_LOG_ROTATION_BYTES", 10**9)

        try:
            manager.start_langgraph_dev(workspace_dir=env.tmp_path)
        except FileNotFoundError:
            pass
        assert env.captured["offset"] == 512

    def test_offset_zero_after_forced_rotation(
        self, start_langgraph_dev_capture, monkeypatch
    ):
        """Forced rotation moves the old log away; the fresh ``open('ab')``
        starts empty → offset 0 (scan the whole new file)."""
        env = start_langgraph_dev_capture
        env.log.write_bytes(b"x" * 4096)
        monkeypatch.setattr(manager, "_LOG_ROTATION_BYTES", 1024)

        try:
            manager.start_langgraph_dev(workspace_dir=env.tmp_path)
        except FileNotFoundError:
            pass
        assert (env.tmp_path / "langgraph_dev.log.1").exists()  # rotation happened
        assert env.captured["offset"] == 0


class TestStartLanggraphDevPropagatesPort:
    """``start_langgraph_dev`` must export the effective bind port into the
    subprocess env as ``EVOSCIENTIST_LANGGRAPH_DEV_PORT`` so the deployed
    main agent's ``cfg.langgraph_dev_port`` matches what langgraph dev
    actually bound to. Without this, ``EvoSci deploy --port X`` binds to X
    but the deployed agent reads the persisted ``langgraph_dev_port``
    (whatever ``EvoSci config set langgraph_dev_port`` last wrote), and
    every ``start_async_task`` fails with "All connection attempts failed"
    because the self-loop URL points at an unbound port.
    """

    def test_env_carries_explicit_bind_port(self, start_langgraph_dev_capture):
        """The ``env`` passed to Popen must set the env var to ``str(port)``
        matching whatever the caller resolved, AND that value must match the
        ``--port`` argv the subprocess is spawned with. Comparing both closes
        the exact desync class this PR fixes — a future refactor that changed
        how argv gets its port (or introduced a second port variable) would
        slip past a pure env-only assertion."""
        env = start_langgraph_dev_capture
        try:
            manager.start_langgraph_dev(workspace_dir=env.tmp_path, port=6617)
        except FileNotFoundError:
            pass
        assert env.captured["env"]["EVOSCIENTIST_LANGGRAPH_DEV_PORT"] == "6617"
        argv = env.captured["args"]
        assert (
            argv[argv.index("--port") + 1]
            == env.captured["env"]["EVOSCIENTIST_LANGGRAPH_DEV_PORT"]
        )

    def test_env_value_replaces_inherited(
        self, start_langgraph_dev_capture, monkeypatch
    ):
        """A stray parent-shell export of ``EVOSCIENTIST_LANGGRAPH_DEV_PORT``
        must NOT shadow the caller-resolved bind port in the subprocess env.
        Dict assignment on ``sub_env`` already guarantees this, but the
        argv-vs-env cross-check pins the invariant against a future refactor
        that decoupled the two."""
        monkeypatch.setenv("EVOSCIENTIST_LANGGRAPH_DEV_PORT", "9999")
        env = start_langgraph_dev_capture
        try:
            manager.start_langgraph_dev(workspace_dir=env.tmp_path, port=6606)
        except FileNotFoundError:
            pass
        # Parent's 9999 replaced; caller's 6606 wins in both env and argv.
        assert env.captured["env"]["EVOSCIENTIST_LANGGRAPH_DEV_PORT"] == "6606"
        argv = env.captured["args"]
        assert (
            argv[argv.index("--port") + 1]
            == env.captured["env"]["EVOSCIENTIST_LANGGRAPH_DEV_PORT"]
        )


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
