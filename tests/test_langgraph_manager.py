"""Happy-path tests for langgraph_dev.manager.

Mocks httpx, psutil, subprocess.Popen, and module-level state so the tests
run on CI without requiring the langgraph CLI to be installed or any port
to be available.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from EvoScientist.config.settings import EvoScientistConfig
from EvoScientist.langgraph_dev import manager
from EvoScientist.langgraph_dev.boundary import is_loopback_host
from tests.fakes.langgraph_dev import Server


@pytest.fixture(autouse=True)
def reset_module_state():
    """Reset manager module globals before each test for isolation."""
    manager._PROCESS = None
    manager._ASYNC_SUBAGENTS_AVAILABLE = False
    manager._LOG_OFFSET_AT_START = 0
    yield
    manager._PROCESS = None
    manager._ASYNC_SUBAGENTS_AVAILABLE = False
    manager._LOG_OFFSET_AT_START = 0


# =============================================================================
# Bind host vs. probe host
# =============================================================================


class TestProbeHost:
    """``_probe_host`` is the seam that makes host support safe: only bind()
    uses the configured interface, every client falls back to something
    actually reachable."""

    @pytest.mark.parametrize(
        ("wildcard", "loopback"),
        [("0.0.0.0", "127.0.0.1"), ("", "127.0.0.1"), ("::", "::1"), ("[::]", "::1")],
    )
    def test_wildcards_map_to_loopback(self, wildcard, loopback):
        assert manager._probe_host(wildcard) == loopback

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

    @pytest.mark.parametrize(
        "host", ["127.0.0.1", "::1", "[::1]", "localhost", "  LOCALHOST  "]
    )
    def test_loopback_recognized(self, host):
        assert is_loopback_host(host) is True

    @pytest.mark.parametrize(
        "host", ["0.0.0.0", "::", "192.168.1.5", "example.test", "[::1", "::1]"]
    )
    def test_exposed_hosts_rejected(self, host):
        assert is_loopback_host(host) is False


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
    def test_probe_bypasses_proxy(self, mock_get):
        # The loopback probe must set trust_env=False: on Windows httpx otherwise
        # routes 127.0.0.1 through the system (registry) proxy and never reaches
        # the local server, so a healthy backend reads as unhealthy.
        mock_get.return_value = MagicMock(status_code=200)
        manager.is_langgraph_dev_running(port=6174)
        assert mock_get.call_args.kwargs.get("trust_env") is False

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
# ensure_server — whether a server is needed
# =============================================================================


class TestEnsureServer:
    def test_starts_when_async_disabled_but_memory_workers_enabled(
        self, tmp_path, monkeypatch, runtime_paths
    ):
        """EvoMemory workers can require langgraph dev even without async subagents."""
        cfg = EvoScientistConfig()
        cfg.enable_async_subagents = False
        cfg.memory_workers_enabled = True
        server = Server(monkeypatch)

        result = manager.ensure_server(cfg)

        assert result.served is True
        assert len(server.spawns) == 1
        assert manager.is_async_subagents_available() is True

    def test_skips_when_async_and_memory_workers_disabled(
        self, tmp_path, monkeypatch, runtime_paths
    ):
        """No background server is needed without async subagents, workers, or scheduler."""
        cfg = EvoScientistConfig()
        cfg.enable_async_subagents = False
        cfg.memory_workers_enabled = False
        cfg.enable_scheduler = False  # scheduler crons also require the backend
        cfg.memory_skill_synthesis_enabled = False
        server = Server(monkeypatch)

        result = manager.ensure_server(cfg, workspace_dir=tmp_path)

        # No server is needed, so the session has what it needs.
        assert result.served is True
        assert result.available is False
        assert server.spawns == []
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

    def test_needs_langgraph_dev_for_server_backend_only(self):
        """gateway_backend=langgraph_server needs the server even with every
        other trigger off (execution itself is server-routed)."""
        cfg = EvoScientistConfig()
        cfg.enable_async_subagents = False
        cfg.memory_workers_enabled = False
        cfg.memory_skill_synthesis_enabled = False
        cfg.enable_scheduler = False
        assert manager.needs_langgraph_dev(cfg) is False
        cfg.gateway_backend = "langgraph_server"
        assert manager.needs_langgraph_dev(cfg) is True

    def test_needs_langgraph_dev_for_anthropic_oauth(self):
        """Anthropic's OAuth goes through the server's ccproxy."""
        cfg = EvoScientistConfig()
        cfg.enable_async_subagents = False
        cfg.memory_workers_enabled = False
        cfg.memory_skill_synthesis_enabled = False
        cfg.enable_scheduler = False
        cfg.anthropic_auth_mode = "oauth"
        # Whatever is configured: /model may pick an Anthropic model.
        cfg.provider = "openai"
        assert manager.needs_langgraph_dev(cfg) is True

    def test_needs_langgraph_dev_explicit_backend_overrides_global(self):
        """The explicit ``backend`` param wins over ``config.gateway_backend``.

        A surface resolved to the server backend must force the dev server even
        when the global flag is still ``local`` (per-surface override), and a
        surface resolved to ``local`` must not be forced on by a server global
        when it has no other trigger.
        """
        cfg = EvoScientistConfig()
        cfg.enable_async_subagents = False
        cfg.memory_workers_enabled = False
        cfg.memory_skill_synthesis_enabled = False
        cfg.enable_scheduler = False
        cfg.gateway_backend = "local"
        assert manager.needs_langgraph_dev(cfg, backend="langgraph_server") is True
        cfg.gateway_backend = "langgraph_server"
        assert manager.needs_langgraph_dev(cfg, backend="local") is False

    def test_reuses_the_recorded_server(self, monkeypatch, runtime_paths):
        """When the server is already running, nothing is spawned."""
        cfg = EvoScientistConfig()
        cfg.enable_async_subagents = True
        server = Server(monkeypatch)
        server.run()

        result = manager.ensure_server(cfg)

        assert server.spawns == []
        assert result.served is True
        assert manager.is_async_subagents_available() is True


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
        ``_spawn_server`` invocation will try again.
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


class TestSpawnRotatesLog:
    """``_spawn_server`` must call ``_rotate_log_if_needed`` before
    opening the log handle, so each session starts with either an
    existing-but-fresh log or a brand-new file."""

    def test_rotate_called_before_open(self, spawn_capture, monkeypatch):
        spawn_capture.log.write_bytes(b"x" * 2048)
        monkeypatch.setattr(manager, "_LOG_ROTATION_BYTES", 1024)

        with pytest.raises(FileNotFoundError):
            manager._spawn_server()

        assert spawn_capture.log.with_name("langgraph_dev.log.1").exists()


@pytest.fixture
def spawn_capture(tmp_path, monkeypatch, runtime_paths):
    """Prereq patches + capturing ``_fake_popen`` for ``_spawn_server``
    tests: records the argv, env, and ``_LOG_OFFSET_AT_START`` at the instant
    ``Popen`` is invoked and raises ``FileNotFoundError`` to stop before the
    real spawn. Callers may seed the log file (``env.log``, in the server
    home) before invoking ``_spawn_server``.
    """
    fake_config = tmp_path / "langgraph.json"
    fake_config.write_text("{}")
    monkeypatch.setattr(manager, "_langgraph_exe", lambda: "/fake/langgraph")
    monkeypatch.setattr(manager, "_packaged_langgraph_config", lambda: fake_config)

    captured: dict = {}

    def _fake_popen(args, **kwargs):
        captured["args"] = args
        captured["env"] = kwargs["env"]
        captured["kwargs"] = kwargs
        captured["offset"] = manager._LOG_OFFSET_AT_START
        raise FileNotFoundError("stop before real spawn")

    monkeypatch.setattr(
        "EvoScientist.langgraph_dev.manager.subprocess.Popen", _fake_popen
    )
    return SimpleNamespace(
        tmp_path=tmp_path, log=runtime_paths.log_file, captured=captured
    )


class TestStartLanggraphDevCapturesLogOffset:
    """``_spawn_server`` must capture ``_LOG_OFFSET_AT_START`` at the
    right moment — after ``_rotate_log_if_needed`` + ``open('ab')`` but
    before ``subprocess.Popen`` — so ``read_tunnel_url`` scans only this
    session's bytes. The existing ``TestReadTunnelUrl`` tests monkeypatch
    the offset directly (consumer side); these guard the producer side, so
    a regression moving the capture line would actually be caught.
    """

    def test_offset_equals_existing_log_size(self, spawn_capture, monkeypatch):
        """No rotation → offset is the pre-existing (appended-to) log size,
        so a stale URL above that offset is never re-read."""
        env = spawn_capture
        env.log.write_bytes(b"x" * 512)
        # Keep the log well under the rotation threshold so it is NOT rotated.
        monkeypatch.setattr(manager, "_LOG_ROTATION_BYTES", 10**9)

        try:
            manager._spawn_server()
        except FileNotFoundError:
            pass
        assert env.captured["offset"] == 512

    def test_offset_zero_after_forced_rotation(self, spawn_capture, monkeypatch):
        """Forced rotation moves the old log away; the fresh ``open('ab')``
        starts empty → offset 0 (scan the whole new file)."""
        env = spawn_capture
        env.log.write_bytes(b"x" * 4096)
        monkeypatch.setattr(manager, "_LOG_ROTATION_BYTES", 1024)

        try:
            manager._spawn_server()
        except FileNotFoundError:
            pass
        assert env.log.with_name("langgraph_dev.log.1").exists()  # rotation happened
        assert env.captured["offset"] == 0


class TestStartLanggraphDevPropagatesPort:
    """``_spawn_server`` must export the effective bind port into the
    subprocess env as ``EVOSCIENTIST_LANGGRAPH_DEV_PORT`` so the deployed
    main agent's ``cfg.langgraph_dev_port`` matches what langgraph dev
    actually bound to. Without this, ``EvoSci deploy --port X`` binds to X
    but the deployed agent reads the persisted ``langgraph_dev_port``
    (whatever ``EvoSci config set langgraph_dev_port`` last wrote), and
    every ``start_async_task`` fails with "All connection attempts failed"
    because the self-loop URL points at an unbound port.
    """

    def test_env_carries_explicit_bind_port(self, spawn_capture):
        """The ``env`` passed to Popen must set the env var to ``str(port)``
        matching whatever the caller resolved, AND that value must match the
        ``--port`` argv the subprocess is spawned with. Comparing both closes
        the exact desync class this PR fixes — a future refactor that changed
        how argv gets its port (or introduced a second port variable) would
        slip past a pure env-only assertion."""
        env = spawn_capture
        try:
            manager._spawn_server(port=6617)
        except FileNotFoundError:
            pass
        assert env.captured["env"]["EVOSCIENTIST_LANGGRAPH_DEV_PORT"] == "6617"
        argv = env.captured["args"]
        assert (
            argv[argv.index("--port") + 1]
            == env.captured["env"]["EVOSCIENTIST_LANGGRAPH_DEV_PORT"]
        )

    def test_env_value_replaces_inherited(self, spawn_capture, monkeypatch):
        """A stray parent-shell export of ``EVOSCIENTIST_LANGGRAPH_DEV_PORT``
        must NOT shadow the caller-resolved bind port in the subprocess env.
        Dict assignment on ``sub_env`` already guarantees this, but the
        argv-vs-env cross-check pins the invariant against a future refactor
        that decoupled the two."""
        monkeypatch.setenv("EVOSCIENTIST_LANGGRAPH_DEV_PORT", "9999")
        env = spawn_capture
        try:
            manager._spawn_server(port=6606)
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
        log = runtime_paths.log_file
        log.write_text(
            "INFO server up\n"
            "[cloudflared] Your quick Tunnel has been created! Visit it at:\n"
            "[cloudflared] https://happy-tiger-demo.trycloudflare.com\n"
        )
        monkeypatch.setattr(manager, "_LOG_OFFSET_AT_START", 0)

        assert (
            manager.read_tunnel_url(timeout=1.0)
            == "https://happy-tiger-demo.trycloudflare.com"
        )

    @pytest.mark.parametrize("api_host", ["api", "API"])
    def test_ignores_api_url(self, tmp_path, runtime_paths, monkeypatch, api_host):
        """langgraph-api names cloudflared's API endpoint when no tunnel came up."""
        log = runtime_paths.log_file
        log.write_text(f"- API: https://{api_host}.trycloudflare.com\n")
        monkeypatch.setattr(manager, "_LOG_OFFSET_AT_START", 0)

        assert manager.read_tunnel_url(timeout=0.2, poll_interval=0.05) is None

    @pytest.mark.parametrize(
        "failure",
        [
            "failed to build quick tunnel request: net/http: invalid method",
            "failed to request quick Tunnel: Post "
            '"https://API.trycloudflare.com/tunnel": i/o timeout',
            "failed to read quick-tunnel response: unexpected EOF",
            # HTTP 429 (rate limited): the body is "error code: 1015".
            "failed to unmarshal quick Tunnel: invalid character 'e' looking "
            "for beginning of value",
            "failed to parse quick Tunnel ID: invalid UUID length: 0",
        ],
    )
    def test_raises_at_once_on_tunnel_failure(
        self, tmp_path, runtime_paths, monkeypatch, failure
    ):
        prefix = "INFO:langgraph_api.tunneling.cloudflare:[cloudflared]"
        log = runtime_paths.log_file
        log.write_text(
            f"{prefix} 2026-10-07T15:39:12Z INF Requesting new quick Tunnel "
            "on trycloudflare.com...\n"
            f"{prefix} 2026-10-07T15:39:12Z ERR Error unmarshaling "
            'QuickTunnel response: error code: 1015 status_code="429"\n'
            f"{prefix} {failure}\n"
            "- API: https://api.trycloudflare.com\n"
        )
        monkeypatch.setattr(manager, "_LOG_OFFSET_AT_START", 0)

        def _no_wait(_seconds):
            pytest.fail("read_tunnel_url waited after cloudflared reported a failure")

        monkeypatch.setattr(
            manager,
            "time",
            SimpleNamespace(monotonic=manager.time.monotonic, sleep=_no_wait),
        )
        with pytest.raises(manager.TunnelFailedError) as info:
            manager.read_tunnel_url(timeout=5.0)
        assert str(info.value) == failure

    def test_connection_errors_are_not_tunnel_failures(
        self, tmp_path, runtime_paths, monkeypatch
    ):
        log = runtime_paths.log_file
        log.write_text(
            "[cloudflared] ERR Failed to serve tunnel connection "
            'error="Application error 0x0 (remote)" connIndex=0\n'
            '[cloudflared] ERR Serve tunnel error error="Application error 0x0 '
            '(remote)" connIndex=0\n'
        )
        monkeypatch.setattr(manager, "_LOG_OFFSET_AT_START", 0)

        assert manager.read_tunnel_url(timeout=0.2, poll_interval=0.05) is None

    def test_ignores_a_failure_before_offset(
        self, tmp_path, runtime_paths, monkeypatch
    ):
        stale = "[cloudflared] failed to request quick Tunnel: i/o timeout\n"
        log = runtime_paths.log_file
        log.write_text(stale)
        monkeypatch.setattr(manager, "_LOG_OFFSET_AT_START", len(stale.encode()))

        assert manager.read_tunnel_url(timeout=0.2, poll_interval=0.05) is None

    @pytest.mark.parametrize("tunnel_host", ["happy-tiger-demo", "api-friendly-demo"])
    def test_returns_tunnel_url_after_api_url(
        self, tmp_path, runtime_paths, monkeypatch, tunnel_host
    ):
        log = runtime_paths.log_file
        public_url = f"https://{tunnel_host}.trycloudflare.com"
        log.write_text(
            "[cloudflared] failed to request quick Tunnel: Post "
            '"https://api.trycloudflare.com/tunnel": i/o timeout\n'
            f"[cloudflared] {public_url}\n"
        )
        monkeypatch.setattr(manager, "_LOG_OFFSET_AT_START", 0)

        assert manager.read_tunnel_url(timeout=1.0) == public_url

    def test_returns_none_on_timeout(self, tmp_path, runtime_paths, monkeypatch):
        log = runtime_paths.log_file
        log.write_text("INFO server up — but no tunnel line ever printed\n")
        monkeypatch.setattr(manager, "_LOG_OFFSET_AT_START", 0)

        assert manager.read_tunnel_url(timeout=0.2, poll_interval=0.05) is None

    def test_ignores_stale_url_before_offset(
        self, tmp_path, runtime_paths, monkeypatch
    ):
        """A URL from a previous session (before the offset) must be skipped;
        only this session's bytes count."""
        stale = "[cloudflared] https://old-stale-url.trycloudflare.com\n"
        log = runtime_paths.log_file
        log.write_text(stale)
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


class TestStartLanggraphDevSpawnFlags:
    """On Windows the langgraph dev child gets its own process group (Ctrl+C
    does not reach it); a lasting one also gets a console of its own, so
    closing the terminal does not end it, while one a session owns keeps the
    terminal's console. On POSIX it gets its own session."""

    @pytest.mark.parametrize(
        ("lasting", "flags"), [(True, 0x08000200), (False, 0x00000200)]
    )
    def test_windows_uses_new_process_group(
        self, spawn_capture, monkeypatch, lasting, flags
    ):
        env = spawn_capture
        monkeypatch.setattr(manager.os, "name", "nt", raising=False)
        monkeypatch.setattr(
            manager.subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200, raising=False
        )
        monkeypatch.setattr(
            manager.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False
        )
        try:
            manager._spawn_server(lasting=lasting)
        except FileNotFoundError:
            pass  # _fake_popen stops before the real spawn
        kw = env.captured["kwargs"]
        assert kw.get("creationflags") == flags
        assert "start_new_session" not in kw

    def test_posix_uses_new_session(self, spawn_capture, monkeypatch):
        env = spawn_capture
        monkeypatch.setattr(manager.os, "name", "posix", raising=False)
        try:
            manager._spawn_server()
        except FileNotFoundError:
            pass
        kw = env.captured["kwargs"]
        assert kw.get("start_new_session") is True
        assert "creationflags" not in kw


class TestWaitForPortBindable:
    def test_free_port_is_bindable(self):
        import socket

        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        assert manager._wait_for_port_bindable(port, "127.0.0.1") is True

    def test_live_listener_is_waited_out_then_reported(self, monkeypatch):
        import socket

        monkeypatch.setattr(manager, "_BIND_WAIT", 0.05)
        monkeypatch.setattr(manager, "_POLL_INTERVAL", 0.01)
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        try:
            port = listener.getsockname()[1]
            assert manager._wait_for_port_bindable(port, "127.0.0.1") is False
        finally:
            listener.close()
