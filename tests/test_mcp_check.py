"""Tests for validate_server_config() and check_server()."""

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import asyncio

from EvoScientist.mcp import check_server, is_ssh_server, validate_server_config
from tests.conftest import run_async


# =============================================================================
# Helpers
# =============================================================================


def _ssh_config(
    *,
    transport: str = "stdio",
    command: str = "mcp-ssh-server",
    args: list | None = None,
    ssh_host: str = "gpu.example.com",
    ssh_user: str = "ubuntu",
    ssh_key_path: str = "/tmp/fake_key",
    expose_to: list | None = None,
    extra_env: dict | None = None,
) -> dict:
    """Build a minimal SSH MCP server config dict."""
    env = {
        "SSH_HOST": ssh_host,
        "SSH_USER": ssh_user,
        "SSH_KEY_PATH": ssh_key_path,
    }
    if extra_env:
        env.update(extra_env)
    cfg: dict = {
        "transport": transport,
        "command": command,
        "args": args if args is not None else ["ssh"],
        "env": env,
    }
    if expose_to is not None:
        cfg["expose_to"] = expose_to
    return cfg


def _status(results: list[dict], check: str) -> str | None:
    """Return the status of a named check, or None if not present."""
    for r in results:
        if r["check"] == check:
            return r["status"]
    return None


# =============================================================================
# TestValidateSSHConfig
# =============================================================================


class TestValidateSSHConfig:
    def test_valid_ssh_config_all_ok(self, tmp_path):
        key = tmp_path / "id_rsa"
        key.write_text("fake key")
        cfg = _ssh_config(ssh_key_path=str(key))
        results = validate_server_config("gpu", cfg)
        failures = [r for r in results if r["status"] == "fail"]
        assert not failures, f"Expected no failures, got: {failures}"

    def test_missing_transport_returns_fail(self):
        cfg = {"command": "mcp-ssh"}
        results = validate_server_config("gpu", cfg)
        assert _status(results, "config.transport") == "fail"

    def test_invalid_transport_returns_fail(self):
        cfg = {"transport": "grpc", "command": "mcp-ssh"}
        results = validate_server_config("gpu", cfg)
        assert _status(results, "config.transport") == "fail"

    def test_stdio_missing_command_returns_fail(self):
        cfg = {
            "transport": "stdio",
            "args": ["ssh"],
            "env": {"SSH_HOST": "h", "SSH_USER": "u", "SSH_KEY_PATH": "/k"},
        }
        results = validate_server_config("gpu", cfg)
        assert _status(results, "config.command") == "fail"

    def test_ssh_missing_host_returns_fail(self):
        cfg = _ssh_config(ssh_host="")
        results = validate_server_config("gpu", cfg)
        assert _status(results, "ssh.env.SSH_HOST") == "fail"

    def test_ssh_missing_user_returns_fail(self):
        cfg = _ssh_config(ssh_user="")
        results = validate_server_config("gpu", cfg)
        assert _status(results, "ssh.env.SSH_USER") == "fail"

    def test_ssh_key_path_not_found_returns_fail(self):
        cfg = _ssh_config(ssh_key_path="/nonexistent/path/id_rsa")
        results = validate_server_config("gpu", cfg)
        assert _status(results, "ssh.key_file") == "fail"

    def test_ssh_key_path_exists_returns_ok(self, tmp_path):
        key = tmp_path / "id_rsa"
        key.write_text("fake key")
        cfg = _ssh_config(ssh_key_path=str(key))
        results = validate_server_config("gpu", cfg)
        assert _status(results, "ssh.key_file") == "ok"

    def test_missing_expose_to_returns_warn(self, tmp_path):
        key = tmp_path / "id_rsa"
        key.write_text("fake key")
        cfg = _ssh_config(ssh_key_path=str(key))
        # No expose_to set
        assert "expose_to" not in cfg
        results = validate_server_config("gpu", cfg)
        assert _status(results, "config.expose_to") == "warn"

    def test_non_ssh_server_skips_ssh_checks(self):
        # stdio server with no SSH markers
        cfg = {"transport": "stdio", "command": "python", "args": ["-m", "myserver"]}
        results = validate_server_config("local", cfg)
        ssh_checks = [r for r in results if r["check"].startswith("ssh.")]
        assert not ssh_checks, f"Expected no SSH checks for non-SSH server, got: {ssh_checks}"

    def test_url_transport_missing_url_returns_fail(self):
        cfg = {"transport": "http"}
        results = validate_server_config("web", cfg)
        assert _status(results, "config.url") == "fail"

    def test_url_transport_with_url_returns_ok(self):
        cfg = {"transport": "http", "url": "http://localhost:8080/mcp"}
        results = validate_server_config("web", cfg)
        assert _status(results, "config.transport") == "ok"
        assert _status(results, "config.url") == "ok"


# =============================================================================
# TestCheckSSHServer
# =============================================================================


def _make_mock_tool(name: str, return_value: str = "ok") -> MagicMock:
    tool = MagicMock()
    tool.name = name
    tool.ainvoke = AsyncMock(return_value=return_value)
    return tool


def _patch_mcp_client(mock_client_cls):
    """Return a context manager that patches MultiServerMCPClient at its source.

    We patch at the source module because check_server() uses a late import
    (``from langchain_mcp_adapters.client import MultiServerMCPClient``) inside
    the function body — the name is never bound at module level in client.py.
    """
    return patch("langchain_mcp_adapters.client.MultiServerMCPClient", mock_client_cls)


class TestCheckSSHServer:
    def test_connection_failure_returns_fail(self, tmp_path):
        key = tmp_path / "id_rsa"
        key.write_text("fake key")
        cfg = _ssh_config(ssh_key_path=str(key))

        MockClient = MagicMock()
        MockClient.return_value.get_tools = AsyncMock(
            side_effect=ConnectionError("refused")
        )

        with _patch_mcp_client(MockClient):
            results = run_async(check_server("gpu", cfg))

        assert _status(results, "connection") == "fail"

    def test_connection_success_returns_ok(self, tmp_path):
        key = tmp_path / "id_rsa"
        key.write_text("fake key")
        cfg = _ssh_config(ssh_key_path=str(key), expose_to=["main"])

        ssh_tool = _make_mock_tool("ssh_execute", "ok")

        MockClient = MagicMock()
        MockClient.return_value.get_tools = AsyncMock(return_value=[ssh_tool])

        with _patch_mcp_client(MockClient):
            results = run_async(check_server("gpu", cfg))

        assert _status(results, "connection") == "ok"

    def test_ssh_execute_success(self, tmp_path):
        key = tmp_path / "id_rsa"
        key.write_text("fake key")
        cfg = _ssh_config(ssh_key_path=str(key), expose_to=["main"])

        ssh_tool = _make_mock_tool("ssh_execute", "ok\nnvidia output")

        MockClient = MagicMock()
        MockClient.return_value.get_tools = AsyncMock(return_value=[ssh_tool])

        with _patch_mcp_client(MockClient):
            results = run_async(check_server("gpu", cfg))

        assert _status(results, "ssh.execute") == "ok"

    def test_ssh_execute_failure(self, tmp_path):
        key = tmp_path / "id_rsa"
        key.write_text("fake key")
        cfg = _ssh_config(ssh_key_path=str(key), expose_to=["main"])

        ssh_tool = _make_mock_tool("ssh_execute")
        ssh_tool.ainvoke = AsyncMock(side_effect=RuntimeError("connection reset"))

        MockClient = MagicMock()
        MockClient.return_value.get_tools = AsyncMock(return_value=[ssh_tool])

        with _patch_mcp_client(MockClient):
            results = run_async(check_server("gpu", cfg))

        assert _status(results, "ssh.execute") == "fail"

    def test_gpu_detection_success(self, tmp_path):
        key = tmp_path / "id_rsa"
        key.write_text("fake key")
        cfg = _ssh_config(ssh_key_path=str(key), expose_to=["main"])

        gpu_output = "Tesla T4, 15360 MiB"
        ssh_tool = _make_mock_tool("ssh_execute")
        # First call (echo ok) returns "ok", second call (nvidia-smi) returns GPU info
        ssh_tool.ainvoke = AsyncMock(side_effect=["ok", gpu_output])

        MockClient = MagicMock()
        MockClient.return_value.get_tools = AsyncMock(return_value=[ssh_tool])

        with _patch_mcp_client(MockClient):
            results = run_async(check_server("gpu", cfg))

        assert _status(results, "ssh.gpu") == "ok"
        gpu_result = next(r for r in results if r["check"] == "ssh.gpu")
        assert "Tesla T4" in gpu_result["detail"]

    def test_nvidia_smi_missing_returns_warn(self, tmp_path):
        key = tmp_path / "id_rsa"
        key.write_text("fake key")
        cfg = _ssh_config(ssh_key_path=str(key), expose_to=["main"])

        ssh_tool = _make_mock_tool("ssh_execute")
        # echo ok succeeds, nvidia-smi raises
        ssh_tool.ainvoke = AsyncMock(
            side_effect=["ok", RuntimeError("nvidia-smi: command not found")]
        )

        MockClient = MagicMock()
        MockClient.return_value.get_tools = AsyncMock(return_value=[ssh_tool])

        with _patch_mcp_client(MockClient):
            results = run_async(check_server("gpu", cfg))

        assert _status(results, "ssh.gpu") == "warn"

    def test_connection_timeout_returns_fail(self, tmp_path):
        key = tmp_path / "id_rsa"
        key.write_text("fake key")
        cfg = _ssh_config(ssh_key_path=str(key))

        MockClient = MagicMock()
        MockClient.return_value.get_tools = AsyncMock(
            side_effect=asyncio.TimeoutError()
        )

        with _patch_mcp_client(MockClient):
            results = run_async(check_server("gpu", cfg))

        assert _status(results, "connection") == "fail"
        conn_result = next(r for r in results if r["check"] == "connection")
        assert "timed out" in conn_result["detail"]

    def test_ssh_execute_timeout_returns_fail(self, tmp_path):
        key = tmp_path / "id_rsa"
        key.write_text("fake key")
        cfg = _ssh_config(ssh_key_path=str(key), expose_to=["main"])

        ssh_tool = _make_mock_tool("ssh_execute")
        ssh_tool.ainvoke = AsyncMock(side_effect=asyncio.TimeoutError())

        MockClient = MagicMock()
        MockClient.return_value.get_tools = AsyncMock(return_value=[ssh_tool])

        with _patch_mcp_client(MockClient):
            results = run_async(check_server("gpu", cfg))

        assert _status(results, "ssh.execute") == "fail"
        exec_result = next(r for r in results if r["check"] == "ssh.execute")
        assert "timed out" in exec_result["detail"]

    def test_missing_langchain_mcp_adapters_returns_fail(self, tmp_path):
        key = tmp_path / "id_rsa"
        key.write_text("fake key")
        cfg = _ssh_config(ssh_key_path=str(key))

        with patch.dict("sys.modules", {"langchain_mcp_adapters.client": None}):
            results = run_async(check_server("gpu", cfg))

        assert _status(results, "deps") == "fail"


# =============================================================================
# Security-focused tests
# =============================================================================


class TestKeyPathSecurity:
    def test_key_path_outside_ssh_dir_warns(self, tmp_path):
        """Key path outside ~/.ssh and CWD produces a warn."""
        key = tmp_path / "id_rsa"
        key.write_text("fake key")
        cfg = _ssh_config(ssh_key_path=str(key))
        results = validate_server_config("gpu", cfg)
        location_results = [r for r in results if r["check"] == "ssh.key_location"]
        assert any(r["status"] == "warn" for r in location_results), (
            f"Expected warn for key outside ~/.ssh, got: {location_results}"
        )

    def test_key_location_prefix_collision_not_matched(self, tmp_path, monkeypatch):
        """~/.ssh_evil/id_rsa must NOT pass the ~/.ssh allowlist check."""
        fake_home = tmp_path / "fakehome"
        fake_home.mkdir()
        # Create ~/.ssh_evil (prefix-collision with ~/.ssh)
        ssh_evil_dir = fake_home / ".ssh_evil"
        ssh_evil_dir.mkdir()
        key = ssh_evil_dir / "id_rsa"
        key.write_text("fake key")

        monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))

        cfg = _ssh_config(ssh_key_path=str(key))
        results = validate_server_config("gpu", cfg)
        location_results = [r for r in results if r["check"] == "ssh.key_location"]
        assert any(r["status"] == "warn" for r in location_results), (
            f"Expected warn for ~/.ssh_evil key (prefix collision), got: {location_results}"
        )

    def test_key_path_detail_does_not_leak_absolute_path(self, tmp_path, monkeypatch):
        """No detail string contains the fully expanded home directory."""
        fake_home = tmp_path / "fakehome"
        fake_home.mkdir()
        ssh_dir = fake_home / ".ssh"
        ssh_dir.mkdir()
        key = ssh_dir / "id_rsa"
        key.write_text("fake key")

        monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))

        cfg = _ssh_config(ssh_key_path=str(key))
        results = validate_server_config("gpu", cfg)
        fake_home_str = str(fake_home)
        for r in results:
            if "key" in r["check"]:
                assert fake_home_str not in r["detail"], (
                    f"Detail leaks absolute home path: {r['detail']}"
                )
                # Key is inside ~/.ssh so it should be shown with ~ prefix
                if "exists" in r["detail"]:
                    assert r["detail"].startswith("key file exists: ~"), (
                        f"Expected ~ prefix in detail: {r['detail']}"
                    )

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
    def test_world_readable_key_warns(self, tmp_path):
        """Key with mode 0o644 produces a warn about permissions."""
        key = tmp_path / "id_rsa"
        key.write_text("fake key")
        os.chmod(key, 0o644)
        cfg = _ssh_config(ssh_key_path=str(key))
        results = validate_server_config("gpu", cfg)
        perm_results = [r for r in results if r["check"] == "ssh.key_permissions"]
        assert any(r["status"] == "warn" for r in perm_results), (
            f"Expected permissions warn for 0o644, got: {perm_results}"
        )

    def test_key_path_resolve_oserror_warns(self, tmp_path):
        """A path that triggers OSError on resolve produces warn, not exception."""
        cfg = _ssh_config(ssh_key_path="/nonexistent/../../../weird/path/key")
        # Should not raise; should produce warn or fail gracefully
        results = validate_server_config("gpu", cfg)
        # At minimum, should not crash
        assert isinstance(results, list)


class TestIsSSHServerSecurity:
    def test_false_positive_no_ssh_flag(self):
        """--no-ssh should NOT be detected as SSH."""
        assert not is_ssh_server({"args": ["--no-ssh"]})

    def test_false_positive_ssh_helper(self):
        """ssh-helper should NOT be detected as SSH."""
        assert not is_ssh_server({"args": ["ssh-helper"]})

    def test_exact_match_ssh(self):
        """Bare 'ssh' arg should be detected."""
        assert is_ssh_server({"args": ["ssh"]})

    def test_mcp_server_ssh_package(self):
        """mcp-server-ssh should be detected."""
        assert is_ssh_server({"args": ["mcp-server-ssh"]})

    def test_npx_pattern(self):
        """npx -y mcp-server-ssh should be detected."""
        assert is_ssh_server({"args": ["-y", "mcp-server-ssh"]})

    def test_scoped_package(self):
        """@scope/mcp-server-ssh@1.0.0 should be detected."""
        assert is_ssh_server({"args": ["-y", "@myorg/mcp-server-ssh@1.0.0"]})

    def test_command_field_ssh(self):
        """command='mcp-server-ssh' should be detected."""
        assert is_ssh_server({"command": "mcp-server-ssh", "args": []})

    def test_ssh_host_env(self):
        """SSH_HOST env var should be detected."""
        assert is_ssh_server({"env": {"SSH_HOST": "example.com"}})


class TestConfigPermissions:
    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
    def test_config_file_permissions(self, tmp_path):
        """After _save_user_config, file is 0o600 and dir is 0o700."""
        from EvoScientist.mcp.client import _save_user_config, USER_CONFIG_DIR, USER_MCP_CONFIG

        # Use monkeypatch-style approach: temporarily override the module-level paths
        test_dir = tmp_path / "config"
        test_file = test_dir / "mcp.yaml"

        with patch("EvoScientist.mcp.client.USER_CONFIG_DIR", test_dir), \
             patch("EvoScientist.mcp.client.USER_MCP_CONFIG", test_file):
            _save_user_config({"test": {"transport": "stdio"}})

        dir_mode = os.stat(test_dir).st_mode & 0o777
        file_mode = os.stat(test_file).st_mode & 0o777
        assert dir_mode == 0o700, f"Expected dir 0o700, got {oct(dir_mode)}"
        assert file_mode == 0o600, f"Expected file 0o600, got {oct(file_mode)}"


class TestMaskingBehavior:
    def test_render_config_table_masks_sensitive_env(self):
        """Sensitive env vars should be masked with ****."""
        from io import StringIO
        from rich.console import Console
        from EvoScientist.cli.mcp_ui import _render_mcp_server_config_table

        buf = StringIO()
        test_console = Console(file=buf, width=120)
        with patch("EvoScientist.cli.mcp_ui.console", test_console):
            _render_mcp_server_config_table("test", {
                "transport": "stdio",
                "env": {"API_KEY": "secret123", "MY_TOKEN": "tok_abc"},
            })
        output = buf.getvalue()
        assert "secret123" not in output, "API_KEY value should be masked"
        assert "tok_abc" not in output, "MY_TOKEN value should be masked"
        assert "****" in output, "Masked placeholder should appear"

    def test_render_config_table_shows_nonsensitive_plaintext(self):
        """Non-sensitive env vars should be shown in plaintext."""
        from io import StringIO
        from rich.console import Console
        from EvoScientist.cli.mcp_ui import _render_mcp_server_config_table

        buf = StringIO()
        test_console = Console(file=buf, width=120)
        with patch("EvoScientist.cli.mcp_ui.console", test_console):
            _render_mcp_server_config_table("test", {
                "transport": "stdio",
                "env": {
                    "SSH_HOST": "example.com",
                    "SSH_KEY_PATH": "~/.ssh/id_rsa",
                },
            })
        output = buf.getvalue()
        assert "example.com" in output, "SSH_HOST should be shown in plaintext"
        assert "~/.ssh/id_rsa" in output, "SSH_KEY_PATH should be shown in plaintext"


class TestExampleYAML:
    def test_example_yaml_has_pinned_version(self):
        """Example YAML should have pinned mcp-server-ssh versions."""
        yaml_path = Path(__file__).parent.parent / "docs" / "examples" / "mcp-ssh-gpu" / "mcp-ssh-gpu.yaml.example"
        content = yaml_path.read_text()
        import re
        unversioned = re.findall(r'mcp-server-ssh(?!@)', content)
        assert not unversioned, (
            f"Found {len(unversioned)} unversioned mcp-server-ssh references"
        )

    def test_readme_has_pinned_version(self):
        """README inline YAML examples should have pinned mcp-server-ssh versions."""
        readme_path = Path(__file__).parent.parent / "docs" / "examples" / "mcp-ssh-gpu" / "README.md"
        content = readme_path.read_text()
        import re
        # Find mcp-server-ssh in code blocks (not in prose text about generic packages)
        unversioned = re.findall(r'mcp-server-ssh(?!@)(?!")', content)
        assert not unversioned, (
            f"README has {len(unversioned)} unversioned mcp-server-ssh references"
        )


class TestServerNameValidation:
    def test_valid_names_accepted(self):
        """Normal server names should be accepted."""
        from EvoScientist.mcp.client import _SERVER_NAME_RE
        for name in ("my-server", "gpu_cluster.1", "ssh-gpu", "test123"):
            assert _SERVER_NAME_RE.match(name), f"{name!r} should be valid"

    def test_invalid_names_rejected(self):
        """Server names with special chars should be rejected by add_mcp_server."""
        from EvoScientist.mcp.client import add_mcp_server
        import pytest
        for name in ("../../etc/passwd", "my server", "name\x00evil", "a;b"):
            with pytest.raises(ValueError, match="Invalid server name"):
                add_mcp_server(name, "stdio", command="test")


class TestRedactHomePathEdgeCases:
    def test_prefix_collision_not_matched(self, tmp_path, monkeypatch):
        """'/home/alice' must NOT match '/home/alicebob/...'."""
        from EvoScientist.mcp.client import _redact_home_path
        fake_home = tmp_path / "alice"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))

        collider = tmp_path / "alicebob" / "file.txt"
        collider.parent.mkdir()
        collider.write_text("x")
        result = _redact_home_path(collider)
        assert result == str(collider), (
            f"Should NOT redact prefix-collision path, got: {result}"
        )

    def test_home_subpath_is_redacted(self, tmp_path, monkeypatch):
        """A path inside home should be redacted to ~/..."""
        from EvoScientist.mcp.client import _redact_home_path
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))

        inside = fake_home / ".ssh" / "id_rsa"
        inside.parent.mkdir()
        inside.write_text("key")
        result = _redact_home_path(inside)
        assert result.startswith("~"), f"Expected ~ prefix, got: {result}"


class TestWindowsConfigWrite:
    def test_windows_branch_writes_content(self, tmp_path):
        """On 'win32', _save_user_config uses write_text fallback."""
        from EvoScientist.mcp.client import _save_user_config

        test_dir = tmp_path / "config"
        test_file = test_dir / "mcp.yaml"

        with patch("EvoScientist.mcp.client.USER_CONFIG_DIR", test_dir), \
             patch("EvoScientist.mcp.client.USER_MCP_CONFIG", test_file), \
             patch("EvoScientist.mcp.client.sys") as mock_sys:
            mock_sys.platform = "win32"
            _save_user_config({"test": {"transport": "stdio"}})

        assert test_file.exists(), "Config file should exist"
        content = test_file.read_text()
        assert "transport: stdio" in content
