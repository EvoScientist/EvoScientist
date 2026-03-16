"""Tests for validate_ssh_config() and check_ssh_server()."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from EvoScientist.mcp.client import check_ssh_server, validate_ssh_config
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
        results = validate_ssh_config("gpu", cfg)
        failures = [r for r in results if r["status"] == "fail"]
        assert not failures, f"Expected no failures, got: {failures}"

    def test_missing_transport_returns_fail(self):
        cfg = {"command": "mcp-ssh"}
        results = validate_ssh_config("gpu", cfg)
        assert _status(results, "config.transport") == "fail"

    def test_invalid_transport_returns_fail(self):
        cfg = {"transport": "grpc", "command": "mcp-ssh"}
        results = validate_ssh_config("gpu", cfg)
        assert _status(results, "config.transport") == "fail"

    def test_stdio_missing_command_returns_fail(self):
        cfg = {
            "transport": "stdio",
            "args": ["ssh"],
            "env": {"SSH_HOST": "h", "SSH_USER": "u", "SSH_KEY_PATH": "/k"},
        }
        results = validate_ssh_config("gpu", cfg)
        assert _status(results, "config.command") == "fail"

    def test_ssh_missing_host_returns_fail(self):
        cfg = _ssh_config(ssh_host="")
        results = validate_ssh_config("gpu", cfg)
        assert _status(results, "ssh.env.SSH_HOST") == "fail"

    def test_ssh_missing_user_returns_fail(self):
        cfg = _ssh_config(ssh_user="")
        results = validate_ssh_config("gpu", cfg)
        assert _status(results, "ssh.env.SSH_USER") == "fail"

    def test_ssh_key_path_not_found_returns_fail(self):
        cfg = _ssh_config(ssh_key_path="/nonexistent/path/id_rsa")
        results = validate_ssh_config("gpu", cfg)
        assert _status(results, "ssh.key_file") == "fail"

    def test_ssh_key_path_exists_returns_ok(self, tmp_path):
        key = tmp_path / "id_rsa"
        key.write_text("fake key")
        cfg = _ssh_config(ssh_key_path=str(key))
        results = validate_ssh_config("gpu", cfg)
        assert _status(results, "ssh.key_file") == "ok"

    def test_missing_expose_to_returns_warn(self, tmp_path):
        key = tmp_path / "id_rsa"
        key.write_text("fake key")
        cfg = _ssh_config(ssh_key_path=str(key))
        # No expose_to set
        assert "expose_to" not in cfg
        results = validate_ssh_config("gpu", cfg)
        assert _status(results, "config.expose_to") == "warn"

    def test_non_ssh_server_skips_ssh_checks(self):
        # stdio server with no SSH markers
        cfg = {"transport": "stdio", "command": "python", "args": ["-m", "myserver"]}
        results = validate_ssh_config("local", cfg)
        ssh_checks = [r for r in results if r["check"].startswith("ssh.")]
        assert not ssh_checks, f"Expected no SSH checks for non-SSH server, got: {ssh_checks}"

    def test_url_transport_missing_url_returns_fail(self):
        cfg = {"transport": "http"}
        results = validate_ssh_config("web", cfg)
        assert _status(results, "config.url") == "fail"

    def test_url_transport_with_url_returns_ok(self):
        cfg = {"transport": "http", "url": "http://localhost:8080/mcp"}
        results = validate_ssh_config("web", cfg)
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
    """Return a context manager that patches MultiServerMCPClient at its source."""
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
            results = run_async(check_ssh_server("gpu", cfg))

        assert _status(results, "connection") == "fail"

    def test_connection_success_returns_ok(self, tmp_path):
        key = tmp_path / "id_rsa"
        key.write_text("fake key")
        cfg = _ssh_config(ssh_key_path=str(key), expose_to=["main"])

        ssh_tool = _make_mock_tool("ssh_execute", "ok")

        MockClient = MagicMock()
        MockClient.return_value.get_tools = AsyncMock(return_value=[ssh_tool])

        with _patch_mcp_client(MockClient):
            results = run_async(check_ssh_server("gpu", cfg))

        assert _status(results, "connection") == "ok"

    def test_ssh_execute_success(self, tmp_path):
        key = tmp_path / "id_rsa"
        key.write_text("fake key")
        cfg = _ssh_config(ssh_key_path=str(key), expose_to=["main"])

        ssh_tool = _make_mock_tool("ssh_execute", "ok\nnvidia output")

        MockClient = MagicMock()
        MockClient.return_value.get_tools = AsyncMock(return_value=[ssh_tool])

        with _patch_mcp_client(MockClient):
            results = run_async(check_ssh_server("gpu", cfg))

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
            results = run_async(check_ssh_server("gpu", cfg))

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
            results = run_async(check_ssh_server("gpu", cfg))

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
            results = run_async(check_ssh_server("gpu", cfg))

        assert _status(results, "ssh.gpu") == "warn"

    def test_missing_langchain_mcp_adapters_returns_fail(self, tmp_path):
        key = tmp_path / "id_rsa"
        key.write_text("fake key")
        cfg = _ssh_config(ssh_key_path=str(key))

        with patch.dict("sys.modules", {"langchain_mcp_adapters.client": None}):
            results = run_async(check_ssh_server("gpu", cfg))

        assert _status(results, "deps") == "fail"
