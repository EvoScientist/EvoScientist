"""Tests for EvoScientist.mcp module."""

import textwrap
from types import SimpleNamespace

import pytest
import yaml

from EvoScientist.mcp.client import (
    _build_connections,
    _filter_tools,
    _interpolate_env,
    _resolve_command,
    _route_tools,
    add_mcp_server,
    edit_mcp_server,
    load_mcp_config,
    parse_mcp_add_args,
    parse_mcp_edit_args,
    remove_mcp_server,
)

# ---- _interpolate_env ----


class TestInterpolateEnv:
    def test_substitutes_env_var(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "secret123")
        assert _interpolate_env("Bearer ${MY_KEY}") == "Bearer secret123"

    def test_multiple_vars(self, monkeypatch):
        monkeypatch.setenv("HOST", "localhost")
        monkeypatch.setenv("PORT", "8080")
        assert _interpolate_env("${HOST}:${PORT}") == "localhost:8080"

    def test_missing_var_returns_empty(self, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_VAR_XYZ", raising=False)
        assert _interpolate_env("${NONEXISTENT_VAR_XYZ}") == ""

    def test_no_vars_unchanged(self):
        assert _interpolate_env("plain text") == "plain text"

    def test_empty_string(self):
        assert _interpolate_env("") == ""


# ---- load_mcp_config ----


@pytest.fixture
def mcp_config_file(monkeypatch, tmp_path):
    """Point USER_MCP_CONFIG to a temp file for isolated testing."""
    cfg = tmp_path / "mcp.yaml"
    monkeypatch.setattr("EvoScientist.mcp.client.USER_MCP_CONFIG", cfg)
    return cfg


class TestLoadMcpConfig:
    def test_missing_file_returns_empty(self, mcp_config_file):
        # File doesn't exist yet
        assert load_mcp_config() == {}

    def test_valid_file_parses(self, mcp_config_file):
        mcp_config_file.write_text(
            textwrap.dedent("""\
            my-server:
              transport: stdio
              command: echo
              args: ["hello"]
        """)
        )
        result = load_mcp_config()
        assert "my-server" in result
        assert result["my-server"]["transport"] == "stdio"

    def test_empty_file_returns_empty(self, mcp_config_file):
        mcp_config_file.write_text("")
        assert load_mcp_config() == {}

    def test_comments_only_returns_empty(self, mcp_config_file):
        mcp_config_file.write_text("# just a comment\n# another comment\n")
        assert load_mcp_config() == {}

    def test_env_var_interpolation(self, mcp_config_file, monkeypatch):
        monkeypatch.setenv("TEST_TOKEN", "tok_abc")
        mcp_config_file.write_text(
            textwrap.dedent("""\
            my-server:
              transport: http
              url: "http://localhost:8080/mcp"
              headers:
                Authorization: "Bearer ${TEST_TOKEN}"
        """)
        )
        result = load_mcp_config()
        assert result["my-server"]["headers"]["Authorization"] == "Bearer tok_abc"


# ---- _build_connections ----


# ---- _resolve_command ----


class TestResolveCommand:
    def test_absolute_path_returned_as_is(self, tmp_path):
        """Absolute paths are never modified, even if the file doesn't exist."""
        fake = str(tmp_path / "mytool")
        assert _resolve_command(fake) == fake

    def test_found_on_path(self):
        """Commands found via shutil.which are returned as full paths."""
        result = _resolve_command("python")
        assert result.endswith("python") or result.endswith("python3")
        assert result != "python"  # resolved, not the bare name

    def test_found_in_python_bin(self, tmp_path, monkeypatch):
        """Falls back to sys.executable's directory when not in PATH."""

        # Create a fake executable next to sys.executable
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        fake_exe = bin_dir / "my-mcp-tool"
        fake_exe.write_text("#!/bin/sh\n")
        fake_exe.chmod(0o755)

        monkeypatch.setattr("shutil.which", lambda _: None)
        monkeypatch.setattr(
            "EvoScientist.mcp.client.sys.executable", str(bin_dir / "python")
        )

        assert _resolve_command("my-mcp-tool") == str(fake_exe)

    def test_not_found_returns_original(self, monkeypatch):
        """Returns the original command when not found anywhere (let OS report the error)."""
        monkeypatch.setattr("shutil.which", lambda _: None)
        monkeypatch.setattr(
            "EvoScientist.mcp.client.sys.executable", "/nonexistent/bin/python"
        )
        assert _resolve_command("unknown-tool-xyz") == "unknown-tool-xyz"

    def test_build_connections_resolves_command(self, monkeypatch):
        """_build_connections uses _resolve_command so the full path appears in output."""
        monkeypatch.setattr(
            "EvoScientist.mcp.client._resolve_command", lambda cmd: f"/resolved/{cmd}"
        )
        config = {"srv": {"transport": "stdio", "command": "mytool", "args": []}}
        conns = _build_connections(config)
        assert conns["srv"]["command"] == "/resolved/mytool"


class TestBuildConnections:
    def test_stdio_connection(self):
        config = {
            "fs": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "server"],
            }
        }
        conns = _build_connections(config)
        assert "fs" in conns
        assert conns["fs"]["transport"] == "stdio"
        assert conns["fs"]["command"].endswith("npx")
        assert conns["fs"]["args"] == ["-y", "server"]

    def test_stdio_with_env(self, monkeypatch):
        for var in (
            "http_proxy",
            "https_proxy",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "all_proxy",
            "ALL_PROXY",
            "no_proxy",
            "NO_PROXY",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
            "REQUESTS_CA_BUNDLE",
            "CURL_CA_BUNDLE",
            "NODE_EXTRA_CA_CERTS",
        ):
            monkeypatch.delenv(var, raising=False)
        config = {
            "fs": {
                "transport": "stdio",
                "command": "npx",
                "args": [],
                "env": {"FOO": "bar"},
            }
        }
        conns = _build_connections(config)
        assert conns["fs"]["env"] == {"FOO": "bar"}

    def test_stdio_forwards_proxy_and_cert_env(self, monkeypatch):
        """Proxy and CA bundle env vars are forwarded to stdio subprocesses."""
        monkeypatch.setenv("https_proxy", "http://proxy:3128")
        monkeypatch.setenv("SSL_CERT_FILE", "/etc/ssl/certs/ca.crt")
        config = {"fs": {"transport": "stdio", "command": "npx", "args": []}}
        conns = _build_connections(config)
        assert conns["fs"]["env"]["https_proxy"] == "http://proxy:3128"
        assert conns["fs"]["env"]["SSL_CERT_FILE"] == "/etc/ssl/certs/ca.crt"

    def test_stdio_user_env_overrides_forwarded(self, monkeypatch):
        """User-configured env takes precedence over auto-forwarded values."""
        monkeypatch.setenv("https_proxy", "http://host-proxy:3128")
        config = {
            "fs": {
                "transport": "stdio",
                "command": "npx",
                "args": [],
                "env": {"https_proxy": "http://user-proxy:9000"},
            }
        }
        conns = _build_connections(config)
        assert conns["fs"]["env"]["https_proxy"] == "http://user-proxy:9000"

    def test_http_connection(self):
        config = {
            "api": {
                "transport": "http",
                "url": "http://localhost:8080/mcp",
                "headers": {"Authorization": "Bearer xxx"},
            }
        }
        conns = _build_connections(config)
        assert conns["api"]["transport"] == "http"
        assert conns["api"]["url"] == "http://localhost:8080/mcp"
        assert conns["api"]["headers"]["Authorization"] == "Bearer xxx"

    def test_sse_connection(self):
        config = {
            "sse-srv": {
                "transport": "sse",
                "url": "http://localhost:9090/sse",
            }
        }
        conns = _build_connections(config)
        assert conns["sse-srv"]["transport"] == "sse"
        assert conns["sse-srv"]["url"] == "http://localhost:9090/sse"

    def test_websocket_connection(self):
        config = {
            "ws": {
                "transport": "websocket",
                "url": "ws://localhost:8765",
            }
        }
        conns = _build_connections(config)
        assert conns["ws"]["transport"] == "websocket"

    def test_unknown_transport_skipped(self):
        config = {
            "bad": {
                "transport": "carrier_pigeon",
                "url": "coo://rooftop",
            }
        }
        conns = _build_connections(config)
        assert conns == {}

    def test_mixed_transports(self):
        config = {
            "a": {"transport": "stdio", "command": "cmd", "args": []},
            "b": {"transport": "http", "url": "http://x"},
            "c": {"transport": "unknown"},
        }
        conns = _build_connections(config)
        assert set(conns.keys()) == {"a", "b"}


# ---- _filter_tools ----


def _make_tool(name: str):
    """Create a minimal mock tool with a .name attribute."""
    return SimpleNamespace(name=name)


class TestFilterTools:
    def test_none_allowlist_passes_all(self):
        tools = [_make_tool("a"), _make_tool("b"), _make_tool("c")]
        assert _filter_tools(tools, None) == tools

    def test_allowlist_filters(self):
        tools = [_make_tool("a"), _make_tool("b"), _make_tool("c")]
        result = _filter_tools(tools, ["a", "c"])
        assert [t.name for t in result] == ["a", "c"]

    def test_empty_allowlist_filters_all(self):
        tools = [_make_tool("a"), _make_tool("b")]
        assert _filter_tools(tools, []) == []

    def test_allowlist_with_nonexistent_name(self):
        tools = [_make_tool("a")]
        result = _filter_tools(tools, ["a", "nonexistent"])
        assert [t.name for t in result] == ["a"]

    def test_empty_tools_list(self):
        assert _filter_tools([], ["a"]) == []
        assert _filter_tools([], None) == []

    # Wildcard tests

    def test_wildcard_star_suffix(self):
        """Test *_exa pattern matching."""
        tools = [
            _make_tool("web_search_exa"),
            _make_tool("get_code_context_exa"),
            _make_tool("company_research_exa"),
            _make_tool("unrelated_tool"),
        ]
        result = _filter_tools(tools, ["*_exa"])
        assert [t.name for t in result] == [
            "web_search_exa",
            "get_code_context_exa",
            "company_research_exa",
        ]

    def test_wildcard_star_prefix(self):
        """Test read_* pattern matching."""
        tools = [
            _make_tool("read_file"),
            _make_tool("read_directory"),
            _make_tool("read_link"),
            _make_tool("write_file"),
        ]
        result = _filter_tools(tools, ["read_*"])
        assert [t.name for t in result] == [
            "read_file",
            "read_directory",
            "read_link",
        ]

    def test_wildcard_star_middle(self):
        """Test pattern with * in the middle."""
        tools = [
            _make_tool("get_user_data"),
            _make_tool("get_admin_data"),
            _make_tool("get_file"),
        ]
        result = _filter_tools(tools, ["get_*_data"])
        assert [t.name for t in result] == ["get_user_data", "get_admin_data"]

    def test_wildcard_star_only(self):
        """Test * matches everything."""
        tools = [_make_tool("a"), _make_tool("b"), _make_tool("c")]
        result = _filter_tools(tools, ["*"])
        assert [t.name for t in result] == ["a", "b", "c"]

    def test_wildcard_question_mark(self):
        """Test ? matches single character."""
        tools = [
            _make_tool("tool_1"),
            _make_tool("tool_2"),
            _make_tool("tool_10"),
        ]
        result = _filter_tools(tools, ["tool_?"])
        assert [t.name for t in result] == ["tool_1", "tool_2"]

    def test_wildcard_character_class(self):
        """Test [seq] matches characters in sequence."""
        tools = [
            _make_tool("tool_a"),
            _make_tool("tool_b"),
            _make_tool("tool_c"),
            _make_tool("tool_d"),
        ]
        result = _filter_tools(tools, ["tool_[abc]"])
        assert [t.name for t in result] == ["tool_a", "tool_b", "tool_c"]

    def test_wildcard_character_class_range(self):
        """Test [0-9] matches digit range."""
        tools = [
            _make_tool("tool_0"),
            _make_tool("tool_5"),
            _make_tool("tool_9"),
            _make_tool("tool_a"),
        ]
        result = _filter_tools(tools, ["tool_[0-9]"])
        assert [t.name for t in result] == ["tool_0", "tool_5", "tool_9"]

    def test_wildcard_negated_character_class(self):
        """Test [!seq] matches characters not in sequence."""
        tools = [
            _make_tool("tool_a"),
            _make_tool("tool_b"),
            _make_tool("tool_1"),
            _make_tool("tool_2"),
        ]
        result = _filter_tools(tools, ["tool_[!0-9]"])
        assert [t.name for t in result] == ["tool_a", "tool_b"]

    def test_wildcard_mixed_with_exact(self):
        """Test mixing wildcard and exact patterns."""
        tools = [
            _make_tool("web_search_exa"),
            _make_tool("get_code_context_exa"),
            _make_tool("specific_tool"),
            _make_tool("another_tool"),
        ]
        result = _filter_tools(tools, ["*_exa", "specific_tool"])
        assert [t.name for t in result] == [
            "web_search_exa",
            "get_code_context_exa",
            "specific_tool",
        ]

    def test_wildcard_multiple_patterns(self):
        """Test multiple wildcard patterns."""
        tools = [
            _make_tool("read_file"),
            _make_tool("write_file"),
            _make_tool("delete_file"),
            _make_tool("search_database"),
        ]
        result = _filter_tools(tools, ["read_*", "write_*"])
        assert [t.name for t in result] == ["read_file", "write_file"]

    def test_wildcard_no_match(self):
        """Test wildcard pattern that doesn't match anything."""
        tools = [_make_tool("foo"), _make_tool("bar")]
        result = _filter_tools(tools, ["baz_*"])
        assert result == []

    def test_wildcard_overlapping_patterns(self):
        """Test overlapping patterns don't duplicate results."""
        tools = [_make_tool("tool_abc"), _make_tool("tool_xyz")]
        result = _filter_tools(tools, ["tool_*", "*_abc"])
        # Should include each tool only once
        assert [t.name for t in result] == ["tool_abc", "tool_xyz"]

    def test_wildcard_complex_pattern(self):
        """Test complex wildcard pattern."""
        tools = [
            _make_tool("get_user_info_v1"),
            _make_tool("get_user_info_v2"),
            _make_tool("get_admin_info_v1"),
            _make_tool("set_user_info"),
        ]
        result = _filter_tools(tools, ["get_*_info_v?"])
        assert [t.name for t in result] == [
            "get_user_info_v1",
            "get_user_info_v2",
            "get_admin_info_v1",
        ]

    def test_exact_match_performance_path(self):
        """Test exact matching still works (fast path)."""
        # This test verifies backward compatibility with exact matching
        tools = [_make_tool("a"), _make_tool("b"), _make_tool("c")]
        result = _filter_tools(tools, ["a", "c"])
        assert [t.name for t in result] == ["a", "c"]


# ---- _route_tools ----


class TestRouteTools:
    def test_default_routes_to_main(self):
        config = {"srv": {"transport": "stdio"}}
        server_tools = {"srv": [_make_tool("x")]}
        result = _route_tools(config, server_tools)
        assert "main" in result
        assert [t.name for t in result["main"]] == ["x"]

    def test_expose_to_named_agent(self):
        config = {"srv": {"transport": "stdio", "expose_to": ["code-agent"]}}
        server_tools = {"srv": [_make_tool("x"), _make_tool("y")]}
        result = _route_tools(config, server_tools)
        assert "code-agent" in result
        assert "main" not in result
        assert [t.name for t in result["code-agent"]] == ["x", "y"]

    def test_expose_to_multiple_agents(self):
        config = {"srv": {"transport": "stdio", "expose_to": ["main", "code-agent"]}}
        server_tools = {"srv": [_make_tool("x")]}
        result = _route_tools(config, server_tools)
        assert [t.name for t in result["main"]] == ["x"]
        assert [t.name for t in result["code-agent"]] == ["x"]

    def test_tool_filter_applied(self):
        config = {"srv": {"transport": "stdio", "tools": ["b"]}}
        server_tools = {"srv": [_make_tool("a"), _make_tool("b"), _make_tool("c")]}
        result = _route_tools(config, server_tools)
        assert [t.name for t in result["main"]] == ["b"]

    def test_multiple_servers(self):
        config = {
            "s1": {"transport": "stdio", "expose_to": ["main"]},
            "s2": {"transport": "http", "expose_to": ["research-agent"]},
        }
        server_tools = {
            "s1": [_make_tool("a")],
            "s2": [_make_tool("b")],
        }
        result = _route_tools(config, server_tools)
        assert [t.name for t in result["main"]] == ["a"]
        assert [t.name for t in result["research-agent"]] == ["b"]

    def test_expose_to_string_not_list(self):
        config = {"srv": {"transport": "stdio", "expose_to": "debug-agent"}}
        server_tools = {"srv": [_make_tool("x")]}
        result = _route_tools(config, server_tools)
        assert "debug-agent" in result

    def test_empty_server_tools(self):
        config = {"srv": {"transport": "stdio"}}
        server_tools = {"srv": []}
        result = _route_tools(config, server_tools)
        assert result.get("main", []) == []


# ---- add_mcp_server / remove_mcp_server ----


@pytest.fixture
def user_mcp_dir(tmp_path, monkeypatch):
    """Redirect user MCP config to a temp directory."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    cfg_file = cfg_dir / "mcp.yaml"
    monkeypatch.setattr("EvoScientist.mcp.client.USER_CONFIG_DIR", cfg_dir)
    monkeypatch.setattr("EvoScientist.mcp.client.USER_MCP_CONFIG", cfg_file)
    return cfg_file


class TestAddMcpServer:
    def test_add_stdio_server(self, user_mcp_dir):
        entry = add_mcp_server(
            "fs", "stdio", command="npx", args=["-y", "server", "/tmp"]
        )
        assert entry["transport"] == "stdio"
        assert entry["command"] == "npx"
        assert entry["args"] == ["-y", "server", "/tmp"]
        # Verify persisted
        data = yaml.safe_load(user_mcp_dir.read_text())
        assert "fs" in data

    def test_add_http_server(self, user_mcp_dir):
        entry = add_mcp_server(
            "api",
            "http",
            url="http://localhost:8080/mcp",
            headers={"Authorization": "Bearer tok"},
        )
        assert entry["url"] == "http://localhost:8080/mcp"
        assert entry["headers"]["Authorization"] == "Bearer tok"

    def test_add_sse_server(self, user_mcp_dir):
        entry = add_mcp_server("sse-srv", "sse", url="http://localhost:9090/sse")
        assert entry["transport"] == "sse"

    def test_add_websocket_server(self, user_mcp_dir):
        entry = add_mcp_server("ws", "websocket", url="ws://localhost:8765")
        assert entry["transport"] == "websocket"

    def test_add_with_tools_and_expose_to(self, user_mcp_dir):
        entry = add_mcp_server(
            "fs",
            "stdio",
            command="npx",
            args=[],
            tools=["read_file"],
            expose_to=["main", "code-agent"],
        )
        assert entry["tools"] == ["read_file"]
        assert entry["expose_to"] == ["main", "code-agent"]

    def test_add_replaces_existing(self, user_mcp_dir):
        add_mcp_server("srv", "stdio", command="old")
        add_mcp_server("srv", "http", url="http://new")
        data = yaml.safe_load(user_mcp_dir.read_text())
        assert data["srv"]["transport"] == "http"

    def test_add_invalid_transport_raises(self, user_mcp_dir):
        with pytest.raises(ValueError, match="Unknown transport"):
            add_mcp_server("bad", "carrier_pigeon", url="coo://rooftop")

    def test_stdio_without_command_raises(self, user_mcp_dir):
        with pytest.raises(ValueError, match="requires a command"):
            add_mcp_server("bad", "stdio")

    def test_http_without_url_raises(self, user_mcp_dir):
        with pytest.raises(ValueError, match="requires a url"):
            add_mcp_server("bad", "http")

    def test_add_with_env(self, user_mcp_dir):
        entry = add_mcp_server(
            "fs", "stdio", command="npx", args=[], env={"FOO": "bar"}
        )
        assert entry["env"] == {"FOO": "bar"}

    def test_add_multiple_servers(self, user_mcp_dir):
        add_mcp_server("a", "stdio", command="cmd1")
        add_mcp_server("b", "http", url="http://x")
        data = yaml.safe_load(user_mcp_dir.read_text())
        assert "a" in data
        assert "b" in data


class TestRemoveMcpServer:
    def test_remove_existing(self, user_mcp_dir):
        add_mcp_server("fs", "stdio", command="npx")
        assert remove_mcp_server("fs") is True
        data = yaml.safe_load(user_mcp_dir.read_text()) or {}
        assert "fs" not in data

    def test_remove_nonexistent(self, user_mcp_dir):
        assert remove_mcp_server("nope") is False

    def test_remove_preserves_others(self, user_mcp_dir):
        add_mcp_server("a", "stdio", command="cmd1")
        add_mcp_server("b", "http", url="http://x")
        remove_mcp_server("a")
        data = yaml.safe_load(user_mcp_dir.read_text())
        assert "a" not in data
        assert "b" in data


# ---- _parse_mcp_add_args (CLI arg parser) ----


class TestParseMcpAddArgs:
    def test_stdio_basic(self):
        r = parse_mcp_add_args(["fs", "npx", "-y", "server", "/tmp"])
        assert r["name"] == "fs"
        assert r["transport"] == "stdio"
        assert r["command"] == "npx"
        assert r["args"] == ["-y", "server", "/tmp"]

    def test_http_auto_detected(self):
        r = parse_mcp_add_args(["api", "http://localhost:8080/mcp"])
        assert r["transport"] == "http"
        assert r["url"] == "http://localhost:8080/mcp"

    def test_https_auto_detected(self):
        r = parse_mcp_add_args(["api", "https://example.com/mcp"])
        assert r["transport"] == "http"
        assert r["url"] == "https://example.com/mcp"

    def test_ws_auto_detected(self):
        r = parse_mcp_add_args(["ws", "ws://localhost:9090"])
        assert r["transport"] == "websocket"

    def test_explicit_transport_override(self):
        r = parse_mcp_add_args(["srv", "https://example.com/sse", "--transport", "sse"])
        assert r["transport"] == "sse"
        assert r["url"] == "https://example.com/sse"

    def test_explicit_transport_short_flag(self):
        r = parse_mcp_add_args(["srv", "https://x", "-T", "websocket"])
        assert r["transport"] == "websocket"

    def test_tools_flag(self):
        r = parse_mcp_add_args(["srv", "http://x", "--tools", "a,b"])
        assert r["tools"] == ["a", "b"]

    def test_expose_to_flag(self):
        r = parse_mcp_add_args(["srv", "http://x", "--expose-to", "main,code-agent"])
        assert r["expose_to"] == ["main", "code-agent"]

    def test_header_flag(self):
        r = parse_mcp_add_args(
            ["srv", "http://x", "--header", "Authorization:Bearer tok"]
        )
        assert r["headers"] == {"Authorization": "Bearer tok"}

    def test_env_flag(self):
        r = parse_mcp_add_args(["srv", "cmd", "--env", "FOO=bar"])
        assert r["env"] == {"FOO": "bar"}

    def test_too_few_tokens_raises(self):
        with pytest.raises(ValueError, match="Usage"):
            parse_mcp_add_args(["fs"])

    def test_double_dash_ignored(self):
        r = parse_mcp_add_args(["srv", "npx", "--", "-y", "pkg"])
        assert r["command"] == "npx"
        assert r["args"] == ["-y", "pkg"]
        assert "--" not in r["args"]

    def test_env_ref_flag(self):
        r = parse_mcp_add_args(["srv", "cmd", "--env-ref", "FOO"])
        assert r["env"] == {"FOO": "${FOO}"}

    def test_env_ref_and_env_combined(self):
        r = parse_mcp_add_args(
            ["srv", "cmd", "--env", "DEBUG=true", "--env-ref", "API_KEY"]
        )
        assert r["env"] == {"DEBUG": "true", "API_KEY": "${API_KEY}"}

    def test_missing_command_or_url_raises(self):
        with pytest.raises(ValueError, match="command or URL is required"):
            parse_mcp_add_args(["fs", "--tools", "a"])


# ---- edit_mcp_server ----


class TestEditMcpServer:
    def test_edit_expose_to(self, user_mcp_dir):
        add_mcp_server("fs", "stdio", command="npx", args=[])
        entry = edit_mcp_server("fs", expose_to=["main", "code-agent"])
        assert entry["expose_to"] == ["main", "code-agent"]
        assert entry["command"] == "npx"  # unchanged

    def test_edit_tools(self, user_mcp_dir):
        add_mcp_server("fs", "stdio", command="npx", args=[])
        entry = edit_mcp_server("fs", tools=["read_file"])
        assert entry["tools"] == ["read_file"]

    def test_edit_clear_tools(self, user_mcp_dir):
        add_mcp_server("fs", "stdio", command="npx", tools=["read_file"])
        entry = edit_mcp_server("fs", tools=None)
        assert "tools" not in entry

    def test_edit_url(self, user_mcp_dir):
        add_mcp_server("api", "http", url="http://old:8080/mcp")
        entry = edit_mcp_server("api", url="http://new:9090/mcp")
        assert entry["url"] == "http://new:9090/mcp"
        assert entry["transport"] == "http"  # unchanged

    def test_edit_nonexistent_raises(self, user_mcp_dir):
        with pytest.raises(KeyError, match="not found"):
            edit_mcp_server("nope", tools=["a"])

    def test_edit_invalid_transport_raises(self, user_mcp_dir):
        add_mcp_server("fs", "stdio", command="npx")
        with pytest.raises(ValueError, match="Unknown transport"):
            edit_mcp_server("fs", transport="carrier_pigeon")

    def test_edit_removes_required_field_raises(self, user_mcp_dir):
        add_mcp_server("fs", "stdio", command="npx")
        with pytest.raises(ValueError, match="requires a command"):
            edit_mcp_server("fs", command=None)

    def test_edit_preserves_unrelated_fields(self, user_mcp_dir):
        add_mcp_server(
            "fs",
            "stdio",
            command="npx",
            args=["-y", "srv"],
            tools=["a"],
            expose_to=["main"],
        )
        entry = edit_mcp_server("fs", expose_to=["code-agent"])
        assert entry["tools"] == ["a"]
        assert entry["args"] == ["-y", "srv"]
        assert entry["expose_to"] == ["code-agent"]


# ---- parse_mcp_edit_args ----


class TestParseMcpEditArgs:
    def test_basic_field(self):
        name, fields = parse_mcp_edit_args(["srv", "--url", "http://new"])
        assert name == "srv"
        assert fields["url"] == "http://new"

    def test_tools_none_clears(self):
        _, fields = parse_mcp_edit_args(["srv", "--tools", "none"])
        assert fields["tools"] is None

    def test_expose_to_csv(self):
        _, fields = parse_mcp_edit_args(["srv", "--expose-to", "main,code-agent"])
        assert fields["expose_to"] == ["main", "code-agent"]

    def test_multiple_fields(self):
        _, fields = parse_mcp_edit_args(["srv", "--url", "http://x", "--tools", "a,b"])
        assert fields["url"] == "http://x"
        assert fields["tools"] == ["a", "b"]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Usage"):
            parse_mcp_edit_args([])

    def test_no_fields_raises(self):
        with pytest.raises(ValueError, match="No fields"):
            parse_mcp_edit_args(["srv"])


# ---- uv tool compatibility ----


class TestUvToolCompat:
    """Tests for uv tool environment detection and compatible install helpers."""

    # -- _is_uv_tool_env --

    def test_is_uv_tool_env_false_when_no_virtual_env(self, monkeypatch):
        from EvoScientist.mcp.registry import _is_uv_tool_env

        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        assert _is_uv_tool_env() is False

    def test_is_uv_tool_env_false_for_regular_venv(self, monkeypatch):
        from EvoScientist.mcp.registry import _is_uv_tool_env

        monkeypatch.setenv("VIRTUAL_ENV", "/home/user/projects/myapp/.venv")
        assert _is_uv_tool_env() is False

    def test_is_uv_tool_env_true_unix(self, monkeypatch):
        from EvoScientist.mcp.registry import _is_uv_tool_env

        monkeypatch.setenv(
            "VIRTUAL_ENV", "/home/user/.local/share/uv/tools/evoscientist"
        )
        assert _is_uv_tool_env() is True

    def test_is_uv_tool_env_true_windows_backslashes(self, monkeypatch):
        from EvoScientist.mcp.registry import _is_uv_tool_env

        monkeypatch.setenv(
            "VIRTUAL_ENV", r"C:\Users\user\AppData\Local\uv\tools\evoscientist"
        )
        assert _is_uv_tool_env() is True

    # -- _uv_tool_name --

    def test_uv_tool_name_returns_name(self, monkeypatch):
        import EvoScientist.mcp.registry as reg

        monkeypatch.setenv(
            "VIRTUAL_ENV", "/home/user/.local/share/uv/tools/evoscientist"
        )
        assert reg._uv_tool_name() == "evoscientist"

    def test_uv_tool_name_returns_none_when_not_uv(self, monkeypatch):
        import EvoScientist.mcp.registry as reg

        monkeypatch.setenv("VIRTUAL_ENV", "/home/user/projects/myapp/.venv")
        assert reg._uv_tool_name() is None

    def test_uv_tool_name_returns_none_when_no_virtual_env(self, monkeypatch):
        import EvoScientist.mcp.registry as reg

        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        assert reg._uv_tool_name() is None

    # -- _uv_tool_existing_requirements --

    def test_existing_requirements_from_receipt(self, monkeypatch, tmp_path):
        import EvoScientist.mcp.registry as reg

        venv = tmp_path / "uv" / "tools" / "evoscientist"
        venv.mkdir(parents=True)
        receipt = venv / "uv-receipt.toml"
        receipt.write_text(
            "[tool]\nrequirements = [\n"
            '  { name = "evoscientist" },\n'
            '  { name = "arxiv-mcp-server" },\n'
            '  { name = "rich" },\n'
            "]\n"
        )
        monkeypatch.setenv("VIRTUAL_ENV", str(venv))
        result = reg._uv_tool_existing_requirements()
        assert result == {"arxiv-mcp-server": "arxiv-mcp-server", "rich": "rich"}

    def test_existing_requirements_preserves_specifiers_and_extras(
        self, monkeypatch, tmp_path
    ):
        import EvoScientist.mcp.registry as reg

        venv = tmp_path / "uv" / "tools" / "evoscientist"
        venv.mkdir(parents=True)
        receipt = venv / "uv-receipt.toml"
        receipt.write_text(
            "[tool]\nrequirements = [\n"
            '  { name = "evoscientist" },\n'
            '  { name = "rich", specifier = ">=13.0" },\n'
            '  { name = "requests", extras = ["socks"] },\n'
            '  { name = "lark-oapi", specifier = ">=1.4.0", extras = ["oauth"] },\n'
            "]\n"
        )
        monkeypatch.setenv("VIRTUAL_ENV", str(venv))
        result = reg._uv_tool_existing_requirements()
        assert result == {
            "rich": "rich>=13.0",
            "requests": "requests[socks]",
            "lark-oapi": "lark-oapi[oauth]>=1.4.0",
        }

    def test_existing_requirements_excludes_tool_name(self, monkeypatch, tmp_path):
        import EvoScientist.mcp.registry as reg

        venv = tmp_path / "uv" / "tools" / "evoscientist"
        venv.mkdir(parents=True)
        receipt = venv / "uv-receipt.toml"
        receipt.write_text(
            '[tool]\nrequirements = [\n  { name = "evoscientist" },\n]\n'
        )
        monkeypatch.setenv("VIRTUAL_ENV", str(venv))
        assert reg._uv_tool_existing_requirements() == {}

    def test_existing_requirements_no_receipt(self, monkeypatch, tmp_path):
        import EvoScientist.mcp.registry as reg

        venv = tmp_path / "uv" / "tools" / "evoscientist"
        venv.mkdir(parents=True)
        monkeypatch.setenv("VIRTUAL_ENV", str(venv))
        assert reg._uv_tool_existing_requirements() == {}

    # -- pip_install_hint --

    def test_pip_install_hint_uv_tool(self, monkeypatch):
        import EvoScientist.mcp.registry as reg

        monkeypatch.setattr(reg, "_is_uv_tool_env", lambda: True)
        hint = reg.pip_install_hint()
        assert "uv tool install --reinstall evoscientist --with" in hint

    def test_pip_install_hint_uv_no_tool(self, monkeypatch):
        import EvoScientist.mcp.registry as reg

        monkeypatch.setattr(reg, "_is_uv_tool_env", lambda: False)
        monkeypatch.setattr(
            reg.shutil, "which", lambda x: "/usr/bin/uv" if x == "uv" else None
        )
        assert reg.pip_install_hint() == "uv pip install"

    def test_pip_install_hint_plain_pip(self, monkeypatch):
        import EvoScientist.mcp.registry as reg

        monkeypatch.setattr(reg, "_is_uv_tool_env", lambda: False)
        monkeypatch.setattr(reg.shutil, "which", lambda x: None)
        assert reg.pip_install_hint() == "pip install"

    # -- install_pip_package --

    def test_install_pip_package_uv_tool_env_uses_uv_tool_install(
        self, monkeypatch, tmp_path
    ):
        """In a uv tool env, should use ``uv tool install --with`` for durability."""
        import EvoScientist.mcp.registry as reg

        # Set up a fake uv tool env with receipt
        venv = tmp_path / "uv" / "tools" / "evoscientist"
        venv.mkdir(parents=True)
        receipt = venv / "uv-receipt.toml"
        receipt.write_text(
            "[tool]\nrequirements = [\n"
            '  { name = "evoscientist" },\n'
            '  { name = "existing-pkg" },\n'
            "]\n"
        )
        monkeypatch.setenv("VIRTUAL_ENV", str(venv))

        captured: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            captured.append(list(cmd))
            return type("R", (), {"returncode": 0})()

        monkeypatch.setattr(
            reg.shutil, "which", lambda x: "/usr/bin/uv" if x == "uv" else None
        )
        monkeypatch.setattr(reg.subprocess, "run", fake_run)
        result = reg.install_pip_package("new-mcp-server")
        assert result is True
        assert len(captured) == 1
        cmd = captured[0]
        assert cmd[:3] == ["uv", "tool", "install"]
        assert "evoscientist" in cmd
        # Must preserve existing --with requirement
        assert "--with" in cmd
        with_args = [cmd[i + 1] for i, v in enumerate(cmd) if v == "--with"]
        assert "existing-pkg" in with_args
        assert "new-mcp-server" in with_args

    def test_install_pip_package_uv_tool_env_no_duplicate_with(
        self, monkeypatch, tmp_path
    ):
        """If the package is already in the receipt, don't add it twice."""
        import EvoScientist.mcp.registry as reg

        venv = tmp_path / "uv" / "tools" / "evoscientist"
        venv.mkdir(parents=True)
        receipt = venv / "uv-receipt.toml"
        receipt.write_text(
            "[tool]\nrequirements = [\n"
            '  { name = "evoscientist" },\n'
            '  { name = "arxiv-mcp-server" },\n'
            "]\n"
        )
        monkeypatch.setenv("VIRTUAL_ENV", str(venv))

        captured: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            captured.append(list(cmd))
            return type("R", (), {"returncode": 0})()

        monkeypatch.setattr(
            reg.shutil, "which", lambda x: "/usr/bin/uv" if x == "uv" else None
        )
        monkeypatch.setattr(reg.subprocess, "run", fake_run)
        reg.install_pip_package("arxiv-mcp-server")
        cmd = captured[0]
        with_args = [cmd[i + 1] for i, v in enumerate(cmd) if v == "--with"]
        assert with_args.count("arxiv-mcp-server") == 1

    def test_install_pip_package_uv_tool_preserves_specifiers(
        self, monkeypatch, tmp_path
    ):
        """Existing --with specs with extras/versions must be preserved."""
        import EvoScientist.mcp.registry as reg

        venv = tmp_path / "uv" / "tools" / "evoscientist"
        venv.mkdir(parents=True)
        receipt = venv / "uv-receipt.toml"
        receipt.write_text(
            "[tool]\nrequirements = [\n"
            '  { name = "evoscientist" },\n'
            '  { name = "rich", specifier = ">=13.0" },\n'
            '  { name = "requests", extras = ["socks"] },\n'
            "]\n"
        )
        monkeypatch.setenv("VIRTUAL_ENV", str(venv))

        captured: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            captured.append(list(cmd))
            return type("R", (), {"returncode": 0})()

        monkeypatch.setattr(
            reg.shutil, "which", lambda x: "/usr/bin/uv" if x == "uv" else None
        )
        monkeypatch.setattr(reg.subprocess, "run", fake_run)
        reg.install_pip_package("new-pkg")
        cmd = captured[0]
        with_args = [cmd[i + 1] for i, v in enumerate(cmd) if v == "--with"]
        assert "rich>=13.0" in with_args
        assert "requests[socks]" in with_args
        assert "new-pkg" in with_args

    def test_install_pip_package_uv_tool_dedup_with_version_spec(
        self, monkeypatch, tmp_path
    ):
        """Dedup must match bare name even if package arg has version spec."""
        import EvoScientist.mcp.registry as reg

        venv = tmp_path / "uv" / "tools" / "evoscientist"
        venv.mkdir(parents=True)
        receipt = venv / "uv-receipt.toml"
        receipt.write_text(
            "[tool]\nrequirements = [\n"
            '  { name = "evoscientist" },\n'
            '  { name = "rich", specifier = ">=13.0" },\n'
            "]\n"
        )
        monkeypatch.setenv("VIRTUAL_ENV", str(venv))

        captured: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            captured.append(list(cmd))
            return type("R", (), {"returncode": 0})()

        monkeypatch.setattr(
            reg.shutil, "which", lambda x: "/usr/bin/uv" if x == "uv" else None
        )
        monkeypatch.setattr(reg.subprocess, "run", fake_run)
        # package arg has version constraint — should still dedup against "rich"
        reg.install_pip_package("rich>=14.0")
        cmd = captured[0]
        with_args = [cmd[i + 1] for i, v in enumerate(cmd) if v == "--with"]
        # Should keep the existing spec, not add a duplicate
        assert with_args.count("rich>=13.0") == 1
        assert "rich>=14.0" not in with_args

    def test_install_pip_package_uv_tool_falls_back_on_failure(
        self, monkeypatch, tmp_path
    ):
        """If ``uv tool install --with`` fails, fall back to ``uv pip install``
        when ``verify_command`` isn't set (library install)."""
        import EvoScientist.mcp.registry as reg

        venv = tmp_path / "uv" / "tools" / "evoscientist"
        venv.mkdir(parents=True)
        receipt = venv / "uv-receipt.toml"
        receipt.write_text(
            '[tool]\nrequirements = [\n  { name = "evoscientist" },\n]\n'
        )
        monkeypatch.setenv("VIRTUAL_ENV", str(venv))

        commands: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            commands.append(cmd)
            if cmd[:3] == ["uv", "tool", "install"]:
                return type("R", (), {"returncode": 1})()
            return type("R", (), {"returncode": 0})()

        monkeypatch.setattr(
            reg.shutil, "which", lambda x: "/usr/bin/uv" if x == "uv" else None
        )
        monkeypatch.setattr(reg.subprocess, "run", fake_run)
        result = reg.install_pip_package("some-package")
        assert result is True
        # No verify_command → standalone `uv tool install <pkg>` is skipped.
        # Order: uv tool install --with (fail), uv pip install (ok).
        assert len(commands) == 2
        assert commands[0][:3] == ["uv", "tool", "install"]
        assert "--with" in commands[0]
        assert commands[1][:3] == ["uv", "pip", "install"]

    def test_install_pip_package_uv_tool_with_verify_tries_standalone(
        self, monkeypatch, tmp_path
    ):
        """In uv-tool env with ``verify_command``: ``--with`` fails →
        standalone ``uv tool install`` is tried before pip."""
        import EvoScientist.mcp.registry as reg

        venv = tmp_path / "uv" / "tools" / "evoscientist"
        venv.mkdir(parents=True)
        receipt = venv / "uv-receipt.toml"
        receipt.write_text(
            '[tool]\nrequirements = [\n  { name = "evoscientist" },\n]\n'
        )
        monkeypatch.setenv("VIRTUAL_ENV", str(venv))

        commands: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            commands.append(cmd)
            if cmd[:3] == ["uv", "tool", "install"]:
                return type("R", (), {"returncode": 1})()
            return type("R", (), {"returncode": 0})()

        monkeypatch.setattr(
            reg.shutil, "which", lambda x: "/usr/bin/uv" if x == "uv" else None
        )
        monkeypatch.setattr(reg.subprocess, "run", fake_run)
        result = reg.install_pip_package("some-cli", verify_command="some-cli")
        assert result is True
        assert len(commands) == 3
        assert commands[0][:3] == ["uv", "tool", "install"]
        assert "--with" in commands[0]
        assert commands[1][:3] == ["uv", "tool", "install"]
        assert "--with" not in commands[1]
        assert commands[2][:3] == ["uv", "pip", "install"]

    def test_install_pip_package_skips_uv_tool_without_verify_command(
        self, monkeypatch
    ):
        """Without ``verify_command`` the non-uv-tool path must skip
        ``uv tool install <pkg>`` — library callers rely on the package
        being importable from the active venv, which standalone uv tools
        are not."""
        import sys

        import EvoScientist.mcp.registry as reg

        captured: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            captured.append(cmd)
            return type("R", (), {"returncode": 0})()

        monkeypatch.setattr(reg, "_is_uv_tool_env", lambda: False)
        monkeypatch.setattr(
            reg.shutil, "which", lambda x: "/usr/bin/uv" if x == "uv" else None
        )
        monkeypatch.setattr(reg.subprocess, "run", fake_run)
        result = reg.install_pip_package("some-library")
        assert result is True
        assert len(captured) == 1
        assert captured[0][:3] == ["uv", "pip", "install"]
        assert sys.executable in captured[0]

    def test_install_pip_package_with_verify_prefers_uv_tool_install(
        self, monkeypatch, tmp_path
    ):
        """Non-uv-tool env, ``verify_command`` found in uv-tool bin dir:
        ``uv tool install <pkg>`` is used so the binary survives uv sync."""
        import EvoScientist.mcp.registry as reg

        captured: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            captured.append(cmd)
            return type("R", (), {"returncode": 0})()

        fake_bin = tmp_path / "some-cli"
        fake_bin.write_text("#!/bin/sh\n")
        fake_bin.chmod(0o755)

        monkeypatch.setattr(reg, "_is_uv_tool_env", lambda: False)
        monkeypatch.setattr(
            reg.shutil, "which", lambda x: "/usr/bin/uv" if x == "uv" else None
        )
        monkeypatch.setattr(reg.subprocess, "run", fake_run)
        monkeypatch.setattr(
            reg, "_uv_tool_bin", lambda cmd: fake_bin if cmd == "some-cli" else None
        )
        result = reg.install_pip_package("some-cli", verify_command="some-cli")
        assert result is True
        assert len(captured) == 1
        assert captured[0][:3] == ["uv", "tool", "install"]
        assert "some-cli" in captured[0]

    def test_install_pip_package_verify_command_missing_triggers_fallback(
        self, monkeypatch
    ):
        """When ``verify_command`` isn't in the uv-tool bin dir after
        ``uv tool install``, fall through to ``uv pip install``."""
        import sys

        import EvoScientist.mcp.registry as reg

        captured: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            captured.append(cmd)
            return type("R", (), {"returncode": 0})()

        monkeypatch.setattr(reg, "_is_uv_tool_env", lambda: False)
        monkeypatch.setattr(
            reg.shutil, "which", lambda x: "/usr/bin/uv" if x == "uv" else None
        )
        monkeypatch.setattr(reg.subprocess, "run", fake_run)
        # uv-tool bin dir exists, but the expected binary isn't in it
        # (entry-point-less package).
        monkeypatch.setattr(reg, "_uv_tool_bin", lambda cmd: None)
        result = reg.install_pip_package(
            "lib-without-entrypoint", verify_command="ghost-cli"
        )
        assert result is True
        assert len(captured) == 2
        assert captured[0][:3] == ["uv", "tool", "install"]
        assert "--python" in captured[1]
        assert sys.executable in captured[1]

    def test_install_pip_package_verify_command_present_short_circuits(
        self, monkeypatch, tmp_path
    ):
        """``uv tool install`` success + verify_command present in uv-tool
        bin dir = no pip fallback, even if a stale copy exists in the venv."""
        import EvoScientist.mcp.registry as reg

        captured: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            captured.append(cmd)
            return type("R", (), {"returncode": 0})()

        # Stale venv binary would be returned by shutil.which — irrelevant
        # now because the verify looks at the uv-tool bin dir directly.
        fake_bin = tmp_path / "arxiv-mcp-server"
        fake_bin.write_text("#!/bin/sh\n")
        fake_bin.chmod(0o755)

        def fake_which(x):
            if x == "uv":
                return "/usr/bin/uv"
            if x == "arxiv-mcp-server":
                return "/venv/bin/arxiv-mcp-server"  # stale, should NOT be trusted
            return None

        monkeypatch.setattr(reg, "_is_uv_tool_env", lambda: False)
        monkeypatch.setattr(reg.shutil, "which", fake_which)
        monkeypatch.setattr(reg.subprocess, "run", fake_run)
        monkeypatch.setattr(
            reg, "_uv_tool_bin", lambda cmd: fake_bin if cmd == "arxiv-mcp-server" else None
        )
        result = reg.install_pip_package(
            "arxiv-mcp-server", verify_command="arxiv-mcp-server"
        )
        assert result is True
        assert len(captured) == 1
        assert captured[0][:3] == ["uv", "tool", "install"]

    def test_install_pip_package_falls_back_to_pip_when_no_uv(self, monkeypatch):
        import sys

        import EvoScientist.mcp.registry as reg

        captured: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            captured.append(cmd)
            ns = type("R", (), {"returncode": 0})()
            return ns

        monkeypatch.setattr(reg, "_is_uv_tool_env", lambda: False)
        monkeypatch.setattr(reg.shutil, "which", lambda x: None)
        monkeypatch.setattr(reg.subprocess, "run", fake_run)
        reg.install_pip_package("some-package")
        assert len(captured) == 1
        assert sys.executable in captured[0]
        assert "-m" in captured[0]
        assert "pip" in captured[0]

    # -- _resolve_command_path --

    def test_resolve_command_path_absolute_passthrough(self):
        from EvoScientist.mcp.registry import _resolve_command_path

        assert _resolve_command_path("/usr/bin/my-tool") == "/usr/bin/my-tool"

    def test_resolve_command_path_found_in_bin_dir(self, monkeypatch, tmp_path):
        import sys

        import EvoScientist.mcp.registry as reg

        # Create a fake executable in a temp bin dir
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        fake_exe = bin_dir / "my-mcp-server"
        fake_exe.touch()
        fake_exe.chmod(0o755)

        # Point sys.executable to something in that bin dir
        fake_python = bin_dir / "python"
        fake_python.touch()
        monkeypatch.setattr(sys, "executable", str(fake_python))
        # Ensure shutil.which won't find it on PATH
        monkeypatch.setattr(reg.shutil, "which", lambda x: None)

        result = reg._resolve_command_path("my-mcp-server")
        assert result == str(fake_exe)

    def test_resolve_command_path_windows_exe_suffix(self, monkeypatch, tmp_path):
        import os
        import sys

        import EvoScientist.mcp.registry as reg

        if os.name != "nt":
            pytest.skip("Windows-only behaviour")

        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        fake_exe = bin_dir / "my-mcp-server.exe"
        fake_exe.touch()
        monkeypatch.setattr(sys, "executable", str(bin_dir / "python.exe"))
        monkeypatch.setattr(reg.shutil, "which", lambda x: None)

        result = reg._resolve_command_path("my-mcp-server")
        assert result == str(fake_exe)

    def test_resolve_command_path_returns_bare_when_not_found(
        self, monkeypatch, tmp_path
    ):
        import sys

        import EvoScientist.mcp.registry as reg

        monkeypatch.setattr(reg.shutil, "which", lambda x: None)
        monkeypatch.setattr(reg, "_uv_tool_bin", lambda cmd: None)
        monkeypatch.setattr(sys, "executable", str(tmp_path / "bin" / "python"))
        result = reg._resolve_command_path("nonexistent-tool")
        assert result == "nonexistent-tool"

    def test_resolve_command_path_prefers_uv_tool_over_venv_shadow(
        self, monkeypatch, tmp_path
    ):
        """When both ``uv tool dir --bin`` and a venv's ``bin/`` contain the
        command, prefer the uv-tool location so the path written to mcp.yaml
        survives ``uv sync``."""
        import EvoScientist.mcp.registry as reg

        uv_bin_dir = tmp_path / "uv-bin"
        uv_bin_dir.mkdir()
        uv_copy = uv_bin_dir / "arxiv-mcp-server"
        uv_copy.write_text("#!/bin/sh\n")
        uv_copy.chmod(0o755)

        # Stale venv copy that `uv run` would surface first via PATH.
        venv_copy = tmp_path / "venv-bin" / "arxiv-mcp-server"
        venv_copy.parent.mkdir()
        venv_copy.write_text("#!/bin/sh\n")
        venv_copy.chmod(0o755)

        monkeypatch.setattr(reg, "_uv_tool_bin_dir", lambda: uv_bin_dir)
        monkeypatch.setattr(reg.shutil, "which", lambda x: str(venv_copy))
        result = reg._resolve_command_path("arxiv-mcp-server")
        assert result == str(uv_copy)
