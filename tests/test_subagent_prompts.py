"""Tests for subagent.yaml prompt changes - SSH GPU support."""

from pathlib import Path

import pytest
import yaml


@pytest.fixture(scope="module")
def subagent_config():
    """Load subagent.yaml configuration."""
    config_path = Path(__file__).parent.parent / "EvoScientist" / "subagent.yaml"
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def code_prompt(subagent_config):
    return subagent_config.get("code-agent", {}).get("system_prompt", "")


@pytest.fixture(scope="module")
def debug_prompt(subagent_config):
    return subagent_config.get("debug-agent", {}).get("system_prompt", "")


@pytest.fixture(scope="module")
def template_path():
    return (
        Path(__file__).parent.parent
        / "docs/examples/mcp-ssh-gpu/mcp-ssh-gpu.yaml.example"
    )


class TestSSHGPUSupport:
    """Tests for SSH GPU support in subagent prompts."""

    def test_code_agent_has_ssh_instructions(self, code_prompt):
        assert "ssh_execute" in code_prompt
        assert "ssh_upload" in code_prompt
        assert "ssh_download" in code_prompt

    def test_code_agent_ssh_execute_usage(self, code_prompt):
        assert "GPU-dependent commands" in code_prompt
        assert "training" in code_prompt.lower()
        assert "inference" in code_prompt.lower()

    def test_code_agent_ssh_upload_usage(self, code_prompt):
        assert "sync experiment code" in code_prompt

    def test_code_agent_ssh_download_usage(self, code_prompt):
        assert "retrieve results" in code_prompt
        assert "artifacts" in code_prompt

    def test_code_agent_screen_tmux_guidance(self, code_prompt):
        assert "screen" in code_prompt.lower() or "tmux" in code_prompt.lower()

    def test_code_agent_backward_compatibility(self, code_prompt):
        assert "not available" in code_prompt.lower()
        assert "execute locally" in code_prompt.lower()

    def test_debug_agent_has_ssh_instructions(self, debug_prompt):
        assert "ssh_execute" in debug_prompt
        assert "ssh_download" in debug_prompt

    def test_debug_agent_ssh_execute_usage(self, debug_prompt):
        assert "reproduce failures" in debug_prompt
        assert "remote environment" in debug_prompt

    def test_debug_agent_cuda_version_check(self, debug_prompt):
        assert "CUDA" in debug_prompt

    def test_debug_agent_ssh_download_usage(self, debug_prompt):
        assert "remote logs" in debug_prompt

    def test_debug_agent_backward_compatibility(self, debug_prompt):
        assert "not available" in debug_prompt.lower()
        assert "debug locally" in debug_prompt.lower()

    def test_code_agent_execution_mode_declaration(self, code_prompt):
        p = code_prompt.lower()
        assert "execution mode" in p or ("remote" in p and "locally" in p)

    def test_debug_agent_execution_mode_declaration(self, debug_prompt):
        p = debug_prompt.lower()
        assert "execution mode" in p or ("debugging" in p and "locally" in p)

    def test_debug_agent_has_ssh_upload(self, debug_prompt):
        p = debug_prompt.lower()
        assert "ssh_upload" in p or ("upload" in p and "push" in p)

    def test_flexible_tool_matching_documented(self, code_prompt, debug_prompt):
        assert (
            "run_remote_command" in code_prompt
            or "or equivalent" in code_prompt.lower()
            or "or equivalent" in debug_prompt.lower()
        )


class TestSubagentYAMLStructure:
    """Tests for subagent.yaml structure and formatting."""

    def test_valid_yaml_structure(self, subagent_config):
        assert isinstance(subagent_config, dict)
        assert "code-agent" in subagent_config
        assert "debug-agent" in subagent_config

    def test_all_agents_have_system_prompts(self, subagent_config):
        for agent_name, agent_config in subagent_config.items():
            assert "system_prompt" in agent_config or "system_prompt_ref" in agent_config, (
                f"{agent_name} should have a system prompt or system_prompt_ref"
            )


class TestMcpConfigTemplate:
    """Tests for MCP config template fields."""

    def test_mcp_config_template_exists(self, template_path):
        assert template_path.exists()

    def test_mcp_config_template_has_expose_to(self, template_path):
        assert "expose_to" in template_path.read_text()

    def test_mcp_config_template_has_env(self, template_path):
        content = template_path.read_text()
        assert "env:" in content or "SSH_HOST" in content
