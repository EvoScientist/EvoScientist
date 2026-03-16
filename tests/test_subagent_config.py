"""Tests for subagent.yaml structure and MCP config template."""

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
def template_path():
    return (
        Path(__file__).parent.parent
        / "docs/examples/mcp-ssh-gpu/mcp-ssh-gpu.yaml.example"
    )


class TestSubagentYAMLStructure:
    """Tests for subagent.yaml structure and formatting."""

    def test_valid_yaml_structure(self, subagent_config):
        assert isinstance(subagent_config, dict)
        assert "code-agent" in subagent_config
        assert "debug-agent" in subagent_config

    def test_all_agents_have_system_prompts(self, subagent_config):
        for agent_name, agent_config in subagent_config.items():
            assert (
                "system_prompt" in agent_config or "system_prompt_ref" in agent_config
            ), f"{agent_name} should have a system prompt or system_prompt_ref"


class TestMcpConfigTemplate:
    """Tests for MCP config template fields."""

    def test_mcp_config_template_exists(self, template_path):
        assert template_path.exists()

    def test_mcp_config_template_has_expose_to(self, template_path):
        assert "expose_to" in template_path.read_text()

    def test_mcp_config_template_has_env(self, template_path):
        content = template_path.read_text()
        assert "env:" in content or "SSH_HOST" in content
