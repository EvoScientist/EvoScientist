"""Tests for subagent.yaml prompt changes - SSH GPU support."""

import pytest
from pathlib import Path
import yaml


@pytest.fixture
def subagent_config():
    """Load subagent.yaml configuration."""
    config_path = Path(__file__).parent.parent / "EvoScientist" / "subagent.yaml"
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestSSHGPUSupport:
    """Tests for SSH GPU support in subagent prompts."""

    def test_code_agent_has_ssh_instructions(self, subagent_config):
        """code-agent prompt should contain SSH-aware instructions."""
        code_agent = subagent_config.get("code-agent", {})
        system_prompt = code_agent.get("system_prompt", "")

        assert "ssh_execute" in system_prompt, (
            "code-agent should have ssh_execute instructions"
        )
        assert "ssh_upload" in system_prompt, (
            "code-agent should have ssh_upload instructions"
        )
        assert "ssh_download" in system_prompt, (
            "code-agent should have ssh_download instructions"
        )

    def test_code_agent_ssh_execute_usage(self, subagent_config):
        """code-agent should specify when to use ssh_execute."""
        code_agent = subagent_config.get("code-agent", {})
        system_prompt = code_agent.get("system_prompt", "")

        assert "GPU-dependent commands" in system_prompt, (
            "code-agent should mention GPU-dependent commands for ssh_execute"
        )
        assert "training" in system_prompt.lower(), (
            "code-agent should mention training as ssh_execute use case"
        )
        assert "inference" in system_prompt.lower(), (
            "code-agent should mention inference as ssh_execute use case"
        )

    def test_code_agent_ssh_upload_usage(self, subagent_config):
        """code-agent should specify when to use ssh_upload."""
        code_agent = subagent_config.get("code-agent", {})
        system_prompt = code_agent.get("system_prompt", "")

        assert "sync experiment code" in system_prompt, (
            "code-agent should mention code sync via ssh_upload"
        )

    def test_code_agent_ssh_download_usage(self, subagent_config):
        """code-agent should specify when to use ssh_download."""
        code_agent = subagent_config.get("code-agent", {})
        system_prompt = code_agent.get("system_prompt", "")

        assert "retrieve results" in system_prompt, (
            "code-agent should mention result retrieval via ssh_download"
        )
        assert "artifacts" in system_prompt, (
            "code-agent should mention artifact retrieval via ssh_download"
        )

    def test_code_agent_screen_tmux_guidance(self, subagent_config):
        """code-agent should mention screen/tmux for long-running jobs."""
        code_agent = subagent_config.get("code-agent", {})
        system_prompt = code_agent.get("system_prompt", "")

        assert "screen" in system_prompt.lower() or "tmux" in system_prompt.lower(), (
            "code-agent should mention screen/tmux for long-running remote jobs"
        )

    def test_code_agent_backward_compatibility(self, subagent_config):
        """code-agent should fall back to local execution when no SSH tools."""
        code_agent = subagent_config.get("code-agent", {})
        system_prompt = code_agent.get("system_prompt", "")

        assert "not available" in system_prompt.lower(), (
            "code-agent should mention fallback when SSH tools unavailable"
        )
        assert "execute locally" in system_prompt.lower(), (
            "code-agent should mention local execution as fallback"
        )

    def test_debug_agent_has_ssh_instructions(self, subagent_config):
        """debug-agent prompt should contain SSH-aware instructions."""
        debug_agent = subagent_config.get("debug-agent", {})
        system_prompt = debug_agent.get("system_prompt", "")

        assert "ssh_execute" in system_prompt, (
            "debug-agent should have ssh_execute instructions"
        )
        assert "ssh_download" in system_prompt, (
            "debug-agent should have ssh_download instructions"
        )

    def test_debug_agent_ssh_execute_usage(self, subagent_config):
        """debug-agent should specify when to use ssh_execute."""
        debug_agent = subagent_config.get("debug-agent", {})
        system_prompt = debug_agent.get("system_prompt", "")

        assert "reproduce failures" in system_prompt, (
            "debug-agent should mention reproducing failures via ssh_execute"
        )
        assert "remote environment" in system_prompt, (
            "debug-agent should mention checking remote environment"
        )

    def test_debug_agent_cuda_version_check(self, subagent_config):
        """debug-agent should mention CUDA version checking."""
        debug_agent = subagent_config.get("debug-agent", {})
        system_prompt = debug_agent.get("system_prompt", "")

        assert "CUDA" in system_prompt, (
            "debug-agent should mention CUDA version checking"
        )

    def test_debug_agent_ssh_download_usage(self, subagent_config):
        """debug-agent should specify when to use ssh_download."""
        debug_agent = subagent_config.get("debug-agent", {})
        system_prompt = debug_agent.get("system_prompt", "")

        assert "remote logs" in system_prompt, (
            "debug-agent should mention retrieving remote logs via ssh_download"
        )

    def test_debug_agent_backward_compatibility(self, subagent_config):
        """debug-agent should fall back to local debugging when no SSH tools."""
        debug_agent = subagent_config.get("debug-agent", {})
        system_prompt = debug_agent.get("system_prompt", "")

        assert "not available" in system_prompt.lower(), (
            "debug-agent should mention fallback when SSH tools unavailable"
        )
        assert "debug locally" in system_prompt.lower(), (
            "debug-agent should mention local debugging as fallback"
        )


class TestSubagentYAMLStructure:
    """Tests for subagent.yaml structure and formatting."""

    def test_valid_yaml_structure(self, subagent_config):
        """subagent.yaml should be valid YAML with agent mappings."""
        assert isinstance(subagent_config, dict), (
            "subagent.yaml should be a YAML mapping"
        )
        assert "code-agent" in subagent_config, "subagent.yaml should define code-agent"
        assert "debug-agent" in subagent_config, (
            "subagent.yaml should define debug-agent"
        )

    def test_all_agents_have_system_prompts(self, subagent_config):
        """All agents should have system prompts."""
        for agent_name, agent_config in subagent_config.items():
            assert (
                "system_prompt" in agent_config or "system_prompt_ref" in agent_config
            ), f"{agent_name} should have a system prompt or system_prompt_ref"
