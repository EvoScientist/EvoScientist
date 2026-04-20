"""Tests for the /model command and extract_model_and_provider helper."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import run_async as _run


class TestExtractModelAndProvider:
    """Unit tests for the argument parser helper."""

    def test_known_model_no_provider(self):
        from EvoScientist.commands.implementation.model import (
            extract_model_and_provider,
        )

        name, prov = extract_model_and_provider(
            ["claude-sonnet-4-6"], fallback_provider="anthropic"
        )
        assert name == "claude-sonnet-4-6"
        assert prov == "anthropic"

    def test_known_model_with_provider_override(self):
        from EvoScientist.commands.implementation.model import (
            extract_model_and_provider,
        )

        name, prov = extract_model_and_provider(
            ["claude-sonnet-4-6", "openrouter"], fallback_provider="anthropic"
        )
        assert name == "claude-sonnet-4-6"
        assert prov == "openrouter"

    def test_unknown_model_no_provider_raises(self):
        from EvoScientist.commands.implementation.model import (
            extract_model_and_provider,
        )

        with pytest.raises(ValueError, match="Unknown model"):
            extract_model_and_provider(
                ["nonexistent-model-xyz"], fallback_provider="anthropic"
            )

    def test_unknown_model_with_provider_still_raises(self):
        from EvoScientist.commands.implementation.model import (
            extract_model_and_provider,
        )

        # Unknown models are always rejected, even with an explicit provider
        with pytest.raises(ValueError, match="Unknown model"):
            extract_model_and_provider(
                ["my-custom-model", "custom-openai"], fallback_provider="anthropic"
            )

    def test_provider_override_on_known_model(self):
        from EvoScientist.commands.implementation.model import (
            extract_model_and_provider,
        )

        # Known model with explicit provider override uses the override
        name, prov = extract_model_and_provider(
            ["claude-sonnet-4-6", "openrouter"], fallback_provider="anthropic"
        )
        assert name == "claude-sonnet-4-6"
        assert prov == "openrouter"


class TestModelCommandUnknownModel:
    """Verify error message for unknown models."""

    def test_unknown_model_shows_error(self):
        from EvoScientist.commands.implementation.model import ModelCommand

        cmd = ModelCommand()
        ui = MagicMock()
        ui.supports_interactive = True
        cfg = SimpleNamespace(model="claude-sonnet-4-6", provider="anthropic")

        ctx = MagicMock()
        ctx.ui = ui

        with patch(
            "EvoScientist.EvoScientist._ensure_config",
            return_value=cfg,
        ):
            _run(cmd.execute(ctx, ["nonexistent-model-xyz"]))

        ui.append_system.assert_called_once()
        call_args = ui.append_system.call_args
        assert "Unknown model" in call_args[0][0]
        assert call_args[1]["style"] == "red"


class TestModelCommandPickerCancelled:
    """Verify no-op when the interactive picker is cancelled."""

    def test_picker_returns_none(self):
        from EvoScientist.commands.implementation.model import ModelCommand

        cmd = ModelCommand()
        ui = MagicMock()
        ui.supports_interactive = True
        ui.wait_for_model_pick = AsyncMock(return_value=None)
        cfg = SimpleNamespace(model="claude-sonnet-4-6", provider="anthropic")

        ctx = MagicMock()
        ctx.ui = ui

        with patch(
            "EvoScientist.EvoScientist._ensure_config",
            return_value=cfg,
        ):
            _run(cmd.execute(ctx, []))

        # No model switch should have happened
        ui.append_system.assert_not_called()


class TestModelCommandSwitch:
    """Verify a successful model switch updates config and rebuilds agent."""

    def test_switch_known_model(self):
        from EvoScientist.commands.implementation.model import ModelCommand

        cmd = ModelCommand()
        ui = MagicMock()
        ui.supports_interactive = True
        cfg = SimpleNamespace(model="claude-sonnet-4-6", provider="anthropic")
        new_agent = MagicMock()

        ctx = MagicMock()
        ctx.ui = ui
        ctx.workspace_dir = "/tmp/test"
        ctx.checkpointer = MagicMock()

        with (
            patch(
                "EvoScientist.EvoScientist._ensure_config",
                return_value=cfg,
            ),
            patch(
                "EvoScientist.EvoScientist.set_chat_model",
            ),
            patch(
                "EvoScientist.cli.agent._load_agent",
                return_value=new_agent,
            ),
        ):
            _run(cmd.execute(ctx, ["claude-opus-4-6"]))

        # Config should be updated
        assert cfg.model == "claude-opus-4-6"
        assert cfg.provider == "anthropic"

        # Agent should be replaced on context
        assert ctx.agent == new_agent

        # Success message shown
        ui.append_system.assert_called_once()
        msg = ui.append_system.call_args[0][0]
        assert "claude-opus-4-6" in msg
        assert "anthropic" in msg

    def test_switch_with_save_flag(self):
        from EvoScientist.commands.implementation.model import ModelCommand

        cmd = ModelCommand()
        ui = MagicMock()
        ui.supports_interactive = True
        cfg = SimpleNamespace(model="claude-sonnet-4-6", provider="anthropic")

        ctx = MagicMock()
        ctx.ui = ui
        ctx.workspace_dir = "/tmp/test"
        ctx.checkpointer = MagicMock()

        with (
            patch(
                "EvoScientist.EvoScientist._ensure_config",
                return_value=cfg,
            ),
            patch("EvoScientist.EvoScientist.set_chat_model"),
            patch(
                "EvoScientist.cli.agent._load_agent",
                return_value=MagicMock(),
            ),
            patch(
                "EvoScientist.config.settings.set_config_value"
            ) as mock_save,
        ):
            _run(cmd.execute(ctx, ["claude-opus-4-6", "--save"]))

        # Config file should be updated
        mock_save.assert_any_call("model", "claude-opus-4-6")
        mock_save.assert_any_call("provider", "anthropic")

        # Success message should mention save
        msg = ui.append_system.call_args[0][0]
        assert "saved to config" in msg

    def test_switch_without_save_flag_does_not_persist(self):
        from EvoScientist.commands.implementation.model import ModelCommand

        cmd = ModelCommand()
        ui = MagicMock()
        ui.supports_interactive = True
        cfg = SimpleNamespace(model="claude-sonnet-4-6", provider="anthropic")

        ctx = MagicMock()
        ctx.ui = ui
        ctx.workspace_dir = "/tmp/test"
        ctx.checkpointer = MagicMock()

        with (
            patch(
                "EvoScientist.EvoScientist._ensure_config",
                return_value=cfg,
            ),
            patch("EvoScientist.EvoScientist.set_chat_model"),
            patch(
                "EvoScientist.cli.agent._load_agent",
                return_value=MagicMock(),
            ),
            patch(
                "EvoScientist.config.settings.set_config_value"
            ) as mock_save,
        ):
            _run(cmd.execute(ctx, ["claude-opus-4-6"]))

        # Config file should NOT be updated
        mock_save.assert_not_called()

        # Message should not mention save
        msg = ui.append_system.call_args[0][0]
        assert "saved to config" not in msg


class TestModelCommandFailure:
    """Verify error handling when set_chat_model raises."""

    def test_set_chat_model_error(self):
        from EvoScientist.commands.implementation.model import ModelCommand

        cmd = ModelCommand()
        ui = MagicMock()
        ui.supports_interactive = True
        cfg = SimpleNamespace(model="claude-sonnet-4-6", provider="anthropic")

        ctx = MagicMock()
        ctx.ui = ui

        with (
            patch(
                "EvoScientist.EvoScientist._ensure_config",
                return_value=cfg,
            ),
            patch(
                "EvoScientist.EvoScientist.set_chat_model",
                side_effect=RuntimeError("API key missing"),
            ),
        ):
            _run(cmd.execute(ctx, ["claude-opus-4-6"]))

        ui.append_system.assert_called_once()
        call_args = ui.append_system.call_args
        assert "Failed to switch model" in call_args[0][0]
        assert call_args[1]["style"] == "red"
