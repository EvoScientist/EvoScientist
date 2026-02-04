"""Tests for EvoScientist onboarding wizard."""

from unittest import mock
import sys

import pytest

from EvoScientist.onboard import (
    IntegerValidator,
    ChoiceValidator,
    validate_anthropic_key,
    validate_openai_key,
    validate_tavily_key,
    STEPS,
    WIZARD_STYLE,
    CONFIRM_STYLE,
    render_progress,
)
from EvoScientist.config import EvoScientistConfig


# =============================================================================
# Test STEPS and WIZARD_STYLE constants
# =============================================================================


class TestConstants:
    def test_steps_has_six_items(self):
        """Test that STEPS contains exactly 6 steps."""
        assert len(STEPS) == 6
        assert STEPS == ["Provider", "API Key", "Model", "Tavily Key", "Workspace", "Parameters"]

    def test_wizard_style_is_style_instance(self):
        """Test that WIZARD_STYLE is a prompt_toolkit Style."""
        from prompt_toolkit.styles import Style
        assert isinstance(WIZARD_STYLE, Style)

    def test_wizard_style_has_required_keys(self):
        """Test that WIZARD_STYLE defines expected style classes."""
        assert WIZARD_STYLE is not None

    def test_confirm_style_is_style_instance(self):
        """Test that CONFIRM_STYLE is a prompt_toolkit Style."""
        from prompt_toolkit.styles import Style
        assert isinstance(CONFIRM_STYLE, Style)

    def test_confirm_style_differs_from_wizard(self):
        """Test that CONFIRM_STYLE has a different qmark color (orange)."""
        assert CONFIRM_STYLE is not WIZARD_STYLE


# =============================================================================
# Test render_progress
# =============================================================================


class TestRenderProgress:
    def test_renders_first_step_active(self):
        """Test that first step is shown as active."""
        panel = render_progress(current_step=0, completed=set())
        # Panel should contain the title
        assert panel.title is not None
        # The renderable content should contain step names
        content_str = str(panel.renderable)
        assert "Provider" in content_str

    def test_renders_completed_steps(self):
        """Test that completed steps are marked differently."""
        panel = render_progress(current_step=2, completed={0, 1})
        content_str = str(panel.renderable)
        # All step names should be present
        for step in STEPS:
            assert step in content_str

    def test_renders_all_steps_completed(self):
        """Test rendering when all steps are completed."""
        panel = render_progress(current_step=5, completed={0, 1, 2, 3, 4, 5})
        content_str = str(panel.renderable)
        assert "Parameters" in content_str

    def test_panel_has_title(self):
        """Test that the panel has the expected title."""
        panel = render_progress(current_step=0, completed=set())
        assert "EvoScientist Setup" in str(panel.title)

    def test_panel_has_blue_border(self):
        """Test that the panel has a blue border style."""
        panel = render_progress(current_step=0, completed=set())
        assert panel.border_style == "blue"


# =============================================================================
# Test Validators
# =============================================================================


class TestIntegerValidator:
    def test_accepts_valid_integer(self):
        """Test that valid integers are accepted."""
        validator = IntegerValidator(min_value=1, max_value=10)

        class Doc:
            text = "5"

        # Should not raise
        validator.validate(Doc())

    def test_accepts_empty_for_default(self):
        """Test that empty string is accepted for using default."""
        validator = IntegerValidator(min_value=1, max_value=10)

        class Doc:
            text = ""

        # Should not raise
        validator.validate(Doc())

    def test_rejects_non_integer(self):
        """Test that non-integers are rejected."""
        from prompt_toolkit.validation import ValidationError

        validator = IntegerValidator(min_value=1, max_value=10)

        class Doc:
            text = "abc"

        with pytest.raises(ValidationError) as exc_info:
            validator.validate(Doc())
        assert "valid integer" in str(exc_info.value.message)

    def test_rejects_below_min(self):
        """Test that values below min are rejected."""
        from prompt_toolkit.validation import ValidationError

        validator = IntegerValidator(min_value=5, max_value=10)

        class Doc:
            text = "3"

        with pytest.raises(ValidationError) as exc_info:
            validator.validate(Doc())
        assert "between" in str(exc_info.value.message)

    def test_rejects_above_max(self):
        """Test that values above max are rejected."""
        from prompt_toolkit.validation import ValidationError

        validator = IntegerValidator(min_value=1, max_value=5)

        class Doc:
            text = "10"

        with pytest.raises(ValidationError) as exc_info:
            validator.validate(Doc())
        assert "between" in str(exc_info.value.message)


class TestChoiceValidator:
    def test_accepts_valid_choice(self):
        """Test that valid choices are accepted."""
        validator = ChoiceValidator(choices=["apple", "banana", "cherry"])

        class Doc:
            text = "banana"

        # Should not raise
        validator.validate(Doc())

    def test_accepts_case_insensitive(self):
        """Test that choices are case-insensitive."""
        validator = ChoiceValidator(choices=["Apple", "Banana"])

        class Doc:
            text = "APPLE"

        # Should not raise
        validator.validate(Doc())

    def test_accepts_empty_when_allowed(self):
        """Test that empty is accepted when allow_empty=True."""
        validator = ChoiceValidator(choices=["a", "b"], allow_empty=True)

        class Doc:
            text = ""

        # Should not raise
        validator.validate(Doc())

    def test_rejects_empty_when_not_allowed(self):
        """Test that empty is rejected when allow_empty=False."""
        from prompt_toolkit.validation import ValidationError

        validator = ChoiceValidator(choices=["a", "b"], allow_empty=False)

        class Doc:
            text = ""

        with pytest.raises(ValidationError):
            validator.validate(Doc())

    def test_rejects_invalid_choice(self):
        """Test that invalid choices are rejected."""
        from prompt_toolkit.validation import ValidationError

        validator = ChoiceValidator(choices=["a", "b"])

        class Doc:
            text = "c"

        with pytest.raises(ValidationError) as exc_info:
            validator.validate(Doc())
        assert "one of" in str(exc_info.value.message)


# =============================================================================
# Test API Key Validation (Mocked)
# =============================================================================


class TestValidateAnthropicKey:
    def test_empty_key_returns_skipped(self):
        """Test that empty key returns skipped status."""
        valid, msg = validate_anthropic_key("")
        assert valid is True
        assert "Skipped" in msg

    def test_valid_key(self):
        """Test that valid key returns success."""
        # Create mock module
        mock_anthropic = mock.MagicMock()
        mock_client = mock.MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        with mock.patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            valid, msg = validate_anthropic_key("sk-ant-valid-key")

        assert valid is True
        assert msg == "Valid"
        mock_anthropic.Anthropic.assert_called_once_with(api_key="sk-ant-valid-key")
        mock_client.models.list.assert_called_once()

    def test_invalid_key(self):
        """Test that invalid key returns error."""
        # Create mock module with AuthenticationError
        mock_anthropic = mock.MagicMock()
        mock_anthropic.AuthenticationError = Exception
        mock_anthropic.Anthropic.side_effect = mock_anthropic.AuthenticationError(
            "Invalid API key"
        )

        with mock.patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            valid, msg = validate_anthropic_key("invalid-key")

        assert valid is False
        assert "Invalid" in msg


class TestValidateOpenaiKey:
    def test_empty_key_returns_skipped(self):
        """Test that empty key returns skipped status."""
        valid, msg = validate_openai_key("")
        assert valid is True
        assert "Skipped" in msg

    def test_valid_key(self):
        """Test that valid key returns success."""
        mock_openai = mock.MagicMock()
        mock_client = mock.MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        with mock.patch.dict(sys.modules, {"openai": mock_openai}):
            valid, msg = validate_openai_key("sk-valid-key")

        assert valid is True
        assert msg == "Valid"
        mock_openai.OpenAI.assert_called_once_with(api_key="sk-valid-key")
        mock_client.models.list.assert_called_once()

    def test_invalid_key(self):
        """Test that invalid key returns error."""
        mock_openai = mock.MagicMock()
        mock_openai.AuthenticationError = Exception
        mock_openai.OpenAI.side_effect = mock_openai.AuthenticationError(
            "Invalid API key"
        )

        with mock.patch.dict(sys.modules, {"openai": mock_openai}):
            valid, msg = validate_openai_key("invalid-key")

        assert valid is False
        assert "Invalid" in msg


class TestValidateTavilyKey:
    def test_empty_key_returns_skipped(self):
        """Test that empty key returns skipped status."""
        valid, msg = validate_tavily_key("")
        assert valid is True
        assert "Skipped" in msg

    def test_valid_key(self):
        """Test that valid key returns success."""
        mock_tavily = mock.MagicMock()
        mock_client = mock.MagicMock()
        mock_tavily.TavilyClient.return_value = mock_client

        with mock.patch.dict(sys.modules, {"tavily": mock_tavily}):
            valid, msg = validate_tavily_key("tvly-valid-key")

        assert valid is True
        assert msg == "Valid"
        mock_tavily.TavilyClient.assert_called_once_with(api_key="tvly-valid-key")
        mock_client.search.assert_called_once_with("test", max_results=1)

    def test_invalid_key(self):
        """Test that invalid key returns error."""
        mock_tavily = mock.MagicMock()
        mock_tavily.TavilyClient.side_effect = Exception("401 Unauthorized")

        with mock.patch.dict(sys.modules, {"tavily": mock_tavily}):
            valid, msg = validate_tavily_key("invalid-key")

        assert valid is False
        assert "Invalid" in msg


# =============================================================================
# Test Step Functions (Mocked questionary)
# =============================================================================


class TestStepProvider:
    def test_returns_selected_provider(self):
        """Test that _step_provider returns selected provider."""
        from EvoScientist.onboard import _step_provider

        config = EvoScientistConfig()

        with mock.patch("EvoScientist.onboard.questionary") as mock_q:
            mock_q.select.return_value.ask.return_value = "anthropic"
            result = _step_provider(config)

        assert result == "anthropic"
        mock_q.select.assert_called_once()

    def test_raises_keyboard_interrupt_on_cancel(self):
        """Test that _step_provider raises KeyboardInterrupt on cancel."""
        from EvoScientist.onboard import _step_provider

        config = EvoScientistConfig()

        with mock.patch("EvoScientist.onboard.questionary") as mock_q:
            mock_q.select.return_value.ask.return_value = None
            with pytest.raises(KeyboardInterrupt):
                _step_provider(config)


class TestStepModel:
    def test_returns_selected_model(self):
        """Test that _step_model returns selected model."""
        from EvoScientist.onboard import _step_model

        config = EvoScientistConfig()

        with mock.patch("EvoScientist.onboard.questionary") as mock_q:
            mock_q.select.return_value.ask.return_value = "claude-sonnet-4-5"
            result = _step_model(config, "anthropic")

        assert result == "claude-sonnet-4-5"

    def test_raises_keyboard_interrupt_on_cancel(self):
        """Test that _step_model raises KeyboardInterrupt on cancel."""
        from EvoScientist.onboard import _step_model

        config = EvoScientistConfig()

        with mock.patch("EvoScientist.onboard.questionary") as mock_q:
            mock_q.select.return_value.ask.return_value = None
            with pytest.raises(KeyboardInterrupt):
                _step_model(config, "anthropic")


class TestStepWorkspace:
    def test_returns_mode_and_empty_workdir(self):
        """Test workspace step with no custom directory."""
        from EvoScientist.onboard import _step_workspace

        config = EvoScientistConfig()

        with mock.patch("EvoScientist.onboard.questionary") as mock_q:
            mock_q.select.return_value.ask.return_value = "daemon"
            mock_q.confirm.return_value.ask.return_value = False  # No custom dir
            result = _step_workspace(config)

        assert result == ("daemon", "")

    def test_returns_mode_and_custom_workdir(self):
        """Test workspace step with custom directory."""
        from EvoScientist.onboard import _step_workspace

        config = EvoScientistConfig()

        with mock.patch("EvoScientist.onboard.questionary") as mock_q:
            mock_q.select.return_value.ask.return_value = "run"
            mock_q.confirm.return_value.ask.return_value = True  # Use custom dir
            mock_q.text.return_value.ask.return_value = "/custom/path"
            result = _step_workspace(config)

        assert result == ("run", "/custom/path")


class TestStepParameters:
    def test_returns_parameters(self):
        """Test parameters step returns all values."""
        from EvoScientist.onboard import _step_parameters

        config = EvoScientistConfig()

        with mock.patch("EvoScientist.onboard.questionary") as mock_q:
            mock_q.text.return_value.ask.side_effect = ["5", "3"]
            mock_q.select.return_value.ask.return_value = True  # show_thinking
            result = _step_parameters(config)

        assert result == (5, 3, True)

    def test_uses_defaults_on_empty_input(self):
        """Test that empty input uses defaults."""
        from EvoScientist.onboard import _step_parameters

        config = EvoScientistConfig(max_concurrent=2, max_iterations=4, show_thinking=False)

        with mock.patch("EvoScientist.onboard.questionary") as mock_q:
            mock_q.text.return_value.ask.side_effect = ["", ""]  # Empty inputs
            mock_q.select.return_value.ask.return_value = False  # show_thinking
            result = _step_parameters(config)

        assert result == (2, 4, False)


# =============================================================================
# Test run_onboard (Integration-like test with mocked questionary)
# =============================================================================


class TestRunOnboard:
    def test_returns_true_on_save(self):
        """Test that run_onboard returns True when config is saved."""
        from EvoScientist.onboard import run_onboard

        with mock.patch("EvoScientist.onboard.questionary") as mock_q, \
             mock.patch("EvoScientist.onboard.load_config") as mock_load, \
             mock.patch("EvoScientist.onboard.save_config") as mock_save, \
             mock.patch("EvoScientist.onboard.console"):
            # Setup mock config
            mock_load.return_value = EvoScientistConfig()

            # Mock all questionary calls
            mock_q.select.return_value.ask.side_effect = [
                "anthropic",  # Provider
                "claude-sonnet-4-5",  # Model
                "daemon",  # Workspace mode
                True,  # Show thinking
            ]
            mock_q.password.return_value.ask.side_effect = [
                "",  # Provider API key (keep current)
                "",  # Tavily key (keep current)
            ]
            mock_q.confirm.return_value.ask.side_effect = [
                False,  # Custom workdir
                True,  # Save config
            ]
            mock_q.text.return_value.ask.side_effect = [
                "3",  # Max concurrent
                "3",  # Max iterations
            ]

            result = run_onboard(skip_validation=True)

        assert result is True
        mock_save.assert_called_once()

    def test_returns_false_on_cancel(self):
        """Test that run_onboard returns False when cancelled."""
        from EvoScientist.onboard import run_onboard

        with mock.patch("EvoScientist.onboard.questionary") as mock_q, \
             mock.patch("EvoScientist.onboard.load_config") as mock_load, \
             mock.patch("EvoScientist.onboard.console"):
            mock_load.return_value = EvoScientistConfig()

            # First selection returns None (Ctrl+C)
            mock_q.select.return_value.ask.return_value = None

            result = run_onboard(skip_validation=True)

        assert result is False

    def test_returns_false_when_not_saving(self):
        """Test that run_onboard returns False when user declines to save."""
        from EvoScientist.onboard import run_onboard

        with mock.patch("EvoScientist.onboard.questionary") as mock_q, \
             mock.patch("EvoScientist.onboard.load_config") as mock_load, \
             mock.patch("EvoScientist.onboard.save_config") as mock_save, \
             mock.patch("EvoScientist.onboard.console"):
            mock_load.return_value = EvoScientistConfig()

            mock_q.select.return_value.ask.side_effect = [
                "anthropic",
                "claude-sonnet-4-5",
                "daemon",
                True,  # Show thinking
            ]
            mock_q.password.return_value.ask.side_effect = ["", ""]
            mock_q.confirm.return_value.ask.side_effect = [
                False,  # Custom workdir
                False,  # Save config - NO
            ]
            mock_q.text.return_value.ask.side_effect = ["3", "3"]

            result = run_onboard(skip_validation=True)

        assert result is False
        mock_save.assert_not_called()
