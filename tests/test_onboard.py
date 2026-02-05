"""Tests for EvoScientist onboarding wizard."""

import subprocess
from unittest import mock

import pytest

from EvoScientist.onboard import (
    IntegerValidator,
    ChoiceValidator,
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
    def test_steps_has_seven_items(self):
        """Test that STEPS contains exactly 7 steps."""
        assert len(STEPS) == 7
        assert STEPS == ["Provider", "API Key", "Model", "Tavily Key", "Workspace", "Parameters", "Channels"]

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


class TestValidateImessage:
    def test_valid_when_cli_found_with_rpc(self):
        """Test validate_imessage returns valid when imsg CLI found and RPC works."""
        from EvoScientist.onboard import validate_imessage

        version_result = mock.Mock(returncode=0, stdout="imsg 1.2.3")
        rpc_result = mock.Mock(returncode=0)

        with mock.patch("EvoScientist.onboard.sys") as mock_sys, \
             mock.patch("EvoScientist.channels.imessage.probe.shutil") as mock_shutil, \
             mock.patch("EvoScientist.onboard.subprocess") as mock_sub:
            mock_sys.platform = "darwin"
            mock_shutil.which.return_value = "/opt/homebrew/bin/imsg"
            mock_sub.run.side_effect = [version_result, rpc_result]
            valid, msg = validate_imessage()

        assert valid is True
        assert "imsg" in msg
        assert "1.2.3" in msg

    def test_invalid_when_cli_not_found(self):
        """Test validate_imessage returns not_installed when imsg CLI missing."""
        from EvoScientist.onboard import validate_imessage

        with mock.patch("EvoScientist.onboard.sys") as mock_sys, \
             mock.patch("EvoScientist.channels.imessage.probe.shutil") as mock_shutil:
            mock_sys.platform = "darwin"
            mock_shutil.which.return_value = None
            valid, msg = validate_imessage()

        assert valid is False
        assert msg == "not_installed"

    def test_invalid_on_non_macos(self):
        """Test validate_imessage returns invalid on non-macOS."""
        from EvoScientist.onboard import validate_imessage

        with mock.patch("EvoScientist.onboard.sys") as mock_sys:
            mock_sys.platform = "linux"
            valid, msg = validate_imessage()

        assert valid is False
        assert "macOS" in msg

    def test_invalid_when_rpc_not_supported(self):
        """Test validate_imessage returns invalid when RPC check fails."""
        from EvoScientist.onboard import validate_imessage

        version_result = mock.Mock(returncode=0, stdout="imsg 0.1.0")
        rpc_result = mock.Mock(returncode=1)

        with mock.patch("EvoScientist.onboard.sys") as mock_sys, \
             mock.patch("EvoScientist.channels.imessage.probe.shutil") as mock_shutil, \
             mock.patch("EvoScientist.onboard.subprocess") as mock_sub:
            mock_sys.platform = "darwin"
            mock_shutil.which.return_value = "/usr/local/bin/imsg"
            mock_sub.run.side_effect = [version_result, rpc_result]
            valid, msg = validate_imessage()

        assert valid is False
        assert "RPC not supported" in msg


class TestInstallImsg:
    def test_install_success(self):
        """Test _install_imsg returns True on success."""
        from EvoScientist.onboard import _install_imsg

        with mock.patch("EvoScientist.onboard.subprocess") as mock_sub:
            mock_sub.run.return_value = mock.Mock(returncode=0)
            mock_sub.TimeoutExpired = subprocess.TimeoutExpired
            result = _install_imsg()

        assert result is True

    def test_install_brew_not_found(self):
        """Test _install_imsg handles missing Homebrew."""
        from EvoScientist.onboard import _install_imsg

        with mock.patch("EvoScientist.onboard.subprocess") as mock_sub, \
             mock.patch("EvoScientist.onboard.console"):
            mock_sub.run.side_effect = FileNotFoundError()
            mock_sub.TimeoutExpired = subprocess.TimeoutExpired
            result = _install_imsg()

        assert result is False

    def test_install_failure(self):
        """Test _install_imsg returns False on non-zero exit."""
        from EvoScientist.onboard import _install_imsg

        with mock.patch("EvoScientist.onboard.subprocess") as mock_sub:
            mock_sub.run.return_value = mock.Mock(returncode=1)
            mock_sub.TimeoutExpired = subprocess.TimeoutExpired
            result = _install_imsg()

        assert result is False


class TestSetupImessage:
    def test_already_installed(self):
        """Test _setup_imessage returns True when already installed."""
        from EvoScientist.onboard import _setup_imessage

        with mock.patch("EvoScientist.onboard.validate_imessage", return_value=(True, "imsg at /bin/imsg")), \
             mock.patch("EvoScientist.onboard.console"):
            result = _setup_imessage()

        assert result is True

    def test_not_macos(self):
        """Test _setup_imessage returns False on non-macOS."""
        from EvoScientist.onboard import _setup_imessage

        with mock.patch("EvoScientist.onboard.validate_imessage", return_value=(False, "iMessage requires macOS")), \
             mock.patch("EvoScientist.onboard.console"):
            result = _setup_imessage()

        assert result is False

    def test_install_then_valid(self):
        """Test _setup_imessage installs and re-validates successfully."""
        from EvoScientist.onboard import _setup_imessage

        with mock.patch("EvoScientist.onboard.validate_imessage") as mock_val, \
             mock.patch("EvoScientist.onboard._install_imsg", return_value=True), \
             mock.patch("EvoScientist.onboard.questionary") as mock_q, \
             mock.patch("EvoScientist.onboard.console"):
            mock_val.side_effect = [
                (False, "not_installed"),  # First check
                (True, "imsg at /bin/imsg"),  # After install
            ]
            mock_q.confirm.return_value.ask.return_value = True  # Yes, install
            result = _setup_imessage()

        assert result is True

    def test_user_declines_install(self):
        """Test _setup_imessage returns False when user declines install."""
        from EvoScientist.onboard import _setup_imessage

        with mock.patch("EvoScientist.onboard.validate_imessage", return_value=(False, "not_installed")), \
             mock.patch("EvoScientist.onboard.questionary") as mock_q, \
             mock.patch("EvoScientist.onboard.console"):
            mock_q.confirm.return_value.ask.return_value = False
            result = _setup_imessage()

        assert result is False


class TestStepChannels:
    def test_returns_disabled_when_skip(self):
        """Test channels step returns disabled when user selects skip."""
        from EvoScientist.onboard import _step_channels

        config = EvoScientistConfig()

        with mock.patch("EvoScientist.onboard.questionary") as mock_q:
            mock_q.select.return_value.ask.return_value = "skip"
            result = _step_channels(config)

        assert result == (False, "")

    def test_returns_enabled_when_setup_passes(self):
        """Test channels step returns enabled when setup succeeds."""
        from EvoScientist.onboard import _step_channels

        config = EvoScientistConfig()

        with mock.patch("EvoScientist.onboard.questionary") as mock_q, \
             mock.patch("EvoScientist.onboard._setup_imessage", return_value=True):
            mock_q.select.return_value.ask.return_value = "imessage"
            mock_q.text.return_value.ask.return_value = ""
            result = _step_channels(config)

        assert result == (True, "")

    def test_returns_enabled_with_senders(self):
        """Test channels step returns enabled with specific senders."""
        from EvoScientist.onboard import _step_channels

        config = EvoScientistConfig()

        with mock.patch("EvoScientist.onboard.questionary") as mock_q, \
             mock.patch("EvoScientist.onboard._setup_imessage", return_value=True):
            mock_q.select.return_value.ask.return_value = "imessage"
            mock_q.text.return_value.ask.return_value = "+1234567890,+0987654321"
            result = _step_channels(config)

        assert result == (True, "+1234567890,+0987654321")

    def test_setup_fails_user_declines(self):
        """Test channels step returns disabled when setup fails and user declines."""
        from EvoScientist.onboard import _step_channels

        config = EvoScientistConfig()

        with mock.patch("EvoScientist.onboard.questionary") as mock_q, \
             mock.patch("EvoScientist.onboard._setup_imessage", return_value=False):
            mock_q.select.return_value.ask.return_value = "imessage"
            mock_q.confirm.return_value.ask.return_value = False
            result = _step_channels(config)

        assert result == (False, "")

    def test_setup_fails_user_enables_anyway(self):
        """Test channels step enables when setup fails but user confirms."""
        from EvoScientist.onboard import _step_channels

        config = EvoScientistConfig()

        with mock.patch("EvoScientist.onboard.questionary") as mock_q, \
             mock.patch("EvoScientist.onboard._setup_imessage", return_value=False):
            mock_q.select.return_value.ask.return_value = "imessage"
            mock_q.confirm.return_value.ask.return_value = True
            mock_q.text.return_value.ask.return_value = ""
            result = _step_channels(config)

        assert result == (True, "")

    def test_raises_keyboard_interrupt_on_cancel(self):
        """Test channels step raises KeyboardInterrupt on cancel."""
        from EvoScientist.onboard import _step_channels

        config = EvoScientistConfig()

        with mock.patch("EvoScientist.onboard.questionary") as mock_q:
            mock_q.select.return_value.ask.return_value = None
            with pytest.raises(KeyboardInterrupt):
                _step_channels(config)


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
                "skip",  # Channels: skip
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
                "skip",  # Channels: skip
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
