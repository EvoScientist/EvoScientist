"""Tests for EvoScientist onboarding wizard."""

import subprocess
from unittest.mock import patch, Mock

import pytest

from EvoScientist.config.onboard import (
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
    def test_steps_has_ten_items(self):
        """Test that STEPS contains exactly 10 steps."""
        assert len(STEPS) == 10
        assert STEPS == [
            "UI",
            "Provider",
            "API Key",
            "Model",
            "Tavily Key",
            "Workspace",
            "Thinking",
            "Skills",
            "MCP Servers",
            "Channels",
        ]

    def test_wizard_style_is_style_instance(self):
        """Test that WIZARD_STYLE is a prompt_toolkit Style."""
        from prompt_toolkit.styles import Style

        assert isinstance(WIZARD_STYLE, Style)

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
        from EvoScientist.config.onboard import _step_provider

        config = EvoScientistConfig()

        with patch("EvoScientist.config.onboard.questionary") as mock_q:
            mock_q.select.return_value.ask.return_value = "anthropic"
            result = _step_provider(config)

        assert result == "anthropic"
        mock_q.select.assert_called_once()

    def test_raises_keyboard_interrupt_on_cancel(self):
        """Test that _step_provider raises KeyboardInterrupt on cancel."""
        from EvoScientist.config.onboard import _step_provider

        config = EvoScientistConfig()

        with patch("EvoScientist.config.onboard.questionary") as mock_q:
            mock_q.select.return_value.ask.return_value = None
            with pytest.raises(KeyboardInterrupt):
                _step_provider(config)


class TestStepModel:
    def test_returns_selected_model(self):
        """Test that _step_model returns selected model."""
        from EvoScientist.config.onboard import _step_model

        config = EvoScientistConfig()

        with patch("EvoScientist.config.onboard.questionary") as mock_q:
            mock_q.select.return_value.ask.return_value = "claude-sonnet-4-5"
            result = _step_model(config, "anthropic")

        assert result == "claude-sonnet-4-5"

    def test_raises_keyboard_interrupt_on_cancel(self):
        """Test that _step_model raises KeyboardInterrupt on cancel."""
        from EvoScientist.config.onboard import _step_model

        config = EvoScientistConfig()

        with patch("EvoScientist.config.onboard.questionary") as mock_q:
            mock_q.select.return_value.ask.return_value = None
            with pytest.raises(KeyboardInterrupt):
                _step_model(config, "anthropic")


class TestStepWorkspace:
    def test_returns_daemon_mode(self):
        """Test workspace step returns selected mode."""
        from EvoScientist.config.onboard import _step_workspace

        config = EvoScientistConfig()

        with patch("EvoScientist.config.onboard.questionary") as mock_q:
            mock_q.select.return_value.ask.return_value = "daemon"
            result = _step_workspace(config)

        assert result == "daemon"

    def test_returns_run_mode(self):
        """Test workspace step returns run mode."""
        from EvoScientist.config.onboard import _step_workspace

        config = EvoScientistConfig()

        with patch("EvoScientist.config.onboard.questionary") as mock_q:
            mock_q.select.return_value.ask.return_value = "run"
            result = _step_workspace(config)

        assert result == "run"


class TestPromptAndValidateApiKey:
    def test_keep_existing_key_still_validates(self):
        """Pressing Enter to keep current key should validate the existing key."""
        from EvoScientist.config.onboard import _prompt_and_validate_api_key

        validate_fn = Mock(return_value=(True, "Valid"))

        with (
            patch("EvoScientist.config.onboard.questionary") as mock_q,
            patch("EvoScientist.config.onboard.console"),
        ):
            mock_q.password.return_value.ask.return_value = ""  # keep existing
            result = _prompt_and_validate_api_key(
                "Enter key:",
                current="existing-key",
                validate_fn=validate_fn,
                skip_validation=False,
            )

        assert result is None  # None means "keep existing, don't overwrite"
        validate_fn.assert_called_once_with("existing-key")

    def test_new_key_still_validates(self):
        """Entering a new key should still run validation."""
        from EvoScientist.config.onboard import _prompt_and_validate_api_key

        validate_fn = Mock(return_value=(True, "valid"))

        with patch("EvoScientist.config.onboard.questionary") as mock_q:
            mock_q.password.return_value.ask.return_value = "new-key"
            result = _prompt_and_validate_api_key(
                "Enter key:",
                current="old-key",
                validate_fn=validate_fn,
                skip_validation=False,
            )

        assert result == "new-key"
        validate_fn.assert_called_once_with("new-key")


class TestValidateImessage:
    def test_valid_when_cli_found_with_rpc(self):
        """Test validate_imessage returns valid when imsg CLI found and RPC works."""
        from EvoScientist.config.onboard import validate_imessage

        version_result = Mock(returncode=0, stdout="imsg 1.2.3")
        rpc_result = Mock(returncode=0)

        with (
            patch("EvoScientist.config.onboard.sys") as mock_sys,
            patch("EvoScientist.channels.imessage.probe.shutil") as mock_shutil,
            patch("EvoScientist.config.onboard.subprocess") as mock_sub,
        ):
            mock_sys.platform = "darwin"
            mock_shutil.which.return_value = "/opt/homebrew/bin/imsg"
            mock_sub.run.side_effect = [version_result, rpc_result]
            valid, msg = validate_imessage()

        assert valid is True
        assert "imsg" in msg
        assert "1.2.3" in msg

    def test_invalid_when_cli_not_found(self):
        """Test validate_imessage returns not_installed when imsg CLI missing."""
        from EvoScientist.config.onboard import validate_imessage

        with (
            patch("EvoScientist.config.onboard.sys") as mock_sys,
            patch("EvoScientist.channels.imessage.probe.shutil") as mock_shutil,
        ):
            mock_sys.platform = "darwin"
            mock_shutil.which.return_value = None
            valid, msg = validate_imessage()

        assert valid is False
        assert msg == "not_installed"

    def test_invalid_on_non_macos(self):
        """Test validate_imessage returns invalid on non-macOS."""
        from EvoScientist.config.onboard import validate_imessage

        with patch("EvoScientist.config.onboard.sys") as mock_sys:
            mock_sys.platform = "linux"
            valid, msg = validate_imessage()

        assert valid is False
        assert "macOS" in msg

    def test_invalid_when_rpc_not_supported(self):
        """Test validate_imessage returns invalid when RPC check fails."""
        from EvoScientist.config.onboard import validate_imessage

        version_result = Mock(returncode=0, stdout="imsg 0.1.0")
        rpc_result = Mock(returncode=1)

        with (
            patch("EvoScientist.config.onboard.sys") as mock_sys,
            patch("EvoScientist.channels.imessage.probe.shutil") as mock_shutil,
            patch("EvoScientist.config.onboard.subprocess") as mock_sub,
        ):
            mock_sys.platform = "darwin"
            mock_shutil.which.return_value = "/usr/local/bin/imsg"
            mock_sub.run.side_effect = [version_result, rpc_result]
            valid, msg = validate_imessage()

        assert valid is False
        assert "RPC not supported" in msg


class TestInstallImsg:
    def test_install_success(self):
        """Test _install_imsg returns True on success."""
        from EvoScientist.config.onboard import _install_imsg

        with patch("EvoScientist.config.onboard.subprocess") as mock_sub:
            mock_sub.run.return_value = Mock(returncode=0)
            mock_sub.TimeoutExpired = subprocess.TimeoutExpired
            result = _install_imsg()

        assert result is True

    def test_install_brew_not_found(self):
        """Test _install_imsg handles missing Homebrew."""
        from EvoScientist.config.onboard import _install_imsg

        with (
            patch("EvoScientist.config.onboard.subprocess") as mock_sub,
            patch("EvoScientist.config.onboard.console"),
        ):
            mock_sub.run.side_effect = FileNotFoundError()
            mock_sub.TimeoutExpired = subprocess.TimeoutExpired
            result = _install_imsg()

        assert result is False

    def test_install_failure(self):
        """Test _install_imsg returns False on non-zero exit."""
        from EvoScientist.config.onboard import _install_imsg

        with patch("EvoScientist.config.onboard.subprocess") as mock_sub:
            mock_sub.run.return_value = Mock(returncode=1)
            mock_sub.TimeoutExpired = subprocess.TimeoutExpired
            result = _install_imsg()

        assert result is False


class TestSetupImessage:
    def test_already_installed(self):
        """Test _setup_imessage returns True when already installed."""
        from EvoScientist.config.onboard import _setup_imessage

        with (
            patch(
                "EvoScientist.config.onboard.validate_imessage",
                return_value=(True, "imsg at /bin/imsg"),
            ),
            patch("EvoScientist.config.onboard.console"),
        ):
            result = _setup_imessage()

        assert result is True

    def test_not_macos(self):
        """Test _setup_imessage returns False on non-macOS."""
        from EvoScientist.config.onboard import _setup_imessage

        with (
            patch(
                "EvoScientist.config.onboard.validate_imessage",
                return_value=(False, "iMessage requires macOS"),
            ),
            patch("EvoScientist.config.onboard.console"),
        ):
            result = _setup_imessage()

        assert result is False

    def test_install_then_valid(self):
        """Test _setup_imessage installs and re-validates successfully."""
        from EvoScientist.config.onboard import _setup_imessage

        with (
            patch("EvoScientist.config.onboard.validate_imessage") as mock_val,
            patch("EvoScientist.config.onboard._install_imsg", return_value=True),
            patch("EvoScientist.config.onboard.questionary") as mock_q,
            patch("EvoScientist.config.onboard.console"),
        ):
            mock_val.side_effect = [
                (False, "not_installed"),  # First check
                (True, "imsg at /bin/imsg"),  # After install
            ]
            mock_q.confirm.return_value.ask.return_value = True  # Yes, install
            result = _setup_imessage()

        assert result is True

    def test_user_declines_install(self):
        """Test _setup_imessage returns False when user declines install."""
        from EvoScientist.config.onboard import _setup_imessage

        with (
            patch(
                "EvoScientist.config.onboard.validate_imessage",
                return_value=(False, "not_installed"),
            ),
            patch("EvoScientist.config.onboard.questionary") as mock_q,
            patch("EvoScientist.config.onboard.console"),
        ):
            mock_q.confirm.return_value.ask.return_value = False
            result = _setup_imessage()

        assert result is False


class TestStepSkills:
    def test_returns_empty_when_none_selected(self):
        """Test skills step returns empty list when user selects nothing."""
        from EvoScientist.config.onboard import _step_skills

        with (
            patch("EvoScientist.config.onboard.questionary") as mock_q,
            patch("EvoScientist.config.onboard.console"),
        ):
            mock_q.checkbox.return_value.ask.return_value = []
            result = _step_skills()

        assert result == []

    def test_installs_selected_skills(self):
        """Test skills step installs selected skills and returns sources."""
        from EvoScientist.config.onboard import _step_skills, _RECOMMENDED_SKILLS

        source = _RECOMMENDED_SKILLS[0]["source"]

        with (
            patch("EvoScientist.config.onboard.questionary") as mock_q,
            patch("EvoScientist.config.onboard.console"),
            patch("EvoScientist.tools.skills_manager.install_skill") as mock_install,
        ):
            mock_q.checkbox.return_value.ask.return_value = [source]
            mock_install.return_value = {"success": True, "name": "test"}
            result = _step_skills()

        assert result == [source]
        mock_install.assert_called_once_with(source)

    def test_handles_install_failure(self):
        """Test skills step handles installation errors gracefully."""
        from EvoScientist.config.onboard import _step_skills, _RECOMMENDED_SKILLS

        source = _RECOMMENDED_SKILLS[0]["source"]

        with (
            patch("EvoScientist.config.onboard.questionary") as mock_q,
            patch("EvoScientist.config.onboard.console"),
            patch("EvoScientist.tools.skills_manager.install_skill") as mock_install,
        ):
            mock_q.checkbox.return_value.ask.return_value = [source]
            mock_install.side_effect = Exception("network error")
            result = _step_skills()

        assert result == []

    def test_raises_keyboard_interrupt_on_cancel(self):
        """Test skills step raises KeyboardInterrupt on cancel."""
        from EvoScientist.config.onboard import _step_skills

        with patch("EvoScientist.config.onboard.questionary") as mock_q:
            mock_q.checkbox.return_value.ask.return_value = None
            with pytest.raises(KeyboardInterrupt):
                _step_skills()


class TestStepChannels:
    def test_returns_disabled_when_skip(self):
        """Test channels step returns empty dict when user selects nothing."""
        from EvoScientist.config.onboard import _step_channels

        config = EvoScientistConfig()

        with patch("EvoScientist.config.onboard.questionary") as mock_q:
            mock_q.checkbox.return_value.ask.return_value = []
            result = _step_channels(config)

        assert result == {"channel_enabled": "", "imessage_enabled": False}

    def test_returns_enabled_when_setup_passes(self):
        """Test channels step returns enabled when iMessage setup succeeds."""
        from EvoScientist.config.onboard import _step_channels

        config = EvoScientistConfig()

        with (
            patch("EvoScientist.config.onboard.questionary") as mock_q,
            patch("EvoScientist.config.onboard._setup_imessage", return_value=True),
        ):
            mock_q.checkbox.return_value.ask.return_value = ["imessage"]
            mock_q.text.return_value.ask.return_value = ""
            result = _step_channels(config)

        assert result["channel_enabled"] == "imessage"
        assert result["imessage_enabled"] is True

    def test_returns_enabled_with_senders(self):
        """Test channels step returns enabled with specific senders."""
        from EvoScientist.config.onboard import _step_channels

        config = EvoScientistConfig()

        with (
            patch("EvoScientist.config.onboard.questionary") as mock_q,
            patch("EvoScientist.config.onboard._setup_imessage", return_value=True),
        ):
            mock_q.checkbox.return_value.ask.return_value = ["imessage"]
            mock_q.text.return_value.ask.return_value = "+1234567890,+0987654321"
            result = _step_channels(config)

        assert result["channel_enabled"] == "imessage"
        assert result["imessage_enabled"] is True
        assert result["imessage_allowed_senders"] == "+1234567890,+0987654321"

    def test_setup_fails_user_declines(self):
        """Test channels step skips iMessage when setup fails and user declines."""
        from EvoScientist.config.onboard import _step_channels

        config = EvoScientistConfig()

        with (
            patch("EvoScientist.config.onboard.questionary") as mock_q,
            patch("EvoScientist.config.onboard._setup_imessage", return_value=False),
        ):
            mock_q.checkbox.return_value.ask.return_value = ["imessage"]
            mock_q.confirm.return_value.ask.return_value = False
            result = _step_channels(config)

        assert result["channel_enabled"] == ""
        assert result["imessage_enabled"] is False

    def test_setup_fails_user_enables_anyway(self):
        """Test channels step enables iMessage when setup fails but user confirms."""
        from EvoScientist.config.onboard import _step_channels

        config = EvoScientistConfig()

        with (
            patch("EvoScientist.config.onboard.questionary") as mock_q,
            patch("EvoScientist.config.onboard._setup_imessage", return_value=False),
        ):
            mock_q.checkbox.return_value.ask.return_value = ["imessage"]
            mock_q.confirm.return_value.ask.return_value = True
            mock_q.text.return_value.ask.return_value = ""
            result = _step_channels(config)

        assert result["channel_enabled"] == "imessage"
        assert result["imessage_enabled"] is True

    def test_raises_keyboard_interrupt_on_cancel(self):
        """Test channels step raises KeyboardInterrupt on cancel."""
        from EvoScientist.config.onboard import _step_channels

        config = EvoScientistConfig()

        with patch("EvoScientist.config.onboard.questionary") as mock_q:
            mock_q.checkbox.return_value.ask.return_value = None
            with pytest.raises(KeyboardInterrupt):
                _step_channels(config)

    def test_telegram_channel_selected(self):
        """Test channels step handles Telegram selection."""
        from EvoScientist.config.onboard import _step_channels

        config = EvoScientistConfig()

        _real_import = (
            __builtins__.__import__
            if hasattr(__builtins__, "__import__")
            else __import__
        )

        def _fake_import(name, *args, **kwargs):
            if name == "telegram":
                return  # pretend installed
            return _real_import(name, *args, **kwargs)

        with (
            patch("EvoScientist.config.onboard.questionary") as mock_q,
            patch("EvoScientist.config.onboard._probe_channel"),
            patch("builtins.__import__", side_effect=_fake_import),
        ):
            mock_q.checkbox.return_value.ask.return_value = ["telegram"]
            mock_q.text.return_value.ask.return_value = "test-token"
            result = _step_channels(config)

        assert result["channel_enabled"] == "telegram"
        assert result["telegram_bot_token"] == "test-token"

    def test_discord_channel_selected(self):
        """Test channels step handles Discord selection."""
        from EvoScientist.config.onboard import _step_channels

        config = EvoScientistConfig()

        _real_import = (
            __builtins__.__import__
            if hasattr(__builtins__, "__import__")
            else __import__
        )

        def _fake_import(name, *args, **kwargs):
            if name == "discord":
                return  # pretend installed
            return _real_import(name, *args, **kwargs)

        with (
            patch("EvoScientist.config.onboard.questionary") as mock_q,
            patch("EvoScientist.config.onboard._probe_channel"),
            patch("builtins.__import__", side_effect=_fake_import),
        ):
            mock_q.checkbox.return_value.ask.return_value = ["discord"]
            mock_q.text.return_value.ask.return_value = "discord-token"
            result = _step_channels(config)

        assert result["channel_enabled"] == "discord"
        assert result["discord_bot_token"] == "discord-token"


class TestStepMcpServersNpxFailure:
    def _make_test_servers(self):
        from EvoScientist.mcp.registry import MCPServerEntry

        return [
            MCPServerEntry(
                name="npx-server",
                label="NPX Server",
                tags=["onboarding"],
                command="npx",
                args=["-y", "test-server"],
            ),
            MCPServerEntry(
                name="url-server",
                label="URL Server",
                tags=["onboarding"],
                transport="streamable_http",
                url="https://example.com/mcp",
            ),
        ]

    def test_npx_failure_skips_npx_servers(self):
        """When _ensure_npx returns False, npx-dependent servers must be skipped."""
        from EvoScientist.config.onboard import _step_mcp_servers

        servers = self._make_test_servers()

        with (
            patch(
                "EvoScientist.mcp.registry.fetch_marketplace_index",
                return_value=servers,
            ),
            patch(
                "EvoScientist.config.onboard._checkbox_ask",
                return_value=["npx-server", "url-server"],
            ),
            patch("EvoScientist.config.onboard._ensure_npx", return_value=False),
            patch("EvoScientist.config.onboard._check_npx", return_value=False),
            patch("EvoScientist.mcp.client._load_user_config", return_value={}),
            patch("EvoScientist.mcp.client.add_mcp_server") as mock_add,
            patch("EvoScientist.config.onboard.console"),
        ):
            result = _step_mcp_servers()

        # The npx server must NOT have been added
        added_names = [call.args[0] for call in mock_add.call_args_list]
        assert "npx-server" not in added_names
        # The URL server should still be added
        assert "url-server" in added_names
        assert "url-server" in result
        assert "npx-server" not in result

    def test_npx_failure_returns_empty_when_all_npx(self):
        """When all selected servers are npx-based and npx fails, return []."""
        from EvoScientist.config.onboard import _step_mcp_servers

        servers = self._make_test_servers()
        npx_names = [s.name for s in servers if s.command == "npx"]

        with (
            patch(
                "EvoScientist.mcp.registry.fetch_marketplace_index",
                return_value=servers,
            ),
            patch("EvoScientist.config.onboard._checkbox_ask", return_value=npx_names),
            patch("EvoScientist.config.onboard._ensure_npx", return_value=False),
            patch("EvoScientist.config.onboard._check_npx", return_value=False),
            patch("EvoScientist.mcp.client._load_user_config", return_value={}),
            patch("EvoScientist.mcp.client.add_mcp_server") as mock_add,
            patch("EvoScientist.config.onboard.console"),
        ):
            result = _step_mcp_servers()

        assert result == []
        mock_add.assert_not_called()


class TestStepThinking:
    def test_returns_show_thinking(self):
        """Test thinking step returns selected value."""
        from EvoScientist.config.onboard import _step_thinking

        config = EvoScientistConfig()

        with patch("EvoScientist.config.onboard.questionary") as mock_q:
            mock_q.select.return_value.ask.return_value = True
            result = _step_thinking(config)

        assert result is True

    def test_returns_false_when_off(self):
        """Test thinking step returns False when user selects Off."""
        from EvoScientist.config.onboard import _step_thinking

        config = EvoScientistConfig(show_thinking=False)

        with patch("EvoScientist.config.onboard.questionary") as mock_q:
            mock_q.select.return_value.ask.return_value = False
            result = _step_thinking(config)

        assert result is False


# =============================================================================
# Test run_onboard (Integration-like test with mocked questionary)
# =============================================================================


class TestRunOnboard:
    def test_returns_true_on_save(self):
        """Test that run_onboard returns True when config is saved."""
        from EvoScientist.config.onboard import run_onboard

        with (
            patch("EvoScientist.config.onboard.questionary") as mock_q,
            patch("EvoScientist.config.onboard.load_config") as mock_load,
            patch("EvoScientist.config.onboard.save_config") as mock_save,
            patch("EvoScientist.config.onboard.console"),
        ):
            # Setup mock config
            mock_load.return_value = EvoScientistConfig()

            # Mock all questionary calls
            mock_q.select.return_value.ask.side_effect = [
                "tui",  # UI backend
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
                True,  # Save config
            ]
            mock_q.text.return_value.ask.side_effect = [
                "",  # Workspace directory (empty = use cwd)
            ]
            mock_q.checkbox.return_value.ask.return_value = []  # Skills: skip

            result = run_onboard(skip_validation=True)

        assert result is True
        mock_save.assert_called_once()

    def test_returns_false_on_cancel(self):
        """Test that run_onboard returns False when cancelled."""
        from EvoScientist.config.onboard import run_onboard

        with (
            patch("EvoScientist.config.onboard.questionary") as mock_q,
            patch("EvoScientist.config.onboard.load_config") as mock_load,
            patch("EvoScientist.config.onboard.console"),
        ):
            mock_load.return_value = EvoScientistConfig()

            # First selection returns None (Ctrl+C)
            mock_q.select.return_value.ask.return_value = None

            result = run_onboard(skip_validation=True)

        assert result is False

    def test_returns_false_when_not_saving(self):
        """Test that run_onboard returns False when user declines to save."""
        from EvoScientist.config.onboard import run_onboard

        with (
            patch("EvoScientist.config.onboard.questionary") as mock_q,
            patch("EvoScientist.config.onboard.load_config") as mock_load,
            patch("EvoScientist.config.onboard.save_config") as mock_save,
            patch("EvoScientist.config.onboard.console"),
        ):
            mock_load.return_value = EvoScientistConfig()

            mock_q.select.return_value.ask.side_effect = [
                "tui",  # UI backend
                "anthropic",
                "claude-sonnet-4-5",
                "daemon",
                True,  # Show thinking
                "skip",  # Channels: skip
            ]
            mock_q.password.return_value.ask.side_effect = ["", ""]
            mock_q.confirm.return_value.ask.side_effect = [
                False,  # Save config - NO
            ]
            mock_q.text.return_value.ask.side_effect = [
                "",  # Workspace directory (empty = use cwd)
            ]
            mock_q.checkbox.return_value.ask.return_value = []  # Skills: skip

            result = run_onboard(skip_validation=True)

        assert result is False
        mock_save.assert_not_called()
