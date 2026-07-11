"""Tests for EvoScientist onboarding wizard."""

import subprocess
from contextlib import contextmanager
from unittest.mock import MagicMock, Mock, patch

import pytest

from EvoScientist.config import EvoScientistConfig
from EvoScientist.config.onboard.style import (
    CONFIRM_STYLE,
    WIZARD_STYLE,
)
from EvoScientist.config.onboard.validators import (
    ChoiceValidator,
    IntegerValidator,
)
from EvoScientist.config.onboard.wizard import (
    STEPS,
    render_progress,
)


@contextmanager
def _patch_all_questionary(mock_q):
    """Patch ``questionary`` in every onboard submodule that imports it.

    The wizard delegates to step functions across multiple submodules
    (steps/helpers/channels/style/wizard), and each submodule binds its own
    ``questionary`` name at import time. Patching just one location wouldn't
    intercept calls made from the others.

    Also forces ``_check_npx`` to True so the skills-step easter egg never
    probes the host: on runners where npx is missing/slow, ``_ensure_npx``
    would otherwise fire an extra ``questionary.confirm`` and shift the
    mocked side_effect sequences (platform-dependent StopIteration).
    """
    with (
        patch("EvoScientist.config.onboard.wizard.questionary", mock_q),
        patch("EvoScientist.config.onboard.steps.questionary", mock_q),
        patch("EvoScientist.config.onboard.helpers.questionary", mock_q),
        patch("EvoScientist.config.onboard.channels.questionary", mock_q),
        patch("EvoScientist.config.onboard.style.questionary", mock_q),
        patch("EvoScientist.config.onboard.helpers._check_npx", return_value=True),
    ):
        yield mock_q


# =============================================================================
# Test STEPS and WIZARD_STYLE constants
# =============================================================================


class TestConstants:
    def test_steps_has_thirteen_items(self):
        """Test that STEPS contains exactly 13 steps."""
        assert len(STEPS) == 13
        assert STEPS == [
            "UI",
            "LangGraph Port",
            "Provider",
            "API Key",
            "Model",
            "Auxiliary Model",
            "Tavily Key",
            "Workspace",
            "Thinking",
            "Skills",
            "MCP Servers",
            "LaTeX",
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


class TestSharedConstantsAlignment:
    """Drift guard: the canonical valid-value sets in
    ``EvoScientist.config.onboard.constants`` must match the actual
    ``Choice(value=...)`` ids built by the interactive step functions in
    ``steps.py``. Without this, adding a new provider to one file but not
    the other would silently break either CLI flag validation or the
    interactive picker.
    """

    def test_provider_constants_match_step_choices(self):
        """Every value in `_step_provider`'s Choice list must be in
        ``VALID_PROVIDERS`` — and vice versa."""
        from unittest.mock import MagicMock, patch

        from EvoScientist.config.onboard.constants import VALID_PROVIDERS
        from EvoScientist.config.onboard.steps import _step_provider

        # Intercept questionary.select to capture the choices list before
        # any user prompt happens.
        captured = {}

        def _capture(*args, **kwargs):
            captured["choices"] = kwargs.get("choices") or (
                args[1] if len(args) > 1 else []
            )
            fake = MagicMock()
            fake.ask.return_value = "anthropic"
            return fake

        with patch(
            "EvoScientist.config.onboard.steps.questionary.select",
            side_effect=_capture,
        ):
            _step_provider(EvoScientistConfig())

        actual_provider_ids = {c.value for c in captured["choices"]}
        assert actual_provider_ids == set(VALID_PROVIDERS), (
            "VALID_PROVIDERS in constants.py drifted from _step_provider's "
            f"choices. Only-in-constants: {set(VALID_PROVIDERS) - actual_provider_ids}; "
            f"only-in-choices: {actual_provider_ids - set(VALID_PROVIDERS)}"
        )

    def test_ui_constants_match_step_choices(self):
        from EvoScientist.config.onboard.constants import VALID_UI_BACKENDS

        # _step_ui_backend hard-codes "tui", "cli", and "webui" — small enough
        # to check by direct lookup against the canonical set.
        assert VALID_UI_BACKENDS == frozenset({"tui", "cli", "webui"})

    def test_workspace_mode_constants_match_step_choices(self):
        from EvoScientist.config.onboard.constants import VALID_WORKSPACE_MODES

        assert VALID_WORKSPACE_MODES == frozenset({"daemon", "run"})

    def test_valid_providers_aligns_with_provider_key_attr(self):
        """Every provider in ``VALID_PROVIDERS`` (except ``openai`` and
        ``ollama``, which the wizard handles via fallbacks) must appear in
        ``_PROVIDER_KEY_ATTR`` — otherwise the auth-mode flow won't know
        which config attribute to write the validated key to."""
        from EvoScientist.config.onboard.constants import VALID_PROVIDERS
        from EvoScientist.config.onboard.wizard import _PROVIDER_KEY_ATTR

        expected = set(VALID_PROVIDERS) - {"openai", "ollama"}
        missing = expected - set(_PROVIDER_KEY_ATTR.keys())
        assert not missing, (
            "VALID_PROVIDERS has providers missing from _PROVIDER_KEY_ATTR "
            f"in wizard.py: {missing}"
        )
        extra = set(_PROVIDER_KEY_ATTR.keys()) - expected
        assert not extra, (
            f"_PROVIDER_KEY_ATTR in wizard.py has keys not in VALID_PROVIDERS: {extra}"
        )

    def test_valid_providers_aligns_with_provider_key_info(self):
        """``_provider_key_info`` uses ``mapping.get(provider, <openai
        fallback>)``, so an unknown provider silently behaves like OpenAI.
        Verify every non-``openai`` provider in ``VALID_PROVIDERS`` has an
        explicit entry — detected by checking the returned display name is
        not the OpenAI fallback string."""
        from EvoScientist.config.onboard.constants import VALID_PROVIDERS
        from EvoScientist.config.onboard.helpers import _provider_key_info

        cfg = EvoScientistConfig()
        for provider in VALID_PROVIDERS:
            display_name, _, _ = _provider_key_info(cfg, provider)
            if provider == "openai":
                assert display_name == "OpenAI"
            else:
                assert display_name != "OpenAI", (
                    f"Provider {provider!r} is missing from _provider_key_info "
                    "in helpers.py — it falls through to the OpenAI default, "
                    "which would silently send the wrong validator and "
                    "current-key lookup."
                )


class TestOAuthModeReconcile:
    def test_reconcile_preserves_auxiliary_openai_oauth(self):
        from EvoScientist.config.onboard.wizard import _reconcile_oauth_modes

        config = EvoScientistConfig(
            provider="minimax",
            auxiliary_provider="openai",
            auxiliary_model="gpt-5.5",
            openai_auth_mode="oauth",
            anthropic_auth_mode="oauth",
        )

        _reconcile_oauth_modes(config)

        assert config.openai_auth_mode == "oauth"
        assert config.anthropic_auth_mode == "api_key"

    def test_reconcile_preserves_auxiliary_provider_without_model(self):
        from EvoScientist.config.onboard.wizard import _reconcile_oauth_modes

        config = EvoScientistConfig(
            provider="minimax",
            auxiliary_provider="openai",
            auxiliary_model="",
            openai_auth_mode="oauth",
        )

        _reconcile_oauth_modes(config)

        assert config.openai_auth_mode == "oauth"


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
        from EvoScientist.config.onboard.steps import _step_provider

        config = EvoScientistConfig()

        with patch("EvoScientist.config.onboard.steps.questionary") as mock_q:
            mock_q.select.return_value.ask.return_value = "anthropic"
            result = _step_provider(config)

        assert result == "anthropic"
        mock_q.select.assert_called_once()

    def test_default_value_and_label_override(self):
        """default_value preselects a provider (re-run co-pilot default) and
        label customizes the prompt text."""
        from EvoScientist.config.onboard.steps import _step_provider

        config = EvoScientistConfig(provider="anthropic")
        with patch("EvoScientist.config.onboard.steps.questionary") as mock_q:
            mock_q.select.return_value.ask.return_value = "openai"
            _step_provider(config, label="co-pilot", default_value="openrouter")

        call = mock_q.select.call_args
        assert call.kwargs["default"] == "openrouter"  # override, not config.provider
        assert "co-pilot" in call.args[0]

    def test_raises_keyboard_interrupt_on_cancel(self):
        """Test that _step_provider raises KeyboardInterrupt on cancel."""
        from EvoScientist.config.onboard.steps import _step_provider

        config = EvoScientistConfig()

        with patch("EvoScientist.config.onboard.steps.questionary") as mock_q:
            mock_q.select.return_value.ask.return_value = None
            with pytest.raises(KeyboardInterrupt):
                _step_provider(config)


class TestStepOAuthAuthMode:
    @pytest.mark.parametrize(
        (
            "step_name",
            "config_attr",
            "provider_label",
            "oauth_choice_label",
            "ccproxy_provider",
            "status_label",
            "question_label",
            "login_prompt",
        ),
        [
            (
                "_step_anthropic_auth_mode",
                "anthropic_auth_mode",
                "Anthropic",
                "Claude Code OAuth",
                "claude_api",
                "OAuth",
                "Authentication mode",
                "Log in to Claude now?",
            ),
            (
                "_step_openai_auth_mode",
                "openai_auth_mode",
                "OpenAI",
                "Codex OAuth",
                "codex",
                "Codex OAuth",
                "OpenAI authentication mode",
                "Log in to Codex now?",
            ),
        ],
    )
    def test_oauth_wrappers_use_provider_specific_ccproxy_flow(
        self,
        step_name,
        config_attr,
        provider_label,
        oauth_choice_label,
        ccproxy_provider,
        status_label,
        question_label,
        login_prompt,
    ):
        """Anthropic/OpenAI wrappers share flow but keep provider-specific IDs."""
        from EvoScientist.config.onboard import steps as onboard_steps

        config = EvoScientistConfig(**{config_attr: "oauth"})
        select_question = MagicMock()
        select_question.ask.return_value = "oauth"
        confirm_question = MagicMock()
        confirm_question.ask.return_value = True

        with (
            patch(
                "EvoScientist.ccproxy_manager.is_ccproxy_available", return_value=True
            ),
            patch(
                "EvoScientist.ccproxy_manager.check_ccproxy_auth",
                return_value=(False, "not authenticated"),
            ) as mock_check_auth,
            patch(
                "EvoScientist.config.onboard.prompter.install_navigation_keys"
            ) as mock_nav,
            patch(
                "EvoScientist.config.onboard.steps.questionary.select",
                return_value=select_question,
            ) as mock_select,
            patch(
                "EvoScientist.config.onboard.steps.questionary.confirm",
                return_value=confirm_question,
            ) as mock_confirm,
            patch(
                "EvoScientist.config.onboard.steps._prompt_ccproxy_port"
            ) as mock_port,
            patch("EvoScientist.config.onboard.steps._run_ccproxy_login") as mock_login,
        ):
            result = getattr(onboard_steps, step_name)(config)

        assert result == "oauth"
        mock_nav.assert_called_once_with(select_question, with_back=True)
        mock_port.assert_called_once_with(config)
        mock_check_auth.assert_called_once_with(ccproxy_provider)
        mock_login.assert_called_once_with(ccproxy_provider, status_label)

        select_call = mock_select.call_args
        assert select_call.args[0] == f"{question_label}  [Esc/← to go back]:"
        assert select_call.kwargs["default"] == "oauth"
        choice_titles = [
            choice.title
            for choice in select_call.kwargs["choices"]
            if getattr(choice, "value", None) in {"api_key", "oauth"}
        ]
        assert choice_titles == [
            f"API Key (direct {provider_label} access)",
            f"{oauth_choice_label} (via ccproxy — no API key needed)",
        ]
        mock_confirm.assert_called_once()
        assert mock_confirm.call_args.args[0] == login_prompt


class TestStepModel:
    def test_returns_selected_model(self):
        """Test that _step_model returns selected model."""
        from EvoScientist.config.onboard.steps import _step_model

        config = EvoScientistConfig()

        with patch("EvoScientist.config.onboard.steps.questionary") as mock_q:
            mock_q.select.return_value.ask.return_value = "claude-sonnet-4-6"
            result = _step_model(config, "anthropic")

        assert result == "claude-sonnet-4-6"

    def test_main_model_not_in_provider_list_defaults_to_first(self):
        """Reset/main flow: a config.model that isn't in the chosen provider's
        list (e.g. provider switched to google-genai) defaults to that
        provider's first model, NOT the custom 'Type a model name...' entry."""
        from EvoScientist.config.onboard.steps import _step_model
        from EvoScientist.llm.models import get_models_for_provider

        config = EvoScientistConfig(model="claude-sonnet-4-6")
        entries = get_models_for_provider("google-genai")
        with patch("EvoScientist.config.onboard.steps.questionary") as mock_q:
            mock_q.select.return_value.ask.return_value = entries[0][0]
            _step_model(config, "google-genai")

        default = mock_q.select.call_args.kwargs["default"]
        assert default == entries[0][0]
        assert default != "__custom__"

    def test_custom_default_value_preselects_and_prefills(self):
        """Co-pilot re-run: a saved custom (non-registry) model preselects and
        prefills the 'Type a model name...' entry."""
        from EvoScientist.config.onboard.steps import _step_model

        config = EvoScientistConfig()
        with patch("EvoScientist.config.onboard.steps.questionary") as mock_q:
            mock_q.select.return_value.ask.return_value = "__custom__"
            mock_q.text.return_value.ask.return_value = "my-private/model"
            result = _step_model(config, "openrouter", default_value="my-private/model")

        assert mock_q.select.call_args.kwargs["default"] == "__custom__"
        assert mock_q.text.call_args.kwargs["default"] == "my-private/model"
        assert result == "my-private/model"

    def test_raises_keyboard_interrupt_on_cancel(self):
        """Test that _step_model raises KeyboardInterrupt on cancel."""
        from EvoScientist.config.onboard.steps import _step_model

        config = EvoScientistConfig()

        with patch("EvoScientist.config.onboard.steps.questionary") as mock_q:
            mock_q.select.return_value.ask.return_value = None
            with pytest.raises(KeyboardInterrupt):
                _step_model(config, "anthropic")


class TestStepWorkspace:
    def test_returns_daemon_mode(self):
        """Test workspace step returns selected mode."""
        from EvoScientist.config.onboard.steps import _step_workspace

        config = EvoScientistConfig()

        with patch("EvoScientist.config.onboard.steps.questionary") as mock_q:
            mock_q.select.return_value.ask.return_value = "daemon"
            result = _step_workspace(config)

        assert result == "daemon"

    def test_returns_run_mode(self):
        """Test workspace step returns run mode."""
        from EvoScientist.config.onboard.steps import _step_workspace

        config = EvoScientistConfig()

        with patch("EvoScientist.config.onboard.steps.questionary") as mock_q:
            mock_q.select.return_value.ask.return_value = "run"
            result = _step_workspace(config)

        assert result == "run"


class TestPromptAndValidateApiKey:
    def test_keep_existing_key_still_validates(self):
        """Pressing Enter to keep current key should validate the existing key."""
        from EvoScientist.config.onboard.helpers import _prompt_and_validate_api_key

        validate_fn = Mock(return_value=(True, "Valid"))

        with (
            patch("EvoScientist.config.onboard.helpers.questionary") as mock_q,
            patch("EvoScientist.config.onboard.helpers.console"),
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
        from EvoScientist.config.onboard.helpers import _prompt_and_validate_api_key

        validate_fn = Mock(return_value=(True, "valid"))

        with patch("EvoScientist.config.onboard.helpers.questionary") as mock_q:
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
        from EvoScientist.config.onboard.helpers import validate_imessage

        version_result = Mock(returncode=0, stdout="imsg 1.2.3")
        rpc_result = Mock(returncode=0)

        with (
            patch("EvoScientist.config.onboard.helpers.sys") as mock_sys,
            patch("EvoScientist.channels.imessage.probe.shutil") as mock_shutil,
            patch("EvoScientist.config.onboard.helpers.subprocess") as mock_sub,
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
        from EvoScientist.config.onboard.helpers import validate_imessage

        with (
            patch("EvoScientist.config.onboard.helpers.sys") as mock_sys,
            patch("EvoScientist.channels.imessage.probe.shutil") as mock_shutil,
        ):
            mock_sys.platform = "darwin"
            mock_shutil.which.return_value = None
            valid, msg = validate_imessage()

        assert valid is False
        assert msg == "not_installed"

    def test_invalid_on_non_macos(self):
        """Test validate_imessage returns invalid on non-macOS."""
        from EvoScientist.config.onboard.helpers import validate_imessage

        with patch("EvoScientist.config.onboard.helpers.sys") as mock_sys:
            mock_sys.platform = "linux"
            valid, msg = validate_imessage()

        assert valid is False
        assert "macOS" in msg

    def test_invalid_when_rpc_not_supported(self):
        """Test validate_imessage returns invalid when RPC check fails."""
        from EvoScientist.config.onboard.helpers import validate_imessage

        version_result = Mock(returncode=0, stdout="imsg 0.1.0")
        rpc_result = Mock(returncode=1)

        with (
            patch("EvoScientist.config.onboard.helpers.sys") as mock_sys,
            patch("EvoScientist.channels.imessage.probe.shutil") as mock_shutil,
            patch("EvoScientist.config.onboard.helpers.subprocess") as mock_sub,
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
        from EvoScientist.config.onboard.helpers import _install_imsg

        with patch("EvoScientist.config.onboard.helpers.subprocess") as mock_sub:
            mock_sub.run.return_value = Mock(returncode=0)
            mock_sub.TimeoutExpired = subprocess.TimeoutExpired
            result = _install_imsg()

        assert result is True

    def test_install_brew_not_found(self):
        """Test _install_imsg handles missing Homebrew."""
        from EvoScientist.config.onboard.helpers import _install_imsg

        with (
            patch("EvoScientist.config.onboard.helpers.subprocess") as mock_sub,
            patch("EvoScientist.config.onboard.helpers.console"),
        ):
            mock_sub.run.side_effect = FileNotFoundError()
            mock_sub.TimeoutExpired = subprocess.TimeoutExpired
            result = _install_imsg()

        assert result is False

    def test_install_failure(self):
        """Test _install_imsg returns False on non-zero exit."""
        from EvoScientist.config.onboard.helpers import _install_imsg

        with patch("EvoScientist.config.onboard.helpers.subprocess") as mock_sub:
            mock_sub.run.return_value = Mock(returncode=1)
            mock_sub.TimeoutExpired = subprocess.TimeoutExpired
            result = _install_imsg()

        assert result is False


class TestSetupImessage:
    def test_already_installed(self):
        """Test _setup_imessage returns True when already installed."""
        from EvoScientist.config.onboard.helpers import _setup_imessage

        with (
            patch(
                "EvoScientist.config.onboard.helpers.validate_imessage",
                return_value=(True, "imsg at /bin/imsg"),
            ),
            patch("EvoScientist.config.onboard.helpers.console"),
        ):
            result = _setup_imessage()

        assert result is True

    def test_not_macos(self):
        """Test _setup_imessage returns False on non-macOS."""
        from EvoScientist.config.onboard.helpers import _setup_imessage

        with (
            patch(
                "EvoScientist.config.onboard.helpers.validate_imessage",
                return_value=(False, "iMessage requires macOS"),
            ),
            patch("EvoScientist.config.onboard.helpers.console"),
        ):
            result = _setup_imessage()

        assert result is False

    def test_install_then_valid(self):
        """Test _setup_imessage installs and re-validates successfully."""
        from EvoScientist.config.onboard.helpers import _setup_imessage

        with (
            patch("EvoScientist.config.onboard.helpers.validate_imessage") as mock_val,
            patch(
                "EvoScientist.config.onboard.helpers._install_imsg", return_value=True
            ),
            patch("EvoScientist.config.onboard.helpers.questionary") as mock_q,
            patch("EvoScientist.config.onboard.helpers.console"),
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
        from EvoScientist.config.onboard.helpers import _setup_imessage

        with (
            patch(
                "EvoScientist.config.onboard.helpers.validate_imessage",
                return_value=(False, "not_installed"),
            ),
            patch("EvoScientist.config.onboard.helpers.questionary") as mock_q,
            patch("EvoScientist.config.onboard.helpers.console"),
        ):
            mock_q.confirm.return_value.ask.return_value = False
            result = _setup_imessage()

        assert result is False


class TestStepSkills:
    def test_returns_empty_when_none_selected(self):
        """Test skills step returns empty list when user selects nothing."""
        from EvoScientist.config.onboard.steps import _step_skills

        with (
            patch("EvoScientist.config.onboard.style.questionary") as mock_q,
            patch("EvoScientist.config.onboard.steps.console"),
            # Empty selection triggers the npx easter-egg check; without this
            # mock the real questionary.confirm in _ensure_npx fires on hosts
            # where npx is missing/slow (NoConsoleScreenBufferError on
            # headless Windows CI).
            patch("EvoScientist.config.onboard.steps._ensure_npx", return_value=True),
        ):
            mock_q.checkbox.return_value.ask.return_value = []
            result = _step_skills()

        assert result == []

    def test_installs_selected_skills(self):
        """Test skills step installs selected skills and returns sources."""
        from EvoScientist.config.onboard.steps import _RECOMMENDED_SKILLS, _step_skills

        source = _RECOMMENDED_SKILLS[0]["source"]

        with (
            patch("EvoScientist.config.onboard.style.questionary") as mock_q,
            patch("EvoScientist.config.onboard.steps.console"),
            patch("EvoScientist.tools.skills_manager.install_skill") as mock_install,
        ):
            mock_q.checkbox.return_value.ask.return_value = [source]
            mock_install.return_value = {"success": True, "name": "test"}
            result = _step_skills()

        assert result == [source]
        mock_install.assert_called_once_with(source)

    def test_handles_install_failure(self):
        """Test skills step handles installation errors gracefully."""
        from EvoScientist.config.onboard.steps import _RECOMMENDED_SKILLS, _step_skills

        source = _RECOMMENDED_SKILLS[0]["source"]

        with (
            patch("EvoScientist.config.onboard.style.questionary") as mock_q,
            patch("EvoScientist.config.onboard.steps.console"),
            patch("EvoScientist.tools.skills_manager.install_skill") as mock_install,
        ):
            mock_q.checkbox.return_value.ask.return_value = [source]
            mock_install.side_effect = Exception("network error")
            result = _step_skills()

        assert result == []

    def test_raises_keyboard_interrupt_on_cancel(self):
        """Test skills step raises KeyboardInterrupt on cancel."""
        from EvoScientist.config.onboard.steps import _step_skills

        with patch("EvoScientist.config.onboard.style.questionary") as mock_q:
            mock_q.checkbox.return_value.ask.return_value = None
            with pytest.raises(KeyboardInterrupt):
                _step_skills()

    def test_detects_pack_via_manifest(self, tmp_path):
        """A pack source recorded in the manifest is detected as installed
        even when none of the unpacked child dir names match the source."""
        from EvoScientist.config.onboard.steps import _RECOMMENDED_SKILLS, _step_skills

        # Recreate the EvoSkills-style layout: child dirs in GLOBAL_SKILLS_DIR
        # whose names share nothing with the pack source URL.
        global_dir = tmp_path / "global"
        global_dir.mkdir()
        for child in ("paper-writing", "evo-memory", "research-survey"):
            (global_dir / child).mkdir()
            (global_dir / child / "SKILL.md").write_text(
                f"---\nname: {child}\ndescription: x\n---\n"
            )
        pack_source = _RECOMMENDED_SKILLS[0]["source"]
        (global_dir / ".installed.yaml").write_text(
            f"paper-writing:\n  source: {pack_source}\n"
            f"evo-memory:\n  source: {pack_source}\n"
            f"research-survey:\n  source: {pack_source}\n"
        )
        empty_user = tmp_path / "user"
        empty_user.mkdir()

        captured: list = []

        def _capture(choices, _msg, **_kw):
            captured.extend(choices)
            return []

        with (
            patch("EvoScientist.paths.GLOBAL_SKILLS_DIR", global_dir),
            patch("EvoScientist.paths.USER_SKILLS_DIR", empty_user),
            patch(
                "EvoScientist.config.onboard.steps._checkbox_ask", side_effect=_capture
            ),
            patch("EvoScientist.config.onboard.steps.console"),
            # _capture returns [] (empty selection), which would otherwise
            # reach the real _ensure_npx — see test_returns_empty_when_none_selected.
            patch("EvoScientist.config.onboard.steps._ensure_npx", return_value=True),
        ):
            _step_skills()

        pack_choice = next(c for c in captured if c.value == pack_source)
        # Detection signal: an "installed" hint appears in the choice title,
        # but the choice stays selectable so users can re-sync (important for
        # packs where one child was deleted but the rest keep the entry).
        title_text = "".join(seg[1] for seg in pack_choice.title)
        assert "installed" in title_text, (
            "EvoSkills pack should be marked as installed via the manifest "
            "even though no child dir name matches the pack source string"
        )
        assert not pack_choice.disabled, (
            "Installed packs must remain selectable so users can re-sync them"
        )

    def test_surfaces_update_available_when_upstream_moved(self, tmp_path):
        """When a pack records an install-time commit and `git ls-remote`
        reports a different SHA, the choice label should call out the update."""
        from EvoScientist.config.onboard.steps import _RECOMMENDED_SKILLS, _step_skills

        pack_source = _RECOMMENDED_SKILLS[0]["source"]
        global_dir = tmp_path / "global"
        global_dir.mkdir()
        (global_dir / "paper-writing").mkdir()
        (global_dir / "paper-writing" / "SKILL.md").write_text(
            "---\nname: paper-writing\ndescription: x\n---\n"
        )
        # Recorded install-time commit.
        (global_dir / ".installed.yaml").write_text(
            f"paper-writing:\n  source: {pack_source}\n  commit: aaa111\n"
        )
        empty_user = tmp_path / "user"
        empty_user.mkdir()

        captured: list = []

        def _capture(choices, _msg, **_kw):
            captured.extend(choices)
            return []

        with (
            patch("EvoScientist.paths.GLOBAL_SKILLS_DIR", global_dir),
            patch("EvoScientist.paths.USER_SKILLS_DIR", empty_user),
            # Upstream moved past the recorded commit.
            patch(
                "EvoScientist.tools.skills_manager.resolve_remote_head",
                return_value="bbb222",
            ),
            patch(
                "EvoScientist.config.onboard.steps._checkbox_ask", side_effect=_capture
            ),
            patch("EvoScientist.config.onboard.steps.console"),
            # _capture returns [] (empty selection), which would otherwise
            # reach the real _ensure_npx — see test_returns_empty_when_none_selected.
            patch("EvoScientist.config.onboard.steps._ensure_npx", return_value=True),
        ):
            _step_skills()

        pack_choice = next(c for c in captured if c.value == pack_source)
        title_text = "".join(seg[1] for seg in pack_choice.title)
        assert "update available" in title_text, (
            f"expected an 'update available' hint in the label, got: {title_text!r}"
        )

    def test_no_update_hint_when_remote_check_fails(self, tmp_path):
        """If `git ls-remote` returns None (offline, timeout, etc.), the label
        must fall back to plain 'installed' — never falsely claim 'update'."""
        from EvoScientist.config.onboard.steps import _RECOMMENDED_SKILLS, _step_skills

        pack_source = _RECOMMENDED_SKILLS[0]["source"]
        global_dir = tmp_path / "global"
        global_dir.mkdir()
        (global_dir / "paper-writing").mkdir()
        (global_dir / "paper-writing" / "SKILL.md").write_text(
            "---\nname: paper-writing\ndescription: x\n---\n"
        )
        (global_dir / ".installed.yaml").write_text(
            f"paper-writing:\n  source: {pack_source}\n  commit: aaa111\n"
        )
        empty_user = tmp_path / "user"
        empty_user.mkdir()

        captured: list = []

        def _capture(choices, _msg, **_kw):
            captured.extend(choices)
            return []

        with (
            patch("EvoScientist.paths.GLOBAL_SKILLS_DIR", global_dir),
            patch("EvoScientist.paths.USER_SKILLS_DIR", empty_user),
            patch(
                "EvoScientist.tools.skills_manager.resolve_remote_head",
                return_value=None,
            ),
            patch(
                "EvoScientist.config.onboard.steps._checkbox_ask", side_effect=_capture
            ),
            patch("EvoScientist.config.onboard.steps.console"),
            # _capture returns [] (empty selection), which would otherwise
            # reach the real _ensure_npx — see test_returns_empty_when_none_selected.
            patch("EvoScientist.config.onboard.steps._ensure_npx", return_value=True),
        ):
            _step_skills()

        pack_choice = next(c for c in captured if c.value == pack_source)
        title_text = "".join(seg[1] for seg in pack_choice.title)
        assert "update available" not in title_text, (
            f"must not claim 'update available' when remote check fails: {title_text!r}"
        )
        assert "installed" in title_text


class TestStepChannels:
    def test_returns_disabled_when_skip(self):
        """Test channels step returns empty dict when user selects nothing."""
        from EvoScientist.config.onboard.channels import _step_channels

        config = EvoScientistConfig()

        with patch("EvoScientist.config.onboard.channels.questionary") as mock_q:
            mock_q.checkbox.return_value.ask.return_value = []
            result = _step_channels(config)

        assert result == {"channel_enabled": "", "imessage_enabled": False}

    def test_returns_enabled_when_setup_passes(self):
        """Test channels step returns enabled when iMessage setup succeeds."""
        from EvoScientist.config.onboard.channels import _step_channels

        config = EvoScientistConfig()

        with (
            patch("EvoScientist.config.onboard.channels.questionary") as mock_q,
            patch(
                "EvoScientist.config.onboard.channels._setup_imessage",
                return_value=True,
            ),
        ):
            mock_q.checkbox.return_value.ask.return_value = ["imessage"]
            mock_q.text.return_value.ask.return_value = ""
            result = _step_channels(config)

        assert result["channel_enabled"] == "imessage"
        assert result["imessage_enabled"] is True

    def test_returns_enabled_with_senders(self):
        """Test channels step returns enabled with specific senders."""
        from EvoScientist.config.onboard.channels import _step_channels

        config = EvoScientistConfig()

        with (
            patch("EvoScientist.config.onboard.channels.questionary") as mock_q,
            patch(
                "EvoScientist.config.onboard.channels._setup_imessage",
                return_value=True,
            ),
        ):
            mock_q.checkbox.return_value.ask.return_value = ["imessage"]
            mock_q.text.return_value.ask.return_value = "+1234567890,+0987654321"
            result = _step_channels(config)

        assert result["channel_enabled"] == "imessage"
        assert result["imessage_enabled"] is True
        assert result["imessage_allowed_senders"] == "+1234567890,+0987654321"

    def test_setup_fails_user_declines(self):
        """Test channels step skips iMessage when setup fails and user declines."""
        from EvoScientist.config.onboard.channels import _step_channels

        config = EvoScientistConfig()

        with (
            patch("EvoScientist.config.onboard.channels.questionary") as mock_q,
            patch(
                "EvoScientist.config.onboard.channels._setup_imessage",
                return_value=False,
            ),
        ):
            mock_q.checkbox.return_value.ask.return_value = ["imessage"]
            mock_q.confirm.return_value.ask.return_value = False
            result = _step_channels(config)

        assert result["channel_enabled"] == ""
        assert result["imessage_enabled"] is False

    def test_setup_fails_user_enables_anyway(self):
        """Test channels step enables iMessage when setup fails but user confirms."""
        from EvoScientist.config.onboard.channels import _step_channels

        config = EvoScientistConfig()

        with (
            patch("EvoScientist.config.onboard.channels.questionary") as mock_q,
            patch(
                "EvoScientist.config.onboard.channels._setup_imessage",
                return_value=False,
            ),
        ):
            mock_q.checkbox.return_value.ask.return_value = ["imessage"]
            mock_q.confirm.return_value.ask.return_value = True
            mock_q.text.return_value.ask.return_value = ""
            result = _step_channels(config)

        assert result["channel_enabled"] == "imessage"
        assert result["imessage_enabled"] is True

    def test_raises_keyboard_interrupt_on_cancel(self):
        """Test channels step raises KeyboardInterrupt on cancel."""
        from EvoScientist.config.onboard.channels import _step_channels

        config = EvoScientistConfig()

        with patch("EvoScientist.config.onboard.channels.questionary") as mock_q:
            mock_q.checkbox.return_value.ask.return_value = None
            with pytest.raises(KeyboardInterrupt):
                _step_channels(config)

    def test_telegram_channel_selected(self):
        """Test channels step handles Telegram selection."""
        from EvoScientist.config.onboard.channels import _step_channels

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
            patch("EvoScientist.config.onboard.channels.questionary") as mock_q,
            patch("EvoScientist.config.onboard.channels._probe_channel"),
            patch("builtins.__import__", side_effect=_fake_import),
        ):
            mock_q.checkbox.return_value.ask.return_value = ["telegram"]
            # Bot token is a secret → prompted via questionary.password.
            mock_q.password.return_value.ask.return_value = "test-token"
            mock_q.text.return_value.ask.return_value = ""
            result = _step_channels(config)

        assert result["channel_enabled"] == "telegram"
        assert result["telegram_bot_token"] == "test-token"

    def test_discord_channel_selected(self):
        """Test channels step handles Discord selection."""
        from EvoScientist.config.onboard.channels import _step_channels

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
            patch("EvoScientist.config.onboard.channels.questionary") as mock_q,
            patch("EvoScientist.config.onboard.channels._probe_channel"),
            patch("builtins.__import__", side_effect=_fake_import),
        ):
            mock_q.checkbox.return_value.ask.return_value = ["discord"]
            # Bot token is a secret → prompted via questionary.password.
            mock_q.password.return_value.ask.return_value = "discord-token"
            mock_q.text.return_value.ask.return_value = ""
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
        from EvoScientist.config.onboard.steps import _step_mcp_servers

        servers = self._make_test_servers()

        with (
            patch(
                "EvoScientist.mcp.registry.fetch_marketplace_index",
                return_value=servers,
            ),
            patch(
                "EvoScientist.config.onboard.steps._checkbox_ask",
                return_value=["npx-server", "url-server"],
            ),
            patch("EvoScientist.config.onboard.steps._ensure_npx", return_value=False),
            patch("EvoScientist.config.onboard.helpers._check_npx", return_value=False),
            patch("EvoScientist.mcp.client._load_user_config", return_value={}),
            patch("EvoScientist.mcp.client.add_mcp_server") as mock_add,
            patch("EvoScientist.config.onboard.steps.console"),
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
        from EvoScientist.config.onboard.steps import _step_mcp_servers

        servers = self._make_test_servers()
        npx_names = [s.name for s in servers if s.command == "npx"]

        with (
            patch(
                "EvoScientist.mcp.registry.fetch_marketplace_index",
                return_value=servers,
            ),
            patch(
                "EvoScientist.config.onboard.steps._checkbox_ask",
                return_value=npx_names,
            ),
            patch("EvoScientist.config.onboard.steps._ensure_npx", return_value=False),
            patch("EvoScientist.config.onboard.helpers._check_npx", return_value=False),
            patch("EvoScientist.mcp.client._load_user_config", return_value={}),
            patch("EvoScientist.mcp.client.add_mcp_server") as mock_add,
            patch("EvoScientist.config.onboard.steps.console"),
        ):
            result = _step_mcp_servers()

        assert result == []
        mock_add.assert_not_called()


class TestStepThinking:
    def test_returns_show_thinking(self):
        """Test thinking step returns selected value."""
        from EvoScientist.config.onboard.steps import _step_thinking

        config = EvoScientistConfig()

        with patch("EvoScientist.config.onboard.steps.questionary") as mock_q:
            mock_q.select.return_value.ask.return_value = True
            result = _step_thinking(config)

        assert result is True

    def test_returns_false_when_off(self):
        """Test thinking step returns False when user selects Off."""
        from EvoScientist.config.onboard.steps import _step_thinking

        config = EvoScientistConfig(show_thinking=False)

        with patch("EvoScientist.config.onboard.steps.questionary") as mock_q:
            mock_q.select.return_value.ask.return_value = False
            result = _step_thinking(config)

        assert result is False


# =============================================================================
# Test run_onboard (Integration-like test with mocked questionary)
# =============================================================================


class TestRunOnboard:
    def test_returns_true_on_save(self):
        """Test that run_onboard returns True when config is saved."""
        from EvoScientist.config.onboard.wizard import run_onboard

        mock_q = MagicMock()
        with (
            _patch_all_questionary(mock_q),
            patch("EvoScientist.config.onboard.wizard.load_config") as mock_load,
            patch("EvoScientist.config.onboard.wizard.save_config") as mock_save,
            patch("EvoScientist.config.onboard.wizard.console"),
            patch("EvoScientist.config.onboard.steps.console"),
            patch("EvoScientist.config.onboard.channels.console"),
            patch("EvoScientist.config.onboard.helpers.console"),
            patch("EvoScientist.config.onboard.wizard._step_tinytex"),
        ):
            # Setup mock config
            mock_load.return_value = EvoScientistConfig()

            # Mock all questionary calls — order matches the wizard's select
            # sequence: UI → Provider → Anthropic auth_mode → Model →
            # Workspace → Thinking → (channels skipped via empty checkbox).
            mock_q.select.return_value.ask.side_effect = [
                "tui",  # UI backend
                "anthropic",  # Provider
                "api_key",  # Anthropic auth mode (API key, not OAuth)
                "claude-sonnet-4-6",  # Model
                "skip",  # Auxiliary: Skip (single driver)
                "daemon",  # Workspace mode
                True,  # Show thinking
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
        # Config is autosaved between phases + once at the end.
        mock_save.assert_called()
        # Verify the final saved config has the values we picked — guards
        # against silent shift bugs where a missing side_effect entry causes
        # downstream prompts to consume the wrong values.
        final_config = mock_save.call_args_list[-1].args[0]
        assert final_config.provider == "anthropic"
        assert final_config.model == "claude-sonnet-4-6"
        assert final_config.anthropic_auth_mode == "api_key"
        assert final_config.ui_backend == "tui"
        assert final_config.default_mode == "daemon"

    def test_auxiliary_model_enabled_collects_provider_and_key(self):
        """Enabling the auxiliary step stores its provider, model, and the
        chosen provider's API key (a different company than the main agent)."""
        from EvoScientist.config.onboard.wizard import run_onboard

        mock_q = MagicMock()
        with (
            _patch_all_questionary(mock_q),
            patch("EvoScientist.config.onboard.wizard.load_config") as mock_load,
            patch("EvoScientist.config.onboard.wizard.save_config") as mock_save,
            patch("EvoScientist.config.onboard.wizard.console"),
            patch("EvoScientist.config.onboard.steps.console"),
            patch("EvoScientist.config.onboard.channels.console"),
            patch("EvoScientist.config.onboard.helpers.console"),
            patch("EvoScientist.config.onboard.wizard._step_tinytex"),
        ):
            mock_load.return_value = EvoScientistConfig()

            mock_q.select.return_value.ask.side_effect = [
                "tui",  # UI backend
                "anthropic",  # Provider
                "api_key",  # Anthropic auth mode
                "claude-sonnet-4-6",  # Model
                "assemble",  # Auxiliary: Assemble
                "openai",  # Auxiliary provider (a different company)
                "api_key",  # Auxiliary OpenAI auth mode
                "gpt-5.5",  # Auxiliary model
                "daemon",  # Workspace mode
                True,  # Show thinking
            ]
            mock_q.password.return_value.ask.side_effect = [
                "",  # Main provider API key (keep current)
                "sk-aux-openai",  # Auxiliary provider API key
                "",  # Tavily key (keep current)
            ]
            mock_q.confirm.return_value.ask.side_effect = [
                True,  # Save config
            ]
            mock_q.text.return_value.ask.side_effect = [
                "",  # Workspace directory
            ]
            mock_q.checkbox.return_value.ask.return_value = []  # Skills: skip

            result = run_onboard(skip_validation=True)

        assert result is True
        final_config = mock_save.call_args_list[-1].args[0]
        assert final_config.auxiliary_provider == "openai"
        assert final_config.auxiliary_model == "gpt-5.5"
        assert final_config.openai_auth_mode == "api_key"
        # The auxiliary provider's key is stored in its per-provider field.
        assert final_config.openai_api_key == "sk-aux-openai"
        # Main agent is untouched.
        assert final_config.provider == "anthropic"
        assert final_config.model == "claude-sonnet-4-6"

    def test_auxiliary_same_provider_reuses_main_credentials(self):
        """Same-provider co-pilot should not imply separate credentials exist."""
        from EvoScientist.config.onboard.wizard import run_onboard

        mock_q = MagicMock()
        with (
            _patch_all_questionary(mock_q),
            patch("EvoScientist.config.onboard.wizard.load_config") as mock_load,
            patch("EvoScientist.config.onboard.wizard.save_config") as mock_save,
            patch("EvoScientist.config.onboard.wizard.console"),
            patch("EvoScientist.config.onboard.steps.console"),
        ):
            mock_load.return_value = EvoScientistConfig(
                provider="openai",
                model="gpt-5.5",
                openai_api_key="sk-main-openai",
            )
            mock_q.select.return_value.ask.side_effect = [
                "assemble",  # Auxiliary: Assemble
                "openai",  # Same provider as the main model
                "gpt-5.5",  # Auxiliary model
            ]
            mock_q.confirm.return_value.ask.side_effect = [True]  # Save config

            result = run_onboard(
                skip_validation=True, only_sections={"auxiliary_model"}
            )

        assert result is True
        final_config = mock_save.call_args_list[-1].args[0]
        assert final_config.auxiliary_provider == "openai"
        assert final_config.auxiliary_model == "gpt-5.5"
        assert final_config.openai_api_key == "sk-main-openai"
        mock_q.password.assert_not_called()
        assert mock_q.select.return_value.ask.call_count == 3

    def test_auxiliary_same_provider_prompts_when_shared_key_missing(self):
        """Same-provider reuse should not hide a missing shared API key."""
        from EvoScientist.config.onboard.wizard import run_onboard

        mock_q = MagicMock()
        with (
            _patch_all_questionary(mock_q),
            patch("EvoScientist.config.onboard.wizard.load_config") as mock_load,
            patch("EvoScientist.config.onboard.wizard.save_config") as mock_save,
            patch("EvoScientist.config.onboard.wizard.console"),
            patch("EvoScientist.config.onboard.steps.console"),
            patch("EvoScientist.config.onboard.helpers.console"),
            patch(
                "EvoScientist.ccproxy_manager.is_ccproxy_available", return_value=True
            ),
        ):
            mock_load.return_value = EvoScientistConfig(
                provider="openai",
                model="gpt-5.5",
                openai_auth_mode="api_key",
                openai_api_key="",
            )
            mock_q.select.return_value.ask.side_effect = [
                "assemble",  # Auxiliary: Assemble
                "openai",  # Same provider as the main model
                "api_key",  # Shared OpenAI auth mode
                "gpt-5.5",  # Auxiliary model
            ]
            mock_q.password.return_value.ask.side_effect = [
                "sk-shared-openai",
            ]
            mock_q.confirm.return_value.ask.side_effect = [True]  # Save config

            result = run_onboard(
                skip_validation=True, only_sections={"auxiliary_model"}
            )

        assert result is True
        final_config = mock_save.call_args_list[-1].args[0]
        assert final_config.auxiliary_provider == "openai"
        assert final_config.auxiliary_model == "gpt-5.5"
        assert final_config.openai_auth_mode == "api_key"
        assert final_config.openai_api_key == "sk-shared-openai"
        mock_q.password.assert_called_once()
        assert mock_q.select.return_value.ask.call_count == 4

    def test_auxiliary_openai_oauth_skips_api_key(self):
        """Auxiliary OpenAI now uses the shared auth flow and skips keys on OAuth."""
        from EvoScientist.config.onboard.wizard import run_onboard

        mock_q = MagicMock()
        with (
            _patch_all_questionary(mock_q),
            patch("EvoScientist.config.onboard.wizard.load_config") as mock_load,
            patch("EvoScientist.config.onboard.wizard.save_config") as mock_save,
            patch("EvoScientist.config.onboard.wizard.console"),
            patch("EvoScientist.config.onboard.steps.console"),
            patch("EvoScientist.config.onboard.helpers.console"),
            patch(
                "EvoScientist.ccproxy_manager.is_ccproxy_available", return_value=True
            ),
            patch(
                "EvoScientist.ccproxy_manager.check_ccproxy_auth",
                return_value=(False, "not authenticated"),
            ) as mock_auth,
        ):
            mock_load.return_value = EvoScientistConfig()
            mock_q.select.return_value.ask.side_effect = [
                "assemble",  # Auxiliary: Assemble
                "openai",  # Auxiliary provider
                "oauth",  # OpenAI auth mode
                "gpt-5.5",  # Auxiliary model
            ]
            mock_q.text.return_value.ask.side_effect = [
                "",  # ccproxy port (keep default)
            ]
            mock_q.confirm.return_value.ask.side_effect = [
                False,  # Do not log in to Codex now
                True,  # Save config
            ]

            result = run_onboard(
                skip_validation=True, only_sections={"auxiliary_model"}
            )

        assert result is True
        final_config = mock_save.call_args_list[-1].args[0]
        assert final_config.auxiliary_provider == "openai"
        assert final_config.auxiliary_model == "gpt-5.5"
        assert final_config.openai_auth_mode == "oauth"
        assert final_config.openai_api_key == ""
        mock_q.password.assert_not_called()
        mock_auth.assert_called_once_with("codex")

    def test_auxiliary_reconfigure_clears_unused_openai_oauth(self):
        """Switching co-pilot away from OpenAI clears stale OpenAI OAuth mode."""
        from EvoScientist.config.onboard.wizard import run_onboard

        mock_q = MagicMock()
        with (
            _patch_all_questionary(mock_q),
            patch("EvoScientist.config.onboard.wizard.load_config") as mock_load,
            patch("EvoScientist.config.onboard.wizard.save_config") as mock_save,
            patch("EvoScientist.config.onboard.wizard.console"),
            patch("EvoScientist.config.onboard.steps.console"),
            patch("EvoScientist.config.onboard.helpers.console"),
        ):
            mock_load.return_value = EvoScientistConfig(
                provider="anthropic",
                model="claude-sonnet-4-6",
                anthropic_auth_mode="oauth",
                auxiliary_provider="openai",
                auxiliary_model="gpt-5.5",
                openai_auth_mode="oauth",
            )
            mock_q.select.return_value.ask.side_effect = [
                "assemble",  # Auxiliary: Assemble
                "minimax",  # Auxiliary provider no longer uses OpenAI
                "global",  # MiniMax region
                "minimax-m2",  # Auxiliary model
            ]
            mock_q.password.return_value.ask.side_effect = [
                "sk-minimax",  # MiniMax API key
            ]
            mock_q.confirm.return_value.ask.side_effect = [True]  # Save config

            result = run_onboard(
                skip_validation=True, only_sections={"auxiliary_model"}
            )

        assert result is True
        final_config = mock_save.call_args_list[-1].args[0]
        assert final_config.auxiliary_provider == "minimax"
        assert final_config.openai_auth_mode == "api_key"
        assert final_config.anthropic_auth_mode == "oauth"

    def test_auxiliary_custom_provider_collects_base_url(self):
        """Regression for the custom-provider fix: a custom auxiliary provider
        collects its base URL (provider -> base URL -> key -> model order)."""
        from EvoScientist.config.onboard.wizard import run_onboard

        mock_q = MagicMock()
        with (
            _patch_all_questionary(mock_q),
            patch("EvoScientist.config.onboard.wizard.load_config") as mock_load,
            patch("EvoScientist.config.onboard.wizard.save_config") as mock_save,
            patch("EvoScientist.config.onboard.wizard.console"),
            patch("EvoScientist.config.onboard.steps.console"),
            patch("EvoScientist.config.onboard.channels.console"),
            patch("EvoScientist.config.onboard.helpers.console"),
        ):
            mock_load.return_value = EvoScientistConfig()
            mock_q.select.return_value.ask.side_effect = [
                "assemble",  # Auxiliary: Assemble
                "custom-openai",  # Auxiliary provider
                "gpt-5.5",  # Auxiliary model (from the custom-openai registry)
            ]
            mock_q.text.return_value.ask.side_effect = [
                "https://my-endpoint/v1",  # Auxiliary base URL (custom provider)
            ]
            mock_q.password.return_value.ask.side_effect = [
                "sk-aux-custom",  # Auxiliary provider API key
            ]
            mock_q.confirm.return_value.ask.side_effect = [True]  # Save

            result = run_onboard(
                skip_validation=True, only_sections={"auxiliary_model"}
            )

        assert result is True
        final_config = mock_save.call_args_list[-1].args[0]
        assert final_config.auxiliary_provider == "custom-openai"
        assert final_config.auxiliary_model == "gpt-5.5"
        # Base URL must be collected for the custom auxiliary provider.
        assert final_config.custom_openai_base_url == "https://my-endpoint/v1"
        assert final_config.custom_openai_api_key == "sk-aux-custom"

    def test_returns_false_on_cancel(self):
        """Test that run_onboard returns False when cancelled."""
        from EvoScientist.config.onboard.wizard import run_onboard

        mock_q = MagicMock()
        with (
            _patch_all_questionary(mock_q),
            patch("EvoScientist.config.onboard.wizard.load_config") as mock_load,
            patch("EvoScientist.config.onboard.wizard.console"),
            patch("EvoScientist.config.onboard.steps.console"),
            patch("EvoScientist.config.onboard.channels.console"),
            patch("EvoScientist.config.onboard.helpers.console"),
            patch("EvoScientist.config.onboard.wizard._step_tinytex"),
        ):
            mock_load.return_value = EvoScientistConfig()

            # First selection returns None (Ctrl+C)
            mock_q.select.return_value.ask.return_value = None

            result = run_onboard(skip_validation=True)

        assert result is False

    def test_returns_false_when_not_saving(self):
        """Test that run_onboard returns False when user declines to save."""
        from EvoScientist.config.onboard.wizard import run_onboard

        mock_q = MagicMock()
        with (
            _patch_all_questionary(mock_q),
            patch("EvoScientist.config.onboard.wizard.load_config") as mock_load,
            patch("EvoScientist.config.onboard.wizard.save_config") as mock_save,
            patch("EvoScientist.config.onboard.wizard.console"),
            patch("EvoScientist.config.onboard.steps.console"),
            patch("EvoScientist.config.onboard.channels.console"),
            patch("EvoScientist.config.onboard.helpers.console"),
            patch("EvoScientist.config.onboard.wizard._step_tinytex"),
        ):
            mock_load.return_value = EvoScientistConfig()

            mock_q.select.return_value.ask.side_effect = [
                "tui",  # UI backend
                "anthropic",  # Provider
                "api_key",  # Anthropic auth mode
                "claude-sonnet-4-6",  # Model
                "skip",  # Auxiliary: Skip (single driver)
                "daemon",  # Workspace mode
                True,  # Show thinking
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
        # The wizard autosaves between phases AND writes the original
        # snapshot back when the user declines — so save_config IS called.
        mock_save.assert_called()

    def test_reset_then_no_restores_pre_wizard_config(self, tmp_path):
        """Regression: Reset → No must restore the original user file,
        NOT overwrite it with ``EvoScientistConfig()`` defaults.

        Reproduces a bug where the snapshot was refreshed after Reset, so
        declining the save silently wiped the user's previous settings.
        Verifies the raw-bytes revert path introduced for CodeRabbit's
        "file absence on cancel" comment.
        """
        from EvoScientist.config.onboard.wizard import run_onboard
        from EvoScientist.config.settings import save_config

        config_file = tmp_path / "config.yaml"
        existing = EvoScientistConfig(
            provider="openai",
            model="gpt-5",
            openai_api_key="sk-existing",
            ui_backend="cli",
        )

        mock_q = MagicMock()
        with patch(
            "EvoScientist.config.settings.get_config_path", return_value=config_file
        ):
            save_config(existing)
            # Append a comment that ``save_config`` would strip on
            # re-serialization. The test then proves revert restored the
            # EXACT bytes (not just rewrote the snapshot via save_config
            # and got lucky because logical content matched).
            with config_file.open("ab") as f:
                f.write(b"\n# user manual comment - must survive revert\n")
            original_bytes = config_file.read_bytes()

            with (
                _patch_all_questionary(mock_q),
                patch(
                    "EvoScientist.config.onboard.wizard.get_config_path",
                    return_value=config_file,
                ),
                patch("EvoScientist.config.onboard.wizard.console"),
                patch("EvoScientist.config.onboard.steps.console"),
                patch("EvoScientist.config.onboard.channels.console"),
                patch("EvoScientist.config.onboard.helpers.console"),
                patch("EvoScientist.config.onboard.wizard._step_tinytex"),
            ):
                mock_q.select.return_value.ask.side_effect = [
                    "reset",  # Keep/Modify/Reset → Reset
                    "tui",
                    "anthropic",
                    "api_key",
                    "claude-sonnet-4-6",
                    "skip",  # Auxiliary: Skip (single driver)
                    "daemon",
                    True,
                ]
                mock_q.password.return_value.ask.side_effect = ["", ""]
                mock_q.confirm.return_value.ask.side_effect = [
                    False,  # Save? = No
                ]
                mock_q.text.return_value.ask.side_effect = [""]
                mock_q.checkbox.return_value.ask.return_value = []

                result = run_onboard(skip_validation=True)

        assert result is False
        assert config_file.exists(), "file should still exist after revert"
        assert config_file.read_bytes() == original_bytes, (
            "revert must restore EXACT pre-wizard bytes, not autosaved state"
        )

    def test_revert_removes_file_when_none_existed(self, tmp_path):
        """Brand-new user (no config.yaml) declines the final save: the file
        created by autosaves should be deleted to match the original state.
        """
        from EvoScientist.config.onboard.wizard import run_onboard

        config_file = tmp_path / "config.yaml"
        assert not config_file.exists()

        mock_q = MagicMock()
        with (
            _patch_all_questionary(mock_q),
            patch(
                "EvoScientist.config.settings.get_config_path",
                return_value=config_file,
            ),
            patch(
                "EvoScientist.config.onboard.wizard.get_config_path",
                return_value=config_file,
            ),
            patch("EvoScientist.config.onboard.wizard.console"),
            patch("EvoScientist.config.onboard.steps.console"),
            patch("EvoScientist.config.onboard.channels.console"),
            patch("EvoScientist.config.onboard.helpers.console"),
            patch("EvoScientist.config.onboard.wizard._step_tinytex"),
        ):
            mock_q.select.return_value.ask.side_effect = [
                "tui",
                "anthropic",
                "api_key",
                "claude-sonnet-4-6",
                "skip",  # Auxiliary: Skip (single driver)
                "daemon",
                True,
            ]
            mock_q.password.return_value.ask.side_effect = ["", ""]
            mock_q.confirm.return_value.ask.side_effect = [
                False,  # Save? = No
            ]
            mock_q.text.return_value.ask.side_effect = [""]
            mock_q.checkbox.return_value.ask.return_value = []

            result = run_onboard(skip_validation=True)

        assert result is False
        assert not config_file.exists(), (
            "revert must delete file that was autosaved into existence "
            "during the wizard run"
        )

    def test_preset_tavily_key_rejected_by_validator_is_fatal(self):
        """``--tavily-key`` preset must be validated like ``--api-key``;
        a rejected key raises rather than silently saving."""
        import pytest

        from EvoScientist.config.onboard.prompter import NonInteractivePrompter
        from EvoScientist.config.onboard.wizard import run_onboard

        prompter = NonInteractivePrompter(answers={"tavily_key": "tvly-bad"})

        with (
            patch(
                "EvoScientist.config.onboard.wizard.load_config",
                return_value=EvoScientistConfig(),
            ),
            patch("EvoScientist.config.onboard.wizard.save_config"),
            patch("EvoScientist.config.onboard.wizard.console"),
            patch(
                "EvoScientist.config.onboard.validators.validate_tavily_key",
                return_value=(False, "Invalid API key"),
            ),
        ):
            with pytest.raises(RuntimeError, match="--tavily-key rejected"):
                run_onboard(
                    skip_validation=False,
                    prompter=prompter,
                    only_sections={"tavily"},
                )

    def test_preset_tavily_key_skips_validation_when_flagged(self):
        """``--skip-validation`` must bypass the preset tavily check."""
        from EvoScientist.config.onboard.prompter import NonInteractivePrompter
        from EvoScientist.config.onboard.wizard import run_onboard

        prompter = NonInteractivePrompter(answers={"tavily_key": "tvly-bad"})

        mock_q = MagicMock()
        with (
            _patch_all_questionary(mock_q),
            patch(
                "EvoScientist.config.onboard.wizard.load_config",
                return_value=EvoScientistConfig(),
            ),
            patch("EvoScientist.config.onboard.wizard.save_config"),
            patch("EvoScientist.config.onboard.wizard.console"),
            patch(
                "EvoScientist.config.onboard.validators.validate_tavily_key"
            ) as mock_validate,
        ):
            mock_q.confirm.return_value.ask.side_effect = [True]  # Save? = Yes
            run_onboard(
                skip_validation=True,
                prompter=prompter,
                only_sections={"tavily"},
            )
            mock_validate.assert_not_called()


class TestOnboardCliErrorPresentation:
    """The CLI must turn the wizard's non-interactive ``RuntimeError`` signals
    into a clean message + exit code 1, not a raw Python traceback."""

    def test_runtime_error_becomes_clean_typer_exit(self):
        import typer

        from EvoScientist.cli.commands import _run_onboard_cli

        with (
            patch(
                "EvoScientist.config.onboard.run_onboard",
                side_effect=RuntimeError("--tavily-key rejected by validator: nope"),
            ),
            patch("EvoScientist.cli.commands.console") as mock_console,
        ):
            with pytest.raises(typer.Exit) as exc_info:
                _run_onboard_cli(skip_validation=False)

        assert exc_info.value.exit_code == 1
        # The error message reaches the user via console, not a traceback.
        printed = " ".join(str(c.args[0]) for c in mock_console.print.call_args_list)
        assert "rejected by validator" in printed


# =============================================================================
# Test TinyTeX helpers
# =============================================================================


class TestCheckLatexComponents:
    """Tests for _check_latex_components()."""

    def test_all_available(self):
        """All three components found → all True."""
        from EvoScientist.config.onboard.helpers import _check_latex_components

        with (
            patch("EvoScientist.config.onboard.helpers.shutil") as mock_sh,
            patch("EvoScientist.config.onboard.helpers.subprocess") as mock_sp,
        ):
            mock_sh.which.return_value = "/usr/local/bin/cmd"
            mock_sp.TimeoutExpired = subprocess.TimeoutExpired
            mock_sp.run.return_value = Mock(returncode=0)
            result = _check_latex_components()
            assert result == {"pdflatex": True, "latexmk": True, "tlmgr": True}

    def test_only_pdflatex(self):
        """Only pdflatex available."""
        from EvoScientist.config.onboard.helpers import _check_latex_components

        with (
            patch("EvoScientist.config.onboard.helpers.shutil") as mock_sh,
            patch("EvoScientist.config.onboard.helpers.subprocess") as mock_sp,
        ):
            mock_sh.which.side_effect = lambda cmd: (
                "/usr/local/bin/pdflatex" if cmd == "pdflatex" else None
            )
            mock_sp.TimeoutExpired = subprocess.TimeoutExpired
            mock_sp.run.return_value = Mock(returncode=0)
            result = _check_latex_components()
            assert result == {
                "pdflatex": True,
                "latexmk": False,
                "tlmgr": False,
            }

    def test_none_available(self):
        """Nothing found → all False."""
        from EvoScientist.config.onboard.helpers import _check_latex_components

        with patch("EvoScientist.config.onboard.helpers.shutil") as mock_sh:
            mock_sh.which.return_value = None
            result = _check_latex_components()
            assert result == {
                "pdflatex": False,
                "latexmk": False,
                "tlmgr": False,
            }


class TestAutoInstallLatexmk:
    """Tests for _auto_install_latexmk()."""

    def test_success(self):
        """tlmgr install latexmk succeeds."""
        from EvoScientist.config.onboard.helpers import _auto_install_latexmk

        with (
            patch("EvoScientist.config.onboard.helpers.shutil") as mock_sh,
            patch("EvoScientist.config.onboard.helpers.subprocess") as mock_sp,
            patch("EvoScientist.config.onboard.helpers.console") as mock_con,
        ):
            mock_sh.which.side_effect = lambda cmd: f"/usr/local/bin/{cmd}"
            mock_sp.run.return_value = Mock(returncode=0)
            mock_sp.TimeoutExpired = subprocess.TimeoutExpired
            _auto_install_latexmk()
            success_printed = any(
                "latexmk installed" in str(c) for c in mock_con.print.call_args_list
            )
            assert success_printed

    def test_tlmgr_not_found(self):
        """tlmgr not on PATH → does nothing."""
        from EvoScientist.config.onboard.helpers import _auto_install_latexmk

        with (
            patch("EvoScientist.config.onboard.helpers.shutil") as mock_sh,
            patch("EvoScientist.config.onboard.helpers.subprocess") as mock_sp,
            patch("EvoScientist.config.onboard.helpers.console"),
        ):
            mock_sh.which.return_value = None
            _auto_install_latexmk()
            mock_sp.run.assert_not_called()

    def test_install_fails(self):
        """tlmgr install returns nonzero → warns."""
        from EvoScientist.config.onboard.helpers import _auto_install_latexmk

        with (
            patch("EvoScientist.config.onboard.helpers.shutil") as mock_sh,
            patch("EvoScientist.config.onboard.helpers.subprocess") as mock_sp,
            patch("EvoScientist.config.onboard.helpers.console") as mock_con,
        ):
            mock_sh.which.side_effect = lambda cmd: (
                "/usr/local/bin/tlmgr" if cmd == "tlmgr" else None
            )
            mock_sp.run.return_value = Mock(returncode=1)
            mock_sp.TimeoutExpired = subprocess.TimeoutExpired
            _auto_install_latexmk()
            warn_printed = any(
                "Failed" in str(c) for c in mock_con.print.call_args_list
            )
            assert warn_printed


class TestCheckTinytex:
    """Tests for _check_tinytex()."""

    def test_found_pdflatex(self):
        """pdflatex found and working → True."""
        from EvoScientist.config.onboard.helpers import _check_tinytex

        with (
            patch("EvoScientist.config.onboard.helpers.shutil") as mock_sh,
            patch("EvoScientist.config.onboard.helpers.subprocess") as mock_sp,
        ):
            mock_sh.which.return_value = "/usr/local/bin/pdflatex"
            mock_sp.TimeoutExpired = subprocess.TimeoutExpired
            mock_sp.run.return_value = Mock(returncode=0)
            assert _check_tinytex() is True

    def test_tlmgr_only_not_enough(self):
        """pdflatex missing but tlmgr found → False (pdflatex is required)."""
        from EvoScientist.config.onboard.helpers import _check_tinytex

        with (
            patch("EvoScientist.config.onboard.helpers.shutil") as mock_sh,
            patch("EvoScientist.config.onboard.helpers.subprocess") as mock_sp,
        ):
            mock_sh.which.side_effect = lambda cmd: (
                "/usr/local/bin/tlmgr" if cmd == "tlmgr" else None
            )
            mock_sp.TimeoutExpired = subprocess.TimeoutExpired
            mock_sp.run.return_value = Mock(returncode=0)
            assert _check_tinytex() is False

    def test_not_found(self):
        """Neither pdflatex nor tlmgr found → False."""
        from EvoScientist.config.onboard.helpers import _check_tinytex

        with patch("EvoScientist.config.onboard.helpers.shutil") as mock_sh:
            mock_sh.which.return_value = None
            assert _check_tinytex() is False

    def test_version_timeout(self):
        """Command found but --version times out → False."""
        from EvoScientist.config.onboard.helpers import _check_tinytex

        with (
            patch("EvoScientist.config.onboard.helpers.shutil") as mock_sh,
            patch("EvoScientist.config.onboard.helpers.subprocess") as mock_sp,
        ):
            mock_sh.which.return_value = "/usr/local/bin/pdflatex"
            mock_sp.TimeoutExpired = subprocess.TimeoutExpired
            mock_sp.run.side_effect = subprocess.TimeoutExpired("pdflatex", 10)
            assert _check_tinytex() is False

    def test_version_nonzero(self):
        """Command found but --version returns nonzero → False."""
        from EvoScientist.config.onboard.helpers import _check_tinytex

        with (
            patch("EvoScientist.config.onboard.helpers.shutil") as mock_sh,
            patch("EvoScientist.config.onboard.helpers.subprocess") as mock_sp,
        ):
            mock_sp.TimeoutExpired = subprocess.TimeoutExpired
            mock_sp.run.return_value = Mock(returncode=1)
            # pdflatex fails, tlmgr not found
            mock_sh.which.side_effect = lambda cmd: (
                "/usr/local/bin/pdflatex" if cmd == "pdflatex" else None
            )
            assert _check_tinytex() is False


class TestDetectTinytexInstallMethod:
    """Tests for _detect_tinytex_install_method()."""

    def test_macos_with_curl(self):
        """macOS with curl → curl method."""
        from EvoScientist.config.onboard.helpers import _detect_tinytex_install_method

        with (
            patch("EvoScientist.config.onboard.helpers.sys") as mock_sys,
            patch("EvoScientist.config.onboard.helpers.shutil") as mock_sh,
        ):
            mock_sys.platform = "darwin"
            mock_sh.which.side_effect = lambda cmd: (
                "/usr/bin/curl" if cmd == "curl" else None
            )
            method, command = _detect_tinytex_install_method()
            assert method == "curl"
            assert "install-bin-unix.sh" in command

    def test_linux_wget_fallback(self):
        """Linux without curl, with wget → wget method."""
        from EvoScientist.config.onboard.helpers import _detect_tinytex_install_method

        with (
            patch("EvoScientist.config.onboard.helpers.sys") as mock_sys,
            patch("EvoScientist.config.onboard.helpers.shutil") as mock_sh,
        ):
            mock_sys.platform = "linux"
            mock_sh.which.side_effect = lambda cmd: (
                "/usr/bin/wget" if cmd == "wget" else None
            )
            method, command = _detect_tinytex_install_method()
            assert method == "wget"
            assert "install-bin-unix.sh" in command

    def test_windows_choco(self):
        """Windows with choco → choco method."""
        from EvoScientist.config.onboard.helpers import _detect_tinytex_install_method

        with (
            patch("EvoScientist.config.onboard.helpers.sys") as mock_sys,
            patch("EvoScientist.config.onboard.helpers.shutil") as mock_sh,
        ):
            mock_sys.platform = "win32"
            mock_sh.which.side_effect = lambda cmd: (
                "C:\\choco\\choco.exe" if cmd == "choco" else None
            )
            method, command = _detect_tinytex_install_method()
            assert method == "choco"
            assert "tinytex" in command

    def test_windows_scoop(self):
        """Windows with scoop (no choco) → scoop method."""
        from EvoScientist.config.onboard.helpers import _detect_tinytex_install_method

        with (
            patch("EvoScientist.config.onboard.helpers.sys") as mock_sys,
            patch("EvoScientist.config.onboard.helpers.shutil") as mock_sh,
        ):
            mock_sys.platform = "win32"
            mock_sh.which.side_effect = lambda cmd: (
                "C:\\scoop\\scoop.exe" if cmd == "scoop" else None
            )
            method, command = _detect_tinytex_install_method()
            assert method == "scoop"
            assert "tinytex" in command

    def test_no_tools(self):
        """No tools available → manual method."""
        from EvoScientist.config.onboard.helpers import _detect_tinytex_install_method

        with (
            patch("EvoScientist.config.onboard.helpers.sys") as mock_sys,
            patch("EvoScientist.config.onboard.helpers.shutil") as mock_sh,
        ):
            mock_sys.platform = "linux"
            mock_sh.which.return_value = None
            method, command = _detect_tinytex_install_method()
            assert method == "manual"
            assert "yihui.org" in command


class TestInstallTinytex:
    """Tests for _install_tinytex()."""

    def test_curl_install_success(self):
        """curl install succeeds → True."""
        from EvoScientist.config.onboard.helpers import _install_tinytex

        with patch("EvoScientist.config.onboard.helpers.subprocess") as mock_sp:
            mock_sp.run.return_value = Mock(returncode=0)
            mock_sp.TimeoutExpired = subprocess.TimeoutExpired
            assert _install_tinytex("curl", "curl -sL ... | sh") is True
            mock_sp.run.assert_called_once()
            # Verify shell=True was used for pipe commands
            _, kwargs = mock_sp.run.call_args
            assert kwargs.get("shell") is True

    def test_curl_install_timeout(self):
        """curl install times out → False."""
        from EvoScientist.config.onboard.helpers import _install_tinytex

        with (
            patch("EvoScientist.config.onboard.helpers.subprocess") as mock_sp,
            patch("EvoScientist.config.onboard.helpers.console"),
        ):
            mock_sp.run.side_effect = subprocess.TimeoutExpired("curl", 300)
            mock_sp.TimeoutExpired = subprocess.TimeoutExpired
            assert _install_tinytex("curl", "curl -sL ... | sh") is False

    def test_choco_install_success(self):
        """choco install succeeds → True."""
        from EvoScientist.config.onboard.helpers import _install_tinytex

        with (
            patch("EvoScientist.config.onboard.helpers.subprocess") as mock_sp,
            patch("EvoScientist.config.onboard.helpers.shutil") as mock_sh,
        ):
            mock_sh.which.return_value = "C:\\choco\\choco.exe"
            mock_sp.run.return_value = Mock(returncode=0)
            mock_sp.TimeoutExpired = subprocess.TimeoutExpired
            assert _install_tinytex("choco", "choco install tinytex -y") is True

    def test_manual_returns_false(self):
        """manual method → False immediately."""
        from EvoScientist.config.onboard.helpers import _install_tinytex

        assert _install_tinytex("manual", "https://yihui.org/tinytex/") is False


class TestStepTinytex:
    """Tests for _step_tinytex()."""

    def test_user_declines_prepare(self):
        """User says No to 'Prepare LaTeX environment?' → skipped."""
        from EvoScientist.config.onboard.steps import _step_tinytex

        with (
            patch("EvoScientist.config.onboard.steps.questionary") as mock_q,
            patch("EvoScientist.config.onboard.steps._print_step_skipped") as mock_ps,
            patch("EvoScientist.config.onboard.steps.console"),
        ):
            mock_q.select.return_value.ask.return_value = False
            _step_tinytex()
            mock_ps.assert_called_once_with("LaTeX", "skipped")

    def test_already_installed_all_components(self):
        """User says Yes, all components available → prints detailed status."""
        from EvoScientist.config.onboard.steps import _step_tinytex

        with (
            patch("EvoScientist.config.onboard.steps.questionary") as mock_q,
            patch(
                "EvoScientist.config.onboard.steps._check_latex_components",
                return_value={
                    "pdflatex": True,
                    "latexmk": True,
                    "tlmgr": True,
                },
            ),
            patch(
                "EvoScientist.config.onboard.steps._print_latex_status"
            ) as mock_status,
            patch("EvoScientist.config.onboard.steps.console"),
        ):
            mock_q.select.return_value.ask.return_value = True
            _step_tinytex()
            mock_status.assert_called_once()

    def test_already_installed_missing_latexmk(self):
        """pdflatex + tlmgr present but latexmk missing → auto-installs."""
        from EvoScientist.config.onboard.steps import _step_tinytex

        with (
            patch("EvoScientist.config.onboard.steps.questionary") as mock_q,
            patch(
                "EvoScientist.config.onboard.steps._check_latex_components",
                return_value={
                    "pdflatex": True,
                    "latexmk": False,
                    "tlmgr": True,
                },
            ),
            patch("EvoScientist.config.onboard.steps._print_latex_status"),
            patch(
                "EvoScientist.config.onboard.steps._auto_install_latexmk"
            ) as mock_auto,
            patch("EvoScientist.config.onboard.steps.console"),
        ):
            mock_q.select.return_value.ask.return_value = True
            _step_tinytex()
            mock_auto.assert_called_once()

    def test_user_installs_successfully(self):
        """Yes → not found → confirms install → succeeds → re-check passes."""
        from EvoScientist.config.onboard.steps import _step_tinytex

        all_false = {"pdflatex": False, "latexmk": False, "tlmgr": False}
        all_true = {"pdflatex": True, "latexmk": True, "tlmgr": True}
        with (
            patch(
                "EvoScientist.config.onboard.steps._check_latex_components",
                side_effect=[all_false, all_true],
            ),
            patch(
                "EvoScientist.config.onboard.steps._detect_tinytex_install_method",
                return_value=("curl", "curl ... | sh"),
            ),
            patch(
                "EvoScientist.config.onboard.steps._install_tinytex",
                return_value=True,
            ),
            patch("EvoScientist.config.onboard.steps.questionary") as mock_q,
            patch("EvoScientist.config.onboard.steps._print_step_result") as mock_pr,
            patch("EvoScientist.config.onboard.steps._print_latex_status"),
            patch("EvoScientist.config.onboard.steps.console"),
        ):
            mock_q.select.return_value.ask.return_value = True
            mock_q.confirm.return_value.ask.return_value = True
            _step_tinytex()
            mock_pr.assert_called_once_with("LaTeX", "TinyTeX installed")

    def test_user_declines_install(self):
        """Yes to prepare → not found → declines install → skipped."""
        from EvoScientist.config.onboard.steps import _step_tinytex

        all_false = {"pdflatex": False, "latexmk": False, "tlmgr": False}
        with (
            patch(
                "EvoScientist.config.onboard.steps._check_latex_components",
                return_value=all_false,
            ),
            patch(
                "EvoScientist.config.onboard.steps._detect_tinytex_install_method",
                return_value=("curl", "curl ... | sh"),
            ),
            patch("EvoScientist.config.onboard.steps.questionary") as mock_q,
            patch("EvoScientist.config.onboard.steps._print_step_skipped") as mock_ps,
            patch("EvoScientist.config.onboard.steps.console"),
        ):
            mock_q.select.return_value.ask.return_value = True
            mock_q.confirm.return_value.ask.return_value = False
            _step_tinytex()
            mock_ps.assert_called_once_with("LaTeX", "skipped")

    def test_install_fails(self):
        """Yes → not found → confirms install → install fails."""
        from EvoScientist.config.onboard.steps import _step_tinytex

        all_false = {"pdflatex": False, "latexmk": False, "tlmgr": False}
        with (
            patch(
                "EvoScientist.config.onboard.steps._check_latex_components",
                return_value=all_false,
            ),
            patch(
                "EvoScientist.config.onboard.steps._detect_tinytex_install_method",
                return_value=("curl", "curl ... | sh"),
            ),
            patch(
                "EvoScientist.config.onboard.steps._install_tinytex",
                return_value=False,
            ),
            patch("EvoScientist.config.onboard.steps.questionary") as mock_q,
            patch("EvoScientist.config.onboard.steps._print_step_result") as mock_pr,
            patch("EvoScientist.config.onboard.steps.console"),
        ):
            mock_q.select.return_value.ask.return_value = True
            mock_q.confirm.return_value.ask.return_value = True
            _step_tinytex()
            mock_pr.assert_called_once_with(
                "LaTeX", "installation failed", success=False
            )

    def test_installed_but_not_in_path(self):
        """Install succeeds but pdflatex not yet in PATH → warns user."""
        from EvoScientist.config.onboard.steps import _step_tinytex

        all_false = {"pdflatex": False, "latexmk": False, "tlmgr": False}
        with (
            patch(
                "EvoScientist.config.onboard.steps._check_latex_components",
                side_effect=[all_false, all_false],
            ),
            patch(
                "EvoScientist.config.onboard.steps._detect_tinytex_install_method",
                return_value=("curl", "curl ... | sh"),
            ),
            patch(
                "EvoScientist.config.onboard.steps._install_tinytex",
                return_value=True,
            ),
            patch("EvoScientist.config.onboard.steps.questionary") as mock_q,
            patch("EvoScientist.config.onboard.steps._print_step_result") as mock_pr,
            patch("EvoScientist.config.onboard.steps.console") as mock_con,
        ):
            mock_q.select.return_value.ask.return_value = True
            mock_q.confirm.return_value.ask.return_value = True
            _step_tinytex()
            path_warning_printed = any(
                "PATH" in str(call) for call in mock_con.print.call_args_list
            )
            assert path_warning_printed
            mock_pr.assert_called_once_with(
                "LaTeX", "installed (restart terminal for PATH)"
            )

    def test_manual_method(self):
        """Yes to prepare → not found → manual method → prints URL, no install prompt."""
        from EvoScientist.config.onboard.steps import _step_tinytex

        all_false = {"pdflatex": False, "latexmk": False, "tlmgr": False}
        with (
            patch(
                "EvoScientist.config.onboard.steps._check_latex_components",
                return_value=all_false,
            ),
            patch(
                "EvoScientist.config.onboard.steps._detect_tinytex_install_method",
                return_value=("manual", "https://yihui.org/tinytex/"),
            ),
            patch("EvoScientist.config.onboard.steps.questionary") as mock_q,
            patch("EvoScientist.config.onboard.steps._print_step_skipped") as mock_ps,
            patch("EvoScientist.config.onboard.steps.console"),
        ):
            mock_q.select.return_value.ask.return_value = True
            _step_tinytex()
            mock_ps.assert_called_once_with("LaTeX", "manual install needed")
