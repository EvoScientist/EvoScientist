"""Startup check for ChatGPT sign-in (no network, no real config)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import typer

from EvoScientist.cli import commands
from EvoScientist.llm.chatgpt_oauth import SIGN_IN_AGAIN, SIGN_IN_HINT, ChatGPTLogin


def _config(mode="oauth"):
    return SimpleNamespace(openai_auth_mode=mode, anthropic_auth_mode="api_key")


def _printed(mock_console) -> str:
    return " ".join(str(call.args[0]) for call in mock_console.print.call_args_list)


def test_api_key_mode_checks_nothing():
    with patch("EvoScientist.llm.chatgpt_oauth.status") as mock_status:
        commands._require_chatgpt_sign_in(_config("api_key"), interactive=True)

    mock_status.assert_not_called()


def test_signed_in_passes_silently():
    with (
        patch(
            "EvoScientist.llm.chatgpt_oauth.status",
            return_value=ChatGPTLogin(signed_in=True, plan="plus"),
        ),
        patch.object(commands, "console") as mock_console,
    ):
        commands._require_chatgpt_sign_in(_config(), interactive=True)

    mock_console.print.assert_not_called()


def test_headless_exits_with_the_hint():
    with (
        patch(
            "EvoScientist.llm.chatgpt_oauth.status",
            return_value=ChatGPTLogin(signed_in=False),
        ),
        patch("questionary.confirm") as mock_confirm,
        patch.object(commands, "console") as mock_console,
        pytest.raises(typer.Exit) as exc_info,
    ):
        commands._require_chatgpt_sign_in(_config(), interactive=False)

    assert exc_info.value.exit_code == 1
    assert SIGN_IN_HINT in _printed(mock_console)
    mock_confirm.assert_not_called()


def test_interactive_login_continues_startup():
    question = MagicMock()
    question.ask.return_value = True
    with (
        patch(
            "EvoScientist.llm.chatgpt_oauth.status",
            return_value=ChatGPTLogin(signed_in=False),
        ),
        patch("questionary.confirm", return_value=question) as mock_confirm,
        patch(
            "EvoScientist.config.onboard.helpers._run_chatgpt_login", return_value=True
        ) as mock_login,
    ):
        commands._require_chatgpt_sign_in(_config(), interactive=True)

    assert mock_confirm.call_args.args[0] == "Sign in with ChatGPT now?"
    mock_login.assert_called_once_with()


@pytest.mark.parametrize(
    ("answer", "login_ok"), [(False, None), (None, None), (True, False)]
)
def test_interactive_refusal_or_failure_says_how_to_retry(answer, login_ok):
    question = MagicMock()
    question.ask.return_value = answer
    with (
        patch(
            "EvoScientist.llm.chatgpt_oauth.status",
            return_value=ChatGPTLogin(signed_in=False),
        ),
        patch("questionary.confirm", return_value=question),
        patch(
            "EvoScientist.config.onboard.helpers._run_chatgpt_login",
            return_value=login_ok,
        ) as mock_login,
        patch.object(commands, "console") as mock_console,
        pytest.raises(typer.Exit) as exc_info,
    ):
        commands._require_chatgpt_sign_in(_config(), interactive=True)

    assert exc_info.value.exit_code == 1
    assert SIGN_IN_AGAIN in _printed(mock_console)
    assert "Start EvoSci in a terminal" not in _printed(mock_console)
    assert mock_login.called is (answer is True)


def test_unreadable_token_is_named_before_the_hint():
    with (
        patch(
            "EvoScientist.llm.chatgpt_oauth.status",
            return_value=ChatGPTLogin(
                signed_in=False, problem="token file is unreadable"
            ),
        ),
        patch.object(commands, "console") as mock_console,
        pytest.raises(typer.Exit),
    ):
        commands._require_chatgpt_sign_in(_config(), interactive=False)

    assert "token file is unreadable" in _printed(mock_console)


class _Tty:
    def __init__(self, tty):
        self._tty = tty

    def isatty(self):
        return self._tty


@pytest.mark.parametrize(
    ("prompt", "stdin", "stdout", "expected"),
    [
        (None, _Tty(True), _Tty(True), True),
        ("", _Tty(True), _Tty(True), True),
        ("hi", _Tty(True), _Tty(True), False),
        (None, _Tty(False), _Tty(True), False),
        (None, _Tty(True), _Tty(False), False),
        (None, None, _Tty(True), False),
        (None, _Tty(True), None, False),
    ],
)
def test_interactive_start_needs_a_person_at_the_terminal(
    monkeypatch, prompt, stdin, stdout, expected
):
    monkeypatch.setattr(commands.sys, "stdin", stdin)
    monkeypatch.setattr(commands.sys, "stdout", stdout)

    assert commands._interactive_start(prompt) is expected


class _Checked(Exception):
    """Stops the command once the sign-in check has been reached."""


def _run_until_the_check(monkeypatch, argv):
    from typer.testing import CliRunner

    import EvoScientist.config as config_module
    from EvoScientist.cli._app import app
    from EvoScientist.config import EvoScientistConfig

    seen = []

    def _check(config, *, interactive):
        seen.append((config.openai_auth_mode, interactive))
        raise _Checked

    monkeypatch.setattr(
        config_module,
        "get_effective_config",
        lambda overrides=None: EvoScientistConfig(openai_auth_mode="oauth"),
    )
    monkeypatch.setattr(config_module, "apply_config_to_env", lambda config: None)
    monkeypatch.setattr(commands, "_require_chatgpt_sign_in", _check)
    result = CliRunner().invoke(app, argv, catch_exceptions=True)
    return seen, result


def test_single_shot_start_runs_the_check_without_a_prompt(monkeypatch):
    seen, result = _run_until_the_check(monkeypatch, ["-p", "hi"])

    assert seen == [("oauth", False)]
    assert isinstance(result.exception, _Checked)


def test_serve_runs_the_check_without_a_prompt(monkeypatch):
    seen, result = _run_until_the_check(monkeypatch, ["serve"])

    assert seen == [("oauth", False)]
    assert isinstance(result.exception, _Checked)


@pytest.mark.parametrize(
    ("stored", "flag", "modes"),
    [
        ({"provider": "anthropic"}, "oauth", ("oauth", "api_key")),
        ({"provider": "openai"}, "oauth", ("api_key", "oauth")),
        ({"provider": "openrouter"}, "oauth", ("api_key", "api_key")),
        (
            {"provider": "anthropic", "auxiliary_provider": "openai"},
            "oauth",
            ("oauth", "oauth"),
        ),
        (
            {"provider": "openai", "model_fallbacks": "claude-x:anthropic"},
            "oauth",
            ("oauth", "oauth"),
        ),
        (
            {"provider": "anthropic", "model_fallbacks": " gpt-6-sol : openai ,"},
            "oauth",
            ("oauth", "oauth"),
        ),
        (
            {"provider": "anthropic", "model_fallbacks": "gpt-6-sol:openrouter"},
            "oauth",
            ("oauth", "api_key"),
        ),
        (
            {"provider": "anthropic", "openai_auth_mode": "oauth"},
            "api_key",
            ("api_key", "api_key"),
        ),
    ],
)
def test_auth_mode_flag_applies_oauth_only_to_providers_in_use(
    monkeypatch, stored, flag, modes
):
    import dataclasses

    from typer.testing import CliRunner

    import EvoScientist.config as config_module
    from EvoScientist.cli._app import app
    from EvoScientist.config import EvoScientistConfig

    base = EvoScientistConfig(**stored)
    seen = []

    def _check(config, *, interactive):
        seen.append((config.anthropic_auth_mode, config.openai_auth_mode))
        raise _Checked

    monkeypatch.setattr(
        config_module,
        "get_effective_config",
        lambda overrides=None: dataclasses.replace(base, **(overrides or {})),
    )
    monkeypatch.setattr(config_module, "apply_config_to_env", lambda config: None)
    monkeypatch.setattr(commands, "_require_chatgpt_sign_in", _check)
    CliRunner().invoke(app, ["--auth-mode", flag, "-p", "hi"])

    assert seen == [modes]
