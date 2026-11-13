"""How the Rich CLI reports a failed turn (#782)."""

from __future__ import annotations

import pytest
from rich.console import Console

from EvoScientist.cli import interactive
from EvoScientist.llm.errors import ProviderStreamError
from EvoScientist.middleware.error_normalization import MissingApiKeyError


@pytest.fixture
def recorded(monkeypatch) -> Console:
    console = Console(record=True, width=200)
    monkeypatch.setattr(interactive, "console", console)
    return console


def test_missing_key_prints_the_onboarding_hint_and_returns(recorded):
    # What reaches the loop: the guard's error, normalized at the model boundary.
    missing = MissingApiKeyError(
        "No Anthropic API key is set. Save it with "
        "`EvoSci config set anthropic_api_key <key>`, set the "
        "ANTHROPIC_API_KEY environment variable, or run `EvoSci onboard`."
    )
    exc = ProviderStreamError(
        provider="anthropic",
        class_qualname="EvoScientist.middleware.error_normalization.MissingApiKeyError",
        message=str(missing),
    )

    # Returns instead of raising or ending the session, so the next prompt
    # can still run ``/model`` to pick a model that has a key.
    assert interactive._print_turn_error(exc) is None

    out = recorded.export_text()
    assert "Error: API key not configured." in out
    assert "Run EvoSci onboard to set up your API key." in out


def test_other_errors_print_their_message(recorded):
    interactive._print_turn_error(RuntimeError("Connection error. [x]"))

    out = recorded.export_text()
    assert "Error: Connection error. [x]" in out
    assert "API key not configured" not in out
