"""Onboarding keeps an explicitly saved legacy model without advertising it."""

from unittest.mock import patch

import pytest

from EvoScientist.config.onboard.steps import _step_model
from EvoScientist.config.settings import EvoScientistConfig


@pytest.mark.parametrize("saved", ["minimax-m2.7", "MiniMax-M2.5-highspeed"])
def test_saved_legacy_model_is_still_default(saved):
    config = EvoScientistConfig(provider="minimax", model=saved)
    with patch("EvoScientist.config.onboard.steps.questionary.select") as select:
        select.return_value.ask.return_value = saved
        assert _step_model(config, "minimax") == saved
    assert select.call_args.kwargs["default"] == saved
    choices = select.call_args.kwargs["choices"]
    assert any(
        choice.value == saved and "saved, legacy" in choice.title for choice in choices
    )
    assert config.model == saved


def test_new_minimax_selection_defaults_to_flash():
    config = EvoScientistConfig()
    with patch("EvoScientist.config.onboard.steps.questionary.select") as select:
        select.return_value.ask.return_value = "minimax-m3.1-flash"
        assert _step_model(config, "minimax") == "minimax-m3.1-flash"
    assert select.call_args.kwargs["default"] == "minimax-m3.1-flash"
    assert [choice.value for choice in select.call_args.kwargs["choices"]] == [
        "minimax-m3.1-flash",
        "minimax-m3",
        "__custom__",
    ]


def test_saved_auxiliary_legacy_selection_stays_on_its_provider():
    config = EvoScientistConfig()
    with patch("EvoScientist.config.onboard.steps.questionary.select") as select:
        select.return_value.ask.return_value = "minimax-m2.5"
        assert (
            _step_model(config, "siliconflow", default_value="minimax-m2.5")
            == "minimax-m2.5"
        )
    assert select.call_args.kwargs["default"] == "minimax-m2.5"
