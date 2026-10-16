"""Input validators for the onboarding wizard.

- IntegerValidator / ChoiceValidator: prompt_toolkit Validators
- validate_*_key: per-provider API key validators (live HTTP probes), defined
  in :mod:`EvoScientist.config.provider_keys` and re-exported here
"""

from __future__ import annotations

from prompt_toolkit.validation import ValidationError, Validator

from ..provider_keys import (  # noqa: F401  (re-exported for the wizard)
    _AUTH_FAILURE_HINTS,
    _TRANSIENT_HINTS,
    _classify_validation_error,
    validate_anthropic_key,
    validate_atlascloud_key,
    validate_dashscope_code_key,
    validate_dashscope_key,
    validate_deepseek_key,
    validate_google_key,
    validate_kimi_key,
    validate_minimax_key,
    validate_moonshot_key,
    validate_novita_key,
    validate_nvidia_key,
    validate_openai_key,
    validate_openrouter_key,
    validate_opper_key,
    validate_requesty_key,
    validate_siliconflow_key,
    validate_tavily_key,
    validate_volcengine_key,
    validate_xiaomi_key,
    validate_zhipu_key,
)


class IntegerValidator(Validator):
    """Validates that input is a positive integer."""

    def __init__(self, min_value: int = 1, max_value: int = 100):
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, document) -> None:
        text = document.text.strip()
        if not text:
            return  # Allow empty for default
        try:
            value = int(text)
            if value < self.min_value or value > self.max_value:
                raise ValidationError(
                    message=f"Must be between {self.min_value} and {self.max_value}"
                )
        except ValueError as e:
            raise ValidationError(message="Must be a valid integer") from e


class ChoiceValidator(Validator):
    """Validates that input is one of the allowed choices."""

    def __init__(self, choices: list[str], allow_empty: bool = True):
        self.choices = choices
        self.allow_empty = allow_empty

    def validate(self, document) -> None:
        text = document.text.strip().lower()
        if not text and self.allow_empty:
            return
        if text not in [c.lower() for c in self.choices]:
            raise ValidationError(message=f"Must be one of: {', '.join(self.choices)}")
