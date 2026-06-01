"""Canonical valid-value sets shared by the wizard and CLI flag validation.

The interactive ``_step_*`` functions in ``steps.py`` use these for ``Choice``
construction (or are checked against them by tests). The CLI ``onboard``
command in ``cli/commands.py`` uses them to validate ``--provider`` /
``--ui`` / ``--workspace-mode`` flag inputs.

Single source of truth — adding a new provider here AND to the corresponding
``Choice(value=...)`` in ``steps.py`` is required; a drift test in
``tests/test_onboard.py`` keeps both sides in sync.
"""

from __future__ import annotations

VALID_PROVIDERS: frozenset[str] = frozenset(
    {
        "anthropic",
        "openai",
        "google-genai",
        "minimax",
        "zhipu",
        "zhipu-code",
        "volcengine",
        "dashscope",
        "dashscope-code",
        "deepseek",
        "moonshot",
        "kimi-coding",
        "ollama",
        "nvidia",
        "siliconflow",
        "openrouter",
        "custom-openai",
        "custom-anthropic",
    }
)

VALID_UI_BACKENDS: frozenset[str] = frozenset({"tui", "cli", "webui"})

VALID_WORKSPACE_MODES: frozenset[str] = frozenset({"daemon", "run"})


__all__ = [
    "VALID_PROVIDERS",
    "VALID_UI_BACKENDS",
    "VALID_WORKSPACE_MODES",
]
