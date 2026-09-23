"""deepagents 0.7.0 merges caller middleware into its default stack by `.name`:
a name match silently REPLACES the built-in. None of EvoScientist's middleware
may collide unintentionally. TodoListMiddleware is deliberately absent from
the forbidden set: we pass it on purpose and replacing a profile-added
instance (e.g. the Codex harness profile's) with our identical one is
desired dedup.

SummarizationMiddleware stays in the forbidden set for the no-backend list.
The per-run subclass is appended only when a backend is supplied, and that
one deliberate overlap is asserted separately (#466).
"""

from unittest.mock import MagicMock, patch

DEEPAGENTS_BASE_STACK_NAMES = {
    "SkillsMiddleware",
    "FilesystemMiddleware",
    "SubAgentMiddleware",
    "SummarizationMiddleware",
    "PatchToolCallsMiddleware",
    "AsyncSubAgentMiddleware",
    "AnthropicPromptCachingMiddleware",
}


def test_no_name_collision_with_deepagents_base_stack():
    from EvoScientist.EvoScientist import _get_default_middleware

    ours = {m.name for m in _get_default_middleware()}
    assert not ours & DEEPAGENTS_BASE_STACK_NAMES

    ours_async = {m.name for m in _get_default_middleware(for_async_subagent=True)}
    assert not ours_async & DEEPAGENTS_BASE_STACK_NAMES


@patch(
    "EvoScientist.middleware.create_tool_selector_middleware",
    return_value=[MagicMock(), MagicMock()],
)
@patch("EvoScientist.EvoScientist._ensure_chat_model")
@patch("EvoScientist.EvoScientist._ensure_config")
def test_backend_summarization_is_the_only_deliberate_collision(
    mock_config, mock_model, mock_ts
):
    from EvoScientist.EvoScientist import _get_default_middleware
    from EvoScientist.middleware.summarization import (
        _PerRunLimitsSummarizationMiddleware,
    )

    mock_model.return_value = MagicMock(profile={"max_input_tokens": 200_000})
    cfg = MagicMock()
    cfg.enable_ask_user = False
    cfg.auto_approve = False
    cfg.auxiliary_model = ""
    cfg.auxiliary_provider = ""
    mock_config.return_value = cfg

    mw = _get_default_middleware(backend=MagicMock())
    overlap = [m for m in mw if m.name in DEEPAGENTS_BASE_STACK_NAMES]
    assert [m.name for m in overlap] == ["SummarizationMiddleware"]
    assert isinstance(overlap[0], _PerRunLimitsSummarizationMiddleware)
