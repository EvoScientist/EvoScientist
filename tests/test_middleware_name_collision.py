"""deepagents 0.7.0 merges caller middleware into its default stack by `.name`:
a name match silently REPLACES the built-in. None of EvoScientist's middleware
may collide unintentionally. TodoListMiddleware is deliberately absent from the
forbidden set: we pass it on purpose and replacing a profile-added instance
(e.g. the Codex harness profile's) with our identical one is desired dedup.
"""

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
