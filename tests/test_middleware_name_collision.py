"""deepagents 0.7.0 merges caller middleware into its default stack by `.name`:
a name match silently REPLACES the built-in. None of EvoScientist's middleware
may collide unintentionally. Two deliberate carve-outs exist:

- TodoListMiddleware: we pass it on purpose and replacing a profile-added
  instance (e.g. the Codex harness profile's) with our identical one is
  desired dedup.
- SummarizationMiddleware: we pass a same-named subclass on purpose
  (`_PerRunLimitsSummarizationMiddleware`, appended by
  `_get_default_middleware` only when a backend is supplied) so deepagents'
  name-based merge replaces the frozen-construction-limits built-in with our
  per-run-limits version in the identical core-stack slot (#466).
"""

DEEPAGENTS_BASE_STACK_NAMES = {
    "SkillsMiddleware",
    "FilesystemMiddleware",
    "SubAgentMiddleware",
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
