"""Unit tests for /experts and /expert slash commands."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest

from EvoScientist.commands.base import ChannelRuntime, CommandContext
from EvoScientist.commands.implementation.experts import (
    ExpertCommand,
    ExpertsCommand,
    invalidate_experts_cache,
)


@pytest.fixture(autouse=True)
def _bust_experts_cache_between_tests():
    """The dispatchable-experts cache in ``experts.py`` is module-level; without
    resetting it, a test that patches ``list_expert_skills`` sees the previous
    test's fakes.
    """
    invalidate_experts_cache()
    yield
    invalidate_experts_cache()


class _FakeUI:
    """Minimal CommandUI capturing outputs for assertion."""

    supports_interactive = False

    def __init__(self) -> None:
        self.lines: list[tuple[str, str]] = []
        self.mounted: list[Any] = []

    def append_system(self, text: str, style: str = "dim") -> None:
        self.lines.append((text, style))

    def mount_renderable(self, renderable: Any) -> None:
        self.mounted.append(renderable)


@dataclass
class _FakeSkillInfo:
    """Enough of ``SkillInfo`` for the commands to render."""

    name: str
    description: str = ""
    role: str = ""
    type: str = "expert"
    tags: list[str] = field(default_factory=list)
    source: str = "builtin"
    # Non-empty by default so the fake passes the empty-body filter in
    # ``list_dispatchable_experts``. Tests that specifically want to
    # exercise the empty-body reject path pass ``body=""``.
    body: str = "persona"
    # Legacy-frontmatter expert by default: ``expert_prompt_body`` reads
    # ``body`` for these. Set ``expert_source="agents_md"`` plus
    # ``agents_body`` to fake an expert on the current contract.
    expert_source: str = "frontmatter"
    agents_body: str = ""


def _make_ctx(active_teams: list[str] | None = None) -> tuple[CommandContext, _FakeUI]:
    ui = _FakeUI()
    runtime = ChannelRuntime()
    if active_teams:
        runtime.active_teams = list(active_teams)
    ctx = CommandContext(
        agent=None,
        thread_id="t1",
        ui=ui,
        channel_runtime=runtime,
    )
    return ctx, ui


class TestExpertsList:
    async def test_lists_installed_experts_in_table(self):
        ctx, ui = _make_ctx()
        with patch(
            "EvoScientist.tools.skills_manager.list_expert_skills",
            return_value=[
                _FakeSkillInfo(
                    name="idea-brainstorm",
                    role="Research idea brainstormer",
                ),
            ],
        ):
            await ExpertsCommand().execute(ctx, args=[])
        # A Rich Table was mounted, and the no-experts-invited hint appeared.
        assert len(ui.mounted) == 1
        assert any("No experts invited" in text for text, _ in ui.lines)

    async def test_empty_list_prints_help_hint(self):
        ctx, ui = _make_ctx()
        with patch(
            "EvoScientist.tools.skills_manager.list_expert_skills",
            return_value=[],
        ):
            await ExpertsCommand().execute(ctx, args=[])
        assert any("No expert skills installed" in text for text, _ in ui.lines)
        assert not ui.mounted

    async def test_active_expert_marked_in_table(self):
        ctx, ui = _make_ctx(active_teams=["idea-brainstorm"])
        with patch(
            "EvoScientist.tools.skills_manager.list_expert_skills",
            return_value=[
                _FakeSkillInfo(
                    name="idea-brainstorm",
                    role="Research idea brainstormer",
                ),
            ],
        ):
            await ExpertsCommand().execute(ctx, args=[])
        assert any("Active: idea-brainstorm" in text for text, _ in ui.lines)


class TestExpertToggle:
    async def test_missing_arg_prints_usage(self):
        ctx, ui = _make_ctx()
        await ExpertCommand().execute(ctx, args=[])
        assert any("Usage:" in text for text, _ in ui.lines)

    async def test_unknown_expert_errors(self):
        ctx, ui = _make_ctx()
        with patch(
            "EvoScientist.tools.skills_manager.list_expert_skills",
            return_value=[_FakeSkillInfo(name="idea-brainstorm")],
        ):
            await ExpertCommand().execute(ctx, args=["not-an-expert"])
        assert any(
            "No expert skill named 'not-an-expert'" in text for text, _ in ui.lines
        )
        assert ctx.channel_runtime.active_teams == []

    async def test_async_outage_does_not_block_invite(self):
        """An async outage must not make an installed expert un-invitable.

        The old per-skill classification refused here whenever
        ``enable_async_subagents`` was off or langgraph dev was unreachable.
        Every expert now keeps its in-turn reach, so the outage degrades the
        reach rather than removing the expert.
        """
        ctx, _ui = _make_ctx()
        with (
            patch(
                "EvoScientist.tools.skills_manager.list_expert_skills",
                return_value=[_FakeSkillInfo(name="literature-review")],
            ),
            patch(
                "EvoScientist.langgraph_dev.manager.is_async_subagents_available",
                return_value=False,
            ),
        ):
            await ExpertCommand().execute(ctx, args=["literature-review"])
        assert ctx.channel_runtime.active_teams == ["literature-review"]

    async def test_invite_adds_to_active_teams(self):
        ctx, ui = _make_ctx()
        with patch(
            "EvoScientist.tools.skills_manager.list_expert_skills",
            return_value=[_FakeSkillInfo(name="idea-brainstorm")],
        ):
            await ExpertCommand().execute(ctx, args=["idea-brainstorm"])
        assert ctx.channel_runtime.active_teams == ["idea-brainstorm"]
        assert any("Invited expert: idea-brainstorm" in text for text, _ in ui.lines)

    async def test_invite_matches_name_case_insensitively(self):
        """Execute honours the same case-insensitive match as completion."""
        ctx, ui = _make_ctx()
        with patch(
            "EvoScientist.tools.skills_manager.list_expert_skills",
            return_value=[_FakeSkillInfo(name="idea-brainstorm")],
        ):
            await ExpertCommand().execute(ctx, args=["Idea-Brainstorm"])
        assert ctx.channel_runtime.active_teams == ["idea-brainstorm"]
        assert any("Invited expert: idea-brainstorm" in text for text, _ in ui.lines)

    async def test_toggle_dismisses_when_already_invited(self):
        ctx, ui = _make_ctx(active_teams=["idea-brainstorm"])
        with patch(
            "EvoScientist.tools.skills_manager.list_expert_skills",
            return_value=[_FakeSkillInfo(name="idea-brainstorm")],
        ):
            await ExpertCommand().execute(ctx, args=["idea-brainstorm"])
        assert ctx.channel_runtime.active_teams == []
        assert any("Dismissed expert: idea-brainstorm" in text for text, _ in ui.lines)

    async def test_clear_dismisses_all(self):
        ctx, ui = _make_ctx(active_teams=["idea-brainstorm", "second"])
        await ExpertCommand().execute(ctx, args=["clear"])
        assert ctx.channel_runtime.active_teams == []
        assert any(
            "Dismissed experts: idea-brainstorm, second" in text for text, _ in ui.lines
        )

    async def test_clear_on_empty_list_reports_nothing_to_do(self):
        ctx, ui = _make_ctx()
        await ExpertCommand().execute(ctx, args=["clear"])
        assert ctx.channel_runtime.active_teams == []
        assert any("No experts invited" in text for text, _ in ui.lines)

    async def test_no_channel_runtime_prints_warning(self):
        ui = _FakeUI()
        ctx = CommandContext(agent=None, thread_id="t1", ui=ui, channel_runtime=None)
        await ExpertCommand().execute(ctx, args=["idea-brainstorm"])
        assert any("/expert requires a session runtime" in text for text, _ in ui.lines)


class TestExpertCompletions:
    """``ExpertCommand.get_completions`` mixes dynamic expert names with the
    static ``clear`` subcommand. Regression coverage for the three fixes on
    PR #371: exact-match suppression, past-first-arg guard, and
    case-insensitive matching.
    """

    def _patched_experts(self, *names: str):
        # Patch ``list_dispatchable_experts`` directly (not the underlying
        # ``list_expert_skills``) so the test does not depend on the shipped
        # yaml sub-agent set — the reserved-name filter would otherwise
        # silently reject a fake whose name collides with a future yaml
        # sub-agent.
        return patch(
            "EvoScientist.subagents.expert_container.list_dispatchable_experts",
            return_value=[_FakeSkillInfo(name=n) for n in names],
        )

    def test_lists_installed_experts_and_clear(self):
        cmd = ExpertCommand()
        with self._patched_experts("smoke-test-sync-expert", "smoke-test-alt-expert"):
            completions = cmd.get_completions([""])
        names = {name for name, _ in completions}
        assert names == {"smoke-test-sync-expert", "smoke-test-alt-expert", "clear"}

    def test_case_insensitive_prefix_match(self):
        # Skill dir names sometimes have uppercase; completion typed
        # lowercase must still surface them.
        cmd = ExpertCommand()
        with self._patched_experts("Smoke-Test-Case-Expert", "smoke-test-sync-expert"):
            completions = cmd.get_completions(["smoke-test-c"])
        names = {name for name, _ in completions}
        assert names == {"Smoke-Test-Case-Expert"}

    def test_exact_match_hides_popup_same_case(self):
        cmd = ExpertCommand()
        with self._patched_experts("smoke-test-sync-expert"):
            completions = cmd.get_completions(["smoke-test-sync-expert"])
        assert completions == []

    def test_exact_match_hides_popup_different_case(self):
        # Case-insensitive exact-match suppression: typing the name in a
        # different case than the skill dir still fully completes it and
        # hides the popup.
        cmd = ExpertCommand()
        with self._patched_experts("Smoke-Test-Case-Expert"):
            completions = cmd.get_completions(["smoke-test-case-expert"])
        assert completions == []

    def test_past_first_arg_returns_empty(self):
        # /expert takes a single positional. Trailing space -> tokens == ["n", ""].
        cmd = ExpertCommand()
        with self._patched_experts("smoke-test-sync-expert"):
            assert cmd.get_completions(["smoke-test-sync-expert", ""]) == []
            assert cmd.get_completions(["smoke-test-sync-expert", "foo"]) == []
