"""Tests for the async expert container graph builder + loader middleware.

The full ``build_expert_container_async_graph()`` factory is exercised end-
to-end at langgraph dev startup; here we cover the load-bearing piece —
``ExpertSkillLoaderMiddleware._compose_prompt`` — in isolation so a
regression on skill resolution surfaces without needing a live langgraph
subprocess.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from EvoScientist.subagents.expert_container_async import (
    ExpertContainerState,
    ExpertSkillLoaderMiddleware,
)
from EvoScientist.tools.skills_manager import SkillInfo

# =============================================================================
# _compose_prompt — the load-bearing logic
# =============================================================================


def _skill_info(
    *,
    name: str = "literature-review",
    role: str = "literature-review strategist",
    body: str = "You produce manuscript-quality surveys.\n\nPipeline: ...\n",
    description: str = "d",
) -> SkillInfo:
    return SkillInfo(
        name=name,
        description=description,
        path=Path("/tmp/does-not-matter"),
        source="builtin",
        type="expert",
        role=role,
        body=body,
    )


class TestComposePrompt:
    def test_returns_role_and_body_for_known_skill(self):
        mw = ExpertSkillLoaderMiddleware()
        with patch(
            "EvoScientist.tools.skills_manager.list_expert_skills",
            return_value=[_skill_info()],
        ):
            composed = mw._compose_prompt({"skill_name": "literature-review"})
        # Role prepended, body preserved, trailing newline guaranteed.
        assert composed.startswith("You are literature-review strategist.")
        assert "You produce manuscript-quality surveys." in composed
        assert composed.endswith("\n")

    def test_omits_role_line_when_absent(self):
        mw = ExpertSkillLoaderMiddleware()
        info = _skill_info(role="", body="Second-person persona body.\n")
        with patch(
            "EvoScientist.tools.skills_manager.list_expert_skills",
            return_value=[info],
        ):
            composed = mw._compose_prompt({"skill_name": "literature-review"})
        assert not composed.startswith("You are ")
        assert "Second-person persona body." in composed

    def test_missing_skill_name_returns_error_cue(self):
        mw = ExpertSkillLoaderMiddleware()
        composed = mw._compose_prompt({})
        assert composed.startswith("ERROR:")
        assert "skill_name" in composed
        assert "wiring bug" in composed

    def test_unknown_skill_returns_error_cue_with_installed_list(self):
        mw = ExpertSkillLoaderMiddleware()
        installed = [_skill_info(name="literature-review"), _skill_info(name="other")]
        with patch(
            "EvoScientist.tools.skills_manager.list_expert_skills",
            return_value=installed,
        ):
            composed = mw._compose_prompt({"skill_name": "not-installed"})
        assert composed.startswith("ERROR:")
        assert "'not-installed' is not installed" in composed
        # Names of the installed experts are listed so the LLM's error
        # envelope can suggest the correct spelling.
        assert "literature-review" in composed
        assert "other" in composed

    def test_no_installed_experts_reports_none(self):
        mw = ExpertSkillLoaderMiddleware()
        with patch(
            "EvoScientist.tools.skills_manager.list_expert_skills", return_value=[]
        ):
            composed = mw._compose_prompt({"skill_name": "literature-review"})
        assert composed.startswith("ERROR:")
        assert "(none)" in composed

    def test_empty_body_still_returns_prompt(self):
        """Skill with an empty body: role line + newline, no crash."""
        mw = ExpertSkillLoaderMiddleware()
        with patch(
            "EvoScientist.tools.skills_manager.list_expert_skills",
            return_value=[_skill_info(body="")],
        ):
            composed = mw._compose_prompt({"skill_name": "literature-review"})
        assert composed.startswith("You are literature-review strategist.")
        assert composed.endswith("\n")


# =============================================================================
# ExpertContainerState — state schema smoke check
# =============================================================================


class TestExpertContainerState:
    """The state schema must accept ``skill_name`` and ``output_path`` as
    optional keys, matching what the ``payload`` from
    ``EvoAsyncSubAgentMiddleware.start_async_task`` puts into ``input=``."""

    def test_state_shape(self):
        # TypedDicts don't runtime-validate — we assert the fields are
        # declared so downstream code can rely on ``state.get("skill_name")``.
        annotations = ExpertContainerState.__annotations__
        assert "skill_name" in annotations
        assert "output_path" in annotations


# =============================================================================
# wrap_model_call — override via ModelRequest.override
# =============================================================================


class TestWrapModelCall:
    def test_wrap_replaces_system_message_with_composed_prompt(self):
        mw = ExpertSkillLoaderMiddleware()
        seen = {}

        def handler(new_request):
            seen["system_prompt"] = new_request.system_message.content
            return SimpleNamespace()

        # Build a minimal ModelRequest-like object with .state and
        # .override() returning a new request whose system_message is the
        # composed prompt. We stub ModelRequest via SimpleNamespace to
        # avoid pulling in the full langchain construction; wrap_model_call
        # only calls request.override(system_message=...) and passes the
        # result to handler.
        overridden = SimpleNamespace()

        def override(*, system_message):
            overridden.system_message = system_message
            return overridden

        request = SimpleNamespace(
            state={"skill_name": "literature-review"},
            override=override,
        )

        with patch(
            "EvoScientist.tools.skills_manager.list_expert_skills",
            return_value=[_skill_info()],
        ):
            mw.wrap_model_call(request, handler)

        assert overridden.system_message.content.startswith(
            "You are literature-review strategist."
        )
