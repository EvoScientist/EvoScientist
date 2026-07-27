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

    def test_empty_body_returns_error_cue(self):
        """A skill with an empty SKILL.md body would otherwise run against a
        persona-less system prompt (just the role line). Mirror the sync
        fold-in's policy: refuse to compose a prompt at all and surface the
        skill-authoring bug through the LLM's error envelope."""
        mw = ExpertSkillLoaderMiddleware()
        with patch(
            "EvoScientist.tools.skills_manager.list_expert_skills",
            return_value=[_skill_info(body="")],
        ):
            composed = mw._compose_prompt({"skill_name": "literature-review"})
        assert composed.startswith("ERROR:")
        assert "empty SKILL.md body" in composed
        assert "literature-review" in composed  # names the offending skill

    def test_whitespace_only_body_returns_error_cue(self):
        """A body that's just whitespace (`   \\n\\n`) is still empty in the
        sense that matters — no persona, no pipeline. Same error cue."""
        mw = ExpertSkillLoaderMiddleware()
        with patch(
            "EvoScientist.tools.skills_manager.list_expert_skills",
            return_value=[_skill_info(body="   \n\n  \n")],
        ):
            composed = mw._compose_prompt({"skill_name": "literature-review"})
        assert composed.startswith("ERROR:")
        assert "empty SKILL.md body" in composed

    def test_runtime_context_tail_surfaces_output_path(self):
        """A populated ``output_path`` in state must appear in the composed
        prompt's Runtime-context tail block AND the "write verbatim" cue,
        so the LLM can honour the caller-provided path instead of inventing
        its own filename."""
        mw = ExpertSkillLoaderMiddleware()
        with patch(
            "EvoScientist.tools.skills_manager.list_expert_skills",
            return_value=[_skill_info()],
        ):
            composed = mw._compose_prompt(
                {
                    "skill_name": "literature-review",
                    "output_path": "./artifacts/literature-review/survey.md",
                }
            )
        # Tail block header and the exact path both appear.
        assert "## Runtime context" in composed
        assert (
            "``output_path``: ``./artifacts/literature-review/survey.md``" in composed
        )
        # The imperative "verbatim" cue only emits when output_path is populated.
        assert "verbatim" in composed

    def test_runtime_context_omits_output_path_line_when_absent(self):
        """When state has no ``output_path`` (skill's SKILL.md doesn't require
        one), the tail block emits without the output_path line AND without
        the "write verbatim" cue — a skill that computes its own default
        path shouldn't be told to honour a nonexistent state field."""
        mw = ExpertSkillLoaderMiddleware()
        with patch(
            "EvoScientist.tools.skills_manager.list_expert_skills",
            return_value=[_skill_info()],
        ):
            composed = mw._compose_prompt({"skill_name": "literature-review"})
        assert "## Runtime context" in composed
        assert "``skill_name``: ``literature-review``" in composed
        assert "``output_path``" not in composed
        assert "verbatim" not in composed

    def test_non_string_output_path_returns_error_cue(self):
        """A non-string ``output_path`` (e.g. int from a wire-format bug)
        surfaces as an error cue naming the wrong type, so the LLM halts
        with a well-formed envelope instead of coercing garbage into the
        filesystem."""
        mw = ExpertSkillLoaderMiddleware()
        with patch(
            "EvoScientist.tools.skills_manager.list_expert_skills",
            return_value=[_skill_info()],
        ):
            composed = mw._compose_prompt(
                {"skill_name": "literature-review", "output_path": 42}
            )
        assert composed.startswith("ERROR:")
        assert "int" in composed  # names the offending type
        assert "literature-review" in composed  # names the expert for triage


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
