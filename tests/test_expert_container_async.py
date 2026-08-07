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

from langchain_core.messages import SystemMessage

from EvoScientist.subagents.expert_container_async import (
    _PERSONA_SENTINEL,
    ExpertContainerState,
    ExpertSkillLoaderMiddleware,
)
from EvoScientist.tools.skills_manager import SkillInfo

# Two base-stack witness blocks used across the wrap_model_call tests. Their
# contents mirror the section headers deepagents emits per-turn — regressing
# the compose logic would drop these from the composed system_message.
_TASK_WITNESS = "## `task` (subagent spawner)\n\nUse ``task`` to delegate ..."
_SKILLS_WITNESS = "## Skills System\n\nInstalled skills are mounted under ..."

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

    def test_agents_md_expert_prompted_from_actor_definition(self):
        """AGENTS.md experts are prompted from AGENTS.md, not SKILL.md.

        Both files exist for these skills, so composing from ``.body`` would
        silently work — and hand the expert a knowledge document written for
        a different reader in place of its persona.
        """
        mw = ExpertSkillLoaderMiddleware()
        info = _skill_info(
            name="paper-review",
            role="",
            body="# Knowledge\n\nThe 5-aspect checklist.\n",
        )
        info.expert_source = "agents_md"
        info.agents_body = "## Persona\n\nYou are an adversarial reviewer.\n"
        with patch(
            "EvoScientist.tools.skills_manager.list_expert_skills",
            return_value=[info],
        ):
            composed = mw._compose_prompt({"skill_name": "paper-review"})
        assert "You are an adversarial reviewer." in composed
        assert "5-aspect checklist" not in composed

    def test_empty_actor_definition_names_agents_md_in_error(self):
        """The error cue names the file the author has to fix.

        An AGENTS.md expert with a healthy SKILL.md would otherwise be told
        its SKILL.md body is empty, sending the author to the wrong file.
        """
        mw = ExpertSkillLoaderMiddleware()
        info = _skill_info(
            name="paper-review",
            role="",
            body="# Knowledge\n\nPlenty of content here.\n",
        )
        info.expert_source = "agents_md"
        info.agents_body = "  \n"
        with patch(
            "EvoScientist.tools.skills_manager.list_expert_skills",
            return_value=[info],
        ):
            composed = mw._compose_prompt({"skill_name": "paper-review"})
        assert composed.startswith("ERROR:")
        assert "empty AGENTS.md body" in composed
        assert "paper-review" in composed

    def test_runtime_context_tail_surfaces_skill_name(self):
        """The tail block re-asserts ``skill_name`` on every model call so the
        expert knows its own persona name after summarization. Since
        ``output_path`` moved to the task description (payload dropped in
        PR #391 review X-4), the tail carries no path — the LLM pins it into
        its own todo list per SKILL.md contract."""
        mw = ExpertSkillLoaderMiddleware()
        with patch(
            "EvoScientist.tools.skills_manager.list_expert_skills",
            return_value=[_skill_info()],
        ):
            composed = mw._compose_prompt({"skill_name": "literature-review"})
        assert "## Runtime context" in composed
        assert "``skill_name``: ``literature-review``" in composed
        # Path retention is no longer a middleware responsibility.
        assert "``output_path``" not in composed
        assert "verbatim" not in composed


# =============================================================================
# ExpertContainerState — state schema smoke check
# =============================================================================


class TestExpertContainerState:
    """The state schema carries ``skill_name`` only. ``output_path`` was
    dropped in PR #391 review X-4 — the main agent now embeds the desired
    path in the task description (natural language) and the expert's
    SKILL.md contract pins it via ``write_todos`` on turn 1."""

    def test_state_shape(self):
        # TypedDicts don't runtime-validate — assert the field is declared
        # so downstream code can rely on ``state.get("skill_name")``.
        annotations = ExpertContainerState.__annotations__
        assert "skill_name" in annotations
        assert "output_path" not in annotations


# =============================================================================
# wrap_model_call — override via ModelRequest.override
# =============================================================================


def _system_message_with_sentinel_and_witnesses() -> SystemMessage:
    """Base-stack-shaped ``SystemMessage``: the fallback (sentinel-bearing)
    block, then two witness blocks that represent deepagents' composed
    sections. This is the exact shape our middleware sees at model-call
    time when the container graph was built with
    ``system_prompt=_FALLBACK_SYSTEM_PROMPT`` and the base stack has
    appended its sections on top."""
    from EvoScientist.subagents.expert_container_async import _FALLBACK_SYSTEM_PROMPT

    return SystemMessage(
        content=[
            {"type": "text", "text": _FALLBACK_SYSTEM_PROMPT},
            {"type": "text", "text": _TASK_WITNESS},
            {"type": "text", "text": _SKILLS_WITNESS},
        ]
    )


def _mock_request(system_message: SystemMessage):
    """Stub ``ModelRequest`` supporting ``state`` and ``override``. Returns
    ``(request, seen, handler)`` — ``seen`` is a list the handler pushes the
    post-override ``system_message`` into for post-call assertions."""
    seen: list[SystemMessage] = []
    overridden = SimpleNamespace()

    def override(*, system_message):
        overridden.system_message = system_message
        return overridden

    def handler(new_request):
        seen.append(new_request.system_message)
        return SimpleNamespace()

    request = SimpleNamespace(
        state={"skill_name": "literature-review"},
        system_message=system_message,
        override=override,
    )
    return request, seen, handler


class TestWrapModelCall:
    def test_wrap_composes_persona_into_base_stack_system_message(self):
        """Persona swaps for the sentinel block; base-stack witness blocks
        stay in place. The whole point of the fix — replacing the whole
        system_message (the pre-fix behaviour) dropped every base-stack
        section (measured live: 9,608 → 382 chars) and broke ``task()``
        for async experts."""
        mw = ExpertSkillLoaderMiddleware()
        request, seen, handler = _mock_request(
            _system_message_with_sentinel_and_witnesses()
        )
        with patch(
            "EvoScientist.tools.skills_manager.list_expert_skills",
            return_value=[_skill_info()],
        ):
            mw.wrap_model_call(request, handler)

        composed = seen[0]
        block_texts = [b.get("text", "") for b in composed.content_blocks]
        # Persona landed — role prepend visible.
        assert any(
            t.startswith("You are literature-review strategist.") for t in block_texts
        )
        # Sentinel gone (block was replaced, not appended).
        assert not any(_PERSONA_SENTINEL in t for t in block_texts)
        # Witnesses preserved verbatim — the base-stack sections stay live.
        assert _TASK_WITNESS in block_texts
        assert _SKILLS_WITNESS in block_texts
        # Block count unchanged — replace, not append.
        assert len(block_texts) == 3

    def test_wrap_appends_persona_when_sentinel_missing(self, caplog):
        """When the sentinel block isn't found (e.g. deepagents refactors
        how ``system_prompt=`` reaches ``content_blocks``), the persona is
        appended instead of silently dropped, and the drift is logged."""
        import logging

        mw = ExpertSkillLoaderMiddleware()
        # No sentinel block — only witnesses.
        request, seen, handler = _mock_request(
            SystemMessage(
                content=[
                    {"type": "text", "text": _TASK_WITNESS},
                    {"type": "text", "text": _SKILLS_WITNESS},
                ]
            )
        )
        with (
            patch(
                "EvoScientist.tools.skills_manager.list_expert_skills",
                return_value=[_skill_info()],
            ),
            caplog.at_level(
                logging.WARNING,
                logger="EvoScientist.subagents.expert_container_async",
            ),
        ):
            mw.wrap_model_call(request, handler)

        composed = seen[0]
        block_texts = [b.get("text", "") for b in composed.content_blocks]
        # Persona appended as a new block.
        assert any(
            t.startswith("You are literature-review strategist.") for t in block_texts
        )
        # Witnesses still present.
        assert _TASK_WITNESS in block_texts
        assert _SKILLS_WITNESS in block_texts
        # Original two blocks + persona = 3.
        assert len(block_texts) == 3
        # Drift-detected warning surfaced.
        assert any(
            _PERSONA_SENTINEL in r.message and "not found" in r.message
            for r in caplog.records
        )
