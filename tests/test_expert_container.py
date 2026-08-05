"""Tests for EvoScientist.subagents.expert_container factory."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from EvoScientist.subagents.expert_container import (
    _compose_system_prompt,
    build_expert_subagent_spec,
    build_expert_subagent_specs,
    expert_prompt_body,
    list_dispatchable_experts,
)
from EvoScientist.tools.skills_manager import SkillInfo

# =============================================================================
# Fixtures
# =============================================================================


def _write_expert_skill_file(
    parent: Path,
    name: str,
    *,
    body: str = "You are a test expert.\n\nDo the thing.\n",
    role: str = "test expert",
    description: str = "A test expert skill",
) -> Path:
    """Write a minimal expert SKILL.md file and return the parent directory."""
    skill_dir = parent / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"""---
name: {name}
description: {description}
type: expert
role: {role}
---
{body}"""
    )
    return skill_dir


def _skill_info(
    path: Path,
    *,
    name: str = "expert-a",
    description: str = "A test expert skill",
    role: str = "test expert",
) -> SkillInfo:
    return SkillInfo(
        name=name,
        description=description,
        path=path,
        source="workspace",
        type="expert",
        role=role,
    )


class _FakeTool:
    """Stand-in for a resolved tool callable — the factory only cares that
    the value is present in the registry, not what it is."""

    def __init__(self, name: str) -> None:
        self.name = name


# =============================================================================
# expert_prompt_body
# =============================================================================


class TestExpertPromptBody:
    def test_extracts_body_after_frontmatter(self, tmp_path):
        skill_dir = _write_expert_skill_file(tmp_path, "expert-a")
        info = _skill_info(skill_dir)
        body = expert_prompt_body(info)
        assert body.startswith("You are a test expert.")
        assert "Do the thing." in body
        assert "---" not in body
        assert "type: expert" not in body

    def test_returns_empty_on_missing_file(self, tmp_path, caplog):
        # A SkillInfo pointing at a nonexistent SKILL.md — factory should
        # gracefully degrade with a warning rather than raise.
        info = _skill_info(tmp_path / "nonexistent")
        body = expert_prompt_body(info)
        assert body == ""
        # Warning surfaced — SEV so the malformed skill isn't invisible.
        assert any("could not read SKILL.md" in r.message for r in caplog.records)

    def test_returns_empty_on_non_utf8_file(self, tmp_path, caplog):
        # A SKILL.md whose bytes aren't valid UTF-8. `read_text` raises
        # UnicodeDecodeError (not OSError); the factory must degrade to an
        # empty body rather than aborting agent construction.
        skill_dir = tmp_path / "bad-utf8"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_bytes(b"\xff\xfe garbage")
        info = _skill_info(skill_dir, name="bad-utf8")
        body = expert_prompt_body(info)
        assert body == ""
        assert any("could not read SKILL.md" in r.message for r in caplog.records)

    def test_handles_no_frontmatter(self, tmp_path):
        """A SKILL.md with no frontmatter — body is the whole file."""
        skill_dir = tmp_path / "no-fm"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# Body Only\n\nContent here.\n")
        info = _skill_info(skill_dir, name="no-fm")
        body = expert_prompt_body(info)
        assert "# Body Only" in body
        assert "Content here." in body

    def test_prefers_cached_body_over_disk_read(self, tmp_path):
        """When ``SkillInfo.body`` is populated (the ``_parse_skill_md`` path),
        ``expert_prompt_body`` uses it directly without touching disk. Guards against
        the double-read regression flagged by pre-PR review."""
        info = SkillInfo(
            name="cached",
            description="d",
            path=tmp_path / "does-not-exist",
            source="workspace",
            type="expert",
            body="Cached body content from SkillInfo.",
        )
        body = expert_prompt_body(info)
        assert body == "Cached body content from SkillInfo."

    def test_agents_md_expert_reads_actor_definition_not_skill_md(self, tmp_path):
        """An AGENTS.md expert is prompted from its actor definition.

        SKILL.md stays pure knowledge under the current contract — it is
        reachable in-turn via ``load_skill`` — so leaking it into the system
        prompt would both bloat the prompt and hand the expert a document
        written for a different reader.
        """
        info = SkillInfo(
            name="paper-review",
            description="d",
            path=tmp_path / "paper-review",
            source="builtin",
            type="expert",
            expert_source="agents_md",
            body="# Knowledge\n\nThe 5-aspect checklist.\n",
            agents_body="## Persona\n\nYou are an adversarial reviewer.\n",
        )
        body = expert_prompt_body(info)
        assert body == "## Persona\n\nYou are an adversarial reviewer.\n"
        assert "5-aspect checklist" not in body

    def test_agents_md_expert_falls_back_to_disk(self, tmp_path):
        """A hand-built SkillInfo without ``agents_body`` still resolves.

        Mirrors the SKILL.md fallback below it — external callers construct
        SkillInfo objects without going through ``_parse_skill_md``.
        """
        skill_dir = tmp_path / "on-disk"
        skill_dir.mkdir()
        (skill_dir / "AGENTS.md").write_text("## Persona\n\nFrom disk.\n")
        info = SkillInfo(
            name="on-disk",
            description="d",
            path=skill_dir,
            source="workspace",
            type="expert",
            expert_source="agents_md",
            body="SKILL.md body that must not be used.",
        )
        assert expert_prompt_body(info) == "## Persona\n\nFrom disk.\n"

    def test_agents_md_expert_returns_empty_when_file_missing(self, tmp_path):
        """Declared expert, no resolvable actor definition -> empty.

        Empty is the signal every registration path checks; falling back to
        the SKILL.md body here would register a knowledge document as a
        persona instead of refusing.
        """
        info = SkillInfo(
            name="gone",
            description="d",
            path=tmp_path / "gone",
            source="workspace",
            type="expert",
            expert_source="agents_md",
            body="SKILL.md body that must not be used.",
        )
        assert expert_prompt_body(info) == ""


# =============================================================================
# _compose_system_prompt
# =============================================================================


class TestComposeSystemPrompt:
    def test_prepends_role_line_when_present(self):
        info = SkillInfo(
            name="expert-a",
            description="d",
            path=Path("/tmp"),
            source="workspace",
            type="expert",
            role="research idea brainstormer",
        )
        prompt = _compose_system_prompt(info, "Follow these rules.\n")
        assert prompt.startswith("You are research idea brainstormer.\n")
        assert "Follow these rules." in prompt

    def test_omits_role_line_when_absent(self):
        info = SkillInfo(
            name="expert-a",
            description="d",
            path=Path("/tmp"),
            source="workspace",
            type="expert",
            role="",
        )
        prompt = _compose_system_prompt(info, "Do the thing.\n")
        assert not prompt.startswith("You are")
        assert prompt.rstrip() == "Do the thing."


# =============================================================================
# build_expert_subagent_spec
# =============================================================================


class TestBuildExpertSubagentSpec:
    def test_produces_expected_shape(self, tmp_path):
        skill_dir = _write_expert_skill_file(
            tmp_path,
            "expert-a",
            body="Second-person persona instructions.\n",
            role="research idea brainstormer",
            description="Brainstorms research ideas",
        )
        info = _skill_info(
            skill_dir,
            name="expert-a",
            description="Brainstorms research ideas",
            role="research idea brainstormer",
        )
        registry = {
            "think_tool": _FakeTool("think_tool"),
            "skill_manager": _FakeTool("skill_manager"),
        }
        spec = build_expert_subagent_spec(info, tool_registry=registry)

        # Same field set as `load_subagents._build_one` returns for a YAML subagent.
        assert set(spec.keys()) == {
            "name",
            "description",
            "system_prompt",
            "tools",
            "skills",
            "_async",
        }
        assert spec["name"] == "expert-a"
        assert spec["description"] == "Brainstorms research ideas"
        assert spec["_async"] is False
        assert spec["skills"] == ["/skills/"]
        assert spec["tools"] == [
            registry["think_tool"],
            registry["skill_manager"],
        ]
        # Role prepended, body preserved.
        assert spec["system_prompt"].startswith("You are research idea brainstormer.\n")
        assert "Second-person persona instructions." in spec["system_prompt"]

    def test_missing_tool_in_registry_is_skipped_not_raised(self, tmp_path, caplog):
        skill_dir = _write_expert_skill_file(tmp_path, "expert-a")
        info = _skill_info(skill_dir)
        # Registry has no `think_tool`. Factory logs a warning and returns
        # an empty tools list rather than raising.
        spec = build_expert_subagent_spec(info, tool_registry={})
        assert spec["tools"] == []
        assert any(
            "default tool 'think_tool' not in registry" in r.message
            for r in caplog.records
        )

    def test_tool_registry_optional(self, tmp_path):
        """Passing no registry is legal (used by tests / adhoc introspection)."""
        skill_dir = _write_expert_skill_file(tmp_path, "expert-a")
        info = _skill_info(skill_dir)
        spec = build_expert_subagent_spec(info)
        # No registry → tools empty; other fields still populated.
        assert spec["tools"] == []
        assert spec["name"] == "expert-a"
        assert spec["system_prompt"]


# =============================================================================
# build_expert_subagent_specs (bulk over list_expert_skills)
# =============================================================================


class TestBuildExpertSubagentSpecs:
    def test_returns_one_spec_per_installed_expert_skill(self, tmp_path):
        # Two expert skills + one utility skill.
        _write_expert_skill_file(tmp_path, "expert-a")
        _write_expert_skill_file(tmp_path, "expert-b")
        util = tmp_path / "util-c"
        util.mkdir()
        (util / "SKILL.md").write_text(
            """---
name: util-c
description: Not an expert
---

# Body
"""
        )

        registry = {
            "think_tool": _FakeTool("think_tool"),
            "skill_manager": _FakeTool("skill_manager"),
        }
        # Patch USER_SKILLS_DIR to point at our temp dir; patch GLOBAL and
        # SKILLS_DIR to empty locations so `list_expert_skills(include_system=True)`
        # only surfaces our two experts.
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with (
            patch("EvoScientist.paths.USER_SKILLS_DIR", tmp_path),
            patch("EvoScientist.paths.GLOBAL_SKILLS_DIR", empty_dir),
            patch("EvoScientist.EvoScientist.SKILLS_DIR", str(empty_dir)),
        ):
            specs = build_expert_subagent_specs(tool_registry=registry)

        names = sorted(s["name"] for s in specs)
        assert names == ["expert-a", "expert-b"]
        for s in specs:
            assert s["_async"] is False
            assert s["skills"] == ["/skills/"]
            assert s["tools"] == [
                registry["think_tool"],
                registry["skill_manager"],
            ]

    def test_skips_expert_with_empty_body(self, tmp_path, caplog):
        # A well-formed expert-frontmatter skill whose body is only whitespace.
        # Registering it would advertise a personaless expert in the `task`
        # schema — cleaner to drop it and log.
        _write_expert_skill_file(tmp_path, "expert-a")
        blank = tmp_path / "expert-blank"
        blank.mkdir()
        (blank / "SKILL.md").write_text(
            """---
name: expert-blank
description: An expert with no body
type: expert
role: blank
---

"""
        )
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with (
            patch("EvoScientist.paths.USER_SKILLS_DIR", tmp_path),
            patch("EvoScientist.paths.GLOBAL_SKILLS_DIR", empty_dir),
            patch("EvoScientist.EvoScientist.SKILLS_DIR", str(empty_dir)),
        ):
            specs = build_expert_subagent_specs(tool_registry={})

        assert [s["name"] for s in specs] == ["expert-a"]
        assert any(
            "SKILL.md body is empty" in r.message and "expert-blank" in r.message
            for r in caplog.records
        )

    def test_agents_md_expert_included_in_sync_registry(self, tmp_path):
        """Both contracts land in the in-turn registry, and its prompt is AGENTS.md.

        Every expert gets both reaches, so an AGENTS.md skill appears here
        as well as in the async specs. Sharing the name across the two is
        safe: they land on different tools with separate schemas.
        """
        _write_expert_skill_file(tmp_path, "legacy-expert")
        actor = tmp_path / "new-expert"
        actor.mkdir()
        (actor / "SKILL.md").write_text(
            """---
name: new-expert
description: Knowledge only
---

# Knowledge

The workflow.
"""
        )
        (actor / "AGENTS.md").write_text("## Persona\n\nYou are the new expert.\n")

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with (
            patch("EvoScientist.paths.USER_SKILLS_DIR", tmp_path),
            patch("EvoScientist.paths.GLOBAL_SKILLS_DIR", empty_dir),
            patch("EvoScientist.EvoScientist.SKILLS_DIR", str(empty_dir)),
        ):
            specs = build_expert_subagent_specs(tool_registry={})

        by_name = {s["name"]: s for s in specs}
        assert set(by_name) == {"legacy-expert", "new-expert"}
        # Prompted from the actor definition, not the knowledge file.
        assert "You are the new expert." in by_name["new-expert"]["system_prompt"]
        assert "The workflow." not in by_name["new-expert"]["system_prompt"]

    def test_legacy_async_dispatch_still_in_sync_registry(self, tmp_path):
        """A legacy ``default_dispatch: async`` skill still gets an in-turn spec.

        ``default_dispatch`` is no longer read: every expert is reachable both
        ways and the orchestrator picks per task, so an async-declared legacy
        skill must not be dropped from the sync registry. This is the
        behavioural counterpart to the parser's not-read contract.
        """
        actor = tmp_path / "async-legacy"
        actor.mkdir()
        (actor / "SKILL.md").write_text(
            """---
name: async-legacy
description: Legacy expert that declared async dispatch
type: expert
role: legacy async expert
default_dispatch: async
---

You are the legacy async expert.
"""
        )
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with (
            patch("EvoScientist.paths.USER_SKILLS_DIR", tmp_path),
            patch("EvoScientist.paths.GLOBAL_SKILLS_DIR", empty_dir),
            patch("EvoScientist.EvoScientist.SKILLS_DIR", str(empty_dir)),
        ):
            specs = build_expert_subagent_specs(tool_registry={})
        assert "async-legacy" in {s["name"] for s in specs}

    def test_returns_empty_when_no_expert_skills(self, tmp_path):
        # A utility skill only — no experts.
        util = tmp_path / "util-only"
        util.mkdir()
        (util / "SKILL.md").write_text(
            """---
name: util-only
description: Utility
---

# Body
"""
        )
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with (
            patch("EvoScientist.paths.USER_SKILLS_DIR", tmp_path),
            patch("EvoScientist.paths.GLOBAL_SKILLS_DIR", empty_dir),
            patch("EvoScientist.EvoScientist.SKILLS_DIR", str(empty_dir)),
        ):
            specs = build_expert_subagent_specs(tool_registry={})
        assert specs == []


# =============================================================================
# _fold_expert_subagents (name-collision guard shared by both construction paths)
# =============================================================================


def _spec(name: str) -> dict:
    """Minimal expert spec — the fold helper only reads ``name``."""
    return {"name": name, "description": f"{name} expert"}


class TestFoldExpertSubagents:
    """Both ``_build_base_kwargs`` and ``load_mcp_and_build_kwargs`` delegate
    to ``_fold_expert_subagents``, so testing the helper directly covers the
    "same behaviour in both paths" reviewer requirement."""

    def test_appends_expert_specs_when_no_collisions(self):
        from EvoScientist.EvoScientist import _fold_expert_subagents

        subs: list[dict] = [{"name": "research"}, {"name": "code"}]
        with patch(
            "EvoScientist.subagents.expert_container.build_expert_subagent_specs",
            return_value=[_spec("idea-brainstorm"), _spec("critic")],
        ):
            _fold_expert_subagents(subs, tool_registry={})

        assert [s["name"] for s in subs] == [
            "research",
            "code",
            "idea-brainstorm",
            "critic",
        ]

    def test_skips_expert_that_collides_with_yaml_subagent(self, caplog):
        from EvoScientist.EvoScientist import _fold_expert_subagents

        subs: list[dict] = [{"name": "research"}, {"name": "planner"}]
        with patch(
            "EvoScientist.subagents.expert_container.build_expert_subagent_specs",
            return_value=[_spec("planner"), _spec("idea-brainstorm")],
        ):
            _fold_expert_subagents(subs, tool_registry={})

        # Colliding expert dropped; non-colliding one appended.
        assert [s["name"] for s in subs] == [
            "research",
            "planner",
            "idea-brainstorm",
        ]
        # Original YAML `planner` untouched (not shadowed by the expert).
        assert subs[1] == {"name": "planner"}
        assert any(
            "collides with an existing sub-agent name" in r.message
            and "planner" in r.message
            for r in caplog.records
        )

    def test_skips_duplicate_expert_names(self, caplog):
        from EvoScientist.EvoScientist import _fold_expert_subagents

        subs: list[dict] = []
        with patch(
            "EvoScientist.subagents.expert_container.build_expert_subagent_specs",
            return_value=[_spec("critic"), _spec("critic")],
        ):
            _fold_expert_subagents(subs, tool_registry={})

        assert [s["name"] for s in subs] == ["critic"]
        assert any(
            "collides with an existing sub-agent name" in r.message
            and "critic" in r.message
            for r in caplog.records
        )

    def test_reserves_general_purpose_name(self, caplog):
        """The default subagent slot is reserved even when no ``general-purpose``
        entry exists in ``subs`` yet — ``_ensure_general_purpose_subagent``
        runs right after the fold and would otherwise treat the expert entry
        as the default subagent, silently losing the DeepAgents default prompt."""
        from EvoScientist.EvoScientist import _fold_expert_subagents

        subs: list[dict] = [{"name": "research"}]
        with patch(
            "EvoScientist.subagents.expert_container.build_expert_subagent_specs",
            return_value=[_spec("general-purpose")],
        ):
            _fold_expert_subagents(subs, tool_registry={})

        assert [s["name"] for s in subs] == ["research"]
        assert any(
            "collides with an existing sub-agent name" in r.message
            and "general-purpose" in r.message
            for r in caplog.records
        )

    def test_forwards_tool_registry_to_specs_factory(self):
        from EvoScientist.EvoScientist import _fold_expert_subagents

        registry = {"think_tool": object()}
        with patch(
            "EvoScientist.subagents.expert_container.build_expert_subagent_specs",
            return_value=[],
        ) as mock_specs:
            _fold_expert_subagents([], tool_registry=registry)

        mock_specs.assert_called_once_with(tool_registry=registry)


# =============================================================================
# list_dispatchable_experts honest surface
# =============================================================================


class TestListDispatchableExpertsSurvivesAsyncOutage:
    """``list_dispatchable_experts`` never drops an expert for async reasons.

    Under the old per-skill classification an async-declared expert vanished
    from every surface whenever ``enable_async_subagents`` was off or
    langgraph dev was unreachable — installed, listed in the gallery, and
    reachable by nothing. Every expert now keeps its in-turn reach, so an
    async outage degrades the reach rather than removing the expert.
    """

    def _skill(self, name: str) -> SkillInfo:
        return SkillInfo(
            name=name,
            description=f"{name} description",
            path=Path("/tmp/nope"),
            source="builtin",
            type="expert",
            role=f"{name} role",
            body="persona body\n",
        )

    def test_experts_survive_async_flag_disabled(self):
        cfg = SimpleNamespace(enable_async_subagents=False)
        skills = [self._skill("idea-brainstorm"), self._skill("lit")]
        with patch(
            "EvoScientist.tools.skills_manager.list_expert_skills",
            return_value=skills,
        ):
            result = list_dispatchable_experts(cfg=cfg)
        assert {s.name for s in result} == {"idea-brainstorm", "lit"}

    def test_experts_survive_dev_unreachable(self):
        cfg = SimpleNamespace(enable_async_subagents=True)
        skills = [self._skill("idea-brainstorm"), self._skill("lit")]
        with (
            patch(
                "EvoScientist.tools.skills_manager.list_expert_skills",
                return_value=skills,
            ),
            patch(
                "EvoScientist.langgraph_dev.manager.is_async_subagents_available",
                return_value=False,
            ),
        ):
            result = list_dispatchable_experts(cfg=cfg)
        assert {s.name for s in result} == {"idea-brainstorm", "lit"}

    def test_empty_actor_definition_still_dropped(self):
        """The filters that remain are about broken experts, not reach."""
        cfg = SimpleNamespace(enable_async_subagents=True)
        blank = self._skill("blank")
        blank.body = "   \n"
        with patch(
            "EvoScientist.tools.skills_manager.list_expert_skills",
            return_value=[self._skill("idea-brainstorm"), blank],
        ):
            result = list_dispatchable_experts(cfg=cfg)
        assert [s.name for s in result] == ["idea-brainstorm"]
