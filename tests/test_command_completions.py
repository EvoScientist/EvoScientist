"""Tests for multi-stage command completions, categories, and dynamic completions."""

from typing import ClassVar
from unittest.mock import patch

import pytest

from EvoScientist.commands._completion_engine import (
    SKILL_CATEGORY,
    compute_completions,
)


class TestTopLevelCompletions:
    def test_slash_returns_all_commands(self):
        r = compute_completions("/", 1)
        names = [c.text for c in r.candidates]
        assert "/mcp" in names
        assert "/help" in names
        assert "/new" in names

    def test_prefix_filters(self):
        r = compute_completions("/mc", 3)
        names = [c.text for c in r.candidates]
        assert "/mcp" in names
        assert "/help" not in names

    def test_exact_leaf_hides(self):
        r = compute_completions("/new", 4)
        assert r.kind == "empty"

    def test_exact_with_subcommands_hides_without_space(self):
        r = compute_completions("/mcp", 4)
        assert r.kind == "empty"

    def test_results_have_categories(self):
        r = compute_completions("/", 1)
        cats = {c.category for c in r.candidates}
        assert "Session" in cats
        assert "MCP" in cats
        assert "General" in cats

    def test_category_ordering(self):
        r = compute_completions("/", 1)
        cats = []
        for c in r.candidates:
            if not cats or cats[-1] != c.category:
                cats.append(c.category)
        assert cats.index("Session") < cats.index("General")


class TestAliasVisibility:
    def test_alias_prefix_matches(self):
        r = compute_completions("/fa", 3)
        names = [c.text for c in r.candidates]
        assert "/model-fallback" in names

    def test_alias_exact_hides_leaf(self):
        r = compute_completions("/quit", 5)
        assert r.kind == "empty"

    def test_alias_exact_hides_without_space(self):
        r = compute_completions("/fallback", 9)
        assert r.kind == "empty"

    def test_alias_space_shows_subcommands(self):
        r = compute_completions("/fallback ", 10)
        names = [c.text for c in r.candidates]
        assert "add" in names
        assert "list" in names


class TestSubcommandCompletions:
    def test_space_shows_subcommands(self):
        r = compute_completions("/mcp ", 5)
        names = [c.text for c in r.candidates]
        assert "list" in names
        assert "config" in names
        assert "install" in names

    def test_prefix_filters_subcommands(self):
        r = compute_completions("/mcp c", 6)
        names = [c.text for c in r.candidates]
        assert "config" in names
        assert "list" not in names

    def test_leaf_subcommand_stops(self):
        r = compute_completions("/model-fallback help ", 21)
        assert r.kind == "empty"

    def test_channel_all_types(self):
        r = compute_completions("/channel ", 9)
        names = [c.text for c in r.candidates]
        assert "status" in names
        assert "telegram" in names
        assert "discord" in names

    def test_model_fallback_subcommands(self):
        r = compute_completions("/model-fallback ", 16)
        names = [c.text for c in r.candidates]
        assert "add" in names
        assert "clear" in names


class TestDynamicCompletions:
    def _invalidate_mcp_cache(self):
        from EvoScientist.commands.implementation.mcp import MCPCommand
        from EvoScientist.commands.manager import manager

        cmd = manager.get_command("/mcp")
        if isinstance(cmd, MCPCommand):
            cmd._invalidate_server_cache()

    def test_mcp_config_server_names(self):
        self._invalidate_mcp_cache()
        fake_config = {"myserver": {}, "other": {}}
        with patch("EvoScientist.mcp.load_mcp_config", return_value=fake_config):
            r = compute_completions("/mcp config ", 12)
        names = [c.text for c in r.candidates]
        assert "myserver" in names
        assert "other" in names

    def test_mcp_config_prefix_filters(self):
        self._invalidate_mcp_cache()
        fake_config = {"myserver": {}, "other": {}}
        with patch("EvoScientist.mcp.load_mcp_config", return_value=fake_config):
            r = compute_completions("/mcp config my", 14)
        names = [c.text for c in r.candidates]
        assert names == ["myserver"]

    def test_mcp_remove_shows_servers(self):
        self._invalidate_mcp_cache()
        fake_config = {"srv1": {}}
        with patch("EvoScientist.mcp.load_mcp_config", return_value=fake_config):
            r = compute_completions("/mcp remove ", 12)
        assert [c.text for c in r.candidates] == ["srv1"]


class TestSkillCompletions:
    _INDEX: ClassVar[dict[str, str]] = {
        "paper-writing": "Write a full paper. " + "x" * 200,
        "paper-review": "Review a paper.",
        "new": "A skill that clashes with /new.",
    }

    def _complete(self, text, workspace):
        with patch(
            "EvoScientist.tools.skills_manager.installed_skill_index",
            return_value=self._INDEX,
        ):
            return compute_completions(text, len(text), workspace=workspace)

    def test_prefix_lists_matching_skills(self, workspace):
        r = self._complete("/paper", workspace)
        skills = [c for c in r.candidates if c.category == SKILL_CATEGORY]
        assert [c.text for c in skills] == ["/paper-review", "/paper-writing"]
        assert all(c.description.startswith("(skill) ") for c in skills)
        assert all(len(c.description) <= 80 for c in skills)

    def test_skills_come_after_commands(self, workspace):
        r = self._complete("/", workspace)
        cats = [c.category for c in r.candidates]
        assert SKILL_CATEGORY in cats
        assert cats.index(SKILL_CATEGORY) > max(
            i for i, c in enumerate(cats) if c != SKILL_CATEGORY
        )

    def test_command_name_clash_is_not_offered_as_a_skill(self, workspace):
        r = self._complete("/ne", workspace)
        assert [c.text for c in r.candidates].count("/new") == 1
        assert not any(c.category == SKILL_CATEGORY for c in r.candidates)

    def test_typing_the_message_after_a_skill_stops_completion(self, workspace):
        assert self._complete("/paper-writing ", workspace).kind == "empty"

    def test_no_workspace_no_skills(self):
        r = compute_completions("/paper", 6)
        assert not any(c.category == SKILL_CATEGORY for c in r.candidates)

    def test_names_the_pin_parser_rejects_are_not_offered(self, workspace):
        index = {
            "My_Skill": "Mixed case with an underscore.",
            "foo_bar": "Underscore.",
            "paper-writing": "Write a paper.",
        }
        with patch(
            "EvoScientist.tools.skills_manager.installed_skill_index",
            return_value=index,
        ):
            r = compute_completions("/", 1, workspace=workspace)
        skills = [c.text for c in r.candidates if c.category == SKILL_CATEGORY]
        assert "/paper-writing" in skills
        assert "/My_Skill" not in skills
        assert "/foo_bar" not in skills

    def test_unreadable_skills_directory_leaves_the_commands(self, workspace):
        with patch(
            "EvoScientist.tools.skills_manager.installed_skill_index",
            side_effect=PermissionError("denied"),
        ):
            r = compute_completions("/mo", 3, workspace=workspace)
            nothing = compute_completions("/pa", 3, workspace=workspace)
        assert "/model" in [c.text for c in r.candidates]
        assert not any(c.category == SKILL_CATEGORY for c in r.candidates)
        assert nothing.kind == "empty"

    _SKILL_INDEX: ClassVar[dict[str, str]] = {
        "skill": "A skill.",
        "skill-two": "Another skill.",
    }

    def _complete_skill_prefix(self, text, workspace):
        with patch(
            "EvoScientist.tools.skills_manager.installed_skill_index",
            return_value=self._SKILL_INDEX,
        ):
            return compute_completions(text, len(text), workspace=workspace)

    def test_exact_skill_name_leads_with_its_whole_group(self, workspace):
        r = self._complete_skill_prefix("/skill", workspace)
        texts = [c.text for c in r.candidates]
        assert texts[:2] == ["/skill", "/skill-two"]
        assert "/skills" in texts
        assert "/autoskills" in texts
        categories = [c.category for c in r.candidates]
        skill_count = categories.count(SKILL_CATEGORY)
        assert skill_count == 2
        assert categories[:skill_count] == [SKILL_CATEGORY] * skill_count

    def test_exact_skill_name_keeps_the_command_order(self, workspace):
        exact = self._complete_skill_prefix("/skill", workspace).candidates
        with patch(
            "EvoScientist.tools.skills_manager.installed_skill_index",
            return_value={"skill-two": "Another skill."},
        ):
            other = compute_completions("/skill", 6, workspace=workspace).candidates
        assert exact[2:] == other[:-1]

    def test_partial_skill_name_keeps_the_commands_first(self, workspace):
        r = self._complete_skill_prefix("/ski", workspace)
        texts = [c.text for c in r.candidates]
        categories = [c.category for c in r.candidates]
        assert "/skills" in texts
        assert "/autoskills" in texts
        assert categories[0] != SKILL_CATEGORY
        assert texts[-2:] == ["/skill", "/skill-two"]
        assert categories[-2:] == [SKILL_CATEGORY] * 2


class TestSkillCompletionsFromDisk:
    @pytest.fixture(autouse=True)
    def _isolated_global_skills(self, tmp_path, monkeypatch):
        from EvoScientist.tools import skills_manager

        global_dir = tmp_path / "global_skills"
        global_dir.mkdir()
        monkeypatch.setattr("EvoScientist.paths.GLOBAL_SKILLS_DIR", global_dir)
        skills_manager._skill_index_cache.clear()
        yield
        skills_manager._skill_index_cache.clear()

    def test_non_string_frontmatter_values_do_not_break_completion(self, workspace):
        skill_dir = workspace.skills_dir / "numbers"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: 2024\ndescription: 42\n---\n\nBody.\n", encoding="utf-8"
        )
        r = compute_completions("/", 1, workspace=workspace)
        skills = {
            c.text: c.description for c in r.candidates if c.category == SKILL_CATEGORY
        }
        assert skills["/2024"] == "(skill) 42"

    def test_frontmatter_yaml_cannot_raise_out_of_completion(self, workspace):
        skill_dir = workspace.skills_dir / "dated"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: dated\ndescription: Hi.\ncreated: 2024-13-45\n---\n\nBody.\n",
            encoding="utf-8",
        )
        r = compute_completions("/", 1, workspace=workspace)
        assert "/dated" not in [c.text for c in r.candidates]
