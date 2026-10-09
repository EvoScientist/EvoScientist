"""Tests for skill_slash_names."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from EvoScientist.commands.skill_slash import skill_slash_names
from EvoScientist.tools import skills_manager


@pytest.fixture(autouse=True)
def _isolated_global_skills(tmp_path, monkeypatch):
    """Keep the developer's real global skills out of the index."""
    global_dir = tmp_path / "global_skills"
    global_dir.mkdir()
    monkeypatch.setattr("EvoScientist.paths.GLOBAL_SKILLS_DIR", global_dir)
    skills_manager._skill_index_cache.clear()
    yield
    skills_manager._skill_index_cache.clear()


def _make_skill(workspace, name):
    skill_dir = workspace.skills_dir / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: A skill.\n---\n\nBody.\n",
        encoding="utf-8",
    )


def test_installed_skill(workspace):
    _make_skill(workspace, "paper-writing")
    assert skill_slash_names("/paper-writing intro", workspace) == ["paper-writing"]


def test_typo_stays_unknown(workspace):
    _make_skill(workspace, "paper-writing")
    assert skill_slash_names("/paper-writng intro", workspace) == []


def test_builtin_command_wins_over_a_same_named_skill(workspace):
    _make_skill(workspace, "model")
    assert skill_slash_names("/model", workspace) == []


def test_non_slash_and_no_workspace(workspace):
    _make_skill(workspace, "paper-writing")
    assert skill_slash_names("hello", workspace) == []
    assert skill_slash_names("/paper-writing intro", None) == []


def test_leading_whitespace_is_ignored(workspace):
    _make_skill(workspace, "paper-writing")
    assert skill_slash_names("  /paper-writing intro", workspace) == ["paper-writing"]


def test_registered_alias_wins_over_a_same_named_skill(workspace):
    _make_skill(workspace, "q")
    assert skill_slash_names("/q", workspace) == []


def test_lone_slash_is_not_a_skill(workspace):
    assert skill_slash_names("/", workspace) == []


def test_skill_made_after_an_earlier_call_is_found(workspace):
    assert skill_slash_names("/late-skill go", workspace) == []
    _make_skill(workspace, "late-skill")
    assert skill_slash_names("/late-skill go", workspace) == ["late-skill"]


def test_unreadable_skills_folder_falls_back_to_unknown_command(workspace):
    _make_skill(workspace, "alpha")
    with patch("pathlib.Path.iterdir", side_effect=PermissionError("denied")):
        assert skill_slash_names("/alpha go", workspace) == []


def test_skill_is_pinned_by_its_frontmatter_name(workspace):
    skill_dir = workspace.skills_dir / "folder-name"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: frontmatter-name\ndescription: A skill.\n---\n",
        encoding="utf-8",
    )
    assert skill_slash_names("/frontmatter-name go", workspace) == ["frontmatter-name"]
    assert skill_slash_names("/folder-name go", workspace) == []
