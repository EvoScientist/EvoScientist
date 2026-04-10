"""Tests for EvoScientist.tools.skills_manager module."""

from pathlib import Path
from unittest.mock import patch

import pytest

from EvoScientist.tools.skills_manager import (
    _is_github_url,
    _parse_github_url,
    _parse_skill_md,
    _validate_skill_dir,
    fetch_remote_skill_index,
    get_all_tags,
    install_skill,
    list_skills,
    list_skills_by_tag,
    uninstall_skill,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_skills_dir(tmp_path):
    """Create a temporary skills directory, isolated from the real global tier."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    empty_global = tmp_path / "global_skills"
    empty_global.mkdir()
    with patch("EvoScientist.paths.GLOBAL_SKILLS_DIR", empty_global):
        yield skills_dir


@pytest.fixture
def sample_skill_dir(tmp_path):
    """Create a sample skill directory with SKILL.md."""
    skill_dir = tmp_path / "sample-skill"
    skill_dir.mkdir()
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        """---
name: sample-skill
description: A sample skill for testing
---

# Sample Skill

This is a sample skill for testing purposes.
"""
    )
    return skill_dir


@pytest.fixture
def sample_skill_no_frontmatter(tmp_path):
    """Create a skill directory without YAML frontmatter."""
    skill_dir = tmp_path / "no-frontmatter-skill"
    skill_dir.mkdir()
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        """# No Frontmatter Skill

This skill has no YAML frontmatter.
"""
    )
    return skill_dir


# =============================================================================
# Tests for _parse_skill_md
# =============================================================================


class TestParseSkillMd:
    """Tests for _parse_skill_md function."""

    def test_parse_with_frontmatter(self, sample_skill_dir):
        skill_md = sample_skill_dir / "SKILL.md"
        result = _parse_skill_md(skill_md)

        assert result.name == "sample-skill"
        assert result.description == "A sample skill for testing"

    def test_parse_without_frontmatter(self, sample_skill_no_frontmatter):
        skill_md = sample_skill_no_frontmatter / "SKILL.md"
        result = _parse_skill_md(skill_md)

        # Should use directory name as fallback
        assert result.name == "no-frontmatter-skill"
        assert result.description == "(no description)"

    def test_parse_with_partial_frontmatter(self, tmp_path):
        skill_dir = tmp_path / "partial-skill"
        skill_dir.mkdir()
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(
            """---
name: my-skill
---

# My Skill
"""
        )

        result = _parse_skill_md(skill_md)
        assert result.name == "my-skill"
        assert result.description == "(no description)"


# =============================================================================
# Tests for _parse_github_url
# =============================================================================


class TestParseGithubUrl:
    """Tests for _parse_github_url function."""

    def test_parse_full_url_with_path(self):
        url = "https://github.com/owner/repo/tree/main/my-skill"
        repo, ref, path = _parse_github_url(url)

        assert repo == "owner/repo"
        assert ref == "main"
        assert path == "my-skill"

    def test_parse_full_url_without_path(self):
        url = "https://github.com/owner/repo/tree/develop"
        repo, ref, path = _parse_github_url(url)

        assert repo == "owner/repo"
        assert ref == "develop"
        assert path is None

    def test_parse_simple_repo_url(self):
        url = "https://github.com/owner/repo"
        repo, ref, path = _parse_github_url(url)

        assert repo == "owner/repo"
        assert ref is None
        assert path is None

    def test_parse_shorthand(self):
        url = "owner/repo@my-skill"
        repo, ref, path = _parse_github_url(url)

        assert repo == "owner/repo"
        assert ref is None
        assert path == "my-skill"

    def test_parse_url_without_protocol(self):
        url = "github.com/owner/repo/tree/v1.0/path/to/skill"
        repo, ref, path = _parse_github_url(url)

        assert repo == "owner/repo"
        assert ref == "v1.0"
        assert path == "path/to/skill"

    def test_invalid_url_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            _parse_github_url("not-a-valid-url")


# =============================================================================
# Tests for _is_github_url
# =============================================================================


class TestIsGithubUrl:
    """Tests for _is_github_url function."""

    def test_github_com_url(self):
        assert _is_github_url("https://github.com/owner/repo") is True
        assert _is_github_url("http://github.com/owner/repo/tree/main/skill") is True

    def test_shorthand(self):
        assert _is_github_url("owner/repo@skill-name") is True

    def test_local_path(self):
        assert _is_github_url("./my-skill") is False
        assert _is_github_url("/absolute/path/skill") is False
        assert _is_github_url("../relative/path") is False

    def test_other_urls(self):
        assert _is_github_url("https://gitlab.com/owner/repo") is False
        assert _is_github_url("file:///path/to/file") is False


# =============================================================================
# Tests for _validate_skill_dir
# =============================================================================


class TestValidateSkillDir:
    """Tests for _validate_skill_dir function."""

    def test_valid_skill_dir(self, sample_skill_dir):
        assert _validate_skill_dir(sample_skill_dir) is True

    def test_invalid_skill_dir_no_skillmd(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        assert _validate_skill_dir(empty_dir) is False

    def test_invalid_skill_dir_file_not_dir(self, tmp_path):
        file_path = tmp_path / "file.txt"
        file_path.write_text("not a directory")
        assert _validate_skill_dir(file_path) is False


# =============================================================================
# Tests for install_skill
# =============================================================================


class TestInstallSkill:
    """Tests for install_skill function."""

    def test_install_from_local_path(self, sample_skill_dir, temp_skills_dir):
        result = install_skill(str(sample_skill_dir), str(temp_skills_dir))

        assert result["success"] is True
        assert result["name"] == "sample-skill"
        assert "sample-skill" in result["path"]

        # Verify the skill was copied
        installed_path = Path(result["path"])
        assert installed_path.exists()
        assert (installed_path / "SKILL.md").exists()

    def test_install_nonexistent_path(self, temp_skills_dir):
        result = install_skill("/nonexistent/path", str(temp_skills_dir))

        assert result["success"] is False
        assert "does not exist" in result["error"]

    def test_install_invalid_skill_no_skillmd(self, tmp_path, temp_skills_dir):
        empty_dir = tmp_path / "empty-skill"
        empty_dir.mkdir()

        result = install_skill(str(empty_dir), str(temp_skills_dir))

        assert result["success"] is False
        assert "No SKILL.md" in result["error"]

    def test_install_replaces_existing(self, sample_skill_dir, temp_skills_dir):
        # Install first time
        result1 = install_skill(str(sample_skill_dir), str(temp_skills_dir))
        assert result1["success"] is True

        # Modify the original skill
        skill_md = sample_skill_dir / "SKILL.md"
        skill_md.write_text(
            """---
name: sample-skill
description: Modified description
---

# Modified
"""
        )

        # Install again
        result2 = install_skill(str(sample_skill_dir), str(temp_skills_dir))
        assert result2["success"] is True
        assert result2["description"] == "Modified description"


# =============================================================================
# Tests for list_skills
# =============================================================================


class TestListSkills:
    """Tests for list_skills function."""

    def test_list_empty_dir(self, temp_skills_dir):
        with patch("EvoScientist.paths.USER_SKILLS_DIR", temp_skills_dir):
            skills = list_skills(include_system=False)
            assert skills == []

    def test_list_with_skills(self, sample_skill_dir, temp_skills_dir):
        # Install a skill
        install_skill(str(sample_skill_dir), str(temp_skills_dir))

        with patch("EvoScientist.paths.USER_SKILLS_DIR", temp_skills_dir):
            skills = list_skills(include_system=False)

            assert len(skills) == 1
            assert skills[0].name == "sample-skill"
            assert skills[0].description == "A sample skill for testing"
            assert skills[0].source == "workspace"

    def test_list_multiple_skills(self, tmp_path, temp_skills_dir):
        # Create and install multiple skills
        for i in range(3):
            skill_dir = tmp_path / f"skill-{i}"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                f"""---
name: skill-{i}
description: Skill number {i}
---
"""
            )
            install_skill(str(skill_dir), str(temp_skills_dir))

        with patch("EvoScientist.paths.USER_SKILLS_DIR", temp_skills_dir):
            skills = list_skills(include_system=False)

            assert len(skills) == 3
            names = [s.name for s in skills]
            assert "skill-0" in names
            assert "skill-1" in names
            assert "skill-2" in names


# =============================================================================
# Tests for uninstall_skill
# =============================================================================


class TestUninstallSkill:
    """Tests for uninstall_skill function."""

    def test_uninstall_existing_skill(self, sample_skill_dir, temp_skills_dir):
        # Install first
        install_skill(str(sample_skill_dir), str(temp_skills_dir))

        with patch("EvoScientist.paths.USER_SKILLS_DIR", temp_skills_dir):
            result = uninstall_skill("sample-skill")

            assert result["success"] is True

            # Verify the skill was removed
            skill_path = temp_skills_dir / "sample-skill"
            assert not skill_path.exists()

    def test_uninstall_nonexistent_skill(self, temp_skills_dir):
        with patch("EvoScientist.paths.USER_SKILLS_DIR", temp_skills_dir):
            result = uninstall_skill("nonexistent-skill")

            assert result["success"] is False
            assert "not found" in result["error"]


# =============================================================================
# Tests for batch install
# =============================================================================


class TestBatchInstall:
    """Tests for batch installing multiple skills from one directory."""

    def _make_skill(self, parent: Path, name: str, desc: str) -> Path:
        """Helper to create a minimal skill directory."""
        d = parent / name
        d.mkdir()
        (d / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {desc}\n---\n\n# {name}\n"
        )
        return d

    def test_batch_install_local_multiple_skills(self, tmp_path, temp_skills_dir):
        """Local path with no root SKILL.md but 3 sub-skills installs all."""
        repo = tmp_path / "multi-repo"
        repo.mkdir()
        self._make_skill(repo, "skill-a", "Alpha")
        self._make_skill(repo, "skill-b", "Beta")
        self._make_skill(repo, "skill-c", "Gamma")

        result = install_skill(str(repo), str(temp_skills_dir))

        assert result["success"] is True
        assert result.get("batch") is True
        assert len(result["installed"]) == 3
        assert result["failed"] == []

        names = {r["name"] for r in result["installed"]}
        assert names == {"skill-a", "skill-b", "skill-c"}

        # Verify files copied
        for name in names:
            assert (temp_skills_dir / name / "SKILL.md").exists()

    def test_batch_install_local_single_still_works(self, tmp_path, temp_skills_dir):
        """Local path with root SKILL.md still installs as single."""
        self._make_skill(tmp_path, "single", "Just one")

        result = install_skill(str(tmp_path / "single"), str(temp_skills_dir))

        assert result["success"] is True
        assert result.get("batch") is not True
        assert result["name"] == "single"

    def test_batch_install_local_empty_repo_fails(self, tmp_path, temp_skills_dir):
        """Local path with no skills at any level fails."""
        empty = tmp_path / "empty-repo"
        empty.mkdir()

        result = install_skill(str(empty), str(temp_skills_dir))

        assert result["success"] is False
        assert "No SKILL.md" in result["error"]

    def test_batch_install_local_mixed_dirs(self, tmp_path, temp_skills_dir):
        """Directories without SKILL.md are silently skipped."""
        repo = tmp_path / "mixed"
        repo.mkdir()
        self._make_skill(repo, "real-skill", "Real")
        (repo / "not-a-skill").mkdir()  # no SKILL.md
        (repo / "readme.md").write_text("# Readme")  # file, not dir

        result = install_skill(str(repo), str(temp_skills_dir))

        assert result["success"] is True
        assert result.get("batch") is not True  # only 1 skill → single install
        assert result["name"] == "real-skill"


# =============================================================================
# Tests for tag parsing
# =============================================================================


class TestParseSkillMdTags:
    """Tests for tag extraction in _parse_skill_md."""

    def test_parse_with_metadata_tags(self, tmp_path):
        """Tags under metadata.tags are extracted."""
        skill_dir = tmp_path / "tagged-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            """---
name: tagged-skill
description: A skill with tags
metadata:
  tags: [core, research, ideation]
---
"""
        )
        result = _parse_skill_md(skill_dir / "SKILL.md")
        assert result.tags == ["core", "research", "ideation"]

    def test_parse_with_top_level_tags(self, tmp_path):
        """Top-level tags field takes precedence over metadata.tags."""
        skill_dir = tmp_path / "top-tags"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            """---
name: top-tags
description: Top-level tags
tags: [writing, review]
metadata:
  tags: [should, not, appear]
---
"""
        )
        result = _parse_skill_md(skill_dir / "SKILL.md")
        assert result.tags == ["writing", "review"]

    def test_parse_no_tags(self, tmp_path):
        """Skills without tags return empty list."""
        skill_dir = tmp_path / "no-tags"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            """---
name: no-tags
description: No tags at all
---
"""
        )
        result = _parse_skill_md(skill_dir / "SKILL.md")
        assert result.tags == []

    def test_parse_comma_string_tags(self, tmp_path):
        """Tags given as comma-separated string are split into list."""
        skill_dir = tmp_path / "string-tags"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            """---
name: string-tags
description: Tags as string
tags: "core, research, writing"
---
"""
        )
        result = _parse_skill_md(skill_dir / "SKILL.md")
        assert result.tags == ["core", "research", "writing"]

    def test_parse_no_frontmatter_returns_empty_tags(self, sample_skill_no_frontmatter):
        """Skills without frontmatter return empty tags."""
        result = _parse_skill_md(sample_skill_no_frontmatter / "SKILL.md")
        assert result.tags == []


# =============================================================================
# Tests for list_skills_by_tag
# =============================================================================


class TestListSkillsByTag:
    """Tests for list_skills_by_tag function."""

    def _make_tagged_skill(self, parent: Path, name: str, tags: list[str]) -> Path:
        d = parent / name
        d.mkdir()
        tags_yaml = ", ".join(tags)
        (d / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Skill {name}\n"
            f"metadata:\n  tags: [{tags_yaml}]\n---\n"
        )
        return d

    def test_filter_by_tag(self, tmp_path, temp_skills_dir):
        self._make_tagged_skill(temp_skills_dir, "skill-a", ["core", "writing"])
        self._make_tagged_skill(temp_skills_dir, "skill-b", ["core", "research"])
        self._make_tagged_skill(temp_skills_dir, "skill-c", ["research"])

        with patch("EvoScientist.paths.USER_SKILLS_DIR", temp_skills_dir):
            core = list_skills_by_tag("core")
            assert len(core) == 2
            assert {s.name for s in core} == {"skill-a", "skill-b"}

            research = list_skills_by_tag("research")
            assert len(research) == 2
            assert {s.name for s in research} == {"skill-b", "skill-c"}

            writing = list_skills_by_tag("writing")
            assert len(writing) == 1
            assert writing[0].name == "skill-a"

    def test_filter_case_insensitive(self, tmp_path, temp_skills_dir):
        self._make_tagged_skill(temp_skills_dir, "skill-x", ["Core", "Writing"])

        with patch("EvoScientist.paths.USER_SKILLS_DIR", temp_skills_dir):
            result = list_skills_by_tag("core")
            assert len(result) == 1
            assert result[0].name == "skill-x"

    def test_filter_nonexistent_tag(self, tmp_path, temp_skills_dir):
        self._make_tagged_skill(temp_skills_dir, "skill-y", ["core"])

        with patch("EvoScientist.paths.USER_SKILLS_DIR", temp_skills_dir):
            result = list_skills_by_tag("nonexistent")
            assert result == []


# =============================================================================
# Tests for get_all_tags
# =============================================================================


class TestGetAllTags:
    """Tests for get_all_tags function."""

    def _make_tagged_skill(self, parent: Path, name: str, tags: list[str]) -> Path:
        d = parent / name
        d.mkdir()
        tags_yaml = ", ".join(tags)
        (d / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Skill {name}\n"
            f"metadata:\n  tags: [{tags_yaml}]\n---\n"
        )
        return d

    def test_returns_tags_with_counts(self, tmp_path, temp_skills_dir):
        self._make_tagged_skill(temp_skills_dir, "skill-a", ["core", "writing"])
        self._make_tagged_skill(temp_skills_dir, "skill-b", ["core", "research"])
        self._make_tagged_skill(temp_skills_dir, "skill-c", ["research"])

        with patch("EvoScientist.paths.USER_SKILLS_DIR", temp_skills_dir):
            tags = get_all_tags()

        tag_dict = dict(tags)
        assert tag_dict["core"] == 2
        assert tag_dict["research"] == 2
        assert tag_dict["writing"] == 1

    def test_empty_when_no_skills(self, temp_skills_dir):
        with patch("EvoScientist.paths.USER_SKILLS_DIR", temp_skills_dir):
            tags = get_all_tags()
            assert tags == []


# =============================================================================
# Tests for fetch_remote_skill_index
# =============================================================================


class TestFetchRemoteSkillIndex:
    """Tests for fetch_remote_skill_index function."""

    def test_fetch_from_local_clone(self, tmp_path):
        """Verify index is built correctly from cloned skills."""
        # Create a fake repo structure
        skills_root = tmp_path / "repo" / "skills"
        skills_root.mkdir(parents=True)

        for name, tags in [
            ("skill-a", ["core", "writing"]),
            ("skill-b", ["core", "research"]),
        ]:
            d = skills_root / name
            d.mkdir()
            tags_yaml = ", ".join(tags)
            (d / "SKILL.md").write_text(
                f"---\nname: {name}\ndescription: Skill {name}\n"
                f"metadata:\n  tags: [{tags_yaml}]\n---\n"
            )

        # Mock _clone_repo to copy our fake repo to the temp dir
        def fake_clone(repo, ref, dest):
            import shutil

            shutil.copytree(tmp_path / "repo", dest)

        with patch(
            "EvoScientist.tools.skills_manager._clone_repo", side_effect=fake_clone
        ):
            # Clear cache to ensure fresh fetch
            from EvoScientist.tools.skills_manager import _REMOTE_INDEX_CACHE

            _REMOTE_INDEX_CACHE.clear()

            index = fetch_remote_skill_index(repo="test/repo", path="skills")

        assert len(index) == 2
        names = {s["name"] for s in index}
        assert names == {"skill-a", "skill-b"}

        # Verify tags are populated
        skill_a = next(s for s in index if s["name"] == "skill-a")
        assert "core" in skill_a["tags"]
        assert "writing" in skill_a["tags"]

        # Verify install_source is set
        assert "test/repo@" in skill_a["install_source"]

    def test_fetch_caches_results(self, tmp_path):
        """Second call within TTL uses cache without cloning again."""
        skills_root = tmp_path / "repo" / "skills"
        skills_root.mkdir(parents=True)
        d = skills_root / "cached-skill"
        d.mkdir()
        (d / "SKILL.md").write_text(
            "---\nname: cached-skill\ndescription: Cached\n"
            "metadata:\n  tags: [core]\n---\n"
        )

        call_count = 0

        def fake_clone(repo, ref, dest):
            nonlocal call_count
            call_count += 1
            import shutil

            shutil.copytree(tmp_path / "repo", dest)

        with patch(
            "EvoScientist.tools.skills_manager._clone_repo", side_effect=fake_clone
        ):
            from EvoScientist.tools.skills_manager import _REMOTE_INDEX_CACHE

            _REMOTE_INDEX_CACHE.clear()

            index1 = fetch_remote_skill_index(repo="cache/test", path="skills")
            index2 = fetch_remote_skill_index(repo="cache/test", path="skills")

        assert call_count == 1  # Only cloned once
        assert index1 == index2
