"""Tests for EvoScientist.tools.skills_manager module."""

from pathlib import Path
from unittest.mock import patch

import pytest

from EvoScientist.tools.skills_manager import (
    SkillInfo,
    _is_github_url,
    _load_manifest,
    _parse_github_url,
    _parse_skill_md,
    _record_install,
    _validate_skill_dir,
    fetch_remote_skill_index,
    get_all_tags,
    install_skill,
    installed_provenance,
    installed_sources,
    list_expert_skills,
    list_skills,
    list_skills_by_tag,
    resolve_remote_head,
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
# Tests for install manifest
# =============================================================================


class TestInstallManifest:
    """Tests for the per-tier .installed.yaml manifest."""

    def _make_skill(self, parent: Path, name: str) -> Path:
        d = parent / name
        d.mkdir()
        (d / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {name} description\n---\n\n# {name}\n"
        )
        return d

    def test_local_install_records_source(self, sample_skill_dir, temp_skills_dir):
        result = install_skill(str(sample_skill_dir), str(temp_skills_dir))
        assert result["success"]

        manifest = _load_manifest(temp_skills_dir)
        # Local installs have no upstream commit — record source only.
        assert manifest == {"sample-skill": {"source": str(sample_skill_dir)}}

    def test_pack_install_records_one_source_for_all_children(
        self, tmp_path, temp_skills_dir
    ):
        """A pack (root has no SKILL.md, multiple child skills) should record
        the same user-facing source for every child skill, so detection by
        source still works."""
        repo = tmp_path / "evoskills-fake"
        repo.mkdir()
        self._make_skill(repo, "paper-writing")
        self._make_skill(repo, "evo-memory")
        self._make_skill(repo, "research-survey")

        result = install_skill(str(repo), str(temp_skills_dir))
        assert result["success"]
        assert result["batch"]

        manifest = _load_manifest(temp_skills_dir)
        assert set(manifest) == {"paper-writing", "evo-memory", "research-survey"}
        sources = {entry["source"] for entry in manifest.values()}
        assert sources == {str(repo)}

    def test_uninstall_clears_manifest_entry(self, sample_skill_dir, temp_skills_dir):
        install_skill(str(sample_skill_dir), str(temp_skills_dir))
        assert "sample-skill" in _load_manifest(temp_skills_dir)

        with patch("EvoScientist.paths.USER_SKILLS_DIR", temp_skills_dir):
            result = uninstall_skill("sample-skill")

        assert result["success"]
        assert "sample-skill" not in _load_manifest(temp_skills_dir)

    def test_save_is_atomic_and_leaves_no_temp(self, sample_skill_dir, temp_skills_dir):
        """_save_manifest must rename a temp file into place, not overwrite,
        so a crash mid-write can't leave a half-written manifest behind."""
        install_skill(str(sample_skill_dir), str(temp_skills_dir))

        manifest_files = sorted(p.name for p in temp_skills_dir.iterdir())
        # Only the manifest itself should remain — no leftover .tmp siblings.
        assert ".installed.yaml" in manifest_files
        assert not any(name.endswith(".tmp") for name in manifest_files), (
            f"unexpected temp file left behind: {manifest_files}"
        )

    def test_installed_sources_filters_missing_dirs(
        self, sample_skill_dir, temp_skills_dir, tmp_path
    ):
        """If a skill dir was removed manually but the manifest entry lingers,
        installed_sources() must not report it as installed."""
        empty_global = tmp_path / "empty_global"
        empty_global.mkdir()
        with (
            patch("EvoScientist.paths.USER_SKILLS_DIR", temp_skills_dir),
            patch("EvoScientist.paths.GLOBAL_SKILLS_DIR", empty_global),
        ):
            install_skill(str(sample_skill_dir), str(temp_skills_dir))
            assert installed_sources() == {str(sample_skill_dir)}

            # Manually wipe the dir, manifest still has the entry.
            import shutil as _shutil

            _shutil.rmtree(temp_skills_dir / "sample-skill")
            assert "sample-skill" in _load_manifest(temp_skills_dir)
            assert installed_sources() == set()

    def test_record_install_persists_commit(self, temp_skills_dir, sample_skill_dir):
        """When _record_install gets a commit SHA it makes it through the
        normalize-on-write pass and is readable as provenance."""
        install_skill(str(sample_skill_dir), str(temp_skills_dir))
        # Simulate a github install by manually recording a commit.
        _record_install(
            temp_skills_dir,
            "sample-skill",
            "owner/repo@skill",
            commit="abc123def456",
        )
        with (
            patch("EvoScientist.paths.USER_SKILLS_DIR", temp_skills_dir),
            patch(
                "EvoScientist.paths.GLOBAL_SKILLS_DIR",
                temp_skills_dir.parent / "missing",
            ),
        ):
            prov = installed_provenance()
        assert prov == {"owner/repo@skill": {"commit": "abc123def456"}}


class TestResolveRemoteHead:
    """Tests for resolve_remote_head — the upstream-SHA helper used by onboard."""

    def test_returns_none_for_local_path(self):
        assert resolve_remote_head("/tmp/some/local/path") is None

    def test_returns_none_when_git_unavailable(self):
        with patch("EvoScientist.tools.skills_manager.subprocess.run") as run:
            run.side_effect = FileNotFoundError("git not on PATH")
            assert resolve_remote_head("owner/repo@skill") is None

    def test_returns_none_on_timeout(self):
        import subprocess as _sp

        with patch("EvoScientist.tools.skills_manager.subprocess.run") as run:
            run.side_effect = _sp.TimeoutExpired(cmd="git", timeout=5)
            assert resolve_remote_head("owner/repo@skill") is None

    def test_parses_first_sha_from_ls_remote_output(self):
        proc = type(
            "P",
            (),
            {
                "returncode": 0,
                "stdout": "deadbeef0000000000000000000000000000abcd\trefs/heads/main\n",
                "stderr": "",
            },
        )()
        with patch(
            "EvoScientist.tools.skills_manager.subprocess.run", return_value=proc
        ):
            sha = resolve_remote_head("owner/repo@skill")
        assert sha == "deadbeef0000000000000000000000000000abcd"


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


# =============================================================================
# Tests for skill_manager tool — action="list" filtering
# =============================================================================


class TestSkillManagerList:
    """Tests for the skill_manager() tool's action='list' output.

    These tests verify that the source-based filtering in skill_manager.py
    correctly maps skills_manager.py's tier names ("workspace", "global",
    "builtin") to the User Skills / System Skills display sections.
    """

    def _make_skill(self, tmp_path, name, description="A skill"):
        skill_dir = tmp_path / name
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {description}\n---\n"
        )
        return skill_dir

    def test_list_user_skills_workspace(self, tmp_path):
        """Workspace-tier skills appear under 'User Skills'."""
        from EvoScientist.tools.skill_manager import skill_manager

        workspace_dir = tmp_path / "workspace"
        workspace_dir.mkdir()
        global_dir = tmp_path / "global"
        global_dir.mkdir()
        self._make_skill(tmp_path, "ws-skill")
        install_skill(str(tmp_path / "ws-skill"), str(workspace_dir))

        with (
            patch("EvoScientist.paths.USER_SKILLS_DIR", workspace_dir),
            patch("EvoScientist.paths.GLOBAL_SKILLS_DIR", global_dir),
        ):
            result = skill_manager.invoke({"action": "list", "include_system": False})

        assert "User Skills (1)" in result
        assert "ws-skill" in result
        assert "System Skills" not in result

    def test_list_user_skills_global(self, tmp_path):
        """Global-tier skills appear under 'User Skills'."""
        from EvoScientist.tools.skill_manager import skill_manager

        workspace_dir = tmp_path / "workspace"
        workspace_dir.mkdir()
        global_dir = tmp_path / "global"
        global_dir.mkdir()
        self._make_skill(tmp_path, "global-skill")
        install_skill(str(tmp_path / "global-skill"), str(global_dir))

        with (
            patch("EvoScientist.paths.USER_SKILLS_DIR", workspace_dir),
            patch("EvoScientist.paths.GLOBAL_SKILLS_DIR", global_dir),
        ):
            result = skill_manager.invoke({"action": "list", "include_system": False})

        assert "User Skills (1)" in result
        assert "global-skill" in result

    def test_list_include_system_shows_both_sections(self, tmp_path):
        """include_system=True shows both User Skills and System Skills sections."""
        from EvoScientist.tools.skill_manager import skill_manager
        from EvoScientist.tools.skills_manager import SkillInfo

        workspace_dir = tmp_path / "workspace"
        workspace_dir.mkdir()
        global_dir = tmp_path / "global"
        global_dir.mkdir()
        self._make_skill(tmp_path, "user-skill")
        install_skill(str(tmp_path / "user-skill"), str(workspace_dir))

        builtin_skill = SkillInfo(
            name="builtin-skill",
            description="A built-in skill",
            path=tmp_path / "builtin-skill",
            source="builtin",
        )

        with (
            patch("EvoScientist.paths.USER_SKILLS_DIR", workspace_dir),
            patch("EvoScientist.paths.GLOBAL_SKILLS_DIR", global_dir),
            patch(
                "EvoScientist.tools.skills_manager.list_skills",
                return_value=[
                    SkillInfo(
                        name="user-skill",
                        description="A user skill",
                        path=workspace_dir / "user-skill",
                        source="workspace",
                    ),
                    builtin_skill,
                ],
            ),
        ):
            result = skill_manager.invoke({"action": "list", "include_system": True})

        assert "User Skills (1)" in result
        assert "user-skill" in result
        assert "System Skills (1)" in result
        assert "builtin-skill" in result

    def test_list_no_user_skills_returns_message(self, tmp_path):
        """Empty workspace and global dirs return the 'no user skills' message."""
        from EvoScientist.tools.skill_manager import skill_manager

        workspace_dir = tmp_path / "workspace"
        workspace_dir.mkdir()
        global_dir = tmp_path / "global"
        global_dir.mkdir()

        with (
            patch("EvoScientist.paths.USER_SKILLS_DIR", workspace_dir),
            patch("EvoScientist.paths.GLOBAL_SKILLS_DIR", global_dir),
        ):
            result = skill_manager.invoke({"action": "list", "include_system": False})

        assert "No user skills installed" in result


# =============================================================================
# Tests for expert-skill fields (agent-teams v1)
# =============================================================================


def _write_expert_skill(
    parent: Path,
    name: str,
    *,
    role: str = "One-line role",
    byline: str = "Test persona",
    capability_tags: list[str] | None = None,
    avatar_hint: str = "star",
    default_dispatch: str = "sync",
    include_description: bool = True,
) -> Path:
    """Write an expert SKILL.md under parent/<name>/ and return the skill dir."""
    skill_dir = parent / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    tags_str = "[" + ", ".join(capability_tags or []) + "]" if capability_tags else "[]"
    desc_line = f"description: A {name} expert skill\n" if include_description else ""
    (skill_dir / "SKILL.md").write_text(
        f"""---
name: {name}
{desc_line}type: expert
role: {role}
byline: {byline}
capability_tags: {tags_str}
avatar_hint: {avatar_hint}
default_dispatch: {default_dispatch}
---

# {name}

Expert-skill body.
"""
    )
    return skill_dir


class TestParseSkillMdExpertFields:
    """`_parse_skill_md` extracts expert-skill frontmatter fields onto SkillInfo."""

    def test_utility_default_when_type_absent(self, sample_skill_dir):
        """Existing skills (no `type` field) default to utility with empty expert fields."""
        result = _parse_skill_md(sample_skill_dir / "SKILL.md")
        assert result.type == "utility"
        assert result.role == ""
        assert result.byline == ""
        assert result.capability_tags == []
        assert result.avatar_hint == ""
        assert result.default_dispatch == ""

    def test_expert_fields_extracted(self, tmp_path):
        skill_dir = _write_expert_skill(
            tmp_path,
            "expert-a",
            role="Expert in A",
            byline="A Byline",
            capability_tags=["tag-1", "tag-2"],
            avatar_hint="atom",
            default_dispatch="panel",
        )
        result = _parse_skill_md(skill_dir / "SKILL.md")
        assert result.type == "expert"
        assert result.role == "Expert in A"
        assert result.byline == "A Byline"
        assert result.capability_tags == ["tag-1", "tag-2"]
        assert result.avatar_hint == "atom"
        assert result.default_dispatch == "panel"

    def test_body_populated_from_skill_md(self, tmp_path):
        """`SkillInfo.body` carries the post-frontmatter content so the expert
        container factory can build the system_prompt without re-reading disk."""
        skill_dir = _write_expert_skill(tmp_path, "expert-body")
        result = _parse_skill_md(skill_dir / "SKILL.md")
        assert "# expert-body" in result.body
        assert "Expert-skill body." in result.body
        # Frontmatter fences must NOT leak into the body.
        assert "---" not in result.body

    def test_body_captured_when_no_frontmatter(self, tmp_path):
        """Skills without a frontmatter block still cache their full text as
        the body — the expert-container factory can then use it without a
        second disk read."""
        skill_dir = tmp_path / "no-frontmatter"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("Just body content, no fences.\n")
        result = _parse_skill_md(skill_dir / "SKILL.md")
        assert result.body.strip() == "Just body content, no fences."

    def test_unknown_type_falls_back_to_utility(self, tmp_path, caplog):
        """A typo in `type` (e.g. `charcter`) must not silently register as an expert
        and must log so authors can debug a missing-expert case."""
        skill_dir = tmp_path / "typo-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            """---
name: typo-skill
description: Has a bad type value
type: charcter
role: This should be ignored
---

# Body
"""
        )
        result = _parse_skill_md(skill_dir / "SKILL.md")
        assert result.type == "utility"
        assert any(
            "unrecognized type" in r.message and "charcter" in r.message
            for r in caplog.records
        )

    def test_invalid_default_dispatch_falls_back_to_empty(self, tmp_path):
        skill_dir = tmp_path / "bad-dispatch"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            """---
name: bad-dispatch
description: Has a bad default_dispatch
type: expert
role: Some role
default_dispatch: asynchronous
---

# Body
"""
        )
        result = _parse_skill_md(skill_dir / "SKILL.md")
        assert result.type == "expert"
        assert result.default_dispatch == ""  # rejected, not passed through

    def test_capability_tags_accepts_comma_string(self, tmp_path):
        """capability_tags falls back to comma-separated string parsing (like `tags`)."""
        skill_dir = tmp_path / "comma-tags"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            """---
name: comma-tags
description: Comma-separated capability tags
type: expert
capability_tags: alpha, beta, gamma
---

# Body
"""
        )
        result = _parse_skill_md(skill_dir / "SKILL.md")
        assert result.capability_tags == ["alpha", "beta", "gamma"]

    def test_unreadable_skill_md_returns_placeholder(self, tmp_path, caplog):
        """A non-UTF-8 SKILL.md must degrade to an ``(unreadable)`` placeholder
        rather than raise. ``list_skills`` sits on the agent-construction hot
        path (via ``_fold_expert_subagents``); an uncaught ``UnicodeDecodeError``
        would abort CLI startup for a single malformed skill in any tier."""
        skill_dir = tmp_path / "bad-utf8"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_bytes(b"---\nname: bad\n---\n\xff\xfe")
        result = _parse_skill_md(skill_dir / "SKILL.md")
        assert result.name == "bad-utf8"
        assert result.description == "(unreadable)"
        assert result.type == "utility"
        assert result.body == ""
        assert any("could not read" in r.message for r in caplog.records)

    def test_empty_name_value_coerces_to_parent_dir(self, tmp_path):
        """A ``name:`` line with no value parses to ``None`` in YAML. Without
        the ``.get(...) or parent.name`` guard, ``SkillInfo.name`` becomes
        ``None`` and slips past the ``_fold_expert_subagents`` collision
        check (a ``None``-named expert would land in the ``task`` schema)."""
        skill_dir = tmp_path / "empty-name"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            """---
name:
description: A skill with an empty name value
type: expert
role: some role
---

Body content.
"""
        )
        result = _parse_skill_md(skill_dir / "SKILL.md")
        assert result.name == "empty-name"

    def test_invalid_frontmatter_yaml_warns(self, tmp_path, caplog):
        """Malformed YAML frontmatter must return the ``(invalid frontmatter)``
        placeholder AND log so the author sees why the skill didn't register."""
        skill_dir = tmp_path / "bad-yaml"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            """---
name: bad-yaml
description: broken YAML below
tags: [unterminated
---

Body.
"""
        )
        result = _parse_skill_md(skill_dir / "SKILL.md")
        assert result.description == "(invalid frontmatter)"
        assert any("invalid frontmatter YAML" in r.message for r in caplog.records)


class TestListExpertSkills:
    """`list_expert_skills()` filters `list_skills()` to `type == 'expert'`."""

    def test_returns_only_expert_skills(self, tmp_path):
        workspace_dir = tmp_path / "workspace"
        workspace_dir.mkdir()
        global_dir = tmp_path / "global"
        global_dir.mkdir()
        # An expert skill and a utility skill, both in workspace tier.
        _write_expert_skill(workspace_dir, "expert-a")
        util = workspace_dir / "util-b"
        util.mkdir()
        (util / "SKILL.md").write_text(
            """---
name: util-b
description: Plain utility skill
---

# Body
"""
        )
        with (
            patch("EvoScientist.paths.USER_SKILLS_DIR", workspace_dir),
            patch("EvoScientist.paths.GLOBAL_SKILLS_DIR", global_dir),
        ):
            all_skills = list_skills()
            expert_skills = list_expert_skills(include_system=False)
        assert {s.name for s in all_skills} == {"expert-a", "util-b"}
        assert [s.name for s in expert_skills] == ["expert-a"]

    def test_empty_when_no_expert_skills_installed(self, tmp_path):
        workspace_dir = tmp_path / "workspace"
        workspace_dir.mkdir()
        global_dir = tmp_path / "global"
        global_dir.mkdir()
        util = workspace_dir / "util-only"
        util.mkdir()
        (util / "SKILL.md").write_text(
            """---
name: util-only
description: Utility
---

# Body
"""
        )
        with (
            patch("EvoScientist.paths.USER_SKILLS_DIR", workspace_dir),
            patch("EvoScientist.paths.GLOBAL_SKILLS_DIR", global_dir),
        ):
            expert_skills = list_expert_skills(include_system=False)
        assert expert_skills == []


class TestSkillManagerToolExpertSurface:
    """`skill_manager` @tool exposes the expert-skill fields and filter."""

    def _mock_skills(self):
        return [
            SkillInfo(
                name="expert-a",
                description="A brainstorm expert",
                path=Path("/skills/expert-a"),
                source="builtin",
                tags=["research"],
                type="expert",
                role="Research idea brainstormer",
                byline="Ideation persona",
                capability_tags=["Iteration", "ELO"],
                avatar_hint="lightbulb",
                default_dispatch="sync",
            ),
            SkillInfo(
                name="util-b",
                description="A utility skill",
                path=Path("/skills/util-b"),
                source="workspace",
                tags=["core"],
            ),
        ]

    def test_list_filters_to_expert_when_skill_type_set(self):
        from EvoScientist.tools.skill_manager import skill_manager

        with patch(
            "EvoScientist.tools.skills_manager.list_skills",
            return_value=self._mock_skills(),
        ):
            out = skill_manager.invoke(
                {"action": "list", "include_system": True, "skill_type": "expert"}
            )
        assert "expert-a" in out
        assert "util-b" not in out

    def test_skill_type_enum_contains_no_empty_string(self):
        """Gemini's function-declaration schema rejects empty enum values
        (`GenerateContentRequest.tools[N].function_declarations[N].parameters.properties[skill_type].enum[0]: cannot be empty`).
        The `skill_type` argument must use a non-empty sentinel (`"all"`)
        as its no-filter default, never `""`.

        This test guards against silently reintroducing the empty-string
        default that broke the live agent-teams smoke on 2026-07-17.
        """
        from EvoScientist.tools.skill_manager import skill_manager

        schema = skill_manager.args_schema.model_json_schema()
        skill_type_prop = schema.get("properties", {}).get("skill_type", {})
        # Pydantic/JSON-schema serialization of a Literal[...] shows up as
        # `enum` on the property directly OR nested under `anyOf`.
        enum_values: list[str] = []
        if "enum" in skill_type_prop:
            enum_values = list(skill_type_prop["enum"])
        else:
            for branch in skill_type_prop.get("anyOf", []):
                if "enum" in branch:
                    enum_values.extend(branch["enum"])
        assert enum_values, "skill_type Literal should surface as enum in the schema"
        assert "" not in enum_values, (
            f"Empty string in skill_type enum will break Gemini: {enum_values}"
        )

    def test_list_all_sentinel_is_no_filter(self):
        """`skill_type='all'` (the default) must return every skill —
        it's the no-filter case, not a bucket that only 'all' skills fall into."""
        from EvoScientist.tools.skill_manager import skill_manager

        with patch(
            "EvoScientist.tools.skills_manager.list_skills",
            return_value=self._mock_skills(),
        ):
            out = skill_manager.invoke(
                {"action": "list", "include_system": True, "skill_type": "all"}
            )
        # Both should appear — 'all' is not a filter to a bucket named "all".
        assert "expert-a" in out
        assert "util-b" in out

    def test_list_all_when_skill_type_absent(self):
        from EvoScientist.tools.skill_manager import skill_manager

        with patch(
            "EvoScientist.tools.skills_manager.list_skills",
            return_value=self._mock_skills(),
        ):
            out = skill_manager.invoke({"action": "list", "include_system": True})
        assert "expert-a" in out
        assert "util-b" in out

    def test_list_returns_message_when_filter_matches_nothing(self):
        from EvoScientist.tools.skill_manager import skill_manager

        with patch(
            "EvoScientist.tools.skills_manager.list_skills",
            return_value=self._mock_skills()[1:],  # only the utility
        ):
            out = skill_manager.invoke(
                {"action": "list", "include_system": True, "skill_type": "expert"}
            )
        assert "No expert skills found" in out

    def test_info_surfaces_expert_fields(self):
        from EvoScientist.tools.skill_manager import skill_manager

        with patch(
            "EvoScientist.tools.skills_manager.get_skill_info",
            return_value=self._mock_skills()[0],
        ):
            out = skill_manager.invoke({"action": "info", "name": "expert-a"})
        assert "Type: expert" in out
        assert "Role: Research idea brainstormer" in out
        assert "Byline: Ideation persona" in out
        assert "Capability tags: Iteration, ELO" in out
        assert "Avatar hint: lightbulb" in out
        assert "Default dispatch: sync" in out

    def test_info_omits_expert_block_for_utility_skills(self):
        from EvoScientist.tools.skill_manager import skill_manager

        with patch(
            "EvoScientist.tools.skills_manager.get_skill_info",
            return_value=self._mock_skills()[1],
        ):
            out = skill_manager.invoke({"action": "info", "name": "util-b"})
        assert "Type: expert" not in out
        assert "Role:" not in out
        assert "Byline:" not in out
        assert "Capability tags:" not in out
        assert "Default dispatch:" not in out
