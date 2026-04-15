"""Tests for EvoScientist.paths — set_workspace_root and ensure_dirs."""

import os
from unittest import mock

import pytest

from EvoScientist import paths


@pytest.fixture(autouse=True)
def _restore_paths():
    """Snapshot module-level path globals and restore after each test."""
    orig = {
        "WORKSPACE_ROOT": paths.WORKSPACE_ROOT,
        "RUNS_DIR": paths.RUNS_DIR,
        "MEMORY_DIR": paths.MEMORY_DIR,
        "USER_SKILLS_DIR": paths.USER_SKILLS_DIR,
        "_active_workspace": paths._active_workspace,
    }
    yield
    paths.WORKSPACE_ROOT = orig["WORKSPACE_ROOT"]
    paths.RUNS_DIR = orig["RUNS_DIR"]
    paths.MEMORY_DIR = orig["MEMORY_DIR"]
    paths.USER_SKILLS_DIR = orig["USER_SKILLS_DIR"]
    paths._active_workspace = orig["_active_workspace"]


class TestSetWorkspaceRoot:
    """Tests for set_workspace_root()."""

    def test_updates_derived_dirs(self, tmp_path):
        """set_workspace_root should update WORKSPACE_ROOT and all derived dirs."""
        new_root = tmp_path / "my_workspace"
        new_root.mkdir()

        paths.set_workspace_root(new_root)

        assert paths.WORKSPACE_ROOT == new_root.resolve()
        assert paths.RUNS_DIR == new_root.resolve() / "runs"
        assert paths.MEMORY_DIR == new_root.resolve() / "memory"
        assert paths.USER_SKILLS_DIR == new_root.resolve() / "skills"

    def test_resets_active_workspace(self, tmp_path):
        """set_workspace_root should reset _active_workspace to new root."""
        new_root = tmp_path / "ws"
        new_root.mkdir()

        # Set active workspace to something different first
        paths._active_workspace = tmp_path / "other"

        paths.set_workspace_root(new_root)

        assert paths._active_workspace == new_root.resolve()

    def test_preserves_env_overrides(self, tmp_path):
        """Dirs set via env vars should NOT be overwritten by set_workspace_root."""
        custom_mem = tmp_path / "custom_memory"
        custom_skills = tmp_path / "custom_skills"
        custom_runs = tmp_path / "custom_runs"

        env = {
            "EVOSCIENTIST_MEMORY_DIR": str(custom_mem),
            "EVOSCIENTIST_SKILLS_DIR": str(custom_skills),
            "EVOSCIENTIST_RUNS_DIR": str(custom_runs),
        }

        new_root = tmp_path / "ws"
        new_root.mkdir()

        with mock.patch.dict(os.environ, env):
            paths.set_workspace_root(new_root)

            # WORKSPACE_ROOT and _active_workspace should still update
            assert paths.WORKSPACE_ROOT == new_root.resolve()
            assert paths._active_workspace == new_root.resolve()

            # Derived dirs should reflect the env overrides, not the new root
            assert paths.MEMORY_DIR == custom_mem.expanduser()
            assert paths.USER_SKILLS_DIR == custom_skills.expanduser()
            assert paths.RUNS_DIR == custom_runs.expanduser()

    def test_accepts_string_path(self, tmp_path):
        """set_workspace_root should accept str as well as Path."""
        new_root = tmp_path / "str_ws"
        new_root.mkdir()

        paths.set_workspace_root(str(new_root))

        assert paths.WORKSPACE_ROOT == new_root.resolve()


class TestEnsureDirsUsesUpdatedPaths:
    """ensure_dirs should create dirs at the currently set paths."""

    def test_ensure_dirs_uses_updated_paths(self, tmp_path):
        """After set_workspace_root, ensure_dirs creates dirs at new location."""
        new_root = tmp_path / "workspace"
        new_root.mkdir()

        paths.set_workspace_root(new_root)
        paths.ensure_dirs()

        assert (new_root / "memory").is_dir()
        assert not (
            new_root / "skills"
        ).exists()  # skills created on demand by install_skill()
