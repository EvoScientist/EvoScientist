"""Tests for the --name/-n run-name deduplication logic."""

from pathlib import Path

from EvoScientist.cli import _deduplicate_run_name

# =============================================================================
# _deduplicate_run_name
# =============================================================================


class TestDeduplicateRunName:
    def test_fresh_name(self, tmp_path: Path):
        assert _deduplicate_run_name("experiment", tmp_path) == "experiment"

    def test_single_collision(self, tmp_path: Path):
        (tmp_path / "experiment").mkdir()
        assert _deduplicate_run_name("experiment", tmp_path) == "experiment_1"

    def test_multiple_collisions(self, tmp_path: Path):
        (tmp_path / "experiment").mkdir()
        (tmp_path / "experiment_1").mkdir()
        (tmp_path / "experiment_2").mkdir()
        assert _deduplicate_run_name("experiment", tmp_path) == "experiment_3"

    def test_gap_in_sequence(self, tmp_path: Path):
        """Picks the first available suffix, not the highest + 1."""
        (tmp_path / "run").mkdir()
        (tmp_path / "run_2").mkdir()
        # _1 doesn't exist, so it should pick _1
        assert _deduplicate_run_name("run", tmp_path) == "run_1"

    def test_does_not_collide_with_files(self, tmp_path: Path):
        """A regular file (not dir) with the same name still counts as taken."""
        (tmp_path / "run").write_text("")
        assert _deduplicate_run_name("run", tmp_path) == "run_1"
