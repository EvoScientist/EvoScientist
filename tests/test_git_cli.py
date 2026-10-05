"""Tests for EvoScientist.git_cli."""

import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from EvoScientist.git_cli import (
    CLONE_TIMEOUT,
    GitNotFoundError,
    clone_repo,
    run_git,
)

_RUN = "EvoScientist.git_cli.subprocess.run"


def _proc(returncode=0, stdout="", stderr=""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


class TestRunGit:
    def test_missing_git_raises_git_not_found(self):
        missing = FileNotFoundError(2, "No such file or directory", "git")
        with patch(_RUN, side_effect=missing):
            with pytest.raises(GitNotFoundError) as excinfo:
                run_git(["--version"], timeout=5)

        assert excinfo.value.__cause__ is missing
        message = str(excinfo.value)
        assert message.startswith("git was not found on PATH.")
        assert "https://git-scm.com/downloads" in message

    def test_real_cause_is_logged(self, caplog):
        too_many = OSError(24, "Too many open files")
        with caplog.at_level("DEBUG", logger="EvoScientist.git_cli"):
            with patch(_RUN, side_effect=too_many):
                with pytest.raises(GitNotFoundError):
                    run_git(["--version"], timeout=5)

        record = next(r for r in caplog.records if r.name == "EvoScientist.git_cli")
        assert record.getMessage() == "could not start git"
        assert record.exc_info[1] is too_many

    def test_git_not_found_is_a_runtime_error(self):
        # Callers already turn RuntimeError into a clean error result.
        assert issubclass(GitNotFoundError, RuntimeError)

    def test_other_start_errors_name_the_cause(self):
        """git is on PATH but cannot start: no install hint, the real cause."""
        with patch(_RUN, side_effect=PermissionError(13, "Permission denied")):
            with pytest.raises(GitNotFoundError) as excinfo:
                run_git(["--version"], timeout=5)

        assert str(excinfo.value) == "git could not be started: Permission denied"

    def test_git_not_found_error_pickles_and_copies(self):
        import copy
        import pickle

        for error in (
            GitNotFoundError(),
            GitNotFoundError("git could not be started: x"),
        ):
            assert str(pickle.loads(pickle.dumps(error))) == str(error)
            assert str(copy.copy(error)) == str(error)

    def test_timeout_propagates(self):
        with patch(_RUN, side_effect=subprocess.TimeoutExpired("git", 5)):
            with pytest.raises(subprocess.TimeoutExpired):
                run_git(["ls-remote", "x"], timeout=5)

    def test_runs_git_with_args_and_returns_result(self):
        with patch(_RUN, return_value=_proc(stdout="git version 2.43.0\n")) as run:
            result = run_git(["--version"], timeout=7)

        assert result.stdout == "git version 2.43.0\n"
        assert run.call_args.args[0] == ["git", "--version"]
        assert run.call_args.kwargs["timeout"] == 7


class TestCloneRepo:
    def test_shallow_clone_command(self):
        with patch(_RUN, return_value=_proc()) as run:
            clone_repo("owner/repo", None, "/tmp/dest")

        assert run.call_args.args[0] == [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/owner/repo.git",
            "/tmp/dest",
        ]
        assert run.call_args.kwargs["timeout"] == CLONE_TIMEOUT

    def test_ref_becomes_branch(self):
        with patch(_RUN, return_value=_proc()) as run:
            clone_repo("owner/repo", "v1", "/tmp/dest")

        assert run.call_args.args[0][4:6] == ["--branch", "v1"]

    def test_missing_git_raises_git_not_found(self):
        with patch(_RUN, side_effect=FileNotFoundError(2, "No such file", "git")):
            with pytest.raises(GitNotFoundError):
                clone_repo("owner/repo", None, "/tmp/dest")

    def test_timeout_reports_timed_out(self):
        with patch(_RUN, side_effect=subprocess.TimeoutExpired("git", CLONE_TIMEOUT)):
            with pytest.raises(RuntimeError, match="git clone timed out") as excinfo:
                clone_repo("owner/repo", None, "/tmp/dest")

        assert not isinstance(excinfo.value, GitNotFoundError)

    def test_failed_clone_reports_stderr(self):
        failed = _proc(returncode=128, stderr="fatal: repository not found\n")
        with patch(_RUN, return_value=failed):
            with pytest.raises(
                RuntimeError, match="git clone failed: fatal: repository not found"
            ) as excinfo:
                clone_repo("owner/repo", None, "/tmp/dest")

        assert not isinstance(excinfo.value, GitNotFoundError)
