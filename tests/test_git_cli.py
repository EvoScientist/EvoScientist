"""Tests for EvoScientist.git_cli."""

import shutil
import subprocess
import sys
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


_NON_INTERACTIVE = ["-c", "core.askPass="]


@pytest.fixture(autouse=True)
def _no_recorded_git(tmp_path, monkeypatch):
    """On Windows a missing git triggers a retry with a recorded PortableGit;
    an empty DATA_DIR keeps the developer's real record out of these tests."""
    from EvoScientist import paths

    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")


class TestRunGit:
    def test_missing_git_raises_git_not_found(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        missing = FileNotFoundError(2, "No such file or directory", "git")
        with patch(_RUN, side_effect=missing):
            with pytest.raises(GitNotFoundError) as excinfo:
                run_git(["--version"], timeout=5)

        assert excinfo.value.__cause__ is missing
        message = str(excinfo.value)
        assert message.startswith("git was not found on PATH.")
        assert "https://git-scm.com/downloads" in message

    def test_windows_message_points_to_evosci_setup(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        assert str(GitNotFoundError()) == (
            "git was not found on PATH. EvoScientist uses git to download skills "
            "and the MCP server index; install Git from "
            "https://git-scm.com/downloads, or run `EvoSci setup`, and try again."
        )

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

        message = str(excinfo.value)
        assert message.startswith("git could not be started:")
        assert "Permission denied" in message
        assert "git-scm.com" not in message

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
        assert run.call_args.args[0] == ["git", *_NON_INTERACTIVE, "--version"]
        assert run.call_args.kwargs["timeout"] == 7
        assert run.call_args.kwargs["encoding"] == "utf-8"
        assert run.call_args.kwargs["errors"] == "replace"

    @pytest.mark.skipif(shutil.which("git") is None, reason="needs a real git")
    def test_non_ascii_path_in_git_output_is_decoded(self, tmp_path):
        """git echoes paths as UTF-8; under a non-UTF-8 code page (Windows) the
        old decoding lost stderr. A profile name such as `Łukasz` in %TEMP%."""
        missing = tmp_path / "Łukasz" / "missing-repo"
        result = run_git(["clone", str(missing), str(tmp_path / "dest")], timeout=30)

        assert result.returncode != 0
        assert result.stderr is not None
        assert "Łukasz" in result.stderr

    def test_windows_retries_with_a_portablegit_set_up_since_start(self, monkeypatch):
        """`EvoSci setup` run in another terminal records PortableGit; the
        running process puts it on PATH and retries once."""
        from pathlib import Path

        from EvoScientist.setup import git as setup_git

        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(setup_git, "activate_runtime", lambda: Path("C:/pg/cmd"))
        missing = FileNotFoundError(2, "No such file or directory", "git")
        with patch(_RUN, side_effect=[missing, _proc(stdout="ok")]) as run:
            result = run_git(["--version"], timeout=5)

        assert result.stdout == "ok"
        assert run.call_count == 2

    def test_windows_without_portablegit_raises_after_one_attempt(self, monkeypatch):
        from EvoScientist.setup import git as setup_git

        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(setup_git, "activate_runtime", lambda: None)
        with patch(
            _RUN, side_effect=FileNotFoundError(2, "No such file", "git")
        ) as run:
            with pytest.raises(GitNotFoundError):
                run_git(["--version"], timeout=5)
        assert run.call_count == 1

    def test_windows_private_git_missing_too_raises(self, monkeypatch):
        from pathlib import Path

        from EvoScientist.setup import git as setup_git

        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(setup_git, "activate_runtime", lambda: Path("C:/pg/cmd"))
        with patch(
            _RUN, side_effect=FileNotFoundError(2, "No such file", "git")
        ) as run:
            with pytest.raises(GitNotFoundError):
                run_git(["--version"], timeout=5)
        assert run.call_count == 2

    def test_windows_retries_only_for_a_missing_git(self, monkeypatch):
        """A git that is there but cannot start is not replaced by PortableGit."""
        from EvoScientist.setup import git as setup_git

        def boom():
            raise AssertionError("activate_runtime must not run")

        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(setup_git, "activate_runtime", boom)
        with patch(_RUN, side_effect=PermissionError(13, "Permission denied")) as run:
            with pytest.raises(GitNotFoundError) as excinfo:
                run_git(["--version"], timeout=5)
        assert run.call_count == 1
        assert str(excinfo.value).startswith("git could not be started:")

    def test_other_platforms_do_not_look_for_portablegit(self, monkeypatch):
        from EvoScientist.setup import git as setup_git

        def boom():
            raise AssertionError("activate_runtime must not run")

        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(setup_git, "activate_runtime", boom)
        with patch(_RUN, side_effect=FileNotFoundError(2, "No such file", "git")):
            with pytest.raises(GitNotFoundError):
                run_git(["--version"], timeout=5)

    def test_never_prompts_but_keeps_the_users_helpers(self, monkeypatch):
        """No terminal prompt, no askpass program, no Credential Manager window;
        the user's own credential helpers (e.g. from `gh auth setup-git`) still
        answer, so private repositories that clone on `main` keep cloning."""
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setenv("GIT_ASKPASS", "/usr/lib/ssh/ssh-askpass")
        monkeypatch.setenv("SSH_ASKPASS", "/usr/lib/ssh/ssh-askpass")
        monkeypatch.setenv("GIT_TERMINAL_PROMPT", "1")
        monkeypatch.setenv("GCM_INTERACTIVE", "auto")
        monkeypatch.setenv("KEEP_ME", "yes")
        with patch(_RUN, return_value=_proc()) as run:
            run_git(["ls-remote", "https://github.com/o/r.git"], timeout=5)

        argv = run.call_args.args[0]
        assert argv[1:3] == _NON_INTERACTIVE
        assert "credential.helper=" not in argv
        env = run.call_args.kwargs["env"]
        assert env["GIT_TERMINAL_PROMPT"] == "0"
        assert env["GCM_INTERACTIVE"] == "0"
        assert "GIT_ASKPASS" not in env
        assert "SSH_ASKPASS" not in env
        assert env["KEEP_ME"] == "yes"

    @pytest.mark.parametrize("private", [True, False])
    def test_windows_resets_the_helper_list_only_for_portablegit(
        self, monkeypatch, private
    ):
        from EvoScientist.setup import git as setup_git

        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(setup_git, "private_git_on_path", lambda: private)
        with patch(_RUN, return_value=_proc()) as run:
            run_git(["--version"], timeout=5)

        assert ("credential.helper=" in run.call_args.args[0]) is private


class TestCloneRepo:
    def test_shallow_clone_command(self):
        with patch(_RUN, return_value=_proc()) as run:
            clone_repo("owner/repo", None, "/tmp/dest")

        assert run.call_args.args[0] == [
            "git",
            *_NON_INTERACTIVE,
            # LF endings stay LF: CRLF breaks shell scripts in skills. eol=lf
            # also covers files a `text=auto` attribute marks as text.
            "-c",
            "core.autocrlf=false",
            "-c",
            "core.eol=lf",
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

        argv = run.call_args.args[0]
        assert argv[argv.index("--branch") + 1] == "v1"

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
