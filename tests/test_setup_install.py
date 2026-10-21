"""Tests for EvoScientist.setup._install (helpers shared by the setup stages)."""

from __future__ import annotations

import os

import pytest

from EvoScientist.setup import _install


@pytest.fixture
def no_retry_pause(monkeypatch):
    monkeypatch.setattr(_install, "_MOVE_RETRY_DELAYS", (0, 0, 0, 0, 0))


def _leftovers(directory):
    return [p.name for p in directory.iterdir() if p.name.endswith(".tmp")]


def test_atomic_write_creates_the_file_and_its_directory(tmp_path):
    target = tmp_path / "tools ąę" / "git.json"
    _install.atomic_write_text(target, '{"a": 1}')
    assert target.read_text(encoding="utf-8") == '{"a": 1}'
    assert _leftovers(target.parent) == []


def test_atomic_write_replaces_existing_content(tmp_path):
    target = tmp_path / "git.json"
    target.write_text("old", encoding="utf-8")
    _install.atomic_write_text(target, "new")
    assert target.read_text(encoding="utf-8") == "new"


def test_atomic_write_retries_a_brief_permission_error(
    tmp_path, monkeypatch, no_retry_pause
):
    """A virus scanner briefly holding the file on Windows."""
    target = tmp_path / "git.json"
    real_replace = os.replace
    denied = [0]

    def replace(src, dst):
        if denied[0] < 2:
            denied[0] += 1
            raise PermissionError(13, "Access is denied")
        return real_replace(src, dst)

    monkeypatch.setattr(_install.os, "replace", replace)
    _install.atomic_write_text(target, "content")
    assert denied[0] == 2
    assert target.read_text(encoding="utf-8") == "content"
    assert _leftovers(tmp_path) == []


def test_atomic_write_failure_leaves_no_temp_file_and_keeps_the_old_one(
    tmp_path, monkeypatch, no_retry_pause
):
    target = tmp_path / "git.json"
    target.write_text("old", encoding="utf-8")

    def replace(src, dst):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(_install.os, "replace", replace)
    with pytest.raises(PermissionError):
        _install.atomic_write_text(target, "new")
    assert target.read_text(encoding="utf-8") == "old"
    assert _leftovers(tmp_path) == []


def test_atomic_write_uses_a_unique_temp_name(tmp_path, monkeypatch):
    """Concurrent writers must never share one temporary file."""
    target = tmp_path / "git.json"
    sources: list[str] = []
    real_replace = os.replace

    def replace(src, dst):
        sources.append(os.path.basename(src))
        return real_replace(src, dst)

    monkeypatch.setattr(_install.os, "replace", replace)
    _install.atomic_write_text(target, "1")
    _install.atomic_write_text(target, "2")
    assert len(set(sources)) == 2
    assert all(s.startswith(".git.json.") and s.endswith(".tmp") for s in sources)


# --------------------------------------------------------------------------- #
# Path rules and PATH lookups
# --------------------------------------------------------------------------- #
def _exe(directory, name):
    """An executable ``name`` in ``directory`` (created), as the lookup sees it."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text("")
    path.chmod(0o755)
    return path


@pytest.fixture
def windows_lookup(monkeypatch):
    """Apply the Windows lookup rules (PATHEXT, no execute bit) on any OS."""
    monkeypatch.setattr(_install, "_WINDOWS", True)
    monkeypatch.setenv("PATHEXT", ".COM;.EXE;.BAT;.CMD")


def test_same_path_ignores_quotes_and_trailing_separators(tmp_path):
    folder = tmp_path / "Program Files ąę" / "Git"
    folder.mkdir(parents=True)
    assert _install.same_path(f'"{folder}"', folder)
    assert _install.same_path(f"{folder}{os.sep}", folder)
    assert not _install.same_path(folder, folder.parent)


@pytest.mark.skipif(os.name != "nt", reason="case folding is Windows-only")
def test_same_path_ignores_case_on_windows(tmp_path):
    folder = tmp_path / "Tools"
    folder.mkdir()
    assert _install.same_path(str(folder).upper(), str(folder).lower())


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_same_path_resolves_links(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "link").symlink_to(real)
    assert _install.same_path(tmp_path / "link", real)


def test_prepend_to_path_drops_other_spellings_of_the_entry(tmp_path, monkeypatch):
    folder = tmp_path / "node-v24"
    folder.mkdir()
    other = str(tmp_path / "other")
    monkeypatch.setenv(
        "PATH", os.pathsep.join([other, f'"{folder}"', f"{folder}{os.sep}"])
    )
    _install.prepend_to_path(folder)
    assert os.environ["PATH"].split(os.pathsep) == [str(folder), other]


def test_which_in_needs_the_execute_bit_on_posix(tmp_path):
    if os.name == "nt":
        pytest.skip("Windows has no execute bit")
    (tmp_path / "node").write_text("")
    assert _install.which_in("node", tmp_path) is None
    (tmp_path / "node").chmod(0o755)
    assert _install.which_in("node", tmp_path) == str(tmp_path / "node")


def test_which_in_tries_pathext_on_windows(tmp_path, windows_lookup):
    (tmp_path / "npx.cmd").write_text("")
    assert _install.which_in("npx", tmp_path) == str(tmp_path / "npx.cmd")
    assert _install.which_in("npm", tmp_path) is None


def test_which_in_never_looks_in_the_current_directory(
    tmp_path, monkeypatch, windows_lookup
):
    """An ``npm.cmd`` where EvoScientist runs is not the one next to a Node."""
    (tmp_path / "cwd").mkdir()
    (tmp_path / "cwd" / "npm.cmd").write_text("")
    (tmp_path / "node-dir").mkdir()
    monkeypatch.chdir(tmp_path / "cwd")
    assert _install.which_in("npm", tmp_path / "node-dir") is None


def test_which_on_path_skips_the_current_directory(tmp_path, monkeypatch):
    """A tool in the current directory, or behind an empty or relative PATH
    entry, is not reported; the real one further down PATH is."""
    name = "git.exe" if os.name == "nt" else "git"
    cwd = tmp_path / "cwd"
    _exe(cwd, name)
    real = _exe(tmp_path / "Git" / "cmd", name)
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("PATH", os.pathsep.join(["", ".", "relative", str(real.parent)]))
    assert _install.which_on_path("git") == str(real)


def test_which_on_path_skips_a_bat_wrapper_in_the_current_directory(
    tmp_path, monkeypatch, windows_lookup
):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "git.bat").write_text("@echo not git")
    real = tmp_path / "Git" / "cmd"
    real.mkdir(parents=True)
    (real / "git.exe").write_text("")
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("PATH", str(real))
    assert _install.which_on_path("git") == str(real / "git.exe")


def test_which_on_path_skips_entries_under_outside(tmp_path, monkeypatch):
    name = "node.exe" if os.name == "nt" else "node"
    tools = tmp_path / "tools"
    private = _exe(tools / "node-v24" / "bin", name)
    system = _exe(tmp_path / "usr" / "bin", name)
    monkeypatch.setenv(
        "PATH", os.pathsep.join([str(private.parent), str(system.parent)])
    )
    assert _install.which_on_path("node") == str(private)
    assert _install.which_on_path("node", outside=tools) == str(system)


def test_which_on_path_reads_quoted_entries(tmp_path, monkeypatch):
    name = "git.exe" if os.name == "nt" else "git"
    real = _exe(tmp_path / "Program Files" / "Git" / "cmd", name)
    monkeypatch.setenv("PATH", f'"{real.parent}"')
    assert _install.which_on_path("git") == str(real)
