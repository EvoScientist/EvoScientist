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
