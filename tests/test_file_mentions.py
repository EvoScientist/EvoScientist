"""Tests for EvoScientist.cli.file_mentions module."""

from __future__ import annotations

from pathlib import Path

from EvoScientist.cli.file_mentions import (
    complete_file_mention,
    invalidate_file_cache,
    parse_file_mentions,
    resolve_file_mentions,
)

# ---------------------------------------------------------------------------
# parse_file_mentions
# ---------------------------------------------------------------------------


class TestParseFileMentions:
    def test_single_mention(self, tmp_path: Path) -> None:
        f = tmp_path / "paper.tex"
        f.write_text("hello")
        result = parse_file_mentions(f"read @{f}", cwd=tmp_path)
        assert result == [f.resolve()]

    def test_relative_mention(self, tmp_path: Path) -> None:
        f = tmp_path / "notes.md"
        f.write_text("world")
        result = parse_file_mentions("check @notes.md", cwd=tmp_path)
        assert result == [f.resolve()]

    def test_multiple_mentions(self, tmp_path: Path) -> None:
        a = tmp_path / "a.py"
        b = tmp_path / "b.py"
        a.write_text("a")
        b.write_text("b")
        result = parse_file_mentions(f"diff @{a} @{b}", cwd=tmp_path)
        assert set(result) == {a.resolve(), b.resolve()}

    def test_missing_file_skipped(self, tmp_path: Path) -> None:
        result = parse_file_mentions("@nonexistent.txt", cwd=tmp_path)
        assert result == []

    def test_email_address_ignored(self, tmp_path: Path) -> None:
        result = parse_file_mentions("send to user@example.com", cwd=tmp_path)
        assert result == []

    def test_directory_excluded(self, tmp_path: Path) -> None:
        d = tmp_path / "subdir"
        d.mkdir()
        result = parse_file_mentions(f"@{d}", cwd=tmp_path)
        assert result == []

    def test_bare_at_sign_ignored(self, tmp_path: Path) -> None:
        result = parse_file_mentions("hello @ world", cwd=tmp_path)
        assert result == []

    def test_tilde_expansion(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        f = tmp_path / "file.txt"
        f.write_text("x")
        result = parse_file_mentions("@~/file.txt", cwd=tmp_path)
        assert result == [f.resolve()]


# ---------------------------------------------------------------------------
# resolve_file_mentions
# ---------------------------------------------------------------------------


class TestResolveFileMentions:
    def test_no_mentions_returns_original(self, tmp_path: Path) -> None:
        text = "just a normal message"
        original, final = resolve_file_mentions(text, str(tmp_path))
        assert original == text
        assert final == text

    def test_mention_appends_content(self, tmp_path: Path) -> None:
        f = tmp_path / "README.md"
        f.write_text("# Hello")
        text = f"summarise @{f}"
        original, final = resolve_file_mentions(text, str(tmp_path))
        assert original == text
        assert "## Referenced Files" in final
        assert "README.md" in final
        assert "# Hello" in final

    def test_large_file_reference_only(self, tmp_path: Path) -> None:
        f = tmp_path / "big.bin"
        # Write more than 256 KB
        f.write_bytes(b"x" * (260 * 1024))
        text = f"read @{f}"
        _, final = resolve_file_mentions(text, str(tmp_path))
        assert "too large to embed" in final
        assert "read_file" in final

    def test_multiple_files_all_embedded(self, tmp_path: Path) -> None:
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("AAA")
        b.write_text("BBB")
        _, final = resolve_file_mentions(f"@{a} and @{b}", str(tmp_path))
        assert "AAA" in final
        assert "BBB" in final

    def test_original_text_always_first(self, tmp_path: Path) -> None:
        f = tmp_path / "f.txt"
        f.write_text("content")
        text = f"look at @{f} please"
        _, final = resolve_file_mentions(text, str(tmp_path))
        assert final.startswith(text)

    def test_none_workspace_uses_cwd(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        f = tmp_path / "local.txt"
        f.write_text("local")
        _, final = resolve_file_mentions("@local.txt", None)
        assert "local" in final


# ---------------------------------------------------------------------------
# complete_file_mention
# ---------------------------------------------------------------------------


class TestCompleteFileMention:
    def setup_method(self) -> None:
        # Ensure cache is clean between tests
        invalidate_file_cache()

    def test_no_at_sign_returns_empty(self, tmp_path: Path) -> None:
        result = complete_file_mention("hello world", str(tmp_path))
        assert result == []

    def test_at_sign_alone_lists_files(self, tmp_path: Path) -> None:
        (tmp_path / "paper.tex").write_text("")
        (tmp_path / "code.py").write_text("")
        result = complete_file_mention("read @", str(tmp_path))
        paths = [p for p, _ in result]
        names = [Path(p.lstrip("@")).name for p in paths]
        assert "paper.tex" in names
        assert "code.py" in names

    def test_partial_prefix_filters(self, tmp_path: Path) -> None:
        (tmp_path / "paper.tex").write_text("")
        (tmp_path / "data.csv").write_text("")
        result = complete_file_mention("@pa", str(tmp_path))
        paths = [p for p, _ in result]
        assert any("paper" in p for p in paths)
        assert not any("data" in p for p in paths)

    def test_directory_has_trailing_slash(self, tmp_path: Path) -> None:
        (tmp_path / "workspace").mkdir()
        result = complete_file_mention("@work", str(tmp_path))
        paths = [p for p, _ in result]
        assert any(p.endswith("/") for p in paths)

    def test_hidden_files_excluded(self, tmp_path: Path) -> None:
        (tmp_path / ".hidden").write_text("")
        (tmp_path / "visible.txt").write_text("")
        result = complete_file_mention("@", str(tmp_path))
        paths = [p for p, _ in result]
        assert not any(".hidden" in p for p in paths)
        assert any("visible" in p for p in paths)

    def test_mid_text_at_completes(self, tmp_path: Path) -> None:
        (tmp_path / "intro.tex").write_text("")
        result = complete_file_mention("please look at @intro", str(tmp_path))
        paths = [p for p, _ in result]
        assert any("intro.tex" in p for p in paths)

    def test_no_at_in_text_returns_empty(self, tmp_path: Path) -> None:
        result = complete_file_mention("no mention here", str(tmp_path))
        assert result == []

    def test_returns_tuples(self, tmp_path: Path) -> None:
        (tmp_path / "report.md").write_text("")
        result = complete_file_mention("@rep", str(tmp_path))
        assert result  # at least one result
        assert isinstance(result[0], tuple)
        assert len(result[0]) == 2

    def test_type_hint_extension(self, tmp_path: Path) -> None:
        (tmp_path / "script.py").write_text("")
        (tmp_path / "notes.md").write_text("")
        (tmp_path / "data.csv").write_text("")
        result = complete_file_mention("@", str(tmp_path))
        hint_map = {Path(p.lstrip("@")).name: h for p, h in result}
        assert hint_map.get("script.py") == "py"
        assert hint_map.get("notes.md") == "md"
        assert hint_map.get("data.csv") == "csv"

    def test_type_hint_no_extension(self, tmp_path: Path) -> None:
        (tmp_path / "Makefile").write_text("")
        result = complete_file_mention("@Make", str(tmp_path))
        hint_map = {Path(p.lstrip("@")).name: h for p, h in result}
        assert hint_map.get("Makefile") == "file"

    def test_directory_type_hint(self, tmp_path: Path) -> None:
        (tmp_path / "subdir").mkdir()
        result = complete_file_mention("@sub", str(tmp_path))
        pairs = dict(result)
        # The dir entry ends with /
        dir_entries = [(p, h) for p, h in pairs.items() if p.endswith("/")]
        assert dir_entries
        assert dir_entries[0][1] == "dir"

    def test_fuzzy_match_partial_query(self, tmp_path: Path) -> None:
        """'resu' should fuzzy-match 'results.json'."""
        (tmp_path / "results.json").write_text("")
        (tmp_path / "readme.txt").write_text("")
        result = complete_file_mention("@resu", str(tmp_path))
        paths = [p for p, _ in result]
        assert any("results.json" in p for p in paths)

    def test_fuzzy_deep_file(self, tmp_path: Path) -> None:
        """Files nested 2+ levels deep are discovered by fuzzy search."""
        nested = tmp_path / "src" / "models"
        nested.mkdir(parents=True)
        (nested / "base.py").write_text("")
        result = complete_file_mention("@base", str(tmp_path))
        paths = [p for p, _ in result]
        assert any("base.py" in p for p in paths)

    def test_cache_reuse(self, tmp_path: Path) -> None:
        """Second call with same workspace does not rescan."""
        (tmp_path / "file1.txt").write_text("")
        ws = str(tmp_path)
        # Warm cache
        complete_file_mention("@", ws)
        # Add a new file — should NOT appear until cache is invalidated
        (tmp_path / "new_file.txt").write_text("")
        result = complete_file_mention("@new", ws)
        paths = [p for p, _ in result]
        assert not any("new_file" in p for p in paths)

    def test_invalidate_cache_triggers_rescan(self, tmp_path: Path) -> None:
        """After invalidation, new files appear in results."""
        (tmp_path / "file1.txt").write_text("")
        ws = str(tmp_path)
        complete_file_mention("@", ws)
        # Add new file then invalidate
        (tmp_path / "new_file.txt").write_text("")
        invalidate_file_cache(ws)
        result = complete_file_mention("@new", ws)
        paths = [p for p, _ in result]
        assert any("new_file" in p for p in paths)
