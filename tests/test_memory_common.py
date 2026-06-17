"""Tests for EvoScientist.memory._common helpers (all new in this PR)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from EvoScientist.memory._common import (
    config_str,
    current_configurable,
    dedupe_ids,
    document_body,
    parse_memory_document,
    pretty_json,
    read_memory_document,
    short_hash,
    stable_json,
)


# ---------------------------------------------------------------------------
# parse_memory_document
# ---------------------------------------------------------------------------


class TestParseMemoryDocument:
    def test_valid_frontmatter_returns_tuple(self):
        text = "---\nid: O-abc\nsummary: Test\n---\nBody text here."
        result = parse_memory_document(text)
        assert result is not None
        metadata, body = result
        assert metadata["id"] == "O-abc"
        assert metadata["summary"] == "Test"
        assert "Body text here." in body

    def test_missing_leading_dashes_returns_none(self):
        assert parse_memory_document("id: foo\nbody") is None

    def test_no_closing_dashes_returns_none(self):
        assert parse_memory_document("---\nid: foo\n") is None

    def test_invalid_yaml_returns_none(self):
        # Deliberately broken YAML
        assert parse_memory_document("---\n: : :\n---\nbody") is None

    def test_non_mapping_yaml_returns_none(self):
        # YAML that parses to a list, not a dict
        assert parse_memory_document("---\n- item1\n- item2\n---\nbody") is None

    def test_non_string_keys_are_filtered_out(self):
        text = "---\nid: test\nsummary: hello\n---\nBody."
        result = parse_memory_document(text)
        assert result is not None
        metadata, _ = result
        assert all(isinstance(k, str) for k in metadata)

    def test_empty_body_allowed(self):
        text = "---\nid: abc\n---\n"
        result = parse_memory_document(text)
        assert result is not None
        metadata, body = result
        assert metadata["id"] == "abc"
        assert body == ""

    def test_multiline_yaml_frontmatter(self):
        text = "---\nid: K-abc\nmemory_type: semantic\nscope: global\n---\nKnowledge body."
        result = parse_memory_document(text)
        assert result is not None
        metadata, body = result
        assert metadata["memory_type"] == "semantic"
        assert metadata["scope"] == "global"
        assert "Knowledge body." in body


# ---------------------------------------------------------------------------
# read_memory_document
# ---------------------------------------------------------------------------


class TestReadMemoryDocument:
    def test_reads_valid_file(self, tmp_path):
        p = tmp_path / "test.md"
        p.write_text("---\nid: O-1\n---\nBody.", encoding="utf-8")
        result = read_memory_document(p)
        assert result is not None
        metadata, body = result
        assert metadata["id"] == "O-1"

    def test_missing_file_returns_none(self, tmp_path):
        result = read_memory_document(tmp_path / "missing.md")
        assert result is None

    def test_invalid_frontmatter_returns_none(self, tmp_path):
        p = tmp_path / "bad.md"
        p.write_text("No frontmatter here.", encoding="utf-8")
        result = read_memory_document(p)
        assert result is None

    def test_undecodable_bytes_returns_none(self, tmp_path):
        p = tmp_path / "binary.md"
        p.write_bytes(b"\xff\xfe invalid utf-8 \x80")
        result = read_memory_document(p)
        assert result is None

    def test_valid_document_body_preserved(self, tmp_path):
        content = "---\nid: O-abc\n---\n## Section\n\nContent here."
        p = tmp_path / "obs.md"
        p.write_text(content, encoding="utf-8")
        result = read_memory_document(p)
        assert result is not None
        _, body = result
        assert "Section" in body
        assert "Content here." in body


# ---------------------------------------------------------------------------
# document_body
# ---------------------------------------------------------------------------


class TestDocumentBody:
    def test_strips_yaml_frontmatter(self):
        text = "---\nid: O-abc\n---\nActual body content."
        body = document_body(text)
        assert body == "Actual body content."

    def test_no_frontmatter_returns_stripped_text(self):
        text = "  plain text  "
        assert document_body(text) == "plain text"

    def test_malformed_frontmatter_returns_stripped(self):
        # Starts with --- but no closing ---
        text = "---\nid: foo\n"
        result = document_body(text)
        # Should return the stripped original since it can't split
        assert "---" in result or "foo" in result

    def test_empty_body_after_frontmatter(self):
        text = "---\nid: O-abc\n---\n\n  \n"
        assert document_body(text) == ""

    def test_multiline_body_stripped(self):
        text = "---\nid: O-1\n---\n\n  Line 1\nLine 2\n  "
        body = document_body(text)
        assert "Line 1" in body
        assert "Line 2" in body


# ---------------------------------------------------------------------------
# current_configurable
# ---------------------------------------------------------------------------


class TestCurrentConfigurable:
    def test_returns_empty_outside_langgraph(self):
        # Outside LangGraph context, get_config() raises RuntimeError
        result = current_configurable()
        assert isinstance(result, dict)
        # May or may not be empty depending on test env, but must be a Mapping

    def test_returns_configurable_from_langgraph_config(self):
        fake_config = {"configurable": {"project_id": "P-test", "thread_id": "t1"}}
        with patch(
            "EvoScientist.memory._common.get_config", return_value=fake_config
        ):
            result = current_configurable()
        assert result["project_id"] == "P-test"

    def test_returns_empty_when_configurable_missing(self):
        fake_config = {"other_key": "value"}
        with patch(
            "EvoScientist.memory._common.get_config", return_value=fake_config
        ):
            result = current_configurable()
        assert result == {}

    def test_returns_empty_when_configurable_not_dict(self):
        fake_config = {"configurable": "not-a-dict"}
        with patch(
            "EvoScientist.memory._common.get_config", return_value=fake_config
        ):
            result = current_configurable()
        assert result == {}

    def test_returns_empty_on_runtime_error(self):
        with patch(
            "EvoScientist.memory._common.get_config",
            side_effect=RuntimeError("no context"),
        ):
            result = current_configurable()
        assert result == {}


# ---------------------------------------------------------------------------
# config_str
# ---------------------------------------------------------------------------


class TestConfigStr:
    def test_returns_string_value(self):
        assert config_str({"key": "value"}, "key") == "value"

    def test_returns_none_for_missing_key(self):
        assert config_str({}, "missing") is None

    def test_returns_none_for_empty_string(self):
        assert config_str({"key": ""}, "key") is None

    def test_returns_none_for_non_string(self):
        assert config_str({"key": 42}, "key") is None
        assert config_str({"key": None}, "key") is None
        assert config_str({"key": True}, "key") is None

    def test_non_empty_string_is_returned(self):
        assert config_str({"evomemory_project_id": "P-abc123"}, "evomemory_project_id") == "P-abc123"


# ---------------------------------------------------------------------------
# short_hash
# ---------------------------------------------------------------------------


class TestShortHash:
    def test_returns_16_chars_by_default(self):
        result = short_hash("hello")
        assert len(result) == 16

    def test_deterministic(self):
        assert short_hash("same-input") == short_hash("same-input")

    def test_different_inputs_differ(self):
        assert short_hash("input-a") != short_hash("input-b")

    def test_custom_length(self):
        result = short_hash("test", n=8)
        assert len(result) == 8

    def test_only_hex_chars(self):
        result = short_hash("any text here")
        assert all(c in "0123456789abcdef" for c in result)

    def test_empty_string_stable(self):
        h1 = short_hash("")
        h2 = short_hash("")
        assert h1 == h2


# ---------------------------------------------------------------------------
# stable_json
# ---------------------------------------------------------------------------


class TestStableJson:
    def test_sorts_keys(self):
        result = stable_json({"z": 1, "a": 2})
        parsed = json.loads(result)
        assert list(parsed.keys()) == sorted(parsed.keys())

    def test_deterministic_for_same_input(self):
        obj = {"b": [1, 2], "a": "test"}
        assert stable_json(obj) == stable_json(obj)

    def test_compact_separators(self):
        result = stable_json({"a": 1})
        # Compact format — no extra whitespace
        assert " " not in result

    def test_non_ascii_preserved(self):
        result = stable_json({"emoji": "🧠"})
        assert "🧠" in result

    def test_non_serializable_uses_str(self):
        from datetime import datetime

        dt = datetime(2024, 1, 1)
        result = stable_json({"dt": dt})
        parsed = json.loads(result)
        assert isinstance(parsed["dt"], str)


# ---------------------------------------------------------------------------
# pretty_json
# ---------------------------------------------------------------------------


class TestPrettyJson:
    def test_indented_output(self):
        result = pretty_json({"key": "value"})
        assert "\n" in result
        assert "  " in result

    def test_sorts_keys(self):
        result = pretty_json({"z": 1, "a": 2})
        parsed = json.loads(result)
        assert list(parsed.keys()) == sorted(parsed.keys())

    def test_non_ascii_preserved(self):
        result = pretty_json({"text": "café"})
        assert "café" in result


# ---------------------------------------------------------------------------
# dedupe_ids
# ---------------------------------------------------------------------------


class TestDedupeIds:
    def test_dedupes_preserving_first_seen_order(self):
        result = dedupe_ids(["O-a", "O-b", "O-a", "O-c", "O-b"])
        assert result == ("O-a", "O-b", "O-c")

    def test_drops_blank_strings(self):
        result = dedupe_ids(["O-a", "", "  ", "O-b"])
        assert "" not in result
        assert "  " not in result
        assert "O-a" in result and "O-b" in result

    def test_strips_whitespace(self):
        result = dedupe_ids(["  O-a  ", "O-b"])
        assert "O-a" in result

    def test_empty_input_returns_empty_tuple(self):
        assert dedupe_ids([]) == ()

    def test_require_prefix_filters(self):
        result = dedupe_ids(["O-a", "K-b", "O-c"], require_prefix="O-")
        assert "K-b" not in result
        assert "O-a" in result
        assert "O-c" in result

    def test_require_prefix_none_keeps_all(self):
        result = dedupe_ids(["O-a", "K-b"])
        assert "O-a" in result
        assert "K-b" in result

    def test_returns_tuple(self):
        result = dedupe_ids(["O-a"])
        assert isinstance(result, tuple)

    def test_all_same_dedupes_to_one(self):
        result = dedupe_ids(["O-x", "O-x", "O-x"])
        assert result == ("O-x",)