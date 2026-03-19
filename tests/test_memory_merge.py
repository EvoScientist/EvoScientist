"""Tests for _merge_memory — backslash-safe regex replacement."""

import pytest

from EvoScientist.middleware.memory import DEFAULT_MEMORY_TEMPLATE, _merge_memory


class TestMergeMemoryBackslashSafety:
    """Ensure values containing regex-special sequences survive _merge_memory."""

    def test_backslash_n_preserved(self):
        """A value containing literal '\\n' must not become a newline."""
        extracted = {
            "user_profile": {"name": r"C:\new_user"},
        }
        result = _merge_memory(DEFAULT_MEMORY_TEMPLATE, extracted)
        assert r"C:\new_user" in result
        # The replacement must not introduce an actual newline inside the Name line
        for line in result.splitlines():
            if "**Name**" in line:
                assert r"C:\new_user" in line
                break
        else:
            pytest.fail("Name line not found")

    def test_backreference_preserved(self):
        r"""A value containing '\\1' must not be treated as a backreference."""
        extracted = {
            "user_profile": {"role": r"A\1B"},
        }
        result = _merge_memory(DEFAULT_MEMORY_TEMPLATE, extracted)
        assert r"A\1B" in result
        for line in result.splitlines():
            if "**Role**" in line:
                assert r"A\1B" in line
                break
        else:
            pytest.fail("Role line not found")

    def test_windows_path_preserved(self):
        r"""A Windows-style path must survive without corruption."""
        extracted = {
            "research_preferences": {
                "preferred_frameworks": r"C:\path\to\file",
            },
        }
        result = _merge_memory(DEFAULT_MEMORY_TEMPLATE, extracted)
        assert r"C:\path\to\file" in result

    def test_multiple_backslash_fields(self):
        """Multiple fields with backslashes all survive."""
        extracted = {
            "user_profile": {
                "name": r"user\name",
                "institution": r"MIT\Lab\42",
            },
            "research_preferences": {
                "hardware": r"GPU\0",
            },
        }
        result = _merge_memory(DEFAULT_MEMORY_TEMPLATE, extracted)
        assert r"user\name" in result
        assert r"MIT\Lab\42" in result
        assert r"GPU\0" in result

    def test_plain_value_still_works(self):
        """Sanity check: normal values without backslashes work fine."""
        extracted = {
            "user_profile": {"name": "Alice", "role": "Researcher"},
        }
        result = _merge_memory(DEFAULT_MEMORY_TEMPLATE, extracted)
        assert "- **Name**: Alice" in result
        assert "- **Role**: Researcher" in result
