"""Tests for EvoScientist/tools.py — only non-API tools."""

from EvoScientist.tools import think_tool


class TestThinkTool:
    def test_returns_confirmation(self):
        result = think_tool.invoke({"reflection": "I need more data on topic X"})
        assert isinstance(result, str)
        assert "I need more data on topic X" in result

    def test_reflection_recorded(self):
        result = think_tool.invoke({"reflection": "gap analysis"})
        assert "Reflection recorded" in result

    def test_empty_reflection(self):
        result = think_tool.invoke({"reflection": ""})
        assert "Reflection recorded" in result

    def test_docstring_has_resource_dimension(self):
        assert "resource" in think_tool.description.lower()
        assert "300" in think_tool.description
