"""Tests for EvoScientist/prompts.py."""

from EvoScientist.prompts import (
    DELEGATION_STRATEGY,
    EXPERIMENT_WORKFLOW,
    get_system_prompt,
)


class TestGetSystemPrompt:
    def test_returns_non_empty(self):
        result = get_system_prompt()
        assert isinstance(result, str)
        assert len(result) > 100

    def test_contains_workflow(self):
        result = get_system_prompt()
        assert "Experiment Workflow" in result

    def test_contains_delegation(self):
        result = get_system_prompt()
        assert "Sub-Agent Delegation" in result

    def test_no_numeric_limits(self):
        result = get_system_prompt()
        assert "{max_concurrent}" not in result
        assert "{max_iterations}" not in result

    def test_workflow_constant_not_empty(self):
        assert len(EXPERIMENT_WORKFLOW) > 0

    def test_delegation_no_placeholders(self):
        assert "{max_concurrent}" not in DELEGATION_STRATEGY
        assert "{max_iterations}" not in DELEGATION_STRATEGY

    def test_shell_guidelines_mention_timeout_limit(self):
        assert "300" in EXPERIMENT_WORKFLOW
        assert "124" in EXPERIMENT_WORKFLOW

    def test_shell_guidelines_mention_background(self):
        assert "background" in EXPERIMENT_WORKFLOW.lower()

    def test_contains_todays_date(self):
        from datetime import datetime

        expected = datetime.now().strftime("%Y-%m-%d")
        assert expected in get_system_prompt()
