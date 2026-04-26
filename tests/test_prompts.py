"""Tests for EvoScientist/prompts.py."""

from EvoScientist.prompts import (
    DELEGATION_STRATEGY,
    EVOSCIENTIST_IDENTITY,
    EXPERIMENT_WORKFLOW,
    REPORT_TEMPLATE,
    SHELL_GUIDELINES,
    WRITING_GUIDELINES,
    get_system_prompt,
)


class TestGetSystemPrompt:
    def test_returns_non_empty(self):
        result = get_system_prompt()
        assert isinstance(result, str)
        assert len(result) > 100

    def test_contains_identity(self):
        result = get_system_prompt()
        assert "EvoScientist" in result
        assert "self-evolving" in result

    def test_contains_workflow(self):
        result = get_system_prompt()
        assert "Experiment Workflow" in result

    def test_contains_report_template(self):
        result = get_system_prompt()
        assert "Experiment Report Template" in result

    def test_contains_writing_guidelines(self):
        result = get_system_prompt()
        assert "Writing Guidelines" in result

    def test_contains_shell_guidelines(self):
        result = get_system_prompt()
        assert "Shell Execution Guidelines" in result

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

    def test_section_ordering(self):
        """Identity must precede workflow; workflow must precede delegation."""
        result = get_system_prompt()
        idx_identity = result.find("# Identity")
        idx_workflow = result.find("# Experiment Workflow")
        idx_delegation = result.find("# Sub-Agent Delegation")
        assert 0 <= idx_identity < idx_workflow < idx_delegation

    def test_does_not_contain_static_date(self):
        """Date is injected per-turn by EvoMemoryMiddleware, not baked into static prompt.

        Static prompt must stay byte-stable across midnight so the cache prefix
        survives. See EvoMemoryMiddleware.modify_request for runtime injection.
        """
        import re

        result = get_system_prompt()
        # No literal "Today's date is YYYY-MM-DD." in the static prompt.
        assert not re.search(r"Today's date is \d{4}-\d{2}-\d{2}", result)

    def test_mentions_skill_manager_for_discovery(self):
        """Agent must know it can browse/install skills from the EvoSkills catalog."""
        result = get_system_prompt()
        assert "skill_manager" in result
        assert "EvoSkills" in result

    def test_no_stale_memory_path_singular(self):
        """Backend route is `/memories/`, not `/memory/`. Catch silent-bug regressions.

        Anything sent to `/memory/...` falls through to CustomSandboxBackend
        (workspace files), bypassing the persistent FilesystemBackend that
        owns ideation-memory.md / experiment-memory.md.
        """
        result = get_system_prompt()
        # `/memory/` as a filesystem path (after a backtick or whitespace, before
        # an alpha char or another /). Excludes word-list usages like
        # "context/memory/web search".
        import re

        assert not re.search(r"[\s`]/memory/[a-zA-Z]", result), (
            "Found `/memory/<file>` in system prompt — should be `/memories/<file>`"
        )


class TestEvoScientistIdentity:
    def test_constant_not_empty(self):
        assert len(EVOSCIENTIST_IDENTITY) > 0

    def test_states_role(self):
        assert "You are EvoScientist" in EVOSCIENTIST_IDENTITY

    def test_mentions_human_on_the_loop_paradigm(self):
        # Behavioral cue: agent should know it isn't asking permission for every action
        assert "on-the-loop" in EVOSCIENTIST_IDENTITY


class TestReportTemplate:
    def test_constant_not_empty(self):
        assert len(REPORT_TEMPLATE) > 0

    def test_contains_six_sections(self):
        # Match the six recommended sections (lowercased to be tolerant of phrasing)
        body = REPORT_TEMPLATE.lower()
        for section in (
            "summary",
            "experiment plan",
            "setup",
            "baselines",
            "results",
            "limitations",
        ):
            assert section in body, section

    def test_not_duplicated_in_workflow_step5(self):
        """Step 5 should reference REPORT_TEMPLATE, not redefine the schema."""
        # Positive: Step 5 must actually reference the report template.
        assert "Experiment Report Template" in EXPERIMENT_WORKFLOW
        # Negative: section headers unique to REPORT_TEMPLATE must not appear
        # inlined inside EXPERIMENT_WORKFLOW (would mean the schema was
        # duplicated again, regardless of indentation style).
        assert "Baselines and comparisons" not in EXPERIMENT_WORKFLOW


class TestWritingGuidelines:
    def test_constant_not_empty(self):
        assert len(WRITING_GUIDELINES) > 0

    def test_mentions_first_person_avoidance(self):
        assert (
            "first-person" in WRITING_GUIDELINES.lower()
            or "I ..." in WRITING_GUIDELINES
        )


class TestShellGuidelines:
    def test_constant_not_empty(self):
        assert len(SHELL_GUIDELINES) > 0

    def test_mentions_timeout_limit(self):
        assert "300" in SHELL_GUIDELINES
        assert "124" in SHELL_GUIDELINES

    def test_mentions_background_execution(self):
        assert "background" in SHELL_GUIDELINES.lower()

    def test_not_duplicated_in_workflow(self):
        """SHELL_GUIDELINES content should live ONLY in its own constant."""
        # Sentinel phrase unique to SHELL_GUIDELINES
        assert "Sandbox limits" not in EXPERIMENT_WORKFLOW
