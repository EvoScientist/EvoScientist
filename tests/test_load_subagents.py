"""Tests for ``EvoScientist.utils.load_subagents``.

Focused on schema-validation paths that are easy to silently misuse from
yaml — primarily the ``async:`` flag type check that prevents quoted-string
or integer values from being misinterpreted as booleans.
"""

from __future__ import annotations

import textwrap

import pytest

from EvoScientist.utils import load_subagents


def _write_yaml(tmp_path, name: str, body: str):
    """Write ``body`` to ``tmp_path/name`` and return the directory path."""
    (tmp_path / name).write_text(textwrap.dedent(body))
    return tmp_path


def test_async_flag_accepts_real_bool(tmp_path):
    """``async: true`` (real yaml boolean) is accepted and carried through."""
    config_path = _write_yaml(
        tmp_path,
        "writing.yaml",
        """
        writing-agent:
          description: Drafts reports
          system_prompt: ""
          tools: []
          async: true
        """,
    )
    subs = load_subagents(config_path, tool_registry={})
    assert len(subs) == 1
    assert subs[0]["name"] == "writing-agent"
    assert subs[0]["_async"] is True


def test_async_flag_defaults_to_false_when_omitted(tmp_path):
    """No ``async:`` field → ``_async`` defaults to False."""
    config_path = _write_yaml(
        tmp_path,
        "planner.yaml",
        """
        planner-agent:
          description: Plans experiments
          system_prompt: ""
          tools: []
        """,
    )
    subs = load_subagents(config_path, tool_registry={})
    assert subs[0]["_async"] is False


def test_async_flag_rejects_quoted_string(tmp_path):
    """``async: "false"`` (quoted) is a real user trap — bool("false") is True.

    Without the explicit isinstance check, this would silently flip the agent
    into async mode. We require the validator to fail loud instead.
    """
    config_path = _write_yaml(
        tmp_path,
        "bad.yaml",
        """
        bad-agent:
          description: ""
          system_prompt: ""
          tools: []
          async: "false"
        """,
    )
    with pytest.raises(ValueError, match=r"'async' must be a boolean"):
        load_subagents(config_path, tool_registry={})


def test_async_flag_rejects_integer(tmp_path):
    """``async: 1`` is also rejected — yaml integers are not booleans."""
    config_path = _write_yaml(
        tmp_path,
        "bad.yaml",
        """
        bad-agent:
          description: ""
          system_prompt: ""
          tools: []
          async: 1
        """,
    )
    with pytest.raises(ValueError, match=r"'async' must be a boolean"):
        load_subagents(config_path, tool_registry={})


def test_async_flag_error_includes_agent_name(tmp_path):
    """Error message must include the offending agent name for triage."""
    config_path = _write_yaml(
        tmp_path,
        "bad.yaml",
        """
        my-bad-agent:
          description: ""
          system_prompt: ""
          tools: []
          async: "yes"
        """,
    )
    with pytest.raises(ValueError, match=r"my-bad-agent"):
        load_subagents(config_path, tool_registry={})


def test_non_dict_spec_raises(tmp_path):
    """Yaml entries that aren't mappings must fail loud, not be silently dropped.

    Previously ``_build_one`` had a ``if not isinstance(spec, dict): continue``
    fallback that swallowed malformed entries — users would see their agent
    quietly disappear with no error. Now caught during the merge loop.
    """
    config_path = _write_yaml(
        tmp_path,
        "bad.yaml",
        """
        bad-agent: 123
        """,
    )
    with pytest.raises(ValueError, match=r"must map to a spec dict"):
        load_subagents(config_path, tool_registry={})


def test_non_dict_spec_error_includes_filename_and_name(tmp_path):
    """Error must surface BOTH the offending file path and agent name."""
    config_path = _write_yaml(
        tmp_path,
        "weird.yaml",
        """
        weird-agent: "just a string"
        """,
    )
    with pytest.raises(ValueError, match=r"weird\.yaml.*weird-agent"):
        load_subagents(config_path, tool_registry={})


def test_missing_tool_on_sync_subagent_logs_warning(tmp_path, caplog):
    """Sync sub-agents with a tool missing from the registry log at WARNING.

    Sync sub-agents run in-process under the main agent and rely on the
    in-process registry to wire every tool they declare. A missing tool
    IS a genuine degradation — surfaces it as a warning.
    """
    config_path = _write_yaml(
        tmp_path,
        "planner.yaml",
        """
        planner-agent:
          description: Plans experiments
          system_prompt: ""
          tools: [nonexistent_tool]
        """,
    )
    with caplog.at_level("DEBUG", logger="EvoScientist.utils"):
        subs = load_subagents(config_path, tool_registry={})
    assert subs[0]["_async"] is False
    assert subs[0]["tools"] == []
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("nonexistent_tool" in r.getMessage() for r in warnings)


def test_missing_tool_on_async_subagent_logs_debug_when_swap_pending(tmp_path, caplog):
    """Sync in-process callers pass ``async_swap_pending=True`` — a tool
    missing for an ``async: true`` spec logs at DEBUG because the async
    graph's own registry will re-resolve it downstream
    (``subagents/_factory.py``).

    Regression guard for the spurious startup WARNING that pre-fix logs
    fired on every ``EvoSci`` startup even though the tool was wired at
    runtime by the deployed graph.
    """
    config_path = _write_yaml(
        tmp_path,
        "scheduler.yaml",
        """
        scheduler:
          description: Fires on cron
          system_prompt: ""
          tools: [nonexistent_tool]
          async: true
        """,
    )
    with caplog.at_level("DEBUG", logger="EvoScientist.utils"):
        subs = load_subagents(config_path, tool_registry={}, async_swap_pending=True)
    assert subs[0]["_async"] is True
    assert subs[0]["tools"] == []
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert not any("nonexistent_tool" in r.getMessage() for r in warnings)
    debugs = [r for r in caplog.records if r.levelname == "DEBUG"]
    assert any(
        "nonexistent_tool" in r.getMessage() and "async graph" in r.getMessage()
        for r in debugs
    )


def test_missing_tool_on_async_subagent_logs_warning_at_terminal_registry(
    tmp_path, caplog
):
    """When ``async_swap_pending`` is False (the default), the caller IS the
    terminal registry — the factory boundary
    (``subagents/_factory.build_async_subagent_graph``). A tool missing for
    an ``async: true`` spec is a genuine typo that won't be resolved anywhere
    downstream, so log at WARNING.

    Without this guard, an earlier version of the fix (unconditional DEBUG for
    every async spec) silently hid factory-boundary typos. Reviewer flagged
    this as the last remaining gap.
    """
    config_path = _write_yaml(
        tmp_path,
        "scheduler.yaml",
        """
        scheduler:
          description: Fires on cron
          system_prompt: ""
          tools: [nonexistent_tool]
          async: true
        """,
    )
    with caplog.at_level("DEBUG", logger="EvoScientist.utils"):
        # Default: async_swap_pending=False → factory-boundary semantics.
        subs = load_subagents(config_path, tool_registry={})
    assert subs[0]["_async"] is True
    assert subs[0]["tools"] == []
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("nonexistent_tool" in r.getMessage() for r in warnings)
    debugs = [r for r in caplog.records if r.levelname == "DEBUG"]
    assert not any(
        "nonexistent_tool" in r.getMessage() and "async graph" in r.getMessage()
        for r in debugs
    )
