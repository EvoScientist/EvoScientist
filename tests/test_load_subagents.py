"""Tests for ``EvoScientist.utils.load_subagents``.

Focused on schema-validation paths that are easy to silently misuse from
yaml — primarily the ``async:`` flag type check that prevents quoted-string
or integer values from being misinterpreted as booleans.
"""

from __future__ import annotations

import textwrap

import pytest

from EvoScientist.utils import load_subagents, resolve_subagent_tools


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
    subs = load_subagents(config_path)
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
    subs = load_subagents(config_path)
    assert subs[0]["_async"] is False


def test_tool_names_are_deferred_until_the_spec_is_selected(tmp_path, caplog):
    selected_tool = object()
    config_path = _write_yaml(
        tmp_path,
        "agents.yaml",
        """
        selected-agent:
          tools: [available]
        remote-agent:
          tools: [remote_only]
          async: true
        """,
    )

    subs = load_subagents(config_path)

    assert not caplog.records
    assert subs[0]["tools"] == []
    assert subs[0]["_tool_names"] == ["available"]
    resolve_subagent_tools(subs[0], {"available": selected_tool})
    assert subs[0]["tools"] == [selected_tool]
    assert "_tool_names" not in subs[0]


def test_resolve_subagent_tools_preserves_injected_tools():
    injected = object()
    selected_tool = object()
    subagent = {
        "name": "selected-agent",
        "tools": [injected],
        "_tool_names": ["available"],
    }

    resolve_subagent_tools(subagent, {"available": selected_tool})

    assert subagent["tools"] == [injected, selected_tool]
    assert "_tool_names" not in subagent


def test_agent_without_tools_preserves_parent_tool_inheritance(tmp_path):
    config_path = _write_yaml(
        tmp_path,
        "general.yaml",
        """
        general-purpose:
          description: Handles general tasks
          system_prompt: ""
        """,
    )

    subagent = load_subagents(config_path)[0]
    resolve_subagent_tools(subagent, {"available": object()})

    assert "tools" not in subagent
    assert "_tool_names" not in subagent


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
        load_subagents(config_path)


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
        load_subagents(config_path)


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
        load_subagents(config_path)


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
        load_subagents(config_path)


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
        load_subagents(config_path)


def test_missing_tool_on_sync_subagent_warns_at_resolution(tmp_path, caplog):
    """A selected sync spec warns when its terminal registry lacks a tool."""
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
        subs = load_subagents(config_path)
    assert subs[0]["_async"] is False
    assert subs[0]["tools"] == []
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert not warnings

    resolve_subagent_tools(subs[0], {})
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("nonexistent_tool" in r.getMessage() for r in warnings)


def test_missing_tool_on_async_subagent_is_deferred_without_logging(tmp_path, caplog):
    """Async tool names do not emit warnings against the caller registry."""
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
        subs = load_subagents(config_path)
    assert subs[0]["_async"] is True
    assert subs[0]["tools"] == []
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert not any("nonexistent_tool" in r.getMessage() for r in warnings)
    debugs = [r for r in caplog.records if r.levelname == "DEBUG"]
    assert not any("nonexistent_tool" in r.getMessage() for r in debugs)
    assert subs[0]["_tool_names"] == ["nonexistent_tool"]


def test_missing_async_tool_warns_at_terminal_resolution(tmp_path, caplog):
    """The selected async graph warns against its terminal registry."""
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
        subs = load_subagents(config_path)
        resolve_subagent_tools(subs[0], {})
    assert subs[0]["_async"] is True
    assert subs[0]["tools"] == []
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("nonexistent_tool" in r.getMessage() for r in warnings)
    debugs = [r for r in caplog.records if r.levelname == "DEBUG"]
    assert not any(
        "nonexistent_tool" in r.getMessage() and "async graph" in r.getMessage()
        for r in debugs
    )
