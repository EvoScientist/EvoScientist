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
