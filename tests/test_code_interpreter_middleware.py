"""Regression tests for the code_interpreter PTC allowlist.

langchain-quickjs >=0.3 reserves the ``task`` sub-agent dispatch tool as the
top-level REPL global and raises ``ValueError`` if ``task`` appears in the
``ptc`` allowlist. EvoScientist must therefore keep ``task`` out of the
allowlist (``task()`` stays reachable as the REPL global, with responseSchema).
"""

from __future__ import annotations

import pytest

from EvoScientist.middleware.code_interpreter import (
    _DEFAULT_PTC_ALLOWLIST,
    create_code_interpreter_middleware,
)


def test_task_excluded_from_ptc_allowlist():
    # Exposing `task` via ptc raises on quickjs >=0.3 — it is the REPL global.
    assert "task" not in _DEFAULT_PTC_ALLOWLIST


def test_async_dispatch_tools_remain_in_allowlist():
    # Guard against over-deletion: async fan-out is the killer PTC use case.
    for name in ("start_async_task", "check_async_task", "list_async_tasks"):
        assert name in _DEFAULT_PTC_ALLOWLIST


def test_filter_tools_for_ptc_accepts_default_allowlist():
    # End-to-end guard: the live quickjs filter must accept our allowlist even
    # when a `task` tool is present in the agent toolset.
    _ptc = pytest.importorskip("langchain_quickjs._ptc")
    from langchain_core.tools import tool

    @tool
    def task(description: str) -> str:
        """dummy sub-agent dispatch tool"""
        return "ok"

    _ptc.filter_tools_for_ptc(
        [task], _DEFAULT_PTC_ALLOWLIST, self_tool_name="code_interpreter"
    )


def test_create_code_interpreter_middleware_builds():
    assert create_code_interpreter_middleware() is not None
