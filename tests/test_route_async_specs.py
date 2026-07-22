"""Tests for the AsyncSubAgent → EvoAsyncSubAgentMiddleware routing helper.

Covers:
- ``_route_async_specs_through_evo_middleware`` splits AsyncSubAgent specs
  out of the ``subs`` list and folds them into the base middleware.
- ``build_expert_async_subagent_specs`` filters by
  ``default_dispatch == "async"`` and respects the async-enable flag +
  langgraph dev reachability.
- ``build_expert_subagent_specs`` (sync fold-in) excludes async experts so
  a single skill never surfaces twice in the main agent's tool schema.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from EvoScientist.subagents.expert_container import build_expert_subagent_specs
from EvoScientist.subagents.expert_container_async import (
    build_expert_async_subagent_specs,
)
from EvoScientist.tools.skills_manager import SkillInfo


def _skill(name: str, dispatch: str) -> SkillInfo:
    return SkillInfo(
        name=name,
        description=f"{name} description",
        path=Path("/tmp/does-not-matter"),
        source="builtin",
        type="expert",
        role=f"{name} role",
        default_dispatch=dispatch,
        body="body\n",
    )


# =============================================================================
# build_expert_async_subagent_specs
# =============================================================================


class TestBuildExpertAsyncSubagentSpecs:
    def test_empty_when_async_disabled(self):
        cfg = SimpleNamespace(enable_async_subagents=False)
        with patch(
            "EvoScientist.tools.skills_manager.list_expert_skills",
            return_value=[_skill("literature-review", "async")],
        ):
            specs = build_expert_async_subagent_specs(cfg=cfg)
        assert specs == []

    def test_empty_when_langgraph_dev_unreachable(self):
        cfg = SimpleNamespace(enable_async_subagents=True, langgraph_dev_port=6174)
        with (
            patch(
                "EvoScientist.tools.skills_manager.list_expert_skills",
                return_value=[_skill("literature-review", "async")],
            ),
            patch(
                "EvoScientist.langgraph_dev.manager.is_async_subagents_available",
                return_value=False,
            ),
        ):
            specs = build_expert_async_subagent_specs(cfg=cfg)
        assert specs == []

    def test_filters_by_default_dispatch(self):
        """Only ``default_dispatch: async`` skills become AsyncSubAgent specs."""
        cfg = SimpleNamespace(enable_async_subagents=True, langgraph_dev_port=6174)
        skills = [
            _skill("idea-brainstorm", "sync"),
            _skill("literature-review", "async"),
            _skill("panel-expert", "panel"),
        ]
        with (
            patch(
                "EvoScientist.tools.skills_manager.list_expert_skills",
                return_value=skills,
            ),
            patch(
                "EvoScientist.langgraph_dev.manager.is_async_subagents_available",
                return_value=True,
            ),
        ):
            specs = build_expert_async_subagent_specs(cfg=cfg)
        assert len(specs) == 1
        assert specs[0]["name"] == "literature-review"
        assert specs[0]["graph_id"] == "expert-container-async"
        assert specs[0]["is_expert"] is True
        assert "http://localhost:6174" in specs[0]["url"]


# =============================================================================
# build_expert_subagent_specs (sync side) — must exclude async experts
# =============================================================================


class TestBuildExpertSubagentSpecsExcludesAsync:
    """The sync fold-in must not emit specs for ``default_dispatch: async`` skills.

    A skill in both lists would produce two competing tool schemas — one
    under ``task(subagent_type='<name>')`` and one under
    ``start_async_task(subagent_type='<name>')`` — from the main agent's
    perspective. Ambiguous. The partition is: async goes async, everything
    else goes sync.
    """

    def test_async_expert_skipped_by_sync_fold_in(self):
        skills = [
            _skill("idea-brainstorm", "sync"),
            _skill("literature-review", "async"),
            _skill("panel-expert", "panel"),
        ]
        with patch(
            "EvoScientist.tools.skills_manager.list_expert_skills",
            return_value=skills,
        ):
            specs = build_expert_subagent_specs(tool_registry={})
        names = [s["name"] for s in specs]
        assert "idea-brainstorm" in names
        assert "panel-expert" in names
        assert "literature-review" not in names


# =============================================================================
# _route_async_specs_through_evo_middleware
# =============================================================================


class TestRouteAsyncSpecs:
    """The routing helper splits AsyncSubAgent specs from ``subs`` and
    hands them to ``EvoAsyncSubAgentMiddleware``. Verifies:
    - Sync subagents pass through untouched.
    - AsyncSubAgent specs are stripped from the returned ``subs``.
    - Expert async specs (from ``build_expert_async_subagent_specs``) are
      merged in.
    - The middleware is appended to ``base_middleware`` only when there
      are async specs (either standard or expert).
    """

    def _cfg(self, *, enable_async: bool = True, port: int = 6174):
        return SimpleNamespace(
            enable_async_subagents=enable_async, langgraph_dev_port=port
        )

    def test_sync_subagents_pass_through(self):
        from EvoScientist.EvoScientist import _route_async_specs_through_evo_middleware

        subs = [{"name": "sync-a", "system_prompt": ""}]
        middleware: list = []
        # Disable async path via cfg + patched reachability.
        with patch(
            "EvoScientist.langgraph_dev.manager.is_async_subagents_available",
            return_value=False,
        ):
            result = _route_async_specs_through_evo_middleware(
                subs, middleware, cfg=self._cfg(enable_async=False)
            )
        assert result == [{"name": "sync-a", "system_prompt": ""}]
        assert middleware == []  # no async → no middleware added

    def test_async_specs_moved_to_middleware(self):
        from EvoScientist.EvoScientist import _route_async_specs_through_evo_middleware
        from EvoScientist.middleware.expert_async_subagent import (
            EvoAsyncSubAgentMiddleware,
        )

        subs = [
            {"name": "sync-a", "system_prompt": ""},
            {
                "name": "writing-agent",
                "description": "std",
                "graph_id": "writing_agent",
                "url": "http://localhost:6174",
            },
        ]
        middleware: list = []
        # Disable expert-async fold-in to isolate the standard-spec routing.
        with patch(
            "EvoScientist.langgraph_dev.manager.is_async_subagents_available",
            return_value=False,
        ):
            result = _route_async_specs_through_evo_middleware(
                subs, middleware, cfg=self._cfg(enable_async=False)
            )
        # `writing-agent` stripped from subs (it has graph_id).
        assert [s["name"] for s in result] == ["sync-a"]
        # Middleware appended.
        assert len(middleware) == 1
        assert isinstance(middleware[0], EvoAsyncSubAgentMiddleware)

    def test_expert_async_specs_merged_in(self):
        from EvoScientist.EvoScientist import _route_async_specs_through_evo_middleware
        from EvoScientist.middleware.expert_async_subagent import (
            EvoAsyncSubAgentMiddleware,
        )

        subs = [{"name": "sync-a", "system_prompt": ""}]
        middleware: list = []
        cfg = self._cfg(enable_async=True)
        # Enable expert-async by patching skills list + reachability.
        with (
            patch(
                "EvoScientist.tools.skills_manager.list_expert_skills",
                return_value=[_skill("literature-review", "async")],
            ),
            patch(
                "EvoScientist.langgraph_dev.manager.is_async_subagents_available",
                return_value=True,
            ),
        ):
            result = _route_async_specs_through_evo_middleware(
                subs, middleware, cfg=cfg
            )
        # sync-a stays; middleware got the expert spec.
        assert [s["name"] for s in result] == ["sync-a"]
        assert len(middleware) == 1
        mw = middleware[0]
        assert isinstance(mw, EvoAsyncSubAgentMiddleware)
        # The middleware's start tool schema advertises literature-review.
        start = next(t for t in mw.tools if t.name == "start_async_task")
        assert "literature-review" in start.description
