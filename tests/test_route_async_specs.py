"""Tests for the AsyncSubAgent → EvoAsyncSubAgentMiddleware routing helper.

Covers:
- ``_route_async_specs_through_evo_middleware`` splits AsyncSubAgent specs
  out of the ``subs`` list and folds them into the base middleware.
- ``build_expert_async_subagent_specs`` gives every installed expert a
  background reach, gated only on the async-enable flag + langgraph dev
  reachability.
- ``build_expert_subagent_specs`` (in-turn fold-in) covers the same
  experts, so each name holds both reaches without colliding.
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


def _skill(name: str) -> SkillInfo:
    return SkillInfo(
        name=name,
        description=f"{name} description",
        path=Path("/tmp/does-not-matter"),
        source="builtin",
        type="expert",
        role=f"{name} role",
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
            return_value=[_skill("literature-review")],
        ):
            specs = build_expert_async_subagent_specs(cfg=cfg)
        assert specs == []

    def test_empty_when_langgraph_dev_unreachable(self):
        cfg = SimpleNamespace(enable_async_subagents=True, langgraph_dev_port=6174)
        with (
            patch(
                "EvoScientist.tools.skills_manager.list_expert_skills",
                return_value=[_skill("literature-review")],
            ),
            patch(
                "EvoScientist.langgraph_dev.manager.is_async_subagents_available",
                return_value=False,
            ),
        ):
            specs = build_expert_async_subagent_specs(cfg=cfg)
        assert specs == []

    def test_every_expert_gets_a_background_reach(self):
        """No classification: every installed expert becomes an AsyncSubAgent spec."""
        cfg = SimpleNamespace(enable_async_subagents=True, langgraph_dev_port=6174)
        skills = [
            _skill("idea-brainstorm"),
            _skill("literature-review"),
            _skill("panel-expert"),
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
        assert {s["name"] for s in specs} == {
            "idea-brainstorm",
            "literature-review",
            "panel-expert",
        }
        for spec in specs:
            assert spec["graph_id"] == "expert-container-async"
            assert spec["is_expert"] is True
            assert "http://localhost:6174" in spec["url"]

    def test_empty_body_experts_skipped(self):
        """Empty-body async experts are filtered out at spec-build time so
        ``start_async_task``'s tool schema never advertises a broken skill.
        Mirrors the sync fold-in in
        ``expert_container.py::build_expert_subagent_specs``."""
        cfg = SimpleNamespace(enable_async_subagents=True, langgraph_dev_port=6174)
        skills = [
            _skill("literature-review"),  # normal body from _skill()
            _skill("empty-persona"),
        ]
        # Second skill has no body — dataclass field default is ``""``, but
        # helper sets it to "body\n" — override to empty.
        skills[1].body = ""
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
        assert [s["name"] for s in specs] == ["literature-review"]

    def test_agents_md_expert_registered_and_gated_on_its_own_file(self):
        """AGENTS.md experts reach async dispatch, and their gate is AGENTS.md.

        The empty-persona gate has to follow the skill's contract: a healthy
        SKILL.md must not vouch for a skill whose actor definition is blank.
        """
        cfg = SimpleNamespace(enable_async_subagents=True, langgraph_dev_port=6174)
        healthy = _skill("paper-review")
        healthy.expert_source = "agents_md"
        healthy.agents_body = "## Persona\n\nYou are an adversarial reviewer.\n"
        blank_actor = _skill("blank-actor")
        blank_actor.expert_source = "agents_md"
        blank_actor.agents_body = "   \n"  # SKILL.md body is fine; actor isn't
        with (
            patch(
                "EvoScientist.tools.skills_manager.list_expert_skills",
                return_value=[healthy, blank_actor],
            ),
            patch(
                "EvoScientist.langgraph_dev.manager.is_async_subagents_available",
                return_value=True,
            ),
        ):
            specs = build_expert_async_subagent_specs(cfg=cfg)
        assert [s["name"] for s in specs] == ["paper-review"]

    def test_reserved_name_collision_skipped(self, caplog):
        """A skill named after a yaml async sub-agent (or ``general-purpose``)
        must skip async-dispatch registration with a warning, not raise. Without
        this guard ``AsyncSubAgentMiddleware.__init__`` would ``ValueError:
        Duplicate async subagent names`` on the merged spec list and kill CLI
        startup — see reviewer thread on PR #391."""
        import logging

        cfg = SimpleNamespace(enable_async_subagents=True, langgraph_dev_port=6174)
        skills = [
            _skill("writing-agent"),  # collides with yaml async agent
            _skill("literature-review"),
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
            patch(
                "EvoScientist.subagents.expert_container._reserved_subagent_names",
                return_value=frozenset({"writing-agent", "general-purpose"}),
            ),
            caplog.at_level(
                logging.WARNING,
                logger="EvoScientist.subagents.expert_container_async",
            ),
        ):
            specs = build_expert_async_subagent_specs(cfg=cfg)
        assert [s["name"] for s in specs] == ["literature-review"]
        assert any(
            "writing-agent" in r.message and "collides" in r.message
            for r in caplog.records
        )

    def test_workspace_duplicate_name_skipped(self, caplog):
        """Two workspace-tier expert skills sharing a frontmatter ``name`` must
        register only the first — the workspace listing uses
        ``check_seen=False`` so both survive to this point. Without a local
        seen-set the second would collide inside
        ``AsyncSubAgentMiddleware.__init__``."""
        import logging

        cfg = SimpleNamespace(enable_async_subagents=True, langgraph_dev_port=6174)
        skills = [
            _skill("literature-review"),
            _skill("literature-review"),  # duplicate name
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
            patch(
                "EvoScientist.subagents.expert_container._reserved_subagent_names",
                return_value=frozenset({"general-purpose"}),
            ),
            caplog.at_level(
                logging.WARNING,
                logger="EvoScientist.subagents.expert_container_async",
            ),
        ):
            specs = build_expert_async_subagent_specs(cfg=cfg)
        # Only the first `literature-review` survives.
        assert [s["name"] for s in specs] == ["literature-review"]
        assert any(
            "literature-review" in r.message and "collides" in r.message
            for r in caplog.records
        )


# =============================================================================
# build_expert_subagent_specs (in-turn side) — same experts, second reach
# =============================================================================


class TestBuildExpertSubagentSpecsCoversEveryExpert:
    """The in-turn fold-in emits a spec for every expert, async ones included.

    Sharing a name across the two registries is intentional. They land on
    different tools with separate schemas (``task`` vs
    ``start_async_task``), and deepagents' duplicate-name check is scoped to
    the async list alone, so one expert holding both reaches never collides.
    """

    def test_every_expert_gets_an_in_turn_reach(self):
        skills = [
            _skill("idea-brainstorm"),
            _skill("literature-review"),
            _skill("panel-expert"),
        ]
        with patch(
            "EvoScientist.tools.skills_manager.list_expert_skills",
            return_value=skills,
        ):
            specs = build_expert_subagent_specs(tool_registry={})
        assert {s["name"] for s in specs} == {
            "idea-brainstorm",
            "literature-review",
            "panel-expert",
        }


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
                return_value=[_skill("literature-review")],
            ),
            patch(
                "EvoScientist.langgraph_dev.manager.is_async_subagents_available",
                return_value=True,
            ),
        ):
            result = _route_async_specs_through_evo_middleware(
                subs, middleware, cfg=cfg
            )
        # sync-a stays; middleware got the expert spec, and an
        # AsyncWatcherMiddleware was installed so expert launches spawn
        # completion watchers (previously the watcher's client cache had no
        # entry for the expert name, KeyErrored on `get_async`, and silently
        # dropped the notification).
        from EvoScientist.middleware.async_watcher import AsyncWatcherMiddleware

        assert [s["name"] for s in result] == ["sync-a"]
        assert len(middleware) == 2
        evo_mw = next(
            m for m in middleware if isinstance(m, EvoAsyncSubAgentMiddleware)
        )
        watcher_mw = next(
            m for m in middleware if isinstance(m, AsyncWatcherMiddleware)
        )
        # The middleware's start tool schema advertises literature-review.
        start = next(t for t in evo_mw.tools if t.name == "start_async_task")
        assert "literature-review" in start.description
        # The watcher's client cache knows how to construct a client for the
        # expert so the completion nudge can spawn.
        assert "literature-review" in watcher_mw._clients._agents

    def test_watcher_cache_extends_when_yaml_watcher_preinstalled(self):
        """Default deployed shape — ``_maybe_swap_async_subagents`` installed
        ``AsyncWatcherMiddleware`` for a yaml async agent, then the routing
        helper extends the cache with expert specs. Without the extension
        branch, an expert completion nudge would KeyError on the watcher's
        ``get_async(<expert>)`` and silently drop the notification."""
        from EvoScientist.cli import async_notifier
        from EvoScientist.EvoScientist import _route_async_specs_through_evo_middleware
        from EvoScientist.middleware.async_watcher import AsyncWatcherMiddleware
        from EvoScientist.middleware.expert_async_subagent import (
            EvoAsyncSubAgentMiddleware,
        )

        yaml_async_spec = {
            "name": "writing-agent",
            "description": "std",
            "graph_id": "writing_agent",
            "url": "http://localhost:6174",
        }
        subs = [{"name": "sync-a", "system_prompt": ""}, yaml_async_spec]
        # Simulate the state after ``_maybe_swap_async_subagents``: watcher is
        # already installed and carries the yaml async agent.
        middleware: list = [
            AsyncWatcherMiddleware(
                {"writing-agent": yaml_async_spec}, notifier=async_notifier
            )
        ]
        cfg = self._cfg(enable_async=True)
        with (
            patch(
                "EvoScientist.tools.skills_manager.list_expert_skills",
                return_value=[_skill("literature-review")],
            ),
            patch(
                "EvoScientist.langgraph_dev.manager.is_async_subagents_available",
                return_value=True,
            ),
            patch(
                "EvoScientist.subagents.expert_container._reserved_subagent_names",
                return_value=frozenset({"general-purpose"}),
            ),
        ):
            result = _route_async_specs_through_evo_middleware(
                subs, middleware, cfg=cfg
            )
        # `writing-agent` (graph_id-carrying) stripped from subs; sync-a stays.
        assert [s["name"] for s in result] == ["sync-a"]
        evo_mw = next(
            m for m in middleware if isinstance(m, EvoAsyncSubAgentMiddleware)
        )
        watcher_mw = next(
            m for m in middleware if isinstance(m, AsyncWatcherMiddleware)
        )
        # Both the yaml async agent and the expert reach the start-task schema.
        start = next(t for t in evo_mw.tools if t.name == "start_async_task")
        assert "writing-agent" in start.description
        assert "literature-review" in start.description
        # The pre-existing watcher was extended in place — both names route.
        assert "writing-agent" in watcher_mw._clients._agents
        assert "literature-review" in watcher_mw._clients._agents
