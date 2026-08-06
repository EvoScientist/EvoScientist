"""The cue's expert set must match the set the seated agent was built from.

Every other test of this machinery substitutes a fake for one of the two
things that have to agree: the tree walk, or the token read that decides
whether to rebuild. This one patches only the data source
(``list_expert_skills``) and drives the real ``BackgroundAgentLoader``
against the real ``dispatchable_experts_token``, so a rebuild reads the
token as many times as the production path reads it — which is what any
scheme keyed on "how many reads happened" gets wrong.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from EvoScientist.cli._agent_loader import BackgroundAgentLoader
from EvoScientist.subagents.expert_container import (
    dispatchable_experts_token,
    list_dispatchable_experts,
    publish_dispatchable_experts,
)
from EvoScientist.tools.skills_manager import SkillInfo

_SKILLS = "EvoScientist.tools.skills_manager.list_expert_skills"


def _expert(name: str) -> SkillInfo:
    return SkillInfo(
        name=name,
        description=f"{name} description",
        path=Path("/tmp/nope") / name,
        source="builtin",
        type="expert",
        role=f"{name} role",
        body="persona body\n",
    )


def _cue_names() -> list[str]:
    """What the active-expert cue and the ``/expert`` popup would report."""
    return [s.name for s in list_dispatchable_experts()]


def _loader(fail_rebuild: bool) -> tuple[BackgroundAgentLoader, object, list[str]]:
    seated = object()
    rebuilt = object()
    builds: list[int] = []
    reported: list[str] = []

    def loader_fn(**_kwargs: object) -> object:
        builds.append(1)
        if len(builds) == 1:
            return seated
        if fail_rebuild:
            raise RuntimeError("rebuild boom")
        return rebuilt

    loader: BackgroundAgentLoader = BackgroundAgentLoader(
        loader_fn,
        on_rebuild_failed=lambda exc: reported.append(str(exc)),
        build_token=dispatchable_experts_token,
        publish_build=publish_dispatchable_experts,
    )
    return loader, seated if fail_rebuild else rebuilt, reported


async def test_a_failed_rebuild_leaves_the_cue_on_the_seated_expert_set():
    """The divergence this whole mechanism exists to prevent, on the branch
    where it is hardest to get right.

    ``await_ready`` reads the token and ``start`` reads it again, so any
    scheme that stamps at decision time and undoes the stamp on failure has
    two writes to undo and one slot to undo them with.
    """
    loader, expected_agent, reported = _loader(fail_rebuild=True)

    with patch(_SKILLS, return_value=[_expert("alpha")]):
        loader.start()
        assert await loader.await_ready() is expected_agent
        assert _cue_names() == ["alpha"]

    # ``beta`` is authored mid-session. The rebuild it triggers fails, so the
    # seated agent still cannot route ``task()`` to it.
    with patch(_SKILLS, return_value=[_expert("alpha"), _expert("beta")]):
        assert await loader.await_ready() is expected_agent
        assert _cue_names() == ["alpha"]

    assert reported == ["rebuild boom"]


async def test_a_successful_rebuild_publishes_the_new_expert_set():
    """The other half: withholding the stamp must not withhold it forever."""
    loader, expected_agent, reported = _loader(fail_rebuild=False)

    with patch(_SKILLS, return_value=[_expert("alpha")]):
        loader.start()
        await loader.await_ready()
        assert _cue_names() == ["alpha"]

    with patch(_SKILLS, return_value=[_expert("alpha"), _expert("beta")]):
        assert await loader.await_ready() is expected_agent
        assert _cue_names() == ["alpha", "beta"]

    assert reported == []
