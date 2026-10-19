"""The workspace reaches every consumer explicitly, and two workspaces stay apart."""

from __future__ import annotations

from pathlib import Path

import pytest

from EvoScientist.paths import Workspace
from EvoScientist.tools.skills_manager import list_skills
from tests.fakes import StubChannel


def _write_skill(skills_dir: Path, name: str) -> None:
    skill = skills_dir / name
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {name} skill\n---\n\nBody.\n",
        encoding="utf-8",
    )


@pytest.fixture
def two_workspaces(tmp_path, monkeypatch):
    import EvoScientist.paths as paths

    # Keep the global tier empty so only workspace skills show up.
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", tmp_path / "global-skills")
    a = Workspace(tmp_path / "a")
    b = Workspace(tmp_path / "b")
    _write_skill(a.skills_dir, "only-in-a")
    _write_skill(b.skills_dir, "only-in-b")
    return a, b


def test_list_skills_reads_only_the_given_workspace(two_workspaces):
    a, b = two_workspaces
    assert {s.name for s in list_skills(workspace=a)} == {"only-in-a"}
    assert {s.name for s in list_skills(workspace=b)} == {"only-in-b"}


def test_skill_manager_tool_is_bound_to_its_workspace(two_workspaces):
    from EvoScientist.tools import make_skill_manager_tool

    a, b = two_workspaces
    tool_a = make_skill_manager_tool(a)
    tool_b = make_skill_manager_tool(b)

    assert tool_a.name == tool_b.name == "skill_manager"
    out_a = tool_a.invoke({"action": "list"})
    out_b = tool_b.invoke({"action": "list"})
    assert "only-in-a" in out_a
    assert "only-in-b" not in out_a
    assert "only-in-b" in out_b
    assert "only-in-a" not in out_b


def test_local_install_goes_into_the_given_workspace(two_workspaces, tmp_path):
    from EvoScientist.tools.skills_manager import install_skill

    a, b = two_workspaces
    source = tmp_path / "incoming"
    _write_skill(source, "fresh")

    result = install_skill(str(source / "fresh"), global_install=False, workspace=a)
    assert result["success"], result
    assert (a.skills_dir / "fresh" / "SKILL.md").is_file()
    assert not (b.skills_dir / "fresh").exists()


def test_skill_manager_install_resolves_virtual_source_in_its_work_dir(
    two_workspaces, tmp_path
):
    import EvoScientist.paths as paths
    from EvoScientist.tools import make_skill_manager_tool

    a, _ = two_workspaces
    work_dir = a.runs_dir / "r1"
    _write_skill(work_dir, "from-run")
    tool = make_skill_manager_tool(a, work_dir=work_dir)

    out = tool.invoke({"action": "install", "source": "/from-run"})

    assert "Successfully installed skill: from-run" in out
    # The tool installs into the global tier (patched to a temp folder).
    assert (Path(paths.GLOBAL_SKILLS_DIR) / "from-run" / "SKILL.md").is_file()
    assert "from-run" in tool.invoke({"action": "list"})


def test_install_resolves_virtual_source_against_work_dir(two_workspaces, tmp_path):
    from EvoScientist.tools.skills_manager import install_skill

    a, _ = two_workspaces
    work_dir = tmp_path / "a" / "runs" / "r1"
    _write_skill(work_dir, "from-run")

    result = install_skill(
        "/from-run", global_install=False, workspace=a, work_dir=work_dir
    )
    assert result["success"], result
    assert (a.skills_dir / "from-run" / "SKILL.md").is_file()


def test_default_backend_roots_sandbox_at_work_dir(two_workspaces, tmp_path):
    from EvoScientist.EvoScientist import _get_default_backend

    a, _ = two_workspaces
    work_dir = tmp_path / "a" / "runs" / "r1"
    backend = _get_default_backend(a, work_dir=work_dir)

    assert Path(backend.default.cwd) == work_dir.resolve()
    assert backend.default._skills_dir == a.skills_dir
    assert Path(backend.routes["/skills/"]._primary.cwd) == a.skills_dir


def test_default_backend_defaults_to_workspace_root(two_workspaces):
    from EvoScientist.EvoScientist import _get_default_backend

    a, _ = two_workspaces
    backend = _get_default_backend(a)
    assert Path(backend.default.cwd) == a.root


def test_channel_media_goes_to_the_injected_folder(tmp_path):
    from EvoScientist.channels.bus import MessageBus
    from EvoScientist.channels.channel_manager import ChannelManager

    media_dir = tmp_path / "ws" / "media"
    manager = ChannelManager(MessageBus(), media_dir=media_dir)
    channel = StubChannel()
    manager.register(channel)

    path = channel._media_path("photo.jpg")
    assert path == media_dir / "photo.jpg"
    assert media_dir.is_dir()


def test_channel_without_media_folder_fails_loudly():
    channel = StubChannel()
    with pytest.raises(RuntimeError, match="no media folder"):
        channel._media_path("photo.jpg")


def test_standalone_without_agent_does_not_load_config(tmp_path, monkeypatch):
    """A headless channel with no agent reads no config (as before)."""
    import EvoScientist.channels.standalone as standalone
    import EvoScientist.config as config_mod

    monkeypatch.delenv("EVOSCIENTIST_WORKSPACE_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    def _no_config(*_a, **_k):
        raise AssertionError("config must not be loaded without an agent")

    captured = {}

    async def _fake_main(*_args, workspace, **_kwargs):
        captured["workspace"] = workspace

    monkeypatch.setattr(config_mod, "get_effective_config", _no_config)
    monkeypatch.setattr(standalone, "_async_main", _fake_main)

    standalone.run_standalone(StubChannel(), object(), use_agent=False)

    assert captured["workspace"] == Workspace(tmp_path)


def test_compact_offloads_into_the_work_dir(two_workspaces, tmp_path, monkeypatch):
    """/compact's history backend is rooted where the agent works."""
    import EvoScientist.EvoScientist as evo

    a, _ = two_workspaces
    work_dir = a.runs_dir / "r1"
    seen = {}

    def _spy(workspace, *, work_dir=None, **_kwargs):
        seen["workspace"] = workspace
        seen["work_dir"] = work_dir
        raise RuntimeError("stop after capturing")

    monkeypatch.setattr(evo, "_get_default_backend", _spy)
    monkeypatch.setattr(evo, "_ensure_chat_model", lambda: object())

    import asyncio

    from EvoScientist.cli.commands import compact_conversation
    from EvoScientist.gateway import GraphTarget
    from tests.fakes import FakeGraphGateway

    gateway = FakeGraphGateway()

    async def _state(*_a, **_k):
        from langchain_core.messages import AIMessage, HumanMessage

        return {"messages": [HumanMessage("hi"), AIMessage("hello")] * 20}

    monkeypatch.setattr(gateway, "get_state_values", _state, raising=False)
    try:
        asyncio.run(
            compact_conversation(
                gateway,
                "tid",
                GraphTarget(workspace_dir=str(work_dir)),
                workspace=a,
            )
        )
    except RuntimeError:
        pass
    assert seen == {"workspace": a, "work_dir": str(work_dir)}
