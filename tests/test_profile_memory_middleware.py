from __future__ import annotations

import hashlib
from types import SimpleNamespace

from langchain_core.messages import SystemMessage

from EvoScientist import paths
from EvoScientist.middleware.memory import create_memory_middleware


def _request():
    request = SimpleNamespace(
        state={},
        runtime=object(),
        system_message=SystemMessage(content="base system"),
    )
    request.override = lambda **kwargs: SimpleNamespace(
        **{
            "state": request.state,
            "runtime": request.runtime,
            "system_message": kwargs.get("system_message", request.system_message),
        }
    )
    return request


def _system_text(modified) -> str:
    system_message = modified.system_message
    assert system_message is not None
    return str(system_message.content)


def _path_project_id(workspace) -> str:
    source = f"path:{workspace.resolve()}"
    return f"P-{hashlib.sha256(source.encode('utf-8')).hexdigest()[:16]}"


def _profile_texts(memories):
    return [
        path.read_text(encoding="utf-8")
        for path in (memories / "profile").rglob("*.md")
    ]


def test_profile_memory_bootstraps_and_injects_profile_files(tmp_path, monkeypatch):
    memories = tmp_path / "memories"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", workspace)

    middleware = create_memory_middleware(str(memories))
    modified = middleware.modify_request(_request())
    system_text = _system_text(modified)

    assert "Today's date" not in system_text
    assert "<profile_memory>" in system_text
    assert "# User profile" in system_text
    assert "/memories/profile/USER_PROFILE.md" in system_text
    assert (memories / "profile" / "SOUL.md").exists()
    assert (memories / "profile" / "USER_PROFILE.md").exists()
    assert (memories / "profile" / "RESEARCH_TASTE.md").exists()
    assert list((memories / "profile" / "projects").glob("*/PROJECT_PROFILE.md"))


def test_profile_memory_uses_path_pointers_when_profiles_exceed_budget(
    tmp_path, monkeypatch
):
    memories = tmp_path / "memories"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", workspace)

    middleware = create_memory_middleware(str(memories), max_inline_profile_chars=10)
    modified = middleware.modify_request(_request())
    system_text = _system_text(modified)

    assert "Profile files are available at:" in system_text
    assert "File: /memories/profile/SOUL.md" not in system_text
    assert "/memories/profile/USER_PROFILE.md" in system_text


def test_profile_memory_async_path_bootstraps_and_injects(
    tmp_path, monkeypatch, run_async
):
    memories = tmp_path / "memories"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", workspace)

    async def _handler(request):
        return request

    middleware = create_memory_middleware(str(memories))
    modified = run_async(middleware.awrap_model_call(_request(), _handler))
    system_text = _system_text(modified)

    assert "<profile_memory>" in system_text
    assert "/memories/profile/USER_PROFILE.md" in system_text
    assert (memories / "profile" / "USER_PROFILE.md").exists()


def test_profile_memory_write_failure_uses_path_pointers(tmp_path, monkeypatch):
    memories = tmp_path / "memories"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", workspace)

    middleware = create_memory_middleware(str(memories))
    monkeypatch.setattr(middleware, "_write_text", lambda _path, _content: False)

    modified = middleware.modify_request(_request())
    system_text = _system_text(modified)

    assert "Profile files are available at:" in system_text
    assert "File: /memories/profile/SOUL.md" not in system_text
    assert "# User profile" not in system_text
    assert not (memories / "profile" / "USER_PROFILE.md").exists()


def test_profile_memory_migrates_legacy_memory_once(tmp_path, monkeypatch):
    memories = tmp_path / "memories"
    memories.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", workspace)
    (memories / "MEMORY.md").write_text(
        "\n".join(
            [
                "# EvoScientist Memory",
                "",
                "## User Profile",
                "- **Name**: Alice",
                "",
                "## Research Preferences",
                "- **Primary Domain**: RL",
                "",
                "## Experiment History",
                "### [2026-01-01] Baseline",
                "- **Conclusion**: Worked",
                "",
                "## Learned Preferences",
                "- Prefers concise plans.",
            ]
        ),
        encoding="utf-8",
    )

    middleware = create_memory_middleware(str(memories))
    middleware.modify_request(_request())
    middleware.modify_request(_request())

    user_profile = (memories / "profile" / "USER_PROFILE.md").read_text(
        encoding="utf-8"
    )
    research_taste = (memories / "profile" / "RESEARCH_TASTE.md").read_text(
        encoding="utf-8"
    )

    assert user_profile.count("- **Name**: Alice") == 1
    assert user_profile.count("Prefers concise plans.") == 1
    assert user_profile.count("### Experiment History") == 1
    assert user_profile.count("- **Conclusion**: Worked") == 1
    assert research_taste.count("- **Primary Domain**: RL") == 1
    assert "Migrated from /memories/MEMORY.md" not in user_profile
    assert "Migrated from /memories/MEMORY.md" not in research_taste
    assert not (memories / "MEMORY.md").exists()


def test_profile_memory_uses_explicit_workspace_for_project_profile(
    tmp_path, monkeypatch
):
    memories = tmp_path / "memories"
    global_workspace = tmp_path / "global-workspace"
    active_workspace = tmp_path / "active-workspace"
    global_workspace.mkdir()
    active_workspace.mkdir()
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", global_workspace)

    middleware = create_memory_middleware(
        str(memories), workspace_dir=str(active_workspace)
    )
    modified = middleware.modify_request(_request())
    system_text = _system_text(modified)

    expected_project_id = _path_project_id(active_workspace)
    wrong_project_id = _path_project_id(global_workspace)

    assert (
        f"/memories/profile/projects/{expected_project_id}/PROJECT_PROFILE.md"
        in system_text
    )
    assert (
        memories / "profile" / "projects" / expected_project_id / "PROJECT_PROFILE.md"
    ).exists()
    assert wrong_project_id not in system_text
    assert not (
        memories / "profile" / "projects" / wrong_project_id / "PROJECT_PROFILE.md"
    ).exists()


def test_profile_memory_preserves_unmapped_legacy_memory(tmp_path, monkeypatch):
    memories = tmp_path / "memories"
    memories.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", workspace)
    legacy_path = memories / "MEMORY.md"
    custom_note = "Keep this custom deployment note."
    legacy_path.write_text(
        "\n".join(
            [
                "# EvoScientist Memory",
                "",
                "## User Profile",
                "- **Name**: Alice",
                "",
                "## Custom Notes",
                custom_note,
            ]
        ),
        encoding="utf-8",
    )

    middleware = create_memory_middleware(str(memories))
    middleware.modify_request(_request())

    user_profile = (memories / "profile" / "USER_PROFILE.md").read_text(
        encoding="utf-8"
    )
    assert custom_note in user_profile
    assert not legacy_path.exists()


def test_profile_memory_skips_legacy_unknown_placeholders(tmp_path, monkeypatch):
    memories = tmp_path / "memories"
    memories.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", workspace)
    (memories / "MEMORY.md").write_text(
        "\n".join(
            [
                "# EvoScientist Memory",
                "",
                "## User Profile",
                "- **Name**: (unknown)",
                "- **Role**: (unknown)",
                "",
                "## Research Preferences",
                "- **Primary Domain**: (unknown)",
                "- **Preferred Methods**: (unknown)",
                "",
                "## Experiment History",
                "(No experiments yet)",
                "",
                "## Learned Preferences",
                "- (none yet)",
            ]
        ),
        encoding="utf-8",
    )

    middleware = create_memory_middleware(str(memories))
    middleware.modify_request(_request())

    migrated_profile_text = "\n".join(_profile_texts(memories))
    assert "(unknown)" not in migrated_profile_text
    assert "Imported from legacy MEMORY.md" not in migrated_profile_text
    assert not (memories / "MEMORY.md").exists()
