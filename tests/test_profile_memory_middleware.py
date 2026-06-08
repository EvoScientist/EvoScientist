from __future__ import annotations

import threading
from types import SimpleNamespace

from blockbuster import BlockBuster
from langchain_core.messages import SystemMessage

import EvoScientist.middleware.memory as memory_module
from EvoScientist import paths
from EvoScientist.memory.observations import (
    MemoryScope,
    MemorySourceType,
    MemoryType,
    record_observation_file,
)


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


def _path_project_id(workspace) -> str:
    return memory_module._resolve_project_id(workspace)


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

    middleware = memory_module.create_memory_middleware(str(memories))
    middleware.modify_request(_request())

    assert [tool.name for tool in middleware.tools] == ["record_observation"]
    assert (memories / "profile" / "SOUL.md").exists()
    assert (memories / "profile" / "USER_PROFILE.md").exists()
    assert (memories / "profile" / "RESEARCH_TASTE.md").exists()
    assert list((memories / "profile" / "projects").glob("*/PROJECT_PROFILE.md"))
    assert (memories / "observations" / "global").is_dir()
    assert list((memories / "observations" / "projects").glob("P-*"))


def test_profile_memory_can_disable_observation_tool(tmp_path, monkeypatch):
    memories = tmp_path / "memories"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", workspace)

    middleware = memory_module.create_memory_middleware(
        str(memories),
        enable_observation_tool=False,
    )
    middleware.modify_request(_request())

    assert middleware.tools == []
    assert (memories / "profile" / "USER_PROFILE.md").exists()


def test_memory_middleware_can_disable_all_memory_injection(tmp_path, monkeypatch):
    memories = tmp_path / "memories"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", workspace)

    middleware = memory_module.create_memory_middleware(
        str(memories),
        enable_profile_memory=False,
        enable_observation_memory=False,
    )
    request = _request()
    modified = middleware.modify_request(request)

    assert modified is request
    assert middleware.tools == []
    assert not (memories / "profile").exists()
    assert not (memories / "observations").exists()


def test_observation_memory_can_be_read_only_without_profile(tmp_path, monkeypatch):
    memories = tmp_path / "memories"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", workspace)

    middleware = memory_module.create_memory_middleware(
        str(memories),
        enable_profile_memory=False,
        enable_observation_memory=True,
        enable_observation_tool=False,
    )
    modified = middleware.modify_request(_request())
    content = str(modified.system_message.content)

    assert middleware.tools == []
    assert not (memories / "profile").exists()
    assert (memories / "observations" / "global").is_dir()
    assert list((memories / "observations" / "projects").glob("P-*"))
    assert "<observation_memory>" in content
    assert "Memory preflight:" in content
    assert "record_observation" not in content


def test_observation_index_loads_summary_frontmatter_once(tmp_path, monkeypatch):
    memories = tmp_path / "memories"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", workspace)
    project_id = _path_project_id(workspace)
    global_result = record_observation_file(
        memory_dir=memories,
        project_id=project_id,
        memory_type=MemoryType.SEMANTIC,
        summary="A global fact is available for future lookup.",
        observation="A global fact should be indexed.",
        why_it_matters="Future agents can decide whether to read it.",
        scope=MemoryScope.GLOBAL,
        source_type=MemorySourceType.SUBAGENT,
        source_session_id="thread-1",
        source_agent="research-agent",
    )
    project_result = record_observation_file(
        memory_dir=memories,
        project_id=project_id,
        memory_type=MemoryType.PROCEDURAL,
        summary="A project recipe is available for future lookup.",
        observation="A project recipe should be indexed.",
        why_it_matters="Future agents can choose it for this workspace.",
        scope=MemoryScope.PROJECT,
        source_type=MemorySourceType.SUBAGENT,
        source_session_id="thread-1",
        source_agent="code-agent",
    )
    (memories / "observations" / "global" / "O-old.md").write_text(
        "\n".join(
            [
                "---",
                'id: "O-old"',
                "memory_type: semantic",
                "scope: global",
                "---",
                "",
                "## Observation",
                "",
                "Old observations without summary are not indexed.",
            ]
        ),
        encoding="utf-8",
    )

    middleware = memory_module.create_memory_middleware(str(memories))
    indexed = {
        record.observation_id: (record.memory_type, record.scope, record.summary)
        for record in middleware._observation_index_records
    }
    record_observation_file(
        memory_dir=memories,
        project_id=project_id,
        memory_type=MemoryType.SEMANTIC,
        summary="This later observation is not in the cached index.",
        observation="Observation written after middleware construction.",
        why_it_matters="Prompt memory should stay stable during the agent run.",
        scope=MemoryScope.GLOBAL,
        source_type=MemorySourceType.SUBAGENT,
        source_session_id="thread-2",
        source_agent="research-agent",
    )

    assert indexed == {
        global_result["observation_id"]: (
            MemoryType.SEMANTIC,
            MemoryScope.GLOBAL,
            "A global fact is available for future lookup.",
        ),
        project_result["observation_id"]: (
            MemoryType.PROCEDURAL,
            MemoryScope.PROJECT,
            "A project recipe is available for future lookup.",
        ),
    }
    assert {
        record.observation_id for record in middleware._observation_index_records
    } == set(indexed)


def test_observation_index_omits_summaries_when_budget_exceeded(tmp_path, monkeypatch):
    memories = tmp_path / "memories"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", workspace)
    middleware = memory_module.create_memory_middleware(str(memories))
    records = [
        memory_module.ObservationIndexRecord(
            observation_id="O-large",
            memory_path="/observations/global/O-large.md",
            memory_type=MemoryType.PROCEDURAL,
            scope=MemoryScope.GLOBAL,
            summary="Do not inline this summary when the index exceeds budget.",
        )
    ]

    context = middleware._observation_index_context_from_records(
        records,
        max_inline_chars=1,
    )

    assert "Do not inline this summary" not in context


def test_profile_memory_uses_path_pointers_when_profiles_exceed_budget(
    tmp_path, monkeypatch
):
    memories = tmp_path / "memories"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", workspace)

    middleware = memory_module.create_memory_middleware(
        str(memories), max_inline_profile_chars=10
    )
    middleware.modify_request(_request())
    records = middleware._read_profile_records()

    assert middleware._profile_context_from_records(records) == (
        middleware._profile_pointer_context
    )


def test_profile_memory_async_path_bootstraps_and_injects(
    tmp_path, monkeypatch, run_async
):
    memories = tmp_path / "memories"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", workspace)

    async def _handler(request):
        return request

    middleware = memory_module.create_memory_middleware(str(memories))
    run_async(middleware.awrap_model_call(_request(), _handler))

    assert (memories / "profile" / "USER_PROFILE.md").exists()


def test_profile_memory_write_failure_uses_path_pointers(tmp_path, monkeypatch):
    memories = tmp_path / "memories"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", workspace)

    monkeypatch.setattr(
        memory_module.EvoMemoryMiddleware,
        "_write_text",
        lambda _self, _path, _content: False,
    )
    middleware = memory_module.create_memory_middleware(str(memories))

    middleware.modify_request(_request())

    assert not (memories / "profile" / "USER_PROFILE.md").exists()


def test_profile_memory_read_failure_uses_path_pointers_without_overwriting(
    tmp_path, monkeypatch
):
    memories = tmp_path / "memories"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", workspace)

    profile_dir = memories / "profile"
    profile_dir.mkdir(parents=True)
    soul_path = profile_dir / "SOUL.md"
    original_bytes = b"\xff\xfe\xfa existing profile bytes"
    soul_path.write_bytes(original_bytes)

    middleware = memory_module.create_memory_middleware(str(memories))
    middleware.modify_request(_request())

    assert soul_path.read_bytes() == original_bytes


def test_profile_memory_async_path_inlines_content_under_blockbuster(
    tmp_path, monkeypatch, run_async
):
    memories = tmp_path / "memories"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", workspace)

    middleware = memory_module.create_memory_middleware(str(memories))
    middleware.modify_request(_request())
    user_profile = memories / "profile" / "USER_PROFILE.md"
    user_profile.write_text(
        user_profile.read_text(encoding="utf-8")
        + "\n\n- Async profile content should be inlined.",
        encoding="utf-8",
    )

    call_threads = []
    original_read = middleware._read_profile_memory

    def tracked_read_profile_memory():
        call_threads.append(threading.get_ident())
        return original_read()

    monkeypatch.setattr(middleware, "_read_profile_memory", tracked_read_profile_memory)

    async def run():
        event_loop_thread = threading.get_ident()
        blocker = BlockBuster(scanned_modules=memory_module)
        blocker.activate()
        try:
            modified = await middleware.amodify_request(_request())
        finally:
            blocker.deactivate()
        return event_loop_thread, modified

    event_loop_thread, modified = run_async(run())

    assert call_threads
    assert all(thread_id != event_loop_thread for thread_id in call_threads)
    assert "Async profile content should be inlined." in str(
        modified.system_message.content
    )


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

    middleware = memory_module.create_memory_middleware(str(memories))
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


def test_profile_memory_deletes_blank_legacy_memory(tmp_path, monkeypatch):
    memories = tmp_path / "memories"
    memories.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", workspace)
    legacy_path = memories / "MEMORY.md"
    legacy_path.write_text("  \n\n", encoding="utf-8")

    middleware = memory_module.create_memory_middleware(str(memories))
    middleware.modify_request(_request())

    assert not legacy_path.exists()


def test_profile_memory_uses_explicit_workspace_for_project_profile(
    tmp_path, monkeypatch
):
    memories = tmp_path / "memories"
    global_workspace = tmp_path / "global-workspace"
    active_workspace = tmp_path / "active-workspace"
    global_workspace.mkdir()
    active_workspace.mkdir()
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", global_workspace)

    middleware = memory_module.create_memory_middleware(
        str(memories), workspace_dir=str(active_workspace)
    )
    middleware.modify_request(_request())

    expected_project_id = _path_project_id(active_workspace)
    wrong_project_id = _path_project_id(global_workspace)

    assert (
        memories / "profile" / "projects" / expected_project_id / "PROJECT_PROFILE.md"
    ).exists()
    assert not (
        memories / "profile" / "projects" / wrong_project_id / "PROJECT_PROFILE.md"
    ).exists()


def test_profile_memory_resolves_project_id_once_per_middleware(
    tmp_path, monkeypatch, run_async
):
    memories = tmp_path / "memories"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    calls = []

    def _resolve_project_id(workspace_dir):
        calls.append(workspace_dir)
        return "P-cached-project"

    monkeypatch.setattr(memory_module, "_resolve_project_id", _resolve_project_id)

    middleware = memory_module.create_memory_middleware(
        str(memories), workspace_dir=str(workspace), max_inline_profile_chars=10
    )
    middleware.modify_request(_request())
    run_async(middleware.amodify_request(_request()))

    assert calls == [workspace]
    assert middleware.project_id == "P-cached-project"
    assert any(
        path == "/profile/projects/P-cached-project/PROJECT_PROFILE.md"
        for path, _template in middleware._profile_specs
    )


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

    middleware = memory_module.create_memory_middleware(str(memories))
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

    middleware = memory_module.create_memory_middleware(str(memories))
    middleware.modify_request(_request())

    migrated_profile_text = "\n".join(_profile_texts(memories))
    assert "(unknown)" not in migrated_profile_text
    assert "Imported from legacy MEMORY.md" not in migrated_profile_text
    assert not (memories / "MEMORY.md").exists()
