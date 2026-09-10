from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from blockbuster import BlockBuster
from langchain_core.messages import SystemMessage

import EvoScientist.middleware.memory as memory_module
from EvoScientist import paths
from EvoScientist.memory.observations import (
    MemoryScope,
    MemorySourceType,
    MemoryType,
    build_observation_index_context,
    list_observation_documents,
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
    return memory_module.resolve_project_id(workspace)


def _profile_texts(memories):
    return [
        path.read_text(encoding="utf-8")
        for path in (memories / "profile").rglob("*.md")
    ]


def _sorted_tool_names(middleware) -> list[str]:
    return sorted(tool.name for tool in middleware.tools)


def test_profile_memory_bootstraps_profiles_without_observation_project_dirs(
    tmp_path, monkeypatch
):
    memories = tmp_path / "memories"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", workspace)

    middleware = memory_module.create_memory_middleware(str(memories))
    middleware.modify_request(_request())

    assert (
        memories / "profile" / "projects" / middleware.project_id / "PROJECT_PROFILE.md"
    ).exists()
    assert (memories / "observations" / "global").is_dir()
    assert not (memories / "observations" / "projects").exists()


def test_append_to_system_message_preserves_metadata():
    system_message = SystemMessage(
        content="base system",
        id="system-1",
        name="root-system",
        additional_kwargs={"cache_control": {"type": "ephemeral"}},
        response_metadata={"provider": "test"},
    )

    updated = memory_module.append_to_system_message(
        system_message,
        "memory context",
    )

    assert updated.id == "system-1"
    assert updated.name == "root-system"
    assert updated.additional_kwargs == {"cache_control": {"type": "ephemeral"}}
    assert updated.response_metadata == {"provider": "test"}
    assert updated.content_blocks == [
        {"type": "text", "text": "base system"},
        {"type": "text", "text": "memory context"},
    ]


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

    assert _sorted_tool_names(middleware) == [
        "read_memory",
        "search_observations",
    ]
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

    assert _sorted_tool_names(middleware) == [
        "read_memory",
        "search_observations",
    ]
    assert not (memories / "profile").exists()
    assert (memories / "observations" / "global").is_dir()
    assert "<observation_memory>" in content
    assert "search_observations" in content
    assert "read_memory" in content
    assert "record_observation" not in content


def test_observation_index_refreshes_summary_frontmatter(tmp_path, monkeypatch):
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
        document.observation_id: (
            document.memory_type,
            document.scope,
            document.summary,
        )
        for document in list_observation_documents(
            memory_dir=memories,
            project_id=project_id,
        )
    }
    later_result = record_observation_file(
        memory_dir=memories,
        project_id=project_id,
        memory_type=MemoryType.SEMANTIC,
        summary="This later observation is refreshed into the index.",
        observation="Observation written after middleware construction.",
        why_it_matters="Prompt memory should reflect worker writes during the session.",
        scope=MemoryScope.GLOBAL,
        source_type=MemorySourceType.SUBAGENT,
        source_session_id="thread-2",
        source_agent="research-agent",
    )
    modified = middleware.modify_request(_request())
    refreshed_ids = {
        document.observation_id
        for document in list_observation_documents(
            memory_dir=memories,
            project_id=project_id,
        )
    }

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
    assert refreshed_ids == {*indexed, later_result["observation_id"]}
    assert "This later observation is refreshed into the index." in str(
        modified.system_message.content
    )


def test_observation_index_omits_summaries_when_budget_exceeded(tmp_path, monkeypatch):
    memories = tmp_path / "memories"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", workspace)
    memory_module.create_memory_middleware(str(memories))
    record_observation_file(
        memory_dir=memories,
        project_id=_path_project_id(workspace),
        memory_type=MemoryType.PROCEDURAL,
        summary="Do not inline this summary when the index exceeds budget.",
        observation="A large-index observation exists.",
        why_it_matters="Future prompts should fall back to search hints.",
        scope=MemoryScope.GLOBAL,
        source_type=MemorySourceType.SUBAGENT,
        source_session_id="thread-1",
        source_agent="research-agent",
    )

    context = build_observation_index_context(
        memory_dir=memories,
        project_id=_path_project_id(workspace),
        max_inline_chars=1,
    )

    assert "Do not inline this summary" not in context


def test_observation_index_over_budget_keeps_entries_that_fit(tmp_path, monkeypatch):
    memories = tmp_path / "memories"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", workspace)
    project_id = _path_project_id(workspace)
    record_observation_file(
        memory_dir=memories,
        project_id=project_id,
        memory_type=MemoryType.PROCEDURAL,
        summary="First over-budget observation " + ("x" * 320),
        observation="First large-index observation.",
        why_it_matters="Index truncation should retain entries when possible.",
        scope=MemoryScope.GLOBAL,
        source_type=MemorySourceType.SUBAGENT,
        source_session_id="thread-1",
        source_agent="research-agent",
    )
    record_observation_file(
        memory_dir=memories,
        project_id=project_id,
        memory_type=MemoryType.PROCEDURAL,
        summary="Second over-budget observation " + ("y" * 320),
        observation="Second large-index observation.",
        why_it_matters="Index truncation should retain entries when possible.",
        scope=MemoryScope.GLOBAL,
        source_type=MemorySourceType.SUBAGENT,
        source_session_id="thread-2",
        source_agent="research-agent",
    )

    context = build_observation_index_context(
        memory_dir=memories,
        project_id=project_id,
        max_inline_chars=1_350,
    )

    assert "Observation index truncated to entries that fit." in context
    assert len(context) <= 1_350
    assert "over-budget observation" in context


def test_construction_defers_observation_index_read_to_first_request(
    tmp_path, monkeypatch
):
    """Building the middleware must not read the observation store.

    The prompt-facing index is rebuilt fresh on every model call, so the
    stored construction-time value never reaches a prompt. Reading the whole
    store to seed it was pure cost, paid once per middleware and 12x per
    deployed-graph rebuild (main agent + 11 sub-agents). Construction still
    creates the search dirs, and the first ``modify_request`` still injects a
    current index built from the store.
    """
    memories = tmp_path / "memories"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", workspace)

    record_observation_file(
        memory_dir=memories,
        project_id=_path_project_id(workspace),
        memory_type=MemoryType.PROCEDURAL,
        summary="Continue the diarization bottleneck investigation.",
        observation="A prior-session observation the agent should surface.",
        why_it_matters="First-turn 'hello' should list resumable topics.",
        scope=MemoryScope.GLOBAL,
        source_type=MemorySourceType.SUBAGENT,
        source_session_id="thread-1",
        source_agent="research-agent",
    )

    calls: list[int] = []
    real_build = memory_module.build_observation_index_context

    def _counting_build(*args, **kwargs):
        calls.append(1)
        return real_build(*args, **kwargs)

    monkeypatch.setattr(
        memory_module, "build_observation_index_context", _counting_build
    )

    middleware = memory_module.create_memory_middleware(str(memories))

    # Construction must not read the observation store ...
    assert calls == []
    assert middleware._observation_index_context == ""
    # ... but must still create the cross-project search dir it prompts agents
    # to look in.
    assert (memories / "observations" / "global").is_dir()

    # The first model call injects a freshly built index containing the
    # prior-session observation, despite the empty construction seed.
    modified = middleware.modify_request(_request())
    content = str(modified.system_message.content)
    assert "Indexed observations:" in content
    assert "Continue the diarization bottleneck investigation." in content
    assert len(calls) == 1


def test_two_middlewares_share_the_observation_cache(tmp_path, monkeypatch):
    """Two ``EvoMemoryMiddleware`` instances over the same ``memory_dir`` must
    share the underlying ``list_observation_documents`` cache so the observation
    store is parsed once, not once per middleware.

    This is the 12x-redundant-work scenario the PR exists to fix: a deployed
    graph builds the main agent + 11 sub-agents, each with its own memory
    middleware.  Construction no longer reads the store (deferred to first
    use), and the first ``modify_request`` from any middleware warms the
    process-scoped cache for the rest.
    """
    from EvoScientist.memory.observations import store

    memories = tmp_path / "memories"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", workspace)

    record_observation_file(
        memory_dir=memories,
        project_id=_path_project_id(workspace),
        memory_type=MemoryType.PROCEDURAL,
        summary="Shared observation for both middlewares.",
        observation="Body text.",
        why_it_matters="Cache sharing across middleware instances.",
        scope=MemoryScope.GLOBAL,
        source_type=MemorySourceType.SUBAGENT,
        source_session_id="thread-1",
        source_agent="research-agent",
    )

    parse_calls: list[int] = []
    real_parse = store._parse_observation_search_document

    def _counting_parse(*args, **kwargs):
        parse_calls.append(1)
        return real_parse(*args, **kwargs)

    monkeypatch.setattr(store, "_parse_observation_search_document", _counting_parse)

    store._file_parse_cache.clear()
    store._cached_max_files = None

    middleware_a = memory_module.create_memory_middleware(
        str(memories), workspace_dir=workspace
    )
    middleware_b = memory_module.create_memory_middleware(
        str(memories), workspace_dir=workspace
    )

    # First middleware's model call parses the store and warms the cache.
    modified_a = middleware_a.modify_request(_request())
    assert "Shared observation for both middlewares." in str(
        modified_a.system_message.content
    )
    assert len(parse_calls) == 1

    # Second middleware's model call must hit the cache, not re-parse.
    modified_b = middleware_b.modify_request(_request())
    assert "Shared observation for both middlewares." in str(
        modified_b.system_message.content
    )
    assert len(parse_calls) == 1, (
        "second middleware must share the cache; re-parsing means the "
        "process-scoped cache is not working across middleware instances"
    )


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


async def test_profile_memory_async_path_bootstraps_and_injects(tmp_path, monkeypatch):
    memories = tmp_path / "memories"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", workspace)

    async def _handler(request):
        return request

    middleware = memory_module.create_memory_middleware(str(memories))
    await middleware.awrap_model_call(_request(), _handler)

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


async def test_profile_memory_async_path_inlines_content_under_blockbuster(
    tmp_path, monkeypatch
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

    event_loop_thread = threading.get_ident()
    blocker = BlockBuster(scanned_modules=memory_module)
    blocker.activate()
    try:
        modified = await middleware.amodify_request(_request())
    finally:
        blocker.deactivate()

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


async def test_profile_memory_resolves_project_id_once_per_middleware(
    tmp_path, monkeypatch
):
    memories = tmp_path / "memories"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    calls = []

    def resolve_project_id(workspace_dir):
        calls.append(workspace_dir)
        return "P-cached-project"

    monkeypatch.setattr(memory_module, "resolve_project_id", resolve_project_id)

    middleware = memory_module.create_memory_middleware(
        str(memories), workspace_dir=str(workspace), max_inline_profile_chars=10
    )
    middleware.modify_request(_request())
    await middleware.amodify_request(_request())

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


# ---- profile bootstrap: frontmatter helpers ---------------------------------


def test_split_frontmatter_round_trips_and_preserves_body():
    meta = memory_module._default_user_profile_frontmatter()
    body = "# User profile\n\n## Stable facts\n- remembered\n"

    text = memory_module._join_frontmatter(meta, body)
    parsed_meta, parsed_body = memory_module._split_frontmatter(text)

    # Pins the on-disk form the agent's edit_file targets.
    assert text.startswith("---\nname: ''\n")
    assert parsed_meta == meta
    assert parsed_body == body


def test_split_frontmatter_without_block_returns_text_unchanged():
    text = "# User profile\n\n- remembered\n"

    assert memory_module._split_frontmatter(text) == ({}, text)


def test_split_frontmatter_malformed_yaml_returns_none_meta():
    text = "---\nname: [unclosed\n---\n# User profile\n"

    assert memory_module._split_frontmatter(text) == (None, text)


def test_split_frontmatter_non_mapping_returns_none_meta():
    text = "---\n- just a list\n---\n# User profile\n"

    assert memory_module._split_frontmatter(text) == (None, text)


def test_user_profile_template_starts_with_default_frontmatter():
    template = memory_module.PROFILE_TEMPLATES["/profile/USER_PROFILE.md"]
    meta, body = memory_module._split_frontmatter(template)

    assert meta == memory_module._default_user_profile_frontmatter()
    assert meta["name"] == ""
    assert meta["intro"] == "pending"
    assert meta["evoscientist"] == {
        "sessions": 0,
        "intro_attempts": 0,
        "last_thread": "",
        "intro_asked_thread": "",
    }
    assert body.startswith("# User profile\n")
    assert "## Constraints" in body


# ---- profile bootstrap: atomic writes ---------------------------------------


def test_write_text_is_atomic_and_leaves_no_tmp_sibling(tmp_path, monkeypatch):
    memories = tmp_path / "memories"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", workspace)
    middleware = memory_module.create_memory_middleware(str(memories))
    target = memories / "profile" / "SOUL.md"

    assert middleware._write_text(target, "content") is True

    assert target.read_text(encoding="utf-8") == "content"
    assert list(target.parent.glob(".*.tmp")) == []


def test_write_text_failure_leaves_original_untouched_and_no_tmp_sibling(
    tmp_path, monkeypatch
):
    memories = tmp_path / "memories"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", workspace)
    middleware = memory_module.create_memory_middleware(str(memories))
    target = memories / "profile" / "SOUL.md"
    target.parent.mkdir(parents=True)
    target.write_text("original", encoding="utf-8")

    def _boom(*_a, **_kw):
        raise OSError("boom")

    monkeypatch.setattr(memory_module.os, "replace", _boom)

    assert middleware._write_text(target, "new content") is False

    assert target.read_text(encoding="utf-8") == "original"
    assert list(target.parent.glob(".*.tmp")) == []


# ---- profile bootstrap: migration ------------------------------------------


def _user_profile_meta(memories):
    text = (memories / "profile" / "USER_PROFILE.md").read_text(encoding="utf-8")
    return memory_module._split_frontmatter(text)


def test_existing_user_profile_without_frontmatter_gets_one_with_body_verbatim(
    tmp_path, monkeypatch
):
    memories = tmp_path / "memories"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", workspace)
    profile_dir = memories / "profile"
    profile_dir.mkdir(parents=True)
    body = "# User profile\n\n## Preferences\n- Likes short reports\n"
    (profile_dir / "USER_PROFILE.md").write_text(body, encoding="utf-8")

    middleware = memory_module.create_memory_middleware(str(memories))
    middleware.modify_request(_request())

    meta, parsed_body = _user_profile_meta(memories)
    assert meta == memory_module._default_user_profile_frontmatter()
    assert parsed_body == body


def test_existing_user_profile_with_frontmatter_is_left_alone(tmp_path, monkeypatch):
    memories = tmp_path / "memories"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", workspace)
    profile_dir = memories / "profile"
    profile_dir.mkdir(parents=True)
    original = "---\nname: Ada\nintro: pending\n---\n# User profile\n"
    (profile_dir / "USER_PROFILE.md").write_text(original, encoding="utf-8")

    middleware = memory_module.create_memory_middleware(str(memories))
    middleware.modify_request(_request())

    assert (profile_dir / "USER_PROFILE.md").read_text(encoding="utf-8") == original


def test_write_text_swallows_cleanup_errors_and_returns_false(tmp_path, monkeypatch):
    memories = tmp_path / "memories"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", workspace)
    middleware = memory_module.create_memory_middleware(str(memories))
    target = memories / "profile" / "USER_PROFILE.md"
    target.parent.mkdir(parents=True)
    target.write_text("original", encoding="utf-8")

    def _denied(*_args, **_kwargs):
        raise PermissionError("locked")

    monkeypatch.setattr(memory_module.os, "replace", _denied)
    monkeypatch.setattr(memory_module.Path, "unlink", _denied)

    assert middleware._write_text(target, "new") is False
    assert target.read_text(encoding="utf-8") == "original"


def test_bootstrap_never_replaces_an_empty_user_profile(tmp_path, monkeypatch):
    memories = tmp_path / "memories"
    profile_dir = memories / "profile"
    profile_dir.mkdir(parents=True)
    (profile_dir / "USER_PROFILE.md").write_text("", encoding="utf-8")
    _, middleware = _bootstrap_middleware(tmp_path, monkeypatch)

    system = _system(middleware.modify_request(_bootstrap_request()))

    assert "<profile_bootstrap>" not in system
    assert (profile_dir / "USER_PROFILE.md").read_text(encoding="utf-8") == ""


def test_ensure_profile_files_does_not_migrate_empty_user_profile(
    tmp_path, monkeypatch
):
    memories = tmp_path / "memories"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", workspace)
    profile_dir = memories / "profile"
    profile_dir.mkdir(parents=True)
    (profile_dir / "USER_PROFILE.md").write_text("", encoding="utf-8")

    middleware = memory_module.create_memory_middleware(str(memories))
    middleware.modify_request(_request())

    assert (profile_dir / "USER_PROFILE.md").read_text(encoding="utf-8") == ""


# ---- profile bootstrap: decision --------------------------------------------


def _view(**overrides):
    view = {
        "name": "",
        "intro": "pending",
        "sessions": 1,
        "intro_attempts": 0,
        "last_thread": "t1",
        "intro_asked_thread": "",
    }
    view.update(overrides)
    return view


def _decide(view, *, thread_id="t1", human_messages=1):
    return memory_module._bootstrap_decision(
        view, thread_id=thread_id, human_messages=human_messages
    )


def test_bootstrap_decision_first_session_first_turn():
    assert _decide(_view()) == "first"


def test_bootstrap_decision_none_once_name_is_set():
    assert _decide(_view(name="Ada")) is None


def test_bootstrap_decision_none_once_skipped():
    assert _decide(_view(intro="skipped")) is None


def test_bootstrap_decision_none_after_first_turn():
    assert _decide(_view(), human_messages=2) is None
    assert _decide(_view(), human_messages=0) is None


def test_bootstrap_decision_without_thread_id_is_first():
    assert _decide(_view(intro_attempts=1, sessions=9), thread_id=None) == "first"


def test_bootstrap_decision_same_thread_keeps_variant_during_first_turn():
    asked = _view(intro_attempts=1, intro_asked_thread="t1")
    assert _decide(asked) == "first"
    retried = _view(intro_attempts=2, intro_asked_thread="t1", sessions=5)
    assert _decide(retried) == "retry"


def test_bootstrap_decision_retry_follows_exponential_backoff():
    assert _decide(_view(intro_attempts=1, sessions=1)) is None
    assert _decide(_view(intro_attempts=1, sessions=2, last_thread="t2")) == "retry"
    assert _decide(_view(intro_attempts=2, sessions=3, last_thread="t3")) is None
    assert _decide(_view(intro_attempts=2, sessions=4, last_thread="t4")) == "retry"
    assert _decide(_view(intro_attempts=3, sessions=7, last_thread="t7")) is None
    assert _decide(_view(intro_attempts=3, sessions=8, last_thread="t8")) == "retry"


def test_bootstrap_decision_never_gives_up_but_caps_the_exponent():
    assert _decide(_view(intro_attempts=4, sessions=15, last_thread="t15")) is None
    assert _decide(_view(intro_attempts=4, sessions=16, last_thread="t16")) == "retry"
    # A corrupted attempt count must not build a huge exponent.
    assert _decide(_view(intro_attempts=10**9, sessions=99, last_thread="t99")) is None


def test_bootstrap_view_fills_defaults_and_ignores_bad_types():
    view = memory_module._bootstrap_view(
        {
            "name": 42,
            "intro": None,
            "evoscientist": {"sessions": "3", "intro_attempts": True},
        }
    )
    assert view == {
        "name": "42",
        "intro": "pending",
        "sessions": 0,
        "intro_attempts": 0,
        "last_thread": "",
        "intro_asked_thread": "",
    }
    assert memory_module._bootstrap_view({})["intro"] == "pending"


def test_apply_bootstrap_view_keeps_identity_keys_and_extra_keys():
    meta = {"name": "Ada", "publications_checked": "2026-09-01"}
    view = _view(name="Ada", sessions=2, last_thread="t2")

    merged = memory_module._apply_bootstrap_view(meta, view)

    assert merged["name"] == "Ada"
    assert merged["field"] == ""
    assert merged["intro"] == "pending"
    assert merged["publications_checked"] == "2026-09-01"
    assert merged["evoscientist"] == {
        "sessions": 2,
        "intro_attempts": 0,
        "last_thread": "t2",
        "intro_asked_thread": "",
    }
    assert list(merged)[:5] == ["name", "field", "homepage", "intro", "evoscientist"]


# ---- profile bootstrap: prompts ---------------------------------------------


def test_bootstrap_prompts_are_tagged():
    first = memory_module.PROFILE_BOOTSTRAP_FIRST
    retry = memory_module.PROFILE_BOOTSTRAP_RETRY

    for block in (first, retry):
        assert block.strip().startswith("<profile_bootstrap>")
        assert block.strip().endswith("</profile_bootstrap>")
        assert "`edit_file`" in block
        assert "intro: skipped" in block
    assert "`ask_user`" in first
    assert "Do not search the web" in first
    assert "always double-quoting the value" in first
    assert "Ask for consent before any survey" in first
    assert "spend a little time letting you get to" in first
    assert "continue as on first contact" in retry
    assert memory_module._PROFILE_BOOTSTRAP_CORE in first
    assert memory_module._PROFILE_BOOTSTRAP_CORE in retry
    assert memory_module._PROFILE_BOOTSTRAP_CONSENT in first
    assert memory_module._PROFILE_BOOTSTRAP_CONSENT in retry
    assert "yes / later / no" in first
    assert first.index("Ask for consent") < first.index("ask three things")
    assert "follow-up" in first
    assert "Do not repeat the full introduction" in retry
    assert "spend a little time letting you get to" in retry
    assert "yes / later / no" in retry
    assert retry.index("ask for consent") < retry.index("continue as on first contact")
    assert "frontmatter" not in memory_module.PROFILE_MEMORY_INSTRUCTIONS


# ---- profile bootstrap: middleware wiring -----------------------------------


def _bootstrap_request(human_messages: int = 1):
    from langchain_core.messages import HumanMessage

    request = _request()
    request.state = {
        "messages": [HumanMessage(content=f"m{i}") for i in range(human_messages)]
    }
    return request


def _bootstrap_middleware(tmp_path, monkeypatch, *, thread_id="t1", **kwargs):
    memories = tmp_path / "memories"
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", workspace)
    monkeypatch.setattr(memory_module, "_current_thread_id", lambda: thread_id)
    kwargs.setdefault("enable_profile_bootstrap", True)
    return memories, memory_module.create_memory_middleware(str(memories), **kwargs)


def _system(modified) -> str:
    content = modified.system_message.content
    if isinstance(content, str):
        return content
    return "\n".join(
        block.get("text", "") for block in content if isinstance(block, dict)
    )


def test_bootstrap_block_injected_last_on_fresh_first_turn(tmp_path, monkeypatch):
    _memories, middleware = _bootstrap_middleware(tmp_path, monkeypatch)

    system = _system(middleware.modify_request(_bootstrap_request()))

    assert "<profile_bootstrap>" in system
    assert "first exchange with this researcher" in system
    assert system.index("</profile_memory>") < system.index("<profile_bootstrap>")
    assert system.rstrip().endswith("</profile_bootstrap>")


def test_bootstrap_block_absent_by_default(tmp_path, monkeypatch):
    _, middleware = _bootstrap_middleware(
        tmp_path, monkeypatch, enable_profile_bootstrap=False
    )

    system = _system(middleware.modify_request(_bootstrap_request()))

    assert "<profile_bootstrap>" not in system
    meta, _ = _user_profile_meta(tmp_path / "memories")
    assert meta["evoscientist"]["sessions"] == 0


def test_bootstrap_block_absent_after_first_turn(tmp_path, monkeypatch):
    _, middleware = _bootstrap_middleware(tmp_path, monkeypatch)

    system = _system(middleware.modify_request(_bootstrap_request(human_messages=2)))

    assert "<profile_bootstrap>" not in system


def test_bootstrap_block_absent_once_name_or_skip_is_written(tmp_path, monkeypatch):
    memories, middleware = _bootstrap_middleware(tmp_path, monkeypatch)
    middleware.modify_request(_bootstrap_request())
    profile = memories / "profile" / "USER_PROFILE.md"

    profile.write_text(
        profile.read_text(encoding="utf-8").replace("name: ''", "name: Ada", 1),
        encoding="utf-8",
    )
    assert "<profile_bootstrap>" not in _system(
        middleware.modify_request(_bootstrap_request())
    )

    profile.write_text(
        profile.read_text(encoding="utf-8")
        .replace("name: Ada", "name: ''", 1)
        .replace("intro: pending", "intro: skipped", 1),
        encoding="utf-8",
    )
    assert "<profile_bootstrap>" not in _system(
        middleware.modify_request(_bootstrap_request())
    )


def test_bootstrap_bookkeeping_counts_sessions_and_attempts_once_per_thread(
    tmp_path, monkeypatch
):
    memories, middleware = _bootstrap_middleware(tmp_path, monkeypatch)
    body_before = memory_module._split_frontmatter(
        memory_module.PROFILE_TEMPLATES["/profile/USER_PROFILE.md"]
    )[1]

    middleware.modify_request(_bootstrap_request())
    middleware.modify_request(_bootstrap_request())  # same thread, ask_user resume
    meta, body = _user_profile_meta(memories)

    assert body == body_before
    assert meta["evoscientist"] == {
        "sessions": 1,
        "intro_attempts": 1,
        "last_thread": "t1",
        "intro_asked_thread": "t1",
    }

    monkeypatch.setattr(memory_module, "_current_thread_id", lambda: "t2")
    system = _system(middleware.modify_request(_bootstrap_request()))
    middleware.modify_request(_bootstrap_request())  # same thread: no double bump
    meta, _ = _user_profile_meta(memories)

    assert "still has no `name`" in system
    assert meta["evoscientist"]["sessions"] == 2
    assert meta["evoscientist"]["intro_attempts"] == 2


def test_bootstrap_retries_with_exponential_backoff(tmp_path, monkeypatch):
    memories, middleware = _bootstrap_middleware(tmp_path, monkeypatch)
    middleware.modify_request(_bootstrap_request())  # session 1: first ask

    monkeypatch.setattr(memory_module, "_current_thread_id", lambda: "t2")
    system = _system(middleware.modify_request(_bootstrap_request()))
    meta, _ = _user_profile_meta(memories)

    assert "still has no `name`" in system
    assert "first exchange with this researcher" not in system
    assert meta["evoscientist"]["intro_attempts"] == 2
    assert meta["evoscientist"]["intro_asked_thread"] == "t2"

    monkeypatch.setattr(memory_module, "_current_thread_id", lambda: "t3")
    assert "<profile_bootstrap>" not in _system(
        middleware.modify_request(_bootstrap_request())
    )

    monkeypatch.setattr(memory_module, "_current_thread_id", lambda: "t4")
    assert "still has no `name`" in _system(
        middleware.modify_request(_bootstrap_request())
    )

    for thread in ("t5", "t6", "t7"):
        monkeypatch.setattr(memory_module, "_current_thread_id", lambda t=thread: t)
        assert "<profile_bootstrap>" not in _system(
            middleware.modify_request(_bootstrap_request())
        )

    monkeypatch.setattr(memory_module, "_current_thread_id", lambda: "t8")
    assert "still has no `name`" in _system(
        middleware.modify_request(_bootstrap_request())
    )


def test_bootstrap_without_thread_id_injects_without_bookkeeping(tmp_path, monkeypatch):
    memories, middleware = _bootstrap_middleware(tmp_path, monkeypatch, thread_id=None)

    system = _system(middleware.modify_request(_bootstrap_request()))
    meta, _ = _user_profile_meta(memories)

    assert "<profile_bootstrap>" in system
    assert meta["evoscientist"]["sessions"] == 0
    assert meta["evoscientist"]["intro_attempts"] == 0


def test_bootstrap_write_failure_still_injects(tmp_path, monkeypatch):
    _memories, middleware = _bootstrap_middleware(tmp_path, monkeypatch)
    middleware.modify_request(_request())  # profile files exist now
    monkeypatch.setattr(
        memory_module.EvoMemoryMiddleware,
        "_write_text",
        lambda _self, _path, _content: False,
    )

    system = _system(middleware.modify_request(_bootstrap_request()))

    assert "<profile_bootstrap>" in system


def test_migrated_existing_user_gets_bootstrap_on_new_thread(tmp_path, monkeypatch):
    memories = tmp_path / "memories"
    profile_dir = memories / "profile"
    profile_dir.mkdir(parents=True)
    (profile_dir / "USER_PROFILE.md").write_text(
        "# User profile\n\n## Preferences\n- Likes short reports\n", encoding="utf-8"
    )
    _, middleware = _bootstrap_middleware(tmp_path, monkeypatch)

    system = _system(middleware.modify_request(_bootstrap_request()))

    assert "<profile_bootstrap>" in system
    assert "- Likes short reports" in system


def test_bootstrap_skips_unparsable_frontmatter_without_writing(tmp_path, monkeypatch):
    memories = tmp_path / "memories"
    profile_dir = memories / "profile"
    profile_dir.mkdir(parents=True)
    original = "---\nname: [unclosed\n---\n# User profile\n"
    (profile_dir / "USER_PROFILE.md").write_text(original, encoding="utf-8")
    _, middleware = _bootstrap_middleware(tmp_path, monkeypatch)

    system = _system(middleware.modify_request(_bootstrap_request()))

    assert "<profile_bootstrap>" not in system
    assert (profile_dir / "USER_PROFILE.md").read_text(encoding="utf-8") == original


async def test_bootstrap_async_path_matches_sync(tmp_path, monkeypatch):
    memories, middleware = _bootstrap_middleware(tmp_path, monkeypatch)

    system = _system(await middleware.amodify_request(_bootstrap_request()))
    meta, _ = _user_profile_meta(memories)

    assert "<profile_bootstrap>" in system
    assert system.index("</profile_memory>") < system.index("<profile_bootstrap>")
    assert meta["evoscientist"]["intro_attempts"] == 1

    assert "<profile_bootstrap>" not in _system(
        await middleware.amodify_request(_bootstrap_request(human_messages=2))
    )


def test_count_human_messages_and_thread_id_helpers():
    from langchain_core.messages import AIMessage, HumanMessage

    assert memory_module._count_human_messages({}) == 0
    assert memory_module._count_human_messages({"messages": "nope"}) == 0
    assert (
        memory_module._count_human_messages(
            {"messages": [HumanMessage(content="a"), AIMessage(content="b")]}
        )
        == 1
    )
    # Outside a runnable context there is no config, hence no thread id.
    assert memory_module._current_thread_id() is None


def test_count_human_messages_ignores_synthetic_summary_messages():
    from langchain_core.messages import AIMessage, HumanMessage

    summary = HumanMessage(
        content="summary", additional_kwargs={"lc_source": "summarization"}
    )
    assert (
        memory_module._count_human_messages(
            {"messages": [summary, AIMessage(content="a")]}
        )
        == 0
    )
    assert (
        memory_module._count_human_messages(
            {"messages": [summary, HumanMessage(content="hi"), AIMessage(content="a")]}
        )
        == 1
    )


def test_bootstrap_ignores_post_summarization_synthetic_human_message(
    tmp_path, monkeypatch
):
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    _memories, middleware = _bootstrap_middleware(tmp_path, monkeypatch)
    request = _request()
    request.state = {
        "messages": [
            HumanMessage(
                content="summary", additional_kwargs={"lc_source": "summarization"}
            ),
            AIMessage(content="ok"),
            ToolMessage(content="x", tool_call_id="t"),
        ]
    }

    system = _system(middleware.modify_request(request))

    assert "<profile_bootstrap>" not in system


# ---- profile bootstrap: assembly --------------------------------------------


def _assembly_cfg(*, auto_mode: bool):
    cfg = MagicMock()
    cfg.enable_ask_user = False
    cfg.auto_approve = True
    cfg.auto_mode = auto_mode
    cfg.auxiliary_model = ""
    cfg.auxiliary_provider = ""
    return cfg


def _assemble(tmp_path, monkeypatch, *, auto_mode: bool, for_async_subagent=False):
    from EvoScientist.EvoScientist import _get_default_middleware

    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", workspace)
    monkeypatch.setattr(paths, "MEMORIES_DIR", tmp_path / "memories")
    with (
        patch(
            "EvoScientist.middleware.create_tool_selector_middleware", return_value=[]
        ),
        patch("EvoScientist.EvoScientist._ensure_chat_model") as mock_model,
        patch("EvoScientist.EvoScientist._ensure_config") as mock_config,
    ):
        mock_config.return_value = _assembly_cfg(auto_mode=auto_mode)
        mock_model.return_value = MagicMock(profile={"max_input_tokens": 200_000})
        middleware = _get_default_middleware(for_async_subagent=for_async_subagent)
    return next(
        m for m in middleware if isinstance(m, memory_module.EvoMemoryMiddleware)
    )


def test_main_agent_enables_profile_bootstrap(tmp_path, monkeypatch):
    instance = _assemble(tmp_path, monkeypatch, auto_mode=False)
    assert instance._enable_profile_bootstrap is True


def test_auto_mode_disables_profile_bootstrap(tmp_path, monkeypatch):
    instance = _assemble(tmp_path, monkeypatch, auto_mode=True)
    assert instance._enable_profile_bootstrap is False


def test_async_subagent_disables_profile_bootstrap(tmp_path, monkeypatch):
    instance = _assemble(
        tmp_path, monkeypatch, auto_mode=False, for_async_subagent=True
    )
    assert instance._enable_profile_bootstrap is False


def test_sync_subagent_site_does_not_pass_profile_bootstrap(tmp_path, monkeypatch):
    from EvoScientist.EvoScientist import _inject_subagent_middleware

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", workspace)
    monkeypatch.setattr(paths, "MEMORIES_DIR", tmp_path / "memories")
    sub = {"name": "research"}
    with (
        patch("EvoScientist.EvoScientist._ensure_chat_model") as mock_model,
        patch("EvoScientist.EvoScientist._ensure_config") as mock_config,
    ):
        mock_config.return_value = _assembly_cfg(auto_mode=False)
        mock_model.return_value = MagicMock(profile={"max_input_tokens": 200_000})
        _inject_subagent_middleware([sub], workspace_dir=workspace)

    instance = next(
        m for m in sub["middleware"] if isinstance(m, memory_module.EvoMemoryMiddleware)
    )
    assert instance._enable_profile_bootstrap is False
