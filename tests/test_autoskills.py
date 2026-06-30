from __future__ import annotations

import asyncio
from types import SimpleNamespace

from EvoScientist.config import EvoScientistConfig, MemorySkillSynthesisCadence
from EvoScientist.memory.autoskills.candidates import autoskill_candidates
from EvoScientist.memory.autoskills.proposals import (
    approve_skill_proposal,
    autoskill_proposals_dir,
    list_skill_proposals,
    pending_skill_proposal_count,
    reject_skill_proposal,
    submit_autoskill_proposal,
)
from EvoScientist.memory.autoskills.schedule import (
    AUTOSKILL_GRAPH_ID,
    AUTOSKILL_RUN_KIND,
    AUTOSKILL_SCHEDULE_SEARCH_LIMIT,
    alist_autoskill_schedules,
    autoskill_cron,
    reconcile_autoskill_schedule,
)
from EvoScientist.memory.observations import (
    MemoryScope,
    MemorySourceType,
    MemoryType,
    ObservationRelation,
    link_observation_files,
    record_observation_file,
)


def _record(
    memory_dir,
    *,
    summary: str,
    observation: str,
    memory_type: MemoryType = MemoryType.PROCEDURAL,
):
    return record_observation_file(
        memory_dir=memory_dir,
        project_id="P-project",
        memory_type=memory_type,
        summary=summary,
        observation=observation,
        why_it_matters=f"Future agents can reuse this pattern: {summary}",
        scope=MemoryScope.PROJECT,
        source_type=MemorySourceType.TURN,
        source_session_id="thread-1",
        source_agent="EvoScientist",
    )


def _write_skill_folder(memory_dir, skill_name: str, description: str, body: str):
    proposal_dir = autoskill_proposals_dir(memory_dir) / skill_name
    proposal_dir.mkdir(parents=True, exist_ok=True)
    (proposal_dir / "SKILL.md").write_text(
        f"---\nname: {skill_name}\ndescription: {description}\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return proposal_dir


def test_autoskill_cron_uses_presets():
    assert autoskill_cron("nightly", "03:00") == "0 3 * * *"
    assert autoskill_cron("weekly", "04:30") == "30 4 * * 0"
    assert autoskill_cron("monthly", "22:05") == "5 22 1 * *"


def test_autoskill_candidates_use_linked_procedural_clusters(tmp_path):
    memory_dir = tmp_path / "memories"
    first = _record(
        memory_dir,
        summary="Use focused pytest before full suite.",
        observation="Run the focused pytest file before the full test suite.",
    )
    second = _record(
        memory_dir,
        summary="Use ruff on changed Python modules.",
        observation="Run ruff on changed modules before broad validation.",
    )
    third = _record(
        memory_dir,
        summary="Validation workflow benefits from narrow checks.",
        observation="Narrow validation catches regressions before expensive checks.",
        memory_type=MemoryType.SEMANTIC,
    )
    for source, target in ((first, second), (second, third)):
        link_observation_files(
            memory_dir=memory_dir,
            project_id="P-project",
            source_observation_id=source["observation_id"],
            target_observation_id=target["observation_id"],
            reason="These observations describe the same validation workflow.",
        )

    candidates = autoskill_candidates(
        memory_dir=memory_dir,
        project_id="P-project",
    )

    assert len(candidates) == 1
    assert set(candidates[0]["observation_ids"]) == {
        first["observation_id"],
        second["observation_id"],
        third["observation_id"],
    }
    assert candidates[0]["procedural_count"] == 2
    assert candidates[0]["existing_pending_proposal"] is False
    assert candidates[0]["already_processed"] is False


def test_autoskill_candidates_surface_contradiction_clusters(tmp_path):
    memory_dir = tmp_path / "memories"
    first = _record(
        memory_dir,
        summary="Use cached package metadata for offline installs.",
        observation="Cached package metadata works when the network is unavailable.",
    )
    second = _record(
        memory_dir,
        summary="Avoid cached metadata for editable dependency changes.",
        observation="Cached package metadata can hide editable dependency changes.",
    )
    third = _record(
        memory_dir,
        summary="Package validation should check cache freshness.",
        observation="Validation should distinguish offline cache use from stale cache risks.",
        memory_type=MemoryType.SEMANTIC,
    )
    link_observation_files(
        memory_dir=memory_dir,
        project_id="P-project",
        source_observation_id=first["observation_id"],
        target_observation_id=second["observation_id"],
        relation=ObservationRelation.CONTRADICTS,
        reason="Cached metadata helps offline installs but can hide editable changes.",
    )
    link_observation_files(
        memory_dir=memory_dir,
        project_id="P-project",
        source_observation_id=second["observation_id"],
        target_observation_id=third["observation_id"],
        reason="Both observations describe cache-aware package validation.",
    )

    candidates = autoskill_candidates(
        memory_dir=memory_dir,
        project_id="P-project",
    )

    assert len(candidates) == 1
    assert set(candidates[0]["observation_ids"]) == {
        first["observation_id"],
        second["observation_id"],
        third["observation_id"],
    }
    assert any(
        relation["relation"] == ObservationRelation.CONTRADICTS
        for relation in candidates[0]["relations"]
    )


def test_skill_proposal_lifecycle_promotes_to_workspace_skill(tmp_path):
    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"

    _write_skill_folder(
        memory_dir,
        "focused-validation",
        "Use when validating code changes with staged checks.",
        "# Focused validation\n\nRun narrow checks before broad ones.",
    )
    proposal = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="focused-validation",
        cluster_hash="cluster-1",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="Three observations describe the same staged validation practice.",
    )

    assert proposal["submitted"] is True
    assert pending_skill_proposal_count(memory_dir) == 1
    pending = list_skill_proposals(memory_dir, status="pending")
    assert pending[0].skill_name == "focused-validation"
    assert pending[0].proposal_id == "focused-validation"

    approved = approve_skill_proposal(
        memory_dir,
        pending[0].proposal_id,
        skills_dir=skills_dir,
    )

    assert approved["approved"] is True
    skill_md = skills_dir / "focused-validation" / "SKILL.md"
    assert skill_md.exists()
    assert "name: focused-validation" in skill_md.read_text(encoding="utf-8")
    assert pending_skill_proposal_count(memory_dir) == 0
    assert list_skill_proposals(memory_dir)[0].status == "approved"


def test_submit_autoskill_proposal_rejects_invalid_generated_folder(tmp_path):
    memory_dir = tmp_path / "memories"
    _write_skill_folder(
        memory_dir,
        "focused-validation",
        "Use when validating code changes with staged checks.",
        "# Focused validation\n\nTODO: fill this in.",
    )

    proposal = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="focused-validation",
        cluster_hash="cluster-1",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="Three observations describe the same staged validation practice.",
    )

    assert proposal["submitted"] is False
    assert proposal["path"] == "/autoskill-proposals/focused-validation"
    assert "TODO placeholders" in proposal["errors"][0]
    assert pending_skill_proposal_count(memory_dir) == 0


def test_reject_skill_proposal_marks_processed(tmp_path):
    memory_dir = tmp_path / "memories"
    _write_skill_folder(
        memory_dir,
        "reject-me",
        "Use when testing rejected proposals.",
        "# Reject me\n",
    )
    proposal = submit_autoskill_proposal(
        memory_dir=memory_dir,
        skill_name="reject-me",
        cluster_hash="cluster-rejected",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale="Test rejection.",
    )

    rejected = reject_skill_proposal(memory_dir, proposal["proposal_id"])

    assert rejected["rejected"] is True
    assert list_skill_proposals(memory_dir)[0].status == "rejected"
    assert (memory_dir / "autoskills" / "processed" / "cluster-rejected.json").exists()


class _FakeCrons:
    def __init__(self):
        self.rows: list[dict] = []
        self.created: list[dict] = []
        self.deleted: list[str] = []
        self.searches: list[dict] = []

    def search(self, **kwargs):
        self.searches.append(kwargs)
        return list(self.rows)

    def create(self, **kwargs):
        row = {
            "cron_id": f"cron-{len(self.rows) + 1}",
            "assistant_id": kwargs["assistant_id"],
            "schedule": kwargs["schedule"],
            "input": kwargs["input"],
            "metadata": kwargs["metadata"],
            "timezone": kwargs["timezone"],
            "enabled": True,
        }
        self.rows.append(row)
        self.created.append(row)
        return row

    def delete(self, cron_id: str):
        self.deleted.append(cron_id)
        self.rows = [row for row in self.rows if row["cron_id"] != cron_id]


class _AsyncFakeCrons:
    def __init__(self):
        self.searches: list[dict] = []

    async def search(self, **kwargs):
        self.searches.append(kwargs)
        return [{"cron_id": "cron-async"}]


def test_alist_autoskill_schedules_uses_async_client_and_explicit_limit(monkeypatch):
    crons = _AsyncFakeCrons()
    client = SimpleNamespace(crons=crons)
    monkeypatch.setattr("langgraph_sdk.get_client", lambda **_kwargs: client)

    rows = asyncio.run(
        alist_autoskill_schedules(
            EvoScientistConfig(),
            limit=3,
        )
    )

    assert rows == [{"cron_id": "cron-async"}]
    assert crons.searches == [
        {
            "metadata": {"run_kind": AUTOSKILL_RUN_KIND},
            "limit": 3,
        }
    ]


def test_reconcile_autoskill_schedule_creates_updates_and_disables(
    tmp_path,
    monkeypatch,
):
    crons = _FakeCrons()
    client = SimpleNamespace(crons=crons)
    monkeypatch.setattr(
        "EvoScientist.langgraph_dev.manager.is_langgraph_dev_running",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr("langgraph_sdk.get_sync_client", lambda **_kwargs: client)

    cfg = EvoScientistConfig(
        memory_skill_synthesis_enabled=True,
        memory_skill_synthesis_cadence=MemorySkillSynthesisCadence.WEEKLY,
        memory_skill_synthesis_time="03:00",
        scheduler_default_timezone="UTC",
    )

    created = reconcile_autoskill_schedule(cfg, workspace_dir=tmp_path)
    unchanged = reconcile_autoskill_schedule(cfg, workspace_dir=tmp_path)
    updated = reconcile_autoskill_schedule(
        EvoScientistConfig(
            memory_skill_synthesis_enabled=True,
            memory_skill_synthesis_cadence=MemorySkillSynthesisCadence.NIGHTLY,
            memory_skill_synthesis_time="03:00",
            scheduler_default_timezone="UTC",
        ),
        workspace_dir=tmp_path,
    )
    disabled = reconcile_autoskill_schedule(
        EvoScientistConfig(memory_skill_synthesis_enabled=False),
        workspace_dir=tmp_path,
    )

    assert created["status"] == "created"
    assert unchanged["status"] == "unchanged"
    assert updated["status"] == "created"
    assert disabled == {"status": "disabled", "deleted": 1}
    assert crons.deleted == ["cron-1", "cron-1"]
    assert crons.rows == []
    assert created["schedule"] == "0 3 * * 0"
    assert updated["schedule"] == "0 3 * * *"
    assert created["cron_id"] == "cron-1"
    assert all(
        search["limit"] == AUTOSKILL_SCHEDULE_SEARCH_LIMIT for search in crons.searches
    )
    assert [row["assistant_id"] for row in crons.created] == [
        AUTOSKILL_GRAPH_ID,
        AUTOSKILL_GRAPH_ID,
    ]
