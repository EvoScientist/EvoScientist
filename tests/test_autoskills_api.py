"""The AutoSkills routes the WebUI reviews drafts through."""

from __future__ import annotations

import json
import os
import shutil
import sys

import pytest

from EvoScientist import local_secret, paths
from EvoScientist.langgraph_dev import registry
from EvoScientist.langgraph_dev.autoskills_api import MAX_FILE_BYTES, MAX_INLINE_CHARS
from EvoScientist.memory.autoskills.proposals import (
    approve_skill_proposal,
    autoskill_proposals_dir,
    processed_cluster_hashes,
    submit_autoskill_proposal,
)
from EvoScientist.paths import Workspace


@pytest.fixture
def projects(monkeypatch) -> registry.Registry:
    fresh = registry.Registry()
    monkeypatch.setattr(registry, "REGISTRY", fresh)
    return fresh


@pytest.fixture
def server(monkeypatch, no_workspace_setup, projects):
    from starlette.testclient import TestClient

    from EvoScientist.langgraph_dev.http import app

    monkeypatch.setattr(local_secret, "_value", None)
    with TestClient(
        app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000)
    ) as client:
        yield client


@pytest.fixture
def served(projects, workspace) -> Workspace:
    projects.set_registered(workspace.key, True)
    return workspace


def _draft(
    workspace: Workspace,
    name: str,
    *,
    operation: str = "create",
    body: str = "Steps.",
    files: dict[str, str] | None = None,
    workspace_dir: str | None = None,
) -> dict:
    folder = autoskill_proposals_dir(paths.MEMORIES_DIR) / name
    folder.mkdir(parents=True, exist_ok=True)
    # LF on every platform: the routes return the bytes as they are on disk.
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Use when testing {name}.\n---\n\n{body}\n",
        encoding="utf-8",
        newline="\n",
    )
    for relative, text in (files or {}).items():
        (folder / relative).parent.mkdir(parents=True, exist_ok=True)
        (folder / relative).write_text(text, encoding="utf-8", newline="\n")
    result = submit_autoskill_proposal(
        memory_dir=paths.MEMORIES_DIR,
        skill_name=name,
        cluster_hash=f"cluster-{name}",
        source_observation_ids=["O-1", "O-2", "O-3"],
        rationale=f"Three observations describe {name}.",
        operation=operation,
        target_skill_name=name if operation == "update" else None,
        skills_dir=workspace.skills_dir,
        workspace_dir=workspace_dir or workspace.root,
    )
    assert result["submitted"] is True, result
    return result


def _approve(workspace: Workspace, name: str) -> None:
    result = approve_skill_proposal(
        paths.MEMORIES_DIR,
        name,
        skills_dir=workspace.skills_dir,
        workspace_dir=workspace.root,
    )
    assert result["approved"] is True, result


def _installed_skill(
    root, name: str, body: str = "Installed.", *, folder: str | None = None
) -> None:
    folder = root / (folder or name)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Use when testing {name}.\n---\n\n{body}\n",
        encoding="utf-8",
    )


def _edit_manifest(name: str, **changes) -> None:
    path = autoskill_proposals_dir(paths.MEMORIES_DIR) / name / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(changes)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _only_draft(server) -> dict:
    proposals = server.get("/api/autoskills/proposals").json()["proposals"]
    assert len(proposals) == 1
    return proposals[0]


# ---------------------------------------------------------------------------
# GET /api/autoskills/proposals
# ---------------------------------------------------------------------------


def test_list_returns_drafts_of_registered_projects_only(
    server, served, projects, tmp_path
):
    elsewhere = Workspace(tmp_path / "elsewhere")
    elsewhere.root.mkdir()
    projects.set_registered(elsewhere.key, False)
    _draft(served, "focused-validation")
    _draft(elsewhere, "other-skill")
    _draft(served, "no-project")
    _edit_manifest("no-project", workspace_dir="")

    body = server.get("/api/autoskills/proposals").json()

    assert [d["proposal_id"] for d in body["proposals"]] == ["focused-validation"]
    assert list(body["by_project"]) == [served.key]


def test_list_counts_and_by_project(server, served, projects, tmp_path):
    quiet = Workspace(tmp_path / "quiet")
    quiet.root.mkdir()
    projects.set_registered(quiet.key, True)
    _draft(served, "focused-validation")
    _draft(served, "approved-skill")
    _approve(served, "approved-skill")

    body = server.get("/api/autoskills/proposals").json()

    assert body["counts"] == {"pending": 1, "approved": 1, "rejected": 0}
    assert body["by_project"] == {
        served.key: {"pending": 1, "approved": 1, "rejected": 0},
        quiet.key: {"pending": 0, "approved": 0, "rejected": 0},
    }


def test_list_filters_by_status_and_refuses_unknown(server, served):
    _draft(served, "focused-validation")
    _draft(served, "approved-skill")
    _approve(served, "approved-skill")

    approved = server.get("/api/autoskills/proposals?status=approved").json()
    bogus = server.get("/api/autoskills/proposals?status=bogus")

    assert [d["proposal_id"] for d in approved["proposals"]] == ["approved-skill"]
    assert approved["counts"] == {"pending": 1, "approved": 1, "rejected": 0}
    assert bogus.status_code == 400
    assert bogus.json() == {"error": "invalid_status"}


def test_list_inlines_texts_and_lists_files(server, served):
    _draft(served, "focused-validation", files={"scripts/check.py": "print(1)\n"})

    draft = _only_draft(server)

    assert draft["skill_md"].startswith("---\nname: focused-validation")
    assert draft["skill_md_truncated"] is False
    assert draft["rationale_md"].startswith("Three observations")
    assert draft["rationale_md_truncated"] is False
    assert draft["files"] == [
        {"path": "SKILL.md", "size": len(draft["skill_md"]), "installed": False},
        {"path": "scripts/check.py", "size": 9, "installed": False},
    ]
    assert draft["install_target"] == str(served.skills_dir / "focused-validation")
    assert draft["install_base"] is None
    assert draft["install_error"] is None
    assert draft["operation"] == "create"
    assert draft["scope"] == "workspace"
    assert draft["status"] == "pending"
    assert draft["source_observation_ids"] == ["O-1", "O-2", "O-3"]
    assert draft["approved_by"] is None
    assert draft["approved_skill_present"] is None


def test_list_orders_files_by_path_text(server, served):
    _draft(
        served,
        "focused-validation",
        files={
            "a/x.txt": "x\n",
            "a-b/y.txt": "y\n",
            "docs/manifest.json": "{}\n",
            "docs/RATIONALE.md": "Why.\n",
        },
    )

    draft = _only_draft(server)

    assert [f["path"] for f in draft["files"]] == [
        "SKILL.md",
        "a-b/y.txt",
        "a/x.txt",
        "docs/RATIONALE.md",
        "docs/manifest.json",
    ]


@pytest.mark.parametrize("action", ["approve", "reject"])
def test_list_inlines_no_text_for_a_decided_draft(
    server, served, secret, quiet_skill_index, action
):
    _draft(served, "focused-validation")
    _decide(server, "focused-validation", action, secret)

    draft = _only_draft(server)

    for field in (
        "skill_md",
        "skill_md_truncated",
        "rationale_md",
        "rationale_md_truncated",
    ):
        assert draft[field] is None
    assert draft["description"] == "Use when testing focused-validation."
    assert [f["path"] for f in draft["files"]] == ["SKILL.md"]


def test_list_truncates_large_skill_md(server, served):
    _draft(served, "focused-validation", body="x" * (MAX_INLINE_CHARS + 10))

    draft = _only_draft(server)

    assert draft["skill_md_truncated"] is True
    assert len(draft["skill_md"]) == MAX_INLINE_CHARS


def test_list_reports_install_error_for_collision(server, served):
    _draft(served, "focused-validation")
    _installed_skill(served.skills_dir, "focused-validation")

    draft = _only_draft(server)

    assert draft["install_target"] is None
    assert "already exists" in draft["install_error"]


def test_list_update_reports_the_global_skill_it_starts_from(server, served):
    global_dir = paths.GLOBAL_SKILLS_DIR
    _installed_skill(global_dir, "paper-review", body="Global workflow.")
    _draft(
        served,
        "paper-review",
        operation="update",
        files={"scripts/new.py": "print(2)\n"},
    )

    draft = _only_draft(server)

    assert draft["install_base"] == str((global_dir / "paper-review").resolve())
    installed = {f["path"]: f["installed"] for f in draft["files"]}
    assert installed == {"SKILL.md": True, "scripts/new.py": False}


def test_list_workspace_dir_is_registry_key(server, served):
    _draft(served, "focused-validation")
    _edit_manifest("focused-validation", workspace_dir=str(served.root) + "/")

    draft = _only_draft(server)

    assert draft["workspace_dir"] == served.key


def test_list_reports_uninstalled_approval(server, served):
    _draft(served, "focused-validation")
    _approve(served, "focused-validation")
    shutil.rmtree(served.skills_dir / "focused-validation")

    draft = _only_draft(server)

    assert draft["status"] == "approved"
    assert draft["approved_by"] == "user"
    assert draft["approved_at"]
    assert draft["approved_skill_present"] is False
    assert draft["install_target"] is None
    assert draft["install_base"] is None
    assert draft["install_error"] is None


def test_list_reports_only_known_approvers(server, served):
    _draft(served, "focused-validation")
    _approve(served, "focused-validation")
    _edit_manifest("focused-validation", approved_by="admin")

    assert _only_draft(server)["approved_by"] is None


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges")
def test_list_leaves_out_a_draft_folder_outside_the_root(server, served, tmp_path):
    _draft(served, "focused-validation")
    root = autoskill_proposals_dir(paths.MEMORIES_DIR)
    outside = tmp_path / "outside"
    shutil.copytree(root / "focused-validation", outside)
    manifest = json.loads((outside / "manifest.json").read_text(encoding="utf-8"))
    manifest["proposal_id"] = "linked"
    (outside / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "linked").symlink_to(outside, target_is_directory=True)

    body = server.get("/api/autoskills/proposals").json()

    assert [d["proposal_id"] for d in body["proposals"]] == ["focused-validation"]
    assert body["counts"]["pending"] == 1


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges")
@pytest.mark.parametrize("name", ["SKILL.md", "RATIONALE.md"])
def test_list_does_not_inline_a_text_linked_outside(server, served, tmp_path, name):
    _draft(served, "focused-validation")
    secret_file = tmp_path / "secret.txt"
    secret_file.write_text("do not show\n", encoding="utf-8")
    folder = autoskill_proposals_dir(paths.MEMORIES_DIR) / "focused-validation"
    (folder / name).unlink()
    (folder / name).symlink_to(secret_file)

    draft = _only_draft(server)

    assert "do not show" not in draft["skill_md"] + draft["rationale_md"]
    sizes = {f["path"]: f["size"] for f in draft["files"]}
    assert sizes.get("SKILL.md") != len("do not show\n")


def test_list_has_exactly_the_contract_fields(server, served):
    _draft(served, "focused-validation")

    assert set(_only_draft(server)) == {
        "proposal_id", "skill_name", "description", "operation",
        "target_skill_name", "scope", "submitted_scope", "status",
        "workspace_dir", "created_at", "updated_at", "approved_at",
        "approved_by", "rejected_at", "approved_skill_path",
        "approved_skill_present", "upstream_source", "workspace_shadow",
        "source_observation_ids", "skill_md", "skill_md_truncated",
        "rationale_md", "rationale_md_truncated", "files", "install_target",
        "install_base", "install_error",
    }  # fmt: skip


def test_list_reports_rejections_and_auto_approvals(server, served):
    from EvoScientist.memory.autoskills.proposals import reject_skill_proposal

    _draft(served, "rejected-skill")
    reject_skill_proposal(paths.MEMORIES_DIR, "rejected-skill")
    _draft(served, "auto-skill")
    _approve(served, "auto-skill")
    _edit_manifest("auto-skill", approved_by="auto")

    drafts = {
        d["proposal_id"]: d
        for d in server.get("/api/autoskills/proposals").json()["proposals"]
    }

    assert drafts["rejected-skill"]["status"] == "rejected"
    assert drafts["rejected-skill"]["rejected_at"]
    assert drafts["auto-skill"]["approved_by"] == "auto"
    assert drafts["auto-skill"]["approved_skill_present"] is True
    assert drafts["auto-skill"]["approved_skill_path"] == str(
        served.skills_dir / "auto-skill"
    )


def test_empty_status_is_no_filter(server, served):
    _draft(served, "focused-validation")

    body = server.get("/api/autoskills/proposals?status=").json()

    assert [d["proposal_id"] for d in body["proposals"]] == ["focused-validation"]


def test_list_leaves_out_a_copied_draft_folder(server, served):
    _draft(served, "focused-validation")
    root = autoskill_proposals_dir(paths.MEMORIES_DIR)
    shutil.copytree(root / "focused-validation", root / "focused-validation-bak")
    copy_md = root / "focused-validation-bak" / "SKILL.md"
    copy_md.write_text(copy_md.read_text(encoding="utf-8") + "EDITED COPY\n", "utf-8")

    body = server.get("/api/autoskills/proposals").json()

    assert [d["proposal_id"] for d in body["proposals"]] == ["focused-validation"]
    assert "EDITED COPY" not in body["proposals"][0]["skill_md"]
    assert body["counts"]["pending"] == 1


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges")
def test_list_leaves_out_a_manifest_linked_outside(server, served, tmp_path):
    _draft(served, "focused-validation")
    folder = autoskill_proposals_dir(paths.MEMORIES_DIR) / "focused-validation"
    outside = tmp_path / "manifest.json"
    shutil.move(folder / "manifest.json", outside)
    (folder / "manifest.json").symlink_to(outside)

    body = server.get("/api/autoskills/proposals").json()

    assert body["proposals"] == []
    assert body["counts"]["pending"] == 0


def test_list_leaves_out_drafts_of_a_missing_project(server, served, tmp_path):
    _draft(served, "focused-validation")
    shutil.rmtree(served.root)

    body = server.get("/api/autoskills/proposals").json()

    assert body["proposals"] == []
    assert body["by_project"] == {
        served.key: {"pending": 0, "approved": 0, "rejected": 0}
    }
    assert _file(server, "focused-validation", "SKILL.md").status_code == 404


def test_list_leaves_out_a_draft_with_an_unknown_scope(server, served):
    _draft(served, "focused-validation")
    _edit_manifest("focused-validation", scope="project")

    assert server.get("/api/autoskills/proposals").json()["proposals"] == []


def test_one_bad_draft_does_not_fail_the_list(server, served):
    _draft(served, "good-skill")
    _draft(served, "bad-skill")
    _draft(served, "bad-manifest")
    root = autoskill_proposals_dir(paths.MEMORIES_DIR)
    (root / "bad-skill" / "SKILL.md").write_bytes(b"---\nname: bad\xff\n---\n")
    (root / "bad-manifest" / "manifest.json").write_bytes(b'{"x": "\xff"}')

    response = server.get("/api/autoskills/proposals")

    assert response.status_code == 200
    drafts = {d["proposal_id"]: d for d in response.json()["proposals"]}
    assert set(drafts) == {"good-skill", "bad-skill"}
    assert drafts["bad-skill"]["install_error"]
    assert "\ufffd" in drafts["bad-skill"]["skill_md"]


def test_list_leaves_out_drafts_it_cannot_send(server, served):
    _draft(served, "good-skill")
    _draft(served, "surrogate-skill")
    _draft(served, "nested-skill")
    _edit_manifest("surrogate-skill", description="bad \ud800 text")
    nested = autoskill_proposals_dir(paths.MEMORIES_DIR) / "nested-skill"
    (nested / "manifest.json").write_text("[" * 100_000 + "]" * 100_000, "utf-8")

    response = server.get("/api/autoskills/proposals")

    assert response.status_code == 200
    body = response.json()
    assert [d["proposal_id"] for d in body["proposals"]] == ["good-skill"]
    assert body["counts"]["pending"] == 1


def test_list_survives_a_deeply_nested_frontmatter(server, served):
    _draft(served, "focused-validation")
    skill_md = (
        autoskill_proposals_dir(paths.MEMORIES_DIR) / "focused-validation" / "SKILL.md"
    )
    deep = "[" * 5000 + "]" * 5000
    skill_md.write_text(
        f"---\nname: focused-validation\ndescription: d\nmetadata: {deep}\n---\n",
        encoding="utf-8",
    )

    response = server.get("/api/autoskills/proposals")

    assert response.status_code == 200
    assert response.json()["proposals"][0]["install_error"]


def test_counts_do_not_depend_on_the_filter(server, served, monkeypatch):
    from EvoScientist.langgraph_dev import autoskills_api

    _draft(served, "good-skill")
    _draft(served, "broken-skill")
    _approve(served, "broken-skill")
    real = autoskills_api._draft

    def failing(proposal, key):
        if proposal.proposal_id == "broken-skill":
            raise OSError("changed on disk")
        return real(proposal, key)

    monkeypatch.setattr(autoskills_api, "_draft", failing)
    bodies = [
        server.get("/api/autoskills/proposals", params=params).json()
        for params in ({}, {"status": "pending"}, {"status": "approved"})
    ]

    assert {str(body["counts"]) for body in bodies} == {
        str({"pending": 1, "approved": 1, "rejected": 0})
    }
    assert [d["proposal_id"] for d in bodies[0]["proposals"]] == ["good-skill"]
    assert bodies[2]["proposals"] == []


def test_a_filtered_list_builds_only_the_drafts_it_returns(server, served, monkeypatch):
    from EvoScientist.langgraph_dev import autoskills_api

    _draft(served, "good-skill")
    _draft(served, "other-skill")
    _approve(served, "other-skill")
    built = []
    real = autoskills_api._draft

    def recording(proposal, key):
        built.append(proposal.proposal_id)
        return real(proposal, key)

    monkeypatch.setattr(autoskills_api, "_draft", recording)

    body = server.get("/api/autoskills/proposals", params={"status": "pending"}).json()

    assert built == ["good-skill"]
    assert body["counts"] == {"pending": 1, "approved": 1, "rejected": 0}


def test_list_scans_installed_skills_once_per_request(server, served, monkeypatch):
    from EvoScientist.memory.autoskills import proposals as module

    for name in ("first-draft", "second-draft", "third-draft"):
        _draft(served, name)
    _installed_skill(served.skills_dir, "local-skill")
    # Folders named unlike their skills: only a scan of that tier finds them.
    _installed_skill(paths.GLOBAL_SKILLS_DIR, "second-draft", folder="global-folder")
    reads = []
    real = module._installed_skill_name

    def recording(skill_dir):
        reads.append(skill_dir.name)
        return real(skill_dir)

    monkeypatch.setattr(module, "_installed_skill_name", recording)

    first = _install_errors(server)
    once = len(reads)
    _installed_skill(served.skills_dir, "first-draft", folder="later-folder")
    second = _install_errors(server)

    assert first["first-draft"] is None
    assert first["third-draft"] is None
    assert "already exists" in first["second-draft"]
    # One scan per set of roots, however many drafts ask.
    assert max(reads.count(name) for name in set(reads)) <= 4
    assert once <= 8
    # A later request scans again, so it sees the skill installed meanwhile.
    assert "already exists" in second["first-draft"]


def _install_errors(server) -> dict[str, str | None]:
    body = server.get("/api/autoskills/proposals").json()
    return {d["proposal_id"]: d["install_error"] for d in body["proposals"]}


def test_list_reads_the_registry_once(server, served, monkeypatch):
    calls = []
    real = registry.REGISTRY.registered

    def registered():
        calls.append(1)
        return real() if len(calls) == 1 else [*real(), "/elsewhere"]

    monkeypatch.setattr(registry.REGISTRY, "registered", registered)
    _draft(served, "focused-validation")

    response = server.get("/api/autoskills/proposals")

    assert response.status_code == 200
    assert calls == [1]


def test_list_strips_the_recorded_project(server, served):
    _draft(served, "focused-validation")
    _edit_manifest("focused-validation", workspace_dir=f"  {served.root}  ")

    assert _only_draft(server)["workspace_dir"] == served.key


def test_list_reports_an_install_target_it_cannot_compute(server, served):
    _draft(served, "focused-validation")
    served.skills_dir.write_text("not a folder\n", encoding="utf-8")

    response = server.get("/api/autoskills/proposals")

    assert response.status_code == 200
    draft = response.json()["proposals"][0]
    assert draft["install_target"] is None
    assert draft["install_error"]


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="POSIX permissions, not root",
)
def test_list_survives_an_unreadable_rationale(server, served):
    _draft(served, "focused-validation")
    rationale = (
        autoskill_proposals_dir(paths.MEMORIES_DIR)
        / "focused-validation"
        / "RATIONALE.md"
    )
    rationale.chmod(0o000)
    try:
        response = server.get("/api/autoskills/proposals")
    finally:
        rationale.chmod(0o644)

    assert response.status_code == 200
    assert response.json()["proposals"][0]["rationale_md"] == ""


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="POSIX permissions, not root",
)
def test_list_shows_why_a_draft_folder_cannot_be_read(server, served):
    _draft(served, "focused-validation", files={"scripts/check.py": "print(1)\n"})
    locked = autoskill_proposals_dir(paths.MEMORIES_DIR) / "focused-validation"
    locked = locked / "scripts"
    locked.chmod(0o600)
    try:
        listed = server.get("/api/autoskills/proposals")
        read = _file(server, "focused-validation", "SKILL.md")
    finally:
        locked.chmod(0o755)

    assert listed.status_code == 200
    draft = listed.json()["proposals"][0]
    assert "Cannot read proposal files" in draft["install_error"]
    assert draft["files"] == []
    assert read.status_code == 404
    assert read.json() == {"error": "not_found"}


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="POSIX permissions, not root",
)
def test_list_survives_an_installed_skill_it_cannot_search(server, served):
    base = paths.GLOBAL_SKILLS_DIR / "paper-review"
    _installed_skill(paths.GLOBAL_SKILLS_DIR, "paper-review")
    (base / "scripts").mkdir()
    (base / "scripts" / "new.py").write_text("print(1)\n", encoding="utf-8")
    _draft(
        served,
        "paper-review",
        operation="update",
        files={"scripts/new.py": "print(2)\n"},
    )
    (base / "scripts").chmod(0o600)
    try:
        listed = server.get("/api/autoskills/proposals")
        read = _file(server, "paper-review", "scripts/new.py", installed="1")
    finally:
        (base / "scripts").chmod(0o755)

    assert listed.status_code == 200
    files = {f["path"]: f for f in listed.json()["proposals"][0]["files"]}
    assert files["scripts/new.py"]["installed"] is False
    assert read.status_code == 404


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="POSIX permissions, not root",
)
def test_list_keeps_an_approval_whose_skill_it_cannot_search(server, served):
    _draft(served, "focused-validation")
    _approve(served, "focused-validation")
    served.skills_dir.chmod(0o600)
    try:
        listed = server.get("/api/autoskills/proposals")
    finally:
        served.skills_dir.chmod(0o755)

    assert listed.status_code == 200
    assert listed.json()["counts"]["approved"] == 1
    (draft,) = listed.json()["proposals"]
    assert draft["approved_skill_present"] is False


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges")
def test_list_never_reports_the_size_of_a_draft_link_outside(server, served, tmp_path):
    _draft(served, "focused-validation")
    outside = tmp_path / "outside.md"
    outside.write_text("x" * 12345, encoding="utf-8")
    folder = autoskill_proposals_dir(paths.MEMORIES_DIR) / "focused-validation"
    (folder / "big.md").symlink_to(outside)

    draft = _only_draft(server)

    files = {f["path"]: f for f in draft["files"]}
    assert files["big.md"]["size"] != 12345
    assert "outside its folder" in draft["install_error"]


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges")
def test_installed_flag_never_follows_a_link_out_of_the_skill(server, served, tmp_path):
    _installed_skill(paths.GLOBAL_SKILLS_DIR, "paper-review")
    _draft(served, "paper-review", operation="update", files={"notes.md": "n\n"})
    outside = tmp_path / "outside.md"
    outside.write_text("outside\n", encoding="utf-8")
    (paths.GLOBAL_SKILLS_DIR / "paper-review" / "notes.md").symlink_to(outside)

    draft = _only_draft(server)

    assert draft["install_base"] is not None
    files = {f["path"]: f for f in draft["files"]}
    assert files["SKILL.md"]["installed"] is True
    assert files["notes.md"]["installed"] is False


def test_handlers_read_off_the_event_loop(server, served, monkeypatch):
    import asyncio

    from EvoScientist.langgraph_dev import autoskills_api

    _draft(served, "focused-validation")
    on_loop = []

    def recording(real):
        def call(*args, **kwargs):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                on_loop.append(False)
            else:
                on_loop.append(True)
            return real(*args, **kwargs)

        return call

    monkeypatch.setattr(
        autoskills_api, "list_drafts_body", recording(autoskills_api.list_drafts_body)
    )
    monkeypatch.setattr(
        autoskills_api, "read_draft_file", recording(autoskills_api.read_draft_file)
    )
    server.get("/api/autoskills/proposals")
    _file(server, "focused-validation", "SKILL.md")

    assert on_loop == [False, False]


# ---------------------------------------------------------------------------
# GET /api/autoskills/proposals/{id}/file
# ---------------------------------------------------------------------------


def _file(server, name: str, path: str, **params):
    return server.get(
        f"/api/autoskills/proposals/{name}/file", params={"path": path, **params}
    )


def test_file_route_returns_text(server, served):
    _draft(served, "focused-validation", files={"scripts/check.py": "print(1)\n"})

    response = _file(server, "focused-validation", "scripts/check.py")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "charset=utf-8" in response.headers["content-type"]
    assert response.text == "print(1)\n"


@pytest.mark.parametrize(
    "path",
    [
        "../manifest.json",
        "/etc/hosts",
        "",
        "a/../SKILL.md",
        "..\\manifest.json",
        "C:SKILL.md",
        "C:/x/SKILL.md",
    ],
)
def test_file_route_refuses_traversal(server, served, path):
    _draft(served, "focused-validation")

    response = _file(server, "focused-validation", path)

    assert response.status_code == 400
    assert response.json() == {"error": "invalid_path"}


def test_file_route_serves_only_copied_files_and_the_rationale(server, served):
    _draft(served, "focused-validation")

    for path in ("manifest.json", "missing.md", "docs/RATIONALE.md"):
        response = _file(server, "focused-validation", path)
        assert response.status_code == 404
        assert response.json() == {"error": "not_found"}

    rationale = _file(server, "focused-validation", "RATIONALE.md")

    assert rationale.status_code == 200
    assert rationale.text == "Three observations describe focused-validation.\n"


def test_file_route_has_no_installed_rationale(server, served):
    _installed_skill(served.skills_dir, "paper-review")
    (served.skills_dir / "paper-review" / "RATIONALE.md").write_text(
        "Installed notes.\n", encoding="utf-8"
    )
    _draft(served, "paper-review", operation="update")

    response = _file(server, "paper-review", "RATIONALE.md", installed=1)

    assert response.status_code == 404
    assert response.json() == {"error": "not_found"}


def test_file_route_reads_a_decided_draft(server, served, secret):
    _draft(served, "focused-validation")
    _decide(server, "focused-validation", "reject", secret)

    skill_md = _file(server, "focused-validation", "SKILL.md")
    rationale = _file(server, "focused-validation", "RATIONALE.md")

    assert skill_md.status_code == 200
    assert skill_md.text.startswith("---\nname: focused-validation")
    assert rationale.status_code == 200


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges")
def test_file_route_refuses_a_rationale_linked_outside(server, served, tmp_path):
    _draft(served, "focused-validation")
    secret_file = tmp_path / "secret.txt"
    secret_file.write_text("do not show\n", encoding="utf-8")
    folder = autoskill_proposals_dir(paths.MEMORIES_DIR) / "focused-validation"
    (folder / "RATIONALE.md").unlink()
    (folder / "RATIONALE.md").symlink_to(secret_file)

    response = _file(server, "focused-validation", "RATIONALE.md")

    assert response.status_code == 400
    assert "do not show" not in response.text


def test_file_route_reports_a_missing_rationale(server, served):
    _draft(served, "focused-validation")
    folder = autoskill_proposals_dir(paths.MEMORIES_DIR) / "focused-validation"
    (folder / "RATIONALE.md").unlink()

    response = _file(server, "focused-validation", "RATIONALE.md")

    assert response.status_code == 404


def test_file_route_unknown_draft_or_project(server, served, projects):
    _draft(served, "focused-validation")

    assert _file(server, "no-such", "SKILL.md").status_code == 404
    assert _file(server, "Bad..Id", "SKILL.md").status_code == 404
    projects.set_registered(served.key, False)
    assert _file(server, "focused-validation", "SKILL.md").status_code == 404


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges")
def test_file_route_refuses_symlink_out(server, served, tmp_path):
    _draft(served, "focused-validation")
    outside = tmp_path / "outside.py"
    outside.write_text("print('outside')\n", encoding="utf-8")
    folder = autoskill_proposals_dir(paths.MEMORIES_DIR) / "focused-validation"
    (folder / "scripts").mkdir()
    (folder / "scripts" / "link.py").symlink_to(outside)

    response = _file(server, "focused-validation", "scripts/link.py")

    assert response.status_code == 400
    assert response.json() == {"error": "invalid_path"}


def test_file_route_binary_and_large(server, served):
    _draft(served, "focused-validation")
    folder = autoskill_proposals_dir(paths.MEMORIES_DIR) / "focused-validation"
    (folder / "blob.bin").write_bytes(b"abc\x00def")
    (folder / "latin.txt").write_bytes("caf\xe9".encode("latin-1"))
    (folder / "big.txt").write_text("x" * (MAX_FILE_BYTES + 1), encoding="utf-8")

    assert _file(server, "focused-validation", "blob.bin").status_code == 415
    assert _file(server, "focused-validation", "latin.txt").json() == {
        "error": "binary_file"
    }
    big = _file(server, "focused-validation", "big.txt")
    assert big.status_code == 413
    assert big.json() == {"error": "too_large"}


def test_a_file_whose_read_fails_after_its_open_is_not_found(tmp_path, monkeypatch):
    from EvoScientist.langgraph_dev import autoskills_api

    path = tmp_path / "SKILL.md"
    path.write_text("Steps.\n", encoding="utf-8")

    class FailingFstat:
        """The module's ``os``, but ``fstat`` fails; pytest's own ``os`` stays."""

        def __getattr__(self, name):
            return getattr(os, name)

        @staticmethod
        def fstat(_fd):
            raise OSError(5, "Input/output error")

    from EvoScientist.memory.autoskills import proposals

    monkeypatch.setattr(proposals, "os", FailingFstat())

    assert autoskills_api._read_text(path) == (404, {"error": "not_found"})


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"abc", ("abc", False)),
        (b"abcd", ("abc", True)),
        (("\U0001f600" * 3 + "a").encode("utf-8"), ("\U0001f600" * 3, True)),
    ],
)
def test_inline_text_is_cut_at_the_character_limit(
    tmp_path, monkeypatch, data, expected
):
    from EvoScientist.langgraph_dev import autoskills_api

    monkeypatch.setattr(autoskills_api, "MAX_INLINE_CHARS", 3)
    path = tmp_path / "SKILL.md"
    path.write_bytes(data)

    assert autoskills_api._inline(path, tmp_path.resolve()) == expected


def test_file_route_installed_copy(server, served):
    _installed_skill(paths.GLOBAL_SKILLS_DIR, "paper-review", body="Global workflow.")
    _draft(served, "paper-review", operation="update", body="Updated workflow.")
    _draft(served, "focused-validation")

    installed = _file(server, "paper-review", "SKILL.md", installed="1")
    draft = _file(server, "paper-review", "SKILL.md")
    create = _file(server, "focused-validation", "SKILL.md", installed="1")
    missing = _file(server, "paper-review", "scripts/none.py", installed="1")

    assert installed.status_code == 200
    assert "Global workflow." in installed.text
    assert "Updated workflow." in draft.text
    assert create.status_code == 404
    assert missing.status_code == 404


def test_file_route_installed_copy_only_for_pending_drafts(server, served):
    _installed_skill(served.skills_dir, "paper-review", body="Old workflow.")
    _draft(served, "paper-review", operation="update", body="Updated workflow.")
    _approve(served, "paper-review")

    response = _file(server, "paper-review", "SKILL.md", installed="1")

    assert response.status_code == 404
    assert response.json() == {"error": "not_found"}


def test_read_routes_need_no_secret(server, served):
    _draft(served, "focused-validation")

    assert local_secret.HEADER not in server.headers
    assert server.get("/api/autoskills/proposals").status_code == 200
    assert _file(server, "focused-validation", "SKILL.md").status_code == 200


# ---------------------------------------------------------------------------
# POST /api/autoskills/proposals/{id}/approve | reject | revert
# ---------------------------------------------------------------------------

_APPROVED_KEYS = {
    "approved",
    "proposal_id",
    "skill_name",
    "operation",
    "path",
    "scope",
    "workspace_shadow",
    "upstream_source",
    "updated_at",
}


@pytest.fixture
def secret(server) -> dict[str, str]:
    value = (paths.SERVER_HOME / local_secret.FILE_NAME).read_text()
    return {local_secret.HEADER: value}


@pytest.fixture
def quiet_skill_index(monkeypatch):
    """Approving and reverting refresh the skill index; no index here."""
    monkeypatch.setattr(
        "EvoScientist.memory.autoskills.proposals._notify_approved_skill",
        lambda: None,
    )


def _decide(server, name: str, action: str, headers=None, **body):
    return server.post(
        f"/api/autoskills/proposals/{name}/{action}",
        json=body,
        headers=headers or {},
    )


@pytest.mark.parametrize("action", ["approve", "reject", "revert"])
def test_decision_routes_need_the_secret(server, served, action):
    _draft(served, "focused-validation")

    response = _decide(server, "focused-validation", action)

    assert response.status_code == 403
    assert _only_draft(server)["status"] == "pending"


def test_approve_route_installs_into_the_drafts_project(
    server, served, secret, quiet_skill_index
):
    _draft(served, "focused-validation")

    response = _decide(server, "focused-validation", "approve", secret)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == _APPROVED_KEYS
    assert body["approved"] is True
    assert body["scope"] == "workspace"
    assert body["path"] == str(served.skills_dir / "focused-validation")
    assert (served.skills_dir / "focused-validation" / "SKILL.md").is_file()
    draft = _only_draft(server)
    assert draft["status"] == "approved"
    assert draft["approved_by"] == "user"
    assert body["updated_at"] == draft["updated_at"]


def test_approve_route_global_scope(server, served, secret, quiet_skill_index):
    _draft(served, "focused-validation")

    response = _decide(server, "focused-validation", "approve", secret, scope="global")

    assert response.status_code == 200
    assert response.json()["scope"] == "global"
    assert (paths.GLOBAL_SKILLS_DIR / "focused-validation" / "SKILL.md").is_file()
    assert not (served.skills_dir / "focused-validation").exists()


@pytest.mark.parametrize("scope", ["bogus", "Global", 5, ["workspace"]])
def test_approve_route_refuses_an_unknown_scope(server, served, secret, scope):
    _draft(served, "focused-validation")

    response = _decide(server, "focused-validation", "approve", secret, scope=scope)

    assert response.status_code == 400
    assert response.json() == {"error": "invalid_scope"}
    assert _only_draft(server)["status"] == "pending"


def test_approve_route_refuses_a_draft_changed_since_the_page_read_it(
    server, served, secret, quiet_skill_index
):
    _draft(served, "focused-validation")
    current = _only_draft(server)["updated_at"]
    installed = served.skills_dir / "focused-validation"

    stale = _decide(
        server,
        "focused-validation",
        "approve",
        secret,
        expected_updated_at="2000-01-01T00:00:00Z",
    )

    assert stale.status_code == 409
    assert stale.json() == {"error": "draft_changed", "updated_at": current}
    assert not installed.exists()
    assert _only_draft(server)["status"] == "pending"

    fresh = _decide(
        server, "focused-validation", "approve", secret, expected_updated_at=current
    )

    assert fresh.status_code == 200
    assert (installed / "SKILL.md").is_file()


@pytest.mark.parametrize("expected", [5, ["2026-11-06"], {"at": "now"}])
def test_approve_route_refuses_an_expected_updated_at_that_is_not_text(
    server, served, secret, expected
):
    _draft(served, "focused-validation")

    response = _decide(
        server, "focused-validation", "approve", secret, expected_updated_at=expected
    )

    assert response.status_code == 400
    assert response.json() == {"error": "invalid_body"}
    assert _only_draft(server)["status"] == "pending"


def test_approve_route_reports_a_decided_draft_before_a_changed_one(
    server, served, secret
):
    _draft(served, "focused-validation")
    _decide(server, "focused-validation", "reject", secret)

    response = _decide(
        server,
        "focused-validation",
        "approve",
        secret,
        expected_updated_at="2000-01-01T00:00:00Z",
    )

    assert response.status_code == 409
    assert response.json() == {"error": "not_pending", "status": "rejected"}


def test_approve_route_refuses_a_resubmit_between_its_lookup_and_the_approval(
    server, served, secret, monkeypatch
):
    from EvoScientist.langgraph_dev import autoskills_api

    _draft(served, "focused-validation")
    read = _only_draft(server)["updated_at"]
    real_apply = autoskills_api._apply

    def resubmit_then_apply(*args, **kwargs):
        _draft(served, "focused-validation", body="Rewritten.")
        return real_apply(*args, **kwargs)

    monkeypatch.setattr(autoskills_api, "_apply", resubmit_then_apply)
    response = _decide(
        server, "focused-validation", "approve", secret, expected_updated_at=read
    )

    assert response.status_code == 409
    assert response.json()["error"] == "draft_changed"
    assert response.json()["updated_at"] != read
    assert not (served.skills_dir / "focused-validation").exists()


def test_reject_route_refuses_a_draft_changed_since_the_page_read_it(
    server, served, secret
):
    _draft(served, "focused-validation")
    current = _only_draft(server)["updated_at"]

    stale = _decide(
        server,
        "focused-validation",
        "reject",
        secret,
        expected_updated_at="2000-01-01T00:00:00Z",
    )

    assert stale.status_code == 409
    assert stale.json() == {"error": "draft_changed", "updated_at": current}
    assert _only_draft(server)["status"] == "pending"
    assert processed_cluster_hashes(paths.MEMORIES_DIR) == set()

    fresh = _decide(
        server, "focused-validation", "reject", secret, expected_updated_at=current
    )

    assert fresh.status_code == 200
    assert _only_draft(server)["status"] == "rejected"


def test_reject_route_refuses_a_resubmit_between_its_lookup_and_the_rejection(
    server, served, secret, monkeypatch
):
    from EvoScientist.langgraph_dev import autoskills_api

    _draft(served, "focused-validation")
    read = _only_draft(server)["updated_at"]
    real_apply = autoskills_api._apply

    def resubmit_then_apply(*args, **kwargs):
        _draft(served, "focused-validation", body="Rewritten.")
        return real_apply(*args, **kwargs)

    monkeypatch.setattr(autoskills_api, "_apply", resubmit_then_apply)
    response = _decide(
        server, "focused-validation", "reject", secret, expected_updated_at=read
    )

    assert response.status_code == 409
    assert response.json()["error"] == "draft_changed"
    assert response.json()["updated_at"] != read
    assert processed_cluster_hashes(paths.MEMORIES_DIR) == set()


@pytest.mark.parametrize("expected", [5, ["2026-11-06"], {"at": "now"}])
def test_reject_route_refuses_an_expected_updated_at_that_is_not_text(
    server, served, secret, expected
):
    _draft(served, "focused-validation")

    response = _decide(
        server, "focused-validation", "reject", secret, expected_updated_at=expected
    )

    assert response.status_code == 400
    assert response.json() == {"error": "invalid_body"}
    assert _only_draft(server)["status"] == "pending"


def test_reject_route_reports_a_decided_draft_before_a_changed_one(
    server, served, secret, quiet_skill_index
):
    _draft(served, "focused-validation")
    _decide(server, "focused-validation", "approve", secret)

    response = _decide(
        server,
        "focused-validation",
        "reject",
        secret,
        expected_updated_at="2000-01-01T00:00:00Z",
    )

    assert response.status_code == 409
    assert response.json() == {"error": "not_pending", "status": "approved"}


def test_a_stale_undo_of_an_approval_removes_nothing(
    server, served, secret, quiet_skill_index
):
    _draft(served, "focused-validation")
    _decide(server, "focused-validation", "approve", secret)

    stale = _decide(
        server,
        "focused-validation",
        "revert",
        secret,
        expected_updated_at="2000-01-01T00:00:00Z",
    )

    assert stale.status_code == 409
    assert stale.json()["error"] == "draft_changed"
    assert (served.skills_dir / "focused-validation" / "SKILL.md").is_file()
    assert processed_cluster_hashes(paths.MEMORIES_DIR) == {
        "cluster-focused-validation"
    }
    assert _only_draft(server)["status"] == "approved"


@pytest.mark.parametrize("action", ["approve", "reject"])
def test_the_undo_a_decision_names_applies_once(
    server, served, secret, quiet_skill_index, action
):
    _draft(served, "focused-validation")
    decided = _decide(server, "focused-validation", action, secret).json()

    first = _decide(
        server,
        "focused-validation",
        "revert",
        secret,
        expected_updated_at=decided["updated_at"],
    )
    second = _decide(
        server,
        "focused-validation",
        "revert",
        secret,
        expected_updated_at=decided["updated_at"],
    )

    assert first.status_code == 200
    assert second.status_code == 409
    assert second.json() == {
        "error": "draft_changed",
        "updated_at": first.json()["updated_at"],
    }
    assert _only_draft(server)["status"] == "pending"


@pytest.mark.parametrize(
    ("expected", "status", "error"),
    [("2000-01-01T00:00:00Z", 409, "draft_changed"), (5, 400, "invalid_body")],
)
def test_revert_refuses_a_stale_expected_updated_at(
    server, served, secret, expected, status, error
):
    _draft(served, "focused-validation")
    _decide(server, "focused-validation", "reject", secret)

    reverted = _decide(
        server, "focused-validation", "revert", secret, expected_updated_at=expected
    )

    assert reverted.status_code == status
    assert reverted.json()["error"] == error


def test_approve_route_picks_the_exact_id(server, served, secret, quiet_skill_index):
    _draft(served, "paper-review")
    _draft(served, "paper-review-v2")

    response = _decide(server, "paper-review", "approve", secret)

    assert response.status_code == 200
    assert response.json()["proposal_id"] == "paper-review"
    statuses = {
        d["proposal_id"]: d["status"]
        for d in server.get("/api/autoskills/proposals").json()["proposals"]
    }
    assert statuses == {"paper-review": "approved", "paper-review-v2": "pending"}


def test_approve_route_not_pending(server, served, secret, quiet_skill_index):
    _draft(served, "focused-validation")
    _decide(server, "focused-validation", "approve", secret)

    again = _decide(server, "focused-validation", "approve", secret)
    rejected = _decide(server, "focused-validation", "reject", secret)

    assert again.status_code == 409
    assert again.json() == {"error": "not_pending", "status": "approved"}
    assert rejected.status_code == 409
    assert rejected.json() == {"error": "not_pending", "status": "approved"}


@pytest.mark.parametrize("gone", ["unregistered", "folder_missing"])
def test_decisions_on_a_project_not_served(server, served, secret, projects, gone):
    _draft(served, "focused-validation")
    if gone == "unregistered":
        projects.set_registered(served.key, False)
    else:
        shutil.rmtree(served.root)

    for action in ("approve", "reject", "revert"):
        response = _decide(server, "focused-validation", action, secret)
        assert response.status_code == 409
        assert response.json() == {"error": "project_not_served"}
    assert not (served.skills_dir / "focused-validation").exists()


def test_decisions_on_a_draft_without_a_project(server, served, secret):
    _draft(served, "focused-validation")
    _edit_manifest("focused-validation", workspace_dir="")

    response = _decide(server, "focused-validation", "approve", secret)

    assert response.status_code == 409
    assert response.json() == {"error": "no_project"}


@pytest.mark.parametrize("action", ["approve", "reject", "revert"])
@pytest.mark.parametrize("name", ["no-such", "Bad..Id"])
def test_decisions_on_an_unknown_draft(server, served, secret, action, name):
    response = _decide(server, name, action, secret)

    assert response.status_code == 404
    assert response.json() == {"error": "not_found"}


@pytest.mark.parametrize("raw", [b"[1]", b"not json", b'"scope"'])
def test_decision_routes_refuse_a_body_that_is_not_an_object(
    server, served, secret, raw
):
    _draft(served, "focused-validation")

    response = server.post(
        "/api/autoskills/proposals/focused-validation/approve",
        content=raw,
        headers={**secret, "content-type": "application/json"},
    )

    assert response.status_code == 400
    assert response.json() == {"error": "invalid_body"}


def test_decision_routes_accept_an_empty_body(server, served, secret):
    _draft(served, "focused-validation")

    response = server.post(
        "/api/autoskills/proposals/focused-validation/reject", headers=secret
    )

    assert response.status_code == 200


def test_approve_route_reports_the_refusal_in_detail(server, served, secret):
    _draft(served, "focused-validation")
    _installed_skill(served.skills_dir, "focused-validation")

    response = _decide(server, "focused-validation", "approve", secret)

    assert response.status_code == 409
    body = response.json()
    assert body["error"] == "approve_failed"
    assert body["detail"].startswith("Skill already exists")


def test_approve_route_reports_why_a_draft_folder_is_invalid(server, served, secret):
    _draft(served, "focused-validation")
    skill_md = (
        autoskill_proposals_dir(paths.MEMORIES_DIR) / "focused-validation" / "SKILL.md"
    )
    skill_md.write_text(skill_md.read_text(encoding="utf-8") + "TODO\n", "utf-8")

    response = _decide(server, "focused-validation", "approve", secret)

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail.startswith("Proposal skill folder is invalid: ")
    assert "TODO" in detail


def test_reject_then_revert_routes(server, served, secret):
    _draft(served, "focused-validation")

    rejected = _decide(server, "focused-validation", "reject", secret)
    reverted = _decide(server, "focused-validation", "revert", secret)
    again = _decide(server, "focused-validation", "revert", secret)

    assert rejected.status_code == 200
    assert rejected.json() == {
        "rejected": True,
        "proposal_id": "focused-validation",
        "skill_name": "focused-validation",
        "updated_at": rejected.json()["updated_at"],
    }
    assert reverted.status_code == 200
    assert reverted.json() == {
        "reverted": True,
        "proposal_id": "focused-validation",
        "status": "pending",
        "updated_at": _only_draft(server)["updated_at"],
    }
    assert rejected.json()["updated_at"] != reverted.json()["updated_at"]
    assert again.status_code == 409
    assert again.json() == {"error": "not_decided"}
    assert _only_draft(server)["status"] == "pending"


def test_revert_route_removes_a_new_skill(server, served, secret, quiet_skill_index):
    _draft(served, "focused-validation")
    _decide(server, "focused-validation", "approve", secret)

    response = _decide(server, "focused-validation", "revert", secret)

    assert response.status_code == 200
    assert response.json()["status"] == "pending"
    assert not (served.skills_dir / "focused-validation").exists()


def test_revert_route_rejects_an_auto_approval(
    server, served, secret, quiet_skill_index
):
    _draft(served, "focused-validation")
    approved = approve_skill_proposal(
        paths.MEMORIES_DIR,
        "focused-validation",
        skills_dir=served.skills_dir,
        workspace_dir=served.root,
        approved_by="auto",
    )
    assert approved["approved"] is True, approved

    response = _decide(server, "focused-validation", "revert", secret)

    assert response.status_code == 200
    draft = _only_draft(server)
    assert response.json() == {
        "reverted": True,
        "proposal_id": "focused-validation",
        "status": "rejected",
        "updated_at": draft["updated_at"],
    }
    assert draft["status"] == "rejected"


def test_revert_route_refusals(server, served, secret, quiet_skill_index):
    _installed_skill(paths.GLOBAL_SKILLS_DIR, "paper-review")
    _draft(served, "paper-review", operation="update")
    _decide(server, "paper-review", "approve", secret)
    _draft(served, "focused-validation")
    _decide(server, "focused-validation", "approve", secret)
    skill_md = served.skills_dir / "focused-validation" / "SKILL.md"
    skill_md.write_text(skill_md.read_text(encoding="utf-8") + "Mine.\n", "utf-8")

    update = _decide(server, "paper-review", "revert", secret)
    changed = _decide(server, "focused-validation", "revert", secret)

    assert update.status_code == 409
    assert update.json() == {"error": "revert_unavailable"}
    assert changed.status_code == 409
    assert changed.json() == {"error": "skill_changed"}
    assert skill_md.is_file()


def test_revert_route_reports_a_failed_removal(server, served, secret, monkeypatch):
    from EvoScientist.langgraph_dev import autoskills_api

    _draft(served, "focused-validation")
    monkeypatch.setattr(
        autoskills_api,
        "revert_skill_proposal",
        lambda _memory_dir, proposal_id, **_: {
            "reverted": False,
            "proposal_id": proposal_id,
            "error": "revert_failed",
            "detail": "file in use",
        },
    )

    response = _decide(server, "focused-validation", "revert", secret)

    assert response.status_code == 409
    assert response.json() == {"error": "revert_failed", "detail": "file in use"}


@pytest.mark.parametrize("action", ["approve", "reject", "revert"])
@pytest.mark.parametrize("error", [OSError, ValueError, RuntimeError])
def test_a_decision_that_raises_is_reported_not_500(
    server, served, secret, monkeypatch, action, error
):
    from EvoScientist.langgraph_dev import autoskills_api

    _draft(served, "focused-validation")

    def raising(*_args, **_kwargs):
        raise error("disk full")

    monkeypatch.setattr(autoskills_api, f"{action}_skill_proposal", raising)

    response = _decide(server, "focused-validation", action, secret)

    assert response.status_code == 409
    assert response.json() == {"error": f"{action}_failed", "detail": "disk full"}


def test_a_refusal_naming_an_unencodable_path_is_still_sent(
    server, served, secret, monkeypatch
):
    from EvoScientist.langgraph_dev import autoskills_api

    _draft(served, "focused-validation")
    monkeypatch.setattr(
        autoskills_api,
        "approve_skill_proposal",
        lambda *args, **kwargs: {
            "approved": False,
            "error": "Skill already exists: /skills/caf\udce9",
        },
    )

    response = _decide(server, "focused-validation", "approve", secret)

    assert response.status_code == 409
    assert response.json() == {
        "error": "approve_failed",
        "detail": "Skill already exists: /skills/caf\\udce9",
    }


def test_approve_route_installs_into_the_drafts_own_project(
    server, workspace, secret, projects, tmp_path, quiet_skill_index
):
    other = Workspace(tmp_path / "aaa-other")
    other.root.mkdir()
    projects.set_registered(other.key, True)
    projects.set_registered(workspace.key, True)
    _draft(workspace, "focused-validation")

    response = _decide(server, "focused-validation", "approve", secret)

    assert response.status_code == 200
    assert response.json()["path"] == str(workspace.skills_dir / "focused-validation")
    assert not (other.skills_dir / "focused-validation").exists()


def test_approve_route_reports_the_tier_and_the_shadow(
    server, served, secret, quiet_skill_index
):
    _installed_skill(paths.GLOBAL_SKILLS_DIR, "paper-review", body="Global.")
    _draft(served, "paper-review", operation="update", body="Updated.")
    _installed_skill(served.skills_dir, "paper-review", body="Local copy.")

    body = _decide(server, "paper-review", "approve", secret).json()

    assert body["scope"] == "global"
    assert body["workspace_shadow"] == str(served.skills_dir / "paper-review")


def test_approve_route_reports_the_upstream_source(
    server, served, secret, quiet_skill_index
):
    from EvoScientist.tools.skills_manager import _save_manifest

    _installed_skill(served.skills_dir, "paper-review", body="Old.")
    _save_manifest(served.skills_dir, {"paper-review": {"source": "owner/repo"}})
    _draft(served, "paper-review", operation="update", body="New.")

    body = _decide(server, "paper-review", "approve", secret).json()

    assert body["upstream_source"] == "owner/repo"


def test_decision_routes_check_the_secret_before_the_body(server, served):
    _draft(served, "focused-validation")

    response = server.post(
        "/api/autoskills/proposals/focused-validation/approve",
        content=b"not json",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 403


def test_a_deeply_nested_body_is_an_invalid_body(server, served, secret):
    _draft(served, "focused-validation")

    response = server.post(
        "/api/autoskills/proposals/focused-validation/approve",
        content=b"[" * 100_000 + b"]" * 100_000,
        headers={**secret, "content-type": "application/json"},
    )

    assert response.status_code == 400
    assert response.json() == {"error": "invalid_body"}


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="POSIX permissions, not root",
)
def test_a_project_the_server_cannot_stat_is_not_served(
    server, served, secret, tmp_path
):
    _draft(served, "focused-validation")
    tmp_path.chmod(0o600)
    try:
        response = _decide(server, "focused-validation", "approve", secret)
    finally:
        tmp_path.chmod(0o755)

    assert response.status_code == 409
    assert response.json() == {"error": "project_not_served"}


@pytest.mark.parametrize("action", ["approve", "reject"])
def test_a_decision_never_falls_back_to_a_prefix_match(
    server, served, secret, monkeypatch, tmp_path, action, quiet_skill_index
):
    from EvoScientist.langgraph_dev import autoskills_api

    _draft(served, "paper-review")
    _draft(served, "paper-review-v2")
    root = autoskill_proposals_dir(paths.MEMORIES_DIR)
    real_find = autoskills_api.find_skill_proposal

    def find_then_lose_the_folder(memory_dir, proposal_id):
        found = real_find(memory_dir, proposal_id)
        shutil.move(root / "paper-review", tmp_path / "moved-away")
        return found

    monkeypatch.setattr(
        autoskills_api, "find_skill_proposal", find_then_lose_the_folder
    )

    response = _decide(server, "paper-review", action, secret)

    assert response.status_code == 409
    assert response.json()["error"] == f"{action}_failed"
    manifest = json.loads((root / "paper-review-v2" / "manifest.json").read_text())
    assert manifest["status"] == "pending"


def test_approve_route_leaves_nothing_after_a_failed_copy(
    server, served, secret, monkeypatch, quiet_skill_index
):
    from EvoScientist.memory.autoskills import proposals

    _draft(served, "focused-validation", files={"scripts/check.py": "print(1)\n"})
    calls = []
    real_copy = proposals._copy_file_atomic

    def full_disk(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise OSError(28, "No space left on device")
        return real_copy(*args, **kwargs)

    monkeypatch.setattr(proposals, "_copy_file_atomic", full_disk)

    response = _decide(server, "focused-validation", "approve", secret)

    assert response.status_code == 409
    assert response.json()["detail"].startswith("Could not install the skill: ")
    assert not (served.skills_dir / "focused-validation").exists()
    draft = _only_draft(server)
    assert draft["status"] == "pending"
    assert draft["install_error"] is None


def test_concurrent_approvals_install_once(served, monkeypatch, quiet_skill_index):
    import threading
    import time

    from EvoScientist.langgraph_dev import autoskills_api
    from EvoScientist.memory.autoskills import proposals

    _draft(served, "focused-validation")
    real_copy = proposals._copy_proposed_skill
    copies = []

    def slow_copy(*args, **kwargs):
        # Both approvals would pass the pending check before either writes.
        copies.append(1)
        time.sleep(0.2)
        return real_copy(*args, **kwargs)

    monkeypatch.setattr(proposals, "_copy_proposed_skill", slow_copy)
    results = []

    def approve():
        results.append(autoskills_api.decide("focused-validation", "approve"))

    threads = [threading.Thread(target=approve) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(status for status, _ in results) == [200, 409]
    refused = next(payload for status, payload in results if status == 409)
    assert refused == {"error": "not_pending", "status": "approved"}
    assert len(copies) == 1


def test_decisions_run_off_the_event_loop(server, served, secret, monkeypatch):
    import asyncio

    from EvoScientist.langgraph_dev import autoskills_api

    _draft(served, "focused-validation")
    on_loop = []
    real = autoskills_api.decide

    def recording(*args, **kwargs):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            on_loop.append(False)
        else:
            on_loop.append(True)
        return real(*args, **kwargs)

    monkeypatch.setattr(autoskills_api, "decide", recording)
    _decide(server, "focused-validation", "reject", secret)

    assert on_loop == [False]


# ---------------------------------------------------------------------------
# POST /api/autoskills/run
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_runs(monkeypatch):
    """Stand-ins for starting and finding AutoSkills runs on the server."""
    from unittest.mock import AsyncMock

    from EvoScientist.memory.autoskills import schedule

    start = AsyncMock(return_value={"thread_id": "t-new", "run_id": "r-new"})
    find = AsyncMock(return_value=None)
    monkeypatch.setattr(schedule, "arun_autoskill_now", start)
    monkeypatch.setattr(schedule, "afind_running_autoskill", find)
    return start, find


def _run(server, headers=None, **body):
    return server.post("/api/autoskills/run", json=body, headers=headers or {})


def test_run_route_needs_the_secret(server, served, fake_runs):
    start, _find = fake_runs

    response = _run(server, workspace_dir=str(served.root))

    assert response.status_code == 403
    start.assert_not_called()


def test_run_route_starts_a_run_for_the_project(server, served, secret, fake_runs):
    start, find = fake_runs

    response = _run(server, secret, workspace_dir=str(served.root) + "/")

    assert response.status_code == 200
    assert response.json() == {"thread_id": "t-new", "run_id": "r-new"}
    assert start.await_args.kwargs["workspace_dir"] == served.key
    assert find.await_args.kwargs["workspace_dir"] == served.key


def test_run_route_names_the_run_already_in_flight(server, served, secret, fake_runs):
    start, find = fake_runs
    find.return_value = {"thread_id": "t-old", "run_id": "r-old"}

    response = _run(server, secret, workspace_dir=str(served.root))

    assert response.status_code == 409
    assert response.json() == {
        "error": "already_running",
        "thread_id": "t-old",
        "run_id": "r-old",
    }
    start.assert_not_called()


def test_run_route_not_served(server, served, secret, projects, fake_runs):
    start, _find = fake_runs
    projects.set_registered(served.key, False)

    response = _run(server, secret, workspace_dir=str(served.root))

    assert response.status_code == 404
    assert response.json() == {"error": "not_served"}
    start.assert_not_called()


def test_run_route_when_autoskills_is_off(
    server, served, secret, fake_runs, monkeypatch
):
    from EvoScientist import config as config_module
    from EvoScientist.config import EvoScientistConfig

    start, _find = fake_runs
    monkeypatch.setattr(
        config_module,
        "get_effective_config",
        lambda *args, **kwargs: EvoScientistConfig(
            memory_skill_synthesis_enabled=False
        ),
    )

    response = _run(server, secret, workspace_dir=str(served.root))

    assert response.status_code == 409
    assert response.json() == {"error": "autoskills_off"}
    start.assert_not_called()


@pytest.mark.parametrize("value", [None, "", "relative/path", 5])
def test_run_route_refuses_a_path_it_cannot_read(
    server, served, secret, fake_runs, value
):
    response = _run(server, secret, workspace_dir=value)

    assert response.status_code == 400
    assert response.json() == {"error": "invalid_path"}


def test_run_route_refuses_a_body_that_is_not_an_object(
    server, served, secret, fake_runs
):
    response = server.post(
        "/api/autoskills/run",
        content=b"[1]",
        headers={**secret, "content-type": "application/json"},
    )

    assert response.status_code == 400
    assert response.json() == {"error": "invalid_body"}


@pytest.mark.parametrize("failing", ["start", "find"])
def test_run_route_reports_a_failed_start(server, served, secret, fake_runs, failing):
    start, find = fake_runs
    (start if failing == "start" else find).side_effect = RuntimeError("down")

    response = _run(server, secret, workspace_dir=str(served.root))

    assert response.status_code == 502
    assert response.json() == {"error": "run_failed", "detail": "down"}
    if failing == "find":
        start.assert_not_called()


def test_run_route_sends_a_detail_holding_a_lone_surrogate(
    server, served, secret, fake_runs, caplog
):
    import logging

    start, _find = fake_runs
    start.side_effect = RuntimeError("down \udce9")
    # The logged traceback would hold the surrogate, which pytest-xdist cannot send.
    caplog.set_level(logging.ERROR, logger="EvoScientist.langgraph_dev.autoskills_api")

    response = _run(server, secret, workspace_dir=str(served.root))

    assert response.status_code == 502
    assert response.json() == {"error": "run_failed", "detail": "down \\udce9"}


def test_run_route_reports_a_config_it_cannot_read(
    server, served, secret, fake_runs, monkeypatch
):
    from EvoScientist import config as config_module

    start, _find = fake_runs

    def unreadable(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(config_module, "get_effective_config", unreadable)

    response = _run(server, secret, workspace_dir=str(served.root))

    assert response.status_code == 502
    assert response.json()["error"] == "run_failed"
    start.assert_not_called()


def test_a_slow_start_in_one_project_does_not_hold_another(
    server, served, secret, fake_runs, projects, tmp_path
):
    import asyncio
    import threading

    other = Workspace(tmp_path / "other")
    other.root.mkdir()
    projects.set_registered(other.key, True)
    start, _find = fake_runs
    entered = threading.Event()
    release = threading.Event()

    async def start_run(_config, *, workspace_dir):
        if workspace_dir == served.key:
            entered.set()
            while not release.is_set():
                await asyncio.sleep(0.01)
        return {"thread_id": f"t-{workspace_dir}", "run_id": "r"}

    start.side_effect = start_run
    slow = threading.Thread(
        target=lambda: _run(server, secret, workspace_dir=str(served.root))
    )
    slow.start()
    assert entered.wait(5)
    fast_done = []
    fast = threading.Thread(
        target=lambda: fast_done.append(
            _run(server, secret, workspace_dir=str(other.root)).status_code
        )
    )
    fast.start()
    fast.join(5)
    done_while_held = list(fast_done)
    release.set()
    slow.join(5)
    fast.join(5)

    assert done_while_held == [200]


def test_run_route_checks_served_before_off_and_off_before_running(
    server, served, secret, fake_runs, projects, monkeypatch
):
    from EvoScientist import config as config_module
    from EvoScientist.config import EvoScientistConfig

    start, find = fake_runs
    find.return_value = {"thread_id": "t-old", "run_id": "r-old"}
    monkeypatch.setattr(
        config_module,
        "get_effective_config",
        lambda *args, **kwargs: EvoScientistConfig(
            memory_skill_synthesis_enabled=False
        ),
    )

    off = _run(server, secret, workspace_dir=str(served.root))
    projects.set_registered(served.key, False)
    unserved = _run(server, secret, workspace_dir=str(served.root))

    assert off.json() == {"error": "autoskills_off"}
    find.assert_not_called()
    assert unserved.status_code == 404
    assert unserved.json() == {"error": "not_served"}
    start.assert_not_called()


def test_run_route_treats_a_project_it_cannot_stat_as_not_served(
    server, served, secret, fake_runs, monkeypatch
):
    from EvoScientist.langgraph_dev import folder_check

    def unreadable(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(folder_check, "is_served", unreadable)

    response = _run(server, secret, workspace_dir=str(served.root))

    assert response.status_code == 404
    assert response.json() == {"error": "not_served"}


def test_run_locks_are_dropped_once_unused(server, served, secret, fake_runs):
    from EvoScientist.langgraph_dev import autoskills_api

    _start, find = fake_runs
    _run(server, secret, workspace_dir=str(served.root))
    assert autoskills_api._RUN_LOCKS == {}

    find.side_effect = RuntimeError("down")
    assert _run(server, secret, workspace_dir=str(served.root)).status_code == 502
    assert autoskills_api._RUN_LOCKS == {}


async def test_run_now_reads_the_setting_once_it_holds_the_lock(
    served, fake_runs, monkeypatch
):
    import asyncio

    from EvoScientist import config as config_module
    from EvoScientist.config import EvoScientistConfig
    from EvoScientist.langgraph_dev import autoskills_api

    start, _find = fake_runs
    enabled = [True]
    monkeypatch.setattr(
        config_module,
        "get_effective_config",
        lambda *args, **kwargs: EvoScientistConfig(
            memory_skill_synthesis_enabled=enabled[0]
        ),
    )
    entry = (asyncio.get_running_loop(), served.key)
    async with autoskills_api._run_lock(served.key):
        waiting = asyncio.create_task(autoskills_api.run_now(str(served.root)))
        while autoskills_api._RUN_LOCKS[entry][1] < 2:
            await asyncio.sleep(0.01)
        # Turned off while the request waits for another start.
        enabled[0] = False

    assert await waiting == (409, {"error": "autoskills_off"})
    start.assert_not_called()


async def test_run_locks_are_dropped_when_their_users_are_cancelled():
    import asyncio
    import contextlib

    from EvoScientist.langgraph_dev import autoskills_api

    held = asyncio.Event()

    async def holder():
        async with autoskills_api._run_lock("project"):
            held.set()
            await asyncio.sleep(30)

    async def waiter():
        async with autoskills_api._run_lock("project"):
            pass

    holding = asyncio.create_task(holder())
    await held.wait()
    waiting = asyncio.create_task(waiter())
    await asyncio.sleep(0)
    waiting.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await waiting
    holding.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await holding

    assert autoskills_api._RUN_LOCKS == {}


def test_two_run_requests_at_once_start_one_run(server, served, secret, fake_runs):
    import asyncio
    import threading

    start, find = fake_runs
    started = []

    async def slow_start(*_args, **_kwargs):
        await asyncio.sleep(0.2)
        started.append({"thread_id": "t-1", "run_id": "r-1"})
        return started[-1]

    async def find_started(*_args, **_kwargs):
        return started[0] if started else None

    start.side_effect = slow_start
    find.side_effect = find_started
    responses = []

    def request():
        responses.append(_run(server, secret, workspace_dir=str(served.root)))

    threads = [threading.Thread(target=request) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(r.status_code for r in responses) == [200, 409]
    assert len(started) == 1


def test_run_route_reads_its_files_off_the_event_loop(
    server, served, secret, fake_runs, monkeypatch
):
    import asyncio

    from EvoScientist import config as config_module

    on_loop = []
    real = config_module.get_effective_config

    def recording(function):
        def call(*args, **kwargs):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                on_loop.append(False)
            else:
                on_loop.append(True)
            return function(*args, **kwargs)

        return call

    from EvoScientist.langgraph_dev import autoskills_api, folder_check

    monkeypatch.setattr(config_module, "get_effective_config", recording(real))
    monkeypatch.setattr(
        folder_check,
        "read_workspace_or_none",
        recording(folder_check.read_workspace_or_none),
    )
    monkeypatch.setattr(autoskills_api, "_served", recording(autoskills_api._served))
    _run(server, secret, workspace_dir=str(served.root))

    assert on_loop == [False, False, False]


@pytest.fixture
def runtime_sdk(runtime_store, monkeypatch):
    """The SDK clients of the AutoSkills helpers, talking to the in-memory runtime."""
    from contextlib import asynccontextmanager

    import httpx
    from langgraph_api.api.runs import runs_routes
    from langgraph_api.api.threads import threads_routes
    from langgraph_sdk.client import LangGraphClient
    from starlette.applications import Starlette

    from EvoScientist.memory.autoskills import schedule

    app = Starlette(routes=[*threads_routes, *runs_routes])

    def http_client():
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://lg"
        )

    @asynccontextmanager
    async def scoped(_url, **_kwargs):
        http = http_client()
        try:
            yield LangGraphClient(http)
        finally:
            await http.aclose()

    monkeypatch.setattr(
        schedule,
        "get_langgraph_async_client",
        lambda **_k: LangGraphClient(http_client()),
    )
    monkeypatch.setattr(schedule, "langgraph_async_client_scoped", scoped)
    monkeypatch.setattr(
        schedule, "configured_langgraph_dev_url", lambda _c: "http://lg"
    )


async def test_the_finder_ignores_a_cron_run_and_finds_a_manual_one(
    runtime_sdk, runtime_store, workspace
):
    from langgraph_api.cron_scheduler import get_metadata_from_payload
    from langgraph_api.models.run import create_valid_run
    from langgraph_runtime.database import connect

    from EvoScientist.config import EvoScientistConfig
    from EvoScientist.langgraph_dev import server_crons
    from EvoScientist.langgraph_dev.assistants import ensure_workspace_assistants
    from EvoScientist.memory.autoskills import schedule

    await ensure_workspace_assistants(workspace)
    config = EvoScientistConfig()
    config.scheduler_default_timezone = "UTC"
    await server_crons.reconcile_autoskills(config, [workspace])
    # Fire the AutoSkills cron as the cron scheduler does: a temporary thread
    # whose run carries the cron's metadata.
    (cron,) = runtime_store["crons"]
    payload = cron["payload"]
    payload["metadata"] = get_metadata_from_payload(cron, payload)
    async with connect() as conn:
        await create_valid_run(conn, thread_id=None, payload=payload, headers={})

    assert (
        await schedule.afind_running_autoskill(config, workspace_dir=workspace.key)
        is None
    )

    started = await schedule.arun_autoskill_now(config, workspace_dir=workspace.key)

    assert await schedule.afind_running_autoskill(
        config, workspace_dir=workspace.key
    ) == {"thread_id": str(started["thread_id"]), "run_id": str(started["run_id"])}
