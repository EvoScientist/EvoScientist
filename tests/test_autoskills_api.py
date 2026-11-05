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
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Use when testing {name}.\n---\n\n{body}\n",
        encoding="utf-8",
    )
    for relative, text in (files or {}).items():
        (folder / relative).parent.mkdir(parents=True, exist_ok=True)
        (folder / relative).write_text(text, encoding="utf-8")
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


def _installed_skill(root, name: str, body: str = "Installed.") -> None:
    folder = root / name
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
        str({"pending": 1, "approved": 0, "rejected": 0})
    }
    assert [d["proposal_id"] for d in bodies[0]["proposals"]] == ["good-skill"]


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
        autoskills_api, "list_drafts", recording(autoskills_api.list_drafts)
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


def test_file_route_serves_only_files_an_approval_copies(server, served):
    _draft(served, "focused-validation")

    for path in ("manifest.json", "RATIONALE.md", "missing.md"):
        response = _file(server, "focused-validation", path)
        assert response.status_code == 404
        assert response.json() == {"error": "not_found"}


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
