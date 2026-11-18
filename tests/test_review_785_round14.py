"""Round 14 regressions: bind WebUI decisions and pin the reviewed tier."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from EvoScientist import local_secret, paths
from EvoScientist.langgraph_dev import registry
from EvoScientist.memory.autoskills.proposals import (
    approve_skill_proposal,
    autoskill_proposals_dir,
    submit_autoskill_proposal,
)


@pytest.fixture
def server(monkeypatch, no_workspace_setup):
    from starlette.testclient import TestClient

    from EvoScientist.langgraph_dev.http import app

    monkeypatch.setattr(local_secret, "_value", None)
    with TestClient(
        app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000)
    ) as client:
        yield client


@pytest.fixture
def served(workspace):
    registry.REGISTRY.set_registered(workspace.key, True)
    return workspace


@pytest.fixture
def secret(server):
    value = (paths.SERVER_HOME / local_secret.FILE_NAME).read_text()
    return {local_secret.HEADER: value}


@pytest.fixture(autouse=True)
def quiet_skill_index(monkeypatch):
    monkeypatch.setattr(
        "EvoScientist.memory.autoskills.proposals._notify_approved_skill",
        lambda: None,
    )


def _draft(workspace, name: str = "probe-skill") -> None:
    folder = autoskill_proposals_dir(paths.MEMORIES_DIR) / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Review this draft.\n---\n\nShown body.\n",
        encoding="utf-8",
    )
    result = submit_autoskill_proposal(
        memory_dir=paths.MEMORIES_DIR,
        skill_name=name,
        cluster_hash=f"cluster-{name}",
        source_observation_ids=["O-1"],
        rationale="One observation is enough.",
        skills_dir=workspace.skills_dir,
        workspace_dir=workspace.root,
    )
    assert result["submitted"] is True, result


def _listed(server, name: str = "probe-skill") -> dict:
    response = server.get("/api/autoskills/proposals")
    assert response.status_code == 200
    drafts = {draft["proposal_id"]: draft for draft in response.json()["proposals"]}
    return drafts[name]


def _decide(server, secret, name: str = "probe-skill", **body):
    return server.post(
        f"/api/autoskills/proposals/{name}/approve", json=body, headers=secret
    )


def test_web_client_without_review_token_cannot_approve(server, served, secret):
    """A second client cannot consume the first client's rendered review."""
    _draft(served)
    surface_a = _listed(server)

    surface_b = _decide(server, secret)

    assert surface_b.status_code == 409, surface_b.json()
    assert "review token" in surface_b.json()["detail"].lower()
    assert isinstance(surface_a.get("review_token"), str)
    assert surface_a["review_token"]
    assert not (served.skills_dir / "probe-skill").exists()


def test_web_client_with_wrong_review_token_cannot_approve(server, served, secret):
    _draft(served)
    surface_a = _listed(server)

    surface_b = _decide(server, secret, review_token="not-the-rendered-token")

    assert surface_b.status_code == 409, surface_b.json()
    assert "does not match" in surface_b.json()["detail"]
    assert isinstance(surface_a.get("review_token"), str)
    assert not (served.skills_dir / "probe-skill").exists()


def test_web_client_with_rendered_review_token_approves(server, served, secret):
    _draft(served)
    surface_a = _listed(server)
    token = surface_a.get("review_token")
    assert isinstance(token, str)
    assert token

    response = _decide(server, secret, review_token=token)

    assert response.status_code == 200, response.json()
    assert response.json()["approved"] is True
    installed = served.skills_dir / "probe-skill" / "SKILL.md"
    assert "Shown body." in installed.read_text(encoding="utf-8")


def test_workspace_review_cannot_be_approved_global(server, served, secret):
    _draft(served)
    surface_a = _listed(server)
    assert surface_a["scope"] == "workspace"

    response = _decide(
        server,
        secret,
        scope="global",
        review_token=surface_a.get("review_token"),
    )

    assert response.status_code == 409, response.json()
    assert "scope" in response.json()["detail"].lower()
    assert not (paths.GLOBAL_SKILLS_DIR / "probe-skill").exists()
    assert not (served.skills_dir / "probe-skill").exists()


def test_global_review_can_be_approved_global(server, served, secret):
    _draft(served)
    manifest_path = (
        autoskill_proposals_dir(paths.MEMORIES_DIR) / "probe-skill" / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["scope"] = "global"
    manifest["submitted_scope"] = "global"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    surface_a = _listed(server)
    token = surface_a.get("review_token")
    assert surface_a["scope"] == "global"
    assert isinstance(token, str)
    assert token

    response = _decide(server, secret, scope="global", review_token=token)

    assert response.status_code == 200, response.json()
    assert response.json()["scope"] == "global"
    assert (paths.GLOBAL_SKILLS_DIR / "probe-skill" / "SKILL.md").is_file()
    assert not (served.skills_dir / "probe-skill").exists()


def test_cli_review_keeps_tokenless_process_local_approval(served):
    """The single-client CLI surface keeps its existing no-token rule."""
    from EvoScientist.commands.implementation.autoskills import AutoSkillsCommand
    from EvoScientist.memory.autoskills import proposals

    _draft(served)
    mounted = []
    ctx = SimpleNamespace(
        workspace=served,
        ui=SimpleNamespace(
            append_system=lambda *args, **kwargs: None,
            mount_renderable=lambda item, **kwargs: mounted.append(item),
        ),
    )
    asyncio.run(AutoSkillsCommand().execute(ctx, ["review"]))
    assert len(mounted) == 2
    folder = autoskill_proposals_dir(paths.MEMORIES_DIR) / "probe-skill"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    recorded = proposals._reviewed_record(folder, manifest["updated_at"])
    assert recorded is not None
    assert recorded.scope == "workspace"
    assert recorded.token is None

    result = approve_skill_proposal(
        paths.MEMORIES_DIR,
        "probe-skill",
        skills_dir=served.skills_dir,
        workspace_dir=served.root,
    )

    assert result["approved"] is True, result
    assert (served.skills_dir / "probe-skill" / "SKILL.md").is_file()


def test_legacy_review_without_scope_installs_into_workspace(tmp_path, monkeypatch):
    """A pre-scope record defaults to the legacy workspace destination."""
    from EvoScientist.memory.autoskills import proposals

    memory_dir = tmp_path / "memories"
    skills_dir = tmp_path / "skills"
    monkeypatch.setattr(paths, "MEMORIES_DIR", memory_dir)
    workspace = paths.Workspace(tmp_path)
    _draft(workspace)
    folder = autoskill_proposals_dir(memory_dir) / "probe-skill"
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("scope", None)
    manifest.pop("submitted_scope", None)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    proposals.record_reviewed_proposal_by_id(memory_dir, "probe-skill")
    recorded = proposals._reviewed_record(folder, manifest["updated_at"])
    assert recorded is not None
    assert recorded.scope == "workspace"
    assert recorded.token is None
    result = approve_skill_proposal(
        memory_dir,
        "probe-skill",
        skills_dir=skills_dir,
        workspace_dir=tmp_path,
    )

    assert result["approved"] is True, result
    assert result["scope"] == "workspace"
    assert (skills_dir / "probe-skill" / "SKILL.md").is_file()


def test_single_file_web_review_uses_header_token(server, served, secret):
    """The plain-text file response carries its token in a header."""
    _draft(served)
    surface_a = server.get("/api/autoskills/proposals/probe-skill/file?path=SKILL.md")
    assert surface_a.status_code == 200

    surface_b = _decide(server, secret)

    assert surface_b.status_code == 409, surface_b.json()
    token = surface_a.headers.get("X-EvoScientist-Review-Token")
    assert isinstance(token, str)
    assert token
    assert not (served.skills_dir / "probe-skill").exists()

    response = _decide(server, secret, review_token=token)
    assert response.status_code == 200, response.json()
    assert (served.skills_dir / "probe-skill" / "SKILL.md").is_file()
