"""AutoSkills proposal validation and review lifecycle."""

from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from ... import paths

AUTOSKILL_PROPOSALS_DIR = "autoskills/proposals"
AUTOSKILL_PROCESSED_DIR = "autoskills/processed"

_SKILL_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
_SKILL_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\s*\n", re.DOTALL)
_ALLOWED_SKILL_FRONTMATTER = {
    "allowed-tools",
    "compatibility",
    "description",
    "license",
    "metadata",
    "name",
}


@dataclass(frozen=True)
class SkillProposal:
    proposal_id: str
    skill_name: str
    description: str
    status: str
    operation: str
    path: Path
    created_at: str
    updated_at: str
    cluster_hash: str
    source_observation_ids: tuple[str, ...]
    target_skill_name: str | None = None
    workspace_dir: str | None = None
    project_id: str | None = None
    approved_skill_path: str | None = None


_PROPOSAL_OPERATIONS = frozenset({"create", "update"})


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _proposal_root(memory_dir: str | Path) -> Path:
    return Path(memory_dir).expanduser() / AUTOSKILL_PROPOSALS_DIR


def _processed_root(memory_dir: str | Path) -> Path:
    return Path(memory_dir).expanduser() / AUTOSKILL_PROCESSED_DIR


def autoskill_proposals_dir(memory_dir: str | Path) -> Path:
    """Return the real directory exposed as `/autoskill-proposals/` to agents."""
    return _proposal_root(memory_dir)


def sanitize_skill_name(name: str) -> str | None:
    """Return a valid skill name or None when no valid name remains."""
    candidate = re.sub(r"[^a-z0-9-]+", "-", name.strip().lower()).strip("-")
    while "--" in candidate:
        candidate = candidate.replace("--", "-")
    if not candidate:
        return None
    candidate = candidate[:64].strip("-")
    if _SKILL_NAME_RE.fullmatch(candidate) and "--" not in candidate:
        return candidate
    return None


def proposal_virtual_path(skill_name: str) -> str:
    return f"/autoskill-proposals/{skill_name}"


def _read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _manifest_timestamp(manifest: dict[str, Any], key: str) -> str | None:
    value = manifest.get(key)
    if isinstance(value, str) and value.strip():
        return value
    return None


def _created_at_for_manifest(existing: dict[str, Any] | None) -> str:
    if existing is None:
        return _now()
    return _manifest_timestamp(existing, "created_at") or _now()


def _normalize_operation(value: object) -> str | None:
    text = str(value or "create").strip().lower()
    return text if text in _PROPOSAL_OPERATIONS else None


def _normalize_workspace_dir(workspace_dir: str | Path | None) -> str | None:
    if workspace_dir is None:
        return None
    text = str(workspace_dir).strip()
    if not text:
        return None
    return str(Path(text).expanduser().resolve())


def _proposal_from_manifest(
    path: Path, manifest: dict[str, Any]
) -> SkillProposal | None:
    created_at = _manifest_timestamp(manifest, "created_at")
    updated_at = _manifest_timestamp(manifest, "updated_at")
    if created_at is None or updated_at is None:
        return None
    try:
        source_ids = tuple(str(item) for item in manifest["source_observation_ids"])
        return SkillProposal(
            proposal_id=str(manifest["proposal_id"]),
            skill_name=str(manifest["skill_name"]),
            description=str(manifest["description"]),
            status=str(manifest["status"]),
            operation=_normalize_operation(manifest.get("operation")) or "create",
            path=path,
            created_at=created_at,
            updated_at=updated_at,
            cluster_hash=str(manifest["cluster_hash"]),
            source_observation_ids=source_ids,
            target_skill_name=(
                str(manifest["target_skill_name"])
                if manifest.get("target_skill_name")
                else None
            ),
            workspace_dir=(
                str(manifest["workspace_dir"])
                if manifest.get("workspace_dir")
                else None
            ),
            project_id=str(manifest["project_id"])
            if manifest.get("project_id")
            else None,
            approved_skill_path=(
                str(manifest["approved_skill_path"])
                if manifest.get("approved_skill_path")
                else None
            ),
        )
    except (KeyError, TypeError, ValueError):
        return None


def list_skill_proposals(
    memory_dir: str | Path,
    *,
    status: str | None = None,
    workspace_dir: str | Path | None = None,
) -> list[SkillProposal]:
    """List autoskill proposals recorded in memory."""
    proposals: list[SkillProposal] = []
    normalized_workspace = _normalize_workspace_dir(workspace_dir)
    root = _proposal_root(memory_dir)
    if not root.exists():
        return proposals
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = _read_manifest(manifest_path)
        if manifest is None:
            continue
        proposal = _proposal_from_manifest(manifest_path.parent, manifest)
        if proposal is None:
            continue
        if (
            normalized_workspace is not None
            and proposal.workspace_dir != normalized_workspace
        ):
            continue
        if status is not None and proposal.status != status:
            continue
        proposals.append(proposal)
    return proposals


def pending_skill_proposal_count(
    memory_dir: str | Path,
    *,
    workspace_dir: str | Path | None = None,
) -> int:
    return len(
        list_skill_proposals(
            memory_dir,
            status="pending",
            workspace_dir=workspace_dir,
        )
    )


def cluster_hashes_by_status(
    memory_dir: str | Path,
    *,
    workspace_dir: str | Path | None = None,
) -> dict[str, set[str]]:
    by_status: dict[str, set[str]] = defaultdict(set)
    for proposal in list_skill_proposals(memory_dir, workspace_dir=workspace_dir):
        by_status[proposal.status].add(proposal.cluster_hash)
    return by_status


def processed_cluster_hashes(memory_dir: str | Path) -> set[str]:
    root = _processed_root(memory_dir)
    if not root.exists():
        return set()
    return {path.stem for path in root.glob("*.json")}


def mark_cluster_processed(memory_dir: str | Path, cluster_hash: str) -> None:
    root = _processed_root(memory_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{cluster_hash}.json"
    payload = {"cluster_hash": cluster_hash, "processed_at": _now()}
    _write_manifest(path, payload)


def _read_skill_markdown(skill_md: Path) -> tuple[str | None, str | None]:
    try:
        return skill_md.read_text(encoding="utf-8"), None
    except OSError as exc:
        return None, f"Cannot read SKILL.md: {exc}"


def _parse_skill_frontmatter(
    content: str,
) -> tuple[dict[str, Any] | None, str | None, re.Match[str] | None]:
    match = _SKILL_FRONTMATTER_RE.match(content)
    if match is None:
        return None, "SKILL.md must start with YAML frontmatter delimited by ---", None
    try:
        frontmatter = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        return None, f"Invalid YAML frontmatter: {exc}", None
    if not isinstance(frontmatter, dict):
        return None, "SKILL.md frontmatter must be a YAML mapping", None
    return frontmatter, None, match


def _validate_skill_proposal_dir(
    *,
    memory_dir: str | Path,
    skill_name: str,
) -> tuple[bool, list[str], str | None]:
    errors: list[str] = []
    normalized_name = sanitize_skill_name(skill_name)
    if normalized_name != skill_name:
        errors.append(
            "skill_name must be lowercase kebab-case and match the proposal directory"
        )
        return False, errors, None

    proposal_dir = _proposal_root(memory_dir) / skill_name
    if not proposal_dir.is_dir():
        errors.append(
            f"Missing proposal directory: {proposal_virtual_path(skill_name)}"
        )
        return False, errors, None

    skill_md = proposal_dir / "SKILL.md"
    if not skill_md.is_file():
        errors.append(f"Missing {proposal_virtual_path(skill_name)}/SKILL.md")
        return False, errors, None

    content, error = _read_skill_markdown(skill_md)
    if error is not None:
        errors.append(error)
        return False, errors, None
    assert content is not None

    frontmatter, error, match = _parse_skill_frontmatter(content)
    if error is not None:
        errors.append(error)
        return False, errors, None
    assert frontmatter is not None
    assert match is not None

    unexpected = set(frontmatter) - _ALLOWED_SKILL_FRONTMATTER
    if unexpected:
        errors.append(
            "Unexpected SKILL.md frontmatter key(s): "
            + ", ".join(sorted(str(key) for key in unexpected))
        )

    frontmatter_name = str(frontmatter.get("name", "")).strip()
    if frontmatter_name != skill_name:
        errors.append(
            f"SKILL.md frontmatter name must be {skill_name!r}, got {frontmatter_name!r}"
        )

    description = frontmatter.get("description")
    if not isinstance(description, str) or not description.strip():
        errors.append("SKILL.md frontmatter description must be a non-empty string")
        description_text = None
    else:
        description_text = " ".join(description.strip().split())
        if "<" in description_text or ">" in description_text:
            errors.append(
                "SKILL.md frontmatter description cannot contain angle brackets"
            )
        if len(description_text) > 1024:
            errors.append("SKILL.md frontmatter description must be at most 1024 chars")

    body = content[match.end() :]
    if re.search(r"\[TODO:|\bTODO\b", body):
        errors.append("SKILL.md body must not contain TODO placeholders")

    return not errors, errors, description_text


def _skill_frontmatter_name(skill_dir: Path) -> str | None:
    content, error = _read_skill_markdown(skill_dir / "SKILL.md")
    if error is not None or content is None:
        return None
    frontmatter, error, _match = _parse_skill_frontmatter(content)
    if error is not None or frontmatter is None:
        return None
    name = str(frontmatter.get("name", "")).strip()
    return name or None


def _find_installed_user_skill(
    skill_name: str,
    *,
    skills_dir: str | Path | None = None,
) -> Path | None:
    roots = (
        [Path(skills_dir).expanduser()]
        if skills_dir is not None
        else [Path(paths.USER_SKILLS_DIR).expanduser(), Path(paths.GLOBAL_SKILLS_DIR)]
    )
    for root in roots:
        if not root.exists():
            continue
        direct = root / skill_name
        if direct.is_dir() and (direct / "SKILL.md").is_file():
            if _skill_frontmatter_name(direct) == skill_name:
                return direct
        for entry in root.iterdir():
            if not entry.is_dir() or not (entry / "SKILL.md").is_file():
                continue
            if _skill_frontmatter_name(entry) == skill_name:
                return entry
    return None


def _copy_proposed_skill(
    proposal_dir: Path,
    destination: Path,
    *,
    base_dir: Path | None = None,
) -> None:
    if not destination.exists():
        if base_dir is not None:
            shutil.copytree(base_dir, destination)
        else:
            destination.mkdir(parents=True, exist_ok=False)
    for source_path in proposal_dir.rglob("*"):
        if not source_path.is_file():
            continue
        relative = source_path.relative_to(proposal_dir)
        if relative.as_posix() in {"manifest.json", "RATIONALE.md"}:
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)


def submit_autoskill_proposal(
    *,
    memory_dir: str | Path,
    skill_name: str,
    cluster_hash: str,
    source_observation_ids: list[str],
    rationale: str,
    operation: str = "create",
    target_skill_name: str | None = None,
    workspace_dir: str | Path | None = None,
    project_id: str | None = None,
) -> dict[str, Any]:
    """Validate and register a skill proposal folder written by the agent."""
    normalized_name = sanitize_skill_name(skill_name)
    if normalized_name != skill_name:
        return {
            "submitted": False,
            "error": "skill_name must be lowercase kebab-case",
            "skill_name": skill_name,
        }
    normalized_operation = _normalize_operation(operation)
    if normalized_operation is None:
        return {
            "submitted": False,
            "error": "operation must be 'create' or 'update'",
            "skill_name": skill_name,
        }
    if normalized_operation == "create" and target_skill_name is not None:
        return {
            "submitted": False,
            "error": "target_skill_name is only valid for update proposals",
            "skill_name": skill_name,
        }
    if normalized_operation == "update":
        target_name = sanitize_skill_name(target_skill_name or skill_name)
        if target_name != (target_skill_name or skill_name):
            return {
                "submitted": False,
                "error": "target_skill_name must be lowercase kebab-case",
                "skill_name": skill_name,
            }
        if target_name != skill_name:
            return {
                "submitted": False,
                "error": "update proposals must use the target skill name as skill_name",
                "skill_name": skill_name,
                "target_skill_name": target_name,
            }
        if _find_installed_user_skill(skill_name) is None:
            return {
                "submitted": False,
                "error": f"No installed workspace/global skill named {skill_name!r} to update",
                "skill_name": skill_name,
                "operation": normalized_operation,
            }
    cluster_text = cluster_hash.strip()
    if not cluster_text:
        return {"submitted": False, "error": "cluster_hash must not be empty"}
    source_ids = tuple(
        sorted({item.strip() for item in source_observation_ids if item.strip()})
    )
    if not source_ids:
        return {
            "submitted": False,
            "error": "source_observation_ids must not be empty",
        }

    valid, errors, description_text = _validate_skill_proposal_dir(
        memory_dir=memory_dir,
        skill_name=skill_name,
    )
    if not valid:
        return {
            "submitted": False,
            "skill_name": skill_name,
            "errors": errors,
            "path": proposal_virtual_path(skill_name),
        }

    proposal_dir = _proposal_root(memory_dir) / skill_name
    manifest_path = proposal_dir / "manifest.json"
    existing = _read_manifest(manifest_path) if manifest_path.exists() else None
    existing_status = str(existing.get("status", "")) if existing else ""
    if existing_status == "rejected":
        return {
            "submitted": False,
            "proposal_id": skill_name,
            "status": existing_status,
            "path": proposal_virtual_path(skill_name),
        }
    existing_is_approved = existing_status == "approved"
    if existing_is_approved and normalized_operation != "update":
        return {
            "submitted": False,
            "proposal_id": skill_name,
            "status": existing_status,
            "path": proposal_virtual_path(skill_name),
        }

    normalized_workspace = _normalize_workspace_dir(workspace_dir)
    existing_workspace = (
        _normalize_workspace_dir(existing.get("workspace_dir")) if existing else None
    )
    if (
        existing is not None
        and normalized_workspace is not None
        and existing_workspace is not None
        and existing_workspace != normalized_workspace
    ):
        return {
            "submitted": False,
            "proposal_id": skill_name,
            "status": existing.get("status", "pending"),
            "error": "A pending proposal with this skill name belongs to another workspace",
            "path": proposal_virtual_path(skill_name),
        }

    created_at = _now() if existing_is_approved else _created_at_for_manifest(existing)
    (proposal_dir / "RATIONALE.md").write_text(
        rationale.strip() + "\n", encoding="utf-8"
    )

    manifest = {
        "proposal_id": skill_name,
        "skill_name": skill_name,
        "description": description_text,
        "status": "pending",
        "operation": normalized_operation,
        "created_at": created_at,
        "updated_at": _now(),
        "cluster_hash": cluster_text,
        "source_observation_ids": list(source_ids),
    }
    if normalized_operation == "update":
        manifest["target_skill_name"] = skill_name
    if normalized_workspace is not None:
        manifest["workspace_dir"] = normalized_workspace
    if project_id:
        manifest["project_id"] = str(project_id)
    _write_manifest(manifest_path, manifest)
    return {
        "submitted": True,
        "created": existing is None,
        "proposal_id": skill_name,
        "status": "pending",
        "skill_name": skill_name,
        "operation": normalized_operation,
        "target_skill_name": skill_name if normalized_operation == "update" else None,
        "path": proposal_virtual_path(skill_name),
    }


def _proposal_dir_by_id(
    memory_dir: str | Path,
    proposal_id: str,
    *,
    workspace_dir: str | Path | None = None,
) -> Path | None:
    requested = proposal_id.strip()
    if not requested:
        return None
    matches = [
        proposal.path
        for proposal in list_skill_proposals(memory_dir, workspace_dir=workspace_dir)
        if proposal.proposal_id.startswith(requested)
    ]
    return matches[0] if len(matches) == 1 else None


def approve_skill_proposal(
    memory_dir: str | Path,
    proposal_id: str,
    *,
    skills_dir: str | Path | None = None,
    workspace_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Promote one pending proposal into the workspace-local skills tier."""
    proposal_dir = _proposal_dir_by_id(
        memory_dir,
        proposal_id,
        workspace_dir=workspace_dir,
    )
    if proposal_dir is None:
        return {
            "approved": False,
            "error": f"No unique proposal matching {proposal_id!r}",
        }
    manifest_path = proposal_dir / "manifest.json"
    manifest = _read_manifest(manifest_path)
    if manifest is None:
        return {"approved": False, "error": "Proposal manifest is missing or invalid"}
    if manifest.get("status") != "pending":
        return {
            "approved": False,
            "proposal_id": manifest.get("proposal_id"),
            "status": manifest.get("status"),
            "error": "Only pending proposals can be approved",
        }

    skill_name = str(manifest.get("skill_name", "")).strip()
    if sanitize_skill_name(skill_name) != skill_name:
        return {"approved": False, "error": "Proposal skill name is invalid"}
    operation = _normalize_operation(manifest.get("operation")) or "create"
    target_skill_name = str(manifest.get("target_skill_name") or skill_name).strip()
    valid, errors, _description = _validate_skill_proposal_dir(
        memory_dir=memory_dir,
        skill_name=skill_name,
    )
    if not valid:
        return {
            "approved": False,
            "proposal_id": manifest["proposal_id"],
            "error": "Proposal skill folder is invalid",
            "errors": errors,
        }
    if skills_dir is not None:
        destination_root = Path(skills_dir).expanduser()
    else:
        destination_root = Path(paths.USER_SKILLS_DIR).expanduser()
    destination = destination_root / skill_name
    base_skill_dir: Path | None = None
    if operation == "create" and destination.exists():
        return {
            "approved": False,
            "proposal_id": manifest["proposal_id"],
            "error": f"Skill already exists: {destination}",
        }
    if operation == "update":
        if target_skill_name != skill_name:
            return {
                "approved": False,
                "proposal_id": manifest["proposal_id"],
                "error": "Update proposal target must match the proposed skill name",
            }
        local_match = _find_installed_user_skill(
            skill_name,
            skills_dir=destination_root,
        )
        if local_match is not None:
            destination = local_match
        else:
            existing_global = _find_installed_user_skill(skill_name)
            if existing_global is None:
                return {
                    "approved": False,
                    "proposal_id": manifest["proposal_id"],
                    "error": f"No installed workspace/global skill named {skill_name!r} to update",
                }
            if destination.exists():
                return {
                    "approved": False,
                    "proposal_id": manifest["proposal_id"],
                    "error": (
                        f"Local skill path already exists but does not match "
                        f"{skill_name!r}: {destination}"
                    ),
                }
            base_skill_dir = existing_global

    _copy_proposed_skill(proposal_dir, destination, base_dir=base_skill_dir)

    manifest["status"] = "approved"
    manifest["updated_at"] = _now()
    manifest["approved_skill_path"] = str(destination)
    _write_manifest(manifest_path, manifest)
    mark_cluster_processed(memory_dir, str(manifest["cluster_hash"]))
    return {
        "approved": True,
        "proposal_id": manifest["proposal_id"],
        "skill_name": skill_name,
        "operation": operation,
        "path": str(destination),
    }


def reject_skill_proposal(
    memory_dir: str | Path,
    proposal_id: str,
    *,
    workspace_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Mark one pending proposal rejected."""
    proposal_dir = _proposal_dir_by_id(
        memory_dir,
        proposal_id,
        workspace_dir=workspace_dir,
    )
    if proposal_dir is None:
        return {
            "rejected": False,
            "error": f"No unique proposal matching {proposal_id!r}",
        }
    manifest_path = proposal_dir / "manifest.json"
    manifest = _read_manifest(manifest_path)
    if manifest is None:
        return {"rejected": False, "error": "Proposal manifest is missing or invalid"}
    if manifest.get("status") != "pending":
        return {
            "rejected": False,
            "proposal_id": manifest.get("proposal_id"),
            "status": manifest.get("status"),
            "error": "Only pending proposals can be rejected",
        }
    manifest["status"] = "rejected"
    manifest["updated_at"] = _now()
    _write_manifest(manifest_path, manifest)
    mark_cluster_processed(memory_dir, str(manifest["cluster_hash"]))
    return {
        "rejected": True,
        "proposal_id": manifest["proposal_id"],
        "skill_name": manifest["skill_name"],
    }
