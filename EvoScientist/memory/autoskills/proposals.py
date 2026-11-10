"""AutoSkills proposal validation and review lifecycle."""

from __future__ import annotations

import contextlib
import errno
import json
import os
import re
import shutil
import stat
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml

from ... import paths

AUTOSKILL_PROPOSALS_DIR = "autoskills/proposals"
AUTOSKILL_PROCESSED_DIR = "autoskills/processed"

_SKILL_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
_CLUSTER_HASH_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
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
    scope: str | None = None
    submitted_scope: str | None = None
    upstream_source: str | None = None
    upstream_commit: str | None = None
    workspace_shadow: str | None = None
    approved_by: str | None = None
    approved_at: str | None = None
    rejected_at: str | None = None

    @property
    def effective_scope(self) -> str:
        """The scope this proposal installs into.

        A proposal recorded before scopes existed carries no scope and
        installs into the workspace tier.
        """
        return self.scope or _SCOPE_WORKSPACE


_PROPOSAL_OPERATIONS = frozenset({"create", "update"})
_SCOPE_WORKSPACE = "workspace"
_SCOPE_GLOBAL = "global"
_SKILL_SCOPES = frozenset({_SCOPE_WORKSPACE, _SCOPE_GLOBAL})


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _version_stamp(previous: object) -> str:
    """Now to the millisecond, never ``previous``: each resubmit, decision and
    revert gets its own ``updated_at``, the version a page's request names."""
    while True:
        now = datetime.now(UTC)
        stamp = f"{now:%Y-%m-%dT%H:%M:%S}.{now.microsecond // 1000:03d}Z"
        if stamp != previous:
            return stamp


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
    raw = _read_regular(path)
    if raw is None:
        return None
    try:
        payload = json.loads(raw.decode("utf-8"))
        # A value no decision could write back or send (a lone surrogate, NaN)
        # makes the manifest corrupt.
        json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, RecursionError):
        return None
    return payload if isinstance(payload, dict) else None


# Pauses while a reader holds the target open on Windows (under a second in all).
_REPLACE_RETRY_DELAYS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)


def _replace(source: Path, target: Path) -> None:
    """``os.replace`` that outlasts a reader on Windows, where a file another
    handle holds open cannot be replaced for that moment."""
    for delay in (*_REPLACE_RETRY_DELAYS, None):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if os.name != "nt" or delay is None:
                raise
            time.sleep(delay)


def _write_text_atomic(path: Path, content: str) -> None:
    """Replace one file without following its final symlink or hardlink."""
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        _replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    _write_text_atomic(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
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
    """The workspace a proposal belongs to, in the stored form."""
    if workspace_dir is None:
        return None
    text = str(workspace_dir).strip()
    if not text:
        return None
    try:
        return paths.Workspace(text).key
    except (ValueError, OSError, RuntimeError):
        # Matches no workspace filter instead of breaking the listing.
        return text


def upgrade_proposal_workspaces(memory_dir: str | Path) -> int:
    """Rewrite the ``workspace_dir`` of proposals stored before ``run_dir``.

    Proposals made by ``--mode=run`` sessions named the run folder; they
    belong to its workspace. Other proposals, and folders that cannot be
    resolved, are left as they are. Part of the one-time upgrade of stored
    values (see ``sessions._upgrade_stored_dirs``). Returns the number
    rewritten.
    """
    root = _proposal_root(memory_dir)
    if not root.exists():
        return 0
    rewritten = 0
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = _read_manifest(manifest_path)
        stored = manifest.get("workspace_dir") if manifest else None
        if not isinstance(stored, str) or not stored.strip():
            continue
        try:
            dirs = paths.SessionDirs.from_legacy(stored.strip())
        except (ValueError, OSError, RuntimeError):
            continue
        if dirs.run_dir is not None:
            manifest["workspace_dir"] = dirs.workspace.key
            _write_manifest(manifest_path, manifest)
            rewritten += 1
    return rewritten


def _proposal_manifests(memory_dir: str | Path, workspace_key: str):
    """The manifests of *workspace_key*'s proposals, every status: (path,
    manifest) pairs."""
    root = _proposal_root(memory_dir)
    if not root.exists():
        return
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = _read_manifest(manifest_path)
        stored = manifest.get("workspace_dir") if manifest else None
        if not isinstance(stored, str) or not stored.strip():
            continue
        try:
            key = paths.Workspace(stored.strip()).key
        except paths.PATH_ERRORS:
            continue
        if key == workspace_key:
            yield manifest_path, manifest


def proposal_project_ids(memory_dir: str | Path, workspace_key: str) -> set[str]:
    """The memory project ids *workspace_key*'s proposals were made under."""
    return {
        project_id
        for _path, manifest in _proposal_manifests(memory_dir, workspace_key)
        if isinstance(project_id := manifest.get("project_id"), str) and project_id
    }


def move_proposals(
    memory_dir: str | Path,
    old_key: str,
    new_key: str,
    project_ids: dict[str, str],
) -> int:
    """Give every proposal of the project *old_key* (every status, so
    ``/autoskills list`` keeps its history) to the project *new_key*, with
    its memory project id mapped through *project_ids*. Returns the number
    rewritten; running it again rewrites none."""
    moved = 0
    for manifest_path, manifest in list(_proposal_manifests(memory_dir, old_key)):
        if manifest.get("workspace_dir") == new_key:
            continue
        manifest["workspace_dir"] = new_key
        project_id = manifest.get("project_id")
        if isinstance(project_id, str) and project_id in project_ids:
            manifest["project_id"] = project_ids[project_id]
        # The list and Undo read where an approval installed, so it moves too.
        for key in ("approved_skill_path", "workspace_shadow"):
            moved_path = _moved_path(manifest.get(key), old_key, new_key)
            if moved_path is not None:
                manifest[key] = moved_path
        _write_manifest(manifest_path, manifest)
        moved += 1
    return moved


def _moved_path(value: object, old_key: str, new_key: str) -> str | None:
    """*value* at its place under *new_key*, or None unless it is under *old_key*;
    spelled as the key is, so a parent whose case changed still matches."""
    if not isinstance(value, str) or not value:
        return None
    try:
        relative = paths.normalize_path(value).relative_to(old_key)
    except paths.PATH_ERRORS:
        return None
    return str(Path(new_key) / relative)


def _proposal_from_manifest(
    path: Path, manifest: dict[str, Any]
) -> SkillProposal | None:
    """Build a proposal from its manifest, or ``None`` if it is incomplete."""
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
            scope=(str(manifest["scope"]) if manifest.get("scope") else None),
            submitted_scope=(
                str(manifest["submitted_scope"])
                if manifest.get("submitted_scope")
                else None
            ),
            upstream_source=(
                str(manifest["upstream_source"])
                if manifest.get("upstream_source")
                else None
            ),
            upstream_commit=(
                str(manifest["upstream_commit"])
                if manifest.get("upstream_commit")
                else None
            ),
            workspace_shadow=(
                str(manifest["workspace_shadow"])
                if manifest.get("workspace_shadow")
                else None
            ),
            approved_by=(
                str(manifest["approved_by"]) if manifest.get("approved_by") else None
            ),
            approved_at=_manifest_timestamp(manifest, "approved_at"),
            rejected_at=_manifest_timestamp(manifest, "rejected_at"),
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
            and _normalize_workspace_dir(proposal.workspace_dir) != normalized_workspace
        ):
            continue
        if status is not None and proposal.status != status:
            continue
        proposals.append(proposal)
    return proposals


def find_skill_proposal(
    memory_dir: str | Path, proposal_id: str
) -> SkillProposal | None:
    """The draft whose id is exactly ``proposal_id``, in any project."""
    found = _find_with_manifest(memory_dir, proposal_id)
    return found[0] if found else None


def _find_with_manifest(
    memory_dir: str | Path, proposal_id: str
) -> tuple[SkillProposal, dict[str, Any]] | None:
    if not isinstance(proposal_id, str) or not _SKILL_NAME_RE.fullmatch(proposal_id):
        return None
    root = _proposal_root(memory_dir)
    proposal_dir = root / proposal_id
    manifest_path = proposal_dir / "manifest.json"
    resolved_root = root.resolve()
    if not _resolves_within_root(
        proposal_dir, resolved_root
    ) or not _resolves_within_root(manifest_path, resolved_root):
        return None
    manifest = _read_manifest(manifest_path)
    if manifest is None:
        return None
    proposal = _proposal_from_manifest(proposal_dir, manifest)
    if proposal is None or proposal.proposal_id != proposal_id:
        return None
    return proposal, manifest


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
    if not _CLUSTER_HASH_RE.fullmatch(cluster_hash):
        raise ValueError("Invalid cluster hash")
    root = _processed_root(memory_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{cluster_hash}.json"
    payload = {"cluster_hash": cluster_hash, "processed_at": _now()}
    _write_manifest(path, payload)


def unmark_cluster_processed(memory_dir: str | Path, cluster_hash: str) -> None:
    if not _CLUSTER_HASH_RE.fullmatch(cluster_hash):
        raise ValueError("Invalid cluster hash")
    try:
        (_processed_root(memory_dir) / f"{cluster_hash}.json").unlink()
    except FileNotFoundError:
        pass


def _read_skill_markdown(skill_md: Path) -> tuple[str | None, str | None]:
    try:
        return skill_md.read_text(encoding="utf-8"), None
    except (OSError, UnicodeDecodeError) as exc:
        return None, f"Cannot read SKILL.md: {exc}"


def _parse_skill_frontmatter(
    content: str,
) -> tuple[dict[str, Any] | None, str | None, re.Match[str] | None]:
    match = _SKILL_FRONTMATTER_RE.match(content)
    if match is None:
        return None, "SKILL.md must start with YAML frontmatter delimited by ---", None
    try:
        frontmatter = yaml.safe_load(match.group(1))
    except (yaml.YAMLError, ValueError, RecursionError) as exc:
        # An impossible date or a very deep nesting fails outside YAMLError.
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
    skill_md = proposal_dir / "SKILL.md"
    try:
        has_dir = proposal_dir.is_dir()
        has_skill_md = has_dir and _is_regular_file(skill_md)
    except OSError as exc:
        errors.append(f"Cannot read proposal files: {exc}")
        return False, errors, None
    if not has_dir:
        errors.append(
            f"Missing proposal directory: {proposal_virtual_path(skill_name)}"
        )
        return False, errors, None
    if not has_skill_md:
        errors.append(f"Missing {proposal_virtual_path(skill_name)}/SKILL.md")
        return False, errors, None

    proposal_root = _proposal_root(memory_dir).resolve()
    if not _resolves_within_root(
        proposal_dir, proposal_root
    ) or not _resolves_within_root(skill_md, proposal_root):
        errors.append("Proposal files must resolve inside /autoskill-proposals")
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

    # An approval copies a linked file's target; one outside the draft would be
    # installed without being part of what is reviewed.
    resolved_dir = proposal_dir.resolve()
    try:
        for relative in copied_files(proposal_dir):
            if not _resolves_within_root(proposal_dir / relative, resolved_dir):
                errors.append(
                    f"Proposal file links outside its folder: {relative.as_posix()}"
                )
            else:
                # A file the copy could not read would stop an in-place update
                # halfway; non-blocking, so a FIFO swapped in cannot hang the check.
                os.close(os.open(proposal_dir / relative, _OPEN_FLAGS))
    except OSError as exc:
        errors.append(f"Cannot read proposal files: {exc}")

    return not errors, errors, description_text


def _installed_skill_name(skill_dir: Path) -> str | None:
    """The name the runtime loads this knowledge skill under.

    The runtime pins a knowledge skill by the ``name`` in its SKILL.md
    frontmatter, and it only accepts a file whose frontmatter block is
    complete: the closing fence has to be followed by a newline, and the
    block has to carry a name and a description. A skill that also ships an
    EXPERT.md is *registered* as an actor under its directory name - that is
    the expert's identity, not the knowledge skill's, and the two are not
    interchangeable. Reusing the runtime's own metadata contract keeps this
    lookup on exactly the skills the runtime can load.
    """
    from ...tools.skills_manager import _skill_name_and_description

    try:
        raw = (skill_dir / "SKILL.md").read_bytes()
    except OSError:
        return None
    resolved = _skill_name_and_description(raw)
    if resolved is None:
        return None
    name, _description = resolved
    return name


def _resolves_within_root(path: Path, root: Path) -> bool:
    """Whether ``path`` resolves inside an already-resolved root."""
    try:
        return path.resolve().is_relative_to(root)
    except (OSError, RuntimeError):
        return False


_NamedSkill = tuple[Path, str | None]
# Set while one listing is built: every draft looks up the same installed skills.
_installed_scans: ContextVar[dict[tuple[Path, ...], list[_NamedSkill]] | None]
_installed_scans = ContextVar("_installed_scans", default=None)


@contextmanager
def installed_skills_read_once() -> Iterator[None]:
    """Scan each set of skill roots at most once inside the block."""
    token = _installed_scans.set({})
    try:
        yield
    finally:
        _installed_scans.reset(token)


def _visible_skills(roots: list[Path]) -> list[_NamedSkill]:
    """The skill folders the runtime sees in ``roots``, in its order, with names."""
    # The merged runtime first hides lower-tier entries with the same resolved
    # virtual path, then resolves duplicate frontmatter names in path order.
    visible: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        resolved_root = root.resolve()
        for entry in root.iterdir():
            if not entry.is_dir() or not (entry / "SKILL.md").is_file():
                continue
            # The runtime drops entries and reads that resolve outside their tier.
            if not _resolves_within_root(entry, resolved_root):
                continue
            if not _resolves_within_root(entry / "SKILL.md", resolved_root):
                continue
            resolved_entry = entry.resolve()
            virtual_path = resolved_entry.relative_to(resolved_root).as_posix()
            visible.setdefault(virtual_path, resolved_entry)
    return [
        (visible[virtual_path], _installed_skill_name(visible[virtual_path]))
        for virtual_path in sorted(visible, key=lambda name: f"/{name}/")
    ]


def _find_installed_user_skill(
    skill_name: str,
    *,
    roots: list[Path],
) -> Path | None:
    """Find an installed skill by name across the given roots."""
    scans = _installed_scans.get()
    skills = scans.get(tuple(roots)) if scans is not None else None
    if skills is None:
        skills = _visible_skills(roots)
        if scans is not None:
            scans[tuple(roots)] = skills
    match: Path | None = None
    for entry, name in skills:
        # The merged runtime keeps the later metadata entry when multiple
        # visible folders declare the same frontmatter name.
        if name == skill_name:
            match = entry
    return match


def _requested_scope(value: object) -> tuple[str | None, bool]:
    """Parse an explicitly requested scope. Returns ``(scope, valid)``.

    An absent or blank request is valid and means "not specified"; an
    unknown scope is not, so a typo is reported instead of silently
    falling back to a default.
    """
    if value is None:
        return None, True
    text = str(value).strip().lower()
    if not text:
        return None, True
    if text not in _SKILL_SCOPES:
        return None, False
    return text, True


def _scope_root(scope: str, *, skills_dir: str | Path) -> Path:
    """The install root a scope writes to."""
    if scope == _SCOPE_GLOBAL:
        return Path(paths.GLOBAL_SKILLS_DIR)
    return Path(skills_dir).expanduser()


def _installed_provenance(skill_dir: Path | None) -> dict[str, str] | None:
    """Read source metadata for a resolved installed skill, if available."""
    if skill_dir is None:
        return None
    from ...tools.skills_manager import installed_skill_provenance

    return installed_skill_provenance(skill_dir)


def _installed_scope(skill_name: str, *, skills_dir: str | Path) -> str | None:
    """The scope that holds ``skill_name``.

    Resolution follows the same merged virtual-path order as the runtime;
    when both tiers expose the same directory, the workspace copy wins.
    """
    workspace_root = _scope_root(_SCOPE_WORKSPACE, skills_dir=skills_dir).resolve()
    global_root = _scope_root(_SCOPE_GLOBAL, skills_dir=skills_dir).resolve()
    resolved = _find_installed_user_skill(
        skill_name,
        roots=[workspace_root, global_root],
    )
    if resolved is not None:
        if workspace_root == global_root:
            return _SCOPE_GLOBAL
        if resolved.is_relative_to(workspace_root):
            return _SCOPE_WORKSPACE
        if resolved.is_relative_to(global_root):
            return _SCOPE_GLOBAL
    return None


def _copy_file_atomic(source: Path, target: Path) -> None:
    """Copy one file without modifying other hardlinks to ``target``."""
    fd, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        _replace(temporary, target)
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass


# Errors Path.is_file reads as "no file" (a broken or looping link), as pathlib does.
_NO_FILE_ERRNOS = frozenset({errno.ENOENT, errno.ENOTDIR, errno.EBADF, errno.ELOOP})
_NO_FILE_WINERRORS = frozenset({21, 123, 1921})
_OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)


def _read_regular(path: Path, size: int = -1) -> bytes | None:
    """Up to *size* bytes of *path* if it is a regular file, else None: never
    blocks on a FIFO and never leaves its descriptor open."""
    try:
        fd = os.open(path, _OPEN_FLAGS)
    except OSError:
        return None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None
        with os.fdopen(fd, "rb", closefd=False) as handle:
            return handle.read(size)
    except OSError:
        return None
    finally:
        os.close(fd)


def _is_regular_file(path: Path) -> bool:
    """``Path.is_file``, but a permission or I/O error raises on every Python."""
    try:
        return stat.S_ISREG(path.stat().st_mode)
    except OSError as exc:
        if (
            exc.errno in _NO_FILE_ERRNOS
            or getattr(exc, "winerror", None) in _NO_FILE_WINERRORS
        ):
            return False
        raise
    except ValueError:
        return False


def copied_files(proposal_dir: Path) -> list[Path]:
    """The files an approval copies, relative to the draft folder, in POSIX text
    order; an unreadable folder raises, so no draft is approved short of files."""
    files: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(proposal_dir, onerror=_raise):
        base = Path(dirpath)
        for name in filenames:
            source_path = base / name
            if not _is_regular_file(source_path):
                continue
            relative = source_path.relative_to(proposal_dir)
            if relative.as_posix() in {"manifest.json", "RATIONALE.md"}:
                continue
            files.append(relative)
    return sorted(files, key=lambda relative: relative.as_posix())


def _create_destination(destination: Path, *, must_create: bool) -> bool:
    """Make the install folder; True when this call created it.

    A new skill or a fork needs a path no one else holds, so anything already
    there (a folder or a link) raises FileExistsError and is left alone. An
    in-place update writes into the skill it resolved, which must still exist.
    """
    if not must_create:
        if not destination.is_dir():
            raise FileNotFoundError(
                errno.ENOENT,
                "The skill to update is no longer installed",
                str(destination),
            )
        return False
    destination.mkdir(parents=True)
    return True


def _gone(path: Path) -> bool:
    """Whether nothing is left at ``path``; an error checking it counts as left."""
    try:
        os.lstat(path)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return False


def _plain_lstat(path: Path, *, directory: bool) -> bool:
    try:
        return _plain(path.lstat(), directory=directory)
    except OSError:
        return False


def _remove_created(folder: Path) -> bool:
    """Remove a folder this approval created; False if any of it may be left."""
    try:
        shutil.rmtree(folder, ignore_errors=True)
        if _gone(folder):
            return True
        # A copy of a read-only skill keeps its files until its folders are
        # writable; only real folders and files are touched, never a link.
        if not _plain_lstat(folder, directory=True):
            return False
        for dirpath, dirnames, filenames in os.walk(folder):
            base = Path(dirpath)
            with contextlib.suppress(OSError):
                os.chmod(base, stat.S_IRWXU)
            dirnames[:] = [
                name for name in dirnames if _plain_lstat(base / name, directory=True)
            ]
            for name in [*dirnames, *filenames]:
                entry = base / name
                if _plain_lstat(entry, directory=name in dirnames):
                    with contextlib.suppress(OSError):
                        os.chmod(entry, stat.S_IRWXU)
        shutil.rmtree(folder, ignore_errors=True)
        return _gone(folder)
    except OSError:
        return False


def _copy_proposed_skill(proposal_dir: Path, destination: Path) -> Path | None:
    """Copy proposal files, returning a blocked path before any unsafe write."""
    copies = [
        (proposal_dir / relative, destination / relative)
        for relative in copied_files(proposal_dir)
    ]
    resolved_destination = destination.resolve()
    safe_copies: list[tuple[Path, Path]] = []
    for source_path, target in copies:
        if not _resolves_within_root(target.parent, resolved_destination):
            return target
        if not _resolves_within_root(target, resolved_destination):
            return target
        safe_copies.append((source_path, target.resolve()))

    for source_path, target in safe_copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        _copy_file_atomic(source_path, target)
    return None


def submit_autoskill_proposal(
    *,
    memory_dir: str | Path,
    skill_name: str,
    cluster_hash: str,
    source_observation_ids: list[str],
    rationale: str,
    operation: str = "create",
    target_skill_name: str | None = None,
    skills_dir: str | Path,
    workspace_dir: str | Path | None = None,
    project_id: str | None = None,
) -> dict[str, Any]:
    """Validate and register a skill proposal folder written by the agent.

    ``skills_dir`` is the workspace skills tier; an update proposal must
    target a skill installed there or in the global tier.

    New skills install into the workspace tier. Updates record the tier that
    the runtime currently resolves for the target; only approval may override
    that tier explicitly.
    """
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
    # The tier this proposal installs into. An update resolves it once,
    # here, so the approval never re-guesses which copy to touch.
    target_scope = _SCOPE_WORKSPACE
    target_path: Path | None = None
    upstream_provenance: dict[str, str] | None = None
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
        target_scope = _installed_scope(skill_name, skills_dir=skills_dir)
        if target_scope is None:
            return {
                "submitted": False,
                "error": (
                    f"No installed workspace/global skill named "
                    f"{skill_name!r} to update"
                ),
                "skill_name": skill_name,
                "operation": normalized_operation,
            }
        target_path = _find_installed_user_skill(
            skill_name,
            roots=[_scope_root(target_scope, skills_dir=skills_dir)],
        )
        if target_path is None:
            return {
                "submitted": False,
                "error": (
                    f"No installed {target_scope} skill named {skill_name!r} to update"
                ),
                "skill_name": skill_name,
                "operation": normalized_operation,
                "scope": target_scope,
            }
        upstream_provenance = _installed_provenance(target_path)
    cluster_text = cluster_hash.strip()
    if not _CLUSTER_HASH_RE.fullmatch(cluster_text):
        return {
            "submitted": False,
            "error": (
                "cluster_hash must contain only letters, numbers, underscores, "
                "and hyphens"
            ),
        }
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
    _write_text_atomic(proposal_dir / "RATIONALE.md", rationale.strip() + "\n")

    manifest = {
        "proposal_id": skill_name,
        "skill_name": skill_name,
        "description": description_text,
        "status": "pending",
        "operation": normalized_operation,
        "created_at": created_at,
        "updated_at": _version_stamp(existing.get("updated_at") if existing else None),
        "cluster_hash": cluster_text,
        "source_observation_ids": list(source_ids),
    }
    if normalized_operation == "update":
        manifest["target_skill_name"] = skill_name
    # Record the resolved tier: the approval reads it instead of
    # re-deciding where the skill lives.
    manifest["scope"] = target_scope
    manifest["submitted_scope"] = target_scope
    if upstream_provenance:
        manifest["upstream_source"] = upstream_provenance["source"]
        if upstream_provenance.get("commit"):
            manifest["upstream_commit"] = upstream_provenance["commit"]
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
        "scope": target_scope,
        "submitted_scope": target_scope,
        "upstream_source": (
            upstream_provenance.get("source") if upstream_provenance else None
        ),
        "upstream_commit": (
            upstream_provenance.get("commit") if upstream_provenance else None
        ),
        "path": proposal_virtual_path(skill_name),
    }


def _proposal_dir_by_id(
    memory_dir: str | Path,
    proposal_id: str,
    *,
    workspace_dir: str | Path | None = None,
    exact: bool = False,
) -> Path | None:
    requested = proposal_id.strip()
    if not requested:
        return None
    found = find_skill_proposal(memory_dir, requested)
    wanted = _normalize_workspace_dir(workspace_dir)
    if found is not None and (
        wanted is None or _normalize_workspace_dir(found.workspace_dir) == wanted
    ):
        return found.path
    if exact:
        return None
    proposals = list_skill_proposals(memory_dir, workspace_dir=workspace_dir)
    # A folder find_skill_proposal refuses still wins by its exact id, so a
    # draft stays reachable when another one's name extends it.
    for proposal in proposals:
        if proposal.proposal_id == requested and proposal.path.name == requested:
            return proposal.path
    matches = [
        proposal.path
        for proposal in proposals
        if proposal.proposal_id.startswith(requested)
    ]
    return matches[0] if len(matches) == 1 else None


def _notify_approved_skill() -> None:
    from ...tools.skills_manager import _notify_skills_changed

    _notify_skills_changed()


def _failure_text(failure: dict[str, Any]) -> str:
    """A refusal's message followed by its ``errors`` list, as one line."""
    text = str(failure.get("error") or "")
    if failure.get("errors"):
        text += ": " + "; ".join(str(item) for item in failure["errors"])
    return text


@dataclass(frozen=True)
class InstallTarget:
    """Where approving a draft would write, resolved without writing.

    ``base_path`` is the installed folder the result starts from: the skill an
    in-place update changes, or the global skill a project fork copies first.
    """

    scope: str | None
    path: Path | None
    base_path: Path | None
    workspace_shadow: Path | None
    error: str | None


def _resolve_install(
    memory_dir: str | Path,
    proposal_dir: Path,
    manifest: dict[str, Any],
    *,
    skills_dir: str | Path,
    scope: str | None,
) -> tuple[InstallTarget, dict[str, Any] | None]:
    """Run every check an approval makes before writing.

    Returns the target and, when approving would be refused, the exact result
    ``approve_skill_proposal`` returns for it.
    """
    target_scope: str | None = None

    def refuse(failure: dict[str, Any]) -> tuple[InstallTarget, dict[str, Any]]:
        error = _failure_text(failure)
        refused = InstallTarget(
            scope=target_scope,
            path=None,
            base_path=None,
            workspace_shadow=None,
            error=error,
        )
        return refused, failure

    if manifest.get("status") != "pending":
        return refuse(
            {
                "approved": False,
                "proposal_id": manifest.get("proposal_id"),
                "status": manifest.get("status"),
                "error": "Only pending proposals can be approved",
            }
        )
    if not _CLUSTER_HASH_RE.fullmatch(str(manifest.get("cluster_hash", ""))):
        return refuse(
            {
                "approved": False,
                "proposal_id": manifest.get("proposal_id"),
                "error": "Proposal records an invalid cluster hash",
            }
        )

    skill_name = str(manifest.get("skill_name", "")).strip()
    if sanitize_skill_name(skill_name) != skill_name:
        return refuse({"approved": False, "error": "Proposal skill name is invalid"})
    # Validation reads proposals/<skill_name> while the copy reads this folder.
    if proposal_dir.name != skill_name or manifest.get("proposal_id") != skill_name:
        return refuse(
            {
                "approved": False,
                "proposal_id": manifest.get("proposal_id"),
                "error": "Proposal folder, id and skill name must match",
            }
        )
    operation = _normalize_operation(manifest.get("operation")) or "create"
    target_skill_name = str(manifest.get("target_skill_name") or skill_name).strip()
    valid, errors, _description = _validate_skill_proposal_dir(
        memory_dir=memory_dir,
        skill_name=skill_name,
    )
    if not valid:
        return refuse(
            {
                "approved": False,
                "proposal_id": manifest["proposal_id"],
                "error": "Proposal skill folder is invalid",
                "errors": errors,
            }
        )
    # The tier the proposal installs into: an explicit override wins, then
    # the tier recorded at submit time, then the workspace tier for a
    # proposal written before scopes existed.
    requested_scope, scope_valid = _requested_scope(scope)
    if not scope_valid:
        return refuse(
            {
                "approved": False,
                "proposal_id": manifest.get("proposal_id"),
                "error": "scope must be 'workspace' or 'global'",
            }
        )
    raw_scope = manifest.get("scope")
    recorded_scope, recorded_scope_valid = _requested_scope(raw_scope)
    if raw_scope is not None and not recorded_scope_valid:
        # A recorded scope that is neither absent nor a known tier is a
        # broken manifest, not a legacy one: installing into the workspace
        # would silently ignore what the manifest says.
        return refuse(
            {
                "approved": False,
                "proposal_id": manifest.get("proposal_id"),
                "error": f"Proposal records an unknown scope {raw_scope!r}",
            }
        )
    # A proposal written before scopes existed records none. It keeps the
    # old workspace behaviour and never writes to the global tier unless
    # the approval names that tier explicitly.
    legacy = recorded_scope is None
    target_scope = requested_scope or recorded_scope or _SCOPE_WORKSPACE
    destination_root = _scope_root(target_scope, skills_dir=skills_dir)
    destination = destination_root / skill_name
    base_skill_dir: Path | None = None
    workspace_shadow: Path | None = None
    if operation == "create":
        other_root = _scope_root(
            _SCOPE_GLOBAL if target_scope == _SCOPE_WORKSPACE else _SCOPE_WORKSPACE,
            skills_dir=skills_dir,
        )
        existing_skill = _find_installed_user_skill(
            skill_name, roots=[destination_root]
        ) or _find_installed_user_skill(skill_name, roots=[other_root])
        if (
            destination.exists()
            or destination.is_symlink()
            or existing_skill is not None
        ):
            collision = existing_skill or destination
            return refuse(
                {
                    "approved": False,
                    "proposal_id": manifest["proposal_id"],
                    "error": f"Skill already exists: {collision}",
                }
            )
    if operation == "update":
        if target_skill_name != skill_name:
            return refuse(
                {
                    "approved": False,
                    "proposal_id": manifest["proposal_id"],
                    "error": "Update proposal target must match the proposed skill name",
                }
            )
        if (
            target_scope == _SCOPE_WORKSPACE
            and recorded_scope == _SCOPE_GLOBAL
            and requested_scope == _SCOPE_WORKSPACE
        ):
            # A workspace approval may fork the global target. Refuse a
            # destination or matching workspace alias that would remain the
            # runtime-visible copy instead of the fork.
            existing_local = _find_installed_user_skill(
                skill_name,
                roots=[destination_root],
            )
            if (
                destination.exists()
                or destination.is_symlink()
                or existing_local is not None
            ):
                return refuse(
                    {
                        "approved": False,
                        "proposal_id": manifest["proposal_id"],
                        "error": (
                            f"Workspace skill path already exists; refusing to "
                            f"retarget global skill {skill_name!r}"
                        ),
                        "scope": target_scope,
                    }
                )
            existing_global = _find_installed_user_skill(
                skill_name,
                roots=[Path(paths.GLOBAL_SKILLS_DIR)],
            )
            if existing_global is None:
                return refuse(
                    {
                        "approved": False,
                        "proposal_id": manifest["proposal_id"],
                        "error": (
                            f"No installed global skill named {skill_name!r} "
                            "to use as the workspace fork source"
                        ),
                        "scope": target_scope,
                    }
                )
            base_skill_dir = existing_global
        elif legacy and target_scope == _SCOPE_WORKSPACE:
            # Legacy proposals retain their historical workspace shadow
            # behaviour; only an explicit global approval writes globally.
            local_match = _find_installed_user_skill(
                skill_name,
                roots=[destination_root],
            )
            if local_match is not None:
                destination = local_match
            else:
                existing_global = _find_installed_user_skill(
                    skill_name,
                    roots=[Path(paths.GLOBAL_SKILLS_DIR)],
                )
                if existing_global is None:
                    return refuse(
                        {
                            "approved": False,
                            "proposal_id": manifest["proposal_id"],
                            "error": (
                                f"No installed workspace/global skill named "
                                f"{skill_name!r} to update"
                            ),
                        }
                    )
                if destination.exists() or destination.is_symlink():
                    return refuse(
                        {
                            "approved": False,
                            "proposal_id": manifest["proposal_id"],
                            "error": (
                                f"Local skill path already exists but does not match "
                                f"{skill_name!r}: {destination}"
                            ),
                        }
                    )
                base_skill_dir = existing_global
        else:
            # Re-check the target in the tier this proposal records. A skill
            # that moved, or was removed, is an error rather than a reason
            # to install into the other tier.
            local_match = _find_installed_user_skill(
                skill_name,
                roots=[destination_root],
            )
            if local_match is not None:
                destination = local_match
            else:
                return refuse(
                    {
                        "approved": False,
                        "proposal_id": manifest["proposal_id"],
                        "error": (
                            f"No installed {target_scope} skill named "
                            f"{skill_name!r} to update"
                        ),
                        "scope": target_scope,
                    }
                )
            if target_scope == _SCOPE_GLOBAL:
                workspace_root = _scope_root(
                    _SCOPE_WORKSPACE,
                    skills_dir=skills_dir,
                )
                workspace_shadow = _find_installed_user_skill(
                    skill_name,
                    roots=[workspace_root],
                )
                if workspace_shadow is None:
                    candidate = workspace_root / destination.name
                    resolved_workspace = workspace_root.resolve()
                    if (
                        candidate.is_dir()
                        and candidate != destination
                        and _resolves_within_root(candidate, resolved_workspace)
                        and _resolves_within_root(
                            candidate / "SKILL.md", resolved_workspace
                        )
                    ):
                        workspace_shadow = candidate

    base_path = base_skill_dir
    if base_path is None and operation == "update":
        base_path = destination
    target = InstallTarget(
        scope=target_scope,
        path=destination,
        base_path=base_path,
        workspace_shadow=workspace_shadow,
        error=None,
    )
    return target, None


def proposal_install_target(
    memory_dir: str | Path,
    proposal_id: str,
    *,
    skills_dir: str | Path,
    scope: str | None = None,
) -> InstallTarget:
    """Where approving the draft named exactly ``proposal_id`` would write."""
    found = _find_with_manifest(memory_dir, proposal_id)
    if found is None:
        return InstallTarget(
            scope=None,
            path=None,
            base_path=None,
            workspace_shadow=None,
            error=f"No proposal named {proposal_id!r}",
        )
    proposal, manifest = found
    try:
        target, _failure = _resolve_install(
            memory_dir, proposal.path, manifest, skills_dir=skills_dir, scope=scope
        )
    except (OSError, ValueError, RecursionError) as exc:
        return InstallTarget(
            scope=None,
            path=None,
            base_path=None,
            workspace_shadow=None,
            error=f"Could not read the skill folders: {exc}",
        )
    return target


def _record_approval(
    manifest: dict[str, Any],
    *,
    approved_by: str,
    destination: Path,
    scope: str,
    workspace_shadow: Path | None,
    provenance: dict[str, str] | None,
) -> None:
    if provenance:
        manifest["upstream_source"] = provenance["source"]
        if provenance.get("commit"):
            manifest["upstream_commit"] = provenance["commit"]
        else:
            manifest.pop("upstream_commit", None)
    else:
        manifest.pop("upstream_source", None)
        manifest.pop("upstream_commit", None)
    manifest["status"] = "approved"
    manifest["updated_at"] = _version_stamp(manifest.get("updated_at"))
    manifest["approved_by"] = approved_by
    manifest["approved_at"] = manifest["updated_at"]
    manifest["approved_skill_path"] = str(destination)
    # The tier the skill was actually installed into, which an approval may
    # override.
    manifest["scope"] = scope
    if workspace_shadow is not None:
        manifest["workspace_shadow"] = str(workspace_shadow)
    else:
        manifest.pop("workspace_shadow", None)


def approve_skill_proposal(
    memory_dir: str | Path,
    proposal_id: str,
    *,
    skills_dir: str | Path,
    workspace_dir: str | Path | None = None,
    scope: str | None = None,
    approved_by: Literal["user", "auto"] = "user",
    exact: bool = False,
    expected_updated_at: str | None = None,
) -> dict[str, Any]:
    """Promote one pending proposal into the tier its manifest records.

    The tier is resolved once, when the proposal is submitted, and read back
    here; the approval re-checks that the skill is still installed there and
    never falls back to another tier. ``scope`` overrides the recorded tier
    explicitly, which is how a proposal written before scopes existed is
    confirmed to install into the global tier. An approval that fails leaves
    no folder it created behind, and the manifest is written last. With
    ``expected_updated_at``, a pending draft resubmitted since is refused.
    """
    proposal_dir = _proposal_dir_by_id(
        memory_dir,
        proposal_id,
        workspace_dir=workspace_dir,
        exact=exact,
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
    if (
        expected_updated_at is not None
        and manifest.get("status") == "pending"
        and manifest.get("updated_at") != expected_updated_at
    ):
        return {
            "approved": False,
            "proposal_id": manifest.get("proposal_id"),
            "updated_at": manifest.get("updated_at"),
            "error": "Proposal changed since it was read",
        }
    try:
        target, failure = _resolve_install(
            memory_dir, proposal_dir, manifest, skills_dir=skills_dir, scope=scope
        )
    except (OSError, ValueError, RecursionError) as exc:
        return {
            "approved": False,
            "proposal_id": manifest.get("proposal_id"),
            "error": f"Could not read the skill folders: {exc}",
        }
    if failure is not None:
        return failure
    assert target.path is not None
    assert target.scope is not None
    skill_name = str(manifest["skill_name"]).strip()
    operation = _normalize_operation(manifest.get("operation")) or "create"
    destination = target.path
    target_scope = target.scope
    workspace_shadow = target.workspace_shadow
    # A fork copies the global skill first; an in-place update starts from itself.
    base_skill_dir = target.base_path if target.base_path != destination else None

    def install_failed(
        error: str, *, changed: bool, leftover: Path | None, mark_left: bool = False
    ) -> dict[str, Any]:
        if changed:
            error += " (the installed skill may already be changed)"
        if leftover is not None:
            error += f"; the partly installed folder {leftover} could not be removed"
        if mark_left:
            error += "; the processed mark could not be removed"
        return {
            "approved": False,
            "proposal_id": manifest["proposal_id"],
            "error": error,
            "scope": target_scope,
        }

    provenance_path = (
        destination if operation == "update" and base_skill_dir is None else None
    )
    cluster_hash = str(manifest["cluster_hash"])
    mark_left = False
    try:
        upstream_provenance = _installed_provenance(provenance_path)
        created = _create_destination(
            destination, must_create=target.base_path != destination
        )
    except (OSError, ValueError, RecursionError) as exc:
        return install_failed(
            f"Could not install the skill: {exc}", changed=False, leftover=None
        )
    try:
        if created and base_skill_dir is not None:
            shutil.copytree(base_skill_dir, destination, dirs_exist_ok=True)
        blocked_path = _copy_proposed_skill(proposal_dir, destination)
        if blocked_path is None:
            _record_approval(
                manifest,
                approved_by=approved_by,
                destination=destination,
                scope=target_scope,
                workspace_shadow=workspace_shadow,
                provenance=upstream_provenance,
            )
            had_mark = cluster_hash in processed_cluster_hashes(memory_dir)
            mark_cluster_processed(memory_dir, cluster_hash)
            try:
                # The manifest is the commit point, so it is written last.
                _write_manifest(manifest_path, manifest)
            except (OSError, ValueError):
                if not had_mark:
                    try:
                        unmark_cluster_processed(memory_dir, cluster_hash)
                    except OSError:
                        mark_left = True
                raise
    except (OSError, ValueError) as exc:
        leftover = destination if created and not _remove_created(destination) else None
        return install_failed(
            f"Could not install the skill: {exc}",
            changed=not created,
            leftover=leftover,
            mark_left=mark_left,
        )
    if blocked_path is not None:
        leftover = destination if created and not _remove_created(destination) else None
        return install_failed(
            f"Skill install path escapes its destination: {blocked_path}",
            changed=False,
            leftover=leftover,
        )
    _notify_approved_skill()
    return {
        "approved": True,
        "proposal_id": manifest["proposal_id"],
        "skill_name": skill_name,
        "operation": operation,
        "path": str(destination),
        "scope": target_scope,
        "workspace_shadow": str(workspace_shadow) if workspace_shadow else None,
        "upstream_source": (
            upstream_provenance.get("source") if upstream_provenance else None
        ),
        "upstream_commit": (
            upstream_provenance.get("commit") if upstream_provenance else None
        ),
        "updated_at": manifest["updated_at"],
    }


def _raise(error: OSError) -> None:
    raise error


# Windows reparse tags that name another path: junctions and symlinks, not cloud files.
_NAME_SURROGATE_TAG = 0x20000000


def _plain(entry: os.stat_result, *, directory: bool) -> bool:
    """A real folder or file: no symlink or junction."""
    if getattr(entry, "st_reparse_tag", 0) & _NAME_SURROGATE_TAG:
        return False
    return stat.S_ISDIR(entry.st_mode) if directory else stat.S_ISREG(entry.st_mode)


def _regular_files(root: Path) -> dict[str, Path] | None:
    """Every file under ``root`` by relative path; None unless all are plain."""
    files: dict[str, Path] = {}
    for dirpath, dirnames, filenames in os.walk(root, onerror=_raise):
        base = Path(dirpath)
        for name in dirnames:
            if not _plain((base / name).lstat(), directory=True):
                return None
        for name in filenames:
            path = base / name
            if not _plain(path.lstat(), directory=False):
                return None
            files[path.relative_to(root).as_posix()] = path
    return files


_COMPARE_CHUNK = 1 << 20


def _same_bytes(installed: Path, copied: Path) -> bool:
    if installed.lstat().st_size != copied.stat().st_size:
        return False
    # In chunks: a skill may carry a large data file, and revert runs in the server.
    with installed.open("rb") as left, copied.open("rb") as right:
        while True:
            chunk = left.read(_COMPARE_CHUNK)
            if chunk != right.read(_COMPARE_CHUNK):
                return False
            if not chunk:
                return True


# Files a skill gains by being used or browsed (Finder's ``._<name>`` too), which
# Undo removes with it; compared without case, as Windows and macOS names are.
_CACHE_NAMES = frozenset({".ds_store", "thumbs.db", "desktop.ini"})


def _is_cache(name: str) -> bool:
    """Whether a relative path is Python bytecode or a file browser's cache."""
    parts = name.split("/")
    if parts[-1].casefold() in _CACHE_NAMES or parts[-1].startswith("._"):
        return True
    return len(parts) > 1 and parts[-2] == "__pycache__" and name.endswith(".pyc")


def _unlink_checked(path: Path) -> None:
    """Delete a file revert has checked; on Windows a read-only copy (the
    approval kept the draft's attribute) refuses until the attribute is cleared."""
    try:
        path.unlink()
    except PermissionError:
        if os.name != "nt":
            raise
        os.chmod(path, stat.S_IWRITE)
        path.unlink()


def _remove_unchanged_copy(installed: Path, proposal_dir: Path) -> bool:
    """Delete a new skill's files if every one is still one the approval copied.

    Files already gone count as removed, so a revert cut short can run again;
    caches go too, but a file added or edited since keeps the whole folder.
    """
    try:
        top = installed.lstat()
    except FileNotFoundError:
        return True
    if not _plain(top, directory=True):
        return False
    copied = {
        relative.as_posix(): proposal_dir / relative
        for relative in copied_files(proposal_dir)
    }
    present = _regular_files(installed)
    if present is None or not all(
        name in copied or _is_cache(name) for name in present
    ):
        return False

    def unchanged(name: str, path: Path) -> bool:
        # A cache may differ from a copied one: Finder rewrites .DS_Store freely.
        return name not in copied or _is_cache(name) or _same_bytes(path, copied[name])

    if not all(unchanged(name, path) for name, path in present.items()):
        return False
    for path in present.values():
        _unlink_checked(path)
    for dirpath, _dirnames, _filenames in os.walk(
        installed, topdown=False, onerror=_raise
    ):
        Path(dirpath).rmdir()
    return True


def _installed_in_its_tier(manifest: dict[str, Any], installed: Path) -> bool:
    """Whether ``installed`` is the ``<tier>/<skill>`` folder the manifest names."""
    # An approval made before scopes existed installed into the workspace tier.
    scope = manifest.get("scope") or _SCOPE_WORKSPACE
    if scope == _SCOPE_GLOBAL:
        root = Path(paths.GLOBAL_SKILLS_DIR)
    elif scope == _SCOPE_WORKSPACE and manifest.get("workspace_dir"):
        try:
            root = paths.Workspace(str(manifest["workspace_dir"])).skills_dir
        except paths.PATH_ERRORS:
            return False
    else:
        return False
    return (
        installed.is_absolute()
        and installed.name == manifest.get("skill_name")
        and installed.parent.resolve() == root.resolve()
    )


def revert_skill_proposal(
    memory_dir: str | Path,
    proposal_id: str,
    *,
    expected_updated_at: str | None = None,
) -> dict[str, Any]:
    """Put a rejected draft, or a new skill's approval, back to pending; an
    approval auto mode made becomes a rejection, so the agent cannot reinstall it.
    With ``expected_updated_at``, a draft decided or reverted since is refused.

    An approved update, in place or as a project fork, is refused: nothing
    records the installed skill it started from. Each step can be retried: the
    manifest keeps its decision until the skill is gone and the mark is settled.
    """
    proposal = find_skill_proposal(memory_dir, proposal_id)
    manifest_path = proposal.path / "manifest.json" if proposal else None
    manifest = _read_manifest(manifest_path) if manifest_path else None
    if proposal is None or manifest_path is None or manifest is None:
        return {"reverted": False, "proposal_id": proposal_id, "error": "not_found"}
    if (
        expected_updated_at is not None
        and manifest.get("updated_at") != expected_updated_at
    ):
        return {
            "reverted": False,
            "proposal_id": proposal.proposal_id,
            "updated_at": manifest.get("updated_at"),
            "error": "draft_changed",
        }

    def refused(code: str) -> dict[str, Any]:
        return {"reverted": False, "proposal_id": proposal.proposal_id, "error": code}

    status = manifest.get("status")
    if status not in ("approved", "rejected"):
        return refused("not_decided")
    if status == "approved" and proposal.operation != "create":
        return refused("revert_unavailable")
    installed = manifest.get("approved_skill_path") if status == "approved" else None
    if installed is not None and not isinstance(installed, str):
        return refused("skill_changed")
    # Submit never overwrites a rejected draft, so the agent cannot bring it back.
    rejects = status == "approved" and manifest.get("approved_by") == "auto"
    try:
        if installed:
            if not _installed_in_its_tier(manifest, Path(installed)):
                return refused("skill_changed")
            if not _remove_unchanged_copy(Path(installed), proposal.path):
                return refused("skill_changed")
        if status == "approved":
            _notify_approved_skill()
        cluster_hash = str(manifest.get("cluster_hash", ""))
        shared = any(
            other.cluster_hash == cluster_hash
            and other.status in ("approved", "rejected")
            and other.path != proposal.path
            for other in list_skill_proposals(memory_dir)
        )
        if _CLUSTER_HASH_RE.fullmatch(cluster_hash) and not shared and not rejects:
            unmark_cluster_processed(memory_dir, cluster_hash)

        for key in ("approved_by", "approved_at", "approved_skill_path", "rejected_at"):
            manifest.pop(key, None)
        if status == "approved":
            manifest.pop("workspace_shadow", None)
            if manifest.get("submitted_scope"):
                manifest["scope"] = manifest["submitted_scope"]
            else:
                manifest.pop("scope", None)
        manifest["status"] = "rejected" if rejects else "pending"
        manifest["updated_at"] = _version_stamp(manifest.get("updated_at"))
        if rejects:
            manifest["rejected_at"] = manifest["updated_at"]
        _write_manifest(manifest_path, manifest)
    except OSError as exc:
        return {**refused("revert_failed"), "detail": str(exc)}
    return {
        "reverted": True,
        "proposal_id": proposal.proposal_id,
        "status": manifest["status"],
        "updated_at": manifest["updated_at"],
    }


def reject_skill_proposal(
    memory_dir: str | Path,
    proposal_id: str,
    *,
    workspace_dir: str | Path | None = None,
    exact: bool = False,
    expected_updated_at: str | None = None,
) -> dict[str, Any]:
    """Mark one pending proposal rejected; with ``expected_updated_at``, a
    draft resubmitted since is refused, as an approval refuses it."""
    proposal_dir = _proposal_dir_by_id(
        memory_dir,
        proposal_id,
        workspace_dir=workspace_dir,
        exact=exact,
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
    if (
        expected_updated_at is not None
        and manifest.get("updated_at") != expected_updated_at
    ):
        return {
            "rejected": False,
            "proposal_id": manifest.get("proposal_id"),
            "updated_at": manifest.get("updated_at"),
            "error": "Proposal changed since it was read",
        }
    if not _CLUSTER_HASH_RE.fullmatch(str(manifest.get("cluster_hash", ""))):
        return {
            "rejected": False,
            "proposal_id": manifest.get("proposal_id"),
            "error": "Proposal records an invalid cluster hash",
        }
    cluster_hash = str(manifest["cluster_hash"])
    manifest["status"] = "rejected"
    manifest["updated_at"] = _version_stamp(manifest.get("updated_at"))
    manifest["rejected_at"] = manifest["updated_at"]
    mark_left = False
    try:
        had_mark = cluster_hash in processed_cluster_hashes(memory_dir)
        mark_cluster_processed(memory_dir, cluster_hash)
        try:
            # The manifest is the commit point, so it is written last.
            _write_manifest(manifest_path, manifest)
        except (OSError, ValueError):
            if not had_mark:
                try:
                    unmark_cluster_processed(memory_dir, cluster_hash)
                except OSError:
                    mark_left = True
            raise
    except (OSError, ValueError) as exc:
        error = f"Could not reject the proposal: {exc}"
        if mark_left:
            error += "; the processed mark could not be removed"
        return {
            "rejected": False,
            "proposal_id": manifest["proposal_id"],
            "error": error,
        }
    return {
        "rejected": True,
        "proposal_id": manifest["proposal_id"],
        "skill_name": manifest["skill_name"],
        "updated_at": manifest["updated_at"],
    }
