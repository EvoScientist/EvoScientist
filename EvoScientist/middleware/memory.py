"""Memory middleware for EvoScientist.

The middleware owns the markdown files under ``/memories/profile/``: it creates
them when missing, migrates the old ``/memories/MEMORY.md`` file when present,
and injects either profile contents or profile file pointers into model calls.
Agents still read and edit the files through their normal ``/memories/...``
tools; this middleware only handles setup and prompt context.
"""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess
from collections.abc import Awaitable, Callable
from pathlib import Path

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)

from .. import paths as _paths

logger = logging.getLogger(__name__)

DEFAULT_MAX_INLINE_PROFILE_CHARS = 24_000
_LEGACY_MEMORY_FILENAME = "MEMORY.md"
_LEGACY_IMPORT_HEADING = "Imported from legacy MEMORY.md"


PROFILE_INJECTION_TEMPLATE = """<profile_memory>
{profile_content}
</profile_memory>

<profile_memory_instructions>
These profile notes live under `/memories/profile/`.
Every agent can read and update them with normal file tools.

Use these files for:
- `/memories/profile/SOUL.md`: how this copy should usually behave; voice and boundaries.
- `/memories/profile/USER_PROFILE.md`: facts and preferences about the user.
- `/memories/profile/RESEARCH_TASTE.md`: research interests, standards, methods that fit, and things to avoid.
- `/memories/profile/projects/{project_id}/PROJECT_PROFILE.md`: conventions, commands, and pitfalls for this workspace.

Read the relevant file before editing it. Add small bullets under existing
headings, skip duplicates, and leave out temporary task state.
</profile_memory_instructions>"""

PROFILE_TEMPLATES: dict[str, str] = {
    "/profile/SOUL.md": """# EvoScientist soul

Default behavior for this copy of EvoScientist.

## Operating principles

## Voice

## Lines not to cross
""",
    "/profile/USER_PROFILE.md": """# User profile

Things worth remembering about the person using EvoScientist.

## Stable facts

## Preferences

## Collaboration style

## Constraints
""",
    "/profile/RESEARCH_TASTE.md": """# Research taste

Research taste to keep in mind: interests, standards, methods that tend to fit,
and things to avoid.

## Interests

## Standards

## Methods that fit

## Things to avoid
""",
    "/profile/projects/{project_id}/PROJECT_PROFILE.md": """# Project profile

Notes about this workspace: conventions, commands, tests, and traps.

## Workspace conventions

## Commands that work

## Evaluation and testing

## Known traps
""",
}


def _short_hash(text: str, *, n: int = 16) -> str:
    """Return a deterministic hash fragment for generated profile paths."""
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def _run_git(args: list[str], cwd: Path) -> str | None:
    """Run a bounded git query, returning trimmed stdout when it succeeds.

    Failures are treated as missing metadata so profile setup can fall back to
    path-based ids.
    """
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _resolve_project_id(workspace: str | Path | None = None) -> str:
    """Return the stable id used for this workspace's project profile.

    Prefer the git remote when available, then the git root, and finally the
    workspace path.
    """
    root = Path(workspace or _paths.WORKSPACE_ROOT).expanduser().resolve()
    git_root = _run_git(["rev-parse", "--show-toplevel"], root)
    if git_root:
        git_root_path = Path(git_root).expanduser().resolve()
        remote = _run_git(["remote", "get-url", "origin"], git_root_path)
        source = f"git-remote:{remote}" if remote else f"git-root:{git_root_path}"
        return f"P-{_short_hash(source)}"
    return f"P-{_short_hash(f'path:{root}')}"


def _profile_specs(project_id: str) -> list[tuple[str, str]]:
    """Return the profile files owned by this middleware and their templates."""
    return [
        (path.format(project_id=project_id), template)
        for path, template in PROFILE_TEMPLATES.items()
    ]


def _agent_path(memory_path: str) -> str:
    """Translate a memory-relative path to the virtual path agents see."""
    return f"/memories{memory_path}"


def _legacy_sections(content: str) -> tuple[str, list[tuple[str, str]]]:
    """Split the old ``MEMORY.md`` format into preface and top-level sections."""
    pattern = re.compile(
        r"^## (?P<heading>.+?)\n(?P<body>.*?)(?=^## |\Z)",
        flags=re.MULTILINE | re.DOTALL,
    )
    sections = [
        (match.group("heading").strip(), match.group("body").strip())
        for match in pattern.finditer(content)
    ]
    first = pattern.search(content)
    preface = content[: first.start()].strip() if first else content.strip()
    return preface, sections


def _is_legacy_placeholder_line(line: str) -> bool:
    """Return whether a legacy line is only default-template filler."""
    stripped = line.strip()
    if stripped in {"", "- (none yet)", "- (none)", "(No experiments yet)", "(none)"}:
        return True
    return bool(re.fullmatch(r"- \*\*[^*]+\*\*:\s*\(unknown\)", stripped))


def _clean_legacy_body(body: str) -> str:
    """Drop old template placeholders while keeping real legacy notes."""
    lines = [
        line.rstrip()
        for line in body.strip().splitlines()
        if not _is_legacy_placeholder_line(line)
    ]
    return "\n".join(lines).strip()


def _clean_legacy_preface(preface: str) -> str:
    """Remove the old root heading from pre-section legacy text."""
    lines = [
        line.rstrip()
        for line in preface.strip().splitlines()
        if line.strip() != "# EvoScientist Memory"
    ]
    return "\n".join(lines).strip()


def _append_imported_section(content: str, body: str) -> str:
    """Append migrated legacy text under a clear, inspectable heading."""
    return content.rstrip() + f"\n\n## {_LEGACY_IMPORT_HEADING}\n\n{body.strip()}\n"


class EvoMemoryMiddleware(AgentMiddleware):
    """Middleware that maintains the profile memory files used by EvoScientist.

    The middleware bootstraps missing files, migrates legacy memory, and adds
    profile context to model requests.
    """

    def __init__(
        self,
        *,
        memory_dir: str | Path,
        workspace_dir: str | Path | None = None,
        max_inline_profile_chars: int = DEFAULT_MAX_INLINE_PROFILE_CHARS,
    ) -> None:
        self._memory_dir = Path(memory_dir).expanduser()
        workspace = Path(workspace_dir or _paths.WORKSPACE_ROOT).expanduser()
        self._project_id = _resolve_project_id(workspace)
        self._profile_specs = _profile_specs(self._project_id)
        pointer_lines = ["Profile files are available at:"]
        pointer_lines.extend(
            f"- {_agent_path(path)}" for path, _ in self._profile_specs
        )
        self._profile_pointer_context = "\n".join(pointer_lines)
        self._max_inline_profile_chars = max_inline_profile_chars

    def _file_path(self, memory_path: str) -> Path:
        """Resolve a memory-relative path against the memory directory."""
        return self._memory_dir / memory_path.lstrip("/")

    def _read_text(self, path: Path) -> str | None:
        """Read UTF-8 text, returning None only when the file is absent."""
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except (OSError, UnicodeDecodeError) as e:
            logger.warning("Failed to read profile memory %s: %s", path, e)
            raise

    def _write_text(self, path: Path, content: str) -> bool:
        """Write UTF-8 text, creating parent directories as needed."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except OSError as e:
            logger.warning("Failed to write profile memory %s: %s", path, e)
            return False
        return True

    def _delete_legacy_memory(self, legacy_path: Path) -> bool:
        """Remove the old memory file after it has no content left to preserve."""
        try:
            legacy_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as e:
            logger.warning("Failed to delete legacy memory %s: %s", legacy_path, e)
            return False
        return True

    def _ensure_profile_files(self) -> list[tuple[str, str]]:
        """Create the expected profile files if needed and return their contents."""
        records = []
        for memory_path, template in self._profile_specs:
            path = self._file_path(memory_path)
            content = self._read_text(path)
            if content is None:
                if not self._write_text(path, template):
                    raise OSError(f"Failed to bootstrap profile file: {path}")
                content = template
            records.append((memory_path, content))
        return records

    def _migrate_legacy_memory(self) -> bool:
        """Import recognized sections from legacy ``MEMORY.md`` into profiles.

        The legacy file is removed only after real content is copied or the file
        is found to contain only old template placeholders.
        """
        legacy_path = self._memory_dir / _LEGACY_MEMORY_FILENAME
        legacy = self._read_text(legacy_path)
        if legacy is None:
            return True
        if not legacy.strip():
            return self._delete_legacy_memory(legacy_path)

        user_profile_path = "/profile/USER_PROFILE.md"
        research_taste_path = "/profile/RESEARCH_TASTE.md"
        imports: dict[str, list[str]] = {
            user_profile_path: [],
            research_taste_path: [],
        }
        recognized_paths = {
            "User Profile": user_profile_path,
            "Research Preferences": research_taste_path,
            "Experiment History": user_profile_path,
            "Learned Preferences": user_profile_path,
        }

        preface, legacy_sections = _legacy_sections(legacy)
        preface_body = _clean_legacy_preface(preface)
        if preface_body:
            imports[user_profile_path].append(f"### Notes\n{preface_body}")
        for heading, body in legacy_sections:
            cleaned = _clean_legacy_body(body)
            if not cleaned:
                continue
            target_path = recognized_paths.get(heading, user_profile_path)
            imports.setdefault(target_path, []).append(f"### {heading}\n{cleaned}")

        imported_any = False
        for memory_path, bodies in imports.items():
            if not bodies:
                continue
            path = self._file_path(memory_path)
            content = self._read_text(path)
            if content is None:
                logger.warning(
                    "Skipping legacy memory migration for missing profile %s", path
                )
                return False
            body = "\n\n".join(bodies)
            if not self._write_text(path, _append_imported_section(content, body)):
                return False
            imported_any = True

        if not imported_any:
            logger.debug("Legacy MEMORY.md contained no real content to migrate")

        return self._delete_legacy_memory(legacy_path)

    def _read_profile_records(self) -> list[tuple[str, str]]:
        """Load all profile files after bootstrapping and legacy migration."""
        records = self._ensure_profile_files()
        if self._migrate_legacy_memory():
            records = [
                (memory_path, self._read_text(self._file_path(memory_path)) or "")
                for memory_path, _ in records
            ]
        return records

    def _profile_context_from_records(self, records: list[tuple[str, str]]) -> str:
        """Inline profile contents unless they exceed the prompt budget."""
        full = "\n\n".join(
            f"File: {_agent_path(path)}\n\n{content.strip()}"
            for path, content in records
            if content.strip()
        ).strip()
        if len(full) <= self._max_inline_profile_chars:
            return full
        return self._profile_pointer_context

    def _read_profile_memory(self) -> str:
        """Return profile context, falling back to file pointers on setup errors."""
        try:
            records = self._read_profile_records()
            return self._profile_context_from_records(records)
        except Exception as e:
            logger.debug("Failed to read profile memory: %s", e)
            return self._profile_pointer_context

    def _inject_profile_context(
        self, request: ModelRequest, profile_content: str
    ) -> ModelRequest:
        """Append profile context and editing guidance to the system prompt."""
        from deepagents.middleware._utils import append_to_system_message

        injection = PROFILE_INJECTION_TEMPLATE.format(
            profile_content=profile_content,
            project_id=self._project_id,
        )
        new_system = append_to_system_message(request.system_message, injection)
        return request.override(system_message=new_system)

    def modify_request(self, request: ModelRequest) -> ModelRequest:
        """Apply profile memory injection for synchronous model calls."""
        profile_content = self._read_profile_memory()
        if not profile_content:
            profile_content = self._profile_pointer_context
        return self._inject_profile_context(request, profile_content)

    async def amodify_request(self, request: ModelRequest) -> ModelRequest:
        """Async profile injection; file reads run off the event loop."""
        profile_content = await asyncio.to_thread(self._read_profile_memory)
        if not profile_content:
            profile_content = self._profile_pointer_context
        return self._inject_profile_context(request, profile_content)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Middleware hook for injecting context before the sync model handler."""
        return handler(self.modify_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Middleware hook for injecting context before the async model handler."""
        return await handler(await self.amodify_request(request))


def create_memory_middleware(
    memory_dir: str | None = None,
    workspace_dir: str | Path | None = None,
    max_inline_profile_chars: int = DEFAULT_MAX_INLINE_PROFILE_CHARS,
) -> EvoMemoryMiddleware:
    """Build profile-memory middleware, defaulting to the shared memories directory."""

    if memory_dir is None:
        memory_dir = str(_paths.MEMORIES_DIR)

    return EvoMemoryMiddleware(
        memory_dir=memory_dir,
        workspace_dir=workspace_dir,
        max_inline_profile_chars=max_inline_profile_chars,
    )
