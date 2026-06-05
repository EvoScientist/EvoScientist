"""Memory middleware for EvoScientist.

The middleware owns the markdown files under ``/memories/profile/``: it creates
them when missing, migrates the old ``/memories/MEMORY.md`` file when present,
injects either profile contents or profile file pointers into model calls, and
points agents at observation memory under ``/memories/observations/``. Agents
still read and edit profile files through their normal ``/memories/...`` tools;
observation writes go through the structured ``record_observation`` tool.
"""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import yaml
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)

from .. import paths as _paths
from ..memory import (
    MemoryScope,
    MemorySourceType,
    MemoryType,
    create_record_observation_tool,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_INLINE_PROFILE_CHARS = 24_000
DEFAULT_MAX_INLINE_OBSERVATION_INDEX_CHARS = 12_000
_LEGACY_MEMORY_FILENAME = "MEMORY.md"
_LEGACY_IMPORT_HEADING = "Imported from legacy MEMORY.md"


PROFILE_INJECTION_TEMPLATE = """<profile_memory>
{profile_content}
</profile_memory>
{observation_memory}

<memory_instructions>
These profile notes live under `/memories/profile/`.
Every agent can read and update them with normal file tools.

Use these files for:
- `/memories/profile/SOUL.md`: how this copy should usually behave; voice and boundaries.
- `/memories/profile/USER_PROFILE.md`: facts and preferences about the user.
- `/memories/profile/RESEARCH_TASTE.md`: research interests, standards, methods that fit, and things to avoid.
- `/memories/profile/projects/{project_id}/PROJECT_PROFILE.md`: conventions, commands, and pitfalls for this workspace.

Read the relevant file before editing it. Add small bullets under existing
headings, skip duplicates, and leave out temporary task state.

Profile update scope:
- Review the profile context above and the latest trajectory for stable changes
  to user preferences, research taste, collaboration style, or project
  conventions.
- Do not infer profile facts from task content alone. Profile updates need
  stable evidence about the user, their preferences, or this project.
- When a profile update is warranted, edit the relevant
  `/memories/profile/...` file with a small deduplicated bullet under an
  existing heading.
- When the turn only contains task progress, subagent findings, search results,
  command output, or temporary run context, leave profile files unchanged.
{observation_instructions}
</memory_instructions>"""

OBSERVATION_MEMORY_READ_INSTRUCTIONS = """
Observation memory lives under `/memories/observations/`:
- `/memories/observations/global/`: cross-project observations.
- `/memories/observations/projects/{project_id}/`: observations for this workspace.

Memory preflight:
- For main-agent and subagent work, before planning, running commands,
  implementing, debugging, analyzing, or writing reports, run a quick search of
  observation memory unless the task is clearly trivial or the observation
  directories are empty.
- Use file tools, not shell paths: start with `grep` on `/memories/observations/`
  using task keywords, then `read_file` the relevant hits by id/path. Use
  `glob` or `ls` only to inspect what exists when grep returns nothing useful.
- When the task calls for a specific kind of memory, grep frontmatter first:
  `memory_type: procedural` for reusable commands/workarounds, `memory_type:
  semantic` for reusable facts/findings, `scope: project` for workspace-local
  notes, and `scope: global` for cross-project notes.
- Mention the result briefly in your plan or handoff: which observation mattered,
  or that no relevant observation was found. Do not let this become a long detour.
"""

OBSERVATION_MEMORY_WRITE_INSTRUCTIONS = """
Call `record_observation` only for durable, non-obvious, evidence-backed
information that is not already in memory and is likely to change future behavior:
recurring constraints, important decisions, failed approaches future agents might
repeat, verified evaluator outcomes, or tool/workflow workarounds.
Provide a one-line `summary` that is specific enough for future agents to decide
whether to read the full observation.

Distill reusable insight rather than saving raw task output or a transcript of
what happened.

Use procedural/global for general tool or platform behavior that can recur
outside this workspace; use project scope only for workspace-specific facts,
commands, datasets, benchmarks, or config. Do not hand-write observation files.
Do not record routine progress, raw traces, ordinary command output, citation
lists without synthesis, simple filesystem listings, temporary paths/run ids,
one-off environment discoveries, or task summaries."""

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

Research taste to keep in mind: interests, standards, methods that tend to fit, and things to avoid.

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


@dataclass(frozen=True)
class ObservationIndexRecord:
    """One summary-bearing observation listed in the system prompt index."""

    observation_id: str
    memory_path: str
    memory_type: MemoryType
    scope: MemoryScope
    summary: str


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
        source_type: MemorySourceType = MemorySourceType.TURN,
        source_agent: str = "EvoScientist",
        enable_profile_memory: bool = True,
        enable_observation_memory: bool = True,
        enable_observation_tool: bool = True,
    ) -> None:
        self._memory_dir = Path(memory_dir).expanduser()
        workspace = Path(workspace_dir or _paths.WORKSPACE_ROOT).expanduser()
        self._project_id = _resolve_project_id(workspace)
        self._enable_profile_memory = enable_profile_memory
        self._enable_observation_memory = enable_observation_memory
        self._profile_specs = _profile_specs(self._project_id)
        pointer_lines = ["Profile files are available at:"]
        pointer_lines.extend(
            f"- {_agent_path(path)}" for path, _ in self._profile_specs
        )
        self._profile_pointer_context = "\n".join(pointer_lines)
        self._max_inline_profile_chars = max_inline_profile_chars
        self._enable_observation_tool = (
            enable_observation_memory and enable_observation_tool
        )
        self.tools = (
            [
                create_record_observation_tool(
                    memory_dir=self._memory_dir,
                    project_id=self._project_id,
                    source_type=source_type,
                    source_agent=source_agent,
                )
            ]
            if self._enable_observation_tool
            else []
        )
        self._observation_index_records = []
        self._observation_index_context = ""
        if not enable_observation_memory:
            return

        self._ensure_observation_dirs()
        self._observation_index_records = self._read_observation_index_records()
        self._observation_index_context = self._observation_index_context_from_records(
            self._observation_index_records
        )

    @property
    def project_id(self) -> str:
        """Stable project id used for this middleware's project memory paths."""
        return self._project_id

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

    def _ensure_observation_dirs(self) -> None:
        """Create the observation directories agents are prompted to search."""
        for memory_path in (
            "/observations/global",
            f"/observations/projects/{self._project_id}",
        ):
            try:
                self._file_path(memory_path).mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.warning("Failed to create observation memory dir: %s", e)

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

    def _read_bootstrapped_profile_records(self) -> list[tuple[str, str]]:
        records = self._ensure_profile_files()
        if self._migrate_legacy_memory():
            records = [
                (memory_path, self._read_text(self._file_path(memory_path)) or "")
                for memory_path, _ in records
            ]
        return records

    def _read_profile_records(self) -> list[tuple[str, str]]:
        """Load all profile files after bootstrapping and legacy migration."""
        if not self._enable_observation_memory:
            return self._read_bootstrapped_profile_records()

        self._ensure_observation_dirs()
        return self._read_bootstrapped_profile_records()

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
        """Return profile context, falling back to file pointers."""
        try:
            records = self._read_profile_records()
            return (
                self._profile_context_from_records(records)
                or self._profile_pointer_context
            )
        except Exception as e:
            logger.debug("Failed to read profile memory: %s", e)
            return self._profile_pointer_context

    def _observation_memory_paths(self) -> list[Path]:
        """Return summary-indexable observation files for this project context."""
        paths: list[Path] = []
        for memory_path in (
            "/observations/global",
            f"/observations/projects/{self._project_id}",
        ):
            directory = self._file_path(memory_path)
            try:
                paths.extend(sorted(directory.glob("*.md")))
            except OSError as e:
                logger.warning("Failed to list observation memory %s: %s", directory, e)
        return paths

    def _read_observation_frontmatter(self, path: Path) -> dict[str, object] | None:
        """Read explicit YAML frontmatter for an observation file."""
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            logger.warning("Failed to read observation memory %s: %s", path, e)
            return None
        if not text.startswith("---\n"):
            return None
        try:
            frontmatter, _body = text.removeprefix("---\n").split("\n---\n", 1)
            metadata = yaml.safe_load(frontmatter)
        except (ValueError, yaml.YAMLError):
            return None
        if not isinstance(metadata, dict):
            return None
        return {key: value for key, value in metadata.items() if isinstance(key, str)}

    def _observation_index_record_from_path(
        self, path: Path
    ) -> ObservationIndexRecord | None:
        """Return an index record only when explicit summary metadata exists."""
        metadata = self._read_observation_frontmatter(path)
        if metadata is None:
            return None

        observation_id = str(metadata.get("id") or "").strip()
        summary = str(metadata.get("summary") or "").strip()
        memory_type_value = str(metadata.get("memory_type") or "").strip()
        scope_value = str(metadata.get("scope") or "").strip()
        if (
            not observation_id
            or not summary
            or not memory_type_value
            or not scope_value
        ):
            return None
        try:
            memory_type = MemoryType(memory_type_value)
            scope = MemoryScope(scope_value)
        except ValueError:
            return None

        try:
            memory_path = "/" + path.relative_to(self._memory_dir).as_posix()
        except ValueError:
            return None
        return ObservationIndexRecord(
            observation_id=observation_id,
            memory_path=memory_path,
            memory_type=memory_type,
            scope=scope,
            summary=summary,
        )

    def _read_observation_index_records(self) -> list[ObservationIndexRecord]:
        """Load summary-bearing observation records for prompt indexing."""
        records = [
            record
            for path in self._observation_memory_paths()
            if (record := self._observation_index_record_from_path(path)) is not None
        ]
        return sorted(records, key=lambda record: record.observation_id)

    def _observation_index_count_line(
        self, records: list[ObservationIndexRecord]
    ) -> str:
        """Return compact observation counts by scope and memory type."""
        scope_counts = dict.fromkeys(MemoryScope, 0)
        type_counts = dict.fromkeys(MemoryType, 0)
        for record in records:
            scope_counts[record.scope] += 1
            type_counts[record.memory_type] += 1
        return (
            f"Counts: total={len(records)}; "
            f"scope global={scope_counts[MemoryScope.GLOBAL]}, "
            f"project={scope_counts[MemoryScope.PROJECT]}; "
            f"type semantic={type_counts[MemoryType.SEMANTIC]}, "
            f"procedural={type_counts[MemoryType.PROCEDURAL]}, "
            f"episodic={type_counts[MemoryType.EPISODIC]}."
        )

    def _observation_search_hints(self) -> str:
        """Return stable search hints for observation memory."""
        return "\n".join(
            [
                "Search hints:",
                "- Grep by id when you already know it from the index.",
                (
                    "- Grep frontmatter by type when appropriate: "
                    "`memory_type: procedural`, `memory_type: semantic`, or "
                    "`memory_type: episodic`."
                ),
                (
                    "- Grep frontmatter by scope when appropriate: "
                    "`scope: project` or `scope: global`."
                ),
                "- Combine those with task keywords, then read relevant hits.",
            ]
        )

    def _observation_index_context_from_records(
        self,
        records: list[ObservationIndexRecord],
        *,
        max_inline_chars: int = DEFAULT_MAX_INLINE_OBSERVATION_INDEX_CHARS,
    ) -> str:
        """Build the static observation index injected into the system prompt."""
        header = "\n".join(
            [
                "<observation_memory>",
                "Observation index loaded at agent start.",
                self._observation_index_count_line(records),
            ]
        )
        if not records:
            return "\n".join(
                [header, self._observation_search_hints(), "</observation_memory>"]
            )

        lines = [
            f"- {record.observation_id} "
            f"[{record.memory_type.value}/{record.scope.value}] "
            f"{_agent_path(record.memory_path)}: {record.summary}"
            for record in records
        ]
        full = "\n".join(
            [
                header,
                "Indexed observations:",
                *lines,
                self._observation_search_hints(),
                "</observation_memory>",
            ]
        )
        if len(full) <= max_inline_chars:
            return full
        return "\n".join(
            [
                header,
                "Observation summaries are too large to inline; search on demand.",
                self._observation_search_hints(),
                "</observation_memory>",
            ]
        )

    def _observation_memory_instructions(self) -> str:
        if not self._enable_observation_memory:
            return ""

        instructions = OBSERVATION_MEMORY_READ_INSTRUCTIONS.format(
            project_id=self._project_id
        )
        if not self._enable_observation_tool:
            return instructions
        return instructions + OBSERVATION_MEMORY_WRITE_INSTRUCTIONS

    def _inject_profile_context(
        self, request: ModelRequest, profile_content: str
    ) -> ModelRequest:
        """Append profile context and editing guidance to the system prompt."""
        from deepagents.middleware._utils import append_to_system_message

        if not self._enable_profile_memory and not self._enable_observation_memory:
            return request

        observation_instructions = self._observation_memory_instructions()

        if not self._enable_profile_memory:
            injection = "\n\n".join(
                part
                for part in (
                    self._observation_index_context,
                    (
                        "<memory_instructions>\n"
                        f"{observation_instructions.strip()}\n"
                        "</memory_instructions>"
                    )
                    if observation_instructions.strip()
                    else "",
                )
                if part
            )
            new_system = append_to_system_message(request.system_message, injection)
            return request.override(system_message=new_system)

        injection = PROFILE_INJECTION_TEMPLATE.format(
            profile_content=profile_content,
            observation_memory=self._observation_index_context,
            project_id=self._project_id,
            observation_instructions=observation_instructions,
        )
        new_system = append_to_system_message(request.system_message, injection)
        return request.override(system_message=new_system)

    def _profile_context_for_request(self) -> str:
        if not self._enable_profile_memory:
            return ""
        return self._read_profile_memory()

    def modify_request(self, request: ModelRequest) -> ModelRequest:
        """Apply memory injection for synchronous model calls."""
        return self._inject_profile_context(
            request, self._profile_context_for_request()
        )

    async def amodify_request(self, request: ModelRequest) -> ModelRequest:
        """Apply memory injection for asynchronous model calls."""
        profile_context = ""
        if self._enable_profile_memory:
            profile_context = await asyncio.to_thread(self._read_profile_memory)
        return self._inject_profile_context(request, profile_context)

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
    source_type: MemorySourceType = MemorySourceType.TURN,
    source_agent: str = "EvoScientist",
    enable_profile_memory: bool = True,
    enable_observation_memory: bool = True,
    enable_observation_tool: bool = True,
) -> EvoMemoryMiddleware:
    """Build profile-memory middleware, defaulting to the shared memories directory."""

    if memory_dir is None:
        memory_dir = str(_paths.MEMORIES_DIR)

    return EvoMemoryMiddleware(
        memory_dir=memory_dir,
        workspace_dir=workspace_dir,
        max_inline_profile_chars=max_inline_profile_chars,
        source_type=source_type,
        source_agent=source_agent,
        enable_profile_memory=enable_profile_memory,
        enable_observation_memory=enable_observation_memory,
        enable_observation_tool=enable_observation_tool,
    )
