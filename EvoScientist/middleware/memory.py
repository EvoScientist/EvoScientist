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
from collections.abc import Awaitable, Callable
from pathlib import Path

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)

from .. import paths as _paths
from ..memory import (
    MemorySourceType,
    ObservationRecordResult,
    build_observation_index_context,
    create_read_memory_tool,
    create_record_observation_tool,
    create_search_observations_tool,
)
from ..memory.project import resolve_project_id
from ..memory.scheduler import MemoryScheduler, ObservationLinkerContext
from .utils import append_to_system_message

logger = logging.getLogger(__name__)

DEFAULT_MAX_INLINE_PROFILE_CHARS = 24_000
_LEGACY_MEMORY_FILENAME = "MEMORY.md"
_LEGACY_IMPORT_HEADING = "Imported from legacy MEMORY.md"


PROFILE_MEMORY_INSTRUCTIONS = """
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
- Review the profile context below and the latest trajectory for stable changes
  to user preferences, research taste, collaboration style, or project
  conventions.
- Do not infer profile facts from task content alone. Profile updates need
  stable evidence about the user, their preferences, or this project.
- When a profile update is warranted, edit the relevant
  `/memories/profile/...` file with a small deduplicated bullet under an
  existing heading.
- When the turn only contains task progress, subagent findings, search results,
  command output, or temporary run context, leave profile files unchanged.
"""

OBSERVATION_MEMORY_READ_INSTRUCTIONS = """
Observation memory lives under `/memories/observations/`:
- `/memories/observations/global/`: cross-project observations.
- `/memories/observations/projects/{project_id}/`: observations for this workspace.

Required memory preflight:
- For coding, debugging, research, planning, or evaluation tasks, complete this
  preflight before inspecting workspace/task files, running commands, editing
  files, delegating, using `code_interpreter`, or making a plan.
- First use the inlined observation index. If a listed summary clearly matches
  the task, call `read_memory` with that observation ID.
- Otherwise, call `search_observations` with a few distinctive words or short
  phrases that describe the issue, constraint, procedure, or prior result to
  find. If one query misses, try 1-3 focused variants. Use `mode=regex` only
  when exact grep-like matching is required. If a result looks promising but
  the snippet is not enough to act on confidently, call `read_memory` with its
  observation ID.
- After this preflight, use direct tools or `code_interpreter` to do or batch
  the actual workspace work as appropriate.
- Mention the result briefly before continuing: observation IDs used, or that
  no relevant observation was found. Keep this preflight short.
"""

OBSERVATION_MEMORY_WRITE_INSTRUCTIONS = """
Call `record_observation` only for durable, non-obvious, evidence-backed
information that is not already in memory and is likely to change future behavior:
recurring constraints, important decisions, failed approaches future agents might
repeat, verified outcomes, or tool/workflow workarounds.
Provide a one-line `summary` that is specific enough for future agents to decide
whether to read the full observation.

Distill reusable insight rather than saving raw task output or a transcript of
what happened.

Use procedural/global for general tool or platform behavior that can recur
outside this workspace; use project scope only for workspace-specific facts,
commands, resources, evaluation setup, or configuration. Do not hand-write
observation files.
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
        source_type: MemorySourceType = MemorySourceType.TURN,
        source_agent: str = "EvoScientist",
        enable_profile_memory: bool = True,
        enable_observation_memory: bool = True,
        enable_observation_tool: bool = True,
        memory_scheduler: MemoryScheduler | None = None,
    ) -> None:
        self._memory_dir = Path(memory_dir).expanduser()
        workspace = Path(workspace_dir or _paths.WORKSPACE_ROOT).expanduser()
        self._workspace_dir = workspace
        self._project_id = resolve_project_id(workspace)
        self._enable_profile_memory = enable_profile_memory
        self._enable_observation_memory = enable_observation_memory
        self._memory_scheduler = memory_scheduler
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
        self.tools = []
        if enable_observation_memory:
            self.tools.append(
                create_search_observations_tool(
                    memory_dir=self._memory_dir,
                    project_id=self._project_id,
                )
            )
            self.tools.append(
                create_read_memory_tool(
                    memory_dir=self._memory_dir,
                    project_id=self._project_id,
                )
            )
        if self._enable_observation_tool:
            self.tools.append(
                create_record_observation_tool(
                    memory_dir=self._memory_dir,
                    project_id=self._project_id,
                    source_type=source_type,
                    source_agent=source_agent,
                    on_observation_recorded=self._record_observation_created,
                )
            )
        self._observation_index_context = ""
        if not enable_observation_memory:
            return

        self._refresh_observation_index_context()

    @property
    def project_id(self) -> str:
        """Stable project id used for this middleware's project memory paths."""
        return self._project_id

    def _record_observation_created(self, result: ObservationRecordResult) -> None:
        if self._memory_scheduler is None:
            return
        project_id = str(result.get("project_id") or self._project_id)
        self._memory_scheduler.record_observation_created(
            ObservationLinkerContext(
                memory_dir=self._memory_dir,
                workspace_dir=self._workspace_dir,
                project_id=project_id,
                observation_ids=(result["observation_id"],),
            )
        )

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
        """Create non-project observation directories agents are prompted to search."""
        try:
            self._file_path("/observations/global").mkdir(parents=True, exist_ok=True)
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

    def _refresh_observation_index_context(self) -> str:
        """Refresh the prompt observation index from current memory files."""
        if not self._enable_observation_memory:
            return ""
        try:
            self._ensure_observation_dirs()
            context = build_observation_index_context(
                memory_dir=self._memory_dir,
                project_id=self._project_id,
            )
        except OSError as e:
            logger.warning("Failed to refresh observation memory index: %s", e)
            return self._observation_index_context
        except Exception as e:
            logger.debug("Failed to refresh observation memory index: %s", e)
            return self._observation_index_context
        self._observation_index_context = context
        return context

    def _observation_memory_instructions(self) -> str:
        if not self._enable_observation_memory:
            return ""

        instructions = OBSERVATION_MEMORY_READ_INSTRUCTIONS.format(
            project_id=self._project_id
        )
        if not self._enable_observation_tool:
            return instructions
        return instructions + OBSERVATION_MEMORY_WRITE_INSTRUCTIONS

    def _memory_instructions_context(self) -> str:
        """Return static memory instructions for enabled memory features."""
        instructions = []
        if self._enable_profile_memory:
            instructions.append(
                PROFILE_MEMORY_INSTRUCTIONS.format(project_id=self._project_id)
            )
        if observation_instructions := self._observation_memory_instructions():
            instructions.append(observation_instructions)
        if not instructions:
            return ""
        return "\n".join(
            [
                "<memory_instructions>",
                "\n\n".join(part.strip() for part in instructions if part.strip()),
                "</memory_instructions>",
            ]
        )

    def _profile_memory_context(self, profile_content: str) -> str:
        """Return profile memory context for prompt injection."""
        if not self._enable_profile_memory:
            return ""
        return "\n".join(
            [
                "<profile_memory>",
                profile_content,
                "</profile_memory>",
            ]
        )

    def _memory_context_for_request(
        self,
        *,
        observation_index_context: str,
        profile_content: str,
    ) -> str:
        """Build request memory context ordered from static to dynamic."""
        return "\n\n".join(
            part
            for part in (
                self._memory_instructions_context(),
                observation_index_context,
                self._profile_memory_context(profile_content),
            )
            if part
        )

    def _inject_memory_context(
        self,
        request: ModelRequest,
        *,
        observation_index_context: str,
        profile_content: str,
    ) -> ModelRequest:
        """Append memory context and editing guidance to the system prompt."""
        if not self._enable_profile_memory and not self._enable_observation_memory:
            return request

        injection = self._memory_context_for_request(
            observation_index_context=observation_index_context,
            profile_content=profile_content,
        )
        new_system = append_to_system_message(request.system_message, injection)
        return request.override(system_message=new_system)

    def _profile_context_for_request(self) -> str:
        if not self._enable_profile_memory:
            return ""
        return self._read_profile_memory()

    def modify_request(self, request: ModelRequest) -> ModelRequest:
        """Apply memory injection for synchronous model calls."""
        return self._inject_memory_context(
            request,
            observation_index_context=self._refresh_observation_index_context(),
            profile_content=self._profile_context_for_request(),
        )

    async def amodify_request(self, request: ModelRequest) -> ModelRequest:
        """Apply memory injection for asynchronous model calls."""
        observation_index_context = ""
        profile_context = ""

        if self._enable_observation_memory and self._enable_profile_memory:
            observation_index_context, profile_context = await asyncio.gather(
                asyncio.to_thread(self._refresh_observation_index_context),
                asyncio.to_thread(self._read_profile_memory),
            )
        elif self._enable_observation_memory:
            observation_index_context = await asyncio.to_thread(
                self._refresh_observation_index_context
            )
        elif self._enable_profile_memory:
            profile_context = await asyncio.to_thread(self._read_profile_memory)

        return self._inject_memory_context(
            request,
            observation_index_context=observation_index_context,
            profile_content=profile_context,
        )

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
    memory_scheduler: MemoryScheduler | None = None,
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
        memory_scheduler=memory_scheduler,
    )
