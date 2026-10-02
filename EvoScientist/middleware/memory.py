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
import contextlib
import logging
import os
import re
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal

import yaml
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import HumanMessage

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

_PROFILE_BOOTSTRAP_CONSENT = """\
 — one small question (one `ask_user` call with a single multiple choice if
that tool is available, otherwise in the same message): are they willing to
spend a little time letting you get to know them better, so you can grow into
their research assistant? Offer yes / later / no.
- later, or they ignore the question → drop the subject and get to work; you
  will ask again in a future session.
- no → set `intro: skipped` in the frontmatter with `edit_file` and never
  raise it again."""

_PROFILE_BOOTSTRAP_CORE = """\
Ask — with one `ask_user` call if that tool is available, otherwise in one
plain message — for whatever of these you do not have yet:
- How they would like to be addressed. A real name or a nickname both work; a
  real name lets you find their published work later.
- A homepage, Google Scholar, or GitHub link (optional).
- Their field, as a multiple choice: Computer science / AI · Life sciences /
  medicine · Physics / chemistry / materials · Social sciences / psychology ·
  Mathematics / statistics · Not decided yet — new to research.
Record the answers in `/memories/profile/USER_PROFILE.md` with `edit_file`: set
`name:` (required) and `field:` / `homepage:` when given in the frontmatter,
always double-quoting the value (`name: "…"`); put anything else stable they
told you as bullets under the existing headings.
If their answers invite it and they seem engaged, you may continue with one or
two natural follow-up questions (current project, what they are stuck on, how
they like reports) — conversationally, not as another form. Write anything
stable into the profile. Stop as soon as they signal they want to get to work.
Close briefly with how you will grow (their corrections →
`/memories/profile/RESEARCH_TASTE.md`; failed runs and environment traps →
observations; they can edit these files directly), then start their task.
Do not search the web, read papers, or draft research taste in this turn."""

PROFILE_BOOTSTRAP_FIRST = f"""
<profile_bootstrap>
This is your first exchange with this researcher: `USER_PROFILE.md` has no `name` yet.

In this turn, in the user's language: introduce yourself in two or three
sentences. If <profile_memory> already holds notes about them, greet them as a
returning collaborator, not a stranger. If their message is already a task,
acknowledge it first and keep the whole opening shorter.

Ask for consent before any survey{_PROFILE_BOOTSTRAP_CONSENT}
Only after a yes, ask three things:
{_PROFILE_BOOTSTRAP_CORE}
If they brought no task, propose one concrete first task from what they told
you instead; for someone new to research, offer to map the field together
first. Keep the opening light.
</profile_bootstrap>
"""

PROFILE_BOOTSTRAP_RETRY = f"""
<profile_bootstrap>
You have worked with this researcher for several sessions, but `USER_PROFILE.md`
still has no `name`. Once, lightly, in the user's language, ask for consent
again{_PROFILE_BOOTSTRAP_CONSENT}
Only after a yes, continue as on first contact:
{_PROFILE_BOOTSTRAP_CORE}
Do not repeat the full introduction.
</profile_bootstrap>
"""

_USER_PROFILE_PATH = "/profile/USER_PROFILE.md"
_BOOKKEEPING_KEY = "evoscientist"
_FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n?", re.DOTALL)

USER_PROFILE_FRONTMATTER: dict[str, object] = {
    "name": "",
    "field": "",
    "homepage": "",
    "intro": "pending",
    _BOOKKEEPING_KEY: {
        "sessions": 0,
        "intro_attempts": 0,
        "last_thread": "",
        "intro_asked_thread": "",
    },
}

_USER_PROFILE_BODY = """# User profile

Things worth remembering about the person using EvoScientist.

## Stable facts

## Preferences

## Collaboration style

## Constraints
"""


def _default_user_profile_frontmatter() -> dict[str, object]:
    """Fresh copy of the default frontmatter (nested dict included)."""
    return {
        key: dict(value) if isinstance(value, dict) else value
        for key, value in USER_PROFILE_FRONTMATTER.items()
    }


def _split_frontmatter(text: str) -> tuple[dict[str, object] | None, str]:
    """Split a leading YAML frontmatter.

    Returns ``({}, text)`` when no frontmatter block is present, and
    ``(None, text)`` when a block is present but unparsable or not a mapping.
    """
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        return {}, text
    try:
        meta = yaml.safe_load(match.group(1))
    except yaml.YAMLError as e:
        logger.debug("Ignoring malformed profile frontmatter: %s", e)
        return None, text
    if not isinstance(meta, dict):
        return None, text
    return meta, text[match.end() :]


def _join_frontmatter(meta: dict[str, object], body: str) -> str:
    dumped = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True).rstrip("\n")
    return f"---\n{dumped}\n---\n{body}"


# Ask again on sessions 1, 2, 4, 8, ... — never gives up, but ever quieter.
_BOOTSTRAP_MAX_EXPONENT = 62

BootstrapVariant = Literal["first", "retry"]


def _meta_str(value: object) -> str:
    if isinstance(value, bool) or value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    return value.strip() if isinstance(value, str) else ""


def _meta_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _bootstrap_view(meta: dict[str, object]) -> dict[str, object]:
    """Bootstrap-relevant fields of a profile frontmatter with defaults applied."""
    book = meta.get(_BOOKKEEPING_KEY)
    if not isinstance(book, dict):
        book = {}
    return {
        "name": _meta_str(meta.get("name")),
        "intro": _meta_str(meta.get("intro")) or "pending",
        "sessions": _meta_int(book.get("sessions")),
        "intro_attempts": _meta_int(book.get("intro_attempts")),
        "last_thread": _meta_str(book.get("last_thread")),
        "intro_asked_thread": _meta_str(book.get("intro_asked_thread")),
    }


def _bootstrap_decision(
    view: dict[str, object],
    *,
    thread_id: str | None,
    human_messages: int,
) -> BootstrapVariant | None:
    """Which first-contact block (if any) this model call should carry."""
    if view["name"] or view["intro"] == "skipped":
        return None
    if human_messages != 1:
        return None
    if thread_id is None:
        return "first"
    if view["intro_asked_thread"] == thread_id:
        return "first" if view["intro_attempts"] <= 1 else "retry"
    attempts = min(view["intro_attempts"], _BOOTSTRAP_MAX_EXPONENT)
    if view["sessions"] >= 1 << attempts:
        return "first" if attempts == 0 else "retry"
    return None


def _apply_bootstrap_view(
    meta: dict[str, object], view: dict[str, object]
) -> dict[str, object]:
    """Merge bookkeeping from *view* into *meta*, filling missing identity keys."""
    merged = _default_user_profile_frontmatter()
    merged.update({k: v for k, v in meta.items() if k != _BOOKKEEPING_KEY})
    book = meta.get(_BOOKKEEPING_KEY)
    merged[_BOOKKEEPING_KEY] = {
        **(book if isinstance(book, dict) else {}),
        "sessions": view["sessions"],
        "intro_attempts": view["intro_attempts"],
        "last_thread": view["last_thread"],
        "intro_asked_thread": view["intro_asked_thread"],
    }
    return merged


def _current_thread_id() -> str | None:
    """Thread id of the running graph, or None outside a runnable context."""
    try:
        from langgraph.config import get_config

        config = get_config()
    except Exception:
        return None
    if not isinstance(config, dict):
        return None
    configurable = config.get("configurable") or {}
    if not isinstance(configurable, dict):
        return None
    thread_id = configurable.get("thread_id")
    return thread_id if isinstance(thread_id, str) and thread_id else None


def _count_human_messages(state: object) -> int:
    messages = state.get("messages") if isinstance(state, dict) else None
    if not isinstance(messages, (list, tuple)):
        return 0
    # Skip synthetic HumanMessages (e.g. summarization) — not the user's turn.
    return sum(
        1
        for message in messages
        if isinstance(message, HumanMessage)
        and message.additional_kwargs.get("lc_source") is None
    )


PROFILE_TEMPLATES: dict[str, str] = {
    "/profile/SOUL.md": """# EvoScientist soul

Default behavior for this copy of EvoScientist.

## Operating principles

## Voice

## Lines not to cross
""",
    "/profile/USER_PROFILE.md": _join_frontmatter(
        _default_user_profile_frontmatter(), _USER_PROFILE_BODY
    ),
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
        enable_profile_bootstrap: bool = False,
    ) -> None:
        self._memory_dir = Path(memory_dir).expanduser()
        workspace = Path(workspace_dir or _paths.WORKSPACE_ROOT).expanduser()
        self._workspace_dir = workspace
        self._project_id = resolve_project_id(workspace)
        self._enable_profile_memory = enable_profile_memory
        self._enable_profile_bootstrap = enable_profile_bootstrap
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

        # The prompt-facing observation index is rebuilt fresh on every model
        # call (see ``modify_request`` / ``amodify_request``), so this stored
        # value never reaches a prompt — it is only the error-fallback returned
        # by ``_refresh_observation_index_context`` when a refresh raises, and
        # on an unreadable store that eager build already falls through to "".
        # Reading the whole observation store here to seed it therefore buys
        # nothing and costs ~0.5-1.4s per middleware — paid 12x on a
        # deployed-graph rebuild (main agent + 11 sub-agents). Create the search
        # dirs (the load-bearing side effect) and defer the read to first use.
        self._ensure_observation_dirs()

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
        """Write UTF-8 text atomically, creating parent directories as needed."""
        tmp_path: Path | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
            )
            tmp_path = Path(tmp_name)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
            os.replace(tmp_path, path)
        except OSError as e:
            logger.warning("Failed to write profile memory %s: %s", path, e)
            if tmp_path is not None:
                with contextlib.suppress(OSError):
                    tmp_path.unlink()
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
            elif (
                memory_path == _USER_PROFILE_PATH
                and content.strip()
                and _FRONTMATTER_RE.match(content) is None
            ):
                # Pre-frontmatter profiles: prepend defaults, keep the body verbatim.
                content = _join_frontmatter(
                    _default_user_profile_frontmatter(), content
                )
                self._write_text(path, content)
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

    def _bootstrap_context(self, *, thread_id: str | None, human_messages: int) -> str:
        """First-contact block for this call; also bumps the frontmatter bookkeeping."""
        if not (self._enable_profile_memory and self._enable_profile_bootstrap):
            return ""
        path = self._file_path(_USER_PROFILE_PATH)
        try:
            content = self._read_text(path)
        except Exception as e:
            logger.debug("Skipping profile bootstrap: %s", e)
            return ""
        # Empty means a concurrent writer is mid-truncate: never replace it.
        if content is None or not content.strip():
            return ""

        meta, body = _split_frontmatter(content)
        if meta is None:
            logger.warning("Unparsable frontmatter in %s; skipping bootstrap", path)
            return ""
        view = _bootstrap_view(meta)
        dirty = False
        if thread_id is not None and view["last_thread"] != thread_id:
            view["sessions"] += 1
            view["last_thread"] = thread_id
            dirty = True
        variant = _bootstrap_decision(
            view, thread_id=thread_id, human_messages=human_messages
        )
        if (
            variant is not None
            and thread_id is not None
            and view["intro_asked_thread"] != thread_id
        ):
            view["intro_attempts"] += 1
            view["intro_asked_thread"] = thread_id
            dirty = True
        if dirty:
            merged = _apply_bootstrap_view(meta, view)
            self._write_text(path, _join_frontmatter(merged, body))

        if variant == "first":
            return PROFILE_BOOTSTRAP_FIRST
        if variant == "retry":
            return PROFILE_BOOTSTRAP_RETRY
        return ""

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
        bootstrap_context: str = "",
    ) -> str:
        """Build request memory context ordered from static to dynamic."""
        return "\n\n".join(
            part
            for part in (
                self._memory_instructions_context(),
                observation_index_context,
                self._profile_memory_context(profile_content),
                bootstrap_context.strip(),
            )
            if part
        )

    def _inject_memory_context(
        self,
        request: ModelRequest,
        *,
        observation_index_context: str,
        profile_content: str,
        bootstrap_context: str = "",
    ) -> ModelRequest:
        """Append memory context and editing guidance to the system prompt."""
        if not self._enable_profile_memory and not self._enable_observation_memory:
            return request

        injection = self._memory_context_for_request(
            observation_index_context=observation_index_context,
            profile_content=profile_content,
            bootstrap_context=bootstrap_context,
        )
        new_system = append_to_system_message(request.system_message, injection)
        return request.override(system_message=new_system)

    def _profile_context_for_request(self) -> str:
        if not self._enable_profile_memory:
            return ""
        return self._read_profile_memory()

    def modify_request(self, request: ModelRequest) -> ModelRequest:
        """Apply memory injection for synchronous model calls."""
        profile_content = self._profile_context_for_request()
        return self._inject_memory_context(
            request,
            observation_index_context=self._refresh_observation_index_context(),
            profile_content=profile_content,
            bootstrap_context=self._bootstrap_context(
                thread_id=_current_thread_id(),
                human_messages=_count_human_messages(request.state),
            ),
        )

    async def amodify_request(self, request: ModelRequest) -> ModelRequest:
        """Apply memory injection for asynchronous model calls."""
        observation_index_context = ""
        profile_context = ""
        bootstrap_context = ""
        # Resolved on the event-loop thread: get_config() reads a contextvar.
        thread_id = _current_thread_id()
        human_messages = _count_human_messages(request.state)

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

        # After the profile read so the file exists on a brand-new memory dir.
        if self._enable_profile_memory and self._enable_profile_bootstrap:
            bootstrap_context = await asyncio.to_thread(
                self._bootstrap_context,
                thread_id=thread_id,
                human_messages=human_messages,
            )

        return self._inject_memory_context(
            request,
            observation_index_context=observation_index_context,
            profile_content=profile_context,
            bootstrap_context=bootstrap_context,
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
    enable_profile_bootstrap: bool = False,
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
        enable_profile_bootstrap=enable_profile_bootstrap,
    )
