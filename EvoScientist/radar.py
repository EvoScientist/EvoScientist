"""Research Radar — scheduled, preference-aware paper monitoring.

Builds a background LangGraph agent that periodically searches for
papers matching the user's research interests, compares them against
observation memory, and writes structured updates to
``/memories/radar/YYYY-MM-DD_HHMMSS.md``.

Scheduling uses an in-process ``asyncio.Task`` driven by ``croniter``
(cross-platform, no OS dependencies).  The scheduler lives inside the
CLI / TUI process — scans will not run while the process is stopped.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from .config.settings import EvoScientistConfig

logger = logging.getLogger(__name__)

RADAR_GRAPH_ID = "research-radar"
RADAR_DIR_NAME = "radar"
RADAR_PENDING_MARKER = ".radar_pending"


class PaperMatch(BaseModel):
    """A single paper matched by the radar scan."""

    title: str
    url: str
    relevance: str = Field(description="Why this paper matches the user's interests")
    connection: str = Field(
        description="How this paper relates to the user's ongoing research"
    )


class IdeaSketch(BaseModel):
    """A research idea inspired by the radar findings."""

    title: str
    motivation: str
    suggested_approach: str


class RadarUpdate(BaseModel):
    """Structured output from one radar scan."""

    papers: list[PaperMatch] = Field(default_factory=list)
    trends: list[str] = Field(default_factory=list)
    ideas: list[IdeaSketch] = Field(default_factory=list)
    suggested_actions: list[str] = Field(default_factory=list)
    summary_text: str = Field(description="Human-readable formatted update (markdown)")


_RADAR_SYSTEM_PROMPT = """\
You are the Research Radar — a background agent that monitors the research \
landscape on behalf of a researcher.

Your job is to:
1. Read the user's research taste from /memories/profile/RESEARCH_TASTE.md, \
paying special attention to the **Interests**, **Radar monitoring**, and \
**Radar feedback** sections.
2. Read the paper-navigator skill instructions at \
/skills/paper-navigator/SKILL.md to understand available search scripts.
3. Use the paper-navigator scripts to find recent papers matching the \
user's interests. Prefer arxiv_monitor.py (no API key needed). Use \
scholar_search.py only if S2_API_KEY is available. Run 3-5 targeted \
searches with --json flag covering different facets. Pass --days from \
the input message.
4. Cross-reference findings against the user's observation memory to \
understand how papers relate to their ongoing work.
5. After completing ALL searches, produce your final answer as a single \
JSON object (no markdown fences, just raw JSON) with exactly these fields:
{
  "papers": [{"title": "...", "url": "https://arxiv.org/abs/...", "relevance": "why it matches", "connection": "how it relates to ongoing work"}, ...],
  "trends": ["trend 1", "trend 2", ...],
  "ideas": [{"title": "...", "motivation": "...", "suggested_approach": "..."}, ...],
  "suggested_actions": ["action 1", "action 2", ...],
  "summary_text": "# [Research Radar] ...\\n\\nA concise markdown digest..."
}

Include 5-10 papers, 2-4 trends, 2-3 ideas, and 3-5 suggested actions.
Do NOT write any files. Do NOT produce the JSON until you have finished \
all searches and have actual results.

Guidelines:
- Prefer papers with code, benchmarks, or clear experimental protocols.
- Deprioritize pure application papers unless the user's taste says otherwise.
- Respect the user's **Radar feedback** — if they said a direction is \
irrelevant, skip it.
- The summary_text should be self-contained and readable without the \
structured fields.
- Include [Research Radar] prefix in the summary_text title so the main \
agent can recognize it.
- If searches return errors or no results, report the failure in \
summary_text so the user knows what happened.
"""


def is_paper_navigator_installed() -> bool:
    """Check if the paper-navigator skill is available."""
    from . import paths as _paths

    for d in (_paths.USER_SKILLS_DIR, _paths.GLOBAL_SKILLS_DIR):
        if (Path(d) / "paper-navigator" / "SKILL.md").exists():
            return True
    return False


def build_radar_graph() -> CompiledStateGraph:
    """Build the Research Radar agent graph."""
    from deepagents import create_deep_agent
    from deepagents.backends import CompositeBackend, FilesystemBackend

    from . import paths as _paths
    from .backends import CustomSandboxBackend, MergedSkillsBackend
    from .config.settings import MemoryControls, get_effective_config
    from .EvoScientist import _ensure_auxiliary_chat_model
    from .middleware.memory_lifecycle import (
        MemoryLifecycleRole,
        _memory_worker_middleware,
    )

    cfg = get_effective_config()
    memory_controls = MemoryControls.from_config(cfg)
    memory_dir = Path(_paths.MEMORIES_DIR).expanduser()
    workspace_dir = Path(_paths.WORKSPACE_ROOT).expanduser()

    if not os.environ.get("S2_API_KEY"):
        logger.info("S2_API_KEY not set — radar will use arXiv-only search")

    backend = CompositeBackend(
        default=CustomSandboxBackend(
            root_dir=str(workspace_dir),
            virtual_mode=True,
            timeout=cfg.sandbox_execute_timeout,
        ),
        routes={
            "/memories/": FilesystemBackend(
                root_dir=str(memory_dir), virtual_mode=True
            ),
            "/skills/": MergedSkillsBackend(
                primary_dir=str(_paths.USER_SKILLS_DIR),
                secondary_dir=str(Path(__file__).parent / "skills"),
                global_dir=str(_paths.GLOBAL_SKILLS_DIR),
            ),
        },
    )

    agent = create_deep_agent(
        name="research-radar",
        model=_ensure_auxiliary_chat_model(),
        system_prompt=_RADAR_SYSTEM_PROMPT,
        tools=[],
        skills=["/skills/"],
        backend=backend,
        middleware=_memory_worker_middleware(
            memory_dir=memory_dir,
            workspace_dir=workspace_dir,
            role=MemoryLifecycleRole.TURN,
            enable_profile_memory=memory_controls.profile_enabled,
            enable_observation_memory=memory_controls.observations_enabled,
            observation_writer=memory_controls.observation_writer,
        ),
        subagents=[],
    )
    return agent.with_config({"recursion_limit": 500})


def _radar_dir(memory_dir: Path | None = None) -> Path:
    from . import paths as _paths

    base = Path(memory_dir) if memory_dir else Path(_paths.MEMORIES_DIR)
    return base / RADAR_DIR_NAME


def _write_radar_update(
    update: RadarUpdate, memory_dir: Path | None = None
) -> tuple[Path, str]:
    """Write a radar update. Returns (path, timestamp_str)."""
    radar_dir = _radar_dir(memory_dir)
    radar_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y-%m-%d_%H%M%S")
    path = radar_dir / f"{ts}.md"

    content = f"# [Research Radar] {ts}\n\n"
    content += update.summary_text + "\n\n"
    content += "---\n\n"
    content += "## Papers\n\n"
    for p in update.papers:
        content += f"### {p.title}\n"
        content += f"**URL:** {p.url}\n\n"
        content += f"**Relevance:** {p.relevance}\n\n"
        content += f"**Connection:** {p.connection}\n\n"
    content += "## Trends\n\n"
    for t in update.trends:
        content += f"- {t}\n"
    content += "\n## Ideas\n\n"
    for idea in update.ideas:
        content += f"### {idea.title}\n"
        content += f"**Motivation:** {idea.motivation}\n\n"
        content += f"**Approach:** {idea.suggested_approach}\n\n"
    content += "## Suggested Actions\n\n"
    for a in update.suggested_actions:
        content += f"- {a}\n"

    path.write_text(content, encoding="utf-8")

    # Write structured JSON alongside for programmatic access
    json_path = radar_dir / f"{ts}.json"
    json_path.write_text(update.model_dump_json(indent=2), encoding="utf-8")

    # Write pending marker for notification pickup
    marker = radar_dir / RADAR_PENDING_MARKER
    marker.write_text(ts, encoding="utf-8")

    return path, ts


def list_radar_history(memory_dir: Path | None = None) -> list[dict]:
    """List past radar updates sorted by date (newest first).

    Returns list of dicts with keys: date, path, paper_count, summary.
    """
    radar_dir = _radar_dir(memory_dir)
    if not radar_dir.exists():
        return []

    entries = []
    for json_file in sorted(radar_dir.glob("????-??-??_??????.json"), reverse=True):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            entries.append(
                {
                    "date": json_file.stem,
                    "path": json_file.with_suffix(".md"),
                    "paper_count": len(data.get("papers", [])),
                    "summary": (data.get("summary_text", "") or "")[:80],
                }
            )
        except Exception:
            logger.debug("Skipping malformed radar file: %s", json_file)
    return entries


def load_radar_update(
    date_str: str, memory_dir: Path | None = None
) -> RadarUpdate | None:
    """Load a radar update by date string (YYYY-MM-DD_HHMMSS)."""
    radar_dir = _radar_dir(memory_dir)
    json_path = radar_dir / f"{date_str}.json"
    if not json_path.exists():
        return None
    try:
        return RadarUpdate.model_validate_json(json_path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed to parse radar update: %s", json_path)
        return None


def load_latest_radar(memory_dir: Path | None = None) -> tuple[str, RadarUpdate] | None:
    """Load the most recent radar update. Returns (date_str, update) or None."""
    history = list_radar_history(memory_dir)
    if not history:
        return None
    entry = history[0]
    update = load_radar_update(entry["date"], memory_dir)
    if update is None:
        return None
    return entry["date"], update


def check_radar_pending(memory_dir: Path | None = None) -> str | None:
    """Check if there's a pending radar notification. Returns date or None."""
    radar_dir = _radar_dir(memory_dir)
    marker = radar_dir / RADAR_PENDING_MARKER
    if marker.exists():
        return marker.read_text(encoding="utf-8").strip()
    return None


def clear_radar_pending(memory_dir: Path | None = None) -> None:
    """Clear the pending radar notification marker."""
    radar_dir = _radar_dir(memory_dir)
    marker = radar_dir / RADAR_PENDING_MARKER
    marker.unlink(missing_ok=True)


_scheduler_task: asyncio.Task | None = None
_scheduler_next_run: datetime | None = None


async def _radar_loop(schedule: str, tz_name: str) -> None:
    """Run radar scans on the configured cron schedule."""
    global _scheduler_next_run
    from croniter import croniter

    tz = UTC
    if tz_name:
        try:
            from zoneinfo import ZoneInfo

            tz = ZoneInfo(tz_name)
        except Exception:
            pass

    while True:
        now = datetime.now(tz=tz)
        try:
            cron = croniter(schedule, now)
            next_run = cron.get_next(datetime)
        except (ValueError, KeyError, TypeError) as exc:
            logger.error(
                "Radar scheduler: invalid cron expression %r: %s", schedule, exc
            )
            return

        # If we land exactly on (or just past) a boundary, skip to the one after
        if (next_run - now).total_seconds() < 1:
            next_run = cron.get_next(datetime)
        _scheduler_next_run = next_run
        delay = (next_run - now).total_seconds()
        logger.info("Radar scheduler: next run at %s (in %.0fs)", next_run, delay)
        await asyncio.sleep(delay)  # CancelledError here = shutdown
        try:
            logger.info("Radar scheduler: starting scan")
            await asyncio.shield(run_radar_now())
            logger.info("Radar scheduler: scan complete")
        except Exception:
            logger.exception("Radar scheduler: scan failed, will retry next cycle")


def start_radar_scheduler(config: EvoScientistConfig) -> None:
    """Start the in-process radar scheduler."""
    global _scheduler_task
    if _scheduler_task and not _scheduler_task.done():
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.info("Radar scheduler: no running event loop, skipping")
        return
    _scheduler_task = loop.create_task(
        _radar_loop(config.radar_schedule, config.radar_timezone)
    )
    _scheduler_task.add_done_callback(_on_scheduler_done)
    logger.info("Radar scheduler started: %s", config.radar_schedule)


def _on_scheduler_done(task: asyncio.Task) -> None:
    """Log if the scheduler task exits unexpectedly."""
    global _scheduler_task
    if task is _scheduler_task:
        _scheduler_task = None
    if task.cancelled():
        logger.info("Radar scheduler task was cancelled")
    elif exc := task.exception():
        logger.error("Radar scheduler task died: %s", exc, exc_info=exc)
    else:
        logger.warning("Radar scheduler task exited unexpectedly (no error)")


def stop_radar_scheduler() -> None:
    """Stop the in-process radar scheduler."""
    global _scheduler_task, _scheduler_next_run
    if _scheduler_task and not _scheduler_task.done():
        _scheduler_task.cancel()
    _scheduler_task = None
    _scheduler_next_run = None
    logger.info("Radar scheduler stopped")


def is_scheduler_running() -> bool:
    return _scheduler_task is not None and not _scheduler_task.done()


def get_scheduler_next_run() -> datetime | None:
    return _scheduler_next_run


def _compute_lookback_days(schedule: str) -> int:
    """Compute how many days to look back based on the cron interval + 1 day buffer."""
    from croniter import croniter

    try:
        now = datetime.now(UTC)
        cron = croniter(schedule, now)
        t1 = cron.get_next(datetime)
        t2 = cron.get_next(datetime)
        interval_days = (t2 - t1).total_seconds() / 86400
        return max(1, min(31, int(interval_days + 1)))
    except (ValueError, KeyError, TypeError):
        logger.warning(
            "Invalid cron expression %r, defaulting to 7 day lookback", schedule
        )
        return 7


async def run_radar_now() -> tuple[str, RadarUpdate] | None:
    """Run a radar scan in-process. Returns (timestamp, update) or None."""
    from .config.settings import get_effective_config

    cfg = get_effective_config()
    days = _compute_lookback_days(cfg.radar_schedule)
    graph = build_radar_graph()
    inp = {"messages": [("user", f"Run radar scan now. Search with days={days}.")]}
    result = await graph.ainvoke(inp)

    update = _extract_radar_update(result)
    if update:
        path, ts = _write_radar_update(update)
        logger.info("Radar scan complete, wrote %s", path)
        return ts, update
    return None


def _extract_radar_update(result: dict | None) -> RadarUpdate | None:
    """Extract RadarUpdate from the graph's output state."""
    if not result:
        return None
    # DeepAgents puts structured output in result["structured_response"]
    sr = result.get("structured_response")
    if isinstance(sr, RadarUpdate):
        return sr
    if isinstance(sr, dict):
        try:
            return RadarUpdate.model_validate(sr)
        except Exception:
            pass
    # Fallback: try parsing from message content
    messages = result.get("messages", [])
    for msg in reversed(messages):
        content = getattr(msg, "content", None) or (
            msg.get("content") if isinstance(msg, dict) else None
        )
        if not content:
            continue
        # Handle list-of-blocks content (some models return [{"type":"text","text":"..."}])
        if isinstance(content, list):
            content = "".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in content
            )
        if not isinstance(content, str):
            continue
        # Try to extract JSON: code fences anywhere in text, or raw JSON
        candidates: list[str] = []
        # 1. Extract fenced code blocks (```json ... ``` or ``` ... ```)
        for m in re.finditer(r"```(?:json)?\s*\n(.*?)```", content, re.DOTALL):
            candidates.append(m.group(1).strip())
        # 2. Try the whole content as-is
        candidates.append(content.strip())
        # 3. Find the outermost { ... } as last resort
        brace_start = content.find("{")
        brace_end = content.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            candidates.append(content[brace_start : brace_end + 1])
        for candidate in candidates:
            try:
                return RadarUpdate.model_validate_json(candidate)
            except Exception:
                pass
            try:
                data = json.loads(candidate)
                if isinstance(data, dict) and "papers" in data:
                    return RadarUpdate.model_validate(data)
            except Exception:
                pass
    logger.warning(
        "Could not extract RadarUpdate from graph output. "
        "Last message content type: %s, preview: %.200s",
        type(content).__name__ if messages else "no messages",
        str(content)[:200] if messages and content else "empty",
    )
    return None
