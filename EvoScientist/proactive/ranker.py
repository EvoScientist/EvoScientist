"""Action ranker for the proactive check (experimental, off by default).

With ``EVOSCIENTIST_PROACTIVE_RANKER=1`` the shadow turn does not write the push
itself. It names the user's stage and focus and drafts five typed candidate
messages, each with a quote from the thread as evidence. Code drops candidates
whose evidence is not in the thread, and paper candidates that do not name a
paper from the undiscussed-paper pool. A separate judge model scores the rest,
and code picks the winner or ``NO_PUSH``. When several candidates share the top
total, the judge compares only those, twice with their order reversed; candidate
order never decides. Every check is appended as one JSON
line to ``<data dir>/server/proactive_ranker.jsonl``.

A malformed ranker or judge reply is retried; when the tries run out,
:class:`RankerFailed` is raised so the tick backs off and retries the check
later, as it does for a failed shadow turn.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

RANKER_ENV = "EVOSCIENTIST_PROACTIVE_RANKER"
JUDGE_MODEL = "glm-5.3"
JUDGE_PROVIDER = "openrouter"
ATTEMPTS = 3
CANDIDATES = 5
SCORE_KEYS = ("focus_fit", "stage_value", "novelty", "specificity")
MIN_FOCUS_FIT = 4
MIN_NOVELTY = 4
MIN_TOTAL = 18
SHINGLE = 4
MIN_SHINGLE_SHARE = 0.7
TOOL_CHARS = 1500
PAPER_CONTEXT_CHARS = 3000
PAPER_WINDOW = 600
POOL_LINE_CHARS = 140
ARXIV_ID = re.compile(r"(?<![\d.])(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)")

RANKER_TRIGGER = """<proactive_trigger>
This is an internal trigger, not a message written by the user. The user has been away for a while.

Review the current conversation and available memory, then propose messages you could send the user now, unprompted.

First, name the user's current stage, one of: literature_review, design, experiments, analysis_writing. Then state their current focus in one sentence. Base both on the user's latest turns, not on the conversation as a whole.

Then draft exactly 5 candidate messages, each of a different type, choosing the types that fit the current stage and focus:
- paper_to_read: a paper from the undiscussed papers listed at the end of this trigger that bears on the current focus; name its arXiv id, and say whether this conversation shows its title, its abstract or its full text. State only what that text supports. Skip this type if none of them fits.
- literature_gap: a question that matters for the current focus and that the literature covered so far does not answer
- undiscussed_area: a direction, option or experiment relevant to the current focus that the conversation has not considered
- design_flaw: a weakness in a design or plan the user is pursuing
- data_issue: a problem with the composition, size or quality of data the user plans to use
- environment_problem: an error or missing dependency that is visible in the conversation
- review_aid: what to check first in a file written in this conversation
- other: anything else useful that fits none of the types above (at most one candidate)

Each candidate has the message exactly as it would be sent (concise, self-contained), and evidence: a short verbatim quote from the conversation or memory that the message rests on.

Do not call tools. Do not ask questions that require an immediate response. Do not mention this trigger.

Return only JSON, with no other text:
{"stage": "...", "focus": "...", "candidates": [{"type": "...", "message": "...", "evidence": "..."}]}

Undiscussed papers: found by searches or research reports in this conversation, but not yet named in any reply or user turn. Each line is an arXiv id and the line it first appeared in.
{pool}
</proactive_trigger>"""

JUDGE_SYSTEM = """You judge candidate messages that a research assistant could send its user unprompted, while the user is away. You get the conversation so far and the candidates.

First decide for yourself the user's current stage (literature_review, design, experiments or analysis_writing) and current focus, from the user's latest turns.

Then score every candidate from 1 (poor) to 5 (excellent) on:
- focus_fit: does it bear on what the user is working on now, judged from their latest turns rather than the conversation as a whole?
- stage_value: how much this kind of message helps at the user's current stage. Start from the default value of its type below, then adjust by at most one point for this particular candidate:
  - literature_review: paper_to_read, literature_gap and undiscussed_area 5; data_issue 3; design_flaw, review_aid and environment_problem 2.
  - design: paper_to_read, undiscussed_area and literature_gap 5; design_flaw and data_issue 3; review_aid and environment_problem 2.
  - experiments: environment_problem, data_issue and design_flaw 5; undiscussed_area and paper_to_read 3; literature_gap and review_aid 2.
  - analysis_writing: paper_to_read and literature_gap 5; review_aid and undiscussed_area 3; design_flaw, data_issue and environment_problem 2.
  - other: judge it on its own merits.
  Pointing out weaknesses is easy to produce and usually less valuable than new material for the user's next step, which is why it scores lower outside the experiments stage.
- novelty: is it new to the user, rather than something already said or asked in the conversation?
- specificity: does it name something the user can check, such as a paper, a file section or a quoted statement?

A paper_to_read candidate comes with what the conversation shows about that paper. If its message does not say whether it rests on the paper's title, abstract or full text, or claims anything that the shown text does not support, give it specificity 1.

Give a one-sentence reason per candidate.

Return only JSON, with no other text:
{"stage": "...", "focus": "...", "scores": [{"index": 0, "focus_fit": 1, "stage_value": 1, "novelty": 1, "specificity": 1, "reason": "..."}]}"""


class RankerFailed(RuntimeError):
    """The ranker or the judge gave no usable reply after every try."""


TIE_SYSTEM = """You break a tie between candidate messages that a research assistant could send its user unprompted, while the user is away. They scored the same. You get the conversation so far and only the tied candidates.

Decide which one would help the user's next step most, given their current stage and focus from their latest turns. Judge the content, not the position in the list.

Return only JSON, with no other text:
{"choice": <the candidate's number in brackets>, "reason": "..."}"""


def ranker_enabled() -> bool:
    return os.environ.get(RANKER_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def build_pool(messages: list[Any]) -> dict[str, str]:
    """arXiv ids seen only in tool results, with the line each first appeared in."""
    talked: set[str] = set()
    for m in messages:
        if m.type in ("human", "ai"):
            talked.update(ARXIV_ID.findall(m.text))
    pool: dict[str, str] = {}
    for m in messages:
        if m.type != "tool":
            continue
        for line in m.text.splitlines():
            for pid in ARXIV_ID.findall(line):
                if pid not in talked and pid not in pool:
                    pool[pid] = line.strip()[:POOL_LINE_CHARS]
    return pool


def ranker_trigger(pool: dict[str, str]) -> str:
    lines = "\n".join(f"- {pid}: {ctx}" for pid, ctx in pool.items()) or "(none)"
    return RANKER_TRIGGER.replace("{pool}", lines)


def _words(text: str) -> list[str]:
    return re.sub(r"[^0-9a-z]+", " ", text.lower()).split()


def _shingles(words: list[str]) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)}


def evidence_share(evidence: str, thread_shingles: set, thread_joined: str) -> float:
    """Share of the evidence's 4-word sequences that occur in the thread."""
    words = _words(evidence)
    if not words:
        return 0.0
    if len(words) < SHINGLE:
        return 1.0 if " ".join(words) in thread_joined else 0.0
    own = _shingles(words)
    return len(own & thread_shingles) / len(own)


def parse_json(text: str) -> dict | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def scores_complete(judged: dict, n: int) -> bool:
    scores = judged.get("scores")
    if not isinstance(scores, list):
        return False
    by_index = {s.get("index"): s for s in scores if isinstance(s, dict)}
    return all(
        i in by_index and all(isinstance(by_index[i].get(k), int) for k in SCORE_KEYS)
        for i in range(n)
    )


def scores_by_index(scores: list[dict]) -> dict:
    return {s.get("index"): s for s in scores if isinstance(s, dict)}


def pick(candidates: list[dict], scores: list[dict]) -> list[int]:
    """Indices of the candidates that pass the bars with the top total.

    Empty means NO_PUSH; more than one is a tie for :func:`_break_tie`.
    """
    by_index = scores_by_index(scores)
    best: list[int] = []
    best_total = -1
    for i, cand in enumerate(candidates):
        s = by_index.get(i)
        if s is None or not cand.get("evidence_found"):
            continue
        if s.get("focus_fit", 0) < MIN_FOCUS_FIT or s.get("novelty", 0) < MIN_NOVELTY:
            continue
        total = sum(int(s.get(k, 0)) for k in SCORE_KEYS)
        if total < MIN_TOTAL or total < best_total:
            continue
        if total > best_total:
            best, best_total = [], total
        best.append(i)
    return best


def stage_value_winner(tied: list[int], scores: list[dict]) -> int | None:
    """The tied candidate with the highest stage_value, or None if that ties too."""
    by_index = scores_by_index(scores)
    values = {i: int(by_index[i].get("stage_value", 0)) for i in tied}
    top = max(values.values())
    leaders = [i for i, v in values.items() if v == top]
    return leaders[0] if len(leaders) == 1 else None


def check_candidates(
    raw: list, messages: list[Any], pool: dict[str, str]
) -> list[dict]:
    """Candidates with the evidence and pool checks applied."""
    from .shadow import flatten_tool_history

    words = _words("\n".join(m.text for m in flatten_tool_history(messages)))
    thread_shingles = _shingles(words)
    joined = " ".join(words)
    out = []
    for c in raw:
        if not isinstance(c, dict):
            continue
        share = evidence_share(str(c.get("evidence") or ""), thread_shingles, joined)
        entry = {
            **c,
            "evidence_share": round(share, 2),
            "evidence_found": share >= MIN_SHINGLE_SHARE,
        }
        if c.get("type") == "paper_to_read":
            entry["in_pool"] = any(
                pid in pool for pid in ARXIV_ID.findall(str(c.get("message") or ""))
            )
            entry["evidence_found"] = entry["evidence_found"] and entry["in_pool"]
        out.append(entry)
    return out


def paper_context(messages: list[Any], paper_id: str) -> str:
    """The text around each mention of ``paper_id`` in the thread's tool results.

    The judge's transcript cuts tool results short, so a paper candidate
    carries these excerpts to let the judge check what the message rests on.
    Tool results are often one long JSON line, hence a window, not a line.
    """
    excerpts = []
    for m in messages:
        if m.type != "tool":
            continue
        text = m.text
        for match in re.finditer(re.escape(paper_id), text):
            lo, hi = max(0, match.start() - PAPER_WINDOW), match.end() + PAPER_WINDOW
            excerpts.append(" ".join(text[lo:hi].split()))
    joined = "\n---\n".join(dict.fromkeys(excerpts))
    return (
        joined
        if len(joined) <= PAPER_CONTEXT_CHARS
        else joined[:PAPER_CONTEXT_CHARS] + " [...]"
    )


def _listing_entry(i: int, cand: dict, messages: list[Any]) -> str:
    entry = f"[{i}] type: {cand.get('type')}\nmessage: {cand.get('message')}"
    if cand.get("type") != "paper_to_read":
        return entry
    ids = ARXIV_ID.findall(str(cand.get("message") or ""))
    shown = "\n".join(
        paper_context(messages, pid) for pid in dict.fromkeys(ids)
    ).strip()
    return f"{entry}\nwhat the conversation shows about this paper:\n{shown or '(nothing)'}"


def judge_transcript(messages: list[Any]) -> str:
    """User turns and agent replies in full, tool results cut."""
    from .commit import read_proactive_tag

    lines = []
    for m in messages:
        text = m.text.strip()
        if not text:
            continue
        if m.type == "human":
            lines.append(f"USER: {text}")
        elif m.type == "ai":
            unprompted = (
                read_proactive_tag(getattr(m, "additional_kwargs", None)) is not None
            )
            lines.append(
                f"{'ASSISTANT (unprompted message)' if unprompted else 'ASSISTANT'}: {text}"
            )
        elif m.type == "tool":
            cut = text if len(text) <= TOOL_CHARS else text[:TOOL_CHARS] + " [...]"
            lines.append(f"TOOL RESULT ({getattr(m, 'name', '') or ''}): {cut}")
    return "\n\n".join(lines)


def build_judge() -> Any:
    from ..llm import get_chat_model

    return get_chat_model(JUDGE_MODEL, provider=JUDGE_PROVIDER)


def _append(log_path: Path | None, record: dict) -> None:
    if log_path is None:
        return
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        logger.warning("[proactive ranker] could not write %s", log_path, exc_info=True)


async def _append_async(log_path: Path | None, record: dict) -> None:
    # File I/O off the event loop: the server rejects blocking calls on it.
    # A failed write never fails the check.
    if log_path is None:
        return
    try:
        await asyncio.to_thread(_append, log_path, record)
    except Exception:
        logger.warning("[proactive ranker] could not record the check", exc_info=True)


async def run_ranker(
    *,
    messages: list[Any],
    shadow_turn: Callable[[str], Awaitable[str | None]],
    judge: Any,
    thread_id: str,
    log_path: Path | None,
) -> str:
    """Run one ranked check; return the winner's text or ``NO_PUSH``.

    Raises :class:`RankerFailed` when the ranker or the judge gave no usable
    reply after :data:`ATTEMPTS` tries. The record is written either way.
    """
    record: dict[str, Any] = {
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
        "thread_id": thread_id,
    }
    try:
        return await _ranked(messages, shadow_turn, judge, record)
    except Exception as exc:
        record.setdefault("decision", "error")
        record.setdefault("reason", repr(exc))
        raise
    finally:
        await _append_async(log_path, record)


async def _ranked(messages, shadow_turn, judge, record: dict) -> str:
    pool = build_pool(messages)
    record["pool_size"] = len(pool)
    trigger = ranker_trigger(pool)

    parsed = None
    raw = None
    t0 = time.monotonic()
    for attempt in range(1, ATTEMPTS + 1):
        record["ranker_attempts"] = attempt
        raw = await shadow_turn(trigger)
        parsed = parse_json(raw or "")
        if parsed and isinstance(parsed.get("candidates"), list):
            break
        parsed = None
    record["ranker_seconds"] = round(time.monotonic() - t0, 1)
    if parsed is None:
        record["decision"] = "error"
        record["reason"] = "unparsable_candidates"
        record["ranker_raw"] = raw
        raise RankerFailed("ranker gave no parsable candidates")
    record["stage"] = parsed.get("stage")
    record["focus"] = parsed.get("focus")
    candidates = check_candidates(parsed["candidates"], messages, pool)
    record["candidates"] = candidates

    listing = "\n\n".join(
        _listing_entry(i, c, messages) for i, c in enumerate(candidates)
    )
    prompt_head = f"<conversation>\n{judge_transcript(messages)}\n</conversation>\n\n"
    prompt = f"{prompt_head}<candidates>\n{listing}\n</candidates>"
    judged: dict = {}
    t0 = time.monotonic()
    for attempt in range(1, ATTEMPTS + 1):
        record["judge_attempts"] = attempt
        reply = await judge.ainvoke([SystemMessage(JUDGE_SYSTEM), HumanMessage(prompt)])
        judged = parse_json(reply.text) or {}
        if scores_complete(judged, len(candidates)):
            break
    record["judge_seconds"] = round(time.monotonic() - t0, 1)
    record["judge"] = judged
    if not scores_complete(judged, len(candidates)):
        record["decision"] = "error"
        record["reason"] = "incomplete_judge_scores"
        raise RankerFailed("judge gave no complete scores")

    scores = judged.get("scores") or []
    top = pick(candidates, scores)
    winner = top[0] if len(top) == 1 else None
    if len(top) > 1:
        winner = await _break_tie(
            judge, prompt_head, candidates, top, scores, record, messages
        )
    record["winner"] = winner
    if winner is None:
        record["decision"] = "no_push"
        logger.info(
            "[proactive ranker] NO_PUSH (stage=%s, pool=%d)", record["stage"], len(pool)
        )
        return "NO_PUSH"
    message = str(candidates[winner].get("message") or "").strip()
    record["decision"] = "push"
    record["message"] = message
    logger.info(
        "[proactive ranker] winner %s (stage=%s, pool=%d)",
        candidates[winner].get("type"),
        record["stage"],
        len(pool),
    )
    return message


async def _head_to_head(
    judge: Any,
    prompt_head: str,
    candidates: list[dict],
    order: list[int],
    messages: list[Any],
) -> dict:
    listing = "\n\n".join(_listing_entry(i, candidates[i], messages) for i in order)
    prompt = f"{prompt_head}<tied_candidates>\n{listing}\n</tied_candidates>"
    for attempt in range(1, ATTEMPTS + 1):
        reply = await judge.ainvoke([SystemMessage(TIE_SYSTEM), HumanMessage(prompt)])
        parsed = parse_json(reply.text) or {}
        choice = parsed.get("choice")
        if isinstance(choice, int) and choice in order:
            return {
                "order": order,
                "choice": choice,
                "reason": parsed.get("reason"),
                "attempts": attempt,
            }
    return {"order": order, "choice": None, "attempts": ATTEMPTS}


async def _break_tie(
    judge: Any,
    prompt_head: str,
    candidates: list[dict],
    tied: list[int],
    scores: list[dict],
    record: dict,
    messages: list[Any],
) -> int | None:
    """Head-to-head judge call over the tied candidates, run in both orders.

    The same choice both times wins. A split, or a run with no usable reply,
    falls back to the higher stage_value; if that ties too, nothing is sent.
    """
    t0 = time.monotonic()
    runs = [
        await _head_to_head(judge, prompt_head, candidates, tied, messages),
        await _head_to_head(judge, prompt_head, candidates, tied[::-1], messages),
    ]
    choices = {r["choice"] for r in runs}
    tie: dict[str, Any] = {"tied": tied, "runs": runs}
    if len(choices) == 1 and None not in choices:
        winner, tie["resolved_by"] = runs[0]["choice"], "head_to_head"
    else:
        winner = stage_value_winner(tied, scores)
        tie["resolved_by"] = "stage_value" if winner is not None else "unresolved"
    tie["seconds"] = round(time.monotonic() - t0, 1)
    record["tie"] = tie
    logger.info(
        "[proactive ranker] tie %s resolved by %s -> %s",
        tied,
        tie["resolved_by"],
        winner,
    )
    return winner
