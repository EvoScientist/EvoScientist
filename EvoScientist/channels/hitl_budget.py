"""Shared HITL resume-loop budgets (issue #469).

Auto-resolved rounds (session grant, config allow-list, ``auto_approve``)
do not count toward the human budget. The total cap still stops a
pathological stream, including auto-resolved rounds. Callers apply this
*after* those fast paths and *before* prompting a human, so a session
grant picked on the last human round still auto-resolves the next pending.
"""

from __future__ import annotations

MAX_HUMAN_HITL_ROUNDS = 50
MAX_HITL_TOTAL_ROUNDS = 1000


def hitl_budget_stop(
    *,
    human_rounds: int,
    total_rounds: int,
    needs_human: bool,
) -> bool:
    """Return whether this pending must be closed instead of resumed.

    ``needs_human`` is False when the pending would auto-resolve without a
    prompt. The human budget does not apply then. The total cap applies to
    every pending.
    """
    if total_rounds >= MAX_HITL_TOTAL_ROUNDS:
        return True
    return needs_human and human_rounds >= MAX_HUMAN_HITL_ROUNDS


def hitl_completed_round_cap_reached(completed_rounds: int) -> bool:
    """True when the HITL resume loop must stop before starting another stream.

    ``completed_rounds`` is the number of streams already finished — the
    loop counter *before* it is incremented for the next iteration. Zero
    never stops, so the first iteration is unchanged. Pause branches still
    call ``hitl_budget_stop`` before they prompt; this is the loop-level
    bound for a round that stored a pending without building a resume.
    """
    return completed_rounds >= MAX_HITL_TOTAL_ROUNDS
