"""``EvoSci setup``: prepare what EvoScientist needs beyond the Python package.

Each :class:`Stage` installs or checks one dependency and reports progress
through the event protocol in :mod:`.protocol`.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from . import node
from .protocol import PROTOCOL, Emitter, StageError, make_event


@dataclass(frozen=True)
class Stage:
    """One setup stage.

    ``platforms`` lists the ``sys.platform`` values the stage applies to
    (None = every platform). ``run(emit, mirror)`` returns the ``done``
    message and ``detail`` object, or raises :class:`StageError`.
    """

    id: str
    title: str
    platforms: frozenset[str] | None
    run: Callable[[Emitter, str], tuple[str, dict[str, Any]]]

    def applies(self) -> bool:
        return self.platforms is None or sys.platform in self.platforms


STAGES: tuple[Stage, ...] = (Stage("node", "Node.js", None, node.run_stage),)


def get_stage(stage_id: str) -> Stage | None:
    return next((s for s in STAGES if s.id == stage_id), None)


def manifest() -> dict[str, Any]:
    """The stages that apply to this platform, in run order."""
    return {
        "protocol": PROTOCOL,
        "stages": [{"id": s.id, "title": s.title} for s in STAGES if s.applies()],
    }


def run_stages(stages: Iterable[Stage], emit: Emitter, mirror: str) -> int:
    """Run ``stages`` in order; stop at the first error. Returns an exit code."""
    for stage in stages:
        if not stage.applies():
            emit(
                make_event(stage.id, "skipped", message=f"Not needed on {sys.platform}")
            )
            continue
        try:
            message, detail = stage.run(emit, mirror)
        except StageError as exc:
            emit(make_event(stage.id, "error", message=exc.message, code=exc.code))
            return 1
        emit(make_event(stage.id, "done", message=message, detail=detail))
    return 0


__all__ = [
    "PROTOCOL",
    "STAGES",
    "Stage",
    "StageError",
    "get_stage",
    "manifest",
    "run_stages",
]
