"""``EvoSci setup``: prepare what EvoScientist needs beyond the Python package.

Each :class:`Stage` installs or checks one dependency and reports progress
through the event protocol in :mod:`.protocol`.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass
from typing import Any

from . import git, node, research_env, webui
from .protocol import PROTOCOL, Emitter, StageError, StageResult, make_event

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Stage:
    """One setup stage.

    ``platforms`` lists the ``sys.platform`` values the stage applies to
    (None = every platform). ``run(emit, mirror)`` returns a
    :class:`StageResult` (``done`` or ``skipped``) or raises
    :class:`StageError`. It emits only ``running`` events itself; the runner
    emits the one terminal event.
    """

    id: str
    title: str
    platforms: frozenset[str] | None
    run: Callable[[Emitter, str], StageResult]

    def applies(self) -> bool:
        return self.platforms is None or sys.platform in self.platforms


STAGES: tuple[Stage, ...] = (
    Stage("node", "Node.js", None, node.run_stage),
    Stage("git", "Git for Windows", frozenset({"win32"}), git.run_stage),
    Stage("research-env", "Python research environment", None, research_env.run_stage),
    Stage("webui", "WebUI", None, webui.run_stage),
)


def get_stage(stage_id: str) -> Stage | None:
    return next((s for s in STAGES if s.id == stage_id), None)


def manifest() -> dict[str, Any]:
    """The stages that apply to this platform, in run order."""
    return {
        "protocol": PROTOCOL,
        "stages": [{"id": s.id, "title": s.title} for s in STAGES if s.applies()],
    }


def run_stages(
    stages: Iterable[Stage],
    emit: Emitter,
    mirror: str,
    skip: Collection[str] = (),
) -> int:
    """Run every stage in order; returns 1 if any failed, else 0.

    A failed stage (e.g. a blocked Node download) does not stop the rest;
    each still ends with its own terminal event. The one dependency, ``webui``
    on ``node``, needs no ordering here: the webui stage asks ``ensure_node()``
    itself, which raises the Node failure again within the same process.

    A stage in ``skip`` (``EvoSci setup --skip``) ends with ``skipped`` and the
    reason ``skip_option`` in its detail, e.g. the desktop app, which bundles
    its own UI, skips ``webui``.
    """
    failed = False
    for stage in stages:
        if stage.id in skip:
            emit(
                make_event(
                    stage.id,
                    "skipped",
                    message="Skipped (--skip)",
                    detail={"reason": "skip_option"},
                )
            )
            continue
        if not stage.applies():
            emit(
                make_event(stage.id, "skipped", message=f"Not needed on {sys.platform}")
            )
            continue
        try:
            result = stage.run(emit, mirror)
        except StageError as exc:
            emit(make_event(stage.id, "error", message=exc.message, code=exc.code))
            failed = True
            continue
        except Exception as exc:
            logger.exception(f"Setup stage {stage.id} failed unexpectedly")
            emit(
                make_event(
                    stage.id,
                    "error",
                    message=f"Unexpected error: {exc!r}",
                    code="install_failed",
                )
            )
            failed = True
            continue
        emit(
            make_event(
                stage.id, result.status, message=result.message, detail=result.detail
            )
        )
    return 1 if failed else 0


__all__ = [
    "PROTOCOL",
    "STAGES",
    "Stage",
    "StageError",
    "StageResult",
    "get_stage",
    "manifest",
    "run_stages",
]
