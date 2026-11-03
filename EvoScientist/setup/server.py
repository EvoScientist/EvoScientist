"""The ``server`` stage of ``EvoSci setup``: the background agent server after
an upgrade.

The background agent service (the server, and ccproxy with Anthropic OAuth)
outlives the sessions that use it, shared by every session in any project
until ``EvoSci service stop``. The first setup after the upgrade says so,
once. The stage also starts again the server ``EvoSci update`` stopped,
from the project it was started from.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .. import paths
from .protocol import Emitter, StageResult, make_event

logger = logging.getLogger(__name__)

LASTING_MESSAGE = (
    "The background agent service (the server, and ccproxy with Anthropic "
    "OAuth) now keeps running after your sessions end, so sessions in every "
    "project share it. It starts with your next session, from that session's "
    "project; `EvoSci service stop` stops it. The langgraph_dev_keepalive "
    "setting is gone."
)
# A record older than this is not from the update this setup follows.
RESTART_MAX_AGE_SECONDS = 60 * 60


def _told_marker() -> Path:
    from ..config.settings import get_config_dir

    return get_config_dir() / ".lasting-server-announced"


def _restart_record() -> Path:
    return paths.SERVER_HOME / "restart-after-update.json"


def _installed_version() -> str:
    from importlib.metadata import version

    return version("EvoScientist")


def remember_for_restart(workspace: str) -> None:
    """Record the project a lasting server was started from, before
    ``EvoSci update`` stops it, so the setup after the update starts it again.
    Best effort."""
    try:
        paths.SERVER_HOME.mkdir(parents=True, exist_ok=True)
        _restart_record().write_text(
            json.dumps(
                {
                    "workspace": workspace,
                    "stopped_at": time.time(),
                    "from_version": _installed_version(),
                }
            )
        )
    except OSError:
        logger.warning("Could not record the server to restart", exc_info=True)


def forget_restart() -> None:
    """Drop the record: the update stopped nothing after all."""
    _restart_record().unlink(missing_ok=True)


def _tell_once() -> bool:
    """Whether to say the server now outlives sessions: once, after an
    upgrade (a saved config; a new install has none)."""
    from ..config.settings import get_config_path

    marker = _told_marker()
    if marker.exists():
        return False
    upgrade = get_config_path().exists()
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
    except OSError:
        # Said every time otherwise; the release notes say it too.
        logger.warning("Could not record the server notice", exc_info=True)
        return False
    return upgrade


def _restart(emit: Emitter) -> tuple[str, bool] | None:
    """Start the server ``EvoSci update`` stopped: a line saying how it went
    and whether it started, or None when the update stopped none."""
    try:
        record = json.loads(_restart_record().read_text())
        workspace = record["workspace"]
        stopped_at = float(record["stopped_at"])
        from_version = record.get("from_version")
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        forget_restart()
        return None
    forget_restart()
    # An update that failed after it stopped the server left the version as
    # it was; the server then starts with the next session.
    if (
        not isinstance(workspace, str)
        or time.time() - stopped_at > RESTART_MAX_AGE_SECONDS
        or from_version == _installed_version()
    ):
        return None

    from ..config import get_effective_config
    from ..langgraph_dev.manager import serve_project
    from ..langgraph_dev.registration import ServerUnusableError, prepare_workdir

    emit(
        make_event("server", "running", message=f"Starting the server for {workspace}")
    )
    config = get_effective_config()
    try:
        served = serve_project(config, prepare_workdir(workspace))
    except (ServerUnusableError, RuntimeError, ValueError, OSError) as exc:
        return f"Could not start the background agent server again: {exc}", False
    if not served:
        return (
            "Could not start the background agent server again; it starts with "
            "your next session.",
            False,
        )
    return f"Started the background agent server again, from {workspace}.", True


def run_stage(emit: Emitter, _mirror: str) -> StageResult:
    lines = []
    told = _tell_once()
    if told:
        lines.append(LASTING_MESSAGE)
    restart = _restart(emit)
    if restart is not None:
        lines.append(restart[0])
    detail = {"announced": told, "restarted": restart is not None and restart[1]}
    if not lines:
        return StageResult("Nothing to change", detail, status="skipped")
    return StageResult(" ".join(lines), detail)
