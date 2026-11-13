"""A client's side of the project registry: the folder check every launcher
runs first (:func:`prepare_workdir`), and the registration of its project
with the langgraph dev server it uses (:func:`register_folder`).

The server serves a project once it is registered and while its folder
exists. A client registers its project every time it starts or reuses a
server; each registration restores the project's threads.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from ..paths import Workspace
from .boundary import is_loopback_host


def running_server_url() -> str:
    """The URL of the running server: its record when one exists, else the
    configured ``langgraph_dev_port``."""
    from ..config import get_config_value
    from . import manager

    record = manager.read_record()
    if record is not None:
        return record.url
    return f"http://127.0.0.1:{get_config_value('langgraph_dev_port')}"

# Seconds a registration may take: it restores the project's threads, which
# reads every checkpoint in sessions.db.
REGISTRATION_TIMEOUT = 120.0
# A move rewrites a project's threads: seconds, but not bounded.
MOVE_TIMEOUT = 30 * 60.0
# Seconds the read of the server's instance may take.
INSTANCE_TIMEOUT = 10.0
# Seconds each read of ``EvoSci server status`` may take.
STATUS_TIMEOUT = 3.0


class _Refusal(RuntimeError):
    """A refusal a launcher reports: ``code`` names it for programs
    (``deploy --json``), ``fields`` holds its facts."""

    code: str

    def __init__(self, message: str, *, fields: dict[str, str] | None = None):
        super().__init__(message)
        self.fields = dict(fields or {})


class ServerUnusableError(_Refusal):
    """This client cannot use the running server for its project: the client
    stops, or keeps the session it has."""

    code = "server_unusable"


class OlderServerError(ServerUnusableError):
    """The running server predates the project registry."""

    code = "server_outdated"

    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            message
            or "The running server is from an older EvoScientist. Stop it with "
            "`EvoSci server stop`, then start again."
        )


class ProjectFolderMissingError(ServerUnusableError):
    """A folder the registry knows as a project is gone."""

    code = "folder_missing"

    def __init__(self, root: Path) -> None:
        super().__init__(
            f"{root} used to be an EvoScientist project, and its threads and "
            "scheduled tasks are still stored. Restore the folder to keep "
            "working in it, or create it empty to start again there.",
            fields={"workspace": str(root)},
        )


class RegistrationError(_Refusal):
    """The server did not register the project. The client goes on without
    it, as when the server does not start."""

    code = "registration_failed"


class OtherServerError(RegistrationError):
    """The running server does not hold this data folder's local secret."""


def is_known_project(workspace: Workspace) -> bool:
    """Whether the registry (``sessions.db``) knows *workspace*, registered or
    not. Read only; a database or table not created yet knows nothing."""
    from .. import paths
    from ..sessions import get_db_path

    # ``get_db_path`` creates the data folder; a missing one knows nothing.
    if not paths.DATA_DIR.is_dir():
        return False
    try:
        conn = sqlite3.connect(
            f"{get_db_path().as_uri()}?mode=ro", uri=True, timeout=30.0
        )
    except sqlite3.Error:
        return False
    try:
        row = conn.execute(
            "SELECT 1 FROM workspaces WHERE workspace_dir = ?", (workspace.key,)
        ).fetchone()
    except sqlite3.Error:
        return False
    finally:
        conn.close()
    return row is not None


def refuse_missing_project(path: str | Path) -> None:
    """Raise :class:`ProjectFolderMissingError` when *path* is missing but
    the registry knows it as a project: recreating it empty would hide the
    project's threads and scheduled tasks behind a new, empty project."""
    folder = Path(path)
    if folder.is_dir():
        return
    workspace = Workspace(folder)
    if is_known_project(workspace):
        raise ProjectFolderMissingError(workspace.root)


def prepare_workdir(path: str | Path) -> Path:
    """The folder a launcher works in, created when it is new; see
    :func:`refuse_missing_project`."""
    folder = Path(path)
    refuse_missing_project(folder)
    os.makedirs(folder, exist_ok=True)
    return folder


def _other_server(url: str) -> str:
    from .. import paths

    return (
        f"The EvoScientist server at {url} does not hold the local secret in "
        f"{paths.SERVER_HOME}: it uses another data folder "
        "(EVOSCIENTIST_DATA_DIR), runs for another user, or could not load its "
        "secret (see its log)."
    )


def _get(url: str, path: str, *, timeout: float) -> tuple[httpx.Response, object]:
    """GET *path* from the server at *url*, without the local secret: the
    response and its JSON body (``None`` when it has none).
    ``RegistrationError`` when the request cannot be made."""
    from .sdk import LANGGRAPH_DEV_AUTH_HEADERS

    try:
        response = httpx.get(
            f"{url}{path}",
            headers=LANGGRAPH_DEV_AUTH_HEADERS,
            timeout=timeout,
            # Loopback: a system proxy must not intercept it.
            trust_env=False,
        )
    except httpx.HTTPError as exc:
        raise RegistrationError(
            f"Could not reach the EvoScientist server at {url}: {exc}"
        ) from exc
    try:
        body = response.json()
    except ValueError:
        body = None
    return response, body


def server_instance(url: str, *, timeout: float = INSTANCE_TIMEOUT) -> str | None:
    """The instance the server at *url* names (``GET /api/instance``)."""
    response, body = _get(url, "/api/instance", timeout=timeout)
    if response.status_code == 404:
        raise OlderServerError()
    if response.status_code != 200:
        raise RegistrationError(
            f"The EvoScientist server at {url} did not answer GET /api/instance "
            f"(HTTP {response.status_code})."
        )
    return body.get("instance") if isinstance(body, dict) else None


def _require_own_server(url: str, *, timeout: float = INSTANCE_TIMEOUT) -> str:
    """The local secret, once the server at *url* answers with its
    ``instance``. ``OlderServerError`` for a server too old to have an
    instance; ``RegistrationError`` when the secret cannot be read or the
    instance cannot be; ``OtherServerError`` for a server that holds another
    secret."""
    from .. import paths
    from ..local_secret import instance_of, read_local_secret

    # First: a server too old to have created the secret has no instance.
    instance = server_instance(url, timeout=timeout)
    secret = read_local_secret()
    if secret is None:
        raise RegistrationError(
            f"Cannot read the server's local secret in {paths.SERVER_HOME}. "
            "Is EVOSCIENTIST_DATA_DIR the same as the server's, on a file "
            "system that keeps permissions?"
        )
    if instance != instance_of(secret):
        raise OtherServerError(_other_server(url))
    return secret


def _post_with_secret(
    url: str,
    path: str,
    body: dict,
    *,
    what: str,
    timeout: float = REGISTRATION_TIMEOUT,
) -> httpx.Response:
    """POST *body* to the server at *url* with the local secret, once it
    answers with the same ``instance``: a different server on the port (another
    user's, another data folder's) does not get it. The instance is public, so
    this catches a mix-up, not a process that copied it. ``OlderServerError``
    for a server too old to have an instance; ``RegistrationError`` when the
    request cannot be made."""
    from ..local_secret import HEADER
    from .sdk import LANGGRAPH_DEV_AUTH_HEADERS

    secret = _require_own_server(url)
    try:
        return httpx.post(
            f"{url}{path}",
            json=body,
            headers={**LANGGRAPH_DEV_AUTH_HEADERS, HEADER: secret},
            timeout=timeout,
            # Loopback: a system proxy must not intercept it.
            trust_env=False,
        )
    except httpx.HTTPError as exc:
        raise RegistrationError(
            f"Could not {what} the EvoScientist server at {url}: {exc}"
        ) from exc


def served_projects(url: str) -> list[dict[str, object]]:
    """The projects the server at *url* serves (``GET /api/workspaces``),
    each ``{workspace_dir, name, missing}``, once it answers with this data
    folder's instance. The local secret is never sent.

    ``OlderServerError`` for a server too old to have an instance,
    ``OtherServerError`` for another data folder's server, and
    ``RegistrationError`` when the local secret or a read fails, or a read
    times out.
    """
    url = url.rstrip("/")
    _require_own_server(url, timeout=STATUS_TIMEOUT)
    response, body = _get(url, "/api/workspaces", timeout=STATUS_TIMEOUT)
    projects = (
        body.get("workspaces")
        if response.status_code == 200 and isinstance(body, dict)
        else None
    )
    if not isinstance(projects, list):
        raise RegistrationError(
            f"The EvoScientist server at {url} did not list its projects "
            f"(HTTP {response.status_code})."
        )
    return [
        project
        for project in projects
        if isinstance(project, dict) and isinstance(project.get("workspace_dir"), str)
    ]


def ensure_ccproxy(*, url: str) -> str | None:
    """Ask the server at *url* to start its ccproxy when it is not running;
    why it does not serve, or None."""
    from ..ccproxy_manager import START_BUDGET_SECONDS

    url = url.rstrip("/")
    try:
        # A start can take ccproxy's whole start budget.
        response = _post_with_secret(
            url,
            "/api/ccproxy/ensure",
            {},
            what="reach",
            timeout=START_BUDGET_SECONDS + 30,
        )
    except RegistrationError as exc:
        return str(exc)
    try:
        body = response.json()
    except ValueError:
        body = {}
    if not isinstance(body, dict):
        body = {}
    if response.status_code == 200 and body.get("serving") is True:
        return None
    if body.get("error") == "local_secret_required":
        return _other_server(url)
    return str(
        body.get("detail")
        or f"The EvoScientist server at {url} did not start its ccproxy "
        f"(HTTP {response.status_code})."
    )


def running_server_url(fallback: str) -> str:
    """The URL of the running server: the one its record names when there is
    a record, else *fallback* (the configured ``langgraph_dev_port``)."""
    from .manager import manager

    try:
        record = manager.read_record()
    except Exception:
        record = None
    url = getattr(record, "url", None) if record is not None else None
    return str(url or fallback)


def reconcile_autoskills(*, url: str) -> list[str]:
    """Ask the server at *url* to reconcile every served project's AutoSkills
    cron with its configuration (after ``/autoskills`` changed it); why each
    project that failed did. ``RegistrationError`` (or ``OlderServerError``)
    when it could not ask."""
    url = running_server_url(url).rstrip("/")
    response = _post_with_secret(url, "/api/autoskills/reconcile", {}, what="reach")
    if response.status_code == 404:
        raise OlderServerError()
    if response.status_code != 200:
        try:
            body = response.json()
        except ValueError:
            body = {}
        error = body.get("error") if isinstance(body, dict) else None
        if error == "local_secret_required":
            raise RegistrationError(_other_server(url))
        detail = body.get("detail") if isinstance(body, dict) else None
        reason = ": ".join(str(part) for part in (error, detail) if part)
        raise RegistrationError(
            f"The EvoScientist server at {url} did not reconcile AutoSkills "
            f"(HTTP {response.status_code}{f', {reason}' if reason else ''})."
        )
    try:
        failed = response.json().get("failed") or []
    except (ValueError, AttributeError):
        failed = []
    return [str(reason) for reason in failed]


_EXPOSED = (
    "has no auth: anyone who reaches it can read every project it serves and "
    "run the agent, with shell access, in any of them."
)


@dataclass(frozen=True)
class ServerExposure:
    """How a server can be reached from other machines: the address it
    listens on when that is not loopback, and whether it has a public
    tunnel."""

    public_host: str | None = None
    tunnel: bool = False

    @classmethod
    def of(cls, host: object, tunnel: object) -> ServerExposure:
        """The exposure of a server bound to *host*; anything but a loopback
        address counts as public, an empty one (every interface) too."""
        if not isinstance(host, str) or is_loopback_host(host):
            return cls(None, tunnel is True)
        return cls(host.strip() or "every interface", tunnel is True)


def exposure_warnings(exposure: ServerExposure | None) -> list[tuple[str, str]]:
    """The warnings for a server with *exposure* (``None``: not known to be
    exposed), as ``(label, message)``: one wording for every surface."""
    if exposure is None:
        return []
    warnings = []
    if exposure.public_host is not None:
        warnings.append(
            (
                "PUBLIC BIND",
                f"The agent server listens on {exposure.public_host} and "
                f"{_EXPOSED} Trusted networks only.",
            )
        )
    if exposure.tunnel:
        warnings.append(
            (
                "PUBLIC TUNNEL",
                "The agent server is reachable through a public tunnel and "
                f"{_EXPOSED} Share its URL only with people you trust.",
            )
        )
    return warnings


def exposure_messages(exposure: ServerExposure | None) -> list[str]:
    """The messages of :func:`exposure_warnings`, for a surface without
    labels (a list of warnings, a log)."""
    return [message for _, message in exposure_warnings(exposure)]


def apply_autoskills_settings(config: object) -> str | None:
    """Ask the server *config* names, when it runs, to reconcile every
    project's AutoSkills cron with the settings just saved; a note when it
    could not. Without a running server each project follows at its next
    setup."""
    from .manager import is_langgraph_dev_running
    from .sdk import configured_langgraph_dev_url

    url = configured_langgraph_dev_url(config)
    if not is_langgraph_dev_running(base_url=url):
        return None
    try:
        failed = reconcile_autoskills(url=url)
    except (RegistrationError, ServerUnusableError) as exc:
        return (
            "Could not ask the background agent server to apply it; each "
            f"project follows when the server next sets it up: {exc}"
        )
    if failed:
        return "AutoSkills was not rescheduled for: " + "; ".join(failed)
    return None


def move_folder(old: Workspace, new: Workspace, *, url: str) -> dict[str, Any]:
    """Ask the server at *url* to move the registered project *old* to the
    folder *new* (``POST /api/workspaces/move``); what it reports moving.

    Raises :class:`OlderServerError` for a server without the move, and
    :class:`RegistrationError` with the server's reason when the move is
    refused or fails.
    """
    url = url.rstrip("/")
    # First, and unwrapped: a server that does not answer, or is not this
    # data folder's, is moving nothing.
    _require_own_server(url)
    try:
        response = _post_with_secret(
            url,
            "/api/workspaces/move",
            {"old": old.key, "new": new.key},
            what=f"move {old.root} with",
            timeout=MOVE_TIMEOUT,
        )
    except RegistrationError as exc:
        # The request was sent, then cut or timed out: the server may go on.
        if not isinstance(exc.__cause__, httpx.HTTPError):
            raise
        raise RegistrationError(
            f"{exc} The server may still be moving it: run the same command again "
            "to see it finish."
        ) from exc
    try:
        body = response.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        body = {}
    if response.status_code == 200:
        return body
    if response.status_code == 404 and "error" not in body:
        raise OlderServerError()
    error = body.get("error")
    if error == "local_secret_required":
        raise RegistrationError(_other_server(url))
    detail = body.get("detail")
    if response.status_code == 409 and detail:
        raise RegistrationError(str(detail))
    if error == "move_failed" and body.get("pending") is not False:
        raise RegistrationError(
            f"The EvoScientist server at {url} could not finish moving "
            f"{old.root}{f' ({detail})' if detail else ''}. Fix the cause and "
            "run the same command again to finish it."
        )
    reason = ": ".join(str(part) for part in (error, detail) if part)
    raise RegistrationError(
        f"The EvoScientist server at {url} did not move {old.root} "
        f"(HTTP {response.status_code}{f', {reason}' if reason else ''})."
    )


def register_folder(workspace: Workspace, *, url: str) -> ServerExposure:
    """Register *workspace* (a project root, never a run folder) with the
    langgraph dev server at *url*; how that server, which now serves it, can
    be reached from other machines.

    The local secret goes only to a server that answers with the same
    ``instance``, so a different server on the port does not get it (see
    :func:`_post_with_secret`).

    Raises :class:`OlderServerError` for a server without the registry,
    :class:`ProjectFolderMissingError` when the server finds the folder
    missing, and :class:`RegistrationError` for anything else that kept the
    server from registering it.
    """
    url = url.rstrip("/")
    response = _post_with_secret(
        url,
        "/api/workspaces",
        {"workspace_dir": workspace.key},
        what=f"register {workspace.root} with",
    )
    try:
        body = response.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        body = {}
    if response.status_code == 200:
        return ServerExposure.of(body.get("host"), body.get("tunnel"))
    error = body.get("error")
    if error == "folder_missing":
        if not workspace.root.is_dir() and is_known_project(workspace):
            raise ProjectFolderMissingError(workspace.root)
        raise RegistrationError(
            f"The EvoScientist server at {url} cannot see the folder {workspace.root}."
        )
    if error == "local_secret_required":
        raise RegistrationError(_other_server(url))
    detail = body.get("detail")
    reason = ": ".join(str(part) for part in (error, detail) if part)
    raise RegistrationError(
        f"The EvoScientist server at {url} did not register {workspace.root} "
        f"(HTTP {response.status_code}{f', {reason}' if reason else ''})."
    )
