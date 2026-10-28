"""Custom HTTP routes mounted alongside the langgraph dev server.

The langgraph-api host supports a top-level ``http`` key in
``langgraph.json`` that names an ASGI app to mount on the same
process as the graph. It hosts the project registry and registration
(``/api/workspaces``), the settings and local-secret routes, the
AutoSkills reconcile, and the model registry the WebUI's ``/model``
picker needs.

Why this lives here and not as a separate sidecar: the WebUI talks to
``EvoSci deploy``'s langgraph endpoint anyway, so one server keeps the
WebUI's fetch logic simple — no extra port to configure.

The app's middleware, :class:`~EvoScientist.langgraph_dev.boundary.BrowserBoundary`,
wraps every route of the server, langgraph-api's included: langgraph-api runs a
custom app's middleware outside its own.

Why Starlette and not FastAPI: ``langgraph_api`` already depends on
Starlette; adding FastAPI would pull in pydantic v1-vs-v2 reconciliation
the deploy doesn't need. The routes here have no input models, just JSON
bodies, so the lower-level surface is sufficient.

Lightweight by design — module-level imports stick to ``config``,
``llm.models`` (registry only; no chat-model construction), and
Starlette itself. Nothing on this surface should pull the agent into
memory.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from typing import TYPE_CHECKING

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from EvoScientist.config import get_effective_config
from EvoScientist.langgraph_dev import server_home, workspace_setup
from EvoScientist.langgraph_dev.boundary import (
    BOUNDARY_ENV,
    BrowserBoundary,
    Trust,
    tunnel_host,
)
from EvoScientist.llm.models import list_model_picker_entries
from EvoScientist.local_secret import (
    instance_id,
    load_local_secret,
    require_local_secret,
)
from EvoScientist.memory.autoskills.schedule import AUTOSKILL_SETTINGS

if TYPE_CHECKING:
    from EvoScientist.paths import Workspace

logger = logging.getLogger(__name__)


async def get_models(_request: Request) -> JSONResponse:
    """Return the model registry as ``{entries, default}``.

    ``entries`` preserves the registry order so the WebUI picker can
    rank providers per short name the same way the backend would.
    Mirrors the TUI ``/model`` picker by appending locally-pulled
    Ollama models when ``ollama_base_url`` is configured — same
    ``discover_ollama_models()`` call, same 1.5-s timeout, same
    fail-soft semantics (the probe returns ``[]`` on any error, never
    raises). The TUI's "Custom Ollama model…" sentinel is intentionally
    omitted — that's a widget-specific input affordance, not part of
    the registry surface.

    ``default`` reflects the deployment's currently-configured fallback
    (``config.yaml``'s ``model`` / ``provider`` — what ``/model reset``
    would land on). Returned even when the configured pair isn't in
    the registry, so the picker can still label it.

    Uses ``get_effective_config()`` (not ``load_config()``) so env-var
    overrides like ``OLLAMA_BASE_URL`` from ``_ENV_MAPPINGS`` are
    honored — matching the deploy's actual model-building behavior.
    Offloaded to a thread because ``get_effective_config()`` reads the
    config file, blocking I/O that langgraph-dev's ``blockbuster``
    middleware refuses on the async event loop (would surface as a 500).
    """
    cfg = await asyncio.to_thread(get_effective_config)
    entries = [
        {"name": name, "model_id": model_id, "provider": provider}
        for name, model_id, provider in await list_model_picker_entries(
            getattr(cfg, "ollama_base_url", None),
            include_custom_ollama=False,
        )
    ]
    return JSONResponse(
        {
            "entries": entries,
            "default": {"name": cfg.model, "provider": cfg.provider},
        }
    )


async def get_teams(request: Request) -> JSONResponse:
    """Return installed expert skills as ``{teams: [...]}`` for the WebUI gallery.

    A "team" in the WebUI vocabulary is an installed expert skill — a skill
    directory carrying a sibling ``EXPERT.md`` (or, on the deprecated path,
    ``type: expert`` SKILL.md frontmatter). The response is a curated,
    gallery-safe projection: name + description, plus optional ``byline`` /
    ``capability_tags`` / ``avatar_hint`` when the skill populates them.

    Cards for experts on the current contract carry name + description only:
    the decoration fields were actor metadata in SKILL.md frontmatter, which
    that contract removes rather than relocates (``EXPERT.md`` has no
    frontmatter to hold them). The omit-when-unpopulated projection below is
    what makes those cards degrade rather than break; restoring richer cards
    means sourcing decoration from index metadata, not re-adding frontmatter
    fields.

    Backend implementation details (SKILL.md body / system prompt, role
    line, tool list, source tier, filesystem path,
    tags) are intentionally NOT projected. The gallery only needs
    identity + descriptor fields to render the card; anything richer
    belongs in a dedicated info endpoint.

    Sourced from ``list_expert_skills(include_system=True)`` so
    first-party experts shipped as builtin skills surface alongside
    workspace/global installs.

    Offloaded to a thread because the skill loader does synchronous
    filesystem walking + yaml parsing, which langgraph-dev's
    ``blockbuster`` middleware refuses on the async event loop.

    Response shape (each entry): ``{name, description, byline?,
    capability_tags?, avatar_hint?}`` — the WebUI gallery consumes these.

    The ``workspace_dir`` query parameter names the project whose installs
    count (``400`` when it is not a readable absolute path, ``404`` when that
    project is not served); without it, the project ``GET /api/workspace``
    reports, with the same ``503`` when there is none.
    """
    from EvoScientist.langgraph_dev.folder_check import read_workspace_or_none
    from EvoScientist.tools.skills_manager import list_expert_skills

    requested = request.query_params.get("workspace_dir")
    if requested is None:
        workspace = await asyncio.to_thread(_launcher_project)
        if workspace is None:
            return _no_project()
    else:
        workspace = await asyncio.to_thread(read_workspace_or_none, requested)
        if workspace is None:
            return JSONResponse({"error": "invalid_path"}, status_code=400)
        if not await asyncio.to_thread(_served, workspace):
            return JSONResponse({"error": "not_served"}, status_code=404)

    experts = await asyncio.to_thread(list_expert_skills, True, workspace=workspace)
    teams = []
    for info in experts:
        entry = {
            "name": info.name,
            "description": info.description,
        }
        # Optional gallery fields — omit when unpopulated so the WebUI
        # card degrades gracefully (SkillInfo defaults `byline` /
        # `avatar_hint` to "" and `capability_tags` to [], which we
        # treat as "not declared").
        if info.byline:
            entry["byline"] = info.byline
        if info.capability_tags:
            entry["capability_tags"] = list(info.capability_tags)
        if info.avatar_hint:
            entry["avatar_hint"] = info.avatar_hint
        teams.append(entry)
    return JSONResponse({"teams": teams})


async def get_bg_process_status(request: Request) -> JSONResponse:
    """Return a background process's live status for the client-side reader.

    Background processes launched by ``run_in_background`` live in THIS server
    process's registry (the served main graph runs here), so their exit is only
    observable here. The CLI's ``bg_processes`` state reader polls this route to
    detect completion across the process boundary, mirroring how the async-task
    reader polls run status. ``process_id`` is a query param; ``status`` is one of
    ``running`` / ``success`` / ``error`` / ``interrupted`` / ``unknown`` (untracked id).

    Offloaded to a thread: reading the registry takes a ``threading.Lock`` and does
    a ``Popen.poll()`` syscall, which langgraph-dev's ``blockbuster`` middleware
    refuses on the async event loop.
    """
    from EvoScientist import background

    process_id = request.query_params.get("process_id", "")
    status = await asyncio.to_thread(background.poll_status, process_id)
    return JSONResponse({"status": status})


async def get_bg_processes_running(_request: Request) -> JSONResponse:
    """Return every background process still running in THIS server, across all
    threads, as ``{"running": [{"process_id", "name"}, ...]}``.

    A caller about to restart or stop the backend polls this first: a
    restart / stop tree-kills all bg children, so the caller waits for or
    confirms stopping them and names each. Scoped to all threads (not one origin thread)
    because the restart affects every bg job in the backend. Offloaded to a
    thread for the same reason as :func:`get_bg_process_status` (registry lock +
    ``Popen.poll()`` syscall, which blockbuster refuses on the event loop).
    """
    from EvoScientist import background

    records = await asyncio.to_thread(background.running_records)
    running = [{"process_id": r["process_id"], "name": r["name"]} for r in records]
    return JSONResponse({"running": running})


async def post_policy(request: Request) -> JSONResponse:
    """Resolve HITL decisions for action requests, for non-Python clients.

    The one source of truth is ``channels.interaction.resolve_config_decisions``
    (the same chokepoint Python clients call in-process), evaluated against THIS
    server's own config — the caller never mirrors auto_approve / dangerous_mode
    / allow_list, which is exactly the part that drifted in client-side ports.

    Pure policy — no agent, no graph, no side effects.

    Body: ``{"action_requests": list[dict]}`` (the same action request objects
    the interrupt delivers). Response: ``{"decisions": list[dict] | null}`` —
    a full per-request decisions list when config clears every request, or
    ``null`` when a human decision is needed (the client then prompts and
    hands its own decisions to the resume payload). The decisions shape is
    what ``build_hitl_resume`` consumes, so the non-null response can be
    forwarded verbatim.
    """
    from EvoScientist.channels.interaction import resolve_config_decisions

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    action_requests = None
    if isinstance(body, dict):
        candidate = body.get("action_requests")
        if isinstance(candidate, list) and all(isinstance(r, dict) for r in candidate):
            action_requests = candidate
    if action_requests is None:
        return JSONResponse(
            {"error": "'action_requests' (list of objects) is required"},
            status_code=400,
        )
    # resolve_config_decisions reads config from disk (load_config), which
    # blockbuster refuses on the dev-server event loop - offload like every
    # other blocking read this file serves.
    decisions = await asyncio.to_thread(resolve_config_decisions, action_requests)
    return JSONResponse({"decisions": decisions})


def _served(workspace: Workspace) -> bool:
    from EvoScientist.langgraph_dev.folder_check import is_served
    from EvoScientist.langgraph_dev.registry import REGISTRY

    return is_served(workspace, REGISTRY)


def _launcher_project() -> Workspace | None:
    """The project of the client that started this server, when it is served.
    Blocking."""
    from EvoScientist.paths import launcher_workspace

    launcher = launcher_workspace()
    return launcher if launcher is not None and _served(launcher) else None


def _no_project() -> JSONResponse:
    return JSONResponse(
        {
            "error": "no_workspace",
            "detail": "No folder is selected: this server was started without "
            "a project, or that project is not registered or its folder is "
            "missing.",
        },
        status_code=503,
    )


async def get_workspace(_request: Request) -> JSONResponse:
    """Return the project of the client that started this server as
    ``{workspace_dir, run_dir}``, while it is served.

    ``workspace_dir`` is the project root in the form thread and checkpoint
    metadata use (:attr:`~EvoScientist.paths.Workspace.key`), so a client
    can compare it with them and derive the folder's assistant IDs without
    resolving paths itself; ``run_dir`` is always ``null``. ``503`` (not
    ``404``, which a client takes for a server without this route) when
    there is no such project, or it is not registered or its folder is
    missing.

    Offloaded to a thread because resolving the folders touches the
    filesystem, which blockbuster refuses on the event loop.
    """
    from EvoScientist.paths import SessionDirs

    workspace = await asyncio.to_thread(_launcher_project)
    if workspace is None:
        return _no_project()
    # The same serialisation as thread and checkpoint stamps.
    return JSONResponse({"run_dir": None} | SessionDirs(workspace).metadata())


def _registered_projects() -> list[dict[str, object]]:
    from pathlib import Path

    from EvoScientist.langgraph_dev.assistants import folder_name
    from EvoScientist.langgraph_dev.registry import REGISTRY

    return [
        {
            "workspace_dir": key,
            "name": folder_name(key),
            "missing": not Path(key).is_dir(),
        }
        for key in REGISTRY.registered()
    ]


async def list_workspaces(_request: Request) -> JSONResponse:
    """Return the registered projects as ``{"workspaces": [{workspace_dir,
    name, missing}]}``; ``name`` is the folder's label, which ends the names
    of its workspace assistants (``<graph> · <name>``), and two projects can
    share it (``workspace_dir`` tells them apart); ``missing`` marks a project
    whose folder no longer exists."""
    projects = await asyncio.to_thread(_registered_projects)
    return JSONResponse({"workspaces": projects})


async def register_workspace(request: Request) -> JSONResponse:
    """Register the project in the body's ``workspace_dir`` and set it up.

    Answers once its setup is done (see
    :func:`~EvoScientist.langgraph_dev.workspace_setup.register_workspace`),
    with ``{"workspace_dir": <key>, "host": <bind address>, "tunnel":
    <bool>}`` (whether it has a public tunnel): the client warns when the
    server, which serves every project, can be reached from other machines.
    A folder that does not exist is
    refused, and never created; a setup that fails answers ``500`` (a project
    that was not registered stays unregistered). Needs the local secret.
    """
    from EvoScientist.langgraph_dev.folder_check import read_workspace_or_none
    from EvoScientist.paths import SessionDirs

    if (denied := require_local_secret(request)) is not None:
        return denied
    try:
        body = await request.json()
    except ValueError:
        body = None
    value = body.get("workspace_dir") if isinstance(body, dict) else None
    workspace = await asyncio.to_thread(read_workspace_or_none, value)
    if workspace is None:
        return JSONResponse({"error": "invalid_path"}, status_code=400)
    # How the server can be reached from other machines, read before the
    # setup so nothing fails once the project is registered: the bind address
    # the manager resolved into its configuration, and the tunnel its
    # boundary lets requests through.
    cfg = await asyncio.to_thread(get_effective_config)
    exposure = {"host": cfg.langgraph_dev_host, "tunnel": tunnel_host() is not None}
    try:
        async with workspace_setup.registration_lock():
            await workspace_setup.register_workspace(SessionDirs(workspace))
    except workspace_setup.FolderMissingError:
        return JSONResponse({"error": "folder_missing"}, status_code=400)
    except Exception as exc:
        logger.exception("Could not set %s up.", workspace.key)
        return JSONResponse(
            {"error": "setup_failed", "detail": str(exc)}, status_code=500
        )
    return JSONResponse({"workspace_dir": workspace.key} | exposure)


async def reconcile_autoskills(request: Request) -> JSONResponse:
    """Reconcile every served project's AutoSkills cron with the server's
    configuration, after a client changed it: ``{"projects": <count>,
    "failed": [<reason>, ...]}``. Needs the local secret."""
    if (denied := require_local_secret(request)) is not None:
        return denied
    try:
        async with workspace_setup.registration_lock():
            count, failed = await workspace_setup.reconcile_autoskills_everywhere()
    except Exception as exc:
        logger.exception("Could not reconcile AutoSkills.")
        return JSONResponse(
            {"error": "reconcile_failed", "detail": str(exc)}, status_code=500
        )
    return JSONResponse({"projects": count, "failed": failed})


async def _unregister(workspace: Workspace) -> JSONResponse:
    from EvoScientist.langgraph_dev import server_crons
    from EvoScientist.langgraph_dev.registry import REGISTRY

    if not await asyncio.to_thread(REGISTRY.is_registered, workspace.key):
        return JSONResponse({"error": "not_registered"}, status_code=404)
    crons = await server_crons.project_crons(workspace)
    if enabled := server_crons.enabled_user_crons(crons):
        return JSONResponse(
            {
                "error": "enabled_user_crons",
                "cron_ids": [str(cron["cron_id"]) for cron in enabled],
            },
            status_code=409,
        )
    await server_crons.delete_crons(
        [cron["cron_id"] for cron in server_crons.system_crons(crons)]
    )
    await workspace_setup.unregister_workspace(workspace)
    return JSONResponse({"workspace_dir": workspace.key})


async def unregister_workspace(request: Request) -> JSONResponse:
    """Unregister the project in the ``workspace_dir`` query parameter.

    Its folder, threads, assistants and user crons all stay; its AutoSkills
    and proactive crons are deleted (their reconciles recreate them).
    Refused (``409``) while it has an enabled user cron, also when its folder
    is missing: a client disables or deletes those first. Waits for a
    registration in progress. Needs the local secret.
    """
    from EvoScientist.langgraph_dev.folder_check import read_workspace_or_none

    if (denied := require_local_secret(request)) is not None:
        return denied
    workspace = await asyncio.to_thread(
        read_workspace_or_none, request.query_params.get("workspace_dir")
    )
    if workspace is None:
        return JSONResponse({"error": "invalid_path"}, status_code=400)
    try:
        async with workspace_setup.registration_lock():
            return await _unregister(workspace)
    except Exception as exc:
        logger.exception("Could not unregister %s.", workspace.key)
        return JSONResponse(
            {"error": "unregister_failed", "detail": str(exc)}, status_code=500
        )


async def get_settings(_request: Request) -> JSONResponse:
    """Return the configuration for the WebUI's wizard and settings page.

    Secret values are never in the body (each reads as ``{"set", "hint"?}``),
    so no local secret is needed. The body is built by
    :func:`~EvoScientist.langgraph_dev.settings_api.read_settings`.
    """
    from EvoScientist.langgraph_dev import settings_api

    return JSONResponse(await asyncio.to_thread(settings_api.read_settings))


async def patch_settings(request: Request) -> JSONResponse:
    """Update one or more settings; needs the local secret.

    Validates every field first and writes nothing if one fails. See
    :func:`~EvoScientist.langgraph_dev.settings_api.patch_settings`.
    """
    from EvoScientist.langgraph_dev import settings_api

    if (denied := require_local_secret(request)) is not None:
        return denied
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"errors": {"body": "invalid JSON"}}, status_code=400)
    status, payload = await asyncio.to_thread(settings_api.patch_settings, body)
    if status == 200 and isinstance(body, dict) and AUTOSKILL_SETTINGS & set(body):
        # Every project's AutoSkills cron follows the server's settings, in
        # the background: the save does not wait for a registration in
        # progress.
        _reconcile_in_background()
    return JSONResponse(payload, status_code=status)


# Settings-triggered reconciles, kept until done so they are not
# garbage-collected.
_reconciles: set[asyncio.Task] = set()


def _reconcile_in_background() -> None:
    async def reconcile() -> None:
        try:
            async with workspace_setup.registration_lock():
                await workspace_setup.reconcile_autoskills_everywhere()
        except Exception:
            # The settings are saved; each project's next setup reconciles it.
            logger.exception("Could not reconcile AutoSkills after a settings change.")

    task = asyncio.create_task(reconcile())
    _reconciles.add(task)
    task.add_done_callback(_reconciles.discard)


async def post_settings_test_key(request: Request) -> JSONResponse:
    """Check a key (and a base URL) without saving it; needs the local secret,
    since a ``base_url`` makes the server call an address from the request."""
    from EvoScientist.langgraph_dev import settings_api

    if (denied := require_local_secret(request)) is not None:
        return denied
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"errors": {"body": "invalid JSON"}}, status_code=400)
    status, payload = await asyncio.to_thread(settings_api.check_key_request, body)
    return JSONResponse(payload, status_code=status)


async def get_instance(_request: Request) -> JSONResponse:
    """The public name of this server's local secret
    (:func:`~EvoScientist.local_secret.instance_id`), so a client can tell a
    different server on the port before sending its secret. Public, it does
    not prove the server holds the secret."""
    return JSONResponse({"instance": instance_id()})


async def verify_local_secret(request: Request) -> Response:
    """``204`` when the request carries the local secret, ``403`` otherwise, so
    a client can check the value it read."""
    if (denied := require_local_secret(request)) is not None:
        return denied
    return Response(status_code=204)


@contextlib.asynccontextmanager
async def _lifespan(_app: Starlette):
    # langgraph-api runs this inside its own lifespan, after it loaded the
    # graphs and started its cron scheduler and run workers, and before it
    # serves requests; the file I/O goes to a thread because blockbuster
    # refuses it on the event loop.
    await asyncio.to_thread(load_local_secret)
    await workspace_setup.set_up_at_start()
    yield


def _build_app() -> Starlette:
    """The custom app, then the boundary's variables removed from the environment.

    Both are read by then (langgraph-api imports its config before this app,
    and the guard keeps its ``Trust``); the agent's commands, such as a
    ``langgraph dev`` of the user's own project, must not inherit them.
    """
    app = Starlette(
        lifespan=_lifespan,
        # Read when langgraph-api imports this app: LANGGRAPH_API_URL is set by
        # then, and a workspace ``.env`` merged later cannot change it.
        middleware=[Middleware(BrowserBoundary, trust=Trust.from_env())],
        routes=[
            Route("/api/models", get_models, methods=["GET"]),
            Route("/api/teams", get_teams, methods=["GET"]),
            Route("/api/bg_process_status", get_bg_process_status, methods=["GET"]),
            Route(
                "/api/bg_processes/running", get_bg_processes_running, methods=["GET"]
            ),
            Route("/api/policy", post_policy, methods=["POST"]),
            Route("/api/workspace", get_workspace, methods=["GET"]),
            Route("/api/workspaces", list_workspaces, methods=["GET"]),
            Route("/api/workspaces", register_workspace, methods=["POST"]),
            Route("/api/workspaces", unregister_workspace, methods=["DELETE"]),
            Route("/api/autoskills/reconcile", reconcile_autoskills, methods=["POST"]),
            Route("/api/instance", get_instance, methods=["GET"]),
            Route("/api/local-secret/verify", verify_local_secret, methods=["GET"]),
            Route("/api/settings", get_settings, methods=["GET"]),
            Route("/api/settings", patch_settings, methods=["PATCH"]),
            Route("/api/settings/test-key", post_settings_test_key, methods=["POST"]),
        ],
    )
    os.environ.pop("CORS_CONFIG", None)
    os.environ.pop(BOUNDARY_ENV, None)
    return app


# langgraph-api imports this app before its runtime loads the store, so the
# server (the process working in the server home) takes its data folder's lock
# and backs the store up first; anything else importing this module does not.
if server_home.serves_from_home():
    server_home.hold_server_lock()
    server_home.back_up_store()
app = _build_app()
