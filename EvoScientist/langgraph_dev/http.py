"""Custom HTTP routes mounted alongside the langgraph dev server.

The langgraph-api host supports a top-level ``http`` key in
``langgraph.json`` that names an ASGI app to mount on the same
process as the graph. We use it to surface the registry the WebUI's
``/model`` picker needs.

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
import os

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from EvoScientist.config import get_effective_config
from EvoScientist.langgraph_dev.boundary import BOUNDARY_ENV, BrowserBoundary, Trust
from EvoScientist.llm.models import list_model_picker_entries
from EvoScientist.local_secret import load_local_secret, require_local_secret


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
    Offloaded to a thread because ``get_effective_config()`` calls
    ``find_dotenv(usecwd=True)`` which invokes ``os.getcwd()`` — a
    blocking syscall that langgraph-dev's ``blockbuster`` middleware
    refuses to allow on the async event loop (would surface as a 500).
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


async def get_teams(_request: Request) -> JSONResponse:
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
    """
    from EvoScientist.paths import process_workspace
    from EvoScientist.tools.skills_manager import list_expert_skills

    def _list() -> list:
        return list_expert_skills(True, workspace=process_workspace())

    experts = await asyncio.to_thread(_list)
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


async def get_workspace(_request: Request) -> JSONResponse:
    """Return the folder this server serves as ``{workspace_dir, run_dir}``.

    ``workspace_dir`` is the project root in the form thread and checkpoint
    metadata use (:attr:`~EvoScientist.paths.Workspace.key`), so a client
    can compare it with them and derive the folder's assistant IDs without
    resolving paths itself. ``run_dir`` is the run folder of a ``--mode=run``
    session that holds the server, else ``null``.

    Offloaded to a thread because resolving the folders touches the
    filesystem, which blockbuster refuses on the event loop.
    """
    from EvoScientist.paths import process_session_dirs

    dirs = await asyncio.to_thread(process_session_dirs)
    # The same serialisation as thread and checkpoint stamps.
    return JSONResponse({"run_dir": None} | dirs.metadata())


async def verify_local_secret(request: Request) -> Response:
    """``204`` when the request carries the local secret, ``403`` otherwise, so
    a client can check the value it read."""
    if (denied := require_local_secret(request)) is not None:
        return denied
    return Response(status_code=204)


@contextlib.asynccontextmanager
async def _lifespan(_app: Starlette):
    # langgraph-api runs this inside its own lifespan; the file I/O goes to a
    # thread because blockbuster refuses it on the event loop.
    await asyncio.to_thread(load_local_secret)
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
            Route("/api/local-secret/verify", verify_local_secret, methods=["GET"]),
        ],
    )
    os.environ.pop("CORS_CONFIG", None)
    os.environ.pop(BOUNDARY_ENV, None)
    return app


app = _build_app()
