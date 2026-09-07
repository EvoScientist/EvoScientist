"""Custom HTTP routes mounted alongside the langgraph dev server.

The langgraph-api host supports a top-level ``http`` key in
``langgraph.json`` that names an ASGI app to mount on the same
process as the graph. We use it to surface the registry the WebUI's
``/model`` picker needs.

Why this lives here and not as a separate sidecar: the WebUI talks to
``EvoSci deploy``'s langgraph endpoint anyway, so one origin keeps the
WebUI's fetch logic simple — no CORS dance, no extra port to configure.

Why Starlette and not FastAPI: ``langgraph_api`` already depends on
Starlette; adding FastAPI would pull in pydantic v1-vs-v2 reconciliation
the deploy doesn't need. The one route here has no input model, just a
JSON body, so the lower-level surface is sufficient.

Lightweight by design — module-level imports stick to ``config``,
``llm.models`` (registry only; no chat-model construction), and
Starlette itself. Nothing on this surface should pull the agent into
memory.
"""

from __future__ import annotations

import asyncio

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from EvoScientist.config import get_effective_config
from EvoScientist.llm.models import list_model_picker_entries


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
    directory carrying a sibling ``AGENTS.md`` (or, on the deprecated path,
    ``type: expert`` SKILL.md frontmatter). The response is a curated,
    gallery-safe projection: name + description, plus optional ``byline`` /
    ``capability_tags`` / ``avatar_hint`` when the skill populates them.

    Cards for experts on the current contract carry name + description only:
    the decoration fields were actor metadata in SKILL.md frontmatter, which
    that contract removes rather than relocates (``AGENTS.md`` has no
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
    from EvoScientist.tools.skills_manager import list_expert_skills

    experts = await asyncio.to_thread(list_expert_skills, True)
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


async def post_policy(request: Request) -> JSONResponse:
    """Resolve the shell-approval policy for a command, for non-Python clients.

    The one source of truth is ``backends.resolve_action_decision`` (the same
    function Python clients call in-process). Exposing it here lets a non-Python
    client (WebUI) stop re-porting the allow-list / dangerous-command logic and
    ask the backend instead. Pure policy — no agent, no graph, no side effects.

    Body: ``{"command": str, "auto_approve"?: bool, "dangerous_mode"?: bool,
    "allow_list"?: list[str]}``. Response: ``{"decision": "approve"|"reject"|
    "prompt", "reason": str}``.
    """
    from EvoScientist.backends import resolve_action_decision

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    if not isinstance(body, dict) or not isinstance(body.get("command"), str):
        return JSONResponse(
            {"error": "'command' (string) is required"}, status_code=400
        )
    allow_list = body.get("allow_list")
    if allow_list is not None and not (
        isinstance(allow_list, list) and all(isinstance(s, str) for s in allow_list)
    ):
        return JSONResponse(
            {"error": "'allow_list' must be a list of strings"}, status_code=400
        )
    verdict = resolve_action_decision(
        body["command"],
        auto_approve=bool(body.get("auto_approve", False)),
        dangerous_mode=bool(body.get("dangerous_mode", False)),
        allow_list=allow_list,
    )
    return JSONResponse({"decision": verdict.decision.value, "reason": verdict.reason})


app = Starlette(
    routes=[
        Route("/api/models", get_models, methods=["GET"]),
        Route("/api/teams", get_teams, methods=["GET"]),
        Route("/api/bg_process_status", get_bg_process_status, methods=["GET"]),
        Route("/api/policy", post_policy, methods=["POST"]),
    ]
)
