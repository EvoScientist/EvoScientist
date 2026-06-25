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

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from EvoScientist.config import load_config
from EvoScientist.llm.models import list_models_by_provider


async def get_models(_request: Request) -> JSONResponse:
    """Return the model registry as ``{entries, default}``.

    ``entries`` preserves the registry order so the WebUI picker can
    rank providers per short name the same way the backend would.
    ``default`` reflects the deployment's currently-configured fallback
    (``config.yaml``'s ``model`` / ``provider`` — what ``/model reset``
    would land on). Returned even when the configured pair isn't in
    the registry, so the picker can still label it.
    """
    cfg = load_config()
    entries = [
        {"name": name, "model_id": model_id, "provider": provider}
        for name, model_id, provider in list_models_by_provider()
    ]
    return JSONResponse(
        {
            "entries": entries,
            "default": {"name": cfg.model, "provider": cfg.provider},
        }
    )


app = Starlette(
    routes=[
        Route("/api/models", get_models, methods=["GET"]),
    ]
)
