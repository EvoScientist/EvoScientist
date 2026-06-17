"""EvoScientist Agent graph construction.

This module defines the agent graph and its factory functions.  All heavy
initialization (deepagents, backends, LLM, middleware) is deferred to first
use so that importing this module is fast and non-agent CLI commands
(``EvoSci config list``, ``EvoSci onboard``) never pay the cost.

Usage:
    from EvoScientist import EvoScientist_agent
    from EvoScientist.stream.events import stream_agent_events

    # Notebook / programmatic usage
    async for event in stream_agent_events(
        EvoScientist_agent, "your question", thread_id="1"
    ):
        ...
"""

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from langchain.agents.middleware import AgentMiddleware, HumanInTheLoopMiddleware

from . import paths as _paths_mod
from .config import (
    MemoryControls,
    MemoryObservationTarget,
    apply_config_to_env,
    get_effective_config,
)
from .memory import MemorySourceType
from .paths import set_active_workspace, set_workspace_root
from .prompts import get_system_prompt

# Suppress noisy warnings from deepagents skill loader (non-string frontmatter fields, etc.)
logging.getLogger("deepagents.middleware.skills").setLevel(logging.ERROR)

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

# =============================================================================
# Constants
# =============================================================================

SUBAGENTS_CONFIG = Path(__file__).parent / "subagents"
SKILLS_DIR = str(Path(__file__).parent / "skills")
DEFAULT_SKILL_SOURCES = ("/skills/",)

# =============================================================================
# Lazy state — initialized on first use, not at import time
# =============================================================================

_config = None
_chat_model = None
# Track the (model, provider) binding of _chat_model so cache invalidates
# when config.model/provider change (e.g. via /model). Without this,
# _ensure_chat_model() returns the stale cached instance even after
# _ensure_config(new_cfg) has overwritten the active config — causing
# /model switch to lag one step (see issue #179).
_chat_model_key: tuple[str | None, str | None] | None = None

# Auxiliary model for background/helper LLM calls (memory workers + main-agent
# tool selector). Cached separately from the main model; falls back to the main
# instance when the auxiliary_* config fields are empty (see
# _ensure_auxiliary_chat_model).
_auxiliary_chat_model = None
_auxiliary_chat_model_key: tuple[str | None, str | None] | None = None

# Cache MCP tools by the effective config signature to avoid reconnecting
# to MCP servers on every `/new` when config is unchanged.
_MCP_TOOLS_CACHE_KEY: str | None = None
_MCP_TOOLS_CACHE_VALUE: dict[str, list] | None = None

# Default agent (no checkpointer) — used by langgraph dev / LangSmith / notebooks.
# Lazily constructed on first access so MCP tools are included without
# spawning subprocesses at import time.
_EvoScientist_agent = None


# =============================================================================
# Lazy initialization helpers
# =============================================================================


def set_active_config(cfg) -> None:
    """Commit *cfg* as the active module config.

    Public commit path for callers (e.g. ``/model``) that built an agent on
    the pure ``create_cli_agent(config=..., chat_model=...)`` path and now
    want it to become the session-wide active config.  This is the write half
    of ``_ensure_config(cfg)`` extracted so the pure path can defer the commit
    until the agent has been built successfully.
    """
    global _config
    _config = cfg
    apply_config_to_env(cfg)


def _apply_env_from_config(cfg) -> None:
    """Apply *cfg*'s API-key env vars without caching it as ``_config``.

    ``apply_config_to_env`` is set-if-unset (guards on ``not
    os.environ.get(...)``), so this is idempotent and safe to call on the pure
    path, where no module globals may be written.
    """
    apply_config_to_env(cfg)


def _ensure_config(config=None):
    """Return cached config.  If *config* is passed, cache and use it."""
    if config is not None:
        set_active_config(config)
    if _config is None:
        set_active_config(get_effective_config())
    return _config


def _build_chat_model(cfg):
    """Build a chat model from *cfg* without writing any module globals.

    Pure-construction counterpart to ``_ensure_chat_model``: used by ``/model``
    to verify a switch before committing, and threaded into
    ``create_cli_agent(chat_model=...)`` so the new agent binds the requested
    model without touching the cached ``_chat_model``.
    """
    from .llm import get_chat_model

    return get_chat_model(model=cfg.model, provider=cfg.provider)


def _replace_chat_model(instance, key: tuple[str | None, str | None]) -> None:
    """Install a new chat model and propagate the related invariants.

    Single write point for ``_chat_model`` / ``_chat_model_key`` /
    ``_EvoScientist_agent``: both ``_ensure_chat_model`` (cache-miss
    rebuild) and ``set_chat_model`` (explicit switch via ``/model``)
    funnel through here so the three globals can never drift.
    """
    global _chat_model, _chat_model_key, _EvoScientist_agent
    _chat_model = instance
    _chat_model_key = key
    # The lazy default agent captured a reference to the previous
    # ``_chat_model`` at build time, so it must be rebuilt on next access.
    _EvoScientist_agent = None


def _ensure_chat_model():
    """Return cached chat model, rebuilding if cfg.model/provider changed.

    The cache key is the current config's ``(model, provider)``. If it
    differs from the key that built ``_chat_model``, rebuild — this makes
    ``create_cli_agent(config=temp_cfg)`` bind the freshly requested model
    into the new agent without requiring callers to interleave
    ``set_chat_model()`` calls in any particular order.
    """
    cfg = _ensure_config()
    key = (cfg.model, cfg.provider)
    if _chat_model is None or _chat_model_key != key:
        _replace_chat_model(_build_chat_model(cfg), key)
    return _chat_model


def _ensure_auxiliary_chat_model():
    """Return the auxiliary chat model for background/helper LLM calls.

    Resolves ``(cfg.auxiliary_model or cfg.model, cfg.auxiliary_provider or
    cfg.provider)``. When the auxiliary fields are empty — or resolve to the same
    ``(model, provider)`` pair as the main model — returns the main
    ``_ensure_chat_model()`` instance directly, so no second client is built.
    Otherwise it is cached separately under its own key. Onboard sets the
    provider alongside the model, so the ``or cfg.provider`` fallback only
    matters for a model set without an explicit auxiliary provider.
    """
    global _auxiliary_chat_model, _auxiliary_chat_model_key
    from .llm import get_chat_model

    cfg = _ensure_config()
    aux_model = cfg.auxiliary_model or cfg.model
    aux_provider = cfg.auxiliary_provider or cfg.provider
    if (aux_model, aux_provider) == (cfg.model, cfg.provider):
        return _ensure_chat_model()
    key = (aux_model, aux_provider)
    if _auxiliary_chat_model is None or _auxiliary_chat_model_key != key:
        _auxiliary_chat_model = get_chat_model(model=aux_model, provider=aux_provider)
        _auxiliary_chat_model_key = key
    return _auxiliary_chat_model


def set_chat_model(model: str, provider: str | None = None):
    """Replace the cached chat model with a new one.

    Called by ``/model`` to switch the LLM mid-session.  No-op when the
    cache already holds the requested ``(model, provider)`` — avoids
    spawning a second ``get_chat_model`` instance (and its HTTP client)
    under the ``/model`` flow where ``_ensure_chat_model`` has already
    rebuilt ``_chat_model`` during the preceding ``_load_agent`` call.
    Returns the current chat model instance.
    """
    from .llm import get_chat_model

    # Invalidate the auxiliary cache too: when auxiliary_* is empty it mirrors
    # the main model, so a /model switch must let it re-resolve to the new main.
    global _auxiliary_chat_model, _auxiliary_chat_model_key
    _auxiliary_chat_model = None
    _auxiliary_chat_model_key = None

    key = (model, provider)
    if _chat_model is None or _chat_model_key != key:
        _replace_chat_model(get_chat_model(model=model, provider=provider), key)
    return _chat_model


def set_chat_model_instance(instance, key: tuple[str | None, str | None]) -> None:
    """Commit an already-built chat model *instance* as the active model.

    Companion to ``set_active_config`` for the pure path: installs a model that
    ``_build_chat_model`` already constructed (e.g. during a ``/model`` verify)
    without rebuilding it, keeping ``_chat_model`` / ``_chat_model_key`` /
    ``_EvoScientist_agent`` in sync via ``_replace_chat_model``.  Unlike
    ``set_chat_model``, the caller owns the ``(model, provider)`` *key*.
    """
    _replace_chat_model(instance, key)


# =============================================================================
# MCP caching
# =============================================================================


def _load_mcp_config_once() -> tuple[str, dict]:
    """Load MCP config and return ``(signature, config)``."""
    from .mcp.client import load_mcp_config

    cfg = load_mcp_config()
    if not cfg:
        return "", {}
    try:
        sig = json.dumps(cfg, sort_keys=True, ensure_ascii=True)
    except TypeError:
        sig = repr(cfg)
    return sig, cfg


def _load_mcp_tools_cached(on_progress=None) -> dict[str, list]:
    """Load MCP tools with config-aware caching.

    Args:
        on_progress: Optional per-server progress callback forwarded to
            :func:`EvoScientist.mcp.load_mcp_tools`.  Only invoked on a
            cache miss — cached replays don't re-emit progress events.
    """
    global _MCP_TOOLS_CACHE_KEY, _MCP_TOOLS_CACHE_VALUE

    from .mcp import load_mcp_tools

    cfg_key, cfg = _load_mcp_config_once()
    if not cfg_key:
        _MCP_TOOLS_CACHE_KEY = ""
        _MCP_TOOLS_CACHE_VALUE = {}
        return {}

    if _MCP_TOOLS_CACHE_KEY == cfg_key and _MCP_TOOLS_CACHE_VALUE is not None:
        return {k: list(v) for k, v in _MCP_TOOLS_CACHE_VALUE.items()}

    loaded = load_mcp_tools(config=cfg, on_progress=on_progress)
    _MCP_TOOLS_CACHE_KEY = cfg_key
    _MCP_TOOLS_CACHE_VALUE = {k: list(v) for k, v in loaded.items()}
    return {k: list(v) for k, v in loaded.items()}


# =============================================================================
# Agent construction helpers
# =============================================================================


def _configured_system_prompt(cfg) -> str:
    # In dangerous mode the agent works on the real filesystem; give it the real
    # cwd so it can use absolute paths instead of the virtual `/` workspace root.
    real_cwd = str(_paths_mod.resolve_virtual_path("/")) if cfg.dangerous_mode else None
    return get_system_prompt(
        dangerous=cfg.dangerous_mode,
        cwd=real_cwd,
    )


def _inject_subagent_middleware(
    subs: list[dict],
    *,
    workspace_dir: str | Path | None = None,
    cfg=None,
    chat_model=None,
) -> None:
    """Ensure every subagent gets error handling and context management middleware.

    Without this, subagent tool errors are caught by LangGraph's default
    ToolNode handler which produces terse messages without tracebacks or
    retry guidance — reducing the subagent's ability to self-recover.

    *chat_model*, when provided, is forwarded to the subagents'
    ``create_context_editing_middleware`` so the pure ``create_cli_agent``
    path doesn't fall back to the global-writing ``_ensure_chat_model()``.
    """
    from .middleware import (
        ContextOverflowMapperMiddleware,
        MemoryLifecycleRole,
        ToolErrorHandlerMiddleware,
        create_context_editing_middleware,
        create_memory_lifecycle_middleware,
        create_memory_middleware,
        create_runtime_context_middleware,
    )

    cfg = cfg if cfg is not None else _ensure_config()
    memory_controls = MemoryControls.from_config(cfg)
    memory_dir = str(_paths_mod.MEMORIES_DIR)
    for sa in subs:
        name = str(sa.get("name") or "sub-agent")
        source_type = MemorySourceType.SUBAGENT
        memory_middleware = create_memory_middleware(
            memory_dir,
            workspace_dir=workspace_dir,
            source_type=source_type,
            source_agent=name,
            enable_profile_memory=memory_controls.profile_enabled,
            enable_observation_memory=memory_controls.observations_enabled,
            enable_observation_tool=memory_controls.observation_tool_enabled(
                MemoryObservationTarget.AGENT
            ),
        )
        middleware = [
            # Subagents share the main agent's model: use the threaded
            # ``chat_model`` on the pure path, else defer to the factory's
            # ``_ensure_chat_model()`` fallback (when ``chat_model=None``).
            create_context_editing_middleware(chat_model),
            create_runtime_context_middleware(),
            ToolErrorHandlerMiddleware(),
            ContextOverflowMapperMiddleware(),
        ]
        if memory_controls.memory_enabled:
            middleware.append(memory_middleware)
        if memory_controls.worker_needed(MemoryObservationTarget.SUBAGENT_WORKER):
            middleware.append(
                create_memory_lifecycle_middleware(
                    memory_dir,
                    workspace_dir=workspace_dir,
                    project_id=memory_middleware.project_id,
                    role=MemoryLifecycleRole.SUBAGENT,
                    source_agent=name,
                )
            )
        sa.setdefault("middleware", []).extend(middleware)


def _ensure_general_purpose_subagent(subs: list[dict]) -> None:
    """Materialize DeepAgents' default subagent so our middleware wraps it."""
    from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT

    name = GENERAL_PURPOSE_SUBAGENT["name"]
    if any(sa.get("name") == name for sa in subs):
        return

    subs.insert(
        0,
        {
            **GENERAL_PURPOSE_SUBAGENT,
            "skills": list(DEFAULT_SKILL_SOURCES),
        },
    )


def _maybe_swap_async_subagents(
    subs: list, middleware: list | None = None, *, cfg=None
) -> list:
    """Replace ``_async``-flagged sub-agents with ``AsyncSubAgent`` specs when enabled.

    Reads the ``_async`` field carried through by ``utils.load_subagents._build_one``
    (sourced from each yaml's ``async: true`` flag). When
    ``config.enable_async_subagents`` is also set, those sub-agents are
    swapped from synchronous in-process dicts to ``AsyncSubAgent`` references
    pointing at the langgraph dev graph of the same name.

    The deployed graphs live in ``EvoScientist.langgraph_dev.graphs`` and
    are registered in ``EvoScientist/langgraph_dev/langgraph.json``.

    Adding a new async sub-agent requires no change here — flip
    ``async: true`` in its yaml and create the matching deployment graph.

    All return paths strip the internal ``_async`` field from sub-agent dicts
    before handoff, since deepagents may schema-validate the kwarg.

    When async subagents are actually swapped in and ``middleware`` is provided,
    appends ``AsyncWatcherMiddleware`` so launches spawn an
    ``async_notifier`` watcher.
    """
    cfg = cfg if cfg is not None else _ensure_config()
    if not getattr(cfg, "enable_async_subagents", False):
        # Async fully disabled — strip the internal flag before handoff.
        for s in subs:
            s.pop("_async", None)
        return subs

    # Guard: if the langgraph dev subprocess never came up (port conflict,
    # binary missing, etc.), routing sub-agents to a dead URL produces hangs
    # and confusing tool errors. Fall back to in-process sync delegation.
    from .langgraph_dev.manager import is_async_subagents_available

    if not is_async_subagents_available():
        logging.getLogger(__name__).warning(
            "enable_async_subagents=true but langgraph dev is not reachable; "
            "falling back to in-process sync delegation for all sub-agents."
        )
        # Strip the internal ``_async`` flag (carried from ``load_subagents``)
        # before sub-agents reach deepagents — it's never a deepagents key.
        for s in subs:
            s.pop("_async", None)
        return subs

    # The ``_async`` flag was set by ``utils.load_subagents._build_one`` from
    # each yaml's ``async:`` field. No need to re-parse the yaml files here.
    async_specs: dict[str, str] = {
        s["name"]: s.get("description", "") for s in subs if s.get("_async")
    }

    if not async_specs:
        for s in subs:
            s.pop("_async", None)
        return subs

    from deepagents import AsyncSubAgent

    port = int(getattr(cfg, "langgraph_dev_port", 6174))
    out = []
    agent_specs: dict[str, AsyncSubAgent] = {}
    # MCP tools routed to async sub-agents (via ``expose_to: <name>`` in
    # mcp.yaml) ARE delivered — the deployed factory
    # ``subagents/_factory.py:build_async_subagent_graph`` loads its own MCP
    # connection per server (cost: one extra MCP server subprocess per
    # exposed server, since stdio transports can't share across processes).
    for s in subs:
        name = s.get("name")
        if name in async_specs:
            spec = AsyncSubAgent(
                name=name,
                description=async_specs[name],
                graph_id=name,
                url=f"http://localhost:{port}",
            )
            agent_specs[name] = spec
            out.append(spec)
        else:
            # Strip the internal flag before handoff to deepagents.
            s.pop("_async", None)
            out.append(s)

    if agent_specs and middleware is not None:
        from .middleware.async_watcher import AsyncWatcherMiddleware

        middleware.append(AsyncWatcherMiddleware(agent_specs))

    # Forward the CLI's live (model, provider) into deepagents'
    # start/update_async_task tool calls so the deployed graph can
    # re-resolve its chat model per run via ConfigurableModelMiddleware.
    # Idempotent — safe to call on every CLI startup.
    if agent_specs:
        from .llm.patches import _patch_deepagents_model_passthrough

        _patch_deepagents_model_passthrough()

    return out


def _build_base_kwargs(
    base_backend, base_middleware, *, cfg=None, chat_model=None, workspace_dir=None
):
    """Build agent kwargs *without* MCP (fast, no subprocess spawning)."""
    from .tools import skill_manager, tavily_search, think_tool
    from .utils import load_subagents

    cfg = cfg if cfg is not None else _ensure_config()
    tool_registry = {"think_tool": think_tool}
    if os.environ.get("TAVILY_API_KEY"):
        tool_registry["tavily_search"] = tavily_search
    base_tools = [think_tool, skill_manager]

    subs = load_subagents(
        SUBAGENTS_CONFIG,
        tool_registry=tool_registry,
    )
    _ensure_general_purpose_subagent(subs)
    _inject_subagent_middleware(
        subs, workspace_dir=workspace_dir, cfg=cfg, chat_model=chat_model
    )
    subs = _maybe_swap_async_subagents(subs, base_middleware, cfg=cfg)
    return {
        "name": "EvoScientist",
        "model": chat_model if chat_model is not None else _ensure_chat_model(),
        "tools": list(base_tools),
        "backend": base_backend,
        "subagents": subs,
        "middleware": base_middleware,
        "system_prompt": _configured_system_prompt(cfg),
        "skills": list(DEFAULT_SKILL_SOURCES),
    }


def load_mcp_and_build_kwargs(
    base_backend,
    base_middleware,
    *,
    on_mcp_progress=None,
    cfg=None,
    chat_model=None,
    workspace_dir=None,
):
    """Load MCP tools (cached by config) and build agent kwargs.

    Re-connects to MCP servers only when the effective MCP config changes.
    Falls back to base kwargs if no MCP configured.

    Args:
        on_mcp_progress: Optional per-server progress callback.  Forwarded
            to the MCP loader so UIs can render live status.
        cfg: Explicit config to thread through instead of reading the cached
            ``_config``.  Used by the pure ``create_cli_agent`` path.
        chat_model: Explicit chat model to bind instead of
            ``_ensure_chat_model()`` (which would write module globals).
    """
    from .tools import skill_manager, tavily_search, think_tool
    from .utils import load_subagents

    cfg = cfg if cfg is not None else _ensure_config()
    mcp_by_agent = _load_mcp_tools_cached(on_progress=on_mcp_progress)
    if not mcp_by_agent:
        return _build_base_kwargs(
            base_backend,
            base_middleware,
            cfg=cfg,
            chat_model=chat_model,
            workspace_dir=workspace_dir,
        )

    tool_registry = {"think_tool": think_tool}
    if os.environ.get("TAVILY_API_KEY"):
        tool_registry["tavily_search"] = tavily_search
    base_tools = [think_tool, skill_manager]

    # Fresh tool registry — start from base tools + MCP tools
    registry = dict(tool_registry)
    for tools in mcp_by_agent.values():
        for t in tools:
            registry[t.name] = t

    mcp_main = mcp_by_agent.pop("main", [])

    subs = load_subagents(
        SUBAGENTS_CONFIG,
        tool_registry=registry,
    )

    _ensure_general_purpose_subagent(subs)
    _inject_subagent_middleware(
        subs, workspace_dir=workspace_dir, cfg=cfg, chat_model=chat_model
    )

    # Inject MCP tools into subagents by name
    for sa in subs:
        if sa_tools := mcp_by_agent.get(sa["name"], []):
            sa.setdefault("tools", []).extend(sa_tools)

    # Swap selected sub-agents to AsyncSubAgent (must happen AFTER MCP injection
    # since async sub-agents are remote graphs that load their own tools).
    subs = _maybe_swap_async_subagents(subs, base_middleware, cfg=cfg)

    return {
        "name": "EvoScientist",
        "model": chat_model if chat_model is not None else _ensure_chat_model(),
        "tools": base_tools + mcp_main,
        "backend": base_backend,
        "subagents": subs,
        "middleware": base_middleware,
        "system_prompt": _configured_system_prompt(cfg),
        "skills": list(DEFAULT_SKILL_SOURCES),
    }


# =============================================================================
# Default agent (langgraph dev / notebooks)
# =============================================================================


def _get_default_backend():
    """Build the default composite backend from current paths."""
    from deepagents.backends import CompositeBackend, FilesystemBackend

    from .backends import CustomSandboxBackend, MergedSkillsBackend

    cfg = _ensure_config()
    workspace_dir = str(_paths_mod.WORKSPACE_ROOT)
    set_active_workspace(workspace_dir)
    memory_dir = str(_paths_mod.MEMORIES_DIR)
    user_skills_dir = str(_paths_mod.USER_SKILLS_DIR)
    global_skills_dir = str(_paths_mod.GLOBAL_SKILLS_DIR)

    # Dangerous mode opens the workspace (`/`) route to the real filesystem;
    # the /skills/ and /memories/ routes stay confined (virtual_mode=True).
    ws_backend = CustomSandboxBackend(
        root_dir=workspace_dir,
        virtual_mode=True,
        timeout=cfg.sandbox_execute_timeout,
        dangerous=cfg.dangerous_mode,
    )
    sk_backend = MergedSkillsBackend(
        primary_dir=user_skills_dir,
        global_dir=global_skills_dir,
        secondary_dir=SKILLS_DIR,
    )
    mem_backend = FilesystemBackend(
        root_dir=memory_dir,
        virtual_mode=True,
    )
    return CompositeBackend(
        default=ws_backend,
        routes={
            "/skills/": sk_backend,
            "/memories/": mem_backend,
        },
    )


def _get_default_middleware(
    *,
    for_async_subagent: bool = False,
    workspace_dir: str | Path | None = None,
    cfg=None,
    chat_model=None,
    memory_source_agent: str = "EvoScientist",
):
    """Build the default middleware list.

    Args:
        for_async_subagent: When True, omit middleware that would deadlock a
            deployed async sub-agent. Specifically: ``AskUserMiddleware`` uses
            ``interrupt()`` to pause the graph waiting for a user reply, but
            async sub-agents run in the ``langgraph dev`` subprocess where
            the parent only holds a ``task_id`` and has no UI path to surface
            (or resume) an interrupt — the sub-agent would hang forever the
            first time it called ``ask_user``. This mirrors the same reason
            ``subagents/_factory.py`` deliberately skips ``interrupt_on=`` on
            the deepagents level. Defaults to False (full middleware list)
            for the CLI's in-process agent.
        cfg: Explicit config to use instead of the cached ``_config``.
        chat_model: Explicit model to bind instead of ``_ensure_chat_model()``
            (avoids writing module globals on the pure path).
        memory_source_agent: Attribution name for profile/observation writes.
            Async sub-agent factories pass their deployed agent name here.
    """
    from .middleware import (
        ConfigurableModelMiddleware,
        ContextOverflowMapperMiddleware,
        MemoryLifecycleRole,
        ModelFallbackMiddleware,
        ToolErrorHandlerMiddleware,
        create_code_interpreter_middleware,
        create_context_editing_middleware,
        create_memory_lifecycle_middleware,
        create_memory_middleware,
        create_runtime_context_middleware,
        create_tool_selector_middleware,
        load_fallback_chain,
    )

    cfg = cfg if cfg is not None else _ensure_config()
    if cfg.model_fallbacks:
        load_fallback_chain(cfg.model_fallbacks)
    model = chat_model if chat_model is not None else _ensure_chat_model()
    memory_dir = str(_paths_mod.MEMORIES_DIR)
    source_type = (
        MemorySourceType.SUBAGENT if for_async_subagent else MemorySourceType.TURN
    )
    memory_controls = MemoryControls.from_config(cfg)
    worker_target = (
        MemoryObservationTarget.SUBAGENT_WORKER
        if for_async_subagent
        else MemoryObservationTarget.TURN_WORKER
    )
    # ``ConfigurableModelMiddleware`` is placed first so it wraps
    # ``ModelFallbackMiddleware``: a configurable.model override sets the
    # PRIMARY model only, leaving the fallback chain free to try its own
    # alternatives instead of re-overriding every retry to the same model.
    memory_middleware = create_memory_middleware(
        memory_dir,
        workspace_dir=workspace_dir,
        source_type=source_type,
        source_agent=memory_source_agent,
        enable_profile_memory=memory_controls.profile_enabled,
        enable_observation_memory=memory_controls.observations_enabled,
        enable_observation_tool=memory_controls.observation_tool_enabled(
            MemoryObservationTarget.AGENT
        ),
    )
    # Main-agent tool selection may use the auxiliary model; async sub-agents
    # keep the main model (they do real work, not a one-off helper call).
    # context_editing stays on the main model — its model only sizes the
    # context-window trigger for the main agent's own history.
    if for_async_subagent:
        tool_selector_model = model
    elif chat_model is None:
        tool_selector_model = _ensure_auxiliary_chat_model()
    else:
        aux_model = cfg.auxiliary_model or cfg.model
        aux_provider = cfg.auxiliary_provider or cfg.provider
        if (aux_model, aux_provider) == (cfg.model, cfg.provider):
            tool_selector_model = model
        else:
            from .llm import get_chat_model

            tool_selector_model = get_chat_model(model=aux_model, provider=aux_provider)
    mw = [
        ConfigurableModelMiddleware(),
        create_context_editing_middleware(model),
        ModelFallbackMiddleware(),
        ContextOverflowMapperMiddleware(),
        ToolErrorHandlerMiddleware(),
        *create_tool_selector_middleware(model=tool_selector_model),
        # Interpreter prompt must land before runtime/memory context, so this
        # middleware sits ahead of runtime_context in the stack.
        create_code_interpreter_middleware(
            timeout=cfg.code_interpreter_timeout,
            max_result_chars=cfg.code_interpreter_max_result_chars,
        ),
        create_runtime_context_middleware(),
    ]
    if memory_controls.memory_enabled:
        mw.append(memory_middleware)
    if memory_controls.worker_needed(worker_target):
        mw.append(
            create_memory_lifecycle_middleware(
                memory_dir,
                workspace_dir=workspace_dir,
                project_id=memory_middleware.project_id,
                role=(
                    MemoryLifecycleRole.SUBAGENT
                    if for_async_subagent
                    else MemoryLifecycleRole.TURN
                ),
                source_agent=memory_source_agent,
            )
        )

    if cfg.enable_ask_user and not cfg.auto_mode and not for_async_subagent:
        from .middleware.ask_user import AskUserMiddleware

        mw.insert(0, AskUserMiddleware())

    # Background-process tools (run_in_background / check_process / stop_process /
    # list_processes) — main agent only. Async sub-agents run on langgraph-dev and
    # must not spawn local OS processes.
    if not for_async_subagent:
        from .middleware.background import BackgroundExecutionMiddleware

        mw.append(BackgroundExecutionMiddleware())

    return mw


def _get_default_agent():
    """Build the default agent (no checkpointer) on first access.

    MCP loading depends on which subprocess mode (if any) this agent is
    being built in. ``langgraph_dev.manager.start_langgraph_dev`` injects
    ``EVOSCIENTIST_DEPLOY_MODE`` into the subprocess with one of two values:

    - ``EVOSCIENTIST_DEPLOY_MODE=full`` — set by ``EvoSci deploy``. The
      subprocess is the *primary* programmatic entry point (Python scripts,
      Jupyter, integration tests via ``langgraph_sdk``), so it needs the full
      configuration: **load MCP**, and ``_ASYNC_SUBAGENTS_AVAILABLE`` flips on
      at module load so async sub-agents self-loop through this same
      langgraph dev server.

    - ``EVOSCIENTIST_DEPLOY_MODE=stripped`` — set by ``EvoSci`` / ``EvoSci
      serve``. The CLI's in-process main agent already loaded MCP; this
      subprocess only services async sub-agent self-loops, so **skip MCP**
      to avoid running a second copy of the same servers.

    Plain ``from EvoScientist import EvoScientist_agent`` (env var unset)
    loads MCP. Async sub-agents stay disabled in that case because there is
    no langgraph dev server to self-loop into.
    """
    global _EvoScientist_agent
    if _EvoScientist_agent is None:
        from deepagents import create_deep_agent

        cfg = _ensure_config()
        be = _get_default_backend()
        mw = _get_default_middleware()

        # HITL on main agent only (mirrors create_cli_agent). Use middleware,
        # not interrupt_on= kwarg — the kwarg propagates to every subagent and
        # breaks parallel execute calls (multi-pending-interrupt LangGraph
        # error). See PR #202.
        if not cfg.auto_approve:
            mw.append(
                HumanInTheLoopMiddleware(
                    interrupt_on={"execute": True, "run_in_background": True}
                )
            )

        if os.environ.get("EVOSCIENTIST_DEPLOY_MODE", "").lower() == "stripped":
            kwargs = _build_base_kwargs(
                be,
                mw,
                workspace_dir=str(_paths_mod.WORKSPACE_ROOT),
            )
        else:
            kwargs = load_mcp_and_build_kwargs(
                be,
                mw,
                workspace_dir=str(_paths_mod.WORKSPACE_ROOT),
            )

        _EvoScientist_agent = create_deep_agent(
            **kwargs,
        ).with_config({"recursion_limit": cfg.recursion_limit})
    return _EvoScientist_agent


def __getattr__(name: str):
    if name == "EvoScientist_agent":
        return _get_default_agent()
    # Backward compat for module-level names
    if name == "chat_model":
        return _ensure_chat_model()
    if name == "SYSTEM_PROMPT":
        return _configured_system_prompt(_ensure_config())
    if name == "backend":
        return _get_default_backend()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# =============================================================================
# CLI agent factory
# =============================================================================


def create_cli_agent(
    workspace_dir: str | None = None,
    checkpointer=None,
    config=None,
    chat_model=None,
    *,
    on_mcp_progress=None,
) -> "CompiledStateGraph":
    """Create agent with checkpointer for CLI multi-turn support.

    A fresh backend is constructed on every call using the current
    ``paths.WORKSPACE_ROOT`` (or the explicit *workspace_dir*), so
    runtime ``set_workspace_root()`` changes are always respected.

    **Pure path:** when *both* ``config`` and ``chat_model`` are explicit, this
    writes none of the cached config/model module globals (``_config``,
    ``_chat_model``, ``_chat_model_key``, ``_EvoScientist_agent``) — the agent
    is built purely from the passed-in locals.  The caller commits the switch
    on success via ``set_active_config`` / ``set_chat_model_instance`` (see
    ``/model``).  Otherwise the existing module-global path runs (langgraph
    dev, notebooks, and CLI startup, which pass ``config=`` only).

    Args:
        workspace_dir: Per-session workspace directory. If ``None``,
            defaults to the current ``paths.WORKSPACE_ROOT``.
        checkpointer: Optional LangGraph checkpointer. If ``None``,
            falls back to ``InMemorySaver`` (non-persistent).
        config: Optional pre-loaded ``EvoScientistConfig``.  If ``None``,
            loads from file/env/defaults.  Passing this avoids double
            loading when the CLI has already loaded config.
        chat_model: Optional pre-built chat model.  Only triggers the pure
            path when ``config`` is also explicit; otherwise it is ignored in
            favor of the ``_ensure_chat_model()`` fallback.
    """
    import os as _os

    from deepagents import create_deep_agent
    from deepagents.backends import CompositeBackend, FilesystemBackend

    from . import paths as _paths
    from .backends import CustomSandboxBackend, MergedSkillsBackend

    # Pure path only when BOTH config and chat_model are explicit: build from
    # locals and write no module globals. Otherwise keep the legacy
    # global-writing behavior — callers that pass config= only (CLI startup,
    # langgraph dev) rely on it to seat the active config/model.
    if config is not None and chat_model is not None:
        cfg = config
        _apply_env_from_config(cfg)
    else:
        cfg = _ensure_config(config)
        chat_model = None

    if checkpointer is None:
        from langgraph.checkpoint.memory import InMemorySaver

        checkpointer = InMemorySaver()

    # When no explicit workspace_dir is provided, apply config.default_workdir
    # as a fallback.  This covers direct callers (notebooks, iMessage server)
    # that never call set_workspace_root() themselves.  CLI callers always
    # pass workspace_dir explicitly, so their --workdir is never overwritten.
    if workspace_dir is None:
        if cfg.default_workdir:
            set_workspace_root(
                _os.path.abspath(_os.path.expanduser(cfg.default_workdir))
            )
        workspace_dir = str(_paths.WORKSPACE_ROOT)

    # Read paths dynamically so runtime set_workspace_root() changes are picked up
    _mem_dir = str(_paths.MEMORIES_DIR)
    _usr_skills_dir = str(_paths.USER_SKILLS_DIR)
    _global_skills_dir = str(_paths.GLOBAL_SKILLS_DIR)

    # Always construct fresh backends from current paths (avoids stale
    # module-level backend when workspace root changed at runtime).
    set_active_workspace(workspace_dir)
    ws_backend = CustomSandboxBackend(
        root_dir=workspace_dir,
        virtual_mode=True,
        timeout=cfg.sandbox_execute_timeout,
        dangerous=cfg.dangerous_mode,
    )
    sk_backend = MergedSkillsBackend(
        primary_dir=_usr_skills_dir,
        global_dir=_global_skills_dir,
        secondary_dir=SKILLS_DIR,
    )
    mem_backend = FilesystemBackend(
        root_dir=_mem_dir,
        virtual_mode=True,
    )
    be = CompositeBackend(
        default=ws_backend,
        routes={
            "/skills/": sk_backend,
            "/memories/": mem_backend,
        },
    )

    # Delegate middleware construction to the single source of truth so the
    # CLI agent never drifts from the default chain. Anything CLI-specific
    # (e.g. ``HumanInTheLoopMiddleware``) is appended below.
    mw: list[AgentMiddleware] = _get_default_middleware(
        workspace_dir=workspace_dir, cfg=cfg, chat_model=chat_model
    )

    # HITL on main agent only — passing `interrupt_on=` to create_deep_agent
    # would propagate it to every subagent, breaking parallel execute calls
    # (multi-pending-interrupt LangGraph error).
    if not cfg.auto_approve:
        mw.append(
            HumanInTheLoopMiddleware(
                interrupt_on={"execute": True, "run_in_background": True}
            )
        )

    # Re-load MCP tools from current config (picks up /mcp add changes)
    kwargs = load_mcp_and_build_kwargs(
        be,
        mw,
        on_mcp_progress=on_mcp_progress,
        cfg=cfg,
        chat_model=chat_model,
        workspace_dir=workspace_dir,
    )

    return create_deep_agent(
        **kwargs,
        checkpointer=checkpointer,
    ).with_config({"recursion_limit": cfg.recursion_limit})
