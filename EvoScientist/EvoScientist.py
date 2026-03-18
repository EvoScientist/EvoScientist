"""EvoScientist Agent graph construction.

This module defines the agent graph and its factory functions.  All heavy
initialization (deepagents, backends, LLM, middleware) is deferred to first
use so that importing this module is fast and non-agent CLI commands
(``EvoSci config list``, ``EvoSci onboard``) never pay the cost.

Usage:
    from EvoScientist import EvoScientist_agent

    # Notebook / programmatic usage
    for state in EvoScientist_agent.stream(
        {"messages": [HumanMessage(content="your question")]},
        config={"configurable": {"thread_id": "1"}},
        stream_mode="values",
    ):
        ...
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from .config import get_effective_config, apply_config_to_env
from .prompts import RESEARCHER_INSTRUCTIONS, get_system_prompt
from . import paths as _paths_mod
from .paths import set_active_workspace, set_workspace_root

# Suppress noisy warnings from deepagents skill loader (non-string frontmatter fields, etc.)
logging.getLogger("deepagents.middleware.skills").setLevel(logging.ERROR)

# =============================================================================
# Constants
# =============================================================================

SUBAGENTS_CONFIG = Path(__file__).parent / "subagent.yaml"
SKILLS_DIR = str(Path(__file__).parent / "skills")

# =============================================================================
# Lazy state — initialized on first use, not at import time
# =============================================================================

_config = None
_chat_model = None

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


def _ensure_config(config=None):
    """Return cached config.  If *config* is passed, cache and use it."""
    global _config
    if config is not None:
        _config = config
        apply_config_to_env(_config)
    if _config is None:
        _config = get_effective_config()
        apply_config_to_env(_config)
    return _config


def _ensure_chat_model():
    """Return cached chat model, creating it on first call."""
    global _chat_model
    if _chat_model is None:
        from .llm import get_chat_model

        cfg = _ensure_config()
        _chat_model = get_chat_model(model=cfg.model, provider=cfg.provider)
    return _chat_model


# =============================================================================
# MCP caching
# =============================================================================


def _mcp_config_signature() -> str:
    """Return a stable signature for the effective MCP config."""
    from .mcp.client import load_mcp_config

    cfg = load_mcp_config()
    if not cfg:
        return ""
    try:
        return json.dumps(cfg, sort_keys=True, ensure_ascii=True)
    except TypeError:
        # Fallback for non-JSON-serializable values (should be rare)
        return repr(cfg)


def _load_mcp_tools_cached() -> dict[str, list]:
    """Load MCP tools with config-aware caching."""
    global _MCP_TOOLS_CACHE_KEY, _MCP_TOOLS_CACHE_VALUE

    from .mcp import load_mcp_tools

    cfg_key = _mcp_config_signature()
    if not cfg_key:
        _MCP_TOOLS_CACHE_KEY = ""
        _MCP_TOOLS_CACHE_VALUE = {}
        return {}

    if _MCP_TOOLS_CACHE_KEY == cfg_key and _MCP_TOOLS_CACHE_VALUE is not None:
        return {k: list(v) for k, v in _MCP_TOOLS_CACHE_VALUE.items()}

    loaded = load_mcp_tools()
    _MCP_TOOLS_CACHE_KEY = cfg_key
    _MCP_TOOLS_CACHE_VALUE = {k: list(v) for k, v in loaded.items()}
    return {k: list(v) for k, v in loaded.items()}


# =============================================================================
# Agent construction helpers
# =============================================================================


def _inject_subagent_middleware(subs: list[dict]) -> None:
    """Ensure every subagent gets ToolErrorHandlerMiddleware.

    Without this, subagent tool errors are caught by LangGraph's default
    ToolNode handler which produces terse messages without tracebacks or
    retry guidance — reducing the subagent's ability to self-recover.
    """
    from .middleware import ToolErrorHandlerMiddleware

    for sa in subs:
        sa.setdefault("middleware", []).append(ToolErrorHandlerMiddleware())


def _build_prompt_refs() -> dict:
    """Build prompt references with the current date (not frozen at import)."""
    return {
        "RESEARCHER_INSTRUCTIONS": RESEARCHER_INSTRUCTIONS.format(
            date=datetime.now().strftime("%Y-%m-%d"),
        ),
    }


def _build_base_kwargs(base_backend, base_middleware):
    """Build agent kwargs *without* MCP (fast, no subprocess spawning)."""
    from .utils import load_subagents
    from .tools import tavily_search, think_tool, skill_manager

    tool_registry = {"think_tool": think_tool, "tavily_search": tavily_search}
    base_tools = [think_tool, skill_manager]

    subs = load_subagents(
        SUBAGENTS_CONFIG,
        tool_registry=tool_registry,
        prompt_refs=_build_prompt_refs(),
    )
    _inject_subagent_middleware(subs)
    return dict(
        name="EvoScientist",
        model=_ensure_chat_model(),
        tools=list(base_tools),
        backend=base_backend,
        subagents=subs,
        middleware=base_middleware,
        system_prompt=get_system_prompt(),
        skills=["/skills/"],
    )


def load_mcp_and_build_kwargs(base_backend, base_middleware):
    """Load MCP tools (cached by config) and build agent kwargs.

    Re-connects to MCP servers only when the effective MCP config changes.
    Falls back to base kwargs if no MCP configured.
    """
    from .utils import load_subagents
    from .tools import tavily_search, think_tool, skill_manager

    mcp_by_agent = _load_mcp_tools_cached()
    if not mcp_by_agent:
        return _build_base_kwargs(base_backend, base_middleware)

    tool_registry = {"think_tool": think_tool, "tavily_search": tavily_search}
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
        prompt_refs=_build_prompt_refs(),
    )

    _inject_subagent_middleware(subs)

    # Inject MCP tools into subagents by name
    for sa in subs:
        if sa_tools := mcp_by_agent.get(sa["name"], []):
            sa.setdefault("tools", []).extend(sa_tools)

    return dict(
        name="EvoScientist",
        model=_ensure_chat_model(),
        tools=base_tools + mcp_main,
        backend=base_backend,
        subagents=subs,
        middleware=base_middleware,
        system_prompt=get_system_prompt(),
        skills=["/skills/"],
    )


# =============================================================================
# Default agent (langgraph dev / notebooks)
# =============================================================================


def _get_default_backend():
    """Build the default composite backend from current paths."""
    from deepagents.backends import FilesystemBackend, CompositeBackend
    from .backends import CustomSandboxBackend, MergedReadOnlyBackend

    workspace_dir = str(_paths_mod.WORKSPACE_ROOT)
    set_active_workspace(workspace_dir)
    memory_dir = str(_paths_mod.MEMORY_DIR)
    user_skills_dir = str(_paths_mod.USER_SKILLS_DIR)

    ws_backend = CustomSandboxBackend(
        root_dir=workspace_dir,
        virtual_mode=True,
        timeout=300,
    )
    sk_backend = MergedReadOnlyBackend(
        primary_dir=user_skills_dir,
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
            "/memory/": mem_backend,
        },
    )


def _get_default_middleware():
    """Build the default middleware list."""
    from .middleware import create_memory_middleware, ToolErrorHandlerMiddleware

    cfg = _ensure_config()
    memory_dir = str(_paths_mod.MEMORY_DIR)
    mw = [
        ToolErrorHandlerMiddleware(),
        create_memory_middleware(memory_dir, extraction_model=_ensure_chat_model()),
    ]
    if cfg.enable_ask_user and not cfg.auto_approve:
        from .middleware.ask_user import AskUserMiddleware

        mw.insert(0, AskUserMiddleware())
    return mw


def _get_default_agent():
    """Build the default agent (with MCP, no checkpointer) on first access."""
    global _EvoScientist_agent
    if _EvoScientist_agent is None:
        from deepagents import create_deep_agent

        be = _get_default_backend()
        mw = _get_default_middleware()
        kwargs = load_mcp_and_build_kwargs(be, mw)
        _EvoScientist_agent = create_deep_agent(**kwargs).with_config(
            {"recursion_limit": 1000}
        )
    return _EvoScientist_agent


def __getattr__(name: str):
    if name == "EvoScientist_agent":
        return _get_default_agent()
    # Backward compat for module-level names
    if name == "chat_model":
        return _ensure_chat_model()
    if name == "SYSTEM_PROMPT":
        return get_system_prompt()
    if name == "backend":
        return _get_default_backend()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# =============================================================================
# CLI agent factory
# =============================================================================


def create_cli_agent(workspace_dir: str | None = None, checkpointer=None, config=None):
    """Create agent with checkpointer for CLI multi-turn support.

    A fresh backend is constructed on every call using the current
    ``paths.WORKSPACE_ROOT`` (or the explicit *workspace_dir*), so
    runtime ``set_workspace_root()`` changes are always respected.

    Args:
        workspace_dir: Per-session workspace directory. If ``None``,
            defaults to the current ``paths.WORKSPACE_ROOT``.
        checkpointer: Optional LangGraph checkpointer. If ``None``,
            falls back to ``InMemorySaver`` (non-persistent).
        config: Optional pre-loaded ``EvoScientistConfig``.  If ``None``,
            loads from file/env/defaults.  Passing this avoids double
            loading when the CLI has already loaded config.
    """
    import os as _os

    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend, CompositeBackend
    from .backends import CustomSandboxBackend, MergedReadOnlyBackend
    from .middleware import create_memory_middleware, ToolErrorHandlerMiddleware
    from . import paths as _paths

    cfg = _ensure_config(config)

    if checkpointer is None:
        from langgraph.checkpoint.memory import InMemorySaver  # type: ignore[import-untyped]

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
    _mem_dir = str(_paths.MEMORY_DIR)
    _usr_skills_dir = str(_paths.USER_SKILLS_DIR)

    # Always construct fresh backends from current paths (avoids stale
    # module-level backend when workspace root changed at runtime).
    set_active_workspace(workspace_dir)
    ws_backend = CustomSandboxBackend(
        root_dir=workspace_dir,
        virtual_mode=True,
        timeout=300,
    )
    sk_backend = MergedReadOnlyBackend(
        primary_dir=_usr_skills_dir,
        secondary_dir=SKILLS_DIR,
    )
    # Memory always uses SHARED directory (not per-session) for cross-session persistence
    mem_backend = FilesystemBackend(
        root_dir=_mem_dir,
        virtual_mode=True,
    )
    be = CompositeBackend(
        default=ws_backend,
        routes={
            "/skills/": sk_backend,
            "/memory/": mem_backend,
        },
    )

    mw = [
        ToolErrorHandlerMiddleware(),
        create_memory_middleware(_mem_dir, extraction_model=_ensure_chat_model()),
    ]
    if cfg.enable_ask_user and not cfg.auto_approve:
        from .middleware.ask_user import AskUserMiddleware

        mw.insert(0, AskUserMiddleware())

    # Re-load MCP tools from current config (picks up /mcp add changes)
    kwargs = load_mcp_and_build_kwargs(be, mw)

    # HITL: gate shell execution for user approval
    _interrupt_on: dict[str, bool] | None = None
    if not cfg.auto_approve:
        _interrupt_on = {"execute": True}

    return create_deep_agent(
        **kwargs,
        checkpointer=checkpointer,
        interrupt_on=_interrupt_on,
    ).with_config({"recursion_limit": 1000})
