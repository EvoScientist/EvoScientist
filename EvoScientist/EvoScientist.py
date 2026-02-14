"""EvoScientist Agent graph construction.

This module creates and exports the compiled agent graph.
Usage:
    from EvoScientist import agent

    # Notebook / programmatic usage
    for state in agent.stream(
        {"messages": [HumanMessage(content="your question")]},
        config={"configurable": {"thread_id": "1"}},
        stream_mode="values",
    ):
        ...
"""

from datetime import datetime
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend, CompositeBackend

from .backends import CustomSandboxBackend, MergedReadOnlyBackend
from .config import get_effective_config, apply_config_to_env
from .llm import get_chat_model
from .mcp import load_mcp_tools
from .middleware import create_memory_middleware, create_skills_middleware
from .prompts import RESEARCHER_INSTRUCTIONS, get_system_prompt
from .utils import load_subagents
from .tools import tavily_search, think_tool, skill_manager, view_image
from .paths import (
    ensure_dirs,
    default_workspace_dir,
    set_active_workspace,
    MEMORY_DIR as _MEMORY_DIR_PATH,
    USER_SKILLS_DIR as _USER_SKILLS_DIR_PATH,
)

# =============================================================================
# Configuration
# =============================================================================

# Load configuration from file/env/defaults
_config = get_effective_config()
apply_config_to_env(_config)

# Research limits (from config)
MAX_CONCURRENT = _config.max_concurrent
MAX_ITERATIONS = _config.max_iterations

# Workspace settings
ensure_dirs()
WORKSPACE_DIR = str(default_workspace_dir())
set_active_workspace(WORKSPACE_DIR)
MEMORY_DIR = str(_MEMORY_DIR_PATH)  # Shared across sessions (not per-session)
SKILLS_DIR = str(Path(__file__).parent / "skills")
USER_SKILLS_DIR = str(_USER_SKILLS_DIR_PATH)
SUBAGENTS_CONFIG = Path(__file__).parent / "subagent.yaml"

# =============================================================================
# Initialization
# =============================================================================

# Get current date
current_date = datetime.now().strftime("%Y-%m-%d")

# Generate system prompt with limits
SYSTEM_PROMPT = get_system_prompt(
    max_concurrent=MAX_CONCURRENT,
    max_iterations=MAX_ITERATIONS,
)

# Initialize chat model using the LLM module (respects config settings)
chat_model = get_chat_model(
    model=_config.model,
    provider=_config.provider,
)

# Initialize workspace backend
_workspace_backend = CustomSandboxBackend(
    root_dir=WORKSPACE_DIR,
    virtual_mode=True,
    timeout=300,
)

# Skills backend: merge user-installed (./skills/) and system (package) skills
_skills_backend = MergedReadOnlyBackend(
    primary_dir=USER_SKILLS_DIR,                        # user-installed, takes priority
    secondary_dir=SKILLS_DIR,                           # package built-in, fallback
)

# Memory backend: persistent filesystem for long-term memory (shared across sessions)
_memory_backend = FilesystemBackend(
    root_dir=MEMORY_DIR,
    virtual_mode=True,
)

# Composite backend: workspace as default, skills and memory mounted
backend = CompositeBackend(
    default=_workspace_backend,
    routes={
        "/skills/": _skills_backend,
        "/memory/": _memory_backend,
    },
)

tool_registry = {
    "think_tool": think_tool,
    "tavily_search": tavily_search,
    "view_image": view_image,
}

# Base tools that every agent variant gets (before MCP)
BASE_TOOLS = [think_tool, skill_manager, view_image]


def _build_base_kwargs(base_backend, base_middleware):
    """Build agent kwargs *without* MCP (fast, no subprocess spawning)."""
    subs = load_subagents(
        SUBAGENTS_CONFIG,
        tool_registry=tool_registry,
        prompt_refs=prompt_refs,
    )
    return dict(
        name="EvoScientist",
        model=chat_model,
        tools=list(BASE_TOOLS),
        backend=base_backend,
        subagents=subs,
        middleware=base_middleware,
        system_prompt=SYSTEM_PROMPT,
        skills=["/skills/"],
    )


def load_mcp_and_build_kwargs(base_backend, base_middleware):
    """(Re-)load MCP tools and build agent kwargs.

    Called on every ``create_cli_agent()`` call so that ``/new`` picks up
    MCP config changes. Falls back to base kwargs if no MCP configured.
    """
    mcp_by_agent = load_mcp_tools()
    if not mcp_by_agent:
        return _build_base_kwargs(base_backend, base_middleware)

    # Fresh tool registry — start from base tools + MCP tools
    registry = dict(tool_registry)
    for tools in mcp_by_agent.values():
        for t in tools:
            registry[t.name] = t

    mcp_main = mcp_by_agent.pop("main", [])

    subs = load_subagents(
        SUBAGENTS_CONFIG,
        tool_registry=registry,
        prompt_refs=prompt_refs,
    )

    # Inject MCP tools into subagents by name
    for sa in subs:
        if sa_tools := mcp_by_agent.get(sa["name"], []):
            sa.setdefault("tools", []).extend(sa_tools)

    return dict(
        name="EvoScientist",
        model=chat_model,
        tools=BASE_TOOLS + mcp_main,
        backend=base_backend,
        subagents=subs,
        middleware=base_middleware,
        system_prompt=SYSTEM_PROMPT,
        skills=["/skills/"],
    )


prompt_refs = {
    "RESEARCHER_INSTRUCTIONS": RESEARCHER_INSTRUCTIONS.format(date=current_date),
}

base_middleware = [
    create_memory_middleware(MEMORY_DIR, extraction_model=chat_model),
    create_skills_middleware(backend),
]

# Default agent (no checkpointer) — used by langgraph dev / LangSmith / notebooks.
# Lazily constructed on first access so MCP tools are included without
# spawning subprocesses at import time.
_EvoScientist_agent = None


def _get_default_agent():
    """Build the default agent (with MCP, no checkpointer) on first access."""
    global _EvoScientist_agent
    if _EvoScientist_agent is None:
        kwargs = load_mcp_and_build_kwargs(backend, base_middleware)
        _EvoScientist_agent = create_deep_agent(**kwargs).with_config(
            {"recursion_limit": 500}
        )
    return _EvoScientist_agent


def __getattr__(name: str):
    if name == "EvoScientist_agent":
        return _get_default_agent()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def create_cli_agent(workspace_dir: str | None = None, checkpointer=None):
    """Create agent with checkpointer for CLI multi-turn support.

    Args:
        workspace_dir: Optional per-session workspace directory. If provided,
            creates a fresh backend rooted at this path. If None, uses the
            module-level default backend (./workspace).
        checkpointer: Optional LangGraph checkpointer. If None, falls back
            to ``InMemorySaver`` (non-persistent).
    """
    if checkpointer is None:
        from langgraph.checkpoint.memory import InMemorySaver  # type: ignore[import-untyped]
        checkpointer = InMemorySaver()

    if workspace_dir:
        set_active_workspace(workspace_dir)
        ws_backend = CustomSandboxBackend(
            root_dir=workspace_dir,
            virtual_mode=True,
            timeout=300,
        )
        sk_backend = MergedReadOnlyBackend(
            primary_dir=USER_SKILLS_DIR,
            secondary_dir=SKILLS_DIR,
        )
        # Memory always uses SHARED directory (not per-session) for cross-session persistence
        mem_backend = FilesystemBackend(
            root_dir=MEMORY_DIR,
            virtual_mode=True,
        )
        be = CompositeBackend(
            default=ws_backend,
            routes={
                "/skills/": sk_backend,
                "/memory/": mem_backend,
            },
        )
    else:
        be = backend

    mw = [
        create_memory_middleware(MEMORY_DIR, extraction_model=chat_model),
        create_skills_middleware(be),
    ]

    # Re-load MCP tools from current config (picks up /mcp add changes)
    kwargs = load_mcp_and_build_kwargs(be, mw)

    return create_deep_agent(
        **kwargs,
        checkpointer=checkpointer,
    ).with_config({"recursion_limit": 500})
