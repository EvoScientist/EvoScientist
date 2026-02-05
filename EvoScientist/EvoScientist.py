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
from .middleware import create_skills_middleware, create_memory_middleware
from .prompts import RESEARCHER_INSTRUCTIONS, get_system_prompt
from .utils import load_subagents
from .tools import tavily_search, think_tool, skill_manager
from .paths import (
    ensure_dirs,
    default_workspace_dir,
    MEMORY_DIR as _MEMORY_DIR_PATH,
    USER_SKILLS_DIR as _USER_SKILLS_DIR_PATH,
)

# =============================================================================
# Configuration
# =============================================================================

# Load configuration from file/env/defaults
_config = get_effective_config()
apply_config_to_env(_config)

# Backend mode: "sandbox" (with execute) or "filesystem" (read/write only)
BACKEND_MODE = "sandbox"

# Research limits (from config)
MAX_CONCURRENT = _config.max_concurrent
MAX_ITERATIONS = _config.max_iterations

# Workspace settings
ensure_dirs()
WORKSPACE_DIR = str(default_workspace_dir())
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

# Initialize workspace backend based on mode
if BACKEND_MODE == "sandbox":
    _workspace_backend = CustomSandboxBackend(
        root_dir=WORKSPACE_DIR,
        virtual_mode=True,
        timeout=300,
    )
else:
    _workspace_backend = FilesystemBackend(
        root_dir=WORKSPACE_DIR,
        virtual_mode=True,
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
}

prompt_refs = {
    "RESEARCHER_INSTRUCTIONS": RESEARCHER_INSTRUCTIONS.format(date=current_date),
}

subagents = load_subagents(
    SUBAGENTS_CONFIG,
    tool_registry=tool_registry,
    prompt_refs=prompt_refs,
)

# Shared kwargs for agent creation
_AGENT_KWARGS = dict(
    name="EvoScientist",
    model=chat_model,
    tools=[think_tool, skill_manager],
    backend=backend,
    subagents=subagents,
    middleware=[
        create_memory_middleware(MEMORY_DIR, extraction_model=chat_model),
        create_skills_middleware(SKILLS_DIR, user_skills_dir=USER_SKILLS_DIR),
    ],
    system_prompt=SYSTEM_PROMPT,
)

# Default agent (no checkpointer) — used by langgraph dev / LangSmith / notebooks
EvoScientist_agent = create_deep_agent(**_AGENT_KWARGS).with_config({"recursion_limit": 500})


def create_cli_agent(workspace_dir: str | None = None):
    """Create agent with InMemorySaver checkpointer for CLI multi-turn support.

    Args:
        workspace_dir: Optional per-session workspace directory. If provided,
            creates a fresh backend rooted at this path. If None, uses the
            module-level default backend (./workspace).
    """
    from langgraph.checkpoint.memory import InMemorySaver  # type: ignore[import-untyped]

    if workspace_dir:
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
        mw = [
            create_memory_middleware(MEMORY_DIR, extraction_model=chat_model),
            create_skills_middleware(SKILLS_DIR, user_skills_dir=USER_SKILLS_DIR),
        ]
        kwargs = dict(
            _AGENT_KWARGS,
            backend=be,
            middleware=mw,
        )
    else:
        kwargs = dict(_AGENT_KWARGS)

    return create_deep_agent(
        **kwargs,
        checkpointer=InMemorySaver(),
    ).with_config({"recursion_limit": 500})
