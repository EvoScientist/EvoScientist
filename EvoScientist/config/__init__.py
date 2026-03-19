"""Configuration package for EvoScientist.

Re-exports all public symbols from settings and onboard submodules
so that existing ``from EvoScientist.config import X`` imports continue
to work without modification.

The onboard module is loaded lazily because it pulls in heavy dependencies
(langchain, llm) that are not needed for normal config operations.
"""

from .settings import (
    EvoScientistConfig,
    apply_config_to_env,
    get_config_dir,
    get_config_path,
    get_config_value,
    get_effective_config,
    list_config,
    load_config,
    reset_config,
    save_config,
    set_config_value,
)

__all__ = [
    "EvoScientistConfig",
    "apply_config_to_env",
    # settings
    "get_config_dir",
    "get_config_path",
    "get_config_value",
    "get_effective_config",
    "list_config",
    "load_config",
    "reset_config",
    # onboard (lazy)
    "run_onboard",
    "save_config",
    "set_config_value",
]


def __getattr__(name: str):
    if name == "run_onboard":
        from .onboard import run_onboard

        return run_onboard
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
