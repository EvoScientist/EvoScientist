"""Onboarding package.

The wizard's only package-level public entry point is :func:`run_onboard`.
Everything else lives in submodules — import directly from them:

- :mod:`EvoScientist.config.onboard.wizard` — orchestrator, ``run_onboard``,
  ``STEPS``, ``render_progress``
- :mod:`EvoScientist.config.onboard.steps` — per-step functions
- :mod:`EvoScientist.config.onboard.channels` — channel selection + setup
- :mod:`EvoScientist.config.onboard.helpers` — API-key prompt, ccproxy,
  npx/node, LaTeX, iMessage helpers
- :mod:`EvoScientist.config.onboard.style` — Rich styles + ``_checkbox_ask``
- :mod:`EvoScientist.config.onboard.validators` — input validators
- :mod:`EvoScientist.config.onboard.prompter` — ``NonInteractivePrompter``
  (CLI-answer container) + ``select_navigation_active`` / ``GoBack`` for
  keyboard nav
- :mod:`EvoScientist.config.onboard.constants` — canonical valid-value sets

This module used to re-export every symbol from every submodule for
backward compat during the initial refactor; those re-exports have since
been removed to keep the public surface narrow. New code should always
import from the submodule that owns the symbol; test code should use
``patch("EvoScientist.config.onboard.<submodule>.<name>")`` paths.
"""

from __future__ import annotations

# Sole package-level public entry. ``EvoScientist.config`` re-exports this
# (via lazy ``__getattr__``) so ``from EvoScientist.config import
# run_onboard`` keeps working — that import path is used by the CLI and is
# the only documented external API.
from .wizard import run_onboard

__all__ = ["run_onboard"]
