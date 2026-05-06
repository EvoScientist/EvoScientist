"""Deployed graph entry for the main EvoScientist agent.

The main ``EvoScientist_agent`` is exposed via ``__getattr__`` lazy loading
in ``EvoScientist/EvoScientist.py`` so it doesn't construct on plain
``import EvoScientist``. ``langgraph dev`` 's symbol resolver inspects
module attributes directly and doesn't trigger ``__getattr__``, so we
re-export here to make it visible.
"""

from EvoScientist.EvoScientist import EvoScientist_agent

__all__ = ["EvoScientist_agent"]
