"""Record ``config.yaml`` before the served graphs read it.

The graph modules import this first, so the settings API can report which
saved settings changed since the server built its graphs (``restart_fields``).
"""

from EvoScientist.langgraph_dev.settings_api import capture_restart_baseline

capture_restart_baseline()
