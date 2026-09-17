"""EvoScientist Windows desktop shell.

A thin pywebview window that hosts the *existing* WebUI. It owns only the
application window, startup progress/error display, and process lifecycle; all
business functionality stays in the WebUI. Process startup/health/stop is
delegated to the shell-agnostic launcher core
(:mod:`EvoScientist.deploy.launcher`) — the desktop shell is just its first
in-process consumer.

Entry point: ``python -m EvoScientist.desktop`` (the installed shortcut runs a
windowed Python that imports :func:`EvoScientist.desktop.shell.run_desktop`).
"""
