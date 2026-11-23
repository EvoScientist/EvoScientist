"""Tests for ``gateway/types``."""

from __future__ import annotations

from types import SimpleNamespace

from EvoScientist.gateway.types import runs_in_process


def _server_gateway():
    from EvoScientist.gateway.server import LangGraphServerGateway

    return object.__new__(LangGraphServerGateway)


def test_runs_in_process_goes_by_the_gateway_that_executes():
    assert runs_in_process(object()) is True
    assert runs_in_process(_server_gateway()) is False
    # A routing gateway is judged by the one it executes on.
    assert runs_in_process(SimpleNamespace(execute_gateway=_server_gateway())) is False
    assert runs_in_process(SimpleNamespace(execute_gateway=object())) is True
