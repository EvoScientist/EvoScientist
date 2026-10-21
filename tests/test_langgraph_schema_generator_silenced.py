"""Regression tests for the langgraph_api SchemaGenerator silencing patch.

Reproducer: mounting our ``/api/models`` custom Starlette app makes
langgraph_api call ``update_openapi_spec`` at startup, which iterates
EVERY route (ours + upstream's). Endpoints whose docstrings aren't
valid YAML hit a warning + traceback in the deploy log — purely noise,
since the existing fallback path already produces a usable schema
entry. The patch keeps the fallback shape but silences the log spam.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def lgapi_utils(langgraph_api_env):
    """``langgraph_api.utils``, imported with the variables its config needs."""
    import langgraph_api.utils

    import EvoScientist.llm.patches as patches

    # Re-invoke the patch now that the variables are set. Earlier imports of
    # patches.py ran without them, so importing ``langgraph_api.utils``
    # inside the patch failed, its bare ``except`` swallowed that, and
    # module-level code never runs twice. The patch returns early once it
    # has landed.
    patches._patch_langgraph_schema_generator_silence_warnings()
    return langgraph_api.utils


class _FakeEndpoint:
    """Minimal Starlette-like endpoint info for the schema generator."""

    def __init__(self, path: str, method: str, func):
        self.path = path
        self.http_method = method
        self.func = func


class _DocstringFixture:
    """The kinds of docstrings the patched generator must handle."""

    @staticmethod
    def prose_with_colon():
        """Endpoint summary.

        Query params:
            id: The thing you want.
        """

    @staticmethod
    def valid_yaml():
        """
        summary: A valid YAML docstring.
        description: Stays structured.
        """

    @staticmethod
    def no_docstring():
        pass


def _generator(lgapi_utils):
    return lgapi_utils.SchemaGenerator(
        {"openapi": "3.1.0", "info": {"title": "test", "version": "0"}}
    )


def test_prose_docstring_no_longer_logs_warning(lgapi_utils):
    """The patched ``parse_docstring`` must silence upstream's structlog
    WARNING when ``yaml.safe_load`` fails on a prose docstring.

    Inverts the patch first to prove the fixture actually trips
    ``yaml.safe_load`` — without this baseline assertion the test would
    pass vacuously if the fixture stopped triggering the failure path
    (e.g. if upstream changed how docstrings are pre-processed).
    """
    from structlog.testing import capture_logs

    gen = _generator(lgapi_utils)
    endpoint = _FakeEndpoint("/x", "get", _DocstringFixture.prose_with_colon)
    gen.get_endpoints = lambda _routes: [endpoint]

    patched_parse = lgapi_utils.SchemaGenerator.parse_docstring
    # Phase 1: baseline. Drop the subclass override so MRO falls through
    # to Starlette's BaseSchemaGenerator.parse_docstring, which is what
    # production hits before our patch installs.
    del lgapi_utils.SchemaGenerator.parse_docstring
    try:
        with capture_logs() as baseline_records:
            gen.get_schema([])
    finally:
        lgapi_utils.SchemaGenerator.parse_docstring = patched_parse

    baseline_warnings = [r for r in baseline_records if r.get("log_level") == "warning"]
    assert any(
        "Unable to parse docstring" in r.get("event", "") for r in baseline_warnings
    ), "fixture no longer trips parse_docstring — test would pass vacuously"

    # Phase 2: with the patch reinstated, the same call must emit no
    # warning records.
    with capture_logs() as patched_records:
        schema = gen.get_schema([])

    assert [r for r in patched_records if r.get("log_level") == "warning"] == []

    # Schema still has the fallback shape — fixture's prose becomes the
    # description verbatim (with leading/trailing whitespace from the
    # docstring preserved by upstream's fallback path).
    entry = schema["paths"]["/x"]["get"]
    assert "description" in entry
    assert "Query params" in entry["description"]


def test_valid_yaml_docstring_keeps_structured_parse(lgapi_utils):
    """Endpoints with parseable YAML keep their structured metadata —
    we only changed the failure branch, not the success path.
    """
    gen = _generator(lgapi_utils)
    endpoint = _FakeEndpoint("/y", "get", _DocstringFixture.valid_yaml)
    gen.get_endpoints = lambda _routes: [endpoint]
    schema = gen.get_schema([])
    entry = schema["paths"]["/y"]["get"]
    assert entry.get("summary") == "A valid YAML docstring."
    assert entry.get("description") == "Stays structured."


def test_no_docstring_still_handled(lgapi_utils):
    """Endpoints with ``__doc__ = None`` must not raise — fallback uses
    empty string for ``description``.
    """
    gen = _generator(lgapi_utils)
    endpoint = _FakeEndpoint("/z", "get", _DocstringFixture.no_docstring)
    gen.get_endpoints = lambda _routes: [endpoint]
    schema = gen.get_schema([])
    entry = schema["paths"]["/z"]["get"]
    # Either description="" (fallback path) or structured (if YAML parse
    # of None happens to succeed somehow — implementation detail).
    # The contract is just "no exception, entry exists".
    assert isinstance(entry, dict)


def test_patch_flag_set():
    from EvoScientist.llm.patches import _langgraph_schema_silenced_patched

    assert _langgraph_schema_silenced_patched is True
