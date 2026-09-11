"""Unit tests for the proactive-check cron wrapper (mock langgraph_sdk client).

Mirrors tests/test_cron_schedule.py: pins the cron client-call behaviour.
"""

from unittest.mock import MagicMock


def _patch_client(monkeypatch):
    from EvoScientist.proactive import cron

    fake = MagicMock()
    fake.crons.create.return_value = {"cron_id": "p-1", "schedule": "*/10 * * * *"}
    fake.crons.search.return_value = [
        {
            "cron_id": "p-1",
            "schedule": "*/10 * * * *",
            "metadata": {"run_kind": "proactive_check"},
        },
    ]
    monkeypatch.setattr(cron, "_client", lambda: fake)
    monkeypatch.setattr(cron, "_default_timezone", lambda: "Europe/London")
    return cron, fake


def test_create_targets_proactive_graph_with_empty_input(monkeypatch):
    cron, fake = _patch_client(monkeypatch)
    rec = cron.create_proactive_schedule()
    assert rec["cron_id"] == "p-1"
    kw = fake.crons.create.call_args.kwargs
    assert kw["assistant_id"] == cron.PROACTIVE_GRAPH_ID
    assert kw["schedule"] == cron.DEFAULT_PROACTIVE_SCHEDULE
    assert kw["input"] == {}  # non-null (EmptyInputError); graph enumerates itself
    assert kw["metadata"] == {"run_kind": cron.PROACTIVE_RUN_KIND}
    assert kw["timezone"] == "Europe/London"


def test_create_accepts_custom_schedule_and_timezone(monkeypatch):
    cron, fake = _patch_client(monkeypatch)
    cron.create_proactive_schedule(schedule="*/5 * * * *", timezone="UTC")
    kw = fake.crons.create.call_args.kwargs
    assert kw["schedule"] == "*/5 * * * *"
    assert kw["timezone"] == "UTC"


def test_list_uses_server_side_run_kind_filter(monkeypatch):
    cron, fake = _patch_client(monkeypatch)
    out = cron.list_proactive_schedules()
    assert [c["cron_id"] for c in out] == ["p-1"]
    fake.crons.search.assert_called_once_with(
        metadata={"run_kind": cron.PROACTIVE_RUN_KIND},
        limit=1000,
    )


def test_delete_and_set_enabled(monkeypatch):
    cron, fake = _patch_client(monkeypatch)
    cron.delete_proactive_schedule("p-1")
    fake.crons.delete.assert_called_once_with("p-1")
    cron.set_proactive_enabled("p-1", False)
    assert fake.crons.update.call_args.kwargs["enabled"] is False


def test_is_available_gates_on_langgraph_dev(monkeypatch):
    from EvoScientist.langgraph_dev import manager
    from EvoScientist.proactive import cron

    monkeypatch.setattr(cron, "_proactive_url", lambda: "http://localhost:6174")
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_: True)
    assert cron.is_available() is True
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_: False)
    assert cron.is_available() is False


def test_langgraph_json_registers_proactive_graph():
    """langgraph.json must map PROACTIVE_GRAPH_ID to the graph module the cron fires."""
    import json
    from pathlib import Path

    import EvoScientist
    from EvoScientist.proactive import cron

    root = Path(EvoScientist.__file__).parent
    manifest = json.loads(
        (root / "langgraph_dev" / "langgraph.json").read_text(encoding="utf-8")
    )
    assert manifest["graphs"][cron.PROACTIVE_GRAPH_ID] == (
        "EvoScientist.proactive.graph:proactive_graph"
    )


def test_proactive_graph_imports_and_compiles_without_building_deps():
    """The registered graph must import + compile cheaply (langgraph-dev does this
    at startup); heavy deps must stay lazy so a bad-config env can't break startup."""
    from EvoScientist.proactive import graph

    assert graph.proactive_graph is not None
    assert graph._DEPS is None  # deps are built on first tick, never at import
