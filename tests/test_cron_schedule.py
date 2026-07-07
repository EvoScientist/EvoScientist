"""Unit tests for the cron wrapper (mock langgraph_sdk client)."""

from unittest.mock import MagicMock


def _patch_client(monkeypatch):
    from EvoScientist.cron import schedule as crons

    fake = MagicMock()
    fake.crons.create.return_value = {"cron_id": "c-1", "schedule": "*/10 * * * *"}
    fake.crons.search.return_value = [
        {
            "cron_id": "c-1",
            "schedule": "*/10 * * * *",
            "metadata": {"run_kind": "scheduled_task", "name": "weather"},
        },
    ]
    monkeypatch.setattr(crons, "_client", lambda: fake)
    monkeypatch.setattr(crons, "_default_timezone", lambda: "Europe/London")
    return crons, fake


def test_create_schedule_targets_scheduler(monkeypatch):
    crons, fake = _patch_client(monkeypatch)
    rec = crons.create_schedule(
        name="weather",
        schedule="*/10 * * * *",
        prompt="search uk weather and summarize",
    )
    assert rec["cron_id"] == "c-1"
    kw = fake.crons.create.call_args.kwargs
    assert kw["assistant_id"] == crons.SCHEDULER_GRAPH_ID
    assert kw["schedule"] == "*/10 * * * *"
    assert kw["input"] == {
        "messages": [{"role": "user", "content": "search uk weather and summarize"}]
    }
    assert kw["metadata"]["run_kind"] == crons.SCHEDULED_RUN_KIND
    assert kw["metadata"]["name"] == "weather"
    assert kw["timezone"] == "Europe/London"


def test_list_schedules_uses_server_side_filter(monkeypatch):
    crons, fake = _patch_client(monkeypatch)
    out = crons.list_schedules()
    assert [c["cron_id"] for c in out] == ["c-1"]
    # Filtered server-side by run_kind metadata (no client filter); high limit so
    # users with >10 schedules still see them all.
    fake.crons.search.assert_called_once_with(
        metadata={"run_kind": crons.SCHEDULED_RUN_KIND},
        limit=1000,
    )


def test_delete_and_set_enabled(monkeypatch):
    crons, fake = _patch_client(monkeypatch)
    crons.delete_schedule("c-1")
    fake.crons.delete.assert_called_once_with("c-1")
    crons.set_enabled("c-1", False)
    assert fake.crons.update.call_args.kwargs["enabled"] is False


def test_run_now_dispatches_thread_then_run(monkeypatch):
    crons, fake = _patch_client(monkeypatch)
    fake.threads.create.return_value = {"thread_id": "t-1"}
    fake.runs.create.return_value = {"run_id": "r-1"}
    rec = crons.run_now("do the thing")
    assert rec["run_id"] == "r-1"
    fake.threads.create.assert_called_once_with(graph_id=crons.SCHEDULER_GRAPH_ID)
    run_kw = fake.runs.create.call_args.kwargs
    assert run_kw["thread_id"] == "t-1"
    assert run_kw["assistant_id"] == crons.SCHEDULER_GRAPH_ID
    assert run_kw["input"] == {
        "messages": [{"role": "user", "content": "do the thing"}]
    }
    assert run_kw["metadata"]["run_kind"] == crons.SCHEDULED_RUN_KIND
    assert run_kw["metadata"]["prompt"] == "do the thing"


# ---------------------------------------------------------------------------
# Task 3: scheduler registration tests
# ---------------------------------------------------------------------------


def test_scheduler_yaml_loads_as_async():
    """scheduler.yaml must define scheduler with async:true and a system_prompt."""
    from pathlib import Path

    import yaml

    import EvoScientist

    subagents_dir = Path(EvoScientist.__file__).parent / "subagents"
    yaml_path = subagents_dir / "scheduler.yaml"
    assert yaml_path.exists(), f"Missing {yaml_path}"
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert "scheduler" in data, "Top-level key must be 'scheduler'"
    spec = data["scheduler"]
    assert spec.get("async") is True, "scheduler must have 'async: true'"
    assert spec.get("system_prompt"), "scheduler must have a non-empty system_prompt"
    # description drives task-tool routing — must be a non-empty string.
    assert spec.get("description"), "scheduler must have a non-empty description"


def test_langgraph_json_registers_scheduler():
    """langgraph.json must map scheduler to the graphs:scheduler binding."""
    import json
    from pathlib import Path

    import EvoScientist

    root = Path(EvoScientist.__file__).parent
    manifest = json.loads(
        (root / "langgraph_dev" / "langgraph.json").read_text(encoding="utf-8")
    )
    assert manifest["graphs"]["scheduler"] == (
        "EvoScientist.langgraph_dev.graphs:scheduler"
    )


def test_scheduler_graph_id_matches_registration():
    """SCHEDULER_GRAPH_ID must match both langgraph.json and scheduler.yaml top-level key."""
    import json
    from pathlib import Path

    import yaml

    import EvoScientist
    from EvoScientist.cron import schedule as crons

    root = Path(EvoScientist.__file__).parent
    manifest = json.loads(
        (root / "langgraph_dev" / "langgraph.json").read_text(encoding="utf-8")
    )
    assert crons.SCHEDULER_GRAPH_ID in manifest["graphs"], (
        f"SCHEDULER_GRAPH_ID={crons.SCHEDULER_GRAPH_ID!r} not found in langgraph.json graphs"
    )
    spec = yaml.safe_load(
        (root / "subagents" / "scheduler.yaml").read_text(encoding="utf-8")
    )
    assert crons.SCHEDULER_GRAPH_ID in spec, (
        f"SCHEDULER_GRAPH_ID={crons.SCHEDULER_GRAPH_ID!r} is not the top-level key in scheduler.yaml"
    )


def test_production_loaders_accept_utf8_content(tmp_path, monkeypatch):
    """Production config readers must handle localized UTF-8 content."""
    import os

    from EvoScientist.langgraph_dev import manager
    from EvoScientist.mcp import client, registry

    mcp_config = tmp_path / "mcp.yaml"
    mcp_config.write_text(
        """
écho:
  transport: stdio
  command: python
  args: ["-m", "démo"]
  env:
    MESSAGE: "bonjour café"
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(client, "USER_MCP_CONFIG", mcp_config)
    assert client._load_user_config()["écho"]["env"]["MESSAGE"] == "bonjour café"
    assert client.load_mcp_config()["écho"]["args"] == ["-m", "démo"]

    marketplace_yaml = tmp_path / "marketplace-écho.yaml"
    marketplace_yaml.write_text(
        """
name: écho
label: "Écho café"
description: "localized marketplace entry"
tags: "démo, marché"
transport: stdio
command: python
args: ["-m", "écho"]
""".lstrip(),
        encoding="utf-8",
    )
    entry = registry.parse_marketplace_yaml(marketplace_yaml)
    assert entry.name == "écho"
    assert entry.tags == ["démo", "marché"]

    venv = tmp_path / "uv" / "tools" / "evoscientist"
    venv.mkdir(parents=True)
    (venv / "uv-receipt.toml").write_text(
        """
[tool]
requirements = [
  { name = "evoscientist" },
  { name = "mcp-écho", specifier = ">=1.0" },
]
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("VIRTUAL_ENV", str(venv))
    assert registry._uv_tool_existing_requirements()["mcp-écho"] == "mcp-écho>=1.0"

    runtime = manager.LanggraphRuntimePaths.for_directory(tmp_path / "runtime")
    monkeypatch.setattr(manager, "RUNTIME", runtime)
    workspace = tmp_path / "workspace-écho"
    manager._write_workspace_sidecar(workspace, 12345)
    assert manager._read_workspace_sidecar() == {
        "workspace": str(workspace),
        "pid": 12345,
    }

    runtime.pid_file.write_text(str(os.getpid()), encoding="utf-8")

    class _NotLanggraphProcess:
        def __init__(self, pid):
            self.pid = pid

        def cmdline(self):
            return ["python", "localized-test"]

        def kill(self):
            raise AssertionError("non-langgraph pid must not be killed")

    monkeypatch.setattr(manager, "_list_pids_on_port", lambda port: [os.getpid()])
    monkeypatch.setattr(manager.psutil, "Process", _NotLanggraphProcess)
    assert manager._kill_owned_stale_process(6174) is False
    assert not runtime.pid_file.exists()
    assert not runtime.workspace_sidecar.exists()
