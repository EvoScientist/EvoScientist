import json

from EvoScientist.orchestration.run_store import list_recent_runs, resolve_latest_run_id


def test_resolve_latest_run_id_returns_most_recent_run(tmp_path):
    root = tmp_path / "artifacts"
    first = root / "es-1"
    second = root / "es-2"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "status.json").write_text('{"run_id": "es-1", "updated_at": "2026-04-03T10:00:00Z"}')
    (second / "status.json").write_text('{"run_id": "es-2", "updated_at": "2026-04-03T11:00:00Z"}')

    assert resolve_latest_run_id(root) == "es-2"


def test_resolve_latest_run_id_ignores_malformed_runs_and_falls_back_to_mtime(tmp_path):
    root = tmp_path / "artifacts"
    malformed = root / "bad-run"
    older = root / "es-older"
    newer = root / "es-newer"
    malformed.mkdir(parents=True)
    older.mkdir()
    newer.mkdir()

    (malformed / "status.json").write_text("{not json")
    older_run = older / "run.json"
    newer_run = newer / "run.json"
    older_run.write_text(json.dumps({"run_id": "es-older"}))
    newer_run.write_text(json.dumps({"run_id": "es-newer"}))

    older_run.touch()
    newer_run.touch()

    assert resolve_latest_run_id(root) == "es-newer"


def test_list_recent_runs_returns_summary_fields_and_prompt_precedence(tmp_path):
    root = tmp_path / "artifacts"
    run_dir = root / "es-20260403-abc123"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "es-20260403-abc123",
                "prompt": "compare orchestration architectures",
                "metadata": {"prompt": "prompt from metadata should not win"},
            }
        )
    )
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "run_id": "es-20260403-abc123",
                "status": "completed",
                "current_stage": "report",
                "updated_at": "2026-04-03T12:00:00Z",
                "last_error": None,
            }
        )
    )

    rows = list_recent_runs(root, limit=10)

    assert rows == [
        {
            "run_id": "es-20260403-abc123",
            "status": "completed",
            "updated_at": "2026-04-03T12:00:00Z",
            "current_stage": "report",
            "summary": "compare orchestration architectures",
            "last_error": None,
        }
    ]


def test_list_recent_runs_falls_back_to_stage_then_error_and_ignores_invalid_dirs(tmp_path):
    root = tmp_path / "artifacts"
    broken = root / "broken"
    failed = root / "es-failed"
    errored = root / "es-error"
    broken.mkdir(parents=True)
    failed.mkdir()
    errored.mkdir()

    (broken / "status.json").write_text("[]")
    (failed / "status.json").write_text(
        json.dumps(
            {
                "run_id": "es-failed",
                "status": "failed",
                "current_stage": "analysis",
                "updated_at": "2026-04-03T12:00:00Z",
                "last_error": "context overflow",
            }
        )
    )
    (errored / "status.json").write_text(
        json.dumps(
            {
                "run_id": "es-error",
                "status": "failed",
                "current_stage": None,
                "updated_at": "2026-04-03T11:00:00Z",
                "last_error": "provider unavailable",
            }
        )
    )

    rows = list_recent_runs(root, limit=10)

    assert [row["run_id"] for row in rows] == ["es-failed", "es-error"]
    assert rows[0]["summary"] == "analysis"
    assert rows[1]["summary"] == "provider unavailable"


def test_list_recent_runs_uses_metadata_prompt_when_top_level_prompt_missing(tmp_path):
    root = tmp_path / "artifacts"
    run_dir = root / "es-meta"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "es-meta",
                "metadata": {"prompt": "design better orchestration checkpoints"},
            }
        )
    )
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "run_id": "es-meta",
                "status": "running",
                "current_stage": "analysis",
                "updated_at": "2026-04-03T13:00:00Z",
                "last_error": None,
            }
        )
    )

    rows = list_recent_runs(root)

    assert rows[0]["summary"] == "design better orchestration checkpoints"


def test_list_recent_runs_falls_back_to_run_json_run_id_when_status_is_partial(tmp_path):
    root = tmp_path / "artifacts"
    run_dir = root / "es-partial"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "es-partial",
                "updated_at": "2026-04-03T14:00:00Z",
                "status": "completed",
            }
        )
    )
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "status": "running",
                "current_stage": "analysis",
                "updated_at": "2026-04-03T13:00:00Z",
            }
        )
    )

    rows = list_recent_runs(root)

    assert rows[0]["run_id"] == "es-partial"
    assert resolve_latest_run_id(root) == "es-partial"


def test_list_recent_runs_uses_run_json_updated_at_when_status_updated_at_is_malformed(tmp_path):
    root = tmp_path / "artifacts"
    older = root / "es-older"
    newer = root / "es-newer"
    older.mkdir(parents=True)
    newer.mkdir()

    (older / "run.json").write_text(
        json.dumps(
            {
                "run_id": "es-older",
                "updated_at": "2026-04-03T12:00:00Z",
            }
        )
    )
    (older / "status.json").write_text(
        json.dumps(
            {
                "run_id": "es-older",
                "updated_at": "not-a-timestamp",
            }
        )
    )
    (newer / "status.json").write_text(
        json.dumps(
            {
                "run_id": "es-newer",
                "updated_at": "2026-04-03T11:00:00Z",
            }
        )
    )

    rows = list_recent_runs(root)

    assert [row["run_id"] for row in rows] == ["es-older", "es-newer"]
    assert rows[0]["updated_at"] == "2026-04-03T12:00:00Z"
