from EvoScientist.orchestration.report import build_run_report


def test_report_summarizes_run_state_and_outputs(tmp_path):
    artifact_dir = tmp_path / "artifacts" / "es-20260401-abc123"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "deliverables").mkdir()
    (artifact_dir / "status.json").write_text(
        '{\n'
        '  "run_id": "es-20260401-abc123",\n'
        '  "status": "created",\n'
        '  "thread_id": "deadbeef",\n'
        '  "workspace_dir": "/tmp/work",\n'
        '  "artifact_dir": "' + str(artifact_dir).replace('\\', '\\\\') + '",\n'
        '  "current_stage": null,\n'
        '  "completed_stages": [],\n'
        '  "last_error": null,\n'
        '  "suggested_next_action": null,\n'
        '  "updated_at": "2026-04-01T00:00:00Z"\n'
        '}\n'
    )

    report = build_run_report(artifact_dir)

    assert "es-20260401-abc123" in report
    assert "created" in report.lower()
    assert str(artifact_dir) in report
