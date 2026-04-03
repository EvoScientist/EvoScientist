from EvoScientist.orchestration.doctor import run_doctor


def test_doctor_reports_missing_config(tmp_path):
    result = run_doctor(config_path=tmp_path / "config.yaml")
    assert result["ok"] is False
    assert any("config" in item["name"] for item in result["checks"])


def test_doctor_reports_missing_provider_api_key(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("provider: anthropic\nmodel: claude-sonnet-4-5\n")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    result = run_doctor(config_path=config_path)

    assert result["ok"] is False
    assert any(item["name"] == "provider_api_key" for item in result["checks"])


def test_doctor_passes_with_explicit_workdir_and_api_key(tmp_path, monkeypatch):
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "provider: anthropic\n"
        "model: claude-sonnet-4-5\n"
        f"default_workdir: {workdir}\n"
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    result = run_doctor(config_path=config_path)

    assert result["ok"] is True
    assert all(item["ok"] for item in result["checks"])
