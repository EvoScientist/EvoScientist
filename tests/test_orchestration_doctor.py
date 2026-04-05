import pytest

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
    assert any(item["name"] == "provider_credentials" for item in result["checks"])


def test_doctor_accepts_custom_openai_env_key(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("provider: custom-openai\nmodel: gpt-5.4\n")
    monkeypatch.setenv("CUSTOM_OPENAI_API_KEY", "test-key")

    result = run_doctor(config_path=config_path)

    assert result["ok"] is True
    assert any(
        item["name"] == "provider_credentials" and item["ok"] is True
        for item in result["checks"]
    )


def test_doctor_accepts_custom_anthropic_env_key(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("provider: custom-anthropic\nmodel: claude-sonnet-4-5\n")
    monkeypatch.setenv("CUSTOM_ANTHROPIC_API_KEY", "test-key")

    result = run_doctor(config_path=config_path)

    assert result["ok"] is True
    assert any(
        item["name"] == "provider_credentials" and item["ok"] is True
        for item in result["checks"]
    )


@pytest.mark.parametrize(
    ("provider", "env_name", "model"),
    [
        ("deepseek", "DEEPSEEK_API_KEY", "deepseek-v3"),
        ("siliconflow", "SILICONFLOW_API_KEY", "glm-5"),
        ("openrouter", "OPENROUTER_API_KEY", "gpt-5.4"),
        ("zhipu", "ZHIPU_API_KEY", "glm-5"),
        ("zhipu-code", "ZHIPU_API_KEY", "glm-5"),
        ("volcengine", "VOLCENGINE_API_KEY", "doubao-seed-2.0-pro"),
        ("dashscope", "DASHSCOPE_API_KEY", "qwen-max"),
        ("ollama", "OLLAMA_BASE_URL", "llama3.1"),
    ],
)
def test_doctor_accepts_runtime_supported_provider_env_keys(
    tmp_path, monkeypatch, provider, env_name, model
):
    config_path = tmp_path / f"{provider}.yaml"
    config_path.write_text(f"provider: {provider}\nmodel: {model}\n")
    monkeypatch.setenv(env_name, "test-value")

    result = run_doctor(config_path=config_path)

    assert result["ok"] is True
    assert any(
        item["name"] == "provider_credentials" and item["ok"] is True
        for item in result["checks"]
    )


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


@pytest.mark.parametrize(
    ("config_text", "env_name", "env_value"),
    [
        ("provider: anthropic\nmodel: claude-sonnet-4-5\nauth_mode: oauth\n", None, None),
        ("provider: anthropic\nmodel: claude-sonnet-4-5\n", "CC_PROXY_URL", "http://localhost:3456"),
    ],
)
def test_doctor_accepts_oauth_or_ccproxy_without_api_key(
    tmp_path, monkeypatch, config_text, env_name, env_value
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_text)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CC_PROXY_URL", raising=False)
    monkeypatch.delenv("AUTH_MODE", raising=False)
    if env_name is not None and env_value is not None:
        monkeypatch.setenv(env_name, env_value)

    result = run_doctor(config_path=config_path)

    assert result["ok"] is True
    assert any(
        item["name"] == "provider_credentials" and item["ok"] is True
        for item in result["checks"]
    )
