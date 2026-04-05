from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import yaml

from ..config import get_config_path

_PROVIDER_KEY_SOURCES = {
    "anthropic": ("anthropic_api_key", "ANTHROPIC_API_KEY"),
    "custom-anthropic": ("custom_anthropic_api_key", "CUSTOM_ANTHROPIC_API_KEY"),
    "openai": ("openai_api_key", "OPENAI_API_KEY"),
    "custom-openai": ("custom_openai_api_key", "CUSTOM_OPENAI_API_KEY"),
    "google-genai": ("google_api_key", "GOOGLE_API_KEY"),
    "google": ("google_api_key", "GOOGLE_API_KEY"),
    "nvidia": ("nvidia_api_key", "NVIDIA_API_KEY"),
    "minimax": ("minimax_api_key", "MINIMAX_API_KEY"),
    "deepseek": ("deepseek_api_key", "DEEPSEEK_API_KEY"),
    "siliconflow": ("siliconflow_api_key", "SILICONFLOW_API_KEY"),
    "openrouter": ("openrouter_api_key", "OPENROUTER_API_KEY"),
    "zhipu": ("zhipu_api_key", "ZHIPU_API_KEY"),
    "zhipu-code": ("zhipu_api_key", "ZHIPU_API_KEY"),
    "volcengine": ("volcengine_api_key", "VOLCENGINE_API_KEY"),
    "dashscope": ("dashscope_api_key", "DASHSCOPE_API_KEY"),
    "ollama": ("ollama_base_url", "OLLAMA_BASE_URL"),
}


def _check(name: str, ok: bool, message: str) -> dict[str, object]:
    return {"name": name, "ok": ok, "message": message}


def run_doctor(config_path: str | Path | None = None) -> dict[str, object]:
    path = Path(config_path) if config_path is not None else get_config_path()
    checks: list[dict[str, object]] = []

    if not path.exists():
        checks.append(_check("config_file", False, f"Config file not found: {path}"))
        return {"ok": False, "checks": checks}

    checks.append(_check("config_file", True, f"Config file found: {path}"))

    try:
        raw_data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        checks.append(_check("config_parse", False, f"Failed to parse config: {exc}"))
        return {"ok": False, "checks": checks}

    if raw_data is None:
        data: Mapping[str, object] = {}
    elif isinstance(raw_data, Mapping):
        data = raw_data
    else:
        checks.append(_check("config_parse", False, "Config file must contain a mapping"))
        return {"ok": False, "checks": checks}

    provider = str(data.get("provider", "anthropic"))
    if provider not in _PROVIDER_KEY_SOURCES:
        checks.append(_check("provider_supported", False, f"Provider {provider} is unsupported"))
    else:
        config_key, env_name = _PROVIDER_KEY_SOURCES[provider]
        config_value = data.get(config_key, "")
        env_value = os.environ.get(env_name, "")
        auth_mode = str(data.get("auth_mode", "")).lower()
        env_auth_mode = os.environ.get("AUTH_MODE", "").lower()
        has_proxy = bool(os.environ.get("CC_PROXY_URL"))
        has_key = bool(
            config_value
            or env_value
            or auth_mode == "oauth"
            or env_auth_mode == "oauth"
            or has_proxy
        )
        checks.append(
            _check(
                "provider_credentials",
                has_key,
                f"Provider {provider} credentials {'available' if has_key else 'missing'}",
            )
        )

    default_workdir = data.get("default_workdir", "")
    if default_workdir:
        workdir = Path(default_workdir).expanduser()
        workdir_ok = workdir.exists() and workdir.is_dir()
        checks.append(
            _check(
                "default_workdir",
                workdir_ok,
                f"Default workdir {'ok' if workdir_ok else 'missing'}: {workdir}",
            )
        )
    else:
        checks.append(_check("default_workdir", True, "No default workdir configured"))

    artifact_root = Path.cwd() / "artifacts"
    try:
        artifact_root.mkdir(parents=True, exist_ok=True)
        checks.append(_check("artifact_root", True, f"Artifact root writable: {artifact_root}"))
    except Exception as exc:
        checks.append(_check("artifact_root", False, f"Artifact root not writable: {exc}"))

    ok = all(bool(item["ok"]) for item in checks)
    return {"ok": ok, "checks": checks}
