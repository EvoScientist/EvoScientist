from __future__ import annotations

import os
from pathlib import Path

import yaml

from ..config import get_config_path

_PROVIDER_ENV_MAP = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google-genai": "GOOGLE_API_KEY",
    "google": "GOOGLE_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "minimax": "MINIMAX_API_KEY",
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

    data = yaml.safe_load(path.read_text()) or {}
    provider = data.get("provider", "anthropic")
    env_name = _PROVIDER_ENV_MAP.get(provider, "")
    api_key = data.get(f"{provider}_api_key", "") if provider else ""
    env_value = os.environ.get(env_name, "") if env_name else ""

    has_key = bool(api_key or env_value)
    checks.append(
        _check(
            "provider_api_key",
            has_key,
            f"Provider {provider} API key {'available' if has_key else 'missing'}",
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
