"""Tests for the record of where config-mapped environment variables came from
(EvoScientist.config.settings) and its use in the settings API's ``sources``."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from EvoScientist.config import settings
from EvoScientist.config.settings import (
    _ENV_MAPPINGS,
    EvoScientistConfig,
    apply_config_to_env,
    config_value_origin,
    env_origin,
    get_config_path,
    get_effective_config,
)
from EvoScientist.langgraph_dev import settings_api


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for env_key in set(_ENV_MAPPINGS.values()):
        # Set first, so teardown also removes what a test's
        # apply_config_to_env() sets.
        monkeypatch.setenv(env_key, "")
        monkeypatch.delenv(env_key)
    monkeypatch.setattr(settings, "_ENV_ORIGINS", {})
    monkeypatch.setattr(settings_api, "_baseline", None)


def _use_dotenv(tmp_path: Path, monkeypatch) -> None:
    """A workspace ``.env`` (the suite turns ``.env`` discovery off)."""
    dotenv = tmp_path / ".env"
    dotenv.write_text("TAVILY_API_KEY=tvly-dotenv\n", encoding="utf-8")
    monkeypatch.setattr(settings, "find_dotenv", lambda *a, **k: str(dotenv))


@pytest.fixture
def config_file() -> Path:
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class TestConfigValueOrigin:
    def test_file(self, config_file):
        config_file.write_text("langgraph_dev_port: 7000\n", encoding="utf-8")
        assert config_value_origin("langgraph_dev_port", 7000) == "file"
        assert config_value_origin("langgraph_dev_port", "7000") == "file"

    def test_flag_over_the_file(self, config_file):
        config_file.write_text("langgraph_dev_port: 7000\n", encoding="utf-8")
        assert config_value_origin("langgraph_dev_port", 9000) == "cli"

    def test_default(self):
        default = EvoScientistConfig().langgraph_dev_port
        assert config_value_origin("langgraph_dev_port", default) == "default"

    def test_workdir_compares_expanded_paths(self, config_file):
        config_file.write_text("default_workdir: ~/research\n", encoding="utf-8")
        expanded = os.path.expanduser("~/research")
        assert config_value_origin("default_workdir", expanded) == "file"

    def test_workdir_from_the_current_folder_is_the_default(self):
        assert config_value_origin("default_workdir", os.getcwd()) == "default"

    def test_workdir_flag(self, tmp_path):
        assert config_value_origin("default_workdir", str(tmp_path)) == "cli"


class TestRecording:
    def test_apply_config_to_env_records_file_copies(self, config_file):
        config_file.write_text("openai_api_key: sk-file\n", encoding="utf-8")
        apply_config_to_env(get_effective_config())
        assert os.environ["OPENAI_API_KEY"] == "sk-file"
        assert env_origin("OPENAI_API_KEY") == {"source": "file"}

    def test_the_users_export_is_not_recorded(self, config_file, monkeypatch):
        config_file.write_text("openai_api_key: sk-file\n", encoding="utf-8")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-shell")
        apply_config_to_env(get_effective_config())
        assert env_origin("OPENAI_API_KEY") is None

    def test_a_flag_override_is_cli(self):
        apply_config_to_env(get_effective_config({"dangerous_mode": True}))
        assert env_origin("EVOSCIENTIST_DANGEROUS_MODE") == {"source": "cli"}
        # Turned off again: the variable is removed, and so is its record.
        apply_config_to_env(EvoScientistConfig())
        assert env_origin("EVOSCIENTIST_DANGEROUS_MODE") is None

    def test_dotenv_records_its_path(self, tmp_path, monkeypatch):
        _use_dotenv(tmp_path, monkeypatch)
        get_effective_config()
        assert env_origin("TAVILY_API_KEY") == {
            "source": "dotenv",
            "path": str(tmp_path / ".env"),
        }

    def test_ccproxy(self, monkeypatch):
        from EvoScientist.ccproxy_manager import setup_ccproxy_env

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        setup_ccproxy_env(9999)
        assert env_origin("ANTHROPIC_BASE_URL") == {"source": "ccproxy"}
        assert env_origin("ANTHROPIC_API_KEY") == {"source": "ccproxy"}


def test_the_server_reads_and_removes_the_record():
    """Read once at import; removed so the agent's commands never inherit it."""
    code = (
        "import os, json\n"
        "from EvoScientist.config import settings\n"
        "print(json.dumps([os.environ.get(settings.ENV_ORIGINS_VAR), "
        "settings.env_origin('OPENAI_API_KEY')]))"
    )
    env = dict(os.environ)
    env[settings.ENV_ORIGINS_VAR] = json.dumps({"OPENAI_API_KEY": {"source": "file"}})
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    ).stdout
    assert json.loads(out.strip().splitlines()[-1]) == [None, {"source": "file"}]


class TestSources:
    def test_a_saved_key_copied_into_the_environment_reads_file(self, config_file):
        config_file.write_text("openai_api_key: sk-file\n", encoding="utf-8")
        apply_config_to_env(get_effective_config())
        sources = settings_api.read_settings()["sources"]
        assert sources["openai_api_key"] == {"source": "file", "shadowed": False}

    def test_shell_key_shadows_with_and_without_a_saved_one(
        self, config_file, monkeypatch
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-shell")
        sources = settings_api.read_settings()["sources"]
        assert sources["openai_api_key"] == {"source": "env", "shadowed": True}
        config_file.write_text("openai_api_key: sk-file\n", encoding="utf-8")
        apply_config_to_env(get_effective_config())
        sources = settings_api.read_settings()["sources"]
        assert sources["openai_api_key"] == {"source": "env", "shadowed": True}

    def test_dotenv_with_path(self, tmp_path, monkeypatch):
        _use_dotenv(tmp_path, monkeypatch)
        sources = settings_api.read_settings()["sources"]
        assert sources["tavily_api_key"] == {
            "source": "dotenv",
            "path": str(tmp_path / ".env"),
            "shadowed": True,
        }

    def test_launcher_workspace_is_not_shadowed(self, monkeypatch):
        settings.record_env_origin("EVOSCIENTIST_WORKSPACE_DIR", "default")
        monkeypatch.setenv("EVOSCIENTIST_WORKSPACE_DIR", os.getcwd())
        sources = settings_api.read_settings()["sources"]
        assert sources["default_workdir"] == {"source": "default", "shadowed": False}

    def test_a_port_flag_is_cli_and_shadowed(self, monkeypatch):
        settings.record_env_origin("EVOSCIENTIST_LANGGRAPH_DEV_PORT", "cli")
        monkeypatch.setenv("EVOSCIENTIST_LANGGRAPH_DEV_PORT", "9000")
        sources = settings_api.read_settings()["sources"]
        assert sources["langgraph_dev_port"] == {"source": "cli", "shadowed": True}

    def test_changed_saved_key_needs_a_restart_although_copied(self, config_file):
        """The server's environment holds the old copy; a restart applies the
        new value, so it is pending, not shadowed."""
        config_file.write_text("openai_api_key: sk-old-value\n", encoding="utf-8")
        apply_config_to_env(get_effective_config())
        settings_api.capture_restart_baseline()
        config_file.write_text("openai_api_key: sk-new-value\n", encoding="utf-8")
        body = settings_api.read_settings()
        assert body["restart_fields"] == ["openai_api_key"]

    def test_auto_approve_flag_needs_no_restart(self, config_file):
        """A server started by ``EvoSci --auto-approve`` compares the file with
        the file, so the flag is no change."""
        config_file.write_text("auto_approve: false\n", encoding="utf-8")
        apply_config_to_env(get_effective_config({"auto_approve": True}))
        settings_api.capture_restart_baseline()
        assert settings_api.read_settings()["restart_required"] is False


class TestServerRecord:
    """What start_langgraph_dev records for the workspace, port and host."""

    def test_a_dotenv_port_keeps_its_source(self, tmp_path, monkeypatch):
        from EvoScientist.langgraph_dev.manager import _server_env_origins

        dotenv = tmp_path / ".env"
        dotenv.write_text("EVOSCIENTIST_LANGGRAPH_DEV_PORT=6297\n", encoding="utf-8")
        monkeypatch.setattr(settings, "find_dotenv", lambda *a, **k: str(dotenv))
        get_effective_config()
        origins = _server_env_origins(str(tmp_path), "6297", "127.0.0.1")
        assert origins["EVOSCIENTIST_LANGGRAPH_DEV_PORT"] == {
            "source": "dotenv",
            "path": str(dotenv),
        }

    def test_a_symlinked_workdir_reads_file(self, config_file, tmp_path):
        from EvoScientist.langgraph_dev.manager import _server_env_origins

        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        try:
            link.symlink_to(real, target_is_directory=True)
        except OSError:
            pytest.skip("symlinks are not available")
        config_file.write_text(
            json.dumps({"default_workdir": str(link)}), encoding="utf-8"
        )
        origins = _server_env_origins(str(real.resolve()), "6174", "127.0.0.1")
        assert origins["EVOSCIENTIST_WORKSPACE_DIR"] == {"source": "file"}


def test_an_unreadable_config_file_does_not_stop_a_start(config_file):
    config_file.mkdir()
    apply_config_to_env(get_effective_config())
    settings_api.capture_restart_baseline()
    assert settings_api._baseline is not None
