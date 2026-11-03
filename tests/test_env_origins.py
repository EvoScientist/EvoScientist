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
    monkeypatch.setattr(settings, "_ENV_ORIGINS", {})
    monkeypatch.setattr(settings_api, "_baseline", None)


def _use_dotenv(tmp_path: Path, monkeypatch) -> None:
    """A workspace ``.env`` with a setting of the EvoSci command (the suite
    turns ``.env`` discovery off)."""
    dotenv = tmp_path / ".env"
    dotenv.write_text("EVOSCIENTIST_DEFAULT_MODE=run\n", encoding="utf-8")
    monkeypatch.setattr(settings, "find_dotenv", lambda *a, **k: str(dotenv))
    monkeypatch.delenv("EVOSCIENTIST_DEFAULT_MODE", raising=False)


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
        assert env_origin("EVOSCIENTIST_DEFAULT_MODE") == {
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
        assert sources["default_mode"] == {
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
    """What start_langgraph_dev records for the server's environment."""

    def test_dotenv_setting_not_passed_to_server(
        self, tmp_path, monkeypatch, runtime_paths
    ):
        """A project ``.env`` configures its own EvoSci command only: the
        shared server gets neither the setting nor its record."""
        from EvoScientist.langgraph_dev.manager import _server_env_origins
        from tests.fakes.langgraph_dev import PopenAbort, patch_start_prereqs

        dotenv = tmp_path / ".env"
        dotenv.write_text("EVOSCIENTIST_UI_BACKEND=cli\n", encoding="utf-8")
        monkeypatch.setattr(settings, "find_dotenv", lambda *a, **k: str(dotenv))
        monkeypatch.delenv("EVOSCIENTIST_UI_BACKEND", raising=False)
        assert get_effective_config().ui_backend == "cli"
        captured = patch_start_prereqs(monkeypatch, tmp_path, runtime_paths)
        from EvoScientist.langgraph_dev import manager

        with pytest.raises(PopenAbort):
            manager.start_langgraph_dev(workspace_dir=tmp_path, port=16178)

        assert "EVOSCIENTIST_UI_BACKEND" not in captured["env"]
        assert "EVOSCIENTIST_UI_BACKEND" not in _server_env_origins(
            "6174", "x", lasting=True
        )


class TestDotenvForServer:
    """A project ``.env`` sets only the EvoSci command's own settings: every
    agent takes its keys and settings from config.yaml and the environment."""

    def test_dotenv_cannot_move_folders_read_at_import(self, tmp_path, monkeypatch):
        dotenv = tmp_path / ".env"
        dotenv.write_text(f"EVOSCIENTIST_MEMORIES_DIR={tmp_path}/mem\n")
        monkeypatch.setattr(settings, "find_dotenv", lambda *a, **k: str(dotenv))
        monkeypatch.delenv("EVOSCIENTIST_MEMORIES_DIR", raising=False)

        get_effective_config()

        assert "EVOSCIENTIST_MEMORIES_DIR" not in os.environ

    def test_server_reads_no_dotenv(self, monkeypatch):
        """From the server home the search would find one in a parent folder,
        such as ``~/.env``."""
        from EvoScientist import paths

        def found(*_args, **_kwargs):
            raise AssertionError("the server searched for a .env")

        monkeypatch.setattr(paths, "SERVER_PROCESS", True)
        monkeypatch.setattr(settings, "find_dotenv", found)
        get_effective_config()


def test_an_unreadable_config_file_does_not_stop_a_start(config_file):
    config_file.mkdir()
    apply_config_to_env(get_effective_config())
    settings_api.capture_restart_baseline()
    assert settings_api._baseline is not None


def test_apply_config_to_env_reads_the_file_once(config_file, monkeypatch):
    from EvoScientist.config import store

    config_file.write_text(
        "openai_api_key: sk-file-1\ntavily_api_key: tvly-file-2\n"
        "minimax_api_key: sk-file-3\n",
        encoding="utf-8",
    )
    config = get_effective_config()
    reads = []
    real = store.read_config_mapping
    monkeypatch.setattr(store, "read_config_mapping", lambda: reads.append(1) or real())
    apply_config_to_env(config)
    assert len(reads) == 1
    for field in ("openai_api_key", "tavily_api_key", "minimax_api_key"):
        assert env_origin(_ENV_MAPPINGS[field]) == {"source": "file"}
