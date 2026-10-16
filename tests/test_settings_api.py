"""Tests for /api/settings (EvoScientist.langgraph_dev.settings_api + http routes)."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from starlette.testclient import TestClient

from EvoScientist import local_secret, paths
from EvoScientist.config import provider_keys, store
from EvoScientist.config.provider_keys import KEY_SPECS, KeyCheck
from EvoScientist.config.settings import _ENV_MAPPINGS, get_config_path
from EvoScientist.langgraph_dev import settings_api
from EvoScientist.langgraph_dev.http import app


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    """No config-mapped variable from the developer's shell, a fresh baseline."""
    for env_key in set(_ENV_MAPPINGS.values()):
        monkeypatch.delenv(env_key, raising=False)
    monkeypatch.setattr(settings_api, "_baseline", None)


@pytest.fixture
def config_file() -> Path:
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(local_secret, "_value", None)
    with TestClient(
        app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000)
    ) as client:
        yield client


@pytest.fixture
def secret(server) -> dict[str, str]:
    value = (paths.SERVER_HOME / local_secret.FILE_NAME).read_text()
    return {local_secret.HEADER: value}


def _mapping(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _no_checks():
    """Key checks are live requests; every test that saves a key mocks them."""
    return patch.object(
        settings_api, "run_check", return_value=KeyCheck("valid", "Valid")
    )


class TestGet:
    def test_secrets_never_leave_the_server(self, server, config_file):
        config_file.write_text(
            "openai_api_key: sk-openai-1234567890\n"
            "tavily_api_key: tvly-short\n"
            "telegram_bot_token: 123456:TELEGRAMSECRET\n"
            "email_smtp_password: hunter2-long-password\n"
            "feishu_encrypt_key: FEISHUENCRYPT\n",
            encoding="utf-8",
        )
        response = server.get("/api/settings")
        assert response.status_code == 200
        for value in (
            "sk-openai-123456",
            "TELEGRAMSECRET",
            "hunter2",
            "FEISHUENCRYPT",
            "tvly-short",
        ):
            assert value not in response.text
        settings = response.json()["settings"]
        assert settings["openai_api_key"] == {"set": True, "hint": "…7890"}
        assert settings["tavily_api_key"] == {"set": True, "hint": "…hort"}
        assert settings["telegram_bot_token"] == {"set": True}
        assert settings["email_smtp_password"] == {"set": True}
        assert settings["anthropic_api_key"] == {"set": False}
        assert settings["mimo_token_plan_base_url"] == ""

    def test_shape(self, server):
        body = server.get("/api/settings").json()
        assert set(body) == {
            "settings",
            "sources",
            "read_only",
            "providers",
            "setup_needed",
            "restart_required",
            "restart_fields",
            "revision",
            "config_error",
        }
        assert body["revision"] is None
        assert "dangerous_mode" in body["read_only"]
        assert body["providers"]["ollama"] == {
            "label": "Ollama",
            "key_field": None,
            "base_url_field": "ollama_base_url",
            "needs_key": False,
            "needs_base_url": False,
            "has_check": True,
            "lists_models": True,
        }
        assert "tavily" not in body["providers"]

    def test_settings_are_the_file_values(self, server, config_file, monkeypatch):
        config_file.write_text("langgraph_dev_port: 7000\n", encoding="utf-8")
        monkeypatch.setenv("EVOSCIENTIST_LANGGRAPH_DEV_PORT", "7999")
        assert (
            server.get("/api/settings").json()["settings"]["langgraph_dev_port"] == 7000
        )

    def test_unparseable_file(self, server, config_file):
        config_file.write_bytes(b"openai_api_key: sk-x\n  bad: [\n")
        body = server.get("/api/settings").json()
        assert body["config_error"]["error"] == "config_unreadable"
        assert body["config_error"]["detail"].startswith("line ")
        assert "sk-x" not in str(body["config_error"])
        assert body["setup_needed"] is False
        assert body["restart_required"] is False


class TestSetupNeeded:
    def test_no_file(self, server):
        assert server.get("/api/settings").json()["setup_needed"] is True

    def test_key_only_in_the_environment(self, server, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")
        assert server.get("/api/settings").json()["setup_needed"] is False

    def test_ollama(self, server, config_file):
        config_file.write_text("provider: ollama\n", encoding="utf-8")
        assert server.get("/api/settings").json()["setup_needed"] is False

    @pytest.mark.parametrize("provider", ["anthropic", "openai"])
    def test_oauth(self, server, config_file, provider):
        config_file.write_text(
            f"provider: {provider}\n{provider}_auth_mode: oauth\n", encoding="utf-8"
        )
        assert server.get("/api/settings").json()["setup_needed"] is False

    def test_custom_endpoint_needs_a_base_url(self, server, config_file):
        config_file.write_text(
            "provider: custom-openai\ncustom_openai_api_key: k\n", encoding="utf-8"
        )
        assert server.get("/api/settings").json()["setup_needed"] is True

    def test_minimax_has_a_default_address(self, server, config_file):
        config_file.write_text(
            "provider: minimax\nminimax_api_key: k\n", encoding="utf-8"
        )
        assert server.get("/api/settings").json()["setup_needed"] is False


class TestSources:
    def test_file_env_default(self, server, config_file, monkeypatch):
        config_file.write_text("openai_api_key: sk-file\n", encoding="utf-8")
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-env")
        sources = server.get("/api/settings").json()["sources"]
        assert sources["openai_api_key"] == {"source": "file", "shadowed": False}
        assert sources["tavily_api_key"] == {"source": "env", "shadowed": True}
        assert sources["anthropic_api_key"] == {"source": "default", "shadowed": False}

    def test_env_shadows_a_saved_key(self, server, config_file, monkeypatch):
        config_file.write_text("openai_api_key: sk-file\n", encoding="utf-8")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        sources = server.get("/api/settings").json()["sources"]
        assert sources["openai_api_key"]["shadowed"] is True


class TestPatchAuth:
    @pytest.mark.parametrize("headers", [{}, {local_secret.HEADER: "wrong"}])
    def test_refused_without_the_secret(self, server, config_file, headers):
        config_file.write_text("model: m1\n", encoding="utf-8")
        before = config_file.read_bytes()
        response = server.patch("/api/settings", json={"model": "m2"}, headers=headers)
        assert response.status_code == 403
        assert response.json() == {"error": "local_secret_required"}
        assert config_file.read_bytes() == before

    def test_test_key_needs_the_secret(self, server):
        response = server.post(
            "/api/settings/test-key", json={"provider": "openai", "key": "sk-x"}
        )
        assert response.status_code == 403


class TestPatch:
    def test_invalid_field_writes_nothing(self, server, secret, config_file):
        config_file.write_text("model: m1\n", encoding="utf-8")
        before = config_file.read_bytes()
        response = server.patch(
            "/api/settings",
            json={"show_thinking": False, "provider": "nope"},
            headers=secret,
        )
        assert response.status_code == 400
        assert response.json() == {"errors": {"provider": "unknown provider"}}
        assert config_file.read_bytes() == before

    @pytest.mark.parametrize("field", sorted(settings_api.READ_ONLY_FIELDS))
    def test_read_only_fields(self, server, secret, field):
        response = server.patch("/api/settings", json={field: "x"}, headers=secret)
        assert response.status_code == 400
        assert "read-only" in response.json()["errors"][field]

    def test_display_only_field_needs_no_restart(self, server, secret, config_file):
        server.get("/api/settings")  # the baseline, as the server started
        response = server.patch(
            "/api/settings", json={"show_thinking": False}, headers=secret
        )
        assert response.status_code == 200
        body = response.json()
        assert body["restart_required"] is False
        assert body["changed"] == ["show_thinking"]
        assert body["revision"] == store.revision_of(config_file.read_bytes())

    @pytest.mark.parametrize(
        ("change", "field"),
        [
            ({"tavily_api_key": "tvly-new-key-123"}, "tavily_api_key"),
            ({"langgraph_dev_port": 8124}, "langgraph_dev_port"),
            ({"default_workdir": "/"}, "default_workdir"),
        ],
    )
    def test_server_fields_need_a_restart(self, server, secret, change, field):
        server.get("/api/settings")
        with _no_checks():
            response = server.patch("/api/settings", json=change, headers=secret)
        assert response.status_code == 200
        assert response.json()["restart_required"] is True
        assert field in response.json()["restart_fields"]

    def test_pending_restart_survives_a_reload(self, server, secret):
        server.get("/api/settings")
        server.patch("/api/settings", json={"langgraph_dev_port": 8124}, headers=secret)
        body = server.get("/api/settings").json()
        assert body["restart_fields"] == ["langgraph_dev_port"]

    def test_null_removes_and_empty_keeps_a_secret(self, server, secret, config_file):
        config_file.write_text("openai_api_key: sk-saved-123456\n", encoding="utf-8")
        response = server.patch(
            "/api/settings", json={"openai_api_key": ""}, headers=secret
        )
        assert response.status_code == 200
        assert _mapping(config_file)["openai_api_key"] == "sk-saved-123456"
        response = server.patch(
            "/api/settings", json={"openai_api_key": None}, headers=secret
        )
        assert response.status_code == 200
        assert "openai_api_key" not in _mapping(config_file)
        settings = server.get("/api/settings").json()["settings"]
        assert settings["openai_api_key"] == {"set": False}

    def test_null_removes_a_plain_field(self, server, secret, config_file):
        config_file.write_text("mirror: cn\n", encoding="utf-8")
        server.patch("/api/settings", json={"mirror": None}, headers=secret)
        assert _mapping(config_file) == {}

    def test_stale_revision_is_refused(self, server, secret, config_file):
        config_file.write_text("model: m1\n", encoding="utf-8")
        old = server.get("/api/settings").json()["revision"]
        config_file.write_text("model: m1\nshow_thinking: false\n", encoding="utf-8")
        response = server.patch(
            "/api/settings", json={"model": "m2", "revision": old}, headers=secret
        )
        assert response.status_code == 409
        assert response.json() == {
            "error": "revision_mismatch",
            "revision": store.revision_of(config_file.read_bytes()),
        }
        assert _mapping(config_file)["model"] == "m1"

    def test_current_revision_is_accepted(self, server, secret, config_file):
        config_file.write_text("model: m1\n", encoding="utf-8")
        current = server.get("/api/settings").json()["revision"]
        response = server.patch(
            "/api/settings", json={"model": "m2", "revision": current}, headers=secret
        )
        assert response.status_code == 200

    def test_unparseable_file_is_never_overwritten(self, server, secret, config_file):
        broken = b"model: m1\n  bad: [\n"
        config_file.write_bytes(broken)
        response = server.patch("/api/settings", json={"model": "m2"}, headers=secret)
        assert response.status_code == 409
        assert response.json()["error"] == "config_unreadable"
        assert config_file.read_bytes() == broken

    def test_locked_by_the_wizard(self, server, secret):
        holding = threading.Event()
        done = threading.Event()

        def hold():
            with store.config_lock():
                holding.set()
                done.wait(10)

        t = threading.Thread(target=hold)
        t.start()
        try:
            holding.wait(5)
            response = server.patch(
                "/api/settings", json={"model": "m2"}, headers=secret
            )
        finally:
            done.set()
            t.join()
        assert response.status_code == 409
        assert response.json()["error"] == "config_locked"

    def test_invalid_key_is_rejected(self, server, secret, config_file):
        with patch.object(
            settings_api,
            "run_check",
            return_value=KeyCheck("invalid", "Invalid API key"),
        ):
            response = server.patch(
                "/api/settings", json={"openai_api_key": "sk-bad"}, headers=secret
            )
        assert response.status_code == 400
        assert response.json() == {"errors": {"openai_api_key": "Invalid API key"}}
        assert not config_file.exists()

    def test_inconclusive_key_is_saved_with_a_warning(
        self, server, secret, config_file
    ):
        with patch.object(
            settings_api,
            "run_check",
            return_value=KeyCheck("inconclusive", "timed out for sk-offline-1"),
        ):
            response = server.patch(
                "/api/settings", json={"openai_api_key": "sk-offline-1"}, headers=secret
            )
        assert response.status_code == 200
        assert _mapping(config_file)["openai_api_key"] == "sk-offline-1"
        warning = response.json()["warnings"]["openai_api_key"]
        assert "sk-offline-1" not in warning

    def test_unchanged_key_is_not_checked(self, server, secret, config_file):
        config_file.write_text("openai_api_key: sk-same\n", encoding="utf-8")
        with patch.object(settings_api, "run_check") as check:
            server.patch(
                "/api/settings", json={"openai_api_key": "sk-same"}, headers=secret
            )
        check.assert_not_called()

    def test_minimax_key_is_checked_against_the_new_base_url(self, server, secret):
        with _no_checks() as check:
            server.patch(
                "/api/settings",
                json={
                    "minimax_api_key": "mm-key",
                    "minimax_base_url": "https://mm.test",
                },
                headers=secret,
            )
        assert check.call_args.args[1:] == ("mm-key", "https://mm.test")

    def test_default_workdir_warning(self, server, secret, tmp_path):
        response = server.patch(
            "/api/settings",
            json={"default_workdir": str(tmp_path / "new")},
            headers=secret,
        )
        assert response.status_code == 200
        assert "created at start" in response.json()["warnings"]["default_workdir"]

    def test_not_an_object(self, server, secret):
        response = server.patch("/api/settings", json=["x"], headers=secret)
        assert response.status_code == 400


class TestTestKey:
    @pytest.mark.parametrize("name", sorted(KEY_SPECS))
    def test_calls_the_providers_check(self, server, secret, name):
        spec = KEY_SPECS[name]
        with patch.object(
            provider_keys, spec.check, return_value=KeyCheck("valid", "Valid")
        ) as check:
            response = server.post(
                "/api/settings/test-key",
                json={"provider": name, "key": "the-key"},
                headers=secret,
            )
        assert response.status_code == 200
        assert check.call_args.args[0] == "the-key"

    def test_wrong_key(self, server, secret):
        with patch.object(
            provider_keys,
            "check_openai_key",
            return_value=KeyCheck("invalid", "Invalid API key"),
        ):
            body = server.post(
                "/api/settings/test-key",
                json={"provider": "openai", "key": "sk-wrong"},
                headers=secret,
            ).json()
        assert body == {"ok": False, "status": "invalid", "message": "Invalid API key"}

    def test_stored_effective_key_is_used_and_never_returned(
        self, server, secret, monkeypatch
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env-123")
        with patch.object(
            provider_keys,
            "check_openai_key",
            return_value=KeyCheck("inconclusive", "Error: bad key sk-from-env-123"),
        ) as check:
            response = server.post(
                "/api/settings/test-key", json={"provider": "openai"}, headers=secret
            )
        check.assert_called_once_with("sk-from-env-123")
        assert "sk-from-env-123" not in response.text
        assert response.json()["ok"] is None

    def test_empty_key_is_not_ok(self, server, secret):
        body = server.post(
            "/api/settings/test-key", json={"provider": "openai"}, headers=secret
        ).json()
        assert body["status"] == "empty"
        assert body["ok"] is None

    @pytest.mark.parametrize(
        "base_url", ["file:///etc/passwd", "ftp://x", "localhost:11434", 5]
    )
    def test_base_url_must_be_http(self, server, secret, base_url):
        response = server.post(
            "/api/settings/test-key",
            json={"provider": "ollama", "base_url": base_url},
            headers=secret,
        )
        assert response.status_code == 400
        assert "base_url" in response.json()["errors"]

    def test_base_url_only_where_the_check_takes_one(self, server, secret):
        response = server.post(
            "/api/settings/test-key",
            json={"provider": "openai", "key": "k", "base_url": "https://x.test"},
            headers=secret,
        )
        assert response.status_code == 400

    def test_ollama_lists_models(self, server, secret):
        with patch(
            "EvoScientist.llm.ollama_discovery.validate_ollama_connection",
            return_value=(True, "Connected", ["llama3.2"]),
        ) as probe:
            body = server.post(
                "/api/settings/test-key",
                json={"provider": "ollama", "base_url": "http://localhost:11434"},
                headers=secret,
            ).json()
        probe.assert_called_once_with("http://localhost:11434")
        assert body["models"] == ["llama3.2"]
        assert body["ok"] is True

    def test_ollama_without_base_url_probes_the_default(self, server, secret):
        with patch(
            "EvoScientist.llm.ollama_discovery.validate_ollama_connection",
            return_value=(True, "Connected", []),
        ) as probe:
            server.post(
                "/api/settings/test-key", json={"provider": "ollama"}, headers=secret
            )
        probe.assert_called_once_with("http://localhost:11434")

    def test_unknown_provider(self, server, secret):
        response = server.post(
            "/api/settings/test-key", json={"provider": "nope"}, headers=secret
        )
        assert response.status_code == 400
        assert response.json() == {"errors": {"provider": "unknown provider"}}

    def test_nothing_is_saved(self, server, secret, config_file):
        with patch.object(
            provider_keys, "check_openai_key", return_value=KeyCheck("valid", "Valid")
        ):
            server.post(
                "/api/settings/test-key",
                json={"provider": "openai", "key": "sk-x"},
                headers=secret,
            )
        assert not config_file.exists()
