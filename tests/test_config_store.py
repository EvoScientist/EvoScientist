"""Tests for the config file's write path (EvoScientist.config.store) and the
shared setting validation in EvoScientist.config.settings."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from dataclasses import fields
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from EvoScientist.config import store
from EvoScientist.config.settings import (
    API_KEY_FIELDS,
    SECRET_FIELDS,
    EvoScientistConfig,
    SettingsError,
    display_config_value,
    get_config_path,
    load_config,
    reset_config,
    save_config,
    secret_hint,
    set_config_value,
    set_config_values,
    validate_setting,
)

BROKEN = b"provider: openai\nopenai_api_key: sk-secret-value\n  bad: [\n"


@pytest.fixture
def config_file() -> Path:
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _mapping(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


class TestParseGuard:
    def test_set_config_value_never_overwrites_an_unparseable_file(self, config_file):
        config_file.write_bytes(BROKEN)
        with pytest.raises(store.ConfigParseError):
            set_config_value("show_thinking", "false")
        assert config_file.read_bytes() == BROKEN

    def test_save_config_never_overwrites_an_unparseable_file(self, config_file):
        config_file.write_bytes(BROKEN)
        with pytest.raises(store.ConfigParseError):
            save_config(load_config())
        assert config_file.read_bytes() == BROKEN

    def test_error_names_line_and_column_but_not_the_content(self, config_file):
        config_file.write_bytes(BROKEN)
        with pytest.raises(store.ConfigParseError) as info:
            store.read_config_mapping()
        assert info.value.line is not None
        assert "sk-secret" not in str(info.value)
        assert "EvoSci config reset" in str(info.value)

    def test_a_top_level_list_is_refused(self, config_file):
        config_file.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(store.ConfigParseError):
            set_config_value("show_thinking", "false")

    def test_reset_works_on_an_unparseable_file(self, config_file):
        config_file.write_bytes(BROKEN)
        reset_config()
        assert not config_file.exists()

    @pytest.mark.parametrize("content", [b"", b"# only a comment\n"])
    def test_empty_file_is_an_empty_mapping(self, config_file, content):
        config_file.write_bytes(content)
        assert set_config_value("show_thinking", "false")
        assert _mapping(config_file) == {"show_thinking": False}


class TestUpdate:
    def test_only_changed_keys_are_written_and_unknown_keys_survive(self, config_file):
        config_file.write_text("model: m1\nfuture_field: 7\n", encoding="utf-8")
        set_config_values({"show_thinking": False})
        assert _mapping(config_file) == {
            "model": "m1",
            "future_field": 7,
            "show_thinking": False,
        }

    def test_none_removes_the_key(self, config_file):
        config_file.write_text("model: m1\nmirror: cn\n", encoding="utf-8")
        set_config_values({"mirror": None})
        assert _mapping(config_file) == {"model": "m1"}
        assert load_config().mirror == "default"

    def test_nothing_is_written_when_one_field_fails(self, config_file):
        config_file.write_text("model: m1\n", encoding="utf-8")
        before = config_file.read_bytes()
        with pytest.raises(SettingsError) as info:
            set_config_values(
                {"show_thinking": False, "provider": "nope", "no_such": 1}
            )
        assert set(info.value.errors) == {"provider", "no_such"}
        assert config_file.read_bytes() == before

    def test_returns_the_new_revision(self, config_file):
        revision, _ = set_config_values({"model": "m2"})
        assert revision == store.revision_of(config_file.read_bytes())

    def test_stale_revision_is_refused(self, config_file):
        config_file.write_text("model: m1\n", encoding="utf-8")
        old = store.revision_of(config_file.read_bytes())
        set_config_value("show_thinking", "false")
        before = config_file.read_bytes()
        with pytest.raises(store.RevisionMismatchError) as info:
            set_config_values({"model": "m2"}, expected_revision=old)
        assert info.value.current == store.revision_of(before)
        assert config_file.read_bytes() == before

    def test_current_revision_is_accepted(self, config_file):
        config_file.write_text("model: m1\n", encoding="utf-8")
        current = store.revision_of(config_file.read_bytes())
        set_config_values({"model": "m2"}, expected_revision=current)
        assert _mapping(config_file)["model"] == "m2"

    def test_none_revision_means_no_file_expected(self, config_file):
        set_config_values({"model": "m2"}, expected_revision=None)
        with pytest.raises(store.RevisionMismatchError):
            set_config_values({"model": "m3"}, expected_revision=None)

    def test_concurrent_writers_of_different_fields_both_apply(
        self, config_file, monkeypatch
    ):
        monkeypatch.setattr(store, "_LOCK_TIMEOUT", 30)
        config_file.write_text("model: m1\n", encoding="utf-8")
        errors = []

        def write(key, value):
            try:
                for _ in range(20):
                    set_config_values({key: value})
            except Exception as e:  # pragma: no cover - reported below
                errors.append(e)

        threads = [
            threading.Thread(target=write, args=("show_thinking", False)),
            threading.Thread(target=write, args=("mirror", "cn")),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
            assert not t.is_alive(), "a writer is stuck"
        assert not errors
        data = _mapping(config_file)
        assert data["show_thinking"] is False
        assert data["mirror"] == "cn"
        assert data["model"] == "m1"


class TestAtomicWrite:
    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")
    def test_file_is_0600(self, config_file):
        set_config_value("model", "m1")
        assert config_file.stat().st_mode & 0o777 == 0o600

    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges")
    def test_symlinked_config_keeps_its_link(self, config_file, tmp_path):
        real = tmp_path / "real.yaml"
        real.write_text("model: m1\n", encoding="utf-8")
        config_file.symlink_to(real)
        set_config_value("show_thinking", "false")
        assert config_file.is_symlink()
        assert _mapping(real)["show_thinking"] is False

    def test_transient_permission_error_is_retried(self, config_file):
        real_replace = os.replace
        calls = []

        def flaky(src, dst):
            calls.append(src)
            if len(calls) < 3:
                raise PermissionError("in use")
            real_replace(src, dst)

        with (
            patch("EvoScientist.config.store.os.replace", side_effect=flaky),
            patch("EvoScientist.config.store.time.sleep"),
        ):
            set_config_value("model", "m1")
        assert len(calls) == 3
        assert _mapping(config_file)["model"] == "m1"

    def test_persistent_permission_error_leaves_no_temp_file(self, config_file):
        config_file.write_text("model: m1\n", encoding="utf-8")
        with (
            patch(
                "EvoScientist.config.store.os.replace",
                side_effect=PermissionError("in use"),
            ),
            patch("EvoScientist.config.store.time.sleep"),
            pytest.raises(PermissionError),
        ):
            set_config_value("model", "m2")
        assert _mapping(config_file) == {"model": "m1"}
        assert [
            p.name for p in config_file.parent.iterdir() if p.suffix == ".tmp"
        ] == []


class TestLock:
    def test_lock_is_reentrant_on_one_thread(self, config_file):
        with store.config_lock(), store.config_lock():
            set_config_value("model", "m1")
        assert _mapping(config_file)["model"] == "m1"

    def test_writer_is_refused_while_another_process_holds_the_lock(self, config_file):
        lock_path = str(config_file.with_name("config.yaml.lock"))
        holder = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import sys, time; from filelock import FileLock\n"
                f"lock = FileLock({lock_path!r}); lock.acquire()\n"
                "print('held', flush=True); time.sleep(30)",
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            assert holder.stdout.readline().strip() == "held"
            start = time.monotonic()
            with pytest.raises(store.ConfigLockedError) as info:
                set_config_value("model", "m1")
            assert time.monotonic() - start < 10
            assert "EvoSci onboard" in str(info.value)
        finally:
            holder.kill()
            holder.wait()
        assert not config_file.exists()


class TestValidateSetting:
    def test_unknown_setting(self):
        with pytest.raises(ValueError, match="unknown setting"):
            validate_setting("no_such", 1)

    def test_unknown_provider(self):
        with pytest.raises(ValueError, match="unknown provider"):
            validate_setting("provider", "nope")

    def test_auxiliary_provider_may_be_empty(self):
        assert validate_setting("auxiliary_provider", "") == ("", None)

    def test_literal_lists_the_allowed_values(self):
        with pytest.raises(ValueError, match="expected one of"):
            validate_setting("mirror", "eu")

    def test_int_field(self):
        with pytest.raises(ValueError, match="expected int"):
            validate_setting("langgraph_dev_port", "abc")
        assert validate_setting("langgraph_dev_port", "8123") == (8123, None)

    @pytest.mark.parametrize(
        ("key", "value", "reason"),
        [
            ("sandbox_execute_timeout", 0, "greater than 0"),
            ("sandbox_execute_timeout", True, "expected a number"),
            ("memory_skill_synthesis_time", "25:00", "HH:MM"),
            ("proactive_idle_minutes", -1, "0 or more"),
            ("proactive_quiet_hours", "late", "HH:MM-HH:MM"),
            ("proactive_timezone", "Mars/Olympus", "unknown time zone"),
        ],
    )
    def test_existing_checks_still_apply(self, key, value, reason):
        with pytest.raises(ValueError, match=reason):
            validate_setting(key, value)

    def test_default_workdir_must_be_absolute(self):
        with pytest.raises(ValueError, match="absolute"):
            validate_setting("default_workdir", "projects/x")

    def test_default_workdir_must_not_be_a_file(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("x")
        with pytest.raises(ValueError, match="file"):
            validate_setting("default_workdir", str(f))

    def test_missing_default_workdir_is_a_warning(self, tmp_path):
        value, warning = validate_setting("default_workdir", str(tmp_path / "new"))
        assert value == str(tmp_path / "new")
        assert "created at start" in warning

    def test_default_workdir_keeps_the_tilde(self):
        value, _ = validate_setting("default_workdir", "~/research")
        assert value == "~/research"

    def test_empty_default_workdir_means_current_folder(self):
        assert validate_setting("default_workdir", "") == ("", None)


class TestSecrets:
    def test_every_secret_is_a_config_field(self):
        names = {f.name for f in fields(EvoScientistConfig)}
        assert SECRET_FIELDS <= names

    def test_every_api_key_field_is_listed(self):
        names = {f.name for f in fields(EvoScientistConfig)}
        assert {n for n in names if n.endswith("_api_key")} == API_KEY_FIELDS

    def test_mimo_token_plan_base_url_is_not_a_secret(self):
        assert "mimo_token_plan_base_url" not in SECRET_FIELDS

    @pytest.mark.parametrize(
        ("key", "value", "hint"),
        [
            ("openai_api_key", "sk-123456789", "…6789"),
            ("openai_api_key", "sk-12345", None),
            ("email_smtp_password", "a-long-password", None),
            ("telegram_bot_token", "123456:ABCDEFGHIJ", None),
        ],
    )
    def test_hint(self, key, value, hint):
        assert secret_hint(key, value) == hint

    def test_display_masks_channel_secrets(self):
        assert display_config_value("telegram_bot_token", "123456:ABCDEF") == "***"
        assert display_config_value("openai_api_key", "sk-123456789") == "***6789"
        assert display_config_value("model", "m") == "m"
        assert display_config_value("openai_api_key", "") == "(not set)"


def test_config_list_does_not_print_channel_secrets(config_file):
    from typer.testing import CliRunner

    from EvoScientist.cli.commands import config_app

    config_file.write_text(
        "telegram_bot_token: 123456:SECRETTOKEN\nopenai_api_key: sk-abcdefgh1234\n",
        encoding="utf-8",
    )
    import io

    from rich.console import Console

    # The module's rich console is bound at import; render what it was given.
    with patch("EvoScientist.cli.commands.console.print") as printed:
        result = CliRunner().invoke(config_app, ["list"])
    assert result.exit_code == 0, result.output
    out = io.StringIO()
    render = Console(file=out, width=300)
    for call in printed.call_args_list:
        render.print(*call.args)
    text = out.getvalue()
    assert "telegram_bot_token" in text
    assert "SECRETTOKEN" not in text
    assert "sk-abcdefgh" not in text
    assert "***1234" in text


def test_config_set_explains_a_refused_value(config_file):
    from typer.testing import CliRunner

    from EvoScientist.cli.commands import config_app

    # The module's rich console is bound at import, so read what it printed.
    with patch("EvoScientist.cli.commands.console.print") as printed:
        result = CliRunner().invoke(config_app, ["set", "provider", "nope"])
    assert result.exit_code == 1
    assert "unknown provider" in " ".join(
        str(c.args[0]) for c in printed.call_args_list
    )


def test_config_set_explains_a_write_that_keeps_failing(config_file):
    from typer.testing import CliRunner

    from EvoScientist.cli.commands import config_app

    config_file.write_text("model: m1\n", encoding="utf-8")
    with (
        patch.object(store, "_replace", side_effect=PermissionError("in use")),
        patch("EvoScientist.cli.commands.console.print") as printed,
    ):
        result = CliRunner().invoke(config_app, ["set", "model", "m2"])
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "in use" in " ".join(str(c.args[0]) for c in printed.call_args_list)
    assert _mapping(config_file) == {"model": "m1"}


class TestWizard:
    def test_refuses_an_unparseable_file_before_asking(self, config_file):
        from EvoScientist.config.onboard.wizard import run_onboard

        config_file.write_bytes(BROKEN)
        with patch("EvoScientist.config.onboard.wizard._run_onboard") as inner:
            with pytest.raises(store.ConfigParseError):
                run_onboard()
        inner.assert_not_called()
        assert config_file.read_bytes() == BROKEN

    def test_holds_the_lock_while_it_runs(self, config_file):
        from EvoScientist.config.onboard.wizard import run_onboard

        seen = {}

        def inner(**_):
            # Another thread stands in for another writer: it must be refused.
            def other():
                try:
                    set_config_value("model", "from-elsewhere")
                except store.ConfigLockedError:
                    seen["locked"] = True

            t = threading.Thread(target=other)
            t.start()
            t.join()
            # The wizard's own saves re-enter the lock.
            save_config(EvoScientistConfig(model="from-wizard"))
            return True

        with patch(
            "EvoScientist.config.onboard.wizard._run_onboard", side_effect=inner
        ):
            assert run_onboard() is True
        assert seen == {"locked": True}
        assert _mapping(config_file)["model"] == "from-wizard"


@pytest.mark.parametrize("value", ["inf", "-inf", "nan"])
def test_number_must_be_finite(value):
    with pytest.raises(ValueError, match="finite"):
        validate_setting("code_interpreter_timeout", value)


def test_onboard_exits_1_when_refused(config_file):
    from typer.testing import CliRunner

    import EvoScientist.cli.commands  # noqa: F401  registers the commands
    from EvoScientist.cli._app import app

    config_file.write_bytes(BROKEN)
    with patch("EvoScientist.cli.commands.console.print"):
        result = CliRunner().invoke(app, ["onboard", "--non-interactive"])
    assert result.exit_code == 1
    assert config_file.read_bytes() == BROKEN


def test_channel_setup_holds_the_lock_while_it_asks(config_file, monkeypatch):
    from typer.testing import CliRunner

    import EvoScientist.cli.commands  # noqa: F401  registers the commands
    from EvoScientist.cli._app import app

    monkeypatch.setattr(store, "_LOCK_TIMEOUT", 0.2)
    seen = {}

    def step(config, runtime=None):
        def other():
            try:
                set_config_value("model", "from-elsewhere")
            except store.ConfigLockedError:
                seen["locked"] = True

        t = threading.Thread(target=other)
        t.start()
        t.join()
        return {"channel_enabled": "telegram"}

    with (
        patch("EvoScientist.config.onboard.channels._step_channels", side_effect=step),
        patch("EvoScientist.cli.commands.console.print"),
    ):
        result = CliRunner().invoke(app, ["channel", "setup"])
    assert result.exit_code == 0, result.output
    assert seen == {"locked": True}
    data = _mapping(config_file)
    assert data["channel_enabled"] == "telegram"
    assert "model" not in data
