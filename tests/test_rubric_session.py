"""Client-side pieces of ``/rubric``: the checklist text reader and the session store."""

from __future__ import annotations

import pytest

from EvoScientist.rubric_text import RubricTextError, read_rubric_text


class TestReadRubricText:
    def test_inline_text_is_stripped_and_kept_multiline(self):
        assert read_rubric_text("  - a\n- b  \n") == "- a\n- b"

    def test_reads_a_relative_file_from_the_base_dir_and_drops_a_bom(self, tmp_path):
        (tmp_path / "r.md").write_text(chr(0xFEFF) + "- a\n", encoding="utf-8")
        assert read_rubric_text("@r.md", base_dir=tmp_path) == "- a"

    def test_an_absolute_path_ignores_the_base_dir(self, tmp_path):
        target = tmp_path / "abs.md"
        target.write_text("- abs\n", encoding="utf-8")
        assert (
            read_rubric_text(f"@{target}", base_dir=tmp_path / "elsewhere") == "- abs"
        )

    def test_relative_path_without_base_dir_uses_the_cwd(self, tmp_path, monkeypatch):
        (tmp_path / "cwd.md").write_text("- cwd\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        assert read_rubric_text("@cwd.md") == "- cwd"

    def test_empty_text_and_empty_file_raise(self, tmp_path):
        with pytest.raises(RubricTextError, match="is empty"):
            read_rubric_text("   ")
        (tmp_path / "blank.md").write_text(" \n", encoding="utf-8")
        with pytest.raises(RubricTextError, match="is empty"):
            read_rubric_text("@blank.md", base_dir=tmp_path)

    def test_missing_directory_and_binary_files_raise_could_not_read(self, tmp_path):
        with pytest.raises(RubricTextError, match="could not read"):
            read_rubric_text("@missing.md", base_dir=tmp_path)
        with pytest.raises(RubricTextError, match="could not read"):
            read_rubric_text("@.", base_dir=tmp_path)
        (tmp_path / "bin").write_bytes(b"\xff\xfe\x00bad")
        with pytest.raises(RubricTextError, match="could not read"):
            read_rubric_text("@bin", base_dir=tmp_path)

    def test_unknown_user_home_raises_could_not_read(self, tmp_path):
        with pytest.raises(RubricTextError, match="could not read"):
            read_rubric_text("@~no-such-user-xyz/r.md", base_dir=tmp_path)


class TestRubricSession:
    def _session(self):
        from EvoScientist.rubric_session import RubricSession

        return RubricSession()

    def test_operations_carry_the_session_id_and_a_growing_sequence(self):
        session = self._session()
        first = session.record("t", sticky="- a")
        second = session.record("t", next="- b")
        third = session.record("u", sticky=None, next=None)
        assert first["session"] == second["session"] == third["session"]
        assert [first["seq"], second["seq"], third["seq"]] == [1, 2, 3]
        assert session.changes_for("t") == [
            {"session": first["session"], "seq": 1, "sticky": "- a"},
            {"session": first["session"], "seq": 2, "next": "- b"},
        ]
        assert session.changes_for("u") == [
            {"session": first["session"], "seq": 3, "sticky": None, "next": None}
        ]
        assert session.changes_for("v") == []

    def test_acknowledge_drops_only_operations_the_thread_applied(self):
        session = self._session()
        session.record("t", sticky="- a")
        session.record("t", next="- b")
        session.record("t", sticky="- c")
        session.acknowledge("t", {session.session_id: 2})
        assert [op["seq"] for op in session.changes_for("t")] == [3]
        session.acknowledge("t", {"another-session": 99})
        assert [op["seq"] for op in session.changes_for("t")] == [3]
        session.acknowledge("t", {})
        assert [op["seq"] for op in session.changes_for("t")] == [3]
        session.acknowledge("t", {session.session_id: 3})
        assert session.changes_for("t") == []

    def test_changes_are_copies(self):
        session = self._session()
        session.record("t", sticky="- a")
        session.changes_for("t")[0]["sticky"] = "mutated"
        assert session.changes_for("t")[0]["sticky"] == "- a"


class TestConfigurableExtra:
    def test_rubric_extra_carries_mode_and_the_threads_changes(self):
        from EvoScientist.rubric_session import RubricSession, rubric_configurable_extra

        session = RubricSession()
        session.record("t", next="- n")
        session.record("u", sticky="- s")
        extra = rubric_configurable_extra(session, "t", "thread")
        assert extra == {
            "rubric_run": {
                "mode": "thread",
                "changes": [{"session": session.session_id, "seq": 1, "next": "- n"}],
            }
        }
        assert rubric_configurable_extra(session, "v", "followup") == {
            "rubric_run": {"mode": "followup", "changes": []}
        }

    def test_a_notifier_turn_carries_no_changes_and_leaves_them_pending(self):
        from EvoScientist.rubric_session import RubricSession, rubric_configurable_extra

        session = RubricSession()
        session.record("t", sticky="- b")
        assert rubric_configurable_extra(session, "t", "followup") == {
            "rubric_run": {"mode": "followup", "changes": []}
        }
        assert [op["sticky"] for op in session.changes_for("t")] == ["- b"]

    def test_merge_skips_empty_parts_and_returns_none_when_nothing_is_left(self):
        from EvoScientist.rubric_session import merge_configurable_extra

        assert merge_configurable_extra(None, {"a": 1}, {}, {"b": 2}) == {
            "a": 1,
            "b": 2,
        }
        assert merge_configurable_extra(None, {}) is None
        assert merge_configurable_extra() is None


class TestRubricBadge:
    def test_badge_from_thread_values(self):
        from EvoScientist.rubric_session import rubric_badge

        assert rubric_badge({}, []) is None
        assert (
            rubric_badge({"_sticky_rubric": {"text": "- s", "id": "x:1"}}, []) == "on"
        )
        assert (
            rubric_badge({"_next_rubric": {"text": "- n", "id": "x:2"}}, []) == "next"
        )
        both = {
            "_sticky_rubric": {"text": "- s", "id": "x:1"},
            "_next_rubric": {"text": "- n", "id": "x:2"},
        }
        assert rubric_badge(both, []) == "on+next"

    def test_pending_operations_overlay_the_thread_values(self):
        from EvoScientist.rubric_session import rubric_badge

        pending_set = [{"session": "x", "seq": 1, "sticky": "- s"}]
        assert rubric_badge({}, pending_set) == "on"
        pending_clear = [{"session": "x", "seq": 2, "sticky": None, "next": None}]
        assert (
            rubric_badge(
                {"_sticky_rubric": {"text": "- s", "id": "x:1"}}, pending_clear
            )
            is None
        )
        in_order = [
            {"session": "x", "seq": 1, "sticky": None, "next": None},
            {"session": "x", "seq": 2, "next": "- n"},
        ]
        assert (
            rubric_badge({"_sticky_rubric": {"text": "- s", "id": "x:0"}}, in_order)
            == "next"
        )


class TestValuesForBadge:
    def test_rebuilds_the_two_slots_from_a_badge(self):
        from EvoScientist.rubric_session import rubric_badge, values_for_badge

        for badge in (None, "on", "next", "on+next"):
            assert rubric_badge(values_for_badge(badge), []) == badge
        pending = [{"session": "x", "seq": 1, "next": "- n"}]
        assert rubric_badge(values_for_badge("on"), pending) == "on+next"


# -- helpers shared by the interactive CLI and the TUI -------------------------


def _server_gateway():
    from EvoScientist.gateway.server import LangGraphServerGateway

    return object.__new__(LangGraphServerGateway)


def test_runs_in_process_goes_by_the_gateway_that_executes():
    from types import SimpleNamespace

    from EvoScientist.rubric_session import runs_in_process

    assert runs_in_process(object()) is True
    assert runs_in_process(_server_gateway()) is False
    # A routing gateway is judged by the one it executes on.
    assert runs_in_process(SimpleNamespace(execute_gateway=_server_gateway())) is False
    assert runs_in_process(SimpleNamespace(execute_gateway=object())) is True


def test_only_the_sessions_own_thread_on_the_local_gateway_is_marked():
    from EvoScientist.rubric_session import RubricSession, thread_run_extra

    session = RubricSession()
    session.record("mine", sticky="- a")

    def extra(gateway=None, *, thread_id="mine", current="mine", mode="thread"):
        return thread_run_extra(
            session,
            gateway if gateway is not None else object(),
            thread_id=thread_id,
            current_thread_id=current,
            mode=mode,
        )

    marked = extra()
    assert marked["rubric_run"]["mode"] == "thread"
    assert marked["rubric_run"]["changes"] == session.changes_for("mine")
    assert extra(mode="followup")["rubric_run"]["mode"] == "followup"
    assert extra(thread_id="other") is None
    assert extra(thread_id=None) is None
    assert extra(mode=None) is None
    assert extra(mode="") is None
    assert extra(_server_gateway()) is None


async def test_acknowledge_applied_drops_operations_and_sets_the_badge(tmp_path):
    from EvoScientist.paths import SessionDirs, Workspace
    from EvoScientist.rubric_session import RubricSession, acknowledge_applied
    from tests.fakes import FakeGraphGateway

    session = RubricSession()
    session.record("t", sticky="- a")
    (operation,) = session.changes_for("t")
    gateway = FakeGraphGateway(
        state_values={
            "_sticky_rubric": {"text": "- a", "id": "x"},
            "_rubric_applied": {operation["session"]: operation["seq"]},
        }
    )
    badges: dict[str, str | None] = {}
    dirs = SessionDirs(Workspace(tmp_path))

    loaded = object()

    async def ack(agent=loaded, thread_id="t"):
        await acknowledge_applied(
            session, badges, gateway, agent=agent, dirs=dirs, thread_id=thread_id
        )

    await ack()
    assert session.changes_for("t") == []
    assert badges == {"t": "on"}

    # An unreadable state, or no agent to read it with, keeps the last badge.
    gateway.state_error = RuntimeError("database is locked")
    await ack()
    assert badges == {"t": "on"}
    gateway.state_error = None
    gateway.state_values = {}
    await ack(agent=None)
    assert badges == {"t": "on"}
    await ack()
    assert badges == {"t": None}
    await ack(thread_id=None)
    assert badges == {"t": None}
