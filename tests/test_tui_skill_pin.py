"""TUI handling of ``/skill-name`` input: pass-through, live label and history.

These tests boot the real TUI class through Textual's pilot (see
``test_tui_banner_position.capture_app``) and drive the submit handler, the
streaming turn and the history renderer with a fake graph gateway.
"""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage

pytest.importorskip("textual")

from tests.fakes.tui import (
    capture_app,
    finish_turns,
    isolated_skills,  # noqa: F401  (autouse fixture)
    prepare_app,
    submit,
    system_lines,
)


def _write_skill(workspace, name, description=None):
    folder = workspace.skills_dir / name
    folder.mkdir(parents=True)
    description = f"Skill {name}." if description is None else description
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\nBody.\n",
        encoding="utf-8",
    )


def _gateway(messages=None):
    from tests.fakes.common import FakeGraphGateway, FakeThreadStore

    return FakeGraphGateway(
        events=[{"type": "text", "content": "ok"}, {"type": "done", "response": "ok"}],
        thread_store=FakeThreadStore(messages=messages or []),
    )


def _skill_labels(app) -> list[str]:
    return [line for line in system_lines(app) if line.startswith("skill:")]


def _user_lines(app) -> list[str]:
    from EvoScientist.cli.widgets.user_message import UserMessage

    return [
        str(w.render())
        for w in app.query_one("#chat").children
        if isinstance(w, UserMessage)
    ]


async def test_skill_slash_runs_a_turn_with_one_label(monkeypatch, workspace):
    _write_skill(workspace, "alpha")
    _write_skill(workspace, "model")  # clashes with the /model command

    app = await capture_app(monkeypatch, workspace)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        gateway = _gateway()
        prepare_app(app, workspace, gateway)
        commands: list[str] = []

        async def _record(text, _token):
            commands.append(text)

        app._handle_command = _record

        await submit(app, "/alpha go")
        await finish_turns(app, pilot)
        await pilot.pause()

        assert [r.message for r in gateway.requests] == ["/alpha go"]
        assert commands == []
        assert _skill_labels(app) == ["skill: alpha"]
        assert _user_lines(app) == ["> /alpha go"]


async def test_commands_typos_and_case_go_to_the_command_handler(
    monkeypatch, workspace
):
    _write_skill(workspace, "alpha")
    _write_skill(workspace, "model")  # clashes with the /model command

    app = await capture_app(monkeypatch, workspace)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        gateway = _gateway()
        prepare_app(app, workspace, gateway)
        commands: list[str] = []

        async def _record(text, _token):
            commands.append(text)
            # The real handler owns the app while it runs and releases it.
            app._busy = False

        app._handle_command = _record

        sent = ["/model", "/model gpt", "/typo hi", "/new", "/Alpha go"]
        for text in sent:
            await submit(app, text)
            if app._background_tasks:
                await asyncio.gather(*list(app._background_tasks))
        await pilot.pause()

        assert sorted(commands) == sorted(sent)
        assert gateway.requests == []
        assert app._run_task is None
        assert _skill_labels(app) == []


async def test_queued_skill_gets_one_label_on_replay(monkeypatch, workspace):
    _write_skill(workspace, "alpha")

    app = await capture_app(monkeypatch, workspace)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        gateway = _gateway()
        prepare_app(app, workspace, gateway)

        await submit(app, "/alpha first")
        await finish_turns(app, pilot)
        await pilot.pause()
        before = _skill_labels(app).count("skill: alpha")
        assert before == 1

        # Sent while a turn is running: queued, not labeled yet.
        app._busy = True
        await submit(app, "/alpha queued")
        app._busy = False
        assert app._queued_messages == ["/alpha queued"]
        assert _skill_labels(app).count("skill: alpha") == before

        # The next turn finishes and replays the queued message.
        await submit(app, "plain")
        await finish_turns(app, pilot)
        await pilot.pause()

        assert _skill_labels(app).count("skill: alpha") == before + 1
        assert [r.message for r in gateway.requests] == [
            "/alpha first",
            "plain",
            "/alpha queued",
        ]
        assert app._queued_messages == []


async def test_history_shows_the_label_not_the_skill_body(monkeypatch, workspace):
    _write_skill(workspace, "alpha")

    pinned = HumanMessage(
        "<skill>SKILL BODY</skill>",
        additional_kwargs={"lc_source": "pinned_skill", "skill": {"name": "alpha"}},
    )
    gateway = _gateway(messages=[HumanMessage("/alpha go"), pinned, AIMessage("Done.")])

    app = await capture_app(monkeypatch, workspace)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        prepare_app(app, workspace, gateway)

        await app._render_history("tid")
        await pilot.pause()

        assert _skill_labels(app) == ["skill: alpha"]
        assert _user_lines(app) == ["> /alpha go"]
        rendered = system_lines(app) + _user_lines(app)
        assert not any("SKILL BODY" in text for text in rendered)


async def test_typing_slash_with_a_non_string_description_does_not_crash(
    monkeypatch, workspace
):
    _write_skill(workspace, "odd", description="42")

    app = await capture_app(monkeypatch, workspace)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        prepare_app(app, workspace, _gateway())

        prompt = app.query_one("#prompt")
        prompt.focus()
        await pilot.press("slash")
        await pilot.pause()
        await pilot.press("n", "e", "w")
        await pilot.pause()

        assert app.is_running
        assert prompt.value == "/new"


async def test_history_shows_grader_feedback_as_a_label(monkeypatch, workspace):
    feedback = HumanMessage(
        "A grader reviewed your work and asked for revisions.",
        additional_kwargs={"lc_source": "rubric_grader"},
    )
    gateway = _gateway(
        messages=[
            HumanMessage("write the report"),
            AIMessage("Draft."),
            feedback,
            AIMessage("Revised."),
        ]
    )

    app = await capture_app(monkeypatch, workspace)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        prepare_app(app, workspace, gateway)

        await app._render_history("tid")
        await pilot.pause()

        assert "rubric: revision requested" in system_lines(app)
        assert _user_lines(app) == ["> write the report"]
        rendered = system_lines(app) + _user_lines(app)
        assert not any("A grader reviewed" in text for text in rendered)
