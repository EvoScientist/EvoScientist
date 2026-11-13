"""Media forwarding must not infer host access from a changed config file."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from EvoScientist.backends import MergedSkillsBackend
from EvoScientist.gateway.local import LocalGraphGateway
from EvoScientist.gateway.runtime import sandbox_dangerous_mode


@pytest.mark.parametrize("dangerous", [False, True])
def test_serve_uses_the_config_that_built_its_agent(monkeypatch, dangerous):
    from EvoScientist.cli.channel import ChannelMessage, _register_channel_request
    from EvoScientist.cli.commands import _serve_process_message
    from EvoScientist.config import EvoScientistConfig
    from EvoScientist.runtime import AsyncRuntime
    from tests.test_serve_agent_holder import _runtime_state

    config = EvoScientistConfig(dangerous_mode=dangerous)
    monkeypatch.setattr(
        "EvoScientist.config.get_effective_config",
        lambda: EvoScientistConfig(dangerous_mode=not dangerous),
    )
    state = _runtime_state(config=config)
    msg = ChannelMessage(
        msg_id="mode-change",
        content="hello",
        sender="user",
        channel_type="imessage",
        metadata={},
        channel_ref=None,
        bus_ref=None,
        chat_id="chat",
        message_id="mode-change",
    )
    with (
        AsyncRuntime(thread_name="test-media-mode") as runtime,
        patch(
            "EvoScientist.cli.commands.dispatch_channel_slash_command",
            new=AsyncMock(return_value=False),
        ),
        patch("EvoScientist.cli.commands.build_metadata", return_value={}),
        patch(
            "EvoScientist.cli.tui_runtime.run_streaming", return_value="ok"
        ) as stream,
    ):
        state.async_runtime = runtime
        _register_channel_request(msg)
        _serve_process_message(
            msg, runtime_state=state, model="model", show_thinking=False
        )
    assert stream.call_args.kwargs["dangerous"] is dangerous


def test_unknown_mode_never_grants_host_access(monkeypatch):
    monkeypatch.setattr(
        "EvoScientist.config.get_effective_config",
        lambda: SimpleNamespace(dangerous_mode=True),
    )
    assert sandbox_dangerous_mode(LocalGraphGateway(thread_store=MagicMock())) is False


@pytest.mark.parametrize("fallback", [False, True])
def test_colliding_skill_tiers_are_not_guessed(tmp_path, monkeypatch, fallback):
    primary, secondary = tmp_path / "primary", tmp_path / "secondary"
    for folder, text in ((primary, "workspace"), (secondary, "builtin")):
        folder.mkdir()
        (folder / "result.txt").write_text(text)
    backend = MergedSkillsBackend(str(primary), str(secondary))
    if fallback:
        monkeypatch.setattr(backend._primary, "read", lambda *_: "Error: too large")
    result = backend.read("/result.txt")
    assert ("builtin" if fallback else "workspace") in str(result)
    # Forwarding builds its routes separately, so it cannot know which read
    # succeeded. Returning nothing is safer than forwarding the wrong file.
    assert backend.file_path("/result.txt") is None


def test_single_skill_tier_still_resolves(tmp_path):
    primary, secondary = tmp_path / "primary", tmp_path / "secondary"
    secondary.mkdir()
    file = secondary / "result.txt"
    file.write_text("builtin")
    backend = MergedSkillsBackend(str(primary), str(secondary))
    assert backend.file_path("/result.txt") == file


async def test_tui_config_change_does_not_forward_host_file(
    monkeypatch, workspace, tmp_path
):
    import EvoScientist.backends as backends
    from EvoScientist.config import EvoScientistConfig
    from EvoScientist.paths import SessionDirs
    from tests.fakes.tui import run_turn
    from tests.test_channel_media_forwarding import _file_tool_turn

    original = EvoScientistConfig(dangerous_mode=False)
    monkeypatch.setattr("EvoScientist.config.get_effective_config", lambda: original)
    host = tmp_path / "host"
    host.mkdir()
    private = host / "private.pdf"
    private.write_bytes(b"private host file")
    monkeypatch.setattr(backends, "_SYSTEM_PATH_PREFIXES", (f"{host}/",))
    sent = []

    def prepare(app):
        app._dirs = SessionDirs(workspace)
        monkeypatch.setattr(
            "EvoScientist.config.get_effective_config",
            lambda: EvoScientistConfig(dangerous_mode=True),
        )

    await run_turn(
        monkeypatch,
        workspace,
        [_file_tool_turn("read_file", {"file_path": str(private)})],
        prepare=prepare,
        on_media_cb=sent.append,
    )
    assert sent == []
