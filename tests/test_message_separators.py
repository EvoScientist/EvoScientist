"""Joined responses preserve assistant message boundaries across consumers."""

import pytest
from langchain_core.messages import AIMessage

from EvoScientist.stream.state import StreamState
from tests.fakes.stream_v3 import (
    FakeV3Agent,
    collect_events,
    message_delta,
    message_finish,
    whole_message,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "events",
    [
        [
            message_delta("Checking."),
            message_finish(),
            message_delta("Done."),
            message_finish(),
        ],
        [
            whole_message(AIMessage(content="Checking.")),
            whole_message(AIMessage(content="Done.")),
        ],
    ],
)
async def test_joined_messages_have_blank_line_in_events_done_and_state(events):
    out = await collect_events(FakeV3Agent(events))
    text = "".join(e["content"] for e in out if e["type"] == "text")
    state = StreamState()
    for event in out:
        state.handle_event(event)
    assert text == "Checking.\n\nDone."
    assert next(e for e in out if e["type"] == "done")["response"] == text
    assert state.response_text == text


@pytest.mark.asyncio
async def test_deltas_within_a_message_are_not_separated():
    out = await collect_events(
        FakeV3Agent(
            [
                message_delta("Hel"),
                message_delta("lo"),
                message_finish(),
                message_finish(),
                message_delta("World"),
                message_finish(),
            ]
        )
    )
    assert next(e for e in out if e["type"] == "done")["response"] == "Hello\n\nWorld"


@pytest.mark.asyncio
async def test_retry_discards_separator_with_failed_text_in_every_buffer():
    from EvoScientist.channels.consumer import apply_model_discard
    from tests.fakes.stream_v3 import model_call_end, model_call_start, retry_event

    out = await collect_events(
        FakeV3Agent(
            [
                model_call_start("1"),
                message_delta("Kept."),
                message_finish(),
                model_call_end("1"),
                model_call_start("2"),
                message_delta("Failed."),
                model_call_end("2", "failed"),
                retry_event(),
                model_call_start("3"),
                message_delta("Good."),
                message_finish(),
                model_call_end("3"),
            ]
        )
    )
    state = StreamState()
    reply = ""
    for event in out:
        state.handle_event(event)
        if event["type"] == "text":
            reply += event["content"]
        elif event["type"] in ("model_retry", "model_output_discarded"):
            reply = apply_model_discard(event, reply, {}, [])
    retry = next(e for e in out if e["type"] == "model_retry")
    assert retry["discarded"]["text_chars"] == len("\n\nFailed.")
    assert reply == state.response_text == "Kept.\n\nGood."
    assert next(e for e in out if e["type"] == "done")["response"] == reply


@pytest.mark.asyncio
async def test_whole_message_with_existing_blank_line_is_not_double_spaced():
    out = await collect_events(
        FakeV3Agent(
            [
                whole_message(AIMessage(content="First.\n\n")),
                whole_message(AIMessage(content="Second.")),
            ]
        )
    )
    assert (
        next(e for e in out if e["type"] == "done")["response"] == "First.\n\nSecond."
    )


@pytest.mark.asyncio
async def test_standalone_consumer_sends_separated_response(workspace):
    from unittest.mock import MagicMock

    from EvoScientist.channels.bus.events import InboundMessage
    from EvoScientist.channels.bus.message_bus import MessageBus
    from EvoScientist.channels.channel_manager import ChannelManager
    from EvoScientist.channels.consumer import InboundConsumer
    from tests.fakes.common import FakeGraphGateway, StubChannel

    events = await collect_events(
        FakeV3Agent(
            [
                message_delta("Checking."),
                message_finish(),
                message_delta("Done."),
                message_finish(),
            ]
        )
    )
    # Omit done.content so this also exercises the consumer's own text buffer.
    events = [e for e in events if e["type"] != "done"]
    bus = MessageBus()
    manager = ChannelManager(bus, media_dir=workspace.root / "media")
    manager.register(StubChannel())
    consumer = InboundConsumer(
        bus=bus,
        manager=manager,
        agent=MagicMock(),
        thread_id="",
        workspace=workspace,
        graph_gateway=FakeGraphGateway(events=events),
    )
    msg = InboundMessage(channel="stub", sender_id="u1", chat_id="c1", content="go")
    await consumer._stream_with_hitl(msg, None, "t1", msg.session_key)
    queued = []
    while not bus.outbound.empty():
        queued.append(bus.outbound.get_nowait().content)
    assert "Checking.\n\nDone." in queued


@pytest.mark.asyncio
async def test_leading_newlines_in_whole_message_do_not_duplicate_separator():
    out = await collect_events(
        FakeV3Agent(
            [
                whole_message(AIMessage(content="First.")),
                whole_message(AIMessage(content="\n\nSecond.")),
            ]
        )
    )
    assert (
        next(e for e in out if e["type"] == "done")["response"] == "First.\n\nSecond."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", [False, True])
async def test_rubric_revision_separator_is_emitted_and_last_pass_stays_clean(finish):
    from tests.fakes.stream_v3 import custom_subagent_event

    events = [message_delta("Draft.")]
    if finish:
        events.append(message_finish())
    events += [
        custom_subagent_event(
            {"type": "rubric_evaluation_end", "result": "needs_revision"}
        ),
        message_delta("Revision."),
        message_finish(),
    ]
    out = await collect_events(FakeV3Agent(events))
    done = next(e for e in out if e["type"] == "done")
    text = "".join(e["content"] for e in out if e["type"] == "text")
    assert text == done["response"] == "Draft.\n\nRevision."
    assert done["last_pass_response"] == "Revision."


@pytest.mark.asyncio
async def test_discard_restores_message_boundary_before_another_delta():
    from tests.fakes.stream_v3 import model_call_end, model_call_start, retry_event

    out = await collect_events(
        FakeV3Agent(
            [
                model_call_start("1"),
                message_delta("Kept."),
                message_finish(),
                model_call_end("1"),
                model_call_start("2"),
                message_delta("Failed."),
                model_call_end("2", "failed"),
                retry_event(),
                message_delta("Good."),
                message_finish(),
            ]
        )
    )
    assert next(e for e in out if e["type"] == "done")["response"] == "Kept.\n\nGood."
