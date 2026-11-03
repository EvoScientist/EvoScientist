"""The channel consumer drops a failed attempt's text from its reply buffers."""

from EvoScientist.channels.consumer import trim_buffer, trim_text


def test_trim_text_drops_the_tail_and_never_goes_negative():
    assert trim_text("First.Half", 4) == "First."
    assert trim_text("ab", 10) == ""
    assert trim_text("ab", 0) == "ab"


def test_trim_buffer_drops_whole_and_partial_chunks():
    chunks = ["one ", "two ", "thr"]
    trim_buffer(
        chunks, 5
    )  # drop "thr", cut "two " to "two" minus one more -> "tw"? no: 5 = 3 + 2
    assert chunks == ["one ", "tw"]
    trim_buffer(chunks, 99)
    assert chunks == []


def _event(instance_id="", text_chars=0, thinking_chars=0):
    return {
        "type": "model_retry",
        "instance_id": instance_id,
        "discarded": {
            "text_chars": text_chars,
            "thinking_chars": thinking_chars,
            "tool_call_ids": [],
        },
    }


def test_main_discard_trims_the_reply_and_the_unsent_thinking():
    from EvoScientist.channels.consumer import apply_model_discard

    thinking = ["plan ", "failed thought"]
    buffers = {"sa": ("research", ["kept"])}
    reply = apply_model_discard(
        _event(text_chars=4, thinking_chars=len("failed thought")),
        "Good.Half",
        buffers,
        thinking,
    )
    assert reply == "Good."
    assert thinking == ["plan "]
    assert buffers == {"sa": ("research", ["kept"])}


def test_subagent_discard_trims_only_its_own_buffer():
    from EvoScientist.channels.consumer import apply_model_discard

    thinking = ["main thought"]
    buffers = {"sa": ("research", ["done ", "half"]), "other": ("code", ["x"])}
    reply = apply_model_discard(
        _event(instance_id="sa", text_chars=4, thinking_chars=3),
        "Main.",
        buffers,
        thinking,
    )
    assert reply == "Main."
    assert thinking == ["main thought"]
    assert buffers["sa"][1] == ["done "]
    assert buffers["other"][1] == ["x"]


def test_discard_for_an_unknown_subagent_is_ignored():
    from EvoScientist.channels.consumer import apply_model_discard

    assert (
        apply_model_discard(_event(instance_id="gone", text_chars=9), "Main.", {}, [])
        == "Main."
    )
