"""Tests for EvoScientist.proactive.delivery_watcher (serve-side delivery).

Pure/DI'd core — no server or bus. Validates push extraction (dict + BaseMessage),
idempotency across polls (the ``seen`` set), and retry-on-failure (a failed publish
stays un-seen).
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage, HumanMessage

from EvoScientist.proactive.delivery_watcher import (
    deliver_pending_pushes,
    extract_proactive_push,
)


def _push_dict(content, proactive_id):
    return {
        "content": content,
        "additional_kwargs": {
            "evoscientist": {"is_proactive_push": True, "proactive_id": proactive_id}
        },
    }


# ---- extract_proactive_push ----


def test_extract_from_dict_message():
    msgs = [{"content": "hi"}, _push_dict("Checking in!", "p1")]
    assert extract_proactive_push(msgs) == ("p1", "Checking in!")


def test_extract_from_basemessage():
    msgs = [
        HumanMessage(content="hi"),
        AIMessage(
            content="Want to continue?",
            additional_kwargs={
                "evoscientist": {"is_proactive_push": True, "proactive_id": "p2"}
            },
        ),
    ]
    assert extract_proactive_push(msgs) == ("p2", "Want to continue?")


def test_extract_returns_none_without_push():
    assert (
        extract_proactive_push([{"content": "hi"}, HumanMessage(content="yo")]) is None
    )
    assert extract_proactive_push([]) is None


def test_extract_returns_most_recent_push():
    msgs = [_push_dict("old", "p1"), {"content": "reply"}, _push_dict("new", "p2")]
    assert extract_proactive_push(msgs) == ("p2", "new")


def test_extract_skips_push_missing_id_or_content():
    assert extract_proactive_push([_push_dict("", "p1")]) is None
    assert (
        extract_proactive_push(
            [
                {
                    "content": "x",
                    "additional_kwargs": {"evoscientist": {"is_proactive_push": True}},
                }
            ]
        )
        is None
    )


# ---- deliver_pending_pushes ----


def _reader(mapping):
    async def _read(thread_id):
        return mapping.get(thread_id, [])

    return _read


class _Publisher:
    def __init__(self, ok=True):
        self.ok = ok
        self.calls = []

    def __call__(self, thread_id, content):
        self.calls.append((thread_id, content))
        return self.ok


def _deliver(mapping, publisher, seen=None):
    seen = seen if seen is not None else set()
    delivered = asyncio.run(
        deliver_pending_pushes(
            list(mapping.keys()),
            read_messages=_reader(mapping),
            publish=publisher,
            seen=seen,
        )
    )
    return delivered, seen


def test_delivers_unseen_push_and_marks_seen():
    pub = _Publisher(ok=True)
    delivered, seen = _deliver({"t1": [_push_dict("hey", "p1")]}, pub)
    assert delivered == ["p1"]
    assert seen == {"p1"}
    assert pub.calls == [("t1", "hey")]


def test_already_seen_push_not_redelivered():
    pub = _Publisher(ok=True)
    delivered, _ = _deliver({"t1": [_push_dict("hey", "p1")]}, pub, seen={"p1"})
    assert delivered == []
    assert pub.calls == []


def test_failed_publish_stays_unseen_for_retry():
    pub = _Publisher(ok=False)  # no origin / bus down
    delivered, seen = _deliver({"t1": [_push_dict("hey", "p1")]}, pub)
    assert delivered == []
    assert seen == set()  # not marked → retried next poll
    assert pub.calls == [("t1", "hey")]


def test_thread_without_push_is_skipped():
    pub = _Publisher(ok=True)
    delivered, _ = _deliver({"t1": [{"content": "just chatting"}]}, pub)
    assert delivered == []
    assert pub.calls == []


def test_multiple_threads_delivered_independently():
    pub = _Publisher(ok=True)
    mapping = {
        "t1": [_push_dict("a", "p1")],
        "t2": [{"content": "no push"}],
        "t3": [_push_dict("c", "p3")],
    }
    delivered, seen = _deliver(mapping, pub)
    assert set(delivered) == {"p1", "p3"}
    assert seen == {"p1", "p3"}
    assert ("t2", "no push") not in pub.calls
