"""Tests for the proactive action ranker."""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from EvoScientist.proactive import ranker


def _thread():
    return [
        HumanMessage(content="Review LLM judges for long-form answers.", id="h1"),
        AIMessage(
            content="", tool_calls=[{"name": "search", "args": {}, "id": "c1"}], id="a1"
        ),
        ToolMessage(
            content="- Judge bias study (2401.11111): position bias in pairwise judges.\n"
            "- Calibrated judges (2402.22222): calibration with small models.",
            tool_call_id="c1",
            name="search",
            id="t1",
        ),
        AIMessage(content="The key paper is 2401.11111 on position bias.", id="a2"),
    ]


class _Reply:
    def __init__(self, text):
        self.text = text


class _Judge:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    async def ainvoke(self, _messages):
        self.calls += 1
        return _Reply(self.replies.pop(0))


def _candidates(paper_id="2402.22222"):
    return {
        "stage": "literature_review",
        "focus": "LLM judges",
        "candidates": [
            {
                "type": "paper_to_read",
                "message": f"Read {paper_id}, it studies calibration with small models.",
                "evidence": "Calibrated judges (2402.22222): calibration with small models.",
            },
            {
                "type": "literature_gap",
                "message": "No paper covers judges for long-form answers.",
                "evidence": "Review LLM judges for long-form answers.",
            },
        ],
    }


def _scores(*totals):
    out = []
    for i, (ff, sv, nov, sp) in enumerate(totals):
        out.append(
            {
                "index": i,
                "focus_fit": ff,
                "stage_value": sv,
                "novelty": nov,
                "specificity": sp,
            }
        )
    return json.dumps({"stage": "literature_review", "focus": "x", "scores": out})


def test_pool_keeps_only_ids_never_named_in_turns():
    pool = ranker.build_pool(_thread())
    assert list(pool) == ["2402.22222"]
    assert "Calibrated judges" in pool["2402.22222"]


def test_trigger_lists_pool_or_none():
    assert "- 2402.22222: line" in ranker.ranker_trigger({"2402.22222": "line"})
    assert "(none)" in ranker.ranker_trigger({})


def test_evidence_share_accepts_reformatted_quote_and_rejects_invented_one():
    words = ranker._words(
        "Calibrated judges (2402.22222): calibration with small models, as reported."
    )
    shingles, joined = ranker._shingles(words), " ".join(words)
    assert (
        ranker.evidence_share(
            "calibrated judges 2402.22222 calibration with small models",
            shingles,
            joined,
        )
        >= 0.7
    )
    assert (
        ranker.evidence_share(
            "judges were trained on ten million labels", shingles, joined
        )
        == 0.0
    )
    assert ranker.evidence_share("", shingles, joined) == 0.0


def test_scores_complete_needs_every_candidate_and_key():
    assert ranker.scores_complete(json.loads(_scores((5, 5, 5, 5), (4, 4, 4, 4))), 2)
    assert not ranker.scores_complete(
        json.loads(
            _scores(
                (5, 5, 5, 5),
            )
        ),
        2,
    )
    assert not ranker.scores_complete({"scores": [{"index": 0, "focus_fit": 5}]}, 1)
    assert not ranker.scores_complete({}, 1)


def test_pick_applies_evidence_and_bars():
    cands = [
        {"evidence_found": True},
        {"evidence_found": True},
        {"evidence_found": False},
    ]
    scores = [
        {
            "index": 0,
            "focus_fit": 5,
            "stage_value": 5,
            "novelty": 3,
            "specificity": 5,
        },  # novelty < 4
        {
            "index": 1,
            "focus_fit": 4,
            "stage_value": 5,
            "novelty": 4,
            "specificity": 5,
        },  # 18, passes
        {
            "index": 2,
            "focus_fit": 5,
            "stage_value": 5,
            "novelty": 5,
            "specificity": 5,
        },  # no evidence
    ]
    assert ranker.pick(cands, scores) == [1]
    scores[1]["specificity"] = 4  # total 17
    assert ranker.pick(cands, scores) == []


def test_pick_returns_every_candidate_tied_at_the_top():
    cands = [{"evidence_found": True}] * 3
    scores = [
        {
            "index": 0,
            "focus_fit": 4,
            "stage_value": 5,
            "novelty": 4,
            "specificity": 5,
        },  # 18
        {
            "index": 1,
            "focus_fit": 5,
            "stage_value": 4,
            "novelty": 4,
            "specificity": 5,
        },  # 18
        {
            "index": 2,
            "focus_fit": 4,
            "stage_value": 4,
            "novelty": 4,
            "specificity": 5,
        },  # 17
    ]
    assert ranker.pick(cands, scores) == [0, 1]
    assert ranker.stage_value_winner([0, 1], scores) == 0
    scores[1]["stage_value"] = 5
    assert ranker.stage_value_winner([0, 1], scores) is None


@pytest.mark.asyncio
async def test_run_ranker_pushes_pool_paper_and_records(tmp_path):
    log = tmp_path / "r.jsonl"

    async def shadow(_trigger):
        return json.dumps(_candidates())

    judge = _Judge([_scores((5, 5, 5, 5), (4, 4, 4, 4))])
    text = await ranker.run_ranker(
        messages=_thread(), shadow_turn=shadow, judge=judge, thread_id="t", log_path=log
    )
    assert "2402.22222" in text
    record = json.loads(log.read_text().splitlines()[0])
    assert record["decision"] == "push"
    assert record["winner"] == 0
    assert record["pool_size"] == 1


@pytest.mark.asyncio
async def test_paper_outside_pool_is_dropped(tmp_path):
    async def shadow(_trigger):
        return json.dumps(_candidates(paper_id="2401.11111"))  # already discussed

    judge = _Judge([_scores((5, 5, 5, 5), (3, 3, 3, 3))])
    text = await ranker.run_ranker(
        messages=_thread(),
        shadow_turn=shadow,
        judge=judge,
        thread_id="t",
        log_path=None,
    )
    assert text == "NO_PUSH"


@pytest.mark.asyncio
async def test_malformed_replies_are_retried_then_fail(tmp_path):
    log = tmp_path / "r.jsonl"
    calls = []

    async def shadow(_trigger):
        calls.append(1)
        return (
            '{"candidates": [{"type":"x":""}]}'
            if len(calls) < 2
            else json.dumps(_candidates())
        )

    judge = _Judge(
        [
            "not json",
            _scores(
                (5, 5, 5, 5),
            ),
            _scores((5, 5, 5, 5), (4, 4, 4, 4)),
        ]
    )
    text = await ranker.run_ranker(
        messages=_thread(), shadow_turn=shadow, judge=judge, thread_id="t", log_path=log
    )
    assert len(calls) == 2
    assert judge.calls == 3
    assert "2402.22222" in text

    async def broken(_trigger):
        return None

    with pytest.raises(ranker.RankerFailed):
        await ranker.run_ranker(
            messages=_thread(),
            shadow_turn=broken,
            judge=_Judge([]),
            thread_id="t",
            log_path=log,
        )
    last = json.loads(log.read_text().splitlines()[-1])
    assert last["decision"] == "error"
    assert last["ranker_attempts"] == ranker.ATTEMPTS


def test_ranker_enabled_reads_env(monkeypatch):
    monkeypatch.delenv(ranker.RANKER_ENV, raising=False)
    assert not ranker.ranker_enabled()
    monkeypatch.setenv(ranker.RANKER_ENV, "1")
    assert ranker.ranker_enabled()


@pytest.mark.asyncio
async def test_record_write_does_not_block_the_event_loop(tmp_path):
    from blockbuster import blockbuster_ctx

    log = tmp_path / "new_dir" / "r.jsonl"

    async def shadow(_trigger):
        return json.dumps(_candidates())

    judge = _Judge([_scores((5, 5, 5, 5), (4, 4, 4, 4))])
    with blockbuster_ctx():
        text = await ranker.run_ranker(
            messages=_thread(),
            shadow_turn=shadow,
            judge=judge,
            thread_id="t",
            log_path=log,
        )
    assert "2402.22222" in text
    assert json.loads(log.read_text().splitlines()[0])["decision"] == "push"


def _tie_scores():
    return _scores((5, 5, 4, 4), (5, 4, 5, 4))  # both 18; stage_value favours 0


@pytest.mark.asyncio
async def test_tie_head_to_head_agreeing_picks_that_candidate(tmp_path):
    log = tmp_path / "r.jsonl"

    async def shadow(_trigger):
        return json.dumps(_candidates())

    pick_one = json.dumps({"choice": 1, "reason": "gap matters more"})
    judge = _Judge([_tie_scores(), pick_one, pick_one])
    text = await ranker.run_ranker(
        messages=_thread(), shadow_turn=shadow, judge=judge, thread_id="t", log_path=log
    )
    assert text.startswith("No paper covers")
    record = json.loads(log.read_text().splitlines()[0])
    tie = record["tie"]
    assert tie["resolved_by"] == "head_to_head"
    assert record["winner"] == 1
    assert [r["order"] for r in tie["runs"]] == [[0, 1], [1, 0]]


@pytest.mark.asyncio
async def test_tie_split_falls_back_to_stage_value(tmp_path):
    log = tmp_path / "r.jsonl"

    async def shadow(_trigger):
        return json.dumps(_candidates())

    # The judge prefers whichever comes first: a position bias, not a choice.
    judge = _Judge(
        [_tie_scores(), json.dumps({"choice": 0}), json.dumps({"choice": 1})]
    )
    text = await ranker.run_ranker(
        messages=_thread(), shadow_turn=shadow, judge=judge, thread_id="t", log_path=log
    )
    assert "2402.22222" in text
    assert (
        json.loads(log.read_text().splitlines()[0])["tie"]["resolved_by"]
        == "stage_value"
    )


@pytest.mark.asyncio
async def test_unresolved_tie_sends_nothing(tmp_path):
    log = tmp_path / "r.jsonl"

    async def shadow(_trigger):
        return json.dumps(_candidates())

    # Equal stage_value, and every head-to-head reply is unusable.
    judge = _Judge(
        [_scores((5, 5, 4, 4), (4, 5, 5, 4))]
        + ["not json", json.dumps({"choice": 7})] * 3
    )
    text = await ranker.run_ranker(
        messages=_thread(), shadow_turn=shadow, judge=judge, thread_id="t", log_path=log
    )
    assert text == "NO_PUSH"
    record = json.loads(log.read_text().splitlines()[0])
    assert record["tie"]["resolved_by"] == "unresolved"
    assert record["decision"] == "no_push"
    assert judge.calls == 1 + 2 * ranker.ATTEMPTS


def test_paper_candidate_shows_the_judge_what_the_thread_has_on_it():
    msgs = _thread()
    entry = ranker._listing_entry(0, _candidates()["candidates"][0], msgs)
    assert "what the conversation shows about this paper" in entry
    assert "Calibrated judges" in entry
    other = ranker._listing_entry(1, _candidates()["candidates"][1], msgs)
    assert "what the conversation shows" not in other
    unseen = ranker._listing_entry(
        0, {"type": "paper_to_read", "message": "Read 2499.00001."}, msgs
    )
    assert unseen.endswith("(nothing)")


def test_paper_context_windows_a_long_single_line_result():
    blob = "x" * 5000 + " 2402.22222 Calibrated judges " + "y" * 5000
    ctx = ranker.paper_context(
        [ToolMessage(content=blob, tool_call_id="c")], "2402.22222"
    )
    assert "Calibrated judges" in ctx
    assert len(ctx) < 2 * ranker.PAPER_WINDOW + 50
