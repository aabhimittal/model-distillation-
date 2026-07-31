"""Industrial edge-case tests for the Track-A fine-tuning pipeline.

These cover the failure modes that appear at production scale rather than in a
happy-path demo: degenerate teacher generations, refusals, prompt leakage,
truncation, multilingual/adversarial unicode, train/eval contamination, rate
limiting, throttled APIs, and crash-resume of a long generation run.

All pure Python — no torch, no GPU, no network — so the whole file runs in CI in
well under a second.
"""

from __future__ import annotations

import json

import pytest

from src.finetune.budget import (
    bucket_by_length,
    estimate_teacher_cost,
    estimate_tokens,
    length_stats,
    padding_waste,
    record_lengths,
    sequential_batches,
)
from src.finetune.chat import IGNORE_INDEX, build_supervised_labels, has_trainable_labels
from src.finetune.curate import (
    REASON_DEGENERATE,
    REASON_EMPTY,
    REASON_PROMPT_LEAK,
    REASON_REFUSAL,
    REASON_TRUNCATED,
    CurationThresholds,
    classify_record,
    clean_response,
    curate_records,
    is_degenerate,
    is_refusal,
    leaks_prompt,
    looks_truncated,
    max_consecutive_repeats,
    ngram_repeat_ratio,
)
from src.finetune.dedup import (
    canonical,
    decontaminate,
    exact_dedup,
    jaccard,
    near_dedup,
    shingles,
)
from src.finetune.robust import (
    JsonlCheckpoint,
    RateLimiter,
    RetryPolicy,
    merge_checkpoint,
    pending_indices,
    retry_call,
    status_of,
)


def rec(instruction="Explain compound interest in detail please", output="ok", input_text=""):
    return {"instruction": instruction, "input": input_text, "output": output}


# =============================================================================
# Degenerate teacher output — the silent killer of sequence-level KD
# =============================================================================

def test_repetition_loop_detected():
    """A decoder stuck in a loop is fluent, so only n-gram stats catch it."""
    assert is_degenerate("the cat sat " * 30)
    assert max_consecutive_repeats("buy buy buy buy buy buy") >= 5


def test_healthy_prose_survives_repetition_check():
    """Real answers repeat some words; the gate must not eat them."""
    text = (
        "An ETF is a fund that trades on an exchange. The fund holds a basket of "
        "assets, and the price of the fund tracks the value of those assets."
    )
    assert not is_degenerate(text)
    assert ngram_repeat_ratio(text) < 0.3


def test_repeat_ratio_ignores_texts_shorter_than_window():
    assert ngram_repeat_ratio("a b", n=3) == 0.0
    assert max_consecutive_repeats("") == 0


def test_boilerplate_list_is_not_flagged_degenerate():
    """Structured answers repeat scaffolding words without being degenerate."""
    text = "1. Save early. 2. Save often. 3. Invest broadly. 4. Rebalance yearly."
    assert not is_degenerate(text)


# =============================================================================
# Refusals and prompt leakage
# =============================================================================

@pytest.mark.parametrize(
    "text",
    [
        "I'm sorry, I cannot help with that request.",
        "I cannot provide financial advice.",
        "As an AI language model, I am unable to answer.",
        "  \"I apologize, but this is outside my guidelines.",
    ],
)
def test_refusals_detected(text):
    assert is_refusal(text)


def test_domain_answer_mentioning_cannot_is_not_a_refusal():
    """'cannot' deep inside a legitimate answer must not trigger the gate."""
    text = (
        "Under the covenant the borrower cannot draw further tranches once the "
        "leverage ratio exceeds 3.5x, which protects the lender."
    )
    assert not is_refusal(text)


def test_prompt_echo_detected():
    instr = "Explain the difference between a mutual fund and an ETF in detail"
    assert leaks_prompt(instr + " Well, an ETF trades intraday.", instr)


def test_chat_template_markers_are_leakage():
    assert leaks_prompt("<|im_start|>assistant here is the answer", "anything")
    assert leaks_prompt("### Response: the answer is 42", "anything")


def test_short_instruction_not_treated_as_echo():
    """Short instructions collide with legitimate answers; don't over-trigger."""
    assert not leaks_prompt("Yes, definitely.", "Yes?")


# =============================================================================
# Truncation (generation hit max_new_tokens)
# =============================================================================

def test_truncated_generation_detected():
    text = " ".join(["word"] * 20) + " and then the value of the portfolio begins to"
    assert looks_truncated(text)


def test_short_unpunctuated_answer_is_not_truncation():
    """'42' is a complete answer even without a full stop."""
    assert not looks_truncated("42")


def test_cjk_terminal_punctuation_accepted():
    """A complete Japanese sentence ends in 。 — not ASCII '.'."""
    assert not looks_truncated("これは" + "とても長い説明です " * 6 + "。")


# =============================================================================
# Unicode / adversarial text hygiene
# =============================================================================

def test_control_characters_stripped():
    assert "\x00" not in clean_response("bad\x00text\x07here")
    assert clean_response("a\x00b") == "ab"


def test_newlines_and_tabs_collapse_not_crash():
    assert clean_response("line1\n\n\tline2   line3") == "line1 line2 line3"


def test_emoji_and_rtl_text_preserved():
    """Non-Latin content is domain data, not noise — it must survive curation."""
    arabic = "القيمة السوقية للصندوق ترتفع عندما ترتفع أسعار الأصول الأساسية."
    assert classify_record(rec(output=arabic)) is None
    assert classify_record(rec(output="Returns compounded 12% 📈 year over year.")) is None


def test_whitespace_only_output_is_empty():
    assert classify_record(rec(output="   \n\t  ")) == REASON_EMPTY


def test_non_string_output_rejected_not_crashed():
    """Malformed rows from a scraped corpus must not take the run down."""
    assert classify_record({"instruction": "q", "output": None}) is not None
    assert classify_record({"instruction": "q", "output": 12345}) is not None


# =============================================================================
# The curation gate end-to-end
# =============================================================================

def test_curate_reports_each_reason():
    instr = "Explain the difference between a mutual fund and an ETF in detail"
    records = [
        rec(output="An ETF trades on an exchange throughout the day."),
        rec(output=""),
        rec(output="spam " * 40),
        rec(output="I'm sorry, I cannot help with that."),
        rec(instruction=instr, output=instr + " they differ in trading."),
    ]
    kept, report = curate_records(records)
    assert len(kept) == 1
    assert report.total == 5 and report.kept == 1
    for reason in (REASON_EMPTY, REASON_DEGENERATE, REASON_REFUSAL, REASON_PROMPT_LEAK):
        assert report.dropped.get(reason) == 1
    assert 0.0 < report.keep_rate < 1.0


def test_curation_cleans_kept_outputs():
    kept, _ = curate_records([rec(output="answer\x00 with   junk.")])
    assert kept[0]["output"] == "answer with junk."


def test_curate_empty_input_is_safe():
    kept, report = curate_records([])
    assert kept == [] and report.total == 0 and report.keep_rate == 0.0
    assert "no records" in report.summary()


def test_thresholds_can_disable_gates():
    """Some domains legitimately contain refusal-shaped text (e.g. legal corpora)."""
    th = CurationThresholds(drop_refusals=False, drop_truncated=False)
    assert classify_record(rec(output="I cannot disclose the client's name."), th) is None


# =============================================================================
# Deduplication and eval contamination
# =============================================================================

def test_exact_dedup_is_case_and_punctuation_insensitive():
    records = [rec("What is an ETF?"), rec("what is an etf"), rec("Define a bond")]
    kept, removed = exact_dedup(records)
    assert removed == 1 and len(kept) == 2


def test_near_dedup_catches_rewordings():
    records = [
        rec("Explain how compound interest works for a savings account"),
        rec("Explain how compound interest works for a savings account!!"),
        rec("What are the tax implications of a Roth IRA conversion"),
    ]
    kept, removed = near_dedup(records, threshold=0.85)
    assert removed == 1 and len(kept) == 2


def test_near_dedup_keeps_genuinely_distinct_records():
    records = [rec("Explain bond duration"), rec("Explain equity beta")]
    kept, removed = near_dedup(records, threshold=0.85)
    assert removed == 0 and len(kept) == 2


def test_dedup_keeps_first_occurrence_stably():
    records = [rec("Define alpha", output="first"), rec("define alpha!", output="second")]
    kept, _ = near_dedup(records, threshold=0.85)
    assert kept[0]["output"] == "first"


def test_decontamination_removes_eval_leakage():
    """The classic silent-inflation bug: eval questions present in training."""
    evaluation = [rec("What is the capital gains tax rate for long term holdings")]
    train = [
        rec("What is the capital gains tax rate for long term holdings"),   # exact
        rec("what is the capital gains tax rate for long-term holdings?"),  # near
        rec("Explain how municipal bonds are taxed at the state level"),    # clean
    ]
    kept, removed = decontaminate(train, evaluation, threshold=0.85)
    assert removed == 2 and len(kept) == 1


def test_decontamination_never_shrinks_eval_set():
    """Eval is the measuring instrument; only train may be filtered."""
    evaluation = [rec("Q one"), rec("Q two")]
    kept, _ = decontaminate([rec("Q one")], evaluation)
    assert len(evaluation) == 2 and kept == []


def test_decontaminate_with_empty_eval_is_noop():
    train = [rec("a"), rec("b")]
    kept, removed = decontaminate(train, [])
    assert removed == 0 and len(kept) == 2


def test_jaccard_and_shingles_edge_cases():
    assert jaccard(set(), set()) == 0.0
    assert jaccard({"a"}, {"a"}) == 1.0
    assert shingles("") == set()
    assert shingles("ab", k=5) == {"ab"}   # shorter than k -> still comparable
    assert canonical("  Héllo,   WORLD!! ") == canonical("héllo world")


def test_dedup_handles_unicode_without_collision():
    """Different CJK questions must not collapse into one another."""
    kept, removed = near_dedup([rec("株式とは何ですか"), rec("債券とは何ですか")], 0.85)
    assert removed == 0 and len(kept) == 2


# =============================================================================
# Token budgeting and padding waste
# =============================================================================

def test_length_bucketing_beats_naive_batching():
    """The whole point: grouping similar lengths removes most padding."""
    lengths = [5, 500, 7, 480, 6, 510, 8, 495]
    naive = padding_waste(lengths, sequential_batches(len(lengths), 2))
    bucketed = padding_waste(lengths, bucket_by_length(lengths, 2))
    assert bucketed < naive
    assert bucketed < 0.05  # near-uniform batches


def test_padding_waste_is_zero_for_uniform_lengths():
    lengths = [10, 10, 10, 10]
    assert padding_waste(lengths, bucket_by_length(lengths, 2)) == 0.0


def test_bucketing_preserves_every_record_exactly_once():
    lengths = [3, 1, 4, 1, 5, 9, 2, 6]
    flat = [i for b in bucket_by_length(lengths, 3) for i in b]
    assert sorted(flat) == list(range(len(lengths)))


def test_budget_edge_cases():
    assert padding_waste([], []) == 0.0
    assert bucket_by_length([], 4) == []
    with pytest.raises(ValueError):
        bucket_by_length([1, 2], 0)


def test_length_stats_flags_records_over_limit():
    stats = length_stats([10, 20, 5000], max_seq_length=1024)
    assert stats.over_limit == 1 and stats.maximum == 5000
    assert stats.count == 3 and "p95" in stats.as_dict()


def test_length_stats_empty_is_safe():
    stats = length_stats([], max_seq_length=1024)
    assert stats.count == 0 and stats.mean == 0.0 and stats.over_limit == 0


def test_cjk_tokens_not_underestimated():
    """4-chars-per-token would badly under-count space-free scripts."""
    cjk = "投資信託の基準価額は毎営業日に更新されます"
    assert estimate_tokens(cjk) >= len(cjk)
    assert estimate_tokens("") == 0


def test_teacher_cost_is_an_upper_bound():
    cost = estimate_teacher_cost(1000, 120, 256, 0.0002, 0.0006)
    assert cost["input_tokens"] == 120_000
    assert cost["output_tokens"] == 256_000
    assert cost["estimated_cost"] > 0
    with pytest.raises(ValueError):
        estimate_teacher_cost(-1, 10, 10)


def test_record_lengths_sums_all_fields():
    lengths = record_lengths([rec("abcd", "efgh", "ijkl")])
    assert lengths[0] >= 3


# =============================================================================
# Remote-teacher robustness: throttling, retries, resume
# =============================================================================

class FakeHTTPError(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.status_code = status


def test_retries_throttling_then_succeeds():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise FakeHTTPError(429)
        return "ok"

    slept = []
    out = retry_call(flaky, RetryPolicy(max_attempts=5), sleep=slept.append, rand=lambda: 1.0)
    assert out == "ok" and calls["n"] == 3 and len(slept) == 2


def test_client_errors_are_not_retried():
    """Retrying a 400/401 just burns quota — fail fast instead."""
    calls = {"n": 0}

    def bad_request():
        calls["n"] += 1
        raise FakeHTTPError(400)

    with pytest.raises(FakeHTTPError):
        retry_call(bad_request, RetryPolicy(max_attempts=5), sleep=lambda _: None)
    assert calls["n"] == 1


def test_retry_gives_up_and_reraises():
    def always_503():
        raise FakeHTTPError(503)

    with pytest.raises(FakeHTTPError):
        retry_call(always_503, RetryPolicy(max_attempts=3), sleep=lambda _: None)


def test_connection_errors_without_status_are_retried():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionResetError("peer reset")
        return "ok"

    assert retry_call(flaky, sleep=lambda _: None) == "ok"


def test_backoff_grows_and_is_capped():
    policy = RetryPolicy(base_delay=1.0, max_delay=10.0, jitter=False)
    assert policy.delay_for(1) == 1.0
    assert policy.delay_for(2) == 2.0
    assert policy.delay_for(3) == 4.0
    assert policy.delay_for(10) == 10.0  # capped


def test_jitter_stays_within_bounds():
    """Full jitter must never exceed the deterministic ceiling."""
    policy = RetryPolicy(base_delay=2.0, max_delay=100.0, jitter=True)
    assert 0.0 <= policy.delay_for(3, rand=lambda: 0.0) <= 8.0
    assert policy.delay_for(3, rand=lambda: 1.0) == 8.0


def test_status_extraction_across_sdk_shapes():
    class WithResponse(Exception):
        def __init__(self):
            self.response = type("R", (), {"status_code": 429})()

    assert status_of(FakeHTTPError(500)) == 500
    assert status_of(WithResponse()) == 429
    assert status_of(ValueError("plain")) is None


def test_rate_limiter_throttles_a_burst():
    now = {"t": 0.0}
    waits = []

    def clock():
        return now["t"]

    def sleep(s):
        waits.append(s)
        now["t"] += s

    limiter = RateLimiter(rate=2.0, burst=1, clock=clock, sleep=sleep)
    for _ in range(3):
        limiter.acquire()
    assert len(waits) == 2                      # first is free (burst), rest throttled
    assert all(abs(w - 0.5) < 1e-6 for w in waits)  # 2 rps -> 0.5s spacing


def test_rate_limiter_banks_idle_capacity():
    now = {"t": 0.0}
    limiter = RateLimiter(rate=10.0, burst=5, clock=lambda: now["t"], sleep=lambda s: None)
    limiter.acquire()
    now["t"] += 10.0          # long idle gap refills the bucket
    assert limiter.acquire() == 0.0


def test_rate_limiter_rejects_bad_rate():
    with pytest.raises(ValueError):
        RateLimiter(rate=0)


def test_checkpoint_resumes_after_crash(tmp_path):
    """A killed Colab session must not re-pay for finished generations."""
    path = str(tmp_path / "cache.jsonl")
    ckpt = JsonlCheckpoint(path)
    for i, text in enumerate(["a", "b", "c"]):
        ckpt.append(i, text)

    done = JsonlCheckpoint(path).load()
    assert done == {0: "a", 1: "b", 2: "c"}
    assert pending_indices(5, done) == [3, 4]
    assert merge_checkpoint(5, done) == ["a", "b", "c", "", ""]


def test_checkpoint_tolerates_torn_final_line(tmp_path):
    """Process killed mid-flush leaves a partial line — resume, don't crash."""
    path = tmp_path / "cache.jsonl"
    path.write_text(
        json.dumps({"index": 0, "output": "ok"}) + "\n" + '{"index": 1, "outp',
        encoding="utf-8",
    )
    assert JsonlCheckpoint(str(path)).load() == {0: "ok"}


def test_checkpoint_missing_file_is_empty(tmp_path):
    assert JsonlCheckpoint(str(tmp_path / "nope.jsonl")).load() == {}


def test_checkpoint_roundtrips_unicode(tmp_path):
    path = str(tmp_path / "u.jsonl")
    JsonlCheckpoint(path).append(0, "利率は 3% 📈")
    assert JsonlCheckpoint(path).load()[0] == "利率は 3% 📈"


def test_checkpoint_creates_missing_parent_dir(tmp_path):
    path = str(tmp_path / "nested" / "deep" / "cache.jsonl")
    JsonlCheckpoint(path).append(0, "x")
    assert JsonlCheckpoint(path).load() == {0: "x"}


# =============================================================================
# Label-masking edge cases that only bite at scale
# =============================================================================

def test_oversized_prompt_leaves_no_trainable_tokens():
    """A record whose prompt alone fills max_seq_length yields an all-masked
    label row — training on it contributes nothing and NaNs some loss reductions."""
    ex = build_supervised_labels(list(range(64)), [5, 6], eos_id=9, max_length=64)
    assert not has_trainable_labels(ex["labels"])
    assert all(l == IGNORE_INDEX for l in ex["labels"])


def test_empty_response_still_teaches_eos():
    ex = build_supervised_labels([1, 2], [], eos_id=9)
    assert ex["input_ids"] == [1, 2, 9]
    assert has_trainable_labels(ex["labels"])


def test_labels_and_inputs_always_same_length():
    for max_len in (1, 3, 8, 100):
        ex = build_supervised_labels([1, 2, 3], [4, 5], eos_id=9, max_length=max_len)
        assert len(ex["input_ids"]) == len(ex["labels"])


def test_no_eos_id_configured_does_not_crash():
    ex = build_supervised_labels([1], [2], eos_id=None)
    assert ex["input_ids"] == [1, 2]
