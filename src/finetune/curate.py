"""Teacher-output curation — the quality gate on sequence-level distillation.

Sequence-level KD trains the student on whatever the teacher *said*. That makes
the student's ceiling the teacher's worst outputs, not its average one. In
production runs over thousands of prompts, a small but destructive fraction of
teacher generations are unusable:

  * **degenerate loops** — ``"the the the ..."``, a decoder stuck in a cycle;
  * **refusals** — ``"I'm sorry, I cannot help with that"``, which teach the
    student to refuse its own domain;
  * **prompt leakage** — the teacher echoes the instruction or emits raw chat
    template markers, teaching the student to parrot scaffolding;
  * **truncation** — generation hit ``max_new_tokens`` mid-sentence, teaching
    the student to stop early.

Each of these is *fluent*, so a loss curve will not reveal them; they silently
degrade the student. This module detects and quarantines them before training.

Conceptually this is the Track-A analogue of RAD's ``L_CRA`` degenerate-retrieval
term: both ask "is the teacher's signal on this example actually worth imitating?"
The difference is that sequence-level KD has no logits to inspect, so the check is
made on the decoded text instead.

Pure Python — no torch, no tokenizer — so it runs in CI and on CPU.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

# Reasons a record can be rejected. Stable strings — they land in the report and
# in downstream dashboards, so treat them as an API.
REASON_EMPTY = "empty"
REASON_DEGENERATE = "degenerate_repetition"
REASON_REFUSAL = "refusal"
REASON_PROMPT_LEAK = "prompt_leak"
REASON_TRUNCATED = "truncated"
REASON_TOO_SHORT = "too_short"
REASON_NON_TEXT = "non_text"

# Matched case-insensitively against the opening of the response, where a refusal
# almost always appears. Substring matching mid-answer would flag legitimate
# content ("the report says the bank cannot lend..."), so we anchor to the head.
REFUSAL_PATTERNS: Tuple[str, ...] = (
    "i cannot", "i can't", "i can not", "i'm sorry", "i am sorry",
    "i apologize", "i apologise", "as an ai", "as a language model",
    "i'm unable", "i am unable", "i must decline", "cannot assist",
    "can't assist", "i'm not able to", "i am not able to",
    "sorry, but i", "unable to provide", "against my guidelines",
)

# Chat/template scaffolding that must never survive into a training target.
TEMPLATE_MARKERS: Tuple[str, ...] = (
    "### instruction:", "### response:", "### input:",
    "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|eot_id|>",
    "[inst]", "[/inst]", "<<sys>>", "<s>[inst]",
    "human:", "assistant:",
)

# Sentence-final punctuation across scripts we plausibly see (latin, CJK, arabic,
# devanagari). Absence at the tail is the truncation signal.
TERMINAL_PUNCT = '.!?"\'`)]}…。！？；：」』】〉》,，、।॥؟'

_WS = re.compile(r"\s+")
# Unicode control characters other than tab/newline/carriage-return. These arrive
# from scraped corpora and can corrupt tokenisation or terminal output downstream.
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass
class CurationThresholds:
    """Tunable gates. Defaults are deliberately conservative: when in doubt, keep
    the example. Over-filtering a domain corpus is as harmful as under-filtering."""

    # Fraction of repeated n-grams above which the text is a decoder loop.
    max_ngram_repeat_ratio: float = 0.55
    repeat_ngram_size: int = 3
    # Longest run of the *same* word repeated back to back.
    max_consecutive_repeats: int = 5
    # Below this many characters a response carries no usable signal.
    min_chars: int = 8
    # Truncation is only asserted for responses long enough that an author would
    # have punctuated them; short fragments ("42") legitimately lack punctuation.
    truncation_min_words: int = 12
    # Response is a prompt leak if it reproduces this much of the instruction.
    prompt_echo_ratio: float = 0.9
    drop_refusals: bool = True
    drop_truncated: bool = True


@dataclass
class CurationReport:
    """Counts by rejection reason plus the surviving records' share."""

    total: int = 0
    kept: int = 0
    dropped: Dict[str, int] = field(default_factory=dict)

    @property
    def keep_rate(self) -> float:
        return (self.kept / self.total) if self.total else 0.0

    def record_drop(self, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1

    def as_dict(self) -> Dict[str, object]:
        return {
            "total": self.total,
            "kept": self.kept,
            "keep_rate": round(self.keep_rate, 4),
            "dropped": dict(sorted(self.dropped.items())),
        }

    def summary(self) -> str:
        if not self.total:
            return "curation: no records"
        parts = ", ".join(f"{k}={v}" for k, v in sorted(self.dropped.items()))
        head = f"curation: kept {self.kept}/{self.total} ({self.keep_rate:.1%})"
        return f"{head}" + (f" | dropped: {parts}" if parts else "")


# --- text hygiene -------------------------------------------------------------

def strip_control_chars(text: str) -> str:
    """Remove C0/C1 control characters, preserving tab/newline/CR."""
    return _CTRL.sub("", text)


def normalize_whitespace(text: str) -> str:
    """NFKC-normalise and collapse runs of whitespace to single spaces."""
    text = unicodedata.normalize("NFKC", text)
    return _WS.sub(" ", text).strip()


def clean_response(text: str) -> str:
    """Hygiene pass applied before any quality judgement is made."""
    return normalize_whitespace(strip_control_chars(text or ""))


# --- individual detectors -----------------------------------------------------

def ngram_repeat_ratio(text: str, n: int = 3) -> float:
    """Fraction of word n-grams that are repeats. 0.0 = all distinct.

    A healthy paragraph sits well below 0.3; a decoder stuck in a loop approaches
    1.0. Texts shorter than one full n-gram window return 0.0 (nothing to judge).
    """
    words = text.split()
    if len(words) <= n:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - (len(set(grams)) / len(grams))


def max_consecutive_repeats(text: str) -> int:
    """Longest run of an identical word repeated consecutively (case-insensitive)."""
    words = [w.lower() for w in text.split()]
    if not words:
        return 0
    best = run = 1
    for prev, cur in zip(words, words[1:]):
        run = run + 1 if cur == prev else 1
        best = max(best, run)
    return best


def is_degenerate(text: str, th: CurationThresholds | None = None) -> bool:
    """True if the text looks like a decoder repetition loop."""
    th = th or CurationThresholds()
    if max_consecutive_repeats(text) >= th.max_consecutive_repeats:
        return True
    return ngram_repeat_ratio(text, th.repeat_ngram_size) > th.max_ngram_repeat_ratio


def is_refusal(text: str, head_chars: int = 160) -> bool:
    """True if the response opens with a recognised refusal formula."""
    head = text[:head_chars].lower().lstrip("\"'` \t\n-*")
    return any(head.startswith(p) or f" {p}" in head[:80] for p in REFUSAL_PATTERNS)


def leaks_prompt(response: str, instruction: str, ratio: float = 0.9) -> bool:
    """True if the response echoes the instruction or emits template scaffolding."""
    low = response.lower()
    if any(m in low for m in TEMPLATE_MARKERS):
        return True
    instr = normalize_whitespace(instruction).lower()
    if len(instr) < 20:
        return False  # short instructions collide with legitimate answers
    resp = normalize_whitespace(response).lower()
    prefix = instr[: max(20, int(len(instr) * ratio))]
    return resp.startswith(prefix)


def looks_truncated(text: str, min_words: int = 12) -> bool:
    """True if a long response ends without terminal punctuation (hit the cap)."""
    stripped = text.rstrip()
    if len(stripped.split()) < min_words:
        return False
    return stripped[-1] not in TERMINAL_PUNCT if stripped else False


# --- the gate -----------------------------------------------------------------

def classify_record(
    record: Dict[str, str], th: CurationThresholds | None = None
) -> str | None:
    """Return the rejection reason for a record, or ``None`` if it should be kept."""
    th = th or CurationThresholds()
    raw = record.get("output", "")
    if not isinstance(raw, str):
        return REASON_NON_TEXT

    text = clean_response(raw)
    if not text:
        return REASON_EMPTY
    if len(text) < th.min_chars:
        return REASON_TOO_SHORT
    if is_degenerate(text, th):
        return REASON_DEGENERATE
    if th.drop_refusals and is_refusal(text):
        return REASON_REFUSAL
    if leaks_prompt(text, str(record.get("instruction", "")), th.prompt_echo_ratio):
        return REASON_PROMPT_LEAK
    if th.drop_truncated and looks_truncated(text, th.truncation_min_words):
        return REASON_TRUNCATED
    return None


def curate_records(
    records: Sequence[Dict[str, str]], th: CurationThresholds | None = None
) -> Tuple[List[Dict[str, str]], CurationReport]:
    """Filter teacher outputs, returning (kept_records, report).

    Kept records carry the *cleaned* output so hygiene fixes reach training.
    """
    th = th or CurationThresholds()
    report = CurationReport(total=len(records))
    kept: List[Dict[str, str]] = []
    for rec in records:
        reason = classify_record(rec, th)
        if reason:
            report.record_drop(reason)
            continue
        kept.append({**rec, "output": clean_response(rec.get("output", ""))})
    report.kept = len(kept)
    return kept, report
