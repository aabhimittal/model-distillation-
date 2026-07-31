"""Token budgeting and length-bucketed batching — the free-tier cost controls.

Two concrete savings, both measurable before a GPU is ever touched:

1. **Pre-flight budgeting.** Estimate prompt/response token counts and teacher
   API spend *before* launching a run, so a 2,000-record job that would exhaust
   a free NIM quota or a Colab session's wall clock is caught at second zero.

2. **Length-bucketed batching.** Batching randomly-ordered variable-length
   sequences pads every sequence to the batch maximum; on a skewed instruction
   corpus most of the tensor is padding, and padding costs exactly as much T4
   compute as real tokens. Grouping similar lengths into a batch cuts that waste
   substantially with no effect on the loss — the classic seq2seq trick, applied
   here to keep a free T4 inside its time limit.

Pure Python. ``estimate_tokens`` is a heuristic used only for pre-flight planning;
actual training always tokenises for real.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

# Empirical average for English BPE vocabularies. CJK and code are denser, so the
# estimate is intentionally conservative (see estimate_tokens).
CHARS_PER_TOKEN = 4.0


def estimate_tokens(text: str, chars_per_token: float = CHARS_PER_TOKEN) -> int:
    """Approximate token count from character length.

    Heuristic, not exact — for planning only. CJK text tokenises far denser than
    Latin (roughly one token per character), so scripts without spaces are counted
    at a heavier rate rather than being wildly under-estimated.
    """
    if not text:
        return 0
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be > 0")
    # Whitespace-free scripts (CJK) tokenise ~1 token/char; blend by space density.
    n_chars = len(text)
    n_spaces = text.count(" ")
    space_density = n_spaces / n_chars if n_chars else 0.0
    if space_density < 0.02:
        return max(1, n_chars)
    return max(1, round(n_chars / chars_per_token))


def record_lengths(
    records: Sequence[Dict[str, str]],
    length_fn: Callable[[str], int] = estimate_tokens,
) -> List[int]:
    """Estimated total token length (instruction + input + output) per record."""
    return [
        length_fn(r.get("instruction", ""))
        + length_fn(r.get("input", ""))
        + length_fn(r.get("output", ""))
        for r in records
    ]


@dataclass
class LengthStats:
    count: int
    total: int
    mean: float
    p50: int
    p95: int
    maximum: int
    over_limit: int

    def as_dict(self) -> Dict[str, float]:
        return {
            "count": self.count,
            "total_tokens": self.total,
            "mean": round(self.mean, 1),
            "p50": self.p50,
            "p95": self.p95,
            "max": self.maximum,
            "over_limit": self.over_limit,
        }


def _percentile(sorted_vals: Sequence[int], q: float) -> int:
    """Nearest-rank percentile; ``q`` in [0, 1]. Empty input -> 0."""
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def length_stats(lengths: Sequence[int], max_seq_length: int) -> LengthStats:
    """Distribution summary plus how many records exceed the sequence limit.

    ``over_limit`` is the number that will be silently truncated during training —
    the single most useful number for choosing ``max_seq_length``.
    """
    if not lengths:
        return LengthStats(0, 0, 0.0, 0, 0, 0, 0)
    ordered = sorted(lengths)
    return LengthStats(
        count=len(ordered),
        total=sum(ordered),
        mean=sum(ordered) / len(ordered),
        p50=_percentile(ordered, 0.50),
        p95=_percentile(ordered, 0.95),
        maximum=ordered[-1],
        over_limit=sum(1 for x in lengths if x > max_seq_length),
    )


def estimate_teacher_cost(
    n_records: int,
    avg_prompt_tokens: int,
    max_new_tokens: int,
    input_price_per_1k: float = 0.0,
    output_price_per_1k: float = 0.0,
) -> Dict[str, float]:
    """Worst-case teacher token usage and spend for a generation run.

    Assumes every generation runs to ``max_new_tokens``; real runs stop earlier at
    EOS, so this is an upper bound — the right bound for "will I blow my quota?".
    """
    if n_records < 0 or avg_prompt_tokens < 0 or max_new_tokens < 0:
        raise ValueError("counts must be non-negative")
    in_tok = n_records * avg_prompt_tokens
    out_tok = n_records * max_new_tokens
    return {
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "total_tokens": in_tok + out_tok,
        "estimated_cost": round(
            (in_tok / 1000.0) * input_price_per_1k
            + (out_tok / 1000.0) * output_price_per_1k,
            4,
        ),
    }


def bucket_by_length(lengths: Sequence[int], batch_size: int) -> List[List[int]]:
    """Group record indices into length-homogeneous batches.

    Returns lists of original indices (sorted by length, chunked), so the caller
    can reorder records without losing their identity. Shuffling *across* batches
    remains the caller's job — within-batch homogeneity is what saves padding, and
    batch order is what preserves gradient stochasticity.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    return [order[i : i + batch_size] for i in range(0, len(order), batch_size)]


def padding_waste(lengths: Sequence[int], batches: Sequence[Sequence[int]]) -> float:
    """Fraction of positions in the padded tensors that are pure padding.

    ``sum(batch_max · batch_len)`` is the padded footprint; the real tokens are
    ``sum(lengths)``. 0.0 means every batch is length-uniform; 0.6 means 60% of
    the T4's compute would be spent on pad tokens.
    """
    if not batches:
        return 0.0
    padded = sum(
        max((lengths[i] for i in batch), default=0) * len(batch) for batch in batches
    )
    if padded <= 0:
        return 0.0
    real = sum(lengths[i] for batch in batches for i in batch)
    return max(0.0, 1.0 - (real / padded))


def sequential_batches(n: int, batch_size: int) -> List[List[int]]:
    """Naive in-order batching — the baseline ``padding_waste`` is compared against."""
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    return [list(range(i, min(i + batch_size, n))) for i in range(0, n, batch_size)]
