"""Deduplication and train/eval decontamination.

Two failure modes that quietly invalidate a fine-tuning run:

1. **Duplicates.** Scraped instruction corpora are heavily redundant. Duplicated
   prompts get over-weighted in the loss and the student memorises them, which
   looks like fast convergence and is actually overfitting.
2. **Contamination.** If an eval example also appears in training, the reported
   score measures memorisation, not generalisation. With small domain datasets
   (a few thousand rows) this happens constantly, and near-duplicates — same
   question, reworded — evade the exact-match check people usually write.

Both are handled here with character n-gram shingling + Jaccard similarity.
Near-duplicate search uses an inverted index over shingles so only records that
share at least one shingle are ever compared, instead of all O(n²) pairs.

Pure Python — no torch, no embeddings, no network.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Dict, Iterable, List, Sequence, Set, Tuple

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def canonical(text: str) -> str:
    """Aggressive normalisation for matching: NFKC, casefold, strip punctuation.

    Deliberately lossy — the point is that "What is an ETF?" and "what is an etf"
    collapse to the same key. Never use this for anything the model will see.
    """
    text = unicodedata.normalize("NFKC", text or "")
    text = _PUNCT.sub(" ", text.casefold())
    return _WS.sub(" ", text).strip()


def record_key(record: Dict[str, str]) -> str:
    """Canonical identity of a record: its instruction plus any input."""
    return canonical(f"{record.get('instruction', '')} {record.get('input', '')}")


def fingerprint(text: str) -> str:
    """Stable short hash of the canonical form — the exact-duplicate bucket key."""
    return hashlib.blake2b(canonical(text).encode("utf-8"), digest_size=16).hexdigest()


def shingles(text: str, k: int = 5) -> Set[str]:
    """Character k-gram set of the canonical text.

    Character shingles (not word shingles) because they degrade gracefully on
    short texts and are robust to morphology and word order tweaks. Texts shorter
    than ``k`` yield a single shingle so they remain comparable rather than empty.
    """
    canon = canonical(text)
    if not canon:
        return set()
    if len(canon) <= k:
        return {canon}
    return {canon[i : i + k] for i in range(len(canon) - k + 1)}


def jaccard(a: Set[str], b: Set[str]) -> float:
    """|A∩B| / |A∪B|; 0.0 when both are empty (no evidence of similarity)."""
    if not a and not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    return inter / len(a | b)


def exact_dedup(records: Sequence[Dict[str, str]]) -> Tuple[List[Dict[str, str]], int]:
    """Drop records whose canonical instruction+input was already seen. O(n).

    The first occurrence wins, so ordering is stable and reproducible.
    """
    seen: Set[str] = set()
    kept: List[Dict[str, str]] = []
    for rec in records:
        key = fingerprint(record_key(rec))
        if key in seen:
            continue
        seen.add(key)
        kept.append(rec)
    return kept, len(records) - len(kept)


def _build_index(
    records: Sequence[Dict[str, str]], k: int
) -> Tuple[List[Set[str]], Dict[str, List[int]]]:
    """Shingle every record once and invert shingle -> record ids."""
    sets = [shingles(record_key(r), k) for r in records]
    index: Dict[str, List[int]] = {}
    for i, sh in enumerate(sets):
        for s in sh:
            index.setdefault(s, []).append(i)
    return sets, index


def near_dedup(
    records: Sequence[Dict[str, str]], threshold: float = 0.85, k: int = 5
) -> Tuple[List[Dict[str, str]], int]:
    """Drop records within ``threshold`` Jaccard of an earlier kept record.

    Candidates come from the inverted index, so a record is only compared against
    others sharing at least one shingle — far cheaper than all pairs while giving
    identical results (Jaccard is 0 without a shared shingle, so nothing is missed).
    """
    if threshold >= 1.0:
        return exact_dedup(records)
    sets, index = _build_index(records, k)
    kept_ids: List[int] = []
    kept_flag = [False] * len(records)

    for i, sh in enumerate(sets):
        candidates: Set[int] = set()
        for s in sh:
            for j in index.get(s, ()):
                if j < i and kept_flag[j]:
                    candidates.add(j)
        if any(jaccard(sh, sets[j]) >= threshold for j in candidates):
            continue
        kept_flag[i] = True
        kept_ids.append(i)

    return [records[i] for i in kept_ids], len(records) - len(kept_ids)


def decontaminate(
    train: Sequence[Dict[str, str]],
    evaluation: Sequence[Dict[str, str]],
    threshold: float = 0.85,
    k: int = 5,
) -> Tuple[List[Dict[str, str]], int]:
    """Remove training records that match any eval record (exact or near).

    Always removes from *train*, never from eval: the eval set is the measurement
    instrument and must stay fixed, otherwise scores are not comparable across runs.
    """
    if not evaluation:
        return list(train), 0

    eval_sets = [shingles(record_key(r), k) for r in evaluation]
    eval_exact = {fingerprint(record_key(r)) for r in evaluation}
    index: Dict[str, List[int]] = {}
    for i, sh in enumerate(eval_sets):
        for s in sh:
            index.setdefault(s, []).append(i)

    kept: List[Dict[str, str]] = []
    for rec in train:
        if fingerprint(record_key(rec)) in eval_exact:
            continue
        sh = shingles(record_key(rec), k)
        candidates: Set[int] = set()
        for s in sh:
            candidates.update(index.get(s, ()))
        if any(jaccard(sh, eval_sets[j]) >= threshold for j in candidates):
            continue
        kept.append(rec)
    return kept, len(train) - len(kept)
