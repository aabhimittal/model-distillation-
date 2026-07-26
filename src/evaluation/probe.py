"""
Knowledge Retention Probe — the measurement that actually tests the RAD thesis.

The claim behind RAG-Augmented Distillation is specific:

    the student internalises *retrieved* knowledge into its weights,
    so it answers correctly at inference with no retriever attached.

Aggregate EM/F1 cannot test that claim. A student can post a respectable F1 purely
from questions the bare teacher already answered — questions where retrieval was
never the point. The retrieval-dependent subset, which is the only part that
measures knowledge transfer, is a minority of the eval set and gets averaged away.

This probe partitions the eval set using the two teachers as instruments:

    retrieval_dependent  RAG teacher correct, bare teacher wrong
                         → answering needs the retrieved passage
    parametric           bare teacher correct
                         → already in the teacher's weights, retrieval irrelevant
    hard                 neither teacher correct
                         → beyond the teacher; excluded from headline rates

and reports the student separately on each. Two rates fall out, both in [0,1]:

    Retention Rate        student EM on retrieval_dependent
    Parametric Preservation  student EM on parametric

Both denominators are 1.0 by construction — the strata are *defined* by teacher
correctness — so each rate reads directly as "what fraction of this knowledge type
survived distillation". Retention Rate is the headline number of the project.

A third diagnostic closes the loop:

    Retrieval Independence Gap (RIG) = EM(student + retrieval) - EM(student alone)

RIG near 0 means bolting a retriever back on adds nothing: the knowledge really is
in the weights. A large RIG means the student still depends on retrieval and
distillation did not internalise it — the failure this project exists to avoid.

The probe consumes prediction lists that `scripts/evaluate.py` already generates,
so it costs no extra forward passes.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .evaluator import exact_match, token_f1

RETRIEVAL_DEPENDENT = "retrieval_dependent"
PARAMETRIC = "parametric"
HARD = "hard"


class KnowledgeRetentionProbe:
    """Stratified evaluation isolating transferred vs. pre-existing knowledge."""

    def __init__(self, match_fn=exact_match, threshold: float = 1.0):
        """
        match_fn: scoring function, `exact_match` (default) or `token_f1`
        threshold: score at or above which a prediction counts as correct.
                   With token_f1, a softer threshold like 0.6 is reasonable.
        """
        self.match_fn = match_fn
        self.threshold = threshold

    def _correct(self, pred: str, ref: str) -> bool:
        return self.match_fn(pred, ref) >= self.threshold

    def stratify(
        self,
        bare_teacher_preds: Sequence[str],
        rag_teacher_preds: Sequence[str],
        references: Sequence[str],
    ) -> Dict[str, List[int]]:
        """Partition eval indices into the three strata."""
        n = len(references)
        if not (len(bare_teacher_preds) == len(rag_teacher_preds) == n):
            raise ValueError(
                "prediction lists and references must be the same length: "
                f"{len(bare_teacher_preds)}, {len(rag_teacher_preds)}, {n}"
            )

        strata: Dict[str, List[int]] = {RETRIEVAL_DEPENDENT: [], PARAMETRIC: [], HARD: []}
        for i in range(n):
            bare_ok = self._correct(bare_teacher_preds[i], references[i])
            rag_ok = self._correct(rag_teacher_preds[i], references[i])
            if bare_ok:
                strata[PARAMETRIC].append(i)
            elif rag_ok:
                strata[RETRIEVAL_DEPENDENT].append(i)
            else:
                strata[HARD].append(i)
        return strata

    def _score_subset(
        self, preds: Sequence[str], refs: Sequence[str], indices: Sequence[int]
    ) -> Dict[str, float]:
        if not indices:
            return {"exact_match": 0.0, "f1": 0.0, "n": 0}
        em = [exact_match(preds[i], refs[i]) for i in indices]
        f1 = [token_f1(preds[i], refs[i]) for i in indices]
        return {
            "exact_match": float(np.mean(em)),
            "f1": float(np.mean(f1)),
            "n": len(indices),
        }

    def evaluate(
        self,
        student_preds: Sequence[str],
        bare_teacher_preds: Sequence[str],
        rag_teacher_preds: Sequence[str],
        references: Sequence[str],
        student_with_retrieval_preds: Optional[Sequence[str]] = None,
    ) -> Dict[str, object]:
        """
        Full probe. `student_with_retrieval_preds` is optional; supply it to get the
        Retrieval Independence Gap.
        """
        strata = self.stratify(bare_teacher_preds, rag_teacher_preds, references)

        per_stratum = {
            name: self._score_subset(student_preds, references, idxs)
            for name, idxs in strata.items()
        }

        retention = per_stratum[RETRIEVAL_DEPENDENT]["exact_match"]
        preservation = per_stratum[PARAMETRIC]["exact_match"]

        result: Dict[str, object] = {
            "retention_rate": retention,
            "parametric_preservation": preservation,
            "per_stratum": per_stratum,
            "stratum_sizes": {k: len(v) for k, v in strata.items()},
        }

        if student_with_retrieval_preds is not None:
            em_alone = float(
                np.mean([exact_match(p, r) for p, r in zip(student_preds, references)])
            )
            em_with = float(
                np.mean(
                    [
                        exact_match(p, r)
                        for p, r in zip(student_with_retrieval_preds, references)
                    ]
                )
            )
            result["retrieval_independence_gap"] = em_with - em_alone
            result["student_em_alone"] = em_alone
            result["student_em_with_retrieval"] = em_with

        return result

    def format_report(self, result: Dict[str, object]) -> str:
        """Human-readable summary for notebooks and CLI output."""
        sizes = result["stratum_sizes"]
        per = result["per_stratum"]
        lines = [
            "Knowledge Retention Probe",
            "=" * 64,
            f"{'Stratum':<24} {'n':>6} {'EM':>10} {'F1':>10}",
            "-" * 64,
        ]
        for name in (RETRIEVAL_DEPENDENT, PARAMETRIC, HARD):
            m = per[name]
            lines.append(
                f"{name:<24} {sizes[name]:>6} "
                f"{m['exact_match']*100:>9.1f}% {m['f1']*100:>9.1f}%"
            )
        lines += [
            "-" * 64,
            f"Retention Rate           {result['retention_rate']*100:>9.1f}%   "
            "(retrieved knowledge now in student weights)",
            f"Parametric Preservation  {result['parametric_preservation']*100:>9.1f}%   "
            "(pre-existing knowledge kept)",
        ]
        if "retrieval_independence_gap" in result:
            rig = result["retrieval_independence_gap"]
            verdict = "independent" if abs(rig) < 0.02 else "still retrieval-reliant"
            lines.append(
                f"Retrieval Independence   {rig*100:>+9.1f}pp  ({verdict})"
            )
        lines.append("=" * 64)
        return "\n".join(lines)
