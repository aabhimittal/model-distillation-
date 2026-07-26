"""
Calibration under RAG-augmented distillation.

A RAG teacher is confident for a reason it cannot pass on: it has the answer in its
context window. The student trained on those soft labels inherits the *confidence*
without the evidence. That is a specific and dangerous failure mode for a deployed
standalone model — it looks certain precisely where it is guessing from a fact it
half-memorised.

Expected Calibration Error quantifies it (Guo et al. 2017): bin predictions by
confidence, compare each bin's mean confidence against its accuracy.

    ECE = Σ_b (n_b / N) · | acc(b) - conf(b) |

The interesting comparison is not "is the student calibrated" in isolation but
whether ECE rises going teacher → student, and whether it rises more on the
retrieval-dependent stratum than the parametric one. That would show retrieval
confidence transferring without retrieval evidence.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch


def expected_calibration_error(
    confidences: Sequence[float],
    correctness: Sequence[float],
    n_bins: int = 10,
) -> Dict[str, object]:
    """
    confidences: model's confidence in its own prediction, each in [0, 1]
    correctness: 1.0 if that prediction was right, else 0.0
    n_bins: equal-width bins over [0, 1]

    Returns ECE, the signed over/under-confidence, and per-bin detail for plotting
    a reliability diagram.
    """
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correctness, dtype=np.float64)
    if conf.shape != corr.shape:
        raise ValueError(f"shape mismatch: {conf.shape} vs {corr.shape}")
    if conf.size == 0:
        return {"ece": 0.0, "overconfidence": 0.0, "bins": [], "n": 0}

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    signed = 0.0
    bins = []

    for lo, hi in zip(edges[:-1], edges[1:]):
        # Include the left edge; the top bin also takes the right edge so conf==1.0 lands
        in_bin = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        n_b = int(in_bin.sum())
        if n_b == 0:
            bins.append({"lo": float(lo), "hi": float(hi), "n": 0, "acc": 0.0, "conf": 0.0})
            continue
        acc_b = float(corr[in_bin].mean())
        conf_b = float(conf[in_bin].mean())
        weight = n_b / conf.size
        ece += weight * abs(acc_b - conf_b)
        signed += weight * (conf_b - acc_b)
        bins.append(
            {"lo": float(lo), "hi": float(hi), "n": n_b, "acc": acc_b, "conf": conf_b}
        )

    return {
        "ece": float(ece),
        "overconfidence": float(signed),  # >0 means confident beyond accuracy
        "bins": bins,
        "n": int(conf.size),
    }


@torch.no_grad()
def sequence_confidence(
    scores: Sequence[torch.Tensor],
    sequences: torch.Tensor,
    pad_token_id: int,
    eos_token_id: Optional[int] = None,
) -> np.ndarray:
    """
    Per-sequence confidence from a HuggingFace `generate(output_scores=True,
    return_dict_in_generate=True)` result.

    Confidence is the geometric mean of the chosen tokens' probabilities — length
    normalised, so long answers are not automatically judged less confident.

    scores: tuple of (B, V) logit tensors, one per generated step
    sequences: (B, L_gen) token ids as returned by generate()
    returns: (B,) confidences in [0, 1]
    """
    if len(scores) == 0:
        return np.zeros(sequences.size(0), dtype=np.float64)

    # generate() prepends the decoder start token, so generated tokens begin at 1
    gen_tokens = sequences[:, 1 : 1 + len(scores)]

    log_probs = []
    for t, step_logits in enumerate(scores):
        step_lp = torch.log_softmax(step_logits.float(), dim=-1)
        chosen = gen_tokens[:, t].unsqueeze(-1)
        log_probs.append(step_lp.gather(-1, chosen).squeeze(-1))

    lp = torch.stack(log_probs, dim=1)                    # (B, T)

    valid = gen_tokens != pad_token_id
    if eos_token_id is not None:
        valid = valid & (gen_tokens != eos_token_id)
    valid = valid.float()

    n_valid = valid.sum(dim=1).clamp(min=1)
    mean_lp = (lp * valid).sum(dim=1) / n_valid
    return mean_lp.exp().cpu().numpy().astype(np.float64)
