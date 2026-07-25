"""
Retrieval-aware gating — the adaptive core of RAD.

The base L_RAD weights L_RAG and L_KL with fixed global constants (alpha, beta).
That assumes retrieval is uniformly useful, which is false: for some questions the
retriever returns the gold passage and the RAG teacher is far better than the bare
teacher; for others it returns noise and the RAG teacher is *worse*. A fixed alpha
forces the student to imitate the RAG teacher in both cases, injecting noise on the
examples where retrieval failed.

Two gates fix this at different granularities:

  RetrievalUtilityGate (RUG) — per-example.
      Measures whether retrieval raised the teacher's likelihood of the gold answer,
      then routes each example's soft-target mass between the RAG and bare teacher.

  TokenFusionGate (TRF) — per-token.
      Even inside a helpful example only some tokens carry retrieved knowledge
      (entity spans, dates); function words do not. Blends the two teachers into a
      single per-token target using their relative confidence.

Both gates are computed from logits the teacher already produced for L_RAD, so
they add no forward passes and no GPU cost. Both return detached tensors — they
are weightings, never learned parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-9


def per_example_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Mean negative log-likelihood of the gold answer under `logits`, per example.

    logits: (B, L, V) — at temperature 1, we want true likelihood not a softened one
    labels: (B, L) with `ignore_index` at padding
    returns: (B,)
    """
    log_probs = F.log_softmax(logits, dim=-1)

    mask = labels != ignore_index
    # gather() rejects negative indices, so neutralise padding before the lookup
    safe_labels = labels.masked_fill(~mask, 0)

    token_lp = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)  # (B, L)
    token_lp = token_lp * mask

    n_valid = mask.sum(dim=1).clamp(min=1)
    return -(token_lp.sum(dim=1) / n_valid)


def retrieval_utility(
    rag_logits: torch.Tensor,
    bare_logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Per-example retrieval utility:  u = NLL_bare(y*) - NLL_rag(y*)

    u > 0  retrieval raised the teacher's likelihood of the gold answer — it helped
    u = 0  retrieval was neutral
    u < 0  retrieval actively hurt (distractor passages pulled the teacher off)

    This is the signal every other feature in this module is built on. It is also
    what `RetrievalUtilityCurriculum` orders training data by.

    returns: (B,)
    """
    nll_rag = per_example_nll(rag_logits, labels, ignore_index)
    nll_bare = per_example_nll(bare_logits, labels, ignore_index)
    return nll_bare - nll_rag


class RetrievalUtilityGate(nn.Module):
    """
    Per-example gate g in [floor, ceiling] routing soft-target mass between teachers.

        g_i = sigmoid(u_i / tau)

        L_RAG <- mean_i  g_i      * KL(student_i || rag_teacher_i)
        L_KL  <- mean_i  (1-g_i)  * KL(student_i || bare_teacher_i)

    Because g + (1-g) = 1, total soft-target mass is conserved: the gate reallocates
    supervision between teachers rather than scaling it. When retrieval fails for the
    whole batch, g -> 0, L_RAG vanishes and the student falls back to standard KD.

    `floor`/`ceiling` clamp the range so neither teacher is ever fully abandoned;
    the default (0.05, 0.95) keeps a little signal from both, which stabilises early
    training when the teacher's NLL estimates are noisy.
    """

    def __init__(
        self,
        gate_temperature: float = 1.0,
        floor: float = 0.05,
        ceiling: float = 0.95,
        ignore_index: int = -100,
    ):
        super().__init__()
        if not 0.0 <= floor < ceiling <= 1.0:
            raise ValueError(f"require 0 <= floor < ceiling <= 1, got {floor}, {ceiling}")
        self.tau = gate_temperature
        self.floor = floor
        self.ceiling = ceiling
        self.ignore_index = ignore_index

    @torch.no_grad()
    def forward(
        self,
        rag_logits: torch.Tensor,
        bare_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """returns (gate, utility), both (B,) and detached."""
        u = retrieval_utility(rag_logits, bare_logits, labels, self.ignore_index)
        g = torch.sigmoid(u / self.tau)
        g = self.floor + (self.ceiling - self.floor) * g
        return g, u


class TokenFusionGate(nn.Module):
    """
    Per-token gate w in (0,1) blending the two teachers into one target distribution.

        w_{i,t} = sigmoid( (H_bare_{i,t} - H_rag_{i,t}) / kappa )
        p_fused = w * p_rag + (1 - w) * p_bare

    A convex combination of two distributions is a distribution, so KL(student ||
    p_fused) is well-defined. Where the RAG teacher is sharper than the bare teacher
    (it "knows" the answer because it read it), w -> 1 and the fused target follows
    retrieval; on generic function words the two teachers agree, w -> 0.5, and the
    target is their average.

    `center` removes the batch-mean entropy gap before the sigmoid. Without it the
    gate is biased: a teacher conditioned on ~400 extra context tokens is almost
    always lower-entropy regardless of whether the context was *relevant*, so raw
    entropy would push w -> 1 everywhere and the gate would carry no information.
    Centering isolates the per-token deviation, which is the part that actually
    signals retrieved knowledge.
    """

    def __init__(self, fusion_temperature: float = 1.0, center: bool = True):
        super().__init__()
        self.kappa = fusion_temperature
        self.center = center

    @staticmethod
    def _entropy(probs: torch.Tensor) -> torch.Tensor:
        """Shannon entropy over the vocab axis. (B,L,V) -> (B,L)"""
        return -(probs * probs.clamp_min(EPS).log()).sum(dim=-1)

    @torch.no_grad()
    def forward(
        self,
        rag_probs: torch.Tensor,
        bare_probs: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        rag_probs, bare_probs: (B, L, V) temperature-scaled probabilities
        valid_mask: (B, L) bool — padding excluded from the centering statistic
        returns: (B, L) fusion weights, detached
        """
        gap = self._entropy(bare_probs) - self._entropy(rag_probs)  # (B, L)

        if self.center:
            if valid_mask is not None and valid_mask.any():
                mean_gap = gap[valid_mask].mean()
            else:
                mean_gap = gap.mean()
            gap = gap - mean_gap

        return torch.sigmoid(gap / self.kappa)


def fuse_teacher_distributions(
    rag_probs: torch.Tensor,
    bare_probs: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """
    Convex per-token blend of two teacher distributions.

    rag_probs, bare_probs: (B, L, V)
    weights: (B, L) from TokenFusionGate
    returns: (B, L, V), each row a valid probability distribution
    """
    w = weights.unsqueeze(-1)
    return w * rag_probs + (1.0 - w) * bare_probs
