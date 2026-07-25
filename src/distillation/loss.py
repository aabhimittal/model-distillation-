"""
RAD Loss — RAG-Augmented Distillation Loss.

L_RAD = α·L_RAG + β·L_KL + γ·L_CRA + δ·L_CE + ε·L_FUSE

L_RAG:  KL(student || RAG-teacher)          — core novel term, gated per-example by RUG
L_KL:   KL(student || bare teacher)         — standard KD, receives the complement gate
L_CRA:  max(0, margin - KL(p_T+ || p_T-))   — contrastive retrieval alignment
L_CE:   CrossEntropy(student, hard labels)  — grounding
L_FUSE: KL(student || entropy-blended teacher) — token-level retrieval fusion (TRF)

All KL terms are multiplied by T² to preserve gradient magnitude (Hinton 2015) and
are averaged over *valid* positions only — padding is masked out before reduction,
so sequence-length variation does not dilute the loss.

Ablations (each independent):
    use_rug=False   → alpha/beta revert to fixed global weights (original L_RAD)
    epsilon=0.0     → no token-level fusion
    gamma=0.0       → no contrastive retrieval alignment
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gating import (
    RetrievalUtilityGate,
    TokenFusionGate,
    fuse_teacher_distributions,
)

EPS = 1e-9


class RADLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        beta: float = 0.2,
        gamma: float = 0.1,
        delta: float = 0.2,
        epsilon: float = 0.3,
        cra_margin: float = 0.5,
        ignore_index: int = -100,
        use_rug: bool = True,
        gate_temperature: float = 1.0,
        gate_floor: float = 0.05,
        gate_ceiling: float = 0.95,
        fusion_temperature: float = 1.0,
        fusion_center: bool = True,
    ):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.cra_margin = cra_margin
        self.ignore_index = ignore_index
        self.use_rug = use_rug

        self.rug = (
            RetrievalUtilityGate(
                gate_temperature=gate_temperature,
                floor=gate_floor,
                ceiling=gate_ceiling,
                ignore_index=ignore_index,
            )
            if use_rug
            else None
        )
        self.trf = (
            TokenFusionGate(fusion_temperature=fusion_temperature, center=fusion_center)
            if epsilon > 0.0
            else None
        )

    def _kl_per_example(
        self,
        log_student: torch.Tensor,   # (B, L, V) log-probs
        teacher_probs: torch.Tensor, # (B, L, V) probs
        valid_mask: torch.Tensor,    # (B, L) bool
    ) -> torch.Tensor:
        """
        KL(student || teacher) summed over vocab, averaged over valid positions.
        Returns (B,) so callers can apply per-example weights before reducing.
        """
        kl_tok = (
            teacher_probs * (teacher_probs.clamp_min(EPS).log() - log_student)
        ).sum(dim=-1)                       # (B, L)
        kl_tok = kl_tok * valid_mask
        n_valid = valid_mask.sum(dim=1).clamp(min=1)
        return (kl_tok.sum(dim=1) / n_valid) * (self.T ** 2)

    def forward(
        self,
        student_logits: torch.Tensor,       # (B, L, V)
        rag_teacher_logits: torch.Tensor,   # (B, L, V)
        bare_teacher_logits: torch.Tensor,  # (B, L, V)
        neg_teacher_logits: torch.Tensor,   # (B, L, V)
        labels: torch.Tensor,               # (B, L), -100 at padding
    ) -> Dict[str, torch.Tensor]:
        valid_mask = (labels != self.ignore_index).float()

        # Temperature-scaled distributions
        student_log_soft = F.log_softmax(student_logits / self.T, dim=-1)
        rag_soft = F.softmax(rag_teacher_logits / self.T, dim=-1)
        bare_soft = F.softmax(bare_teacher_logits / self.T, dim=-1)
        neg_soft = F.softmax(neg_teacher_logits / self.T, dim=-1)

        kl_rag = self._kl_per_example(student_log_soft, rag_soft, valid_mask)    # (B,)
        kl_bare = self._kl_per_example(student_log_soft, bare_soft, valid_mask)  # (B,)

        # --- Retrieval-Utility Gating (per-example) ---------------------------
        if self.rug is not None:
            gate, utility = self.rug(rag_teacher_logits, bare_teacher_logits, labels)
            # Plain mean (not gate-normalised): when retrieval fails batch-wide the
            # gate -> 0 and L_RAG must genuinely vanish. Normalising by sum(gate)
            # would rescale it back to full magnitude and defeat the mechanism.
            l_rag = (gate * kl_rag).mean()
            l_kl = ((1.0 - gate) * kl_bare).mean()
        else:
            gate = torch.full_like(kl_rag, 0.5)
            utility = torch.zeros_like(kl_rag)
            l_rag = kl_rag.mean()
            l_kl = kl_bare.mean()

        # --- Token-level Retrieval Fusion -------------------------------------
        if self.trf is not None:
            w = self.trf(rag_soft, bare_soft, valid_mask.bool())
            fused = fuse_teacher_distributions(rag_soft, bare_soft, w)
            l_fuse = self._kl_per_example(student_log_soft, fused, valid_mask).mean()
            mean_fusion_w = w[valid_mask.bool()].mean() if valid_mask.any() else w.mean()
        else:
            l_fuse = torch.zeros((), device=student_logits.device)
            mean_fusion_w = torch.zeros((), device=student_logits.device)

        # --- Contrastive Retrieval Alignment ----------------------------------
        # Penalise a teacher whose RAG distribution is indistinguishable from its
        # negative-context distribution — i.e. one that ignored the retrieved text.
        rag_log_soft = F.log_softmax(rag_teacher_logits / self.T, dim=-1)
        kl_pos_neg = (
            rag_soft * (rag_soft.clamp_min(EPS).log() - torch.log(neg_soft.clamp_min(EPS)))
        ).sum(dim=-1)
        kl_pos_neg = ((kl_pos_neg * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1))
        kl_pos_neg = kl_pos_neg.mean() * (self.T ** 2)
        l_cra = torch.clamp(self.cra_margin - kl_pos_neg, min=0.0)

        # --- Hard-label grounding ---------------------------------------------
        l_ce = F.cross_entropy(
            student_logits.reshape(-1, student_logits.size(-1)),
            labels.reshape(-1),
            ignore_index=self.ignore_index,
        )

        total = (
            self.alpha * l_rag
            + self.beta * l_kl
            + self.gamma * l_cra
            + self.delta * l_ce
            + self.epsilon * l_fuse
        )

        return {
            "total": total,
            "L_RAG": l_rag.detach(),
            "L_KL": l_kl.detach(),
            "L_CRA": l_cra.detach(),
            "L_CE": l_ce.detach(),
            "L_FUSE": l_fuse.detach(),
            # Diagnostics — these are what make the gate legible in the loss curves
            "gate_mean": gate.mean().detach(),
            "utility_mean": utility.mean().detach(),
            "fusion_w_mean": mean_fusion_w.detach(),
        }
