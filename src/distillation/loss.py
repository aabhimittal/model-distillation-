"""
RAD Loss — RAG-Augmented Distillation Loss.

L_RAD = α·L_RAG + β·L_KL + γ·L_CRA + δ·L_CE

L_RAG: KL(student || RAG-teacher)          — core novel term
L_KL:  KL(student || bare teacher)         — standard KD regulariser
L_CRA: max(0, margin - KL(p_T+ || p_T-))  — contrastive retrieval alignment
L_CE:  CrossEntropy(student, hard labels)  — grounding

All KL terms are multiplied by T² to preserve gradient magnitude (Hinton 2015).
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class RADLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        beta: float = 0.2,
        gamma: float = 0.1,
        delta: float = 0.2,
        cra_margin: float = 0.5,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.cra_margin = cra_margin
        self.ignore_index = ignore_index

    def _kl(self, log_student: torch.Tensor, teacher_probs: torch.Tensor) -> torch.Tensor:
        """
        KL(student || teacher) summed over vocab, mean over valid positions.
        log_student: (B, L, V) log-probabilities
        teacher_probs: (B, L, V) probabilities
        Returns scalar.
        """
        # kl_div expects (N, V); reshape, compute, reshape back
        B, L, V = log_student.shape
        kl = F.kl_div(
            log_student.reshape(B * L, V),
            teacher_probs.reshape(B * L, V),
            reduction="batchmean",
            log_target=False,
        )
        return kl * (self.T ** 2)

    def forward(
        self,
        student_logits: torch.Tensor,       # (B, L, V)
        rag_teacher_logits: torch.Tensor,   # (B, L, V)
        bare_teacher_logits: torch.Tensor,  # (B, L, V)
        neg_teacher_logits: torch.Tensor,   # (B, L, V)
        labels: torch.Tensor,               # (B, L) hard labels, -100 for padding
    ) -> Dict[str, torch.Tensor]:
        # Temperature-scaled soft distributions
        student_log_soft = F.log_softmax(student_logits / self.T, dim=-1)
        rag_soft = F.softmax(rag_teacher_logits / self.T, dim=-1)
        bare_soft = F.softmax(bare_teacher_logits / self.T, dim=-1)
        neg_soft = F.softmax(neg_teacher_logits / self.T, dim=-1)

        # L_RAG: student matches RAG-augmented teacher
        l_rag = self._kl(student_log_soft, rag_soft)

        # L_KL: student matches bare teacher (standard KD)
        l_kl = self._kl(student_log_soft, bare_soft)

        # L_CRA: RAG teacher must be margin-separated from negative-context teacher.
        # If the teacher ignores retrieved context, this term penalises it.
        rag_log_soft = F.log_softmax(rag_teacher_logits / self.T, dim=-1)
        B, L, V = rag_log_soft.shape
        kl_pos_neg = F.kl_div(
            rag_log_soft.reshape(B * L, V),
            neg_soft.reshape(B * L, V),
            reduction="batchmean",
            log_target=False,
        ) * (self.T ** 2)
        l_cra = torch.clamp(self.cra_margin - kl_pos_neg, min=0.0)

        # L_CE: cross entropy against hard labels for grounding
        # student_logits at original temperature (T=1) for CE
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
        )

        return {
            "total": total,
            "L_RAG": l_rag.detach(),
            "L_KL": l_kl.detach(),
            "L_CRA": l_cra.detach(),
            "L_CE": l_ce.detach(),
        }
