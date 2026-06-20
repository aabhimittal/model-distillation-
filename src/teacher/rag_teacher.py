"""RAG-augmented teacher: three forward passes for RAD loss computation."""

from __future__ import annotations

from typing import List

import torch
from transformers import AutoTokenizer

from .teacher_model import TeacherModel
from src.rag.retriever import Retriever


class RAGTeacher:
    """
    Wraps a TeacherModel with a Retriever to produce three sets of logits:
      1. RAG logits    — teacher conditioned on retrieved contexts (positive)
      2. Bare logits   — teacher without any retrieved context
      3. Negative logits — teacher conditioned on irrelevant contexts (for L_CRA)

    All three passes are batched together into a single forward call to
    minimise GPU round-trips.
    """

    CONTEXT_TEMPLATE = (
        "Context 1: {ctx1}\n"
        "Context 2: {ctx2}\n"
        "Context 3: {ctx3}\n"
        "Question: {question}\n"
        "Answer:"
    )

    def __init__(
        self,
        teacher: TeacherModel,
        retriever: Retriever,
        max_input_length: int = 512,
    ):
        self.teacher = teacher
        self.retriever = retriever
        self.tokenizer: AutoTokenizer = teacher.tokenizer
        self.max_input_length = max_input_length

    def _format_rag_prompt(self, question: str, contexts: List[str]) -> str:
        # Pad to 3 contexts if fewer retrieved
        padded = (contexts + [""] * 3)[:3]
        return self.CONTEXT_TEMPLATE.format(
            ctx1=padded[0], ctx2=padded[1], ctx3=padded[2], question=question
        )

    def _encode_batch(self, texts: List[str]) -> dict:
        return self.tokenizer(
            texts,
            max_length=self.max_input_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    @torch.no_grad()
    def get_all_logits(
        self,
        questions: List[str],
        decoder_input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (rag_logits, bare_logits, neg_logits) each of shape (B, L_dec, V).

        Batches all three forward passes into one large batch of size 3B for
        efficiency, then splits. Negative contexts come from the batch-shifted
        retrieval (shift by n//2).
        """
        B = len(questions)

        # Retrieve positive and negative contexts
        pos_contexts = self.retriever.retrieve_batch(questions, top_k=3)
        neg_contexts = self.retriever.retrieve_negative(questions, top_k=3)

        # Build prompts
        rag_prompts = [self._format_rag_prompt(q, c) for q, c in zip(questions, pos_contexts)]
        bare_prompts = [f"Question: {q}\nAnswer:" for q in questions]
        neg_prompts = [self._format_rag_prompt(q, c) for q, c in zip(questions, neg_contexts)]

        # Combine into one 3B batch
        all_prompts = rag_prompts + bare_prompts + neg_prompts
        enc = self._encode_batch(all_prompts)

        input_ids = enc["input_ids"].to(self.teacher.device)
        attention_mask = enc["attention_mask"].to(self.teacher.device)

        # Tile decoder_input_ids to match 3B
        dec_ids = decoder_input_ids.to(self.teacher.device)
        dec_ids_tiled = dec_ids.repeat(3, 1)

        logits = self.teacher.get_logits(input_ids, attention_mask, dec_ids_tiled)

        rag_logits = logits[:B]
        bare_logits = logits[B : 2 * B]
        neg_logits = logits[2 * B :]

        return rag_logits, bare_logits, neg_logits
