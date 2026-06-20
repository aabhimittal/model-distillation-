"""Frozen teacher model — logits only, always under torch.no_grad()."""

from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class TeacherModel:
    def __init__(self, model_name: str = "google/flan-t5-base", device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass returning raw logits of shape (B, L_dec, V)."""
        outputs = self.model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            decoder_input_ids=decoder_input_ids.to(self.device),
        )
        return outputs.logits  # stays on self.device

    @torch.no_grad()
    def generate_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, max_new_tokens: int = 64) -> list[str]:
        """Generate text for qualitative evaluation."""
        outputs = self.model.generate(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            max_new_tokens=max_new_tokens,
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def encode(self, texts: list[str], max_length: int = 512) -> dict:
        """Tokenise a list of texts and return tensors on self.device."""
        return self.tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
