"""Student model with gradient checkpointing for memory efficiency."""

from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class StudentModel:
    def __init__(self, model_name: str = "google/flan-t5-small", device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.model.gradient_checkpointing_enable()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits of shape (B, L_dec, V)."""
        outputs = self.model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            decoder_input_ids=decoder_input_ids.to(self.device),
        )
        return outputs.logits

    @torch.no_grad()
    def generate_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, max_new_tokens: int = 64
    ) -> list[str]:
        self.model.eval()
        outputs = self.model.generate(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            max_new_tokens=max_new_tokens,
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def parameters(self):
        return self.model.parameters()

    def save(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
