"""Teacher-driven distillation targets (sequence-level knowledge distillation).

We use **sequence-level KD** (Kim & Rush, 2016): a large open-source teacher
generates high-quality responses on the domain prompts, and the small student is
fine-tuned (SFT + QLoRA) to imitate them. Unlike logit distillation this is
tokenizer-agnostic — the teacher can be a *different* model family, or even a
remote API (NVIDIA NIM) — which is exactly what makes it practical on a free
Colab GPU where the teacher may not fit in memory alongside the student.

Three providers:
  * ``passthrough`` — no teacher; distil against the dataset's gold answers.
                      This is ordinary domain fine-tuning and needs no GPU/teacher.
  * ``hf``          — load the teacher locally with transformers and generate.
  * ``nim``         — call an NVIDIA build.nvidia.com / NIM OpenAI-compatible API.
"""

from __future__ import annotations

import os
from typing import Dict, List

from .chat import to_messages
from .config import TeacherConfig

VALID_PROVIDERS = ("passthrough", "hf", "nim")


def resolve_provider(cfg: TeacherConfig) -> str:
    """Normalise + validate the configured provider name (pure, testable)."""
    provider = (cfg.provider or "passthrough").lower().strip()
    if provider not in VALID_PROVIDERS:
        raise ValueError(
            f"Unknown teacher provider {provider!r}; expected one of {VALID_PROVIDERS}"
        )
    return provider


class PassthroughTeacher:
    """Returns the gold answer unchanged — the fine-tuning-only baseline."""

    def generate(self, records: List[Dict[str, str]]) -> List[str]:
        return [r.get("output", "") for r in records]


class NIMTeacher:
    """NVIDIA NIM / build.nvidia.com teacher over the OpenAI-compatible REST API.

    Reads the API key from ``NVIDIA_API_KEY``. Free developer credits at
    https://build.nvidia.com make this a zero-local-GPU way to obtain a
    frontier-class teacher signal.
    """

    def __init__(self, cfg: TeacherConfig):
        self.cfg = cfg
        self.api_key = os.environ.get("NVIDIA_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "provider='nim' requires the NVIDIA_API_KEY environment variable "
                "(get free credits at https://build.nvidia.com)."
            )

    def _client(self):
        # openai SDK speaks the NIM endpoint; imported lazily.
        from openai import OpenAI

        return OpenAI(base_url=self.cfg.nim_base_url, api_key=self.api_key)

    def generate(self, records: List[Dict[str, str]]) -> List[str]:
        client = self._client()
        outputs: List[str] = []
        for r in records:
            messages = to_messages(r.get("instruction", ""), r.get("input", ""))
            resp = client.chat.completions.create(
                model=self.cfg.nim_model,
                messages=messages,
                max_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
            )
            outputs.append((resp.choices[0].message.content or "").strip())
        return outputs


class HFTeacher:
    """Local HuggingFace teacher generating with greedy/sampling decoding."""

    def __init__(self, cfg: TeacherConfig, model_name: str, batch_size: int = 8):
        self.cfg = cfg
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._tok = None

    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self._model.eval()

    def generate(self, records: List[Dict[str, str]]) -> List[str]:
        import torch

        if self._model is None:
            self._load()

        tok = self._tok
        outputs: List[str] = []
        for start in range(0, len(records), self.batch_size):
            chunk = records[start : start + self.batch_size]
            prompts = [
                tok.apply_chat_template(
                    to_messages(r.get("instruction", ""), r.get("input", "")),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for r in chunk
            ]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True)
            enc = {k: v.to(self._model.device) for k, v in enc.items()}
            with torch.no_grad():
                gen = self._model.generate(
                    **enc,
                    max_new_tokens=self.cfg.max_new_tokens,
                    do_sample=self.cfg.temperature > 0,
                    temperature=max(self.cfg.temperature, 1e-5),
                    top_p=self.cfg.top_p,
                    pad_token_id=tok.pad_token_id,
                )
            for i in range(len(chunk)):
                new_tokens = gen[i][enc["input_ids"].shape[1] :]
                outputs.append(tok.decode(new_tokens, skip_special_tokens=True).strip())
        return outputs


def build_teacher(cfg: TeacherConfig, model_name: str = ""):
    """Factory: return the teacher implementation for the configured provider."""
    provider = resolve_provider(cfg)
    if provider == "passthrough":
        return PassthroughTeacher()
    if provider == "nim":
        return NIMTeacher(cfg)
    return HFTeacher(cfg, model_name)


def make_distillation_records(
    records: List[Dict[str, str]], teacher_outputs: List[str]
) -> List[Dict[str, str]]:
    """Replace each record's target with the teacher's response (the distil signal).

    Records whose teacher output came back empty fall back to the original gold
    answer so a flaky generation never injects an empty training target.
    """
    if len(records) != len(teacher_outputs):
        raise ValueError(
            f"record/output length mismatch: {len(records)} vs {len(teacher_outputs)}"
        )
    distilled = []
    for r, out in zip(records, teacher_outputs):
        target = out.strip() if out and out.strip() else r.get("output", "")
        distilled.append({**r, "output": target})
    return distilled
