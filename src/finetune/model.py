"""Build the small student model with 4-bit QLoRA adapters (free-tier friendly).

All heavy imports are lazy. ``build_student`` returns a ``(model, tokenizer)`` pair
where the base weights are frozen + quantised to 4-bit and only small LoRA adapters
are trainable — the combination that lets a 0.5B–1.5B model fine-tune inside a T4's
15 GB. On a CPU-only box (no bitsandbytes/CUDA) it transparently falls back to a
full-precision LoRA setup so the pipeline still runs, just slower.
"""

from __future__ import annotations

from typing import Tuple

from .config import LoraConfig, ModelConfig


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def build_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        # Causal LMs frequently ship without a pad token; reuse EOS for batching.
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def build_student(model_cfg: ModelConfig, lora_cfg: LoraConfig) -> Tuple[object, object]:
    """Return (peft_model, tokenizer) ready for supervised fine-tuning."""
    import torch
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM

    tok = build_tokenizer(model_cfg.student)

    use_4bit = lora_cfg.load_in_4bit and _cuda_available()
    quant_config = None
    if use_4bit:
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.student,
        quantization_config=quant_config,
        torch_dtype=torch.float16 if _cuda_available() else torch.float32,
        device_map="auto" if _cuda_available() else None,
    )
    model.config.use_cache = False  # required with gradient checkpointing

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    peft_config = PeftLoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        target_modules=lora_cfg.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    return model, tok


def count_trainable_parameters(model) -> Tuple[int, int]:
    """Return (trainable, total) parameter counts — for logging the LoRA ratio."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
