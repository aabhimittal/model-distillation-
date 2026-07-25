"""Supervised fine-tuning loop with prompt-masked labels (heavy imports lazy)."""

from __future__ import annotations

from typing import Dict, List

from .chat import build_prompt, build_supervised_labels, to_messages
from .config import FinetuneConfig


def _encode_prompt(tokenizer, instruction: str, input_text: str) -> List[int]:
    """Tokenise the prompt, preferring the tokenizer's own chat template."""
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            to_messages(instruction, input_text),
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = build_prompt(instruction, input_text)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def tokenize_records(records: List[Dict[str, str]], tokenizer, max_length: int):
    """Turn distilled records into a HF Dataset of {input_ids, labels, attention_mask}."""
    from datasets import Dataset

    rows = []
    for r in records:
        prompt_ids = _encode_prompt(tokenizer, r.get("instruction", ""), r.get("input", ""))
        response_ids = tokenizer(r.get("output", ""), add_special_tokens=False)["input_ids"]
        ex = build_supervised_labels(
            prompt_ids,
            response_ids,
            eos_id=tokenizer.eos_token_id,
            max_length=max_length,
        )
        ex["attention_mask"] = [1] * len(ex["input_ids"])
        rows.append(ex)
    return Dataset.from_list(rows)


def train_student(cfg: FinetuneConfig, model, tokenizer, train_records, eval_records=None):
    """Run QLoRA SFT on the distilled records and save the adapter to output_dir."""
    import torch
    from transformers import Trainer, TrainingArguments
    from transformers import DataCollatorForSeq2Seq

    train_ds = tokenize_records(train_records, tokenizer, cfg.model.max_seq_length)
    eval_ds = (
        tokenize_records(eval_records, tokenizer, cfg.model.max_seq_length)
        if eval_records
        else None
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer, label_pad_token_id=-100, padding=True, return_tensors="pt"
    )

    args = TrainingArguments(
        output_dir=cfg.train.output_dir,
        per_device_train_batch_size=cfg.train.batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        learning_rate=cfg.train.learning_rate,
        num_train_epochs=cfg.train.num_epochs,
        warmup_ratio=cfg.train.warmup_ratio,
        weight_decay=cfg.train.weight_decay,
        logging_steps=cfg.train.logging_steps,
        save_steps=cfg.train.save_steps,
        max_grad_norm=cfg.train.max_grad_norm,
        fp16=cfg.train.fp16 and torch.cuda.is_available(),
        bf16=cfg.train.bf16 and torch.cuda.is_available(),
        gradient_checkpointing=True,
        report_to="none",
        seed=cfg.train.seed,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(cfg.train.output_dir)
    tokenizer.save_pretrained(cfg.train.output_dir)
    return trainer
