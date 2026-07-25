#!/usr/bin/env python
"""End-to-end fine-tuning + distillation of a small open-source LLM.

Four phases:
  1. Load a specialised-domain instruction dataset from the HF Hub.
  2. (Optional) Have a larger teacher generate responses -> distillation targets.
  3. QLoRA-fine-tune the small student to imitate those targets.
  4. Save the LoRA adapter (+ tokenizer) to the output directory.

Runs on a free-tier Colab T4. Examples:
    python scripts/finetune_distill.py                          # pure domain fine-tune
    python scripts/finetune_distill.py --teacher hf             # local teacher distillation
    python scripts/finetune_distill.py --teacher nim            # NVIDIA NIM teacher (needs $NVIDIA_API_KEY)
    python scripts/finetune_distill.py --max-train 500 --epochs 1
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.finetune.config import FinetuneConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/finetune_config.yaml")
    p.add_argument("--teacher", choices=["passthrough", "hf", "nim"], default=None,
                   help="Override teacher.provider from the config.")
    p.add_argument("--student", default=None, help="Override the student model id.")
    p.add_argument("--dataset", default=None, help="Override the HF dataset name.")
    p.add_argument("--max-train", type=int, default=None, help="Cap training samples.")
    p.add_argument("--epochs", type=int, default=None, help="Override num_epochs.")
    p.add_argument("--output-dir", default=None, help="Override output dir.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = FinetuneConfig.from_yaml(args.config)

    # --- CLI overrides ---
    if args.teacher:
        cfg.teacher.provider = args.teacher
    if args.student:
        cfg.model.student = args.student
    if args.dataset:
        cfg.data.name = args.dataset
    if args.max_train:
        cfg.data.max_train_samples = args.max_train
    if args.epochs:
        cfg.train.num_epochs = args.epochs
    if args.output_dir:
        cfg.train.output_dir = args.output_dir

    # Heavy imports deferred until after arg parsing so --help stays instant.
    from src.finetune.data import load_domain_dataset, records_to_dicts, train_eval_split
    from src.finetune.distill import build_teacher, make_distillation_records, resolve_provider
    from src.finetune.model import build_student, count_trainable_parameters
    from src.finetune.train import train_student

    provider = resolve_provider(cfg.teacher)
    print(f"[1/4] Loading dataset '{cfg.data.name}' ...")
    ds = load_domain_dataset(cfg.data)
    train_ds, eval_ds = train_eval_split(ds, cfg.data.max_eval_samples)
    train_records = records_to_dicts(train_ds)
    eval_records = records_to_dicts(eval_ds)
    print(f"      train={len(train_records)}  eval={len(eval_records)}")

    print(f"[2/4] Building distillation targets (teacher provider='{provider}') ...")
    if provider == "passthrough":
        print("      passthrough -> fine-tuning on the dataset's gold answers.")
    else:
        teacher = build_teacher(cfg.teacher, cfg.model.teacher)
        teacher_outputs = teacher.generate(train_records)
        train_records = make_distillation_records(train_records, teacher_outputs)
        print(f"      generated {len(teacher_outputs)} teacher responses.")

    print(f"[3/4] Building student '{cfg.model.student}' with QLoRA ...")
    model, tokenizer = build_student(cfg.model, cfg.lora)
    trainable, total = count_trainable_parameters(model)
    print(f"      trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    print("[4/4] Fine-tuning ...")
    train_student(cfg, model, tokenizer, train_records, eval_records)
    print(f"Done. Adapter saved to {cfg.train.output_dir}")


if __name__ == "__main__":
    main()
