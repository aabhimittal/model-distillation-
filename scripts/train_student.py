"""
Phase 3: Train the student model with RAD Loss.

Usage:
    python scripts/train_student.py [--config configs/distillation_config.yaml] [--disable-cra]

If pre-generated soft labels are found in soft_labels/, they are loaded from disk.
Otherwise the RAG teacher runs online during training (requires more GPU memory).
"""

import argparse
import sys
from pathlib import Path
from functools import partial

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
from torch.utils.data import DataLoader

from src.data.dataset_loader import load_squad, filter_answerable, format_for_training, collate_fn
from src.rag.embedder import Embedder
from src.rag.chroma_store import ChromaStore
from src.rag.retriever import Retriever
from src.teacher.teacher_model import TeacherModel
from src.teacher.rag_teacher import RAGTeacher
from src.student.student_model import StudentModel
from src.distillation.loss import RADLoss
from src.distillation.trainer import RADTrainer


def main(config_path: str, disable_cra: bool) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dc = cfg["distillation"]
    tc = cfg["training"]

    print("Loading student model...")
    student = StudentModel(cfg["models"]["student"])

    print("Loading teacher model...")
    teacher = TeacherModel(cfg["models"]["teacher"])

    print("Setting up RAG retriever...")
    embedder = Embedder(cfg["models"]["embedder"])
    store = ChromaStore(
        persist_dir=cfg["rag"]["chroma_persist_dir"],
        collection_name=cfg["rag"]["collection_name"],
    )
    retriever = Retriever(embedder, store)
    rag_teacher = RAGTeacher(
        teacher, retriever, max_input_length=cfg["dataset"]["max_input_length"]
    )

    print("Loading training dataset...")
    train_ds = load_squad(cfg["dataset"]["train_split"])
    train_ds = filter_answerable(train_ds)
    val_ds = load_squad(cfg["dataset"]["val_split"])
    val_ds = filter_answerable(val_ds)

    fmt = partial(
        format_for_training,
        tokenizer=student.tokenizer,
        max_input_length=cfg["dataset"]["max_input_length"],
        max_target_length=cfg["dataset"]["max_target_length"],
    )
    train_formatted = train_ds.map(fmt, remove_columns=train_ds.column_names)
    val_formatted = val_ds.map(fmt, remove_columns=val_ds.column_names)
    train_formatted.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_formatted.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_loader = DataLoader(
        train_formatted,
        batch_size=tc["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_formatted,
        batch_size=tc["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    loss_fn = RADLoss(
        temperature=dc["temperature"],
        alpha=dc["alpha"],
        beta=dc["beta"],
        gamma=dc["gamma"],
        delta=dc["delta"],
        cra_margin=dc["cra_margin"],
    )

    total_steps = len(train_loader) // tc["gradient_accumulation_steps"] * tc["num_epochs"]
    warmup_steps = int(total_steps * tc["warmup_ratio"])

    trainer = RADTrainer(
        student=student,
        rag_teacher=rag_teacher,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=tc["output_dir"],
        learning_rate=tc["learning_rate"],
        warmup_steps=warmup_steps,
        max_grad_norm=tc["max_grad_norm"],
        fp16=tc["fp16"],
        logging_steps=tc["logging_steps"],
        eval_steps=tc["eval_steps"],
        save_steps=tc["save_steps"],
        gradient_accumulation_steps=tc["gradient_accumulation_steps"],
        disable_cra=disable_cra,
    )

    print(f"\nTraining for {tc['num_epochs']} epochs ({total_steps} total steps)...")
    if disable_cra:
        print("  NOTE: Contrastive Retrieval Alignment (L_CRA) is DISABLED (ablation mode)")
    trainer.train(num_epochs=tc["num_epochs"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/distillation_config.yaml")
    parser.add_argument(
        "--disable-cra",
        action="store_true",
        help="Disable L_CRA term for ablation study",
    )
    args = parser.parse_args()
    main(args.config, args.disable_cra)
