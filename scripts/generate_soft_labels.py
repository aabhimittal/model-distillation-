"""
Phase 2: Pre-generate teacher soft labels and save to disk as .npz files.

This decouples the expensive teacher inference from student training,
allowing training to restart cheaply without re-running the teacher.

Usage:
    python scripts/generate_soft_labels.py [--config configs/distillation_config.yaml]

Output: soft_labels/{example_id}.npz with keys: rag_logits, bare_logits, neg_logits
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial

from src.data.dataset_loader import load_squad, filter_answerable, format_for_training, collate_fn
from src.rag.embedder import Embedder
from src.rag.chroma_store import ChromaStore
from src.rag.retriever import Retriever
from src.teacher.teacher_model import TeacherModel
from src.teacher.rag_teacher import RAGTeacher


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["soft_labels"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    dataset = load_squad(cfg["dataset"]["train_split"])
    dataset = filter_answerable(dataset)
    print(f"{len(dataset)} training examples.")

    print("Loading teacher model...")
    teacher = TeacherModel(cfg["models"]["teacher"])

    print("Setting up retriever...")
    embedder = Embedder(cfg["models"]["embedder"])
    store = ChromaStore(
        persist_dir=cfg["rag"]["chroma_persist_dir"],
        collection_name=cfg["rag"]["collection_name"],
    )
    retriever = Retriever(embedder, store)
    rag_teacher = RAGTeacher(
        teacher, retriever, max_input_length=cfg["dataset"]["max_input_length"]
    )

    fmt = partial(
        format_for_training,
        tokenizer=teacher.tokenizer,
        max_input_length=cfg["dataset"]["max_input_length"],
        max_target_length=cfg["dataset"]["max_target_length"],
    )
    formatted = dataset.map(fmt, remove_columns=dataset.column_names)
    formatted.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    loader = DataLoader(
        formatted,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    print(f"Generating soft labels → {output_dir}")
    skipped = 0
    for batch_idx, batch in enumerate(tqdm(loader)):
        example_ids = batch["example_id"]
        labels = batch["labels"]
        questions = batch["question_text"]

        # Build decoder inputs
        pad_id = teacher.tokenizer.pad_token_id
        dec = labels.clone()
        dec[dec == -100] = pad_id
        bos = torch.full((labels.size(0), 1), pad_id, dtype=torch.long)
        decoder_input_ids = torch.cat([bos, dec[:, :-1]], dim=1)

        rag_logits, bare_logits, neg_logits = rag_teacher.get_all_logits(questions, decoder_input_ids)

        for i, eid in enumerate(example_ids):
            out_path = output_dir / f"{eid}.npz"
            if out_path.exists():
                skipped += 1
                continue
            np.savez_compressed(
                str(out_path),
                rag_logits=rag_logits[i].cpu().float().numpy(),
                bare_logits=bare_logits[i].cpu().float().numpy(),
                neg_logits=neg_logits[i].cpu().float().numpy(),
            )

    total = len(dataset)
    print(f"\nDone. Generated {total - skipped} files, skipped {skipped} already-existing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/distillation_config.yaml")
    args = parser.parse_args()
    main(args.config)
