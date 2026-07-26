"""
Phase 2: Pre-generate teacher soft labels and save to disk as .npz files.

This decouples the expensive teacher inference from student training,
allowing training to restart cheaply without re-running the teacher.

Usage:
    python scripts/generate_soft_labels.py [--config ...] [--soft-labels-dir ...] [--chroma-dir ...]

Output: {soft_labels_dir}/{example_id}.npz with keys: rag_logits, bare_logits, neg_logits

Colab example:
    python scripts/generate_soft_labels.py \\
        --soft-labels-dir /content/drive/MyDrive/rad/soft_labels \\
        --chroma-dir /content/drive/MyDrive/rad/chroma_db
"""

import argparse
import json
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
from src.distillation.gating import retrieval_utility


def main(config_path: str, soft_labels_dir: str | None, chroma_dir: str | None, device_map: str | None) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if soft_labels_dir:
        cfg["soft_labels"]["output_dir"] = soft_labels_dir
    if chroma_dir:
        cfg["rag"]["chroma_persist_dir"] = chroma_dir

    output_dir = Path(cfg["soft_labels"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    dataset = load_squad(cfg["dataset"]["train_split"])
    dataset = filter_answerable(dataset)
    print(f"{len(dataset)} training examples.")

    print("Loading teacher model...")
    teacher = TeacherModel(cfg["models"]["teacher"], device_map=device_map or None)

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
    formatted.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
        # Keep question_text / answer_text / example_id as plain Python strings —
        # without this datasets drops them and the RAG teacher loses its queries.
        output_all_columns=True,
    )

    loader = DataLoader(
        formatted,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    print(f"Generating soft labels -> {output_dir}")
    skipped = 0
    utilities: dict[str, float] = {}
    for batch_idx, batch in enumerate(tqdm(loader)):
        example_ids = batch["example_id"]
        labels = batch["labels"]
        questions = batch["question_text"]

        pad_id = teacher.tokenizer.pad_token_id
        dec = labels.clone()
        dec[dec == -100] = pad_id
        bos = torch.full((labels.size(0), 1), pad_id, dtype=torch.long)
        decoder_input_ids = torch.cat([bos, dec[:, :-1]], dim=1)

        rag_logits, bare_logits, neg_logits = rag_teacher.get_all_logits(questions, decoder_input_ids)

        # Retrieval utility per example — free here, since both logit sets are
        # already in memory. Drives the curriculum and is the headline diagnostic
        # for whether the retriever is pulling its weight at all.
        utility = retrieval_utility(
            rag_logits.float().cpu(),
            bare_logits.float().cpu(),
            labels,
        )

        for i, eid in enumerate(example_ids):
            utilities[eid] = float(utility[i])
            out_path = output_dir / f"{eid}.npz"
            if out_path.exists():
                skipped += 1
                continue
            np.savez_compressed(
                str(out_path),
                rag_logits=rag_logits[i].cpu().float().numpy(),
                bare_logits=bare_logits[i].cpu().float().numpy(),
                neg_logits=neg_logits[i].cpu().float().numpy(),
                utility=np.float32(utility[i]),
            )

    utilities_path = output_dir / "utilities.json"
    with open(utilities_path, "w") as f:
        json.dump(utilities, f)

    total = len(dataset)
    print(f"\nDone. Generated {total - skipped} files, skipped {skipped} already-existing.")
    print(f"Soft labels at: {output_dir}")

    u = np.array(list(utilities.values()))
    helped = float((u > 0).mean())
    print(
        f"\nRetrieval utility  mean={u.mean():+.4f}  median={np.median(u):+.4f}\n"
        f"Retrieval helped on {helped:.1%} of examples "
        f"(u > 0 means the retrieved context raised the teacher's likelihood of the gold answer)."
    )
    if helped < 0.5:
        print(
            "  NOTE: retrieval helped on under half the examples. RUG will down-weight\n"
            "  L_RAG accordingly, but consider raising top_k or improving chunking."
        )
    print(f"Utility table at: {utilities_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/distillation_config.yaml")
    parser.add_argument("--soft-labels-dir", default=None, help="Override soft labels output directory")
    parser.add_argument("--chroma-dir", default=None, help="Override ChromaDB directory")
    parser.add_argument("--device-map", default=None, help="HuggingFace device_map (e.g. 'auto' for Kaggle 2xT4)")
    args = parser.parse_args()
    main(args.config, args.soft_labels_dir, args.chroma_dir, args.device_map)
