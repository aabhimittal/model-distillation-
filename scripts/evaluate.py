"""
Phase 4: Evaluate and compare all model conditions.

Conditions:
  1. Teacher (bare)   — flan-t5-base, no retrieval
  2. Teacher + RAG    — flan-t5-base with ChromaDB retrieval
  3. Student (RAD)    — flan-t5-small trained with full L_RAD

Usage:
    python scripts/evaluate.py --student-rad outputs/student_rad/final [--config ...]

Colab example:
    python scripts/evaluate.py \\
        --student-rad /content/drive/MyDrive/rad/outputs/final \\
        --chroma-dir /content/drive/MyDrive/rad/chroma_db \\
        --output-dir /content/drive/MyDrive/rad/outputs
"""

import argparse
import json
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
from src.student.student_model import StudentModel
from src.evaluation.evaluator import Evaluator
from src.evaluation.probe import KnowledgeRetentionProbe


def generate_predictions(model, tokenizer, loader, device: str, use_rag_retriever=None) -> tuple[list, list]:
    model.eval()
    preds, refs = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            if use_rag_retriever:
                questions = batch["question_text"]
                contexts = use_rag_retriever.retrieve_batch(questions, top_k=3)
                rag_prompts = [
                    f"Context 1: {c[0]}\nContext 2: {c[1] if len(c) > 1 else ''}\nQuestion: {q}\nAnswer:"
                    for q, c in zip(questions, contexts)
                ]
                enc = tokenizer(
                    rag_prompts, max_length=512, truncation=True,
                    padding="max_length", return_tensors="pt"
                ).to(device)
                out = model.generate(**enc, max_new_tokens=64)
            else:
                out = model.generate(
                    input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=64
                )
            decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
            preds.extend(decoded)
            refs.extend(batch["answer_text"])
    return preds, refs


def main(
    config_path: str,
    student_rad_path: str | None,
    output_dir: str | None,
    chroma_dir: str | None,
) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if output_dir:
        cfg["training"]["output_dir"] = output_dir
    if chroma_dir:
        cfg["rag"]["chroma_persist_dir"] = chroma_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = Evaluator()

    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Loading validation dataset...")
    val_ds = load_squad(cfg["dataset"]["val_split"])
    val_ds = filter_answerable(val_ds)
    print(f"{len(val_ds)} validation examples.")

    print("Loading teacher...")
    teacher = TeacherModel(cfg["models"]["teacher"], device=device)

    fmt = partial(
        format_for_training,
        tokenizer=teacher.tokenizer,
        max_input_length=cfg["dataset"]["max_input_length"],
        max_target_length=cfg["dataset"]["max_target_length"],
    )
    val_formatted = val_ds.map(fmt, remove_columns=val_ds.column_names)
    val_formatted.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
        # Keep question_text / answer_text / example_id as plain Python strings —
        # without this datasets drops them and the RAG teacher loses its queries.
        output_all_columns=True,
    )
    val_loader = DataLoader(
        val_formatted, batch_size=16, shuffle=False, collate_fn=collate_fn
    )

    print("Setting up RAG retriever...")
    embedder = Embedder(cfg["models"]["embedder"])
    store = ChromaStore(
        persist_dir=cfg["rag"]["chroma_persist_dir"],
        collection_name=cfg["rag"]["collection_name"],
    )
    retriever = Retriever(embedder, store)

    all_results = {}

    print("\n[1/3] Evaluating bare teacher...")
    bare_teacher_preds, refs = generate_predictions(
        teacher.model, teacher.tokenizer, val_loader, device
    )
    all_results["Teacher (bare)"] = evaluator.evaluate(bare_teacher_preds, refs)

    print("[2/3] Evaluating RAG teacher...")
    rag_teacher_preds, refs = generate_predictions(
        teacher.model, teacher.tokenizer, val_loader, device, use_rag_retriever=retriever
    )
    all_results["Teacher + RAG"] = evaluator.evaluate(rag_teacher_preds, refs)

    probe_result = None
    if student_rad_path and Path(student_rad_path).exists():
        print("[3/3] Evaluating student (RAD)...")
        student = StudentModel(student_rad_path, device=device)
        student_preds, refs = generate_predictions(
            student.model, student.tokenizer, val_loader, device
        )
        all_results["Student (RAD)"] = evaluator.evaluate(student_preds, refs)

        # Knowledge Retention Probe — the measurement that actually tests the RAD
        # thesis. Reuses predictions already generated above, so it costs nothing.
        # The student is also run *with* retrieval to measure how much it still
        # depends on it; if RAD worked, that should add close to nothing.
        print("\nRunning Knowledge Retention Probe...")
        student_rag_preds, _ = generate_predictions(
            student.model, student.tokenizer, val_loader, device,
            use_rag_retriever=retriever,
        )
        probe = KnowledgeRetentionProbe()
        probe_result = probe.evaluate(
            student_preds=student_preds,
            bare_teacher_preds=bare_teacher_preds,
            rag_teacher_preds=rag_teacher_preds,
            references=refs,
            student_with_retrieval_preds=student_rag_preds,
        )
        print("\n" + probe.format_report(probe_result))
    else:
        print("[3/3] Student RAD checkpoint not found — skipping.")

    print("\n" + "=" * 62)
    print("RESULTS")
    print("=" * 62)
    print(evaluator.compare_models(all_results))
    print("=" * 62)

    output_root = Path(cfg["training"]["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    results_path = output_root / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    if probe_result is not None:
        probe_path = output_root / "retention_probe.json"
        with open(probe_path, "w") as f:
            json.dump(probe_result, f, indent=2)
        print(f"Retention probe saved to {probe_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAD model conditions")
    parser.add_argument("--config", default="configs/distillation_config.yaml")
    parser.add_argument("--student-rad", default=None, help="Path to RAD student checkpoint")
    parser.add_argument("--output-dir", default=None, help="Override output directory for results JSON")
    parser.add_argument("--chroma-dir", default=None, help="Override ChromaDB directory")
    args = parser.parse_args()
    main(args.config, args.student_rad, args.output_dir, args.chroma_dir)
