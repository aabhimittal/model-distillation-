"""Load and prepare SQuAD v2 for RAG-Augmented Distillation."""

from __future__ import annotations

from typing import List, Dict, Any

from datasets import load_dataset, Dataset


def load_squad(split: str, max_examples: int | None = None) -> Dataset:
    """Load a SQuAD v2 split, optionally capped at max_examples."""
    slice_str = f"{split}[:{max_examples}]" if max_examples else split
    dataset = load_dataset("rajpurkar/squad_v2", split=slice_str)
    return dataset


def filter_answerable(dataset: Dataset) -> Dataset:
    """Remove unanswerable questions (SQuAD v2 has ~40%)."""
    return dataset.filter(lambda ex: len(ex["answers"]["text"]) > 0)


def extract_contexts(dataset: Dataset) -> List[str]:
    """Return deduplicated passage texts for building the ChromaDB corpus."""
    seen = set()
    contexts = []
    for ex in dataset:
        ctx = ex["context"]
        if ctx not in seen:
            seen.add(ctx)
            contexts.append(ctx)
    return contexts


def format_for_training(
    example: Dict[str, Any], tokenizer, max_input_length: int, max_target_length: int
) -> Dict[str, Any]:
    """
    Prepare a single SQuAD example for the student model.

    IMPORTANT: the student input is 'question: {q}' only — no context.
    The student must learn the answer from teacher soft labels, not context leakage.
    The question_text field is kept separately for RAG retrieval in the teacher.
    """
    question = example["question"]
    answer = example["answers"]["text"][0] if example["answers"]["text"] else ""

    student_input = f"question: {question}"
    inputs = tokenizer(
        student_input,
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    targets = tokenizer(
        answer,
        max_length=max_target_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    labels = targets["input_ids"].squeeze()
    # Replace padding token id with -100 so loss ignores it
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "input_ids": inputs["input_ids"].squeeze(),
        "attention_mask": inputs["attention_mask"].squeeze(),
        "labels": labels,
        "question_text": question,
        "answer_text": answer,
        "example_id": example["id"],
    }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Default collate for DataLoader (tensors already padded)."""
    import torch

    result: Dict[str, Any] = {}
    for key in batch[0]:
        if hasattr(batch[0][key], "shape"):
            result[key] = torch.stack([item[key] for item in batch])
        else:
            result[key] = [item[key] for item in batch]
    return result
