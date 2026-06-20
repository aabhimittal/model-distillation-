"""Evaluation metrics and model comparison for RAD."""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import List, Dict, Any

import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def _normalize(text: str) -> str:
    """
    Lowercase, strip punctuation, then strip articles — standard SQuAD order.
    Punctuation must come first so 'U.S.A.' → 'usa' before article removal,
    otherwise the isolated 'a' in 'u.s.a.' matches the article regex.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(_normalize(prediction) == _normalize(ground_truth))


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gt_tokens = _normalize(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def bleu4(prediction: str, ground_truth: str) -> float:
    smoother = SmoothingFunction().method1
    ref = [_normalize(ground_truth).split()]
    hyp = _normalize(prediction).split()
    if not hyp:
        return 0.0
    return sentence_bleu(ref, hyp, smoothing_function=smoother)


class Evaluator:
    def __init__(self):
        pass

    def evaluate(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        em_scores, f1_scores, bleu_scores = [], [], []
        for pred, ref in zip(predictions, references):
            em_scores.append(exact_match(pred, ref))
            f1_scores.append(token_f1(pred, ref))
            bleu_scores.append(bleu4(pred, ref))
        return {
            "exact_match": float(np.mean(em_scores)),
            "f1": float(np.mean(f1_scores)),
            "bleu4": float(np.mean(bleu_scores)),
            "n": len(predictions),
        }

    @torch.no_grad()
    def compute_perplexity(self, model, tokenizer, texts: List[str], device: str = "cpu") -> float:
        """Compute mean perplexity over a list of target strings."""
        model.eval()
        total_nll = 0.0
        total_tokens = 0
        for text in texts:
            enc = tokenizer(text, return_tensors="pt").to(device)
            input_ids = enc["input_ids"]
            outputs = model(**enc, labels=input_ids)
            nll = outputs.loss.item()
            n_tok = input_ids.numel()
            total_nll += nll * n_tok
            total_tokens += n_tok
        return float(np.exp(total_nll / max(total_tokens, 1)))

    def compare_models(
        self,
        results: Dict[str, Dict[str, float]],
    ) -> str:
        """Format a comparison table from a dict of {model_name: metrics_dict}."""
        header = f"{'Model':<30} {'EM':>8} {'F1':>8} {'BLEU-4':>8}"
        sep = "-" * len(header)
        rows = [header, sep]
        for name, metrics in results.items():
            rows.append(
                f"{name:<30} {metrics.get('exact_match', 0)*100:>7.1f}%"
                f" {metrics.get('f1', 0)*100:>7.1f}%"
                f" {metrics.get('bleu4', 0)*100:>7.1f}%"
            )
        return "\n".join(rows)
