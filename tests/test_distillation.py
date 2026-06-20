"""
Tests for RADLoss — uses random tensors, no HF models required.
All tests run on CPU in < 5 seconds.
"""

import pytest
import torch
import torch.nn.functional as F

from src.distillation.loss import RADLoss


B, L, V = 4, 16, 32000  # batch, seq_len, vocab_size (T5-small)
IGNORE = -100


def _random_logits(b=B, l=L, v=V):
    return torch.randn(b, l, v)


def _random_labels(b=B, l=L, v=V, pad_frac=0.3):
    labels = torch.randint(0, v, (b, l))
    mask = torch.rand(b, l) < pad_frac
    labels[mask] = IGNORE
    return labels


def test_rad_loss_returns_scalar():
    loss_fn = RADLoss()
    result = loss_fn(
        student_logits=_random_logits(),
        rag_teacher_logits=_random_logits(),
        bare_teacher_logits=_random_logits(),
        neg_teacher_logits=_random_logits(),
        labels=_random_labels(),
    )
    assert result["total"].shape == torch.Size([])


def test_loss_components_present():
    loss_fn = RADLoss()
    result = loss_fn(
        _random_logits(), _random_logits(), _random_logits(), _random_logits(), _random_labels()
    )
    for key in ("total", "L_RAG", "L_KL", "L_CRA", "L_CE"):
        assert key in result, f"Missing key: {key}"


def test_loss_components_non_negative():
    """KL and CE losses must be >= 0; CRA uses clamp(min=0) so also >= 0."""
    loss_fn = RADLoss()
    result = loss_fn(
        _random_logits(), _random_logits(), _random_logits(), _random_logits(), _random_labels()
    )
    for key in ("L_RAG", "L_KL", "L_CRA", "L_CE"):
        assert result[key].item() >= 0, f"{key} is negative: {result[key].item()}"


def test_cra_zero_when_rag_equals_neg():
    """
    When rag_logits == neg_logits, KL(p_T+ || p_T-) = 0.
    L_CRA = max(0, margin - 0) = margin.
    """
    margin = 0.5
    loss_fn = RADLoss(cra_margin=margin)
    same_logits = _random_logits()
    result = loss_fn(
        student_logits=_random_logits(),
        rag_teacher_logits=same_logits,
        bare_teacher_logits=_random_logits(),
        neg_teacher_logits=same_logits,  # identical to rag
        labels=_random_labels(),
    )
    assert abs(result["L_CRA"].item() - margin) < 1e-3


def test_temperature_scaling_weights():
    """With alpha=1, beta=0, gamma=0, delta=0, total should equal L_RAG."""
    loss_fn = RADLoss(alpha=1.0, beta=0.0, gamma=0.0, delta=0.0)
    s, r, b, n = _random_logits(), _random_logits(), _random_logits(), _random_logits()
    labels = _random_labels()
    result = loss_fn(s, r, b, n, labels)
    assert abs(result["total"].item() - result["L_RAG"].item()) < 1e-4


def test_gradient_flows_through_student_only():
    """
    After a backward pass, only student logits should have gradients.
    Teacher logits are detached inputs, so they must have no grad.
    """
    loss_fn = RADLoss()
    student_logits = _random_logits().requires_grad_(True)
    rag_logits = _random_logits()        # no grad
    bare_logits = _random_logits()       # no grad
    neg_logits = _random_logits()        # no grad

    result = loss_fn(student_logits, rag_logits, bare_logits, neg_logits, _random_labels())
    result["total"].backward()

    assert student_logits.grad is not None
    assert rag_logits.grad is None
    assert bare_logits.grad is None
    assert neg_logits.grad is None


def test_loss_changes_with_different_inputs():
    """Sanity: different teacher logits produce different losses."""
    loss_fn = RADLoss()
    s = _random_logits()
    labels = _random_labels()
    result1 = loss_fn(s, _random_logits(v=V), _random_logits(), _random_logits(), labels)
    result2 = loss_fn(s, _random_logits(v=V), _random_logits(), _random_logits(), labels)
    # Extremely unlikely to be identical with random tensors
    assert result1["total"].item() != result2["total"].item()
