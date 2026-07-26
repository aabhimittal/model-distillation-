"""Tests for retrieval-aware gating — RUG and TRF. No models, no GPU."""

import pytest
import torch

from src.distillation.gating import (
    RetrievalUtilityGate,
    TokenFusionGate,
    fuse_teacher_distributions,
    per_example_nll,
    retrieval_utility,
)

B, L, V = 4, 6, 32
IGNORE = -100


def _labels(pad_from: int | None = None) -> torch.Tensor:
    labels = torch.randint(0, V, (B, L))
    if pad_from is not None:
        labels[:, pad_from:] = IGNORE
    return labels


def _confident_logits(labels: torch.Tensor, strength: float) -> torch.Tensor:
    """Logits that put `strength` extra mass on the gold token at each position."""
    logits = torch.zeros(B, L, V)
    safe = labels.clamp_min(0)
    logits.scatter_(-1, safe.unsqueeze(-1), strength)
    return logits


def test_per_example_nll_ignores_padding():
    labels = _labels(pad_from=3)
    logits = _confident_logits(labels, strength=10.0)

    nll = per_example_nll(logits, labels, ignore_index=IGNORE)

    assert nll.shape == (B,)
    # Gold tokens dominate at every *valid* position, so NLL should be near zero.
    # If padding leaked in, the random-token positions would inflate it well past 1.
    assert torch.all(nll < 0.1), f"padding leaked into NLL: {nll}"


def test_per_example_nll_all_padding_does_not_divide_by_zero():
    labels = torch.full((B, L), IGNORE)
    logits = torch.randn(B, L, V)

    nll = per_example_nll(logits, labels, ignore_index=IGNORE)

    assert torch.isfinite(nll).all()
    assert torch.allclose(nll, torch.zeros(B))


def test_retrieval_utility_positive_when_rag_teacher_is_better():
    labels = _labels()
    rag = _confident_logits(labels, strength=8.0)   # RAG teacher knows the answer
    bare = torch.zeros(B, L, V)                     # bare teacher is uniform

    u = retrieval_utility(rag, bare, labels)

    assert u.shape == (B,)
    assert torch.all(u > 0), f"retrieval helped but utility was not positive: {u}"


def test_retrieval_utility_negative_when_retrieval_hurts():
    labels = _labels()
    bare = _confident_logits(labels, strength=8.0)
    rag = torch.zeros(B, L, V)  # distractor passages washed out the teacher

    u = retrieval_utility(rag, bare, labels)

    assert torch.all(u < 0), f"retrieval hurt but utility was not negative: {u}"


def test_retrieval_utility_zero_when_teachers_identical():
    labels = _labels()
    logits = torch.randn(B, L, V)

    u = retrieval_utility(logits, logits.clone(), labels)

    assert torch.allclose(u, torch.zeros(B), atol=1e-6)


def test_gate_routes_toward_rag_teacher_when_retrieval_helps():
    labels = _labels()
    rag = _confident_logits(labels, strength=8.0)
    bare = torch.zeros(B, L, V)

    gate, u = RetrievalUtilityGate(gate_temperature=1.0)(rag, bare, labels)

    assert gate.shape == (B,)
    assert torch.all(gate > 0.5), f"gate should favour the RAG teacher, got {gate}"
    assert torch.all(u > 0)


def test_gate_routes_toward_bare_teacher_when_retrieval_hurts():
    labels = _labels()
    bare = _confident_logits(labels, strength=8.0)
    rag = torch.zeros(B, L, V)

    gate, _ = RetrievalUtilityGate(gate_temperature=1.0)(rag, bare, labels)

    assert torch.all(gate < 0.5), f"gate should favour the bare teacher, got {gate}"


def test_gate_respects_floor_and_ceiling():
    labels = _labels()
    rag = _confident_logits(labels, strength=50.0)  # extreme — would saturate to 1.0
    bare = torch.zeros(B, L, V)

    gate, _ = RetrievalUtilityGate(floor=0.2, ceiling=0.8)(rag, bare, labels)

    assert torch.all(gate >= 0.2 - 1e-6)
    assert torch.all(gate <= 0.8 + 1e-6)


def test_gate_is_neutral_at_zero_utility():
    labels = _labels()
    logits = torch.randn(B, L, V)

    gate, _ = RetrievalUtilityGate(floor=0.0, ceiling=1.0)(logits, logits.clone(), labels)

    assert torch.allclose(gate, torch.full((B,), 0.5), atol=1e-5)


def test_gate_rejects_invalid_bounds():
    with pytest.raises(ValueError):
        RetrievalUtilityGate(floor=0.9, ceiling=0.1)


def test_gate_output_is_detached():
    labels = _labels()
    rag = torch.randn(B, L, V, requires_grad=True)
    bare = torch.randn(B, L, V, requires_grad=True)

    gate, u = RetrievalUtilityGate()(rag, bare, labels)

    assert not gate.requires_grad, "gate is a weighting, it must not carry gradient"
    assert not u.requires_grad


def test_fusion_gate_favours_the_sharper_teacher():
    # RAG teacher is peaked (low entropy), bare teacher is flat (high entropy)
    rag = torch.zeros(B, L, V)
    rag[..., 0] = 1.0
    bare = torch.full((B, L, V), 1.0 / V)

    # center=False so we read the raw confidence comparison
    w = TokenFusionGate(center=False)(rag, bare)

    assert w.shape == (B, L)
    assert torch.all(w > 0.5), f"should lean on the sharper RAG teacher, got {w}"


def test_fusion_gate_centering_removes_uniform_confidence_bias():
    """
    A context-conditioned teacher is lower-entropy almost everywhere just from
    having more input. Centering must strip that constant offset so the gate
    reflects per-token deviation rather than a global shift.
    """
    torch.manual_seed(0)
    bare = torch.softmax(torch.randn(B, L, V), dim=-1)
    # Same distribution, uniformly sharpened — a pure constant entropy shift
    rag = torch.softmax(torch.log(bare) * 2.0, dim=-1)

    w_raw = TokenFusionGate(center=False)(rag, bare)
    w_centered = TokenFusionGate(center=True)(rag, bare)

    assert w_raw.mean() > 0.7, "uncentered gate should be biased toward RAG"
    assert abs(w_centered.mean().item() - 0.5) < 0.1, (
        f"centered gate should sit near neutral, got {w_centered.mean():.3f}"
    )


def test_fused_distribution_is_a_valid_distribution():
    rag = torch.softmax(torch.randn(B, L, V), dim=-1)
    bare = torch.softmax(torch.randn(B, L, V), dim=-1)
    w = torch.rand(B, L)

    fused = fuse_teacher_distributions(rag, bare, w)

    assert fused.shape == (B, L, V)
    assert torch.allclose(fused.sum(-1), torch.ones(B, L), atol=1e-5)
    assert torch.all(fused >= 0)


def test_fusion_weight_extremes_recover_each_teacher():
    rag = torch.softmax(torch.randn(B, L, V), dim=-1)
    bare = torch.softmax(torch.randn(B, L, V), dim=-1)

    all_rag = fuse_teacher_distributions(rag, bare, torch.ones(B, L))
    all_bare = fuse_teacher_distributions(rag, bare, torch.zeros(B, L))

    assert torch.allclose(all_rag, rag, atol=1e-6)
    assert torch.allclose(all_bare, bare, atol=1e-6)
