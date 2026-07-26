"""Tests for the retrieval-utility curriculum. Pure Python + numpy, no GPU."""

import json

import numpy as np
import pytest

from src.distillation.curriculum import (
    RetrievalUtilityCurriculum,
    competence,
    load_utilities,
)


def test_competence_starts_at_initial_and_reaches_one():
    assert competence(0, 100, initial=0.25) == pytest.approx(0.25)
    assert competence(100, 100, initial=0.25) == pytest.approx(1.0)
    assert competence(200, 100, initial=0.25) == pytest.approx(1.0)  # clamped


def test_competence_is_monotonic():
    vals = [competence(t, 100, initial=0.2) for t in range(0, 101, 5)]
    assert all(b >= a for a, b in zip(vals, vals[1:])), vals


def test_competence_handles_zero_total_steps():
    assert competence(0, 0) == 1.0


def test_curriculum_reveals_highest_utility_examples_first():
    utilities = [0.0, 5.0, -3.0, 2.0, 9.0]  # ranking: 4, 1, 3, 0, 2
    sampler = RetrievalUtilityCurriculum(utilities, total_steps=100, initial_competence=0.4)

    sampler.set_step(0)
    visible = set(sampler)

    # 40% of 5 examples = 2, and those must be the two highest-utility ones
    assert visible == {4, 1}, f"expected the top-utility pair, got {visible}"


def test_curriculum_widens_to_full_dataset():
    utilities = [0.0, 5.0, -3.0, 2.0, 9.0]
    sampler = RetrievalUtilityCurriculum(utilities, total_steps=100, initial_competence=0.4)

    sampler.set_step(100)

    assert set(sampler) == set(range(5))
    assert len(sampler) == 5


def test_curriculum_length_tracks_competence():
    sampler = RetrievalUtilityCurriculum(
        np.arange(100, dtype=float), total_steps=50, initial_competence=0.1
    )

    sampler.set_step(0)
    early = len(sampler)
    sampler.set_step(50)
    late = len(sampler)

    assert early == 10
    assert late == 100
    assert early < late


def test_curriculum_never_yields_empty_pool():
    sampler = RetrievalUtilityCurriculum([1.0, 2.0], total_steps=10, initial_competence=0.0)
    sampler.set_step(0)

    assert len(list(sampler)) >= 1


def test_curriculum_shuffles_within_the_competent_pool():
    """Order inside the visible prefix must vary, or batches correlate with utility."""
    utilities = list(range(50))
    a = RetrievalUtilityCurriculum(utilities, total_steps=10, initial_competence=1.0, seed=1)
    b = RetrievalUtilityCurriculum(utilities, total_steps=10, initial_competence=1.0, seed=2)

    assert list(a) != list(b)
    assert sorted(a) == sorted(b) == list(range(50))


def test_curriculum_rejects_non_1d_utilities():
    with pytest.raises(ValueError, match="1-D"):
        RetrievalUtilityCurriculum([[1.0, 2.0], [3.0, 4.0]], total_steps=10)


def test_load_utilities_aligns_to_example_order(tmp_path):
    path = tmp_path / "utilities.json"
    path.write_text(json.dumps({"a": 1.5, "b": -0.5, "c": 3.0}))

    out = load_utilities(path, ["c", "a", "b"])

    assert np.allclose(out, [3.0, 1.5, -0.5])


def test_load_utilities_defaults_missing_ids_to_neutral(tmp_path):
    """A partial soft-label directory should degrade gracefully, not crash."""
    path = tmp_path / "utilities.json"
    path.write_text(json.dumps({"a": 2.0}))

    out = load_utilities(path, ["a", "missing"])

    assert np.allclose(out, [2.0, 0.0])
