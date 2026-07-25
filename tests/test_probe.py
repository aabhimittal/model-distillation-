"""Tests for the Knowledge Retention Probe and calibration metrics. No GPU."""

import numpy as np
import pytest

from src.evaluation.calibration import expected_calibration_error
from src.evaluation.probe import (
    HARD,
    PARAMETRIC,
    RETRIEVAL_DEPENDENT,
    KnowledgeRetentionProbe,
)


@pytest.fixture
def probe():
    return KnowledgeRetentionProbe()


# Four examples, one per stratum pattern:
#   0: bare wrong, rag right   -> retrieval_dependent
#   1: bare right              -> parametric
#   2: both wrong              -> hard
#   3: bare wrong, rag right   -> retrieval_dependent
REFS = ["paris", "rome", "tokyo", "cairo"]
BARE = ["lyon", "rome", "osaka", "giza"]
RAG = ["paris", "rome", "kyoto", "cairo"]


def test_stratify_assigns_each_example_to_one_stratum(probe):
    strata = probe.stratify(BARE, RAG, REFS)

    assert strata[RETRIEVAL_DEPENDENT] == [0, 3]
    assert strata[PARAMETRIC] == [1]
    assert strata[HARD] == [2]

    total = sum(len(v) for v in strata.values())
    assert total == len(REFS), "strata must partition the eval set exactly"


def test_stratify_rejects_mismatched_lengths(probe):
    with pytest.raises(ValueError, match="same length"):
        probe.stratify(BARE[:2], RAG, REFS)


def test_retention_rate_is_one_when_student_learned_everything(probe):
    student = ["paris", "rome", "tokyo", "cairo"]  # correct on both RD examples

    result = probe.evaluate(student, BARE, RAG, REFS)

    assert result["retention_rate"] == pytest.approx(1.0)


def test_retention_rate_is_zero_when_student_learned_nothing(probe):
    # Wrong on both retrieval-dependent examples (0 and 3), right elsewhere
    student = ["lyon", "rome", "tokyo", "giza"]

    result = probe.evaluate(student, BARE, RAG, REFS)

    assert result["retention_rate"] == pytest.approx(0.0)


def test_retention_rate_is_partial_credit(probe):
    student = ["paris", "rome", "tokyo", "giza"]  # right on 0, wrong on 3

    result = probe.evaluate(student, BARE, RAG, REFS)

    assert result["retention_rate"] == pytest.approx(0.5)


def test_parametric_preservation_detects_catastrophic_forgetting(probe):
    """A student that lost knowledge the teacher already had should score 0 here."""
    student = ["paris", "milan", "tokyo", "cairo"]  # 'rome' -> 'milan' is forgetting

    result = probe.evaluate(student, BARE, RAG, REFS)

    assert result["parametric_preservation"] == pytest.approx(0.0)
    # but retention is untouched — the two rates are independent
    assert result["retention_rate"] == pytest.approx(1.0)


def test_retention_is_not_diluted_by_the_parametric_majority(probe):
    """
    The reason the probe exists: aggregate EM hides retrieval failure when most
    of the eval set is answerable without retrieval.
    """
    refs = ["a"] * 9 + ["z"]
    bare = ["a"] * 9 + ["wrong"]        # 9 parametric, 1 retrieval-dependent
    rag = ["a"] * 9 + ["z"]
    student = ["a"] * 9 + ["wrong"]     # learned nothing retrieval-dependent

    result = KnowledgeRetentionProbe().evaluate(student, bare, rag, refs)
    aggregate_em = np.mean([p == r for p, r in zip(student, refs)])

    assert aggregate_em == pytest.approx(0.9)      # looks great
    assert result["retention_rate"] == pytest.approx(0.0)  # actually learned nothing


def test_retrieval_independence_gap_is_zero_when_student_is_independent(probe):
    student = ["paris", "rome", "tokyo", "cairo"]

    result = probe.evaluate(student, BARE, RAG, REFS, student_with_retrieval_preds=student)

    assert result["retrieval_independence_gap"] == pytest.approx(0.0)


def test_retrieval_independence_gap_positive_when_student_still_needs_retrieval(probe):
    student_alone = ["lyon", "rome", "tokyo", "giza"]
    student_with_rag = ["paris", "rome", "tokyo", "cairo"]

    result = probe.evaluate(
        student_alone, BARE, RAG, REFS, student_with_retrieval_preds=student_with_rag
    )

    assert result["retrieval_independence_gap"] > 0


def test_evaluate_omits_gap_when_no_retrieval_predictions_given(probe):
    result = probe.evaluate(["paris", "rome", "tokyo", "cairo"], BARE, RAG, REFS)

    assert "retrieval_independence_gap" not in result


def test_empty_stratum_scores_zero_without_dividing_by_zero(probe):
    # Every example is parametric — retrieval_dependent stratum is empty
    refs = ["a", "b"]
    result = probe.evaluate(refs, refs, refs, refs)

    assert result["retention_rate"] == 0.0
    assert result["per_stratum"][RETRIEVAL_DEPENDENT]["n"] == 0


def test_format_report_includes_headline_numbers(probe):
    result = probe.evaluate(["paris", "rome", "tokyo", "cairo"], BARE, RAG, REFS)

    report = probe.format_report(result)

    assert "Retention Rate" in report
    assert "Parametric Preservation" in report
    assert RETRIEVAL_DEPENDENT in report


# --- Calibration ----------------------------------------------------------


def test_ece_is_zero_for_a_perfectly_calibrated_model():
    # Confidence exactly matches accuracy within each bin
    conf = [0.95] * 100 + [0.05] * 100
    corr = [1.0] * 95 + [0.0] * 5 + [1.0] * 5 + [0.0] * 95

    result = expected_calibration_error(conf, corr, n_bins=10)

    assert result["ece"] == pytest.approx(0.0, abs=0.01)


def test_ece_detects_overconfidence():
    # Claims 0.95 confidence but is right only half the time
    conf = [0.95] * 100
    corr = [1.0] * 50 + [0.0] * 50

    result = expected_calibration_error(conf, corr, n_bins=10)

    assert result["ece"] == pytest.approx(0.45, abs=0.01)
    assert result["overconfidence"] > 0


def test_ece_reports_signed_underconfidence():
    conf = [0.1] * 100
    corr = [1.0] * 90 + [0.0] * 10

    result = expected_calibration_error(conf, corr, n_bins=10)

    assert result["overconfidence"] < 0, "under-confident model should report negative"


def test_ece_handles_empty_input():
    result = expected_calibration_error([], [], n_bins=10)

    assert result["ece"] == 0.0
    assert result["n"] == 0


def test_ece_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="shape mismatch"):
        expected_calibration_error([0.5, 0.6], [1.0])


def test_ece_bins_cover_all_samples():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.0, 1.0, 500)
    corr = (rng.uniform(size=500) < conf).astype(float)

    result = expected_calibration_error(conf, corr, n_bins=10)

    assert sum(b["n"] for b in result["bins"]) == 500
