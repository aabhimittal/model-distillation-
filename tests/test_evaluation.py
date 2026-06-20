"""Tests for evaluation metrics — pure Python, no models."""

import pytest

from src.evaluation.evaluator import exact_match, token_f1, bleu4, Evaluator


def test_exact_match_identical():
    assert exact_match("New York", "New York") == 1.0


def test_exact_match_normalisation():
    assert exact_match("The U.S.A.", "the usa") == 1.0


def test_exact_match_different():
    assert exact_match("Paris", "London") == 0.0


def test_f1_identical():
    assert token_f1("New York City", "New York City") == pytest.approx(1.0)


def test_f1_partial_credit():
    # "New York" vs "New York City" — 2 common tokens out of 2 and 3
    f1 = token_f1("New York", "New York City")
    assert 0.5 < f1 < 1.0


def test_f1_empty_prediction():
    assert token_f1("", "New York") == 0.0


def test_bleu4_identical():
    score = bleu4("the cat sat on the mat", "the cat sat on the mat")
    assert score > 0.9


def test_bleu4_empty_prediction():
    score = bleu4("", "something")
    assert score == 0.0


def test_evaluator_compare_table_has_all_columns():
    evaluator = Evaluator()
    results = {
        "Teacher (bare)": {"exact_match": 0.45, "f1": 0.60, "bleu4": 0.30},
        "Student (RAD)": {"exact_match": 0.42, "f1": 0.57, "bleu4": 0.27},
    }
    table = evaluator.compare_models(results)
    assert "EM" in table
    assert "F1" in table
    assert "BLEU" in table
    assert "Teacher (bare)" in table
    assert "Student (RAD)" in table


def test_evaluator_evaluate_batch():
    evaluator = Evaluator()
    preds = ["Paris", "Germany", "42"]
    refs = ["Paris", "France", "42"]
    metrics = evaluator.evaluate(preds, refs)
    assert metrics["exact_match"] == pytest.approx(2 / 3)
    assert 0 < metrics["f1"] <= 1.0
    assert "bleu4" in metrics
    assert metrics["n"] == 3
