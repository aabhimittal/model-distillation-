"""CPU-safe unit tests for the fine-tuning + distillation track.

These exercise the tokenizer-agnostic core (config parsing, instruction
formatting, supervised label masking, provider selection, distillation record
assembly) without importing torch / transformers / peft, so they run in the CI
job in seconds with no GPU.
"""

from __future__ import annotations

import os

import pytest

from src.finetune import (
    IGNORE_INDEX,
    FinetuneConfig,
    build_prompt,
    build_supervised_labels,
    has_trainable_labels,
    make_distillation_records,
    normalize_record,
    resolve_provider,
    to_messages,
)
from src.finetune.distill import PassthroughTeacher

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
    "finetune_config.yaml",
)


# --- config -------------------------------------------------------------------

def test_config_defaults():
    cfg = FinetuneConfig()
    assert cfg.model.student
    assert cfg.teacher.provider in ("passthrough", "hf", "nim")
    assert cfg.lora.r > 0
    assert cfg.train.gradient_accumulation_steps >= 1


def test_config_from_yaml_roundtrip():
    cfg = FinetuneConfig.from_yaml(CONFIG_PATH)
    assert cfg.model.student == "Qwen/Qwen2.5-0.5B-Instruct"
    assert cfg.data.name
    assert isinstance(cfg.lora.target_modules, list) and cfg.lora.target_modules


def test_config_ignores_unknown_keys():
    cfg = FinetuneConfig.from_dict({"model": {"student": "x", "bogus": 1}})
    assert cfg.model.student == "x"


# --- record normalisation -----------------------------------------------------

def test_normalize_record_alpaca():
    r = normalize_record({"instruction": " hi ", "input": "", "output": " yo "})
    assert r == {"instruction": "hi", "input": "", "output": "yo"}


def test_normalize_record_column_remap():
    raw = {"question": "What is 2+2?", "answer": "4"}
    r = normalize_record(raw, instruction_key="question", output_key="answer")
    assert r["instruction"] == "What is 2+2?"
    assert r["output"] == "4"
    assert r["input"] == ""


def test_normalize_record_handles_missing_and_none():
    r = normalize_record({"instruction": None})
    assert r == {"instruction": "", "input": "", "output": ""}


# --- prompt / chat formatting -------------------------------------------------

def test_build_prompt_variants():
    with_in = build_prompt("Summarise", "long text")
    no_in = build_prompt("Summarise")
    assert "### Input:" in with_in and "long text" in with_in
    assert "### Input:" not in no_in
    assert both_end_in_response(with_in, no_in)


def both_end_in_response(*prompts):
    return all(p.rstrip().endswith("### Response:") for p in prompts)


def test_to_messages_merges_input():
    msgs = to_messages("Classify", "the sentiment")
    assert msgs[0]["role"] == "user"
    assert "Classify" in msgs[0]["content"] and "the sentiment" in msgs[0]["content"]


# --- supervised label masking -------------------------------------------------

def test_labels_mask_prompt_only():
    prompt = [1, 2, 3]
    response = [4, 5]
    ex = build_supervised_labels(prompt, response, eos_id=9)
    assert ex["input_ids"] == [1, 2, 3, 4, 5, 9]
    assert ex["labels"] == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 4, 5, 9]
    assert has_trainable_labels(ex["labels"])


def test_labels_eos_not_duplicated():
    ex = build_supervised_labels([1], [4, 9], eos_id=9)
    assert ex["input_ids"] == [1, 4, 9]  # existing EOS kept, not doubled


def test_labels_truncation():
    ex = build_supervised_labels([1, 2, 3], [4, 5, 6, 7], eos_id=9, max_length=4)
    assert len(ex["input_ids"]) == 4
    assert len(ex["labels"]) == 4


def test_all_masked_labels_flagged():
    assert not has_trainable_labels([IGNORE_INDEX, IGNORE_INDEX])


# --- teacher provider selection ----------------------------------------------

def test_resolve_provider_normalises():
    from src.finetune.config import TeacherConfig

    assert resolve_provider(TeacherConfig(provider="PASSTHROUGH")) == "passthrough"
    assert resolve_provider(TeacherConfig(provider="hf")) == "hf"


def test_resolve_provider_rejects_unknown():
    from src.finetune.config import TeacherConfig

    with pytest.raises(ValueError):
        resolve_provider(TeacherConfig(provider="gpt5"))


def test_passthrough_teacher_returns_gold():
    records = [{"instruction": "q", "input": "", "output": "gold"}]
    assert PassthroughTeacher().generate(records) == ["gold"]


# --- distillation record assembly --------------------------------------------

def test_make_distillation_records_uses_teacher_output():
    records = [{"instruction": "q", "input": "", "output": "gold"}]
    out = make_distillation_records(records, ["teacher says"])
    assert out[0]["output"] == "teacher says"
    assert out[0]["instruction"] == "q"


def test_make_distillation_records_falls_back_on_empty():
    records = [{"instruction": "q", "input": "", "output": "gold"}]
    out = make_distillation_records(records, ["   "])
    assert out[0]["output"] == "gold"  # empty teacher output -> keep gold


def test_make_distillation_records_length_mismatch():
    with pytest.raises(ValueError):
        make_distillation_records([{"output": "a"}], [])
