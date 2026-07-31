"""Typed configuration for the fine-tuning + distillation track.

Loaded from ``configs/finetune_config.yaml``. Uses only the stdlib + PyYAML so it
can be imported (and unit-tested) without torch/transformers present.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ModelConfig:
    # Small open-source student that fits a free-tier T4 in 4-bit + LoRA.
    student: str = "Qwen/Qwen2.5-0.5B-Instruct"
    # Larger open-source teacher whose behaviour we distil into the student.
    teacher: str = "Qwen/Qwen2.5-3B-Instruct"
    max_seq_length: int = 1024


@dataclass
class DataConfig:
    # Any HF instruction dataset; column names are remapped via *_key below.
    name: str = "gbharti/finance-alpaca"
    split: str = "train"
    max_train_samples: int = 2000
    max_eval_samples: int = 200
    instruction_key: str = "instruction"
    input_key: str = "input"
    output_key: str = "output"
    seed: int = 42


@dataclass
class TeacherConfig:
    # "hf": load the teacher locally with transformers.
    # "nim": call an NVIDIA NIM / build.nvidia.com OpenAI-compatible endpoint.
    # "passthrough": skip teacher generation and distil against the dataset's gold
    #                answers (pure fine-tuning, useful as a baseline / offline run).
    provider: str = "passthrough"
    # NVIDIA NIM settings (provider == "nim"). The API key is read from the
    # NVIDIA_API_KEY environment variable — never hard-code it here.
    nim_base_url: str = "https://integrate.api.nvidia.com/v1"
    nim_model: str = "meta/llama-3.1-8b-instruct"
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class CurationConfig:
    """Quality gate on teacher outputs + dataset hygiene (see curate.py, dedup.py)."""

    enabled: bool = True
    drop_refusals: bool = True
    drop_truncated: bool = True
    max_ngram_repeat_ratio: float = 0.55
    min_chars: int = 8
    # Near-duplicate Jaccard threshold; 1.0 degrades to exact-match dedup only.
    dedup_threshold: float = 0.85
    dedup: bool = True
    # Remove train rows that leak into the eval split (measurement integrity).
    decontaminate: bool = True


@dataclass
class RobustnessConfig:
    """Retry / rate-limit / resume behaviour for remote teacher generation."""

    max_attempts: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    # Requests per second against the teacher endpoint (0 disables limiting).
    requests_per_second: float = 4.0
    # JSONL resume log; empty string disables checkpointing.
    checkpoint_path: str = "./outputs/teacher_cache.jsonl"


@dataclass
class LoraConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    # 4-bit QLoRA quantisation of the frozen base — the key to fitting a T4.
    load_in_4bit: bool = True


@dataclass
class TrainConfig:
    output_dir: str = "./outputs/student_finetuned"
    batch_size: int = 2
    gradient_accumulation_steps: int = 8  # effective batch = 16
    learning_rate: float = 2.0e-4
    num_epochs: int = 1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    logging_steps: int = 10
    save_steps: int = 200
    fp16: bool = False
    bf16: bool = False
    max_grad_norm: float = 0.3
    seed: int = 42


@dataclass
class FinetuneConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    curation: CurationConfig = field(default_factory=CurationConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @staticmethod
    def _section(cls, raw: Optional[Dict[str, Any]]):
        raw = raw or {}
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "FinetuneConfig":
        raw = raw or {}
        return cls(
            model=cls._section(ModelConfig, raw.get("model")),
            data=cls._section(DataConfig, raw.get("data")),
            teacher=cls._section(TeacherConfig, raw.get("teacher")),
            curation=cls._section(CurationConfig, raw.get("curation")),
            robustness=cls._section(RobustnessConfig, raw.get("robustness")),
            lora=cls._section(LoraConfig, raw.get("lora")),
            train=cls._section(TrainConfig, raw.get("train")),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "FinetuneConfig":
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.from_dict(raw)
