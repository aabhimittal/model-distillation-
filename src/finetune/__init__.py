"""Fine-tuning + sequence-level distillation of a small open-source LLM.

Companion track to the RAG-Augmented Distillation (RAD) module: where RAD does
logit-level distillation of a seq2seq model, this track does parameter-efficient
(QLoRA) fine-tuning of a small decoder-only LLM against a larger teacher's
generated responses — the recipe that runs on a free-tier Colab T4.

Only the import-light pieces (config + chat/label helpers) are eagerly exported so
the package can be imported without torch/transformers/peft installed.
"""

from .chat import (
    IGNORE_INDEX,
    build_prompt,
    build_supervised_labels,
    has_trainable_labels,
    normalize_record,
    to_messages,
)
from .budget import (
    bucket_by_length,
    estimate_teacher_cost,
    estimate_tokens,
    length_stats,
    padding_waste,
    record_lengths,
    sequential_batches,
)
from .config import (
    CurationConfig,
    DataConfig,
    FinetuneConfig,
    LoraConfig,
    ModelConfig,
    RobustnessConfig,
    TeacherConfig,
    TrainConfig,
)
from .curate import (
    CurationReport,
    CurationThresholds,
    classify_record,
    curate_records,
    is_degenerate,
    is_refusal,
    leaks_prompt,
    looks_truncated,
)
from .dedup import decontaminate, exact_dedup, jaccard, near_dedup, shingles
from .distill import (
    VALID_PROVIDERS,
    make_distillation_records,
    resolve_provider,
)
from .robust import (
    JsonlCheckpoint,
    RateLimiter,
    RetryPolicy,
    merge_checkpoint,
    pending_indices,
    retry_call,
)

__all__ = [
    "IGNORE_INDEX",
    "build_prompt",
    "build_supervised_labels",
    "has_trainable_labels",
    "normalize_record",
    "to_messages",
    "CurationConfig",
    "DataConfig",
    "FinetuneConfig",
    "LoraConfig",
    "ModelConfig",
    "RobustnessConfig",
    "TeacherConfig",
    "TrainConfig",
    "VALID_PROVIDERS",
    "make_distillation_records",
    "resolve_provider",
    # curation
    "CurationReport",
    "CurationThresholds",
    "classify_record",
    "curate_records",
    "is_degenerate",
    "is_refusal",
    "leaks_prompt",
    "looks_truncated",
    # dedup / decontamination
    "decontaminate",
    "exact_dedup",
    "jaccard",
    "near_dedup",
    "shingles",
    # budgeting
    "bucket_by_length",
    "estimate_teacher_cost",
    "estimate_tokens",
    "length_stats",
    "padding_waste",
    "record_lengths",
    "sequential_batches",
    # robustness
    "JsonlCheckpoint",
    "RateLimiter",
    "RetryPolicy",
    "merge_checkpoint",
    "pending_indices",
    "retry_call",
]
