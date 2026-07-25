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
from .config import (
    DataConfig,
    FinetuneConfig,
    LoraConfig,
    ModelConfig,
    TeacherConfig,
    TrainConfig,
)
from .distill import (
    VALID_PROVIDERS,
    make_distillation_records,
    resolve_provider,
)

__all__ = [
    "IGNORE_INDEX",
    "build_prompt",
    "build_supervised_labels",
    "has_trainable_labels",
    "normalize_record",
    "to_messages",
    "DataConfig",
    "FinetuneConfig",
    "LoraConfig",
    "ModelConfig",
    "TeacherConfig",
    "TrainConfig",
    "VALID_PROVIDERS",
    "make_distillation_records",
    "resolve_provider",
]
