"""
Instruction formatting and supervised-label masking — pure Python, no heavy deps.

This module is the tokenizer-agnostic core of the fine-tuning track. It turns a
raw ``{instruction, input, output}`` record into a single prompt/target pair and,
given already-tokenised ids, builds the ``labels`` tensor with the prompt tokens
masked to ``-100`` so the loss is only computed on the response — the standard
supervised fine-tuning (SFT) recipe.

Keeping this logic import-light means it is fully unit-testable on a CPU runner
with no torch / transformers / GPU, which is what the CI job relies on.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

IGNORE_INDEX = -100

# A compact Alpaca-style template. Deliberately model-agnostic: when a real
# tokenizer with a chat template is available we prefer that (see ``model.py``),
# but this fallback keeps the data layer usable and testable on its own.
PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes the "
    "request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)
PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
)


def normalize_record(
    record: Dict[str, Any],
    instruction_key: str = "instruction",
    input_key: str = "input",
    output_key: str = "output",
) -> Dict[str, str]:
    """Map an arbitrary dataset row onto the canonical instruction schema.

    Missing keys degrade gracefully to empty strings so a dataset that only has,
    say, ``question``/``answer`` columns can be adapted purely through config.
    """
    return {
        "instruction": str(record.get(instruction_key, "") or "").strip(),
        "input": str(record.get(input_key, "") or "").strip(),
        "output": str(record.get(output_key, "") or "").strip(),
    }


def build_prompt(instruction: str, input_text: str = "") -> str:
    """Render the prompt (everything the model conditions on, up to the response)."""
    if input_text and input_text.strip():
        return PROMPT_WITH_INPUT.format(instruction=instruction.strip(), input=input_text.strip())
    return PROMPT_NO_INPUT.format(instruction=instruction.strip())


def to_messages(instruction: str, input_text: str = "") -> List[Dict[str, str]]:
    """Chat-style message list for tokenizers that expose ``apply_chat_template``."""
    user = instruction.strip()
    if input_text and input_text.strip():
        user = f"{user}\n\n{input_text.strip()}"
    return [{"role": "user", "content": user}]


def build_supervised_labels(
    prompt_ids: List[int],
    response_ids: List[int],
    eos_id: Optional[int] = None,
    max_length: Optional[int] = None,
) -> Dict[str, List[int]]:
    """Concatenate prompt+response into ``input_ids`` and mask the prompt in ``labels``.

    Only the response tokens (and the EOS that teaches the model to stop) carry a
    loss signal; prompt positions are set to ``IGNORE_INDEX``. Truncation keeps the
    left/prompt side intact and clips from the right, which preserves the task
    framing while dropping overly long completions.
    """
    resp = list(response_ids)
    if eos_id is not None and (not resp or resp[-1] != eos_id):
        resp = resp + [eos_id]

    input_ids = list(prompt_ids) + resp
    labels = [IGNORE_INDEX] * len(prompt_ids) + resp[:]

    if max_length is not None and len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    return {"input_ids": input_ids, "labels": labels}


def has_trainable_labels(labels: List[int]) -> bool:
    """True if at least one position contributes to the loss (guards empty targets)."""
    return any(l != IGNORE_INDEX for l in labels)
