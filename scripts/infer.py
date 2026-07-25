#!/usr/bin/env python
"""Load a fine-tuned LoRA adapter and generate — the 'private efficient system'.

    python scripts/infer.py --adapter outputs/student_finetuned \
        --prompt "Explain what an ETF is in one sentence."
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.finetune.chat import to_messages


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", required=True, help="Path to the saved LoRA adapter dir.")
    p.add_argument("--base", default=None, help="Base model id (defaults to adapter config).")
    p.add_argument("--prompt", required=True, help="Instruction to answer.")
    p.add_argument("--max-new-tokens", type=int, default=256)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.adapter)
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.adapter,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    text = tok.apply_chat_template(
        to_messages(args.prompt), tokenize=False, add_generation_prompt=True
    )
    enc = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = model.generate(
            **enc, max_new_tokens=args.max_new_tokens, do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    answer = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    print(answer.strip())


if __name__ == "__main__":
    main()
