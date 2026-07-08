"""Load a specialised-domain instruction dataset from the HF Hub and normalise it.

Heavy imports (``datasets``) are done lazily inside functions so this module — and
the config/formatting helpers it re-exports — stay importable on a bare CI runner.
"""

from __future__ import annotations

from typing import Dict, List

from .chat import normalize_record
from .config import DataConfig


def load_domain_dataset(cfg: DataConfig):
    """Return a HuggingFace ``Dataset`` of canonical ``{instruction,input,output}`` rows.

    The source dataset's columns are remapped through ``cfg.*_key`` so any Alpaca-
    style or Q&A-style dataset can be plugged in without code changes. Rows with an
    empty instruction or empty target are dropped — they carry no supervised signal.
    """
    from datasets import load_dataset

    ds = load_dataset(cfg.name, split=cfg.split)
    ds = ds.shuffle(seed=cfg.seed)

    def _map(record: Dict) -> Dict:
        return normalize_record(
            record,
            instruction_key=cfg.instruction_key,
            input_key=cfg.input_key,
            output_key=cfg.output_key,
        )

    ds = ds.map(_map, remove_columns=ds.column_names)
    ds = ds.filter(lambda r: bool(r["instruction"]) and bool(r["output"]))

    if cfg.max_train_samples and len(ds) > cfg.max_train_samples + cfg.max_eval_samples:
        ds = ds.select(range(cfg.max_train_samples + cfg.max_eval_samples))

    return ds


def train_eval_split(ds, n_eval: int):
    """Split into (train, eval). ``n_eval`` is clamped so tiny datasets still work."""
    n_eval = max(1, min(n_eval, max(1, len(ds) // 5)))
    eval_ds = ds.select(range(n_eval))
    train_ds = ds.select(range(n_eval, len(ds)))
    return train_ds, eval_ds


def records_to_dicts(ds) -> List[Dict[str, str]]:
    """Materialise a Dataset into a plain list of dicts (for teacher generation)."""
    return [dict(row) for row in ds]
