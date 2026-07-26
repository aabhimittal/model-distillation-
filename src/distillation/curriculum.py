"""
Retrieval-Utility Curriculum — order training data by how much retrieval helped.

Curriculum learning normally ranks examples by a generic difficulty proxy (sequence
length, loss under a warm-up model). For RAG-augmented distillation there is a much
better ordering signal available for free: the per-example retrieval utility

    u = NLL_bare(y*) - NLL_rag(y*)

Examples with high u are ones where the retriever found the right passage and the
teacher used it — the transfer signal is clean and unambiguous. Examples with low or
negative u are ones where retrieval misfired; their RAG soft labels are noisy.

Training on high-u examples first lets the student establish the retrieval-to-weights
mapping on clean supervision before it has to cope with noisy retrieval. This pairs
naturally with `RetrievalUtilityGate`, which down-weights the same noisy examples once
they do enter the mix.

Competence schedule follows Platanios et al. (2019), "Competence-based Curriculum
Learning for Neural Machine Translation":

    c(t) = min(1, sqrt( t/T · (1 - c₀²) + c₀² ))

which grows quickly at first then flattens, so the model sees the full distribution
well before the end of training.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler


def competence(step: int, total_steps: int, initial: float = 0.25) -> float:
    """
    Fraction of the (utility-sorted) dataset visible at `step`.

    Returns a value in [initial, 1.0]. Reaches 1.0 at `total_steps`.
    """
    if total_steps <= 0:
        return 1.0
    frac = min(1.0, max(0.0, step / total_steps))
    return min(1.0, math.sqrt(frac * (1.0 - initial ** 2) + initial ** 2))


class RetrievalUtilityCurriculum(Sampler[int]):
    """
    Sampler that reveals training examples in descending retrieval-utility order.

    At step t only the top `competence(t)` fraction of the ranking is sampleable;
    indices within that prefix are shuffled so batches stay i.i.d. within the
    competent pool.

    The trainer advances the schedule by calling `set_step()`; competence is
    recomputed when the DataLoader builds a fresh iterator, i.e. once per epoch.

        sampler = RetrievalUtilityCurriculum(utilities, total_steps=len(loader)*epochs)
        loader = DataLoader(ds, batch_size=8, sampler=sampler)
        ...
        sampler.set_step(global_step)   # start of each epoch
    """

    def __init__(
        self,
        utilities: Sequence[float],
        total_steps: int,
        initial_competence: float = 0.25,
        seed: int = 0,
    ):
        self.utilities = np.asarray(utilities, dtype=np.float64)
        if self.utilities.ndim != 1:
            raise ValueError(f"utilities must be 1-D, got shape {self.utilities.shape}")

        self.total_steps = total_steps
        self.initial_competence = initial_competence
        self.rng = np.random.default_rng(seed)
        self.step = 0

        # Descending utility: retrieval-helped examples first
        self.ranking = np.argsort(-self.utilities)

    def set_step(self, step: int) -> None:
        self.step = step

    def current_competence(self) -> float:
        return competence(self.step, self.total_steps, self.initial_competence)

    def __iter__(self) -> Iterator[int]:
        c = self.current_competence()
        n_visible = max(1, int(round(c * len(self.ranking))))
        pool = self.ranking[:n_visible].copy()
        self.rng.shuffle(pool)
        return iter(pool.tolist())

    def __len__(self) -> int:
        c = self.current_competence()
        return max(1, int(round(c * len(self.ranking))))


def load_utilities(path: str | Path, example_ids: Sequence[str]) -> np.ndarray:
    """
    Load per-example utilities written by `generate_soft_labels.py` and align them
    to the dataset's example order.

    Missing ids get utility 0.0 (neutral) so a partially-generated soft-label
    directory degrades to "no curriculum preference" rather than crashing.
    """
    with open(path) as f:
        table = json.load(f)
    return np.array([float(table.get(eid, 0.0)) for eid in example_ids], dtype=np.float64)
