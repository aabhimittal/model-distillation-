"""Device-aware sentence embedder using MiniLM."""

from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_passages(self, passages: List[str], batch_size: int = 64) -> np.ndarray:
        """Embed a list of passages in batches. Returns (N, dim) float32 array."""
        all_embeddings = []
        for i in tqdm(range(0, len(passages), batch_size), desc="Embedding passages"):
            batch = passages[i : i + batch_size]
            embs = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            all_embeddings.append(embs)
        return np.vstack(all_embeddings).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns (dim,) float32 array."""
        return self.model.encode(query, convert_to_numpy=True).astype(np.float32)

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        """Embed multiple queries at once. Returns (N, dim) float32 array."""
        return self.model.encode(queries, convert_to_numpy=True).astype(np.float32)
