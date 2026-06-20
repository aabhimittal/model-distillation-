"""High-level retrieval interface composing Embedder and ChromaStore."""

from __future__ import annotations

from typing import List, Tuple

from .embedder import Embedder
from .chroma_store import ChromaStore


class Retriever:
    def __init__(self, embedder: Embedder, store: ChromaStore):
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Return the top-k passage texts for a query."""
        emb = self.embedder.embed_query(query)
        results = self.store.query(emb, top_k=top_k)
        return [r["document"] for r in results]

    def retrieve_with_scores(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Return (passage, cosine_distance) pairs for inspection."""
        emb = self.embedder.embed_query(query)
        results = self.store.query(emb, top_k=top_k)
        return [(r["document"], r["distance"]) for r in results]

    def retrieve_batch(self, queries: List[str], top_k: int = 3) -> List[List[str]]:
        """Retrieve top-k passages for each query in a batch."""
        embs = self.embedder.embed_queries(queries)
        results = []
        for emb in embs:
            docs = self.store.query(emb, top_k=top_k)
            results.append([d["document"] for d in docs])
        return results

    def retrieve_negative(
        self, queries: List[str], top_k: int = 3
    ) -> List[List[str]]:
        """
        Retrieve contexts for the *other half* of the batch (circular shift by n//2).
        These are used as negative contexts for the Contrastive Retrieval Alignment term.
        Shifting by half maximises question diversity within the batch.
        """
        n = len(queries)
        shift = max(1, n // 2)
        negative_queries = queries[shift:] + queries[:shift]
        return self.retrieve_batch(negative_queries, top_k=top_k)
