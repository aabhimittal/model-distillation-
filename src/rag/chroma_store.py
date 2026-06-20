"""ChromaDB wrapper with idempotent upsert and cosine similarity."""

from __future__ import annotations

import hashlib
from typing import List, Dict, Any, Optional

import numpy as np
import chromadb
from chromadb.config import Settings


class ChromaStore:
    def __init__(self, persist_dir: Optional[str] = None, collection_name: str = "squad_contexts"):
        if persist_dir:
            self.client = chromadb.PersistentClient(path=persist_dir)
        else:
            # EphemeralClient for tests — no disk I/O
            self.client = chromadb.EphemeralClient()

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _passage_id(self, text: str) -> str:
        """Deterministic ID from passage content — enables idempotent upsert."""
        return hashlib.sha256(text.encode()).hexdigest()[:32]

    def add_documents(
        self,
        passages: List[str],
        embeddings: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 512,
    ) -> None:
        """Upsert passages in batches. Safe to call multiple times."""
        if metadata is None:
            metadata = [{"idx": i} for i in range(len(passages))]

        ids = [self._passage_id(p) for p in passages]

        for i in range(0, len(passages), batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_docs = passages[i : i + batch_size]
            batch_embs = embeddings[i : i + batch_size].tolist()
            batch_meta = metadata[i : i + batch_size]

            # ChromaDB rejects empty metadata dicts — ensure at least one key
            safe_meta = [m if m else {"_": "1"} for m in batch_meta]
            self.collection.upsert(
                ids=batch_ids,
                documents=batch_docs,
                embeddings=batch_embs,
                metadatas=safe_meta,
            )

    def query(
        self, embedding: np.ndarray, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Return top_k results as list of {document, distance, metadata}."""
        results = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=min(top_k, self.collection_size()),
            include=["documents", "distances", "metadatas"],
        )
        output = []
        for doc, dist, meta in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        ):
            output.append({"document": doc, "distance": dist, "metadata": meta})
        return output

    def collection_size(self) -> int:
        return self.collection.count()
