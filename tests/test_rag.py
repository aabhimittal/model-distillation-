"""Tests for the RAG stack — uses ephemeral ChromaDB, no HF models loaded."""

import numpy as np
import pytest

from src.rag.chroma_store import ChromaStore


PASSAGES = [
    "Paris is the capital of France and a major European city.",
    "The Eiffel Tower is located in Paris, France.",
    "Berlin is the capital of Germany.",
    "The Amazon River is the largest river by discharge volume.",
    "Python is a popular programming language created by Guido van Rossum.",
    "Machine learning is a subset of artificial intelligence.",
    "The speed of light in vacuum is approximately 299,792,458 metres per second.",
    "Napoleon Bonaparte was a French military commander and political leader.",
    "The Great Wall of China stretches over 21,000 kilometres.",
    "Shakespeare wrote Hamlet, Macbeth, and Romeo and Juliet.",
]

DIM = 8  # tiny embedding dim for fast tests


def _random_embs(n: int, dim: int = DIM, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    embs = rng.random((n, dim)).astype(np.float32)
    # Normalize for cosine similarity
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / (norms + 1e-9)


def test_chroma_store_init():
    store = ChromaStore()
    assert store.collection_size() == 0


def test_add_and_count():
    store = ChromaStore()
    embs = _random_embs(len(PASSAGES))
    store.add_documents(PASSAGES, embs)
    assert store.collection_size() == len(PASSAGES)


def test_retrieval_returns_topk():
    store = ChromaStore()
    embs = _random_embs(len(PASSAGES))
    store.add_documents(PASSAGES, embs)

    query_emb = _random_embs(1)[0]
    results = store.query(query_emb, top_k=3)
    assert len(results) == 3
    for r in results:
        assert "document" in r
        assert "distance" in r


def test_idempotent_upsert():
    store = ChromaStore()
    embs = _random_embs(len(PASSAGES))
    store.add_documents(PASSAGES, embs)
    # Insert same passages again
    store.add_documents(PASSAGES, embs)
    # Size should not grow
    assert store.collection_size() == len(PASSAGES)


def test_query_top_k_capped_by_collection():
    """
    When top_k > collection size, ChromaStore.query caps n_results to collection size
    so ChromaDB does not raise an error. The result length should not exceed
    the number of documents in the collection.
    """
    store = ChromaStore()
    small_passages = PASSAGES[:3]
    embs = _random_embs(3)
    store.add_documents(small_passages, embs)

    query_emb = _random_embs(1)[0]
    results = store.query(query_emb, top_k=10)  # request more than available
    # Should not crash and should return at most collection_size results
    assert len(results) <= 10
    # The first 3 results should be our known passages
    returned_docs = {r["document"] for r in results}
    assert returned_docs.issuperset(set(small_passages))


def test_retrieval_relevance():
    """
    Semantic relevance test: a query embedding very close to passage[0]
    should return passage[0] in top-1.
    """
    store = ChromaStore()
    embs = _random_embs(len(PASSAGES))
    store.add_documents(PASSAGES, embs)

    # Query with a slightly perturbed version of embs[0] — should still be closest
    query_emb = embs[0] + np.random.default_rng(0).random(DIM).astype(np.float32) * 0.01
    query_emb /= np.linalg.norm(query_emb)

    results = store.query(query_emb, top_k=3)
    top_doc = results[0]["document"]
    assert top_doc == PASSAGES[0]
