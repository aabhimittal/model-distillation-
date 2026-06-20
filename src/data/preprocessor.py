"""Chunk long passages for ChromaDB ingestion."""

from __future__ import annotations

from typing import List, Tuple


def chunk_passage(
    text: str, chunk_size: int = 256, overlap: int = 32
) -> List[str]:
    """
    Split a passage into overlapping word-level chunks.
    Word-level chunking avoids mid-word splits that confuse embedders.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def chunk_passages(
    passages: List[str], chunk_size: int = 256, overlap: int = 32
) -> Tuple[List[str], List[int]]:
    """
    Chunk a list of passages.
    Returns (all_chunks, source_indices) so each chunk can be traced back
    to its original passage index.
    """
    all_chunks: List[str] = []
    source_indices: List[int] = []
    for idx, passage in enumerate(passages):
        chunks = chunk_passage(passage, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(chunks)
        source_indices.extend([idx] * len(chunks))
    return all_chunks, source_indices
