"""
Phase 1: Build the ChromaDB vector store from SQuAD v2 contexts.

Usage:
    python scripts/build_vector_db.py [--config ...] [--chroma-dir ...]

Idempotent: checks collection size before embedding, skips if already populated.

Colab example:
    python scripts/build_vector_db.py --chroma-dir /content/drive/MyDrive/rad/chroma_db
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.data.dataset_loader import load_squad, filter_answerable, extract_contexts
from src.data.preprocessor import chunk_passages
from src.rag.embedder import Embedder
from src.rag.chroma_store import ChromaStore


def main(config_path: str, chroma_dir: str | None) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # CLI flag overrides YAML config
    if chroma_dir:
        cfg["rag"]["chroma_persist_dir"] = chroma_dir

    print("Loading SQuAD v2 train split...")
    dataset = load_squad("train")
    dataset = filter_answerable(dataset)
    print(f"Loaded {len(dataset)} answerable examples.")

    passages = extract_contexts(dataset)
    print(f"Extracted {len(passages)} unique passages.")

    chunks, source_indices = chunk_passages(
        passages,
        chunk_size=cfg["rag"]["chunk_size"],
        overlap=cfg["rag"]["chunk_overlap"],
    )
    print(f"Chunked into {len(chunks)} passages (chunk_size={cfg['rag']['chunk_size']}).")

    store = ChromaStore(
        persist_dir=cfg["rag"]["chroma_persist_dir"],
        collection_name=cfg["rag"]["collection_name"],
    )

    existing = store.collection_size()
    if existing >= len(chunks):
        print(f"ChromaDB already contains {existing} documents. Skipping embedding.")
        return

    print(f"ChromaDB has {existing} docs; need {len(chunks)}. Embedding...")
    embedder = Embedder(cfg["models"]["embedder"])
    embeddings = embedder.embed_passages(chunks)

    metadata = [{"source_passage_idx": int(idx)} for idx in source_indices]
    store.add_documents(chunks, embeddings, metadata)

    final_size = store.collection_size()
    print(f"\nDone. ChromaDB '{cfg['rag']['collection_name']}' now has {final_size} documents.")
    print(f"Persisted at: {cfg['rag']['chroma_persist_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/distillation_config.yaml")
    parser.add_argument("--chroma-dir", default=None, help="Override ChromaDB persist directory")
    args = parser.parse_args()
    main(args.config, args.chroma_dir)
