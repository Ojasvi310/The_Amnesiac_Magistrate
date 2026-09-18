"""
Retrieval layer: builds and queries a FAISS index over document chunks.
No training-stack imports -- only faiss, sentence_transformers, numpy, standard lib.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Maps a directory-name prefix to an ISO effective date.
# Extend this table as new regime quarters are introduced.
REGIME_DATE_MAP: dict[str, str] = {
    "regime_q1": "2024-01-01",
    "regime_q2": "2024-04-01",
    "regime_q3": "2024-07-01",
    "regime_q4": "2024-10-01",
    "regime_2025_q1": "2025-01-01",
    "regime_2025_q2": "2025-04-01",
    "regime_2025_q3": "2025-07-01",
    "regime_2025_q4": "2025-10-01",
}

_DEFAULT_EFFECTIVE_DATE = "1970-01-01"


def _resolve_effective_date(dir_name: str) -> str:
    """Return the ISO effective date for a directory name, falling back to a sentinel."""
    for key, date in REGIME_DATE_MAP.items():
        if dir_name.startswith(key):
            return date
    return _DEFAULT_EFFECTIVE_DATE


@dataclass
class IndexMetadata:
    regime: str
    effective_date: str  # ISO format, e.g. "2024-01-01"
    section: str
    text: str


def _chunk_text(raw: str) -> list[str]:
    """Split on double newlines; skip blank chunks."""
    return [p.strip() for p in raw.split("\n\n") if p.strip()]


def build_index(
    docs_root: str | Path,
    embed_model_id: str,
    output_dir: str | Path,
) -> str:
    """
    Scan every .txt file under docs_root, chunk by paragraph, embed with
    SentenceTransformer, and persist a FAISS IndexFlatIP plus chunk metadata.

    Returns the index hash (SHA-256 of the written index.faiss bytes).
    """
    docs_root = Path(docs_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(embed_model_id)

    txt_files = sorted(docs_root.rglob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found under {docs_root}")

    all_chunks: list[IndexMetadata] = []
    all_texts: list[str] = []

    for fpath in txt_files:
        # Derive regime and section from the directory structure.
        rel = fpath.relative_to(docs_root)
        parts = rel.parts
        regime = parts[0] if len(parts) > 1 else "unknown"
        section = fpath.stem

        effective_date = _resolve_effective_date(regime)

        raw = fpath.read_text(encoding="utf-8", errors="replace")
        for chunk_text in _chunk_text(raw):
            meta = IndexMetadata(
                regime=regime,
                effective_date=effective_date,
                section=section,
                text=chunk_text,
            )
            all_chunks.append(meta)
            all_texts.append(chunk_text)

    if not all_texts:
        raise ValueError("Documents found but produced no non-empty chunks.")

    embeddings: np.ndarray = model.encode(
        all_texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_path = output_dir / "index.faiss"
    chunks_path = output_dir / "chunks.json"

    faiss.write_index(index, str(index_path))
    with chunks_path.open("w", encoding="utf-8") as fh:
        json.dump([asdict(c) for c in all_chunks], fh, indent=2)

    return hash_index(output_dir)


def hash_index(index_dir: str | Path) -> str:
    """SHA-256 of index.faiss file contents -- used as the retrieval index version ID."""
    index_path = Path(index_dir) / "index.faiss"
    if not index_path.exists():
        raise FileNotFoundError(f"index.faiss not found in {index_dir}")
    h = hashlib.sha256()
    with index_path.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def load_index(
    index_dir: str | Path,
) -> tuple[faiss.Index, list[IndexMetadata], str]:
    """
    Load a previously built FAISS index from disk.

    Returns (index, chunks, index_hash).
    Raises FileNotFoundError if either artifact is missing.
    """
    index_dir = Path(index_dir)
    index_path = index_dir / "index.faiss"
    chunks_path = index_dir / "chunks.json"

    for p in (index_path, chunks_path):
        if not p.exists():
            raise FileNotFoundError(f"Required index artifact missing: {p}")

    index = faiss.read_index(str(index_path))
    with chunks_path.open("r", encoding="utf-8") as fh:
        raw_chunks = json.load(fh)

    chunks = [IndexMetadata(**c) for c in raw_chunks]
    idx_hash = hash_index(index_dir)
    return index, chunks, idx_hash


def query_index(
    query_embedding: np.ndarray,
    index: faiss.Index,
    chunks: list[IndexMetadata],
    top_k: int,
    regime_filter: Optional[str] = None,
) -> list[tuple[IndexMetadata, float]]:
    """
    Return up to top_k (IndexMetadata, score) pairs sorted by inner-product score
    descending. When regime_filter is supplied, only chunks whose regime field
    matches are returned (temporal filter applied post-retrieval).
    """
    vec = query_embedding.reshape(1, -1).astype(np.float32)

    # Over-fetch when filtering so we have enough candidates to fill top_k.
    fetch_k = top_k * 10 if regime_filter else top_k
    fetch_k = min(fetch_k, index.ntotal)

    scores, indices = index.search(vec, fetch_k)
    scores = scores[0].tolist()
    indices = indices[0].tolist()

    results: list[tuple[IndexMetadata, float]] = []
    for idx, score in zip(indices, scores):
        if idx < 0:
            continue
        meta = chunks[idx]
        if regime_filter and meta.regime != regime_filter:
            continue
        results.append((meta, float(score)))
        if len(results) >= top_k:
            break

    return results


def embed_query(text: str, embed_model: SentenceTransformer) -> np.ndarray:
    """Encode a single query string, returning a float32 1-D array."""
    vec = embed_model.encode(
        [text],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return vec[0].astype(np.float32)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build"])
    parser.add_argument("--docs-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--embed-model-id", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()
    
    if args.command == "build":
        build_index(args.docs_root, args.embed_model_id, args.output_dir)
        print(f"Index built successfully at {args.output_dir}")
