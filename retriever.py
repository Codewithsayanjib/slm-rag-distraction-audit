"""BM25 and dense (cosine) retrievers over a flat list of text chunks.

Dense retrieval uses sentence-transformers/all-MiniLM-L6-v2 embeddings with
pure-numpy cosine search (no FAISS). faiss-cpu causes SIGSEGV on macOS MPS
because its AVX memory allocations corrupt the Metal address space.
"""
from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

_EMBED_MODEL: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        # Pinned to CPU — must never touch the MPS context.
        _EMBED_MODEL = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
        )
    return _EMBED_MODEL


# ── BM25 ─────────────────────────────────────────────────────────────────────

class BM25Retriever:
    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks
        tokenized   = [c.lower().split() for c in chunks]
        self.bm25   = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        scores  = self.bm25.get_scores(query.lower().split())
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [self.chunks[i] for i in top_idx]


# ── Dense (numpy cosine, no FAISS) ────────────────────────────────────────────

class DenseRetriever:
    """All-MiniLM-L6-v2 embeddings + brute-force cosine search via numpy matmul.

    For 16k × 384-dim float32, a single matrix-vector multiply takes <2 ms —
    fast enough and safe on macOS MPS (no native extension memory conflicts).
    """

    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks
        model       = _get_embed_model()
        embs        = model.encode(
            chunks, batch_size=64, show_progress_bar=False,
            normalize_embeddings=True,
        )
        self.embs = embs.astype(np.float32)   # (N, 384)

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        model  = _get_embed_model()
        q_emb  = model.encode(
            [query], normalize_embeddings=True,
        ).astype(np.float32)                  # (1, 384)
        scores  = self.embs @ q_emb.T         # (N, 1)
        top_idx = np.argsort(scores[:, 0])[::-1][:top_k]
        return [self.chunks[i] for i in top_idx]


# ── factory ──────────────────────────────────────────────────────────────────

def build_retriever(
    chunks: list[str], retriever_type: str
) -> BM25Retriever | DenseRetriever:
    if retriever_type == "bm25":
        return BM25Retriever(chunks)
    if retriever_type == "faiss":          # label kept for result CSV compatibility
        return DenseRetriever(chunks)
    raise ValueError(f"Unknown retriever: {retriever_type!r}")
