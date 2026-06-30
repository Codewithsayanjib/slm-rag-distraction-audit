"""Tokenizer-aware text chunking using tiktoken cl100k_base."""
from functools import lru_cache

import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str, chunk_size: int, overlap: int = 0) -> list[str]:
    """Split *text* into chunks of exactly *chunk_size* tokens (no overlap by default).

    Trailing chunk is included only if it has >= 10 tokens.
    """
    tokens = _ENC.encode(text)
    step   = chunk_size - overlap
    chunks = []

    for i in range(0, len(tokens), step):
        slice_ = tokens[i : i + chunk_size]
        if len(slice_) < 10:
            break
        chunks.append(_ENC.decode(slice_))

    return chunks


def token_count(text: str) -> int:
    return len(_ENC.encode(text))
